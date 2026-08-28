import os
import re
import json
import math
import difflib
from io import StringIO
import pymysql
import pymysql.cursors
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from datetime import datetime
from flask import Flask, render_template, request, jsonify, make_response, Response, stream_with_context, redirect, url_for
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from email_validator import validate_email, EmailNotValidError

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'fyp-forecastiq-dev-secret-2025')

login_manager = LoginManager(app)
login_manager.login_view = 'login_page'

import requests as _requests

import chat_engine as ce

OLLAMA_URL   = os.environ.get('OLLAMA_HOST', 'http://localhost:11434')
OLLAMA_MODEL = os.environ.get('OLLAMA_MODEL', 'llama3.2')

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR  = os.path.join(BASE_DIR, '..', 'models')
DATASET_DIR = os.path.join(BASE_DIR, '..', 'dataset')

DB_CONFIG = {
    'host':        os.environ.get('MYSQL_HOST', 'localhost'),
    'port':        int(os.environ.get('MYSQL_PORT', 3306)),
    'user':        os.environ.get('MYSQL_USER', 'root'),
    'password':    os.environ.get('MYSQL_PASSWORD', 'admin'),
    'database':    os.environ.get('MYSQL_DATABASE', 'fyp_forecasting'),
    'cursorclass': pymysql.cursors.DictCursor,
    'autocommit':  True,
}

# ── Database ──────────────────────────────────────────────────────────────────

def get_db():
    return pymysql.connect(**DB_CONFIG)


class User(UserMixin):
    def __init__(self, id, email, restaurant_name):
        self.id = id
        self.email = email
        self.restaurant_name = restaurant_name


@login_manager.user_loader
def load_user(user_id):
    conn = get_db()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT id, email, restaurant_name FROM users WHERE id = %s",
                (int(user_id),)
            )
            row = cur.fetchone()
    finally:
        conn.close()
    if not row:
        return None
    return User(row['id'], row['email'], row['restaurant_name'])


def init_db():
    conn = get_db()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    id              INT AUTO_INCREMENT PRIMARY KEY,
                    email           VARCHAR(255) NOT NULL UNIQUE,
                    password_hash   VARCHAR(255) NOT NULL,
                    restaurant_name VARCHAR(255) NOT NULL,
                    created_at      TEXT NOT NULL
                )
            """)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS sessions (
                    id            INT AUTO_INCREMENT PRIMARY KEY,
                    created_at    TEXT NOT NULL,
                    filename      TEXT NOT NULL,
                    total_records INT  NOT NULL,
                    weeks         INT  NOT NULL,
                    centers       INT  NOT NULL,
                    meals         INT  NOT NULL,
                    horizon       INT  NOT NULL,
                    user_id       INT
                )
            """)
            # Add user_id to sessions if upgrading from an older schema
            try:
                cur.execute("ALTER TABLE sessions ADD COLUMN user_id INT")
            except Exception:
                pass
            cur.execute("""
                CREATE TABLE IF NOT EXISTS forecasts (
                    id             INT AUTO_INCREMENT PRIMARY KEY,
                    session_id     INT  NOT NULL,
                    week           INT  NOT NULL,
                    center_id      INT  NOT NULL,
                    meal_id        INT  NOT NULL,
                    category       TEXT,
                    cuisine        TEXT,
                    ai_forecast    INT  NOT NULL,
                    override_value INT,
                    lstm_forecast  INT,
                    FOREIGN KEY (session_id) REFERENCES sessions(id)
                )
            """)
            try:
                cur.execute("ALTER TABLE forecasts ADD COLUMN lstm_forecast INT")
            except Exception:
                pass
            try:
                cur.execute("ALTER TABLE forecasts ADD COLUMN forecast_min INT")
            except Exception:
                pass
            try:
                cur.execute("ALTER TABLE forecasts ADD COLUMN forecast_max INT")
            except Exception:
                pass
            try:
                cur.execute("ALTER TABLE sessions ADD COLUMN history_json MEDIUMTEXT")
            except Exception:
                pass
            try:
                cur.execute("ALTER TABLE sessions ADD COLUMN forecast_rf_json MEDIUMTEXT")
            except Exception:
                pass
            try:
                cur.execute("ALTER TABLE sessions ADD COLUMN feat_json MEDIUMTEXT")
            except Exception:
                pass
            cur.execute("""
                CREATE TABLE IF NOT EXISTS actuals (
                    id            INT AUTO_INCREMENT PRIMARY KEY,
                    session_id    INT  NOT NULL,
                    week          INT  NOT NULL,
                    actual_orders INT  NOT NULL,
                    created_at    TEXT NOT NULL,
                    FOREIGN KEY (session_id) REFERENCES sessions(id)
                )
            """)
            # One row per (session, meal_id) -- a per-item summary purpose-built for the AI
            # assistant to search/ground on (uploaded item name, whether it's a known trained
            # item or a cold-start estimate, total historical orders, next-week forecast).
            # Decoupled from `forecasts`, which is per-week-per-item and serves the results
            # table/charts.
            cur.execute("""
                CREATE TABLE IF NOT EXISTS session_meals (
                    id                  INT AUTO_INCREMENT PRIMARY KEY,
                    session_id          INT  NOT NULL,
                    meal_id             INT  NOT NULL,
                    item_name           TEXT,
                    category            TEXT,
                    cuisine             TEXT,
                    is_cold_start       TINYINT(1) NOT NULL DEFAULT 0,
                    total_orders        INT NOT NULL DEFAULT 0,
                    next_week_forecast  INT,
                    forecast_min        INT,
                    forecast_max        INT,
                    FOREIGN KEY (session_id) REFERENCES sessions(id)
                )
            """)
            try:
                cur.execute("ALTER TABLE session_meals ADD COLUMN alt_names TEXT")
            except Exception:
                pass
    finally:
        conn.close()


# ── LSTM model definition (must match train_lstm.py) ─────────────────────────

class LSTMForecaster(nn.Module):
    def __init__(self, input_size=7, hidden_size=128, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True, dropout=0.2)
        self.head = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :]).squeeze(-1)


# ── Model artifacts ───────────────────────────────────────────────────────────

xgb_model      = joblib.load(os.path.join(MODELS_DIR, 'xgb_model.pkl'))
rf_model       = joblib.load(os.path.join(MODELS_DIR, 'rf_model.pkl'))
label_encoders = joblib.load(os.path.join(MODELS_DIR, 'label_encoders.pkl'))

with open(os.path.join(MODELS_DIR, 'feature_columns.json')) as f:
    FEATURE_COLS = json.load(f)

with open(os.path.join(MODELS_DIR, 'model_metrics.json')) as f:
    MODEL_METRICS = json.load(f)

# ── LSTM loading (optional — app works without it) ────────────────────────────

lstm_model    = None
lstm_scaler   = None
lstm_cfg      = None

def _load_lstm():
    global lstm_model, lstm_scaler, lstm_cfg
    model_path  = os.path.join(MODELS_DIR, 'lstm_model.pt')
    config_path = os.path.join(MODELS_DIR, 'lstm_config.json')
    scaler_path = os.path.join(MODELS_DIR, 'lstm_scaler.pkl')
    if not all(os.path.exists(p) for p in [model_path, config_path, scaler_path]):
        print('[LSTM] Model files not found — run train_lstm.py to generate them.')
        return
    try:
        with open(config_path) as f:
            cfg = json.load(f)
        model = LSTMForecaster(cfg['input_size'], cfg['hidden_size'], cfg['num_layers'])
        model.load_state_dict(torch.load(model_path, map_location='cpu', weights_only=True))
        model.eval()
        lstm_model  = model
        lstm_scaler = joblib.load(scaler_path)
        lstm_cfg    = cfg
        print(f'[LSTM] Loaded (seq_len={cfg["seq_len"]}, hidden={cfg["hidden_size"]})')
    except Exception as e:
        print(f'[LSTM] Could not load: {e}')

_load_lstm()

meal_info   = pd.read_csv(os.path.join(DATASET_DIR, 'meal_info.csv'))
center_info = pd.read_csv(os.path.join(DATASET_DIR, 'fulfilment_center_info.csv'))

# The fixed, trained vocabulary the label encoders were fit on -- an uploaded file can
# only ever be normalized against these exact values, never expanded with new ones.
MEAL_CATEGORIES = sorted(meal_info['category'].unique())
MEAL_CUISINES   = sorted(meal_info['cuisine'].unique())
KNOWN_MEAL_IDS   = set(meal_info['meal_id'].astype(int))
KNOWN_CENTER_IDS = set(center_info['center_id'].astype(int))
# Most restaurants track a menu category but not a separate "cuisine type" -- cuisine is a
# secondary/auxiliary feature for the model, so rather than force every uncategorized-cuisine
# item to be excluded, unresolved cuisine falls back to a suggested value instead. This is a
# documented, explainable assumption (surfaced in the review step), not a silent guess.
DEFAULT_CUISINE = meal_info['cuisine'].mode().iat[0]
# The most common cuisine WITHIN each category, e.g. most Beverages items in the trained data
# happen to be Thai -- a much more specific, defensible suggestion for a new beverage item
# than the single global default.
CATEGORY_DEFAULT_CUISINE = meal_info.groupby('category')['cuisine'].agg(lambda s: s.mode().iat[0]).to_dict()

_CUISINE_KEYWORDS = {
    'indian': 'Indian', 'north indian': 'Indian', 'south indian': 'Indian',
    'thai': 'Thai',
    'continental': 'Continental', 'western': 'Continental', 'european': 'Continental',
    'italian': 'Italian', 'italiano': 'Italian',
}

init_db()

REQUIRED_COLS = [
    'week', 'center_id', 'meal_id', 'checkout_price', 'base_price',
    'emailer_for_promotion', 'homepage_featured', 'num_orders'
]

MAX_UPLOAD_BYTES = 50 * 1024 * 1024  # 50 MB
ALLOWED_EXTENSIONS = {'.csv', '.xlsx', '.xls'}
ALLOWED_MIMETYPES  = {
    'text/csv', 'application/csv', 'text/plain',
    'application/vnd.ms-excel',
    'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
}

# ── Flexible upload: column mapping ─────────────────────────────────────────

OPTIONAL_COLS = ['category', 'cuisine', 'item_name']

COLUMN_SYNONYMS = {
    'week':                   ['week', 'wk', 'week_no', 'week_number', 'period'],
    'center_id':              ['center_id', 'centre_id', 'outlet_id', 'store_id', 'location_id', 'branch_id'],
    'meal_id':                ['meal_id', 'item_id', 'product_id', 'menu_id', 'sku', 'sku_id', 'dish_id'],
    'checkout_price':         ['checkout_price', 'selling_price', 'sale_price', 'unit_price',
                                'cost', 'item_price', 'amount'],
    'base_price':             ['base_price', 'list_price', 'mrp', 'original_price', 'regular_price', 'menu_price'],
    'emailer_for_promotion':  ['emailer_for_promotion', 'email_promo', 'promo_email', 'emailer', 'promotion'],
    'homepage_featured':      ['homepage_featured', 'featured', 'is_featured', 'homepage'],
    'num_orders':             ['num_orders', 'orders', 'order_count', 'quantity', 'qty', 'units_sold', 'sales'],
    'category':               ['category', 'category_name', 'menu_category', 'food_category', 'type',
                                'item_group', 'product_group', 'menu_group'],
    'cuisine':                ['cuisine', 'cuisine_type', 'food_type'],
    # Free text, no vocabulary to normalize against -- purely a display/lookup field, never
    # fed to the model. Deliberately excludes bare 'item'/'dish' (too collision-prone with
    # meal_id's 'item_id'/'dish_id' synonyms).
    'item_name':              ['item_name', 'meal_name', 'dish_name', 'product_name', 'menu_item',
                                'item_description', 'description'],
}

_MAP_HIGH   = 0.85  # exact / known-synonym-level match -> auto-suggest
_MAP_MEDIUM = 0.65  # plausible fuzzy match -> suggest, but require confirmation
_MAP_LOW    = 0.55  # weak fuzzy match -> still surfaced, but least trusted
# below _MAP_LOW: no deterministic candidate at all -> eligible for the LLM fallback


def _norm(s):
    return re.sub(r'[^a-z0-9]', '', str(s).lower())


def _confidence_tier(score):
    if score >= _MAP_HIGH:
        return 'high'
    if score >= _MAP_MEDIUM:
        return 'medium'
    if score >= _MAP_LOW:
        return 'low'
    return 'none'


def _match_score(a_norm, b_norm):
    """Similarity between two already-normalized strings. Plain SequenceMatcher.ratio()
    unfairly penalizes a short, exact synonym embedded in a longer multi-word header (e.g.
    "cost" inside "currentcost" scores ~0.53 on ratio alone, below even the low-confidence
    cutoff) -- a clean substring match is a strong signal regardless of the length gap, so
    it's boosted to at least medium confidence."""
    if a_norm == b_norm:
        return 1.0
    ratio = difflib.SequenceMatcher(None, a_norm, b_norm).ratio()
    shorter, longer = sorted((a_norm, b_norm), key=len)
    if len(shorter) >= 4 and shorter in longer:
        ratio = max(ratio, _MAP_MEDIUM + 0.1)
    return ratio


def suggest_column_mapping(file_columns):
    """Greedy best-match mapping from every mappable field (required + optional) to a
    column in file_columns, deterministic only (exact name / synonym / fuzzy match).
    Returns {field: {'column': str|None, 'confidence': 'high'|'medium'|'low'|'none',
    'source': 'deterministic'}}."""
    file_norm = {c: _norm(c) for c in file_columns}
    fields = REQUIRED_COLS + OPTIONAL_COLS
    scored = []
    for field in fields:
        syn_norms = [_norm(s) for s in COLUMN_SYNONYMS.get(field, [field])]
        for file_col, fn in file_norm.items():
            score = max(_match_score(fn, sn) for sn in syn_norms)
            scored.append((score, field, file_col))
    scored.sort(key=lambda x: -x[0])

    mapping = {f: {'column': None, 'confidence': 'none', 'source': None} for f in fields}
    used_reqs, used_files = set(), set()
    for score, field, file_col in scored:
        if score < _MAP_LOW or field in used_reqs or file_col in used_files:
            continue
        mapping[field] = {'column': file_col, 'confidence': _confidence_tier(score), 'source': 'deterministic'}
        used_reqs.add(field)
        used_files.add(file_col)
    return mapping


def _suggest_mapping_with_llm(unmapped_columns: list, unmapped_fields: list) -> dict:
    """Last resort for columns deterministic matching couldn't place at all. Ask the local
    LLM to match each leftover column to one of the still-unmapped required fields via
    tool-calling, constrained to that exact enum (+ null) -- it can never invent a field
    name that doesn't exist. Returns {field: column}; always surfaced to the user as a
    low-confidence AI suggestion that must be confirmed, never applied automatically."""
    if not unmapped_columns or not unmapped_fields:
        return {}
    tool = {
        'type': 'function',
        'function': {
            'name': 'map_columns',
            'description': 'Match each leftover spreadsheet column to the closest still-needed data field.',
            'parameters': {
                'type': 'object',
                'properties': {
                    col: {
                        'type': ['string', 'null'],
                        'enum': unmapped_fields + [None],
                        'description': f'Which required field does column "{col}" represent? null if none fit.',
                    }
                    for col in unmapped_columns
                },
                'required': unmapped_columns,
            },
        },
    }
    try:
        r = _requests.post(
            f'{OLLAMA_URL}/api/chat',
            json={
                'model': OLLAMA_MODEL,
                'messages': [
                    {'role': 'system', 'content': (
                        'You map spreadsheet column headers to a fixed set of required data fields for a '
                        'restaurant demand-forecasting system. For each column header, choose the single '
                        'best-matching field from the allowed list, or null if none plausibly match. Only '
                        'ever use one of the exact field names given -- never invent a new one.'
                    )},
                    {'role': 'user', 'content': 'Columns to match: ' + ', '.join(unmapped_columns)},
                ],
                'tools': [tool],
                'stream': False,
                'options': ce.deterministic_ollama_options(),
            },
            timeout=15,
        )
        r.raise_for_status()
        tool_calls = r.json().get('message', {}).get('tool_calls') or []
        if not tool_calls:
            return {}
        args = tool_calls[0].get('function', {}).get('arguments', {})
        if isinstance(args, str):
            args = json.loads(args)
    except Exception:
        return {}

    result = {}
    for col, field in (args or {}).items():
        if col in unmapped_columns and field in unmapped_fields and field not in result:
            result[field] = col
    return result


def _vocab_norm(s):
    return re.sub(r'[^a-z0-9]', '', str(s).lower())


def normalize_vocab_value(raw_value: str, vocab: list, synonyms: dict | None = None) -> dict:
    """Match one raw text value (e.g. a category cell from an uploaded file) against a
    fixed, trained vocabulary (e.g. the 14 real meal categories). Deterministic only --
    exact match, curated synonym, then fuzzy match -- with a confidence tier. Never
    returns a value that isn't literally in `vocab`."""
    raw_norm = _vocab_norm(raw_value)
    vocab_norm = {_vocab_norm(v): v for v in vocab}

    if raw_norm in vocab_norm:
        return {'suggested': vocab_norm[raw_norm], 'confidence': 'high', 'source': 'deterministic'}

    if synonyms:
        mapped = synonyms.get(str(raw_value).strip().lower())
        if mapped and mapped in vocab:
            return {'suggested': mapped, 'confidence': 'high', 'source': 'deterministic'}

    best_score, best_vocab = 0.0, None
    for norm_v, v in vocab_norm.items():
        score = _match_score(raw_norm, norm_v)
        if score > best_score:
            best_score, best_vocab = score, v

    if best_score >= _MAP_MEDIUM:
        return {'suggested': best_vocab, 'confidence': 'medium', 'source': 'deterministic'}
    if best_score >= _MAP_LOW:
        return {'suggested': best_vocab, 'confidence': 'low', 'source': 'deterministic'}
    return {'suggested': None, 'confidence': 'none', 'source': None}


def suggest_vocab_value_with_llm(raw_value: str, vocab: list, field_label: str) -> str | None:
    """Constrained fallback for a raw value deterministic matching couldn't place: ask the
    local LLM to pick from the exact trained vocabulary via tool-calling (or null). Used
    only for low/no-confidence cases, and always surfaced as a suggestion requiring human
    confirmation -- this never lets an unmatched/hallucinated value reach the model."""
    tool = {
        'type': 'function',
        'function': {
            'name': 'pick_value',
            'description': f'Pick the closest matching {field_label} from the allowed list, or null if none fit.',
            'parameters': {
                'type': 'object',
                'properties': {
                    'value': {'type': ['string', 'null'], 'enum': vocab + [None]},
                },
                'required': ['value'],
            },
        },
    }
    try:
        r = _requests.post(
            f'{OLLAMA_URL}/api/chat',
            json={
                'model': OLLAMA_MODEL,
                'messages': [
                    {'role': 'system', 'content': (
                        f'You match messy real-world text to a fixed, trained {field_label} vocabulary for a '
                        'restaurant demand-forecasting model. Pick the single closest match from the allowed '
                        'list, or null if genuinely nothing fits. Only ever return one of the exact allowed '
                        'values -- never invent a new one.'
                    )},
                    {'role': 'user', 'content': f'Value to match: "{raw_value}"'},
                ],
                'tools': [tool],
                'stream': False,
                'options': ce.deterministic_ollama_options(),
            },
            timeout=15,
        )
        r.raise_for_status()
        tool_calls = r.json().get('message', {}).get('tool_calls') or []
        if not tool_calls:
            return None
        args = tool_calls[0].get('function', {}).get('arguments', {})
        if isinstance(args, str):
            args = json.loads(args)
        value = (args or {}).get('value')
    except Exception:
        return None
    return value if value in vocab else None


_BOOL_MAP = {
    'yes': 1, 'y': 1, 'true': 1, 't': 1, '1': 1, '1.0': 1,
    'no': 0, 'n': 0, 'false': 0, 'f': 0, '0': 0, '0.0': 0, '': 0, 'nan': 0,
}


def clean_required_columns(df):
    """Coerce real-world messiness (currency symbols, yes/no flags, stray text) in the
    required columns to the numeric types the model expects."""
    df = df.copy()

    for col in ['week', 'center_id', 'meal_id', 'checkout_price', 'base_price', 'num_orders']:
        if df[col].dtype == object:
            df[col] = df[col].astype(str).str.replace(r'[^\d.\-]', '', regex=True)
        df[col] = pd.to_numeric(df[col], errors='coerce')

    for col in ['emailer_for_promotion', 'homepage_featured']:
        if df[col].dtype == object:
            df[col] = df[col].astype(str).str.strip().str.lower().map(_BOOL_MAP)
        df[col] = pd.to_numeric(df[col], errors='coerce')

    return df


# ── ML helpers ────────────────────────────────────────────────────────────────

def engineer_features(df):
    """Merges in the trained reference tables (meal_info/center_info) for category/cuisine/
    center_type. If the upload itself carried its own (already-normalized, stage 6/7)
    category/cuisine columns, those are used as a fallback ONLY for meal_ids the reference
    table doesn't have -- meal_info stays authoritative whenever a meal_id is known. Rows
    that still can't resolve a real category/cuisine/center_type (unknown center_id, or an
    unknown meal_id with no usable category) are dropped rather than crashing the label
    encoders on an unseen value. Returns (df_feat, known_feat, cold_feat, excluded_unresolved_count,
    resolved_df) -- resolved_df is the retained rows before the (separate) insufficient-history
    filter, so callers can report accurate summary counts against what was actually usable.

    The "need 3 prior weeks for lag features" requirement below only makes sense for known
    meal_ids, which is what the pretrained models were fit on -- cold-start meal_ids (not in
    KNOWN_MEAL_IDS) use a different forecaster that reads their raw history directly, not
    lag/rolling features, so they must NOT be dropped here just for having limited history;
    that's precisely the case the cold-start path exists to handle."""
    for col in ('category', 'cuisine'):
        if col in df.columns:
            df = df.rename(columns={col: f'_uploaded_{col}'})

    df = df.merge(meal_info, on='meal_id', how='left')
    df = df.merge(center_info, on='center_id', how='left')

    for col in ('category', 'cuisine'):
        upload_col = f'_uploaded_{col}'
        if upload_col in df.columns:
            df[col] = df[col].fillna(df[upload_col])
            df = df.drop(columns=[upload_col])

    # Cuisine is secondary to category -- default rather than exclude when it's still
    # unresolved (this only ever affects meal_ids meal_info didn't have, since known
    # meal_ids always get a real cuisine from that merge).
    if 'cuisine' in df.columns:
        df['cuisine'] = df['cuisine'].fillna(DEFAULT_CUISINE)

    before = len(df)
    df = df.dropna(subset=['category', 'cuisine', 'center_type'])
    excluded_unresolved = before - len(df)
    resolved_df = df.copy()

    df['discount_pct'] = (
        (df['base_price'] - df['checkout_price']) / df['base_price']
    ).clip(lower=0)

    for col in ['category', 'cuisine', 'center_type']:
        le = label_encoders[col]
        df[col + '_enc'] = le.transform(df[col].astype(str))

    df = df.sort_values(['center_id', 'meal_id', 'week']).reset_index(drop=True)
    grp = df.groupby(['center_id', 'meal_id'])['num_orders']

    df['lag_1']          = grp.shift(1)
    df['lag_2']          = grp.shift(2)
    df['lag_3']          = grp.shift(3)
    df['rolling_mean_3'] = grp.shift(1).rolling(3, min_periods=1).mean().values
    df['rolling_std_3']  = grp.shift(1).rolling(3, min_periods=1).std().fillna(0).values

    is_known    = df['meal_id'].isin(KNOWN_MEAL_IDS)
    known_feat  = df[is_known].dropna(subset=['lag_1', 'lag_2', 'lag_3'])
    cold_feat   = df[~is_known]  # no lag-history requirement -- the cold-start forecaster
                                  # doesn't use lag/rolling features, it reads raw history directly
    df_feat = pd.concat([known_feat, cold_feat]).sort_values(['center_id', 'meal_id', 'week']).reset_index(drop=True)

    return df_feat, known_feat, cold_feat, excluded_unresolved, resolved_df


def forecast_future(df_feat, horizon=4, model=None, emailer=0, homepage=0, last_week=None):
    if model is None:
        model = xgb_model
    # Callers with a cold-start counterpart MUST pass last_week explicitly and anchor both
    # to the same value -- otherwise a new/cold-start item whose own data happens to extend
    # later than the known items' history skews df_feat's raw max() higher than known items'
    # true last week, and the two groups end up forecasting non-overlapping week ranges
    # (e.g. horizon=4 chosen but 7 distinct weeks produced across known + cold-start rows).
    last_week = int(last_week) if last_week is not None else int(df_feat['week'].max())

    # Build a dict of per-group state once
    groups = {}
    for (center_id, meal_id), grp in df_feat.groupby(['center_id', 'meal_id']):
        grp = grp.sort_values('week')
        last = grp.iloc[-1]
        groups[(center_id, meal_id)] = {
            'recent': grp['num_orders'].tolist(),
            'checkout_price':  last['checkout_price'],
            'base_price':      last['base_price'],
            'discount_pct':    last['discount_pct'],
            'op_area':         last['op_area'],
            'city_code':       last['city_code'],
            'region_code':     last['region_code'],
            'center_type_enc': last['center_type_enc'],
            'category_enc':    last['category_enc'],
            'cuisine_enc':     last['cuisine_enc'],
        }

    keys    = list(groups.keys())
    results = []

    for h in range(1, horizon + 1):
        rows = []
        for (center_id, meal_id) in keys:
            g      = groups[(center_id, meal_id)]
            recent = g['recent']
            lag1   = recent[-1] if len(recent) >= 1 else 0
            lag2   = recent[-2] if len(recent) >= 2 else 0
            lag3   = recent[-3] if len(recent) >= 3 else 0
            rows.append({
                'week':                  last_week + h,
                'center_id':             center_id,
                'meal_id':               meal_id,
                'checkout_price':        g['checkout_price'],
                'base_price':            g['base_price'],
                'discount_pct':          g['discount_pct'],
                'emailer_for_promotion': emailer,
                'homepage_featured':     homepage,
                'op_area':               g['op_area'],
                'city_code':             g['city_code'],
                'region_code':           g['region_code'],
                'center_type_enc':       g['center_type_enc'],
                'category_enc':          g['category_enc'],
                'cuisine_enc':           g['cuisine_enc'],
                'lag_1':                 lag1,
                'lag_2':                 lag2,
                'lag_3':                 lag3,
                'rolling_mean_3':        np.mean(recent[-3:]),
                'rolling_std_3':         np.std(recent[-3:]) if len(recent) >= 2 else 0,
            })

        # Batch predict all groups at once for this horizon step
        X     = pd.DataFrame(rows)[FEATURE_COLS]
        preds = np.maximum(model.predict(X), 0).astype(int)

        for i, (center_id, meal_id) in enumerate(keys):
            pred = int(preds[i])
            results.append({'center_id': int(center_id), 'meal_id': int(meal_id),
                            'week': last_week + h, 'forecast': pred})
            groups[(center_id, meal_id)]['recent'].append(pred)

    if not results:
        return pd.DataFrame(columns=['center_id', 'meal_id', 'week', 'forecast'])
    return pd.DataFrame(results)


def forecast_future_lstm(df_feat, horizon=4, emailer=0.0, homepage=0.0, last_week=None):
    """LSTM multi-step forecast. Returns same shape as forecast_future."""
    if lstm_model is None or lstm_cfg is None:
        return pd.DataFrame(columns=['center_id', 'meal_id', 'week', 'forecast'])

    seq_len      = lstm_cfg['seq_len']
    seq_features = lstm_cfg['seq_features']
    lo_min       = lstm_cfg['lo_min']
    lo_rng       = lstm_cfg['lo_rng']
    n_feat       = len(seq_features)
    # See forecast_future's comment -- must match the cold-start anchor exactly.
    last_week    = int(last_week) if last_week is not None else int(df_feat['week'].max())

    df_w = df_feat.copy()
    df_w['log_orders'] = np.log1p(df_w['num_orders'])

    groups = {}
    for (center_id, meal_id), grp in df_w.groupby(['center_id', 'meal_id']):
        grp  = grp.sort_values('week')
        rows = grp[seq_features].values.astype(np.float32)
        if len(rows) < seq_len:
            pad  = np.tile(rows[0], (seq_len - len(rows), 1))
            rows = np.vstack([pad, rows])
        last = grp.iloc[-1]
        groups[(center_id, meal_id)] = {
            'seq':             rows[-seq_len:].copy(),
            'discount_pct':    float(last['discount_pct']),
            'emailer':         float(emailer),
            'homepage':        float(homepage),
            'category_enc':    float(last['category_enc']),
            'cuisine_enc':     float(last['cuisine_enc']),
            'center_type_enc': float(last['center_type_enc']),
        }

    keys    = list(groups.keys())
    results = []

    for h in range(1, horizon + 1):
        batch = np.array([groups[k]['seq'] for k in keys])      # (N, seq_len, n_feat)
        n     = len(batch)
        batch_scaled = lstm_scaler.transform(
            batch.reshape(-1, n_feat)
        ).reshape(n, seq_len, n_feat)

        with torch.no_grad():
            preds_s = lstm_model(torch.FloatTensor(batch_scaled)).numpy()

        # Inverse-transform the log_orders output (feature index 0)
        preds_log = preds_s * lo_rng + lo_min
        order_preds = np.maximum(np.expm1(preds_log), 0).astype(int)

        for i, (center_id, meal_id) in enumerate(keys):
            pred = int(order_preds[i])
            results.append({
                'center_id': int(center_id),
                'meal_id':   int(meal_id),
                'week':      last_week + h,
                'forecast':  pred,
            })
            g = groups[(center_id, meal_id)]
            new_row = np.array([
                np.log1p(pred), g['discount_pct'], g['emailer'], g['homepage'],
                g['category_enc'], g['cuisine_enc'], g['center_type_enc'],
            ], dtype=np.float32)
            g['seq'] = np.vstack([g['seq'][1:], new_row])

    if not results:
        return pd.DataFrame(columns=['center_id', 'meal_id', 'week', 'forecast'])
    return pd.DataFrame(results)


# ── Cold-start forecasting for meal_ids the trained models have never seen ────────────
# meal_id/center_id are raw numeric features in FEATURE_COLS -- for an unseen meal_id that
# number has no learned meaning, so the pretrained XGBoost/RF/LSTM ensemble can't produce a
# trustworthy prediction for it no matter how well category/cuisine were resolved. Instead:
# blend the item's own uploaded history (a real trend, when there's enough of it) with a
# baseline drawn from *this restaurant's own* known items in the same category/center (not a
# generic cross-restaurant average), scaled for this item's price. The less real history an
# item has, the more the forecast leans on the category baseline and the wider its band gets.

COLD_START_FULL_HISTORY_WEEKS = 4  # weeks of an item's own data needed to fully trust its trend


def _category_baseline_stats(known_feat):
    """Reference demand/price levels for known items, from THIS upload only -- per
    category+center (most specific), falling back to category-only, then an overall
    average if a category has no other known items at all."""
    if known_feat.empty:
        return None
    by_cat_center = known_feat.groupby(['category', 'center_id']).agg(
        avg_orders=('num_orders', 'mean'), avg_price=('checkout_price', 'mean'), n=('num_orders', 'size'),
    )
    by_cat = known_feat.groupby('category').agg(
        avg_orders=('num_orders', 'mean'), avg_price=('checkout_price', 'mean'), n=('num_orders', 'size'),
    )
    return {
        'by_cat_center': by_cat_center,
        'by_cat':        by_cat,
        'overall_orders': float(known_feat['num_orders'].mean()),
        'overall_price':  float(known_feat['checkout_price'].mean()),
    }


def _category_baseline(stats, category, center_id, checkout_price):
    """Expected weekly demand for a new item, from comparable known items' actual demand,
    adjusted for this item being priced above/below what those comparable items charge."""
    if stats is None:
        return 1.0  # no known items anywhere in this upload -- last-resort floor, not a guess

    row = None
    by_cc = stats['by_cat_center']
    if (category, center_id) in by_cc.index and by_cc.loc[(category, center_id), 'n'] >= 3:
        row = by_cc.loc[(category, center_id)]
    elif category in stats['by_cat'].index:
        row = stats['by_cat'].loc[category]

    if row is not None:
        avg_orders, avg_price = float(row['avg_orders']), float(row['avg_price'])
    else:
        avg_orders, avg_price = stats['overall_orders'], stats['overall_price']

    price_ratio = 1.0
    if avg_price > 0 and checkout_price and checkout_price > 0:
        price_ratio = float(np.clip(avg_price / checkout_price, 0.6, 1.6))
    return max(0.0, avg_orders * price_ratio)


def _category_promo_uplift(known_feat, category):
    """How much homepage_featured=1 typically boosts orders for known items in this
    category, in this upload. Falls back to no effect (1.0x) without enough data to trust."""
    if known_feat.empty:
        return 1.0
    pool         = known_feat[known_feat['category'] == category]
    featured     = pool.loc[pool['homepage_featured'] == 1, 'num_orders']
    not_featured = pool.loc[pool['homepage_featured'] == 0, 'num_orders']
    if len(featured) >= 5 and len(not_featured) >= 5 and not_featured.mean() > 0:
        return float(np.clip(featured.mean() / not_featured.mean(), 0.7, 2.0))
    return 1.0


def _cold_start_item_forecast(history_df, baseline, promo_uplift, horizon, last_week, homepage=0):
    """Blends this item's own trend (when it has one) with the category baseline. Weight on
    the item's own history grows with how many real weeks of its own data exist -- 0 weeks
    means pure category baseline, COLD_START_FULL_HISTORY_WEEKS+ means mostly its own trend.
    The confidence band widens the more the forecast leans on the (less certain) baseline."""
    weeks_of_history = len(history_df)
    weight_own = min(1.0, weeks_of_history / COLD_START_FULL_HISTORY_WEEKS)

    own_level = None
    slope = 0.0
    if weeks_of_history >= 2:
        x = np.arange(weeks_of_history, dtype=float)
        y = history_df['num_orders'].to_numpy(dtype=float)
        slope, intercept = np.polyfit(x, y, 1)
        own_level = intercept + slope * (weeks_of_history - 1)
    elif weeks_of_history == 1:
        own_level = float(history_df['num_orders'].iloc[0])

    rows = []
    for h in range(1, horizon + 1):
        own_estimate = max(0.0, own_level + slope * h) if own_level is not None else baseline
        blended = weight_own * own_estimate + (1 - weight_own) * baseline
        if homepage:
            blended *= promo_uplift
        blended = max(0.0, blended)
        uncertainty = 0.15 + 0.35 * (1 - weight_own)  # 15% with full own history, up to 50% with none
        forecast = int(round(blended))
        rows.append({
            'week':         last_week + h,
            'forecast':     forecast,
            'forecast_min': int(round(max(0, blended * (1 - uncertainty)))),
            'forecast_max': int(round(blended * (1 + uncertainty))),
        })
    return rows, weight_own


def cold_start_forecast_all(cold_feat, known_feat, horizon, last_week, homepage=0):
    """Forecasts for every (center_id, meal_id) group whose meal_id isn't in the trained
    vocabulary. Returns a DataFrame matching the known-item forecast shape, plus
    is_cold_start/confidence_weight so callers can tell these apart."""
    cols = ['center_id', 'meal_id', 'week', 'forecast', 'forecast_min', 'forecast_max',
            'is_cold_start', 'confidence_weight']
    if cold_feat.empty:
        return pd.DataFrame(columns=cols)

    stats = _category_baseline_stats(known_feat)
    rows = []
    for (center_id, meal_id), grp in cold_feat.groupby(['center_id', 'meal_id']):
        grp            = grp.sort_values('week')
        category       = grp['category'].iloc[-1]
        checkout_price = grp['checkout_price'].iloc[-1]
        baseline       = _category_baseline(stats, category, center_id, checkout_price)
        promo_uplift   = _category_promo_uplift(known_feat, category)
        forecast_rows, weight_own = _cold_start_item_forecast(
            grp, baseline, promo_uplift, horizon, last_week, homepage=homepage,
        )
        for r in forecast_rows:
            rows.append({
                'center_id': int(center_id), 'meal_id': int(meal_id),
                'week': r['week'], 'forecast': r['forecast'],
                'forecast_min': r['forecast_min'], 'forecast_max': r['forecast_max'],
                'is_cold_start': True, 'confidence_weight': round(weight_own, 2),
            })
    return pd.DataFrame(rows, columns=cols)


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route('/login', methods=['GET', 'POST'])
def login_page():
    if current_user.is_authenticated:
        return redirect(url_for('index'))
    error = None
    if request.method == 'POST':
        email    = (request.form.get('email') or '').strip().lower()
        password =  request.form.get('password') or ''
        conn = get_db()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT id, email, password_hash, restaurant_name FROM users WHERE email = %s",
                    (email,)
                )
                row = cur.fetchone()
        finally:
            conn.close()
        if row and check_password_hash(row['password_hash'], password):
            login_user(User(row['id'], row['email'], row['restaurant_name']), remember=True)
            return redirect(url_for('index'))
        error = 'Invalid email or password.'
    return render_template('login.html', error=error)


@app.route('/register', methods=['GET', 'POST'])
def register_page():
    if current_user.is_authenticated:
        return redirect(url_for('index'))
    error = None
    if request.method == 'POST':
        restaurant = (request.form.get('restaurant_name') or '').strip()
        email      = (request.form.get('email') or '').strip().lower()
        password   =  request.form.get('password') or ''
        confirm    =  request.form.get('confirm_password') or ''
        if not restaurant or not email or not password:
            error = 'All fields are required.'
        elif password != confirm:
            error = 'Passwords do not match.'
        elif len(password) < 8:
            error = 'Password must be at least 8 characters.'
        else:
            try:
                email = validate_email(email, check_deliverability=True).normalized
            except EmailNotValidError as e:
                error = str(e)
        if not error:
            conn = get_db()
            try:
                with conn.cursor() as cur:
                    cur.execute("SELECT id FROM users WHERE email = %s", (email,))
                    if cur.fetchone():
                        error = 'An account with that email already exists.'
                    else:
                        cur.execute("""
                            INSERT INTO users (email, password_hash, restaurant_name, created_at)
                            VALUES (%s, %s, %s, %s)
                        """, (email, generate_password_hash(password), restaurant,
                              datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')))
                        user_id = cur.lastrowid
            finally:
                conn.close()
            if not error:
                login_user(User(user_id, email, restaurant), remember=True)
                return redirect(url_for('index'))
    return render_template('register.html', error=error)


@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login_page'))


@app.route('/')
@login_required
def index():
    return render_template('index.html',
                           restaurant_name=current_user.restaurant_name,
                           user_email=current_user.email)


def _validate_upload_file(file):
    """Shared extension/size/mimetype checks. Returns an error response dict, or None if OK."""
    ext = os.path.splitext(file.filename or '')[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        return {'error': 'Unsupported file type. Please upload a CSV or Excel file (.csv, .xlsx, or .xls).'}

    file.seek(0, 2)
    size = file.tell()
    file.seek(0)
    if size > MAX_UPLOAD_BYTES:
        return {'error': f'Your file is too large ({size // (1024*1024)} MB). Please upload a file smaller than 50 MB.'}

    mimetype = (file.mimetype or '').split(';')[0].strip()
    if mimetype and mimetype not in ALLOWED_MIMETYPES:
        return {'error': 'Unsupported file type. Please upload a CSV or Excel file (.csv, .xlsx, or .xls).'}

    return None


@app.route('/upload/analyze', methods=['POST'])
@login_required
def analyze_upload():
    if 'file' not in request.files or request.files['file'].filename == '':
        return jsonify({'error': 'No file selected. Please choose a file before uploading.'}), 400

    try:
        file = request.files['file']
        err = _validate_upload_file(file)
        if err:
            return jsonify(err), 400

        ext = os.path.splitext(file.filename or '')[1].lower()
        df_head = pd.read_excel(file, nrows=5) if ext in {'.xlsx', '.xls'} else pd.read_csv(file, nrows=5)
        columns = [str(c) for c in df_head.columns]

        # Always compute the full mapping (not just a required-columns check) -- even an
        # exact-schema file might carry an optional category/cuisine column that stage
        # 6/7 (normalize-categories) will want later.
        mapping = suggest_column_mapping(columns)

        unmapped_required = [f for f in REQUIRED_COLS if mapping[f]['column'] is None]
        if unmapped_required:
            used_cols = {m['column'] for m in mapping.values() if m['column']}
            leftover_cols = [c for c in columns if c not in used_cols]
            ai_matches = _suggest_mapping_with_llm(leftover_cols, unmapped_required)
            for field, col in ai_matches.items():
                mapping[field] = {'column': col, 'confidence': 'low', 'source': 'ai'}

        match = all(mapping[f]['column'] == f for f in REQUIRED_COLS)
        sample_rows = df_head.head(3).astype(object).where(df_head.head(3).notnull(), None).values.tolist()
        return jsonify({
            'match':    match,
            'columns':  columns,
            'required': REQUIRED_COLS,
            'optional': OPTIONAL_COLS,
            'mapping':  mapping,
            'sample_rows': sample_rows,
        })
    except Exception:
        app.logger.exception('Could not read uploaded file header')
        return jsonify({'error': 'Could not read your file. Please check that it is a valid '
                                  'CSV or Excel file and try again.'}), 400


def _read_and_rename(file, ext, column_mapping: dict):
    """Re-reads the file (full, not just a header preview) and applies a confirmed
    field -> column mapping. Shared by /upload/normalize-categories and /upload so both
    stages see the data under the same canonical field names."""
    df = pd.read_excel(file) if ext in {'.xlsx', '.xls'} else pd.read_csv(file)
    rename_map = {
        source: field
        for field, source in column_mapping.items()
        if field in (REQUIRED_COLS + OPTIONAL_COLS) and source in df.columns
    }
    if rename_map:
        df = df.rename(columns=rename_map)
    return df


@app.route('/upload/normalize-categories', methods=['POST'])
@login_required
def normalize_categories():
    """Stage 6/7: for any meal_id not already in the trained meal_info.csv (so its
    category/cuisine can't come from that authoritative source), find what raw
    category/cuisine text the upload itself provides and try to normalize it against the
    fixed trained vocabulary. Also flags center_ids the trained model has never seen --
    those have no text to normalize, they're just unresolvable. Nothing here writes
    anything; it only reports what stage 8 (the actual /upload save) will need resolved."""
    if 'file' not in request.files or request.files['file'].filename == '':
        return jsonify({'error': 'No file selected. Please choose a file before uploading.'}), 400

    try:
        file = request.files['file']
        err = _validate_upload_file(file)
        if err:
            return jsonify(err), 400

        column_mapping = {}
        raw_mapping = request.form.get('column_mapping')
        if raw_mapping:
            try:
                column_mapping = json.loads(raw_mapping)
            except ValueError:
                column_mapping = {}

        ext = os.path.splitext(file.filename or '')[1].lower()
        df = _read_and_rename(file, ext, column_mapping)

        if 'meal_id' not in df.columns or 'center_id' not in df.columns:
            return jsonify({
                'error': 'Column mapping must resolve meal_id and center_id before categories can be checked.',
            }), 400

        meal_ids   = pd.to_numeric(df['meal_id'], errors='coerce')
        center_ids = pd.to_numeric(df['center_id'], errors='coerce')

        unknown_center_ids = sorted({
            int(c) for c in center_ids.dropna().unique() if int(c) not in KNOWN_CENTER_IDS
        })
        unknown_mask = meal_ids.notna() & ~meal_ids.round().astype('Int64').isin(KNOWN_MEAL_IDS)
        unknown_meal_ids = sorted({int(m) for m in meal_ids[unknown_mask].unique()})

        def _collect_suggestions(col_name, vocab, synonyms, field_label):
            if col_name not in df.columns or not unknown_mask.any():
                return []
            raw = df.loc[unknown_mask, col_name].dropna().astype(str).str.strip()
            raw = raw[raw != '']
            suggestions = []
            for value in sorted(raw.unique()):
                result = normalize_vocab_value(value, vocab, synonyms)
                if result['confidence'] in ('low', 'none'):
                    ai_guess = suggest_vocab_value_with_llm(value, vocab, field_label)
                    if ai_guess:
                        result = {'suggested': ai_guess, 'confidence': 'low', 'source': 'ai'}
                suggestions.append({'raw_value': value, **result})
            return suggestions

        def _meals_missing_value(col_name):
            """meal_ids among the unknown ones that have no usable text in col_name at all --
            distinct from a value that's present but doesn't match the vocabulary."""
            if col_name not in df.columns or not unknown_mask.any():
                return unknown_meal_ids
            has_value = (
                df.loc[unknown_mask, col_name].notna()
                & (df.loc[unknown_mask, col_name].astype(str).str.strip() != '')
            )
            return sorted({int(m) for m in meal_ids[unknown_mask][~has_value.values].unique()})

        category_present = 'category' in df.columns
        category_suggestions = _collect_suggestions('category', MEAL_CATEGORIES, ce.CATEGORY_KEYWORDS, 'menu category')
        cuisine_suggestions  = _collect_suggestions('cuisine', MEAL_CUISINES, _CUISINE_KEYWORDS, 'cuisine type')

        unresolvable_meal_ids = _meals_missing_value('category') if category_present else unknown_meal_ids
        # meals with no usable cuisine text at all -- but skip ones that are getting excluded
        # anyway for lack of a category, there's no point suggesting a cuisine for those.
        meals_missing_cuisine = _meals_missing_value('cuisine') if 'cuisine' in df.columns else unknown_meal_ids
        meals_missing_cuisine = [m for m in meals_missing_cuisine if m not in set(unresolvable_meal_ids)]

        # raw category text -> the category we'd actually resolve it to, so the cuisine
        # suggestion below is based on where the item is HEADING, not its raw spelling.
        cat_suggestion_by_raw = {s['raw_value']: s['suggested'] for s in category_suggestions}

        cuisine_defaults = []
        for meal_id in meals_missing_cuisine:
            resolved_category = None
            if category_present:
                vals = df.loc[meal_ids == meal_id, 'category'].dropna().astype(str).str.strip()
                vals = vals[vals != '']
                if not vals.empty:
                    raw_category = vals.iloc[0]
                    resolved_category = raw_category if raw_category in MEAL_CATEGORIES else cat_suggestion_by_raw.get(raw_category)

            if resolved_category and resolved_category in CATEGORY_DEFAULT_CUISINE:
                suggested = CATEGORY_DEFAULT_CUISINE[resolved_category]
                cuisine_defaults.append({
                    'meal_id':   meal_id,
                    'category':  resolved_category,
                    'suggested': suggested,
                    'is_default': False,
                    'reason': f'Suggested because other {resolved_category} items in the trained data are usually {suggested}.',
                })
            else:
                cuisine_defaults.append({
                    'meal_id':   meal_id,
                    'category':  resolved_category,
                    'suggested': DEFAULT_CUISINE,
                    'is_default': True,
                    'reason': f'No similar items to base a suggestion on -- using "{DEFAULT_CUISINE}", the most common cuisine overall, as a safe default.',
                })

        return jsonify({
            'needs_review':           bool(unknown_meal_ids or unknown_center_ids),
            'unknown_meal_ids':       unknown_meal_ids,
            'unknown_center_ids':     unknown_center_ids,
            'unresolvable_meal_ids':  unresolvable_meal_ids,
            'cuisine_defaults':       cuisine_defaults,
            'category_column_present': category_present,
            'cuisine_column_present':  'cuisine' in df.columns,
            'category_suggestions':  category_suggestions,
            'cuisine_suggestions':   cuisine_suggestions,
            'categories_vocab':      MEAL_CATEGORIES,
            'cuisines_vocab':        MEAL_CUISINES,
            'default_cuisine':       DEFAULT_CUISINE,
        })
    except Exception:
        app.logger.exception('Could not analyze categories')
        return jsonify({'error': 'Could not analyze the categories in your file. '
                                  'Please check the file and try again.'}), 400


def _json_safe(obj):
    """Recursively replace float NaN with None. Several pandas operations (groupby key
    reconstruction, left-merge fill, scalar column assignment) resurface a missing value as
    NaN even when the source was a real Python None -- jsonify then emits the bare `NaN`
    token, which is invalid per the JSON spec, so browsers' JSON.parse() rejects the whole
    response outright. Applied once here rather than chased at each pandas call site."""
    if isinstance(obj, float) and math.isnan(obj):
        return None
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_safe(v) for v in obj]
    return obj


@app.route('/upload', methods=['POST'])
@login_required
def upload():
    if 'file' not in request.files or request.files['file'].filename == '':
        return jsonify({'error': 'No file selected. Please choose a file before uploading.'}), 400

    try:
        file = request.files['file']
        err = _validate_upload_file(file)
        if err:
            return jsonify(err), 400

        # Optional column mapping from the /upload/analyze confirmation step, e.g.
        # {"week": "Week No", "num_orders": "Units Sold", "category": "Type", ...}
        column_mapping = {}
        raw_mapping = request.form.get('column_mapping')
        if raw_mapping:
            try:
                column_mapping = json.loads(raw_mapping)
            except ValueError:
                column_mapping = {}

        ext = os.path.splitext(file.filename or '')[1].lower()
        df = _read_and_rename(file, ext, column_mapping)

        # Optional category/cuisine VALUE normalization from the /upload/normalize-categories
        # review step, e.g. {"Drinks": "Beverages", "Munchies": "Other Snacks"}. Only values
        # the user actually confirmed are in here. A raw value with no explicit mapping entry
        # still passes through if it's ALREADY an exact (case-insensitive) match to the real
        # trained vocabulary -- e.g. a value the review step didn't need to touch, or this
        # route being called directly without the review step for already-clean data. Anything
        # that's neither explicitly mapped nor already valid becomes NaN and is dropped later
        # (or, for cuisine specifically, defaulted) in engineer_features rather than guessed at
        # or fed to the label encoder. Blank/missing cells are normalized to the "" key so the
        # review step's "no cuisine listed -- pick one" dropdown can target them explicitly.
        for col, param, vocab in (('category', 'category_mapping', MEAL_CATEGORIES),
                                    ('cuisine', 'cuisine_mapping', MEAL_CUISINES)):
            if col not in df.columns:
                continue
            raw_value_map = {}
            raw_param = request.form.get(param)
            if raw_param:
                try:
                    raw_value_map = json.loads(raw_param)
                except ValueError:
                    raw_value_map = {}
            vocab_ci = {v.lower(): v for v in vocab}
            stripped = df[col].fillna('').astype(str).str.strip()
            mapped   = stripped.map(raw_value_map)
            already_valid = stripped.str.lower().map(vocab_ci)
            df[col] = mapped.fillna(already_valid)

        # item_name is free text -- no vocabulary to normalize against, just clean it up so
        # blank cells become real nulls (not the literal string "nan") for the meal_lookup
        # fallback-to-"Meal #id" logic further down to detect correctly.
        if 'item_name' in df.columns:
            cleaned = df['item_name'].fillna('').astype(str).str.strip()
            df['item_name'] = cleaned.replace('', None)

        missing = [c for c in REQUIRED_COLS if c not in df.columns]
        if missing:
            return jsonify({
                'error': 'Your file does not match the required format. Please download the data template and use it as a guide to set up your file correctly.',
                'template_hint': True,
            }), 400

        df = clean_required_columns(df)
        unreadable = int(df[REQUIRED_COLS].isna().any(axis=1).sum())
        df = df.dropna(subset=REQUIRED_COLS)
        if df.empty:
            return jsonify({
                'error': 'We could not read the values in your file after matching the columns. '
                         'Please check for non-numeric prices, quantities, or IDs and try again.',
            }), 400
        for col in ['week', 'center_id', 'meal_id', 'num_orders', 'emailer_for_promotion', 'homepage_featured']:
            df[col] = df[col].astype(int)

        # Per-item cuisine choice from the review step's "no cuisine listed" suggestions,
        # e.g. {"90101": "Thai"} -- keyed by meal_id rather than raw text, since a blank
        # cell carries no text to key off, and different meal_ids can get different
        # category-based suggestions. Only fills cells still blank after cuisine_mapping;
        # anything left unset still falls through to DEFAULT_CUISINE in engineer_features.
        if 'cuisine' in df.columns:
            raw_meal_defaults = request.form.get('cuisine_meal_defaults')
            meal_defaults = {}
            if raw_meal_defaults:
                try:
                    meal_defaults = {int(k): v for k, v in json.loads(raw_meal_defaults).items()}
                except (ValueError, TypeError):
                    meal_defaults = {}
            if meal_defaults:
                still_blank = df['cuisine'].isna()
                if still_blank.any():
                    fallback = df.loc[still_blank, 'meal_id'].map(meal_defaults)
                    df.loc[still_blank, 'cuisine'] = df.loc[still_blank, 'cuisine'].where(fallback.isna(), fallback)

        df_feat, known_feat, cold_feat, excluded_unresolved, resolved_df = engineer_features(df.copy())
        if df_feat.empty:
            return jsonify({'error': 'Not enough sales history in your file. Please make sure you have at least 3 weeks of data for each menu item.'}), 400

        df_feat['predicted'] = np.maximum(
            xgb_model.predict(df_feat[FEATURE_COLS]), 0
        ).astype(int)

        horizon = min(int(request.form.get('horizon', 4)), 12)

        # meal_id/center_id are raw numeric features the trained models split on -- for a
        # meal_id they've never seen, that number carries no learned meaning, so trained-model
        # forecasts are only trustworthy for known items. New items are routed through a
        # separate cold-start estimator (own history trend blended with a category baseline)
        # instead, so they never touch the pretrained ensemble and can't affect known-item
        # forecasts in any way.
        #
        # Anchor every forecast (known-item ensemble AND cold-start) to known_feat's own last
        # week rather than the full df_feat's -- a new/cold-start item that happens to have
        # more recent rows than the known items in this upload would otherwise push df_feat's
        # max() later than known_feat's, giving the two groups non-overlapping forecast week
        # ranges (e.g. horizon=4 chosen but 7 distinct weeks produced overall). Falls back to
        # df_feat's max only when there are no known items at all to anchor against.
        last_week = int(known_feat['week'].max()) if not known_feat.empty else int(df_feat['week'].max())
        # resolved_df (pre-lag-filter) has every usable row for known items, not just the
        # week-4-onward subset known_feat needs for lag features -- more data for the
        # category baseline/promo-uplift stats the cold-start estimator below relies on.
        known_resolved = resolved_df[resolved_df['meal_id'].isin(KNOWN_MEAL_IDS)]

        forecast_xgb_df  = forecast_future(known_feat, horizon, model=xgb_model, last_week=last_week) if not known_feat.empty else pd.DataFrame(columns=['center_id', 'meal_id', 'week', 'forecast'])
        forecast_rf_df   = forecast_future(known_feat, horizon, model=rf_model, last_week=last_week)  if not known_feat.empty else pd.DataFrame(columns=['center_id', 'meal_id', 'week', 'forecast'])
        forecast_lstm_df = forecast_future_lstm(known_feat, horizon, last_week=last_week)             if not known_feat.empty else pd.DataFrame(columns=['center_id', 'meal_id', 'week', 'forecast'])

        # Ensemble: average all available models (XGBoost + RF + LSTM) -- known items only
        if not forecast_xgb_df.empty:
            known_forecast_df = forecast_xgb_df.rename(columns={'forecast': 'xgb'})
            if not forecast_rf_df.empty:
                known_forecast_df = known_forecast_df.merge(
                    forecast_rf_df.rename(columns={'forecast': 'rf'}),
                    on=['center_id', 'meal_id', 'week'], how='left',
                )
            else:
                known_forecast_df['rf'] = np.nan
            if not forecast_lstm_df.empty:
                known_forecast_df = known_forecast_df.merge(
                    forecast_lstm_df.rename(columns={'forecast': 'lstm'}),
                    on=['center_id', 'meal_id', 'week'], how='left',
                )
            else:
                known_forecast_df['lstm'] = np.nan
            model_cols = ['xgb', 'rf', 'lstm']
            known_forecast_df['forecast']     = known_forecast_df[model_cols].median(axis=1, skipna=True).round().astype(int)
            _std = known_forecast_df[model_cols].std(axis=1, skipna=True).fillna(0).clip(upper=known_forecast_df['forecast'] * 0.25)
            known_forecast_df['forecast_min'] = (known_forecast_df['forecast'] - _std).clip(lower=0).round().astype(int)
            known_forecast_df['forecast_max'] = (known_forecast_df['forecast'] + _std).round().astype(int)
            known_forecast_df = known_forecast_df[['center_id', 'meal_id', 'week', 'forecast', 'forecast_min', 'forecast_max']]
        else:
            known_forecast_df = forecast_xgb_df.copy()
            known_forecast_df['forecast_min'] = known_forecast_df.get('forecast')
            known_forecast_df['forecast_max'] = known_forecast_df.get('forecast')
        known_forecast_df['is_cold_start']     = False
        known_forecast_df['is_cold_start']     = known_forecast_df['is_cold_start'].astype(bool)
        known_forecast_df['confidence_weight'] = 1.0

        cold_forecast_df = cold_start_forecast_all(cold_feat, known_resolved, horizon, last_week)
        forecast_df = pd.concat([known_forecast_df, cold_forecast_df], ignore_index=True)
        forecast_df['is_cold_start'] = forecast_df['is_cold_start'].astype(bool)

        history_agg = (
            df_feat.groupby('week')[['num_orders', 'predicted']]
            .sum()
            .reset_index()
        )

        forecast_agg = (
            forecast_df.groupby('week')[['forecast', 'forecast_min', 'forecast_max']].sum().reset_index()
            if not forecast_df.empty
            else pd.DataFrame(columns=['week', 'forecast', 'forecast_min', 'forecast_max'])
        )
        if not forecast_agg.empty:
            forecast_agg['forecast_min'] = forecast_agg['forecast_min'].clip(lower=(forecast_agg['forecast'] * 0.80).astype(int))
            forecast_agg['forecast_max'] = forecast_agg['forecast_max'].clip(upper=(forecast_agg['forecast'] * 1.20).astype(int))

        forecast_rf_agg = pd.DataFrame(columns=['week', 'forecast'])

        forecast_lstm_agg = (
            forecast_lstm_df.groupby('week')['forecast'].sum().reset_index()
            if not forecast_lstm_df.empty
            else pd.DataFrame(columns=['week', 'forecast'])
        )

        # Category/cuisine display lookup comes from df_feat (already resolved -- meal_info
        # where known, the upload's normalized value or the cuisine default otherwise), not
        # a fresh merge against the raw meal_info table, which would reintroduce NaN for any
        # meal_id meal_info doesn't have.
        meal_lookup = df_feat[['meal_id', 'category', 'cuisine']].drop_duplicates('meal_id').copy()
        if 'item_name' in df_feat.columns:
            names = (
                df_feat.dropna(subset=['item_name'])[['meal_id', 'item_name']]
                .drop_duplicates('meal_id', keep='first')
            )
            meal_lookup = meal_lookup.merge(names, on='meal_id', how='left')
            # A meal_id's raw item name can genuinely change across rows in a real upload
            # (menu renamed, rebranded, or a code reused for a different dish over time).
            # display_name/item_name above only keeps the first-seen spelling, which would
            # silently break chatbot lookups for any other name the same meal_id was ever
            # sold under -- so also keep every distinct spelling for matching purposes.
            alt_names_map = (
                df_feat.dropna(subset=['item_name']).groupby('meal_id')['item_name']
                .agg(lambda s: sorted(set(s)))
            )
            meal_lookup['alt_names'] = meal_lookup['meal_id'].map(alt_names_map)
            meal_lookup['alt_names'] = meal_lookup['alt_names'].apply(
                lambda v: v if isinstance(v, list) else [])
        else:
            meal_lookup['item_name'] = None
            meal_lookup['alt_names'] = [[] for _ in range(len(meal_lookup))]
        # Missing names above come out as float NaN (pandas' left-merge fill value / scalar
        # column assignment default), not Python None -- jsonify emits NaN as the bare
        # (invalid-JSON) token `NaN` rather than `null`, which fails JSON.parse() in the
        # browser. Force real None so the response stays valid JSON.
        meal_lookup['item_name'] = meal_lookup['item_name'].astype(object).where(
            meal_lookup['item_name'].notna(), None
        )
        # Always-a-string label for display -- uploaded name when available, otherwise the
        # existing "Meal #id" convention. Callers should prefer this over reimplementing the
        # same fallback in every chart/chat/export site.
        meal_lookup['display_name'] = meal_lookup['item_name'].where(
            meal_lookup['item_name'].notna(), meal_lookup['meal_id'].map(lambda m: f'Meal #{int(m)}')
        )

        top_meals = (
            df_feat.groupby('meal_id')['num_orders'].sum()
            .nlargest(30).reset_index()
            .rename(columns={'num_orders': 'total_orders'})
            .merge(meal_lookup, on='meal_id', how='left')
        )
        top_meals['is_cold_start'] = ~top_meals['meal_id'].isin(KNOWN_MEAL_IDS)

        forecast_detail = (
            forecast_df
            .merge(meal_lookup, on='meal_id', how='left')
            .merge(center_info[['center_id', 'center_type']], on='center_id', how='left')
            if not forecast_df.empty
            else pd.DataFrame()
        )

        if not forecast_detail.empty and not forecast_lstm_df.empty:
            forecast_detail = forecast_detail.merge(
                forecast_lstm_df[['center_id', 'meal_id', 'week', 'forecast']].rename(
                    columns={'forecast': 'lstm_forecast'}),
                on=['center_id', 'meal_id', 'week'], how='left',
            )
            # Cold-start rows never went through the LSTM (or any pretrained model) -- leave
            # them as None rather than fillna(0), which would misleadingly read as "LSTM
            # predicted zero orders" instead of "LSTM wasn't used for this item." Cast to
            # object dtype FIRST so None survives as a true null instead of being silently
            # upcast back to float NaN (which isn't valid JSON and breaks the response).
            is_cold = forecast_detail['is_cold_start'].astype(bool)
            forecast_detail['lstm_forecast'] = forecast_detail['lstm_forecast'].astype(object)
            forecast_detail.loc[~is_cold, 'lstm_forecast'] = forecast_detail.loc[~is_cold, 'lstm_forecast'].fillna(0).astype(int)
            forecast_detail.loc[is_cold, 'lstm_forecast'] = None
        elif not forecast_detail.empty:
            forecast_detail['lstm_forecast'] = None

        # Compact per-category summaries for the chatbot context
        forecast_top_next_week    = pd.DataFrame()
        forecast_bottom_next_week = pd.DataFrame()
        forecast_by_category      = pd.DataFrame()
        if not forecast_detail.empty:
            first_week = int(forecast_detail['week'].min())
            nw = forecast_detail[forecast_detail['week'] == first_week]
            meal_cat_totals = (
                nw.groupby(['meal_id', 'category', 'cuisine', 'item_name', 'display_name', 'is_cold_start'], dropna=False)['forecast']
                .sum().reset_index()
            )
            forecast_top_next_week = (
                meal_cat_totals.sort_values('forecast', ascending=False)
                .head(30)
                .sort_values(['category', 'forecast'], ascending=[True, False])
            )
            forecast_bottom_next_week = (
                meal_cat_totals.sort_values('forecast', ascending=True)
                .head(20)
                .sort_values(['category', 'forecast'], ascending=[True, True])
            )
            forecast_by_category = (
                forecast_detail.groupby(['week', 'category'])['forecast']
                .sum().reset_index()
                .sort_values(['week', 'forecast'], ascending=[True, False])
            )

        summary = {
            'total_records':        len(resolved_df),
            'weeks':                int(resolved_df['week'].nunique()),
            'centers':              int(resolved_df['center_id'].nunique()),
            'meals':                int(resolved_df['meal_id'].nunique()),
            'rows_skipped':         unreadable,
            'rows_excluded_unresolved': excluded_unresolved,
        }

        # Per-item summary purpose-built for the AI assistant -- every meal_id (not just the
        # top-30 shown on the dashboard), with total historical orders across all weeks and
        # the earliest forecasted week's numbers, so it can answer "how popular is X" with
        # real current-session figures instead of falling back to the old trained dataset.
        all_meal_totals = df_feat.groupby('meal_id')['num_orders'].sum().rename('total_orders').reset_index()
        next_week_by_meal = pd.DataFrame(columns=['meal_id', 'next_week_forecast', 'forecast_min', 'forecast_max'])
        if not forecast_detail.empty:
            first_fc_week = int(forecast_detail['week'].min())
            next_week_by_meal = (
                forecast_detail[forecast_detail['week'] == first_fc_week]
                .groupby('meal_id')
                .agg(next_week_forecast=('forecast', 'sum'),
                     forecast_min=('forecast_min', 'sum'),
                     forecast_max=('forecast_max', 'sum'))
                .reset_index()
            )
        session_meals_df = (
            meal_lookup[['meal_id', 'item_name', 'category', 'cuisine', 'alt_names']]
            .merge(all_meal_totals, on='meal_id', how='left')
            .merge(next_week_by_meal, on='meal_id', how='left')
        )
        session_meals_df['is_cold_start'] = ~session_meals_df['meal_id'].isin(KNOWN_MEAL_IDS)
        session_meals = session_meals_df.to_dict(orient='records')

        # Persist session + forecasts
        filename   = file.filename or 'upload.csv'
        session_id = _save_session(
            filename, summary, horizon, forecast_detail, current_user.id,
            has_lstm=not forecast_lstm_df.empty,
            history_agg=history_agg,
            forecast_rf_agg=forecast_rf_agg,
            df_feat=df_feat,
            session_meals=session_meals,
        )

        return jsonify(_json_safe({
            'session_id':               session_id,
            'summary':                  summary,
            'history':                  history_agg.to_dict(orient='records'),
            'forecast':                 forecast_agg.to_dict(orient='records'),
            'forecast_lstm':            forecast_lstm_agg.to_dict(orient='records'),
            'top_meals':                top_meals.to_dict(orient='records'),
            'forecast_detail':          forecast_detail.to_dict(orient='records'),
            'forecast_top_next_week':   forecast_top_next_week.to_dict(orient='records')    if not forecast_top_next_week.empty    else [],
            'forecast_bottom_next_week': forecast_bottom_next_week.to_dict(orient='records') if not forecast_bottom_next_week.empty else [],
            'forecast_by_category':     forecast_by_category.to_dict(orient='records')     if not forecast_by_category.empty     else [],
        }))

    except Exception as e:
        app.logger.exception('Upload processing failed')
        return jsonify({'error': 'Something went wrong while processing your file. Please check that your file is not corrupted and try again.'}), 500


@app.route('/override', methods=['POST'])
@login_required
def save_override():
    """Persist a single override value for a forecast row."""
    data = request.get_json(force=True)
    session_id = data.get('session_id')
    week       = data.get('week')
    center_id  = data.get('center_id')
    meal_id    = data.get('meal_id')
    override   = data.get('override')

    if None in (session_id, week, center_id, meal_id, override):
        return jsonify({'error': 'Missing fields'}), 400

    conn = get_db()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                UPDATE forecasts
                   SET override_value = %s
                 WHERE session_id = %s AND week = %s AND center_id = %s AND meal_id = %s
            """, (int(override), int(session_id), int(week), int(center_id), int(meal_id)))
    finally:
        conn.close()

    return jsonify({'ok': True})


@app.route('/history')
@login_required
def history():
    """Return the 20 most recent forecast sessions for the current user."""
    conn = get_db()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT id, created_at, filename, total_records, weeks, centers, meals, horizon
                  FROM sessions
                 WHERE user_id = %s
                 ORDER BY id DESC
                 LIMIT 20
            """, (current_user.id,))
            rows = cur.fetchall()
    finally:
        conn.close()
    return jsonify(rows)


@app.route('/session/<int:session_id>')
@login_required
def session_detail(session_id):
    """Return all forecast rows for a saved session (scoped to current user)."""
    conn = get_db()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT * FROM sessions WHERE id = %s AND user_id = %s",
                (session_id, current_user.id)
            )
            meta = cur.fetchone()
            if not meta:
                return jsonify({'error': 'Session not found'}), 404
            cur.execute(
                """
                SELECT f.*, sm.item_name, sm.is_cold_start
                FROM forecasts f
                LEFT JOIN session_meals sm ON sm.session_id = f.session_id AND sm.meal_id = f.meal_id
                WHERE f.session_id = %s ORDER BY f.week, f.center_id, f.meal_id
                """,
                (session_id,)
            )
            rows = cur.fetchall()
    finally:
        conn.close()
    return jsonify({'session': meta, 'forecasts': rows})


@app.route('/recalculate-featured', methods=['POST'])
@login_required
def recalculate_featured():
    data       = request.get_json()
    session_id = data.get('session_id')
    items      = data.get('items', [])   # [{center_id, meal_id}, ...]

    if not session_id or not items:
        return jsonify({'error': 'session_id and items required'}), 400

    conn = get_db()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT horizon, feat_json FROM sessions WHERE id = %s AND user_id = %s",
                (session_id, current_user.id)
            )
            row = cur.fetchone()
    finally:
        conn.close()

    if not row or not row['feat_json']:
        return jsonify({'error': 'Session data not available — please re-upload your file.'}), 404

    try:
        horizon = int(row['horizon'])
        df_feat = pd.read_json(StringIO(row['feat_json']), orient='records')

        # Filter to only the flagged (center_id, meal_id) pairs
        pairs  = {(int(it['center_id']), int(it['meal_id'])) for it in items}
        df_sub = df_feat[df_feat.apply(
            lambda r: (int(r['center_id']), int(r['meal_id'])) in pairs, axis=1
        )].copy()

        if df_sub.empty:
            return jsonify({'results': []}), 200

        # Same split as the initial forecast: known meal_ids go through the pretrained
        # ensemble, unseen ones through the cold-start estimator -- never the reverse.
        is_known_sub = df_sub['meal_id'].isin(KNOWN_MEAL_IDS)
        known_sub    = df_sub[is_known_sub]
        cold_sub     = df_sub[~is_known_sub]
        full_known_feat = df_feat[df_feat['meal_id'].isin(KNOWN_MEAL_IDS)]
        # Anchor to the FULL known-item pool's last week (matching how the original forecast
        # was anchored -- see the matching comment in /upload), not just this flagged subset's
        # own max, and not the raw df_feat max either -- either of those can drift from the
        # rest of the session's already-stored forecast weeks if this item's own data coverage
        # differs from the other known items.
        last_week = int(full_known_feat['week'].max()) if not full_known_feat.empty else int(df_feat['week'].max())

        if not known_sub.empty:
            xgb_df  = forecast_future(known_sub, horizon, model=xgb_model, homepage=1, last_week=last_week)
            rf_df   = forecast_future(known_sub, horizon, model=rf_model,  homepage=1, last_week=last_week)
            lstm_df = forecast_future_lstm(known_sub, horizon, homepage=1.0, last_week=last_week)

            merged = xgb_df.rename(columns={'forecast': 'xgb'})
            if not rf_df.empty:
                merged = merged.merge(
                    rf_df[['center_id', 'meal_id', 'week', 'forecast']].rename(columns={'forecast': 'rf'}),
                    on=['center_id', 'meal_id', 'week'], how='left')
            else:
                merged['rf'] = np.nan
            if not lstm_df.empty:
                merged = merged.merge(
                    lstm_df[['center_id', 'meal_id', 'week', 'forecast']].rename(columns={'forecast': 'lstm'}),
                    on=['center_id', 'meal_id', 'week'], how='left')
            else:
                merged['lstm'] = np.nan

            model_cols = [c for c in ['xgb', 'rf', 'lstm'] if c in merged.columns]
            merged['forecast']     = merged[model_cols].median(axis=1, skipna=True).round().astype(int)
            _std = merged[model_cols].std(axis=1, skipna=True).fillna(0).clip(upper=merged['forecast'] * 0.25)
            merged['forecast_min'] = (merged['forecast'] - _std).clip(lower=0).round().astype(int)
            merged['forecast_max'] = (merged['forecast'] + _std).round().astype(int)
            known_results = merged[['center_id', 'meal_id', 'week', 'forecast', 'forecast_min', 'forecast_max']]
        else:
            known_results = pd.DataFrame(columns=['center_id', 'meal_id', 'week', 'forecast', 'forecast_min', 'forecast_max'])

        cold_results = cold_start_forecast_all(cold_sub, full_known_feat, horizon, last_week, homepage=1)
        combined = pd.concat([known_results, cold_results], ignore_index=True)

        results = combined[['center_id', 'meal_id', 'week', 'forecast', 'forecast_min', 'forecast_max']].to_dict(orient='records')
        return jsonify({'results': results})

    except Exception:
        app.logger.exception('recalculate-featured error')
        return jsonify({'error': 'Could not recalculate the forecast for that item. '
                                  'Please try again.'}), 500


@app.route('/actuals', methods=['POST'])
@login_required
def save_actuals():
    data       = request.get_json(force=True)
    session_id = data.get('session_id')
    actuals    = data.get('actuals', [])
    if not session_id or not actuals:
        return jsonify({'error': 'Missing fields'}), 400
    conn = get_db()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT id FROM sessions WHERE id = %s AND user_id = %s",
                        (session_id, current_user.id))
            if not cur.fetchone():
                return jsonify({'error': 'Session not found'}), 404
            cur.execute("DELETE FROM actuals WHERE session_id = %s", (session_id,))
            rows = [
                (session_id, int(a['week']), int(a['actual_orders']),
                 datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S'))
                for a in actuals if a.get('actual_orders') is not None
            ]
            if rows:
                cur.executemany(
                    "INSERT INTO actuals (session_id, week, actual_orders, created_at) VALUES (%s,%s,%s,%s)",
                    rows,
                )
    finally:
        conn.close()
    return jsonify({'ok': True})


@app.route('/actuals/<int:session_id>')
@login_required
def get_actuals(session_id):
    conn = get_db()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT id FROM sessions WHERE id = %s AND user_id = %s",
                        (session_id, current_user.id))
            if not cur.fetchone():
                return jsonify({'error': 'Session not found'}), 404
            cur.execute(
                "SELECT week, actual_orders FROM actuals WHERE session_id = %s ORDER BY week",
                (session_id,),
            )
            rows = cur.fetchall()
    finally:
        conn.close()
    return jsonify(rows)


@app.route('/template')
def download_template():
    csv = 'week,center_id,meal_id,checkout_price,base_price,emailer_for_promotion,homepage_featured,num_orders\n'
    resp = make_response(csv)
    resp.headers['Content-Type'] = 'text/csv'
    resp.headers['Content-Disposition'] = 'attachment; filename="sales_data_template.csv"'
    return resp


@app.route('/metrics')
def metrics():
    try:
        with open(os.path.join(MODELS_DIR, 'model_metrics.json')) as f:
            return jsonify(json.load(f))
    except Exception:
        return jsonify(MODEL_METRICS)


# ── Internal helpers ──────────────────────────────────────────────────────────

def _save_session(filename, summary, horizon, forecast_detail, user_id=None, has_lstm=False,
                  history_agg=None, forecast_rf_agg=None, df_feat=None, session_meals=None):
    history_json_str     = history_agg.to_json(orient='records')     if history_agg     is not None and not history_agg.empty     else None
    forecast_rf_json_str = forecast_rf_agg.to_json(orient='records') if forecast_rf_agg is not None and not forecast_rf_agg.empty else None
    feat_json_str        = df_feat.to_json(orient='records')         if df_feat         is not None and not df_feat.empty         else None

    conn = get_db()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO sessions
                    (created_at, filename, total_records, weeks, centers, meals, horizon, user_id,
                     history_json, forecast_rf_json, feat_json)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """, (
                datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S'),
                filename,
                summary['total_records'],
                summary['weeks'],
                summary['centers'],
                summary['meals'],
                horizon,
                user_id,
                history_json_str,
                forecast_rf_json_str,
                feat_json_str,
            ))
            session_id = cur.lastrowid

            if not forecast_detail.empty:
                rows = []
                for _, r in forecast_detail.iterrows():
                    lstm_val = int(r['lstm_forecast']) if has_lstm and pd.notna(r.get('lstm_forecast')) else None
                    rows.append((
                        session_id,
                        int(r['week']),
                        int(r['center_id']),
                        int(r['meal_id']),
                        r.get('category') or None,
                        r.get('cuisine')  or None,
                        int(r['forecast']),
                        None,
                        lstm_val,
                        int(r['forecast_min']) if r.get('forecast_min') is not None else int(r['forecast']),
                        int(r['forecast_max']) if r.get('forecast_max') is not None else int(r['forecast']),
                    ))
                cur.executemany("""
                    INSERT INTO forecasts
                        (session_id, week, center_id, meal_id, category, cuisine,
                         ai_forecast, override_value, lstm_forecast, forecast_min, forecast_max)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """, rows)

            if session_meals:
                # pd.notna (not truthy `or`/`is not None`) -- these values can be NaN floats
                # from a merge, and NaN is truthy in Python, so `x or None` would silently
                # keep the NaN instead of nulling it out.
                def _clean(v, cast=None):
                    if not pd.notna(v):
                        return None
                    return cast(v) if cast else v

                meal_rows = [(
                    session_id,
                    int(m['meal_id']),
                    _clean(m.get('item_name')),
                    _clean(m.get('category')),
                    _clean(m.get('cuisine')),
                    1 if m.get('is_cold_start') else 0,
                    _clean(m.get('total_orders'), int) or 0,
                    _clean(m.get('next_week_forecast'), int),
                    _clean(m.get('forecast_min'), int),
                    _clean(m.get('forecast_max'), int),
                    json.dumps(m.get('alt_names') or []),
                ) for m in session_meals]
                cur.executemany("""
                    INSERT INTO session_meals
                        (session_id, meal_id, item_name, category, cuisine, is_cold_start,
                         total_orders, next_week_forecast, forecast_min, forecast_max, alt_names)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """, meal_rows)
    finally:
        conn.close()
    return session_id


# ── AI Chat ───────────────────────────────────────────────────────────────────

def _item_label(item):
    """Uploaded item name when available, falling back to the existing 'Meal #id' convention."""
    return item.get('display_name') or f"Meal #{item.get('meal_id')}"


def _item_chart_label(item):
    """Compact chart-axis label -- uploaded item name when available, else the existing
    '#id cuisine' convention."""
    return item.get('display_name') or f"#{item.get('meal_id', '?')} {item.get('cuisine', '')}"


def _chat_system(ctx):
    txt = (
        "You are ForecastIQ Assistant — built into a food demand forecasting platform for "
        "small restaurant managers. Help users understand their forecasts, spot trends, and "
        "get practical advice on reducing food waste, in plain, everyday business language.\n\n"
        "The section below is INTERNAL context to help you reason about the numbers — it is "
        "written in technical terms for your understanding only. NEVER repeat these technical "
        "terms to the user. The user is a restaurant owner/manager with no technical or data "
        "science background:\n"
        "• Users upload weekly sales CSVs with columns: week, center_id, meal_id, checkout_price, "
        "base_price, emailer_for_promotion, homepage_featured, num_orders.\n"
        "• Three ML models run: XGBoost (primary), Random Forest (secondary), and LSTM (deep learning).\n"
        "• Input features: lag orders (1/2/3 weeks back), 3-week rolling mean & std, "
        "discount %, promotional flags, meal category/cuisine encodings, "
        "fulfilment centre type/area/city/region.\n"
        "• Forecast horizons available: 1, 4, 8, or 12 weeks.\n"
        "• A safety buffer (0–20 %) inflates recommended stock quantities above the raw forecast.\n"
        "• Managers can override individual forecast values before exporting to CSV.\n\n"
    )
    s = ctx.get('summary')
    if s:
        txt += (
            "Current uploaded dataset:\n"
            f"  Records: {s.get('total_records', '?'):,}\n"
            f"  Weeks of history: {s.get('weeks', '?')}\n"
            f"  Fulfilment centres: {s.get('centers', '?')}\n"
            f"  Menu items: {s.get('meals', '?')}\n\n"
        )
    top = ctx.get('top_meals', [])
    if top:
        txt += "Top menu items by total historical orders:\n"
        for m in top[:10]:
            txt += (
                f"  {_item_label(m)} — {m.get('category', '')} "
                f"({m.get('cuisine', '')}): {m.get('total_orders', 0):,} orders\n"
            )
        txt += "\n"
    fc = ctx.get('forecast', [])
    if fc:
        txt += "Upcoming AI forecast (total across all locations & meals):\n"
        for r in fc:
            txt += f"  Week {r.get('week')}: {r.get('forecast', 0):,} forecasted orders\n"
        txt += "\n"
    top_nw = ctx.get('forecast_top_next_week', [])
    if top_nw:
        txt += "AI-forecasted top menu items for next week (summed across all locations), by category:\n"
        current_cat = None
        for item in top_nw:
            cat = item.get('category') or 'Unknown'
            if cat != current_cat:
                current_cat = cat
                txt += f"  {cat}:\n"
            txt += (
                f"    {_item_label(item)} ({item.get('cuisine', '')}): "
                f"{item.get('forecast', 0):,} forecasted orders\n"
            )
        txt += "\n"
    bot_nw = ctx.get('forecast_bottom_next_week', [])
    if bot_nw:
        txt += "AI-forecasted LOWEST demand menu items for next week (potential over-stock risk):\n"
        current_cat = None
        for item in bot_nw[:20]:
            cat = item.get('category') or 'Unknown'
            if cat != current_cat:
                current_cat = cat
                txt += f"  {cat}:\n"
            txt += (
                f"    {_item_label(item)} ({item.get('cuisine', '')}): "
                f"{item.get('forecast', 0):,} forecasted orders\n"
            )
        txt += "\n"
    by_cat = ctx.get('forecast_by_category', [])
    if by_cat:
        seen_weeks: list = []
        for item in by_cat:
            w = item.get('week')
            if w not in seen_weeks:
                seen_weeks.append(w)
        txt += "AI-forecasted orders by category per forecast week:\n"
        for w in seen_weeks:
            txt += f"  Week {w}:\n"
            for item in by_cat:
                if item.get('week') == w:
                    txt += f"    {item.get('category', '?')}: {item.get('forecast', 0):,} orders\n"
        txt += "\n"
    if not ctx.get('summary'):
        txt += (
            "IMPORTANT: No sales data has been uploaded yet. "
            "If the user asks anything about specific forecasts, predictions, demand figures, "
            "which items will sell most, or any question that requires actual data, "
            "do NOT make up or guess an answer. "
            "Instead, tell them clearly that no data has been uploaded yet and ask them to "
            "upload their sales CSV file first to get real AI-powered forecasts.\n\n"
        )

    txt += (
        "Keep answers concise and practical. Focus on food demand forecasting, "
        "food waste reduction, and restaurant operations. "
        "CRITICAL: When data is provided above (forecasts, top meals, category breakdowns), "
        "you MUST use those exact numbers to answer the user's question directly. "
        "Do NOT say you lack data or cannot predict — the forecast numbers are already computed "
        "and provided to you above. Cite specific meal IDs and forecasted order counts. "
        "Never fabricate numbers that are not in this prompt. "
        "If the user asks about something completely unrelated to food demand, politely redirect.\n\n"
        "LANGUAGE RULE (always applies): the user is a restaurant owner or manager, not a "
        "programmer or data scientist. NEVER use technical/software/ML jargon in your replies — "
        "no model or algorithm names (XGBoost, Random Forest, LSTM, or any other model name), "
        "no ML/stats terms (features, encodings, lag values, rolling mean/std, hyperparameters, "
        "training data, API, tokens), and no software-developer terms (database, backend, server, "
        "endpoint, JSON, session). If asked how the forecasting works, describe it conceptually "
        "in plain terms instead, e.g. 'it looks at your past sales patterns, seasonality, pricing, "
        "and promotions to predict what you'll likely need next week' — never name the underlying "
        "techniques. Speak the way a helpful business analyst would, not an engineer."
    )
    return txt


_DATA_KEYWORDS = {
    'predict', 'forecast', 'next week', 'which', 'most popular', 'top',
    'best sell', 'sell most', 'order', 'demand', 'drink', 'beverage',
    'food item', 'meal', 'menu', 'centre', 'center', 'location', 'cuisine',
    'category', 'popular', 'highest', 'lowest', 'week', 'quantity', 'how many',
    'how much', 'analysis', 'analyse', 'analyze', 'trend', 'insight',
}

_NO_DATA_REPLY = (
    "I don't have any sales data to work with yet. "
    "Please upload your historical sales CSV file first — "
    "then I can give you real, data-driven forecasts and insights specific to your restaurant."
)

# Shown to the user whenever the assistant backend can't be reached or errors out.
# Deliberately has zero technical detail (no service names, ports, or exception text) --
# the audience is a restaurant owner/manager, not a developer. Real diagnostics go to
# app.logger instead, never to the response body.
_ASSISTANT_UNAVAILABLE_MSG = (
    "The assistant is temporarily unavailable. Please try again in a moment — "
    "if this keeps happening, contact support."
)


_ITEM_NAME_MATCH_THRESHOLD = 0.68


def _find_items_by_name(query_text, session_meals, min_score=_ITEM_NAME_MATCH_THRESHOLD):
    """Fuzzy-matches free text against this session's uploaded item names. Returns every
    item scoring above `min_score`, best first -- there can genuinely be more than one
    match (e.g. "Calamari Ring" matching both "Calamari Rings #1445" and "...#2444")."""
    q = _vocab_norm(query_text)
    if not q or not session_meals:
        return []
    scored = []
    for m in session_meals:
        names = [n for n in [m.get('item_name'), *(m.get('alt_names') or [])] if n]
        if not names:
            continue
        score = max(_match_score(q, _vocab_norm(n)) for n in names)
        if score >= min_score:
            scored.append((score, m))
    scored.sort(key=lambda t: -t[0])
    seen, out = set(), []
    for _, m in scored:
        if m['meal_id'] not in seen:
            seen.add(m['meal_id'])
            out.append(m)
    return out


_STRONG_ITEM_MATCH_THRESHOLD = 0.90


def _has_strong_item_match(last_msg: str, session_meals: list) -> bool:
    """True when `last_msg` contains something close to an uploaded item's FULL name --
    not just a borderline fuzzy overlap with a short category-ish word. Lets a genuine
    specific-item question ("how popular is the Iced Lemon Tea?") keep winning over
    category-ranking priority even though the item's own name happens to contain a real
    category keyword ("...Tea" contains 'tea' -> Beverages); a weak/borderline match
    (like "top soup?" against "Tom Yum Soup", ~0.71) does NOT count as strong enough to
    override a clear category-ranking request. Mirrors _item_name_answer's own
    whole-message-then-n-gram search strategy, just at a much higher confidence bar."""
    if not session_meals:
        return False
    if _find_items_by_name(last_msg, session_meals, min_score=_STRONG_ITEM_MATCH_THRESHOLD):
        return True
    words = re.findall(r"[A-Za-z][A-Za-z'\-]{2,}", last_msg)[:15]
    for n in (4, 3, 2):
        for i in range(len(words) - n + 1):
            if _find_items_by_name(' '.join(words[i:i + n]), session_meals,
                                    min_score=_STRONG_ITEM_MATCH_THRESHOLD):
                return True
    return False


def _item_name_answer(last_msg: str, ctx: dict) -> str | None:
    """Deterministic, data-grounded answer for a question naming a specific uploaded menu
    item -- so the assistant can never claim an item "doesn't exist" just because it wasn't
    in the original trained dataset. Matching the whole message against short item names
    scores poorly (too much surrounding text dilutes the ratio -- "how popular are the
    Calamari Ring" vs "Calamari Rings" scores ~0.53), so this also tries 2-4 word phrases
    pulled from the message. Single-word chunks are deliberately excluded ("rice" alone
    scores ~0.75 against "Rice Bowl", which is too trigger-happy for a generic mention).
    Returns None (falls through to the existing chart/LLM pipeline) if nothing matches."""
    session_meals = ctx.get('session_meals') or []
    if not session_meals:
        return None

    matches = _find_items_by_name(last_msg, session_meals)
    if not matches:
        words = re.findall(r"[A-Za-z][A-Za-z'\-]{2,}", last_msg)[:15]
        for n in (4, 3, 2):
            for i in range(len(words) - n + 1):
                found = _find_items_by_name(' '.join(words[i:i + n]), session_meals)
                if found:
                    matches = found
                    break
            if matches:
                break
    if not matches:
        return None

    matches = matches[:5]
    lines = [f"Here's what your uploaded data shows for **{last_msg.strip()}**:", ""]
    for m in matches:
        name  = m.get('display_name') or m.get('item_name') or f"Meal #{m['meal_id']}"
        cat   = m.get('category') or 'Unknown category'
        cui   = m.get('cuisine') or ''
        total = m.get('total_orders') or 0
        nwf   = m.get('next_week_forecast')

        lines.append(f"**{name}** (#{m['meal_id']}) — {cat}{f', {cui}' if cui else ''}")
        lines.append(f"- Historical orders in your upload: {total:,}")
        if nwf is not None:
            fmin, fmax = m.get('forecast_min'), m.get('forecast_max')
            rng = f" (range {fmin:,}–{fmax:,})" if fmin is not None and fmax is not None and fmin != fmax else ""
            lines.append(f"- Next-week AI forecast: {nwf:,}{rng}")
        else:
            lines.append("- No forecast available for this item yet.")
        lines.append(
            "- Based on this item's full order history." if not m.get('is_cold_start') else
            "- **New item** — this wasn't in your original historical data, so this is an "
            "estimate based on its category, price, and your uploaded sales history."
        )
        lines.append("")

    return "\n".join(lines).strip()


def _load_session_forecasts_df(session_id, user_id) -> pd.DataFrame:
    if not session_id:
        return pd.DataFrame()
    conn = get_db()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT f.week, f.center_id, f.meal_id, f.category, f.cuisine, f.ai_forecast,
                       sm.item_name
                FROM forecasts f
                JOIN sessions s ON s.id = f.session_id
                LEFT JOIN session_meals sm ON sm.session_id = f.session_id AND sm.meal_id = f.meal_id
                WHERE f.session_id = %s AND s.user_id = %s
                """,
                (session_id, user_id),
            )
            rows = cur.fetchall()
    finally:
        conn.close()
    return pd.DataFrame(rows)


def _load_session_history_df(session_id, user_id) -> pd.DataFrame:
    if not session_id:
        return pd.DataFrame()
    conn = get_db()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT feat_json FROM sessions WHERE id = %s AND user_id = %s",
                (session_id, user_id),
            )
            row = cur.fetchone()
    finally:
        conn.close()
    if not row or not row.get('feat_json'):
        return pd.DataFrame()
    try:
        return pd.read_json(StringIO(row['feat_json']), orient='records')
    except Exception:
        return pd.DataFrame()


def _build_chat_context_from_session(session_id, user_id):
    """Rebuilds chat context straight from the DB for this session -- authoritative and
    always current, including uploaded item names and cold-start status -- rather than
    trusting whatever the client happens to have cached (which can go stale across page
    reloads, session switches, or a browser tab left open through a new upload)."""
    if not session_id:
        return None
    conn = get_db()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT total_records, weeks, centers, meals FROM sessions WHERE id = %s AND user_id = %s",
                (session_id, user_id),
            )
            meta = cur.fetchone()
            if not meta:
                return None

            cur.execute(
                "SELECT meal_id, item_name, category, cuisine, is_cold_start, total_orders, "
                "next_week_forecast, forecast_min, forecast_max, alt_names "
                "FROM session_meals WHERE session_id = %s",
                (session_id,),
            )
            meals = cur.fetchall()

            cur.execute(
                "SELECT week, SUM(ai_forecast) AS forecast FROM forecasts "
                "WHERE session_id = %s GROUP BY week ORDER BY week",
                (session_id,),
            )
            weekly_forecast = cur.fetchall()

            cur.execute(
                "SELECT week, category, SUM(ai_forecast) AS forecast FROM forecasts "
                "WHERE session_id = %s AND category IS NOT NULL GROUP BY week, category "
                "ORDER BY week, forecast DESC",
                (session_id,),
            )
            by_category = cur.fetchall()

            cur.execute(
                "SELECT center_id, SUM(ai_forecast) AS forecast FROM forecasts "
                "WHERE session_id = %s GROUP BY center_id ORDER BY forecast DESC LIMIT 30",
                (session_id,),
            )
            by_center = cur.fetchall()
    finally:
        conn.close()

    if not meals:
        # Legacy session created before session_meals existed, or genuinely empty -- let the
        # caller fall back to client-supplied context rather than presenting an emptier one.
        return None

    for m in meals:
        m['is_cold_start'] = bool(m['is_cold_start'])
        m['display_name']  = m['item_name'] or f"Meal #{m['meal_id']}"
        try:
            m['alt_names'] = json.loads(m.get('alt_names') or '[]')
        except (TypeError, ValueError):
            m['alt_names'] = []

    top_meals = sorted(meals, key=lambda m: m['total_orders'] or 0, reverse=True)[:30]

    forecasted = [m for m in meals if m['next_week_forecast'] is not None]
    top_next_week = sorted(forecasted, key=lambda m: m['next_week_forecast'], reverse=True)[:30]
    top_next_week = sorted(top_next_week, key=lambda m: (m['category'] or '', -m['next_week_forecast']))
    top_next_week = [{**m, 'forecast': m['next_week_forecast']} for m in top_next_week]
    bottom_next_week = sorted(forecasted, key=lambda m: m['next_week_forecast'])[:20]
    bottom_next_week = sorted(bottom_next_week, key=lambda m: (m['category'] or '', m['next_week_forecast']))
    bottom_next_week = [{**m, 'forecast': m['next_week_forecast']} for m in bottom_next_week]

    return {
        'summary':                  meta,
        'top_meals':                top_meals,
        'forecast':                 weekly_forecast,
        'forecast_top_next_week':   top_next_week,
        'forecast_bottom_next_week': bottom_next_week,
        'forecast_by_category':     by_category,
        'forecast_by_center':       by_center,
        'session_meals':            meals,  # full unfiltered list -- for the item-name lookup
    }



def _center_type_map() -> dict:
    return center_info.set_index('center_id')['center_type'].to_dict()


def _enrich_history_item_names(history_df: pd.DataFrame, ctx: dict) -> pd.DataFrame:
    """The feature-engineered historical dataframe (feat_json) carries meal_id/category/
    cuisine but not the uploaded item name -- engineer_features() only merges the trained
    meal_info/center_info reference tables, not session_meals. Enrich it here from ctx so
    historical rankings show real uploaded names instead of falling back to 'Meal #id'."""
    if history_df is None or history_df.empty or 'item_name' in history_df.columns:
        return history_df
    meals = ctx.get('session_meals') or []
    name_map = {m['meal_id']: m.get('item_name') for m in meals if m.get('item_name')}
    if not name_map:
        return history_df
    history_df = history_df.copy()
    history_df['item_name'] = history_df['meal_id'].map(name_map)
    return history_df


def _stream_text(text: str):
    chunk_size = 40
    for i in range(0, len(text), chunk_size):
        yield f"data: {json.dumps({'delta': text[i:i + chunk_size]})}\n\n"


# ── Chart-view disambiguation (LLM, structured output) ────────────────────────
# Deterministic keyword rules (chat_engine.detect_chart_view) resolve the vast majority
# of chart requests. This is a last-resort call, used ONLY when a message has already
# been confirmed to want a chart, doesn't read as a plain ranked-items request, and
# doesn't match any of the deterministic aggregate-view keywords -- it can pick which
# CHART SHAPE fits (category share vs. cuisine totals vs. weekly trend vs. locations),
# never a category, number, or ranking. Uses Ollama's structured-output JSON schema so
# the response is constrained to the exact view enum; any failure (unreachable, invalid
# JSON, value outside the enum) falls back to the deterministic default ('ranked_items').

_VIEW_SCHEMA = {
    'type': 'object',
    'properties': {'view': {'type': 'string', 'enum': list(ce.AGGREGATE_CHART_VIEWS)}},
    'required': ['view'],
}

_CHART_VIEW_PLANNER_SYSTEM = (
    "You are the chart-view classifier for a restaurant demand-forecasting app. The "
    "user's message has already been confirmed to want a chart, and doesn't clearly ask "
    "for a ranked list of individual menu items -- your only job is to pick which KIND of "
    "aggregate chart best matches the wording, from the given options. You do not decide "
    "numbers, categories, or item names -- only which chart shape fits the request.\n\n"
    "category_breakdown: share of demand per category (pie/doughnut). "
    "cuisine_breakdown: total demand per cuisine type. "
    "category_trend: how each category changes across weeks (multi-line). "
    "weekly_totals: total demand per week, no category split. "
    "location_breakdown: total demand per fulfilment centre/branch/outlet."
)


def _state_summary_for_llm(prev_state) -> str | None:
    """Short plain-text summary of the last resolved result, handed to the chart-view
    planner (and usable for other LLM calls) so a follow-up like 'and by cuisine
    instead?' can be resolved with the same context a person reading the transcript
    would have -- addresses the earlier gap where the chart planner only ever saw the
    latest message in isolation."""
    if not prev_state or not prev_state.last_result_rows:
        return None
    top = prev_state.last_result_rows[0]
    cats = ', '.join(prev_state.last_result_categories) if prev_state.last_result_categories else 'all menu items'
    return (f"Previously discussed a {prev_state.last_result_metric} ranking of {cats}, "
            f"top result: {top.get('display_name')} ({top.get('value')}).")


def _plan_chart_view_with_llm(last_msg: str, history_messages: list, state_summary: str | None) -> str | None:
    try:
        sys_messages = [{'role': 'system', 'content': _CHART_VIEW_PLANNER_SYSTEM}]
        if state_summary:
            sys_messages.append({'role': 'system', 'content': f'Conversation context so far: {state_summary}'})
        conv = [m for m in history_messages[-6:] if m.get('role') in ('user', 'assistant')]
        r = _requests.post(
            f'{OLLAMA_URL}/api/chat',
            json={
                'model': OLLAMA_MODEL,
                'messages': sys_messages + conv,
                'format': _VIEW_SCHEMA,
                'stream': False,
                'options': ce.deterministic_ollama_options(),
            },
            timeout=20,
        )
        r.raise_for_status()
        content = r.json().get('message', {}).get('content', '')
        parsed  = json.loads(content)
        view = parsed.get('view')
        if view in ce.AGGREGATE_CHART_VIEWS:
            return view
    except Exception:
        app.logger.info('Chart-view planning via Ollama failed or returned invalid output; using deterministic default')
    return None


def _llm_chart_insight(chart_data: dict, messages: list, ctx: dict) -> str:
    """LLM-authored 2-3 sentence commentary on an aggregate chart, grounded in the exact
    rendered data and validated before use (see chat_engine.validate_chart_narrative).
    Any contradiction, omission of the true top value, Ollama being unreachable, or
    malformed output all fall back to the deterministic template -- this call can never
    be the sole source of a factual claim reaching the user."""
    chart_summary = ce.chart_data_summary(chart_data)
    insight_msgs = [{'role': 'system', 'content': _chat_system(ctx)}] + messages + [
        {'role': 'user', 'content': (
            f'A "{chart_data["title"]}" chart was just shown to the user, with this '
            f'exact data: {chart_summary}. The data is already fully visible in the '
            'chart above -- do NOT repeat it as a table, code block, or list. Write '
            'ONLY 2-3 plain prose sentences that highlight the single most important '
            'insight from THIS data -- do not name any item/category as "top", "highest", '
            'or "most popular" other than the one with the largest value shown above. '
            'Do not describe anything not listed in this data. End with one practical '
            'action the restaurant manager should take. No markdown tables, no code '
            'fences, no bullet lists.'
        )}
    ]
    try:
        r = _requests.post(
            f'{OLLAMA_URL}/api/chat',
            json={'model': OLLAMA_MODEL, 'messages': insight_msgs, 'stream': True,
                  'options': ce.deterministic_ollama_options()},
            stream=True, timeout=60,
        )
        # Buffered rather than streamed token-by-token: this reply is short, and buffering
        # lets it be scrubbed (strip_data_dump) and *validated against the chart data*
        # before anything reaches the client -- not possible once tokens are already
        # streamed out one at a time.
        raw_parts = []
        for line in r.iter_lines():
            if not line:
                continue
            try:
                obj   = json.loads(line)
                delta = obj.get('message', {}).get('content', '')
                if delta:
                    raw_parts.append(delta)
                if obj.get('done'):
                    break
            except json.JSONDecodeError:
                pass
        cleaned = ce.strip_data_dump(''.join(raw_parts))
        ok, reason = ce.validate_chart_narrative(cleaned, chart_data)
        if ok and cleaned.strip():
            return cleaned
        app.logger.info('Chart insight from Ollama failed validation (%s); using deterministic template', reason)
    except Exception:
        app.logger.info('Chart insight LLM call failed; using deterministic template')
    return ce.render_chart_insight_fallback(chart_data)


# ── Grounded query orchestration ───────────────────────────────────────────────
# Everything a user could ask that has a factual/numeric answer (a ranking, a chart, an
# ordinal follow-up, a comparison, or a "is X still the top Y" check) is routed through
# here. Ollama is never asked to compute or recall the numbers themselves -- only (a)
# which aggregate chart shape fits ambiguous phrasing, and (b) optional prose commentary
# on an aggregate chart, both validated/overridable as above. Rankings, ordinal
# look-ups, comparisons, and ranked-item chart captions are 100% deterministic Python.

def _handle_grounded_query(spec: ce.QuerySpec, prev_state, session_id, user_id,
                            messages: list, ctx: dict) -> dict | None:
    # Metric ambiguity: default to `forecast` (this app's primary use case is forward
    # planning -- "what should I stock") and the reply always states the metric
    # explicitly ("By next week's AI forecast...") rather than silently guessing. The
    # "genuinely cannot be inferred, ask" path is reserved below for referent
    # resolution failures, where there truly is no reasonable default to fall back on.
    if spec.metric is None:
        spec.metric = ce.DEFAULT_METRIC

    forecast_df = _load_session_forecasts_df(session_id, user_id)
    history_df  = _enrich_history_item_names(_load_session_history_df(session_id, user_id), ctx)

    def _compute(s):
        # A follow-up (ordinal/still_top/compare) must be answered against the SAME
        # thing the previous turn ranked -- if that was locations (spec.rank_by ==
        # 'location', inherited from the prior turn's QuerySpec), it must go through
        # compute_location_ranking, not the item-ranking pipeline. Answering a location
        # follow-up with compute_grounded_result silently reinterprets a center id as a
        # meal id and returns the wrong entity entirely (e.g. "what about the second
        # one?" after "top 2 locations" returned the second MENU ITEM instead of the
        # second location).
        if s.rank_by == 'location':
            return ce.compute_location_ranking(s, session_id, forecast_df, history_df,
                                                center_type_map=_center_type_map())
        return ce.compute_grounded_result(s, session_id, forecast_df, history_df)

    def _resolve_referent():
        if prev_state and prev_state.last_referenced_item:
            return prev_state.last_referenced_item
        if prev_state and prev_state.last_result_rows:
            return prev_state.last_result_rows[0]
        return None

    if spec.intent == 'ordinal':
        gr = _compute(spec)
        if gr is None:
            return {'text': ce.NO_MATCHING_DATA_MSG}
        text = ce.render_ordinal_narrative(gr, spec.ordinal_rank)
        idx = (len(gr.rows) - 1) if spec.ordinal_rank == -1 else (spec.ordinal_rank - 1)
        referenced = gr.rows[idx] if 0 <= idx < len(gr.rows) else None
        return {'text': text, 'grounded': gr, 'referenced_item': referenced}

    if spec.intent == 'still_top':
        referent = _resolve_referent()
        if referent is None:
            return {'text': ce.CLARIFY_NO_REFERENT_MSG}
        gr = _compute(spec)
        if gr is None:
            return {'text': ce.NO_MATCHING_DATA_MSG}
        return {'text': ce.render_still_top_narrative(referent, gr), 'grounded': gr, 'referenced_item': referent}

    if spec.intent == 'compare':
        # A fresh 'compare CategoryA and CategoryB' (or 'CategoryA vs food') request
        # names its own two sides directly -- it does NOT need a referent from a prior
        # turn, and answering it against an unrelated "it" left over from a previous
        # message would be wrong. This must be checked before the referent-based path
        # below, which is for the different, follow-up case ("compare it with X").
        if ce.is_category_comparison(spec.raw_message):
            totals = ce.compute_category_totals(spec, session_id, forecast_df, history_df)
            if not totals:
                return {'text': ce.NO_MATCHING_DATA_MSG}
            cats = ce.extract_categories(spec.raw_message)
            if len(cats) >= 2:
                cat_a, cat_b = cats[0], cats[1]
                label_b = cat_b
            else:
                cat_a = cats[0]
                label_b = 'Food (all other categories)'
            val_a = totals.get(cat_a, 0)
            val_b = totals.get(cat_b, 0) if len(cats) >= 2 else sum(v for k, v in totals.items() if k != cat_a)
            item_a = {'display_name': cat_a, 'value': val_a}
            item_b = {'display_name': label_b, 'value': val_b}
            text = ce.render_compare_narrative(item_a, item_b, spec.metric)
            chart = None
            if spec.wants_chart:
                chart = ce.two_slice_doughnut(cat_a, val_a, label_b, val_b, f'{cat_a} vs. {label_b}')
            return {'text': text, 'chart': chart}

        referent = _resolve_referent()
        if referent is None:
            return {'text': ce.CLARIFY_NO_REFERENT_MSG}
        # "the top food item" is a colloquial, non-category-specific counterpart (much
        # like the "food vs drinks" doughnut split elsewhere) -- there's no dedicated
        # "non-beverage" category to query, so this resolves to the overall top item for
        # any *named* category in the hint, or the overall top item across everything if
        # none was named. If that happens to be the same item as the referent (e.g. no
        # distinct category could be inferred), a wrong same-item comparison is refused
        # in favor of asking, rather than presenting a comparison that doesn't compare.
        other_category = ce.extract_category(spec.compare_item_hint or '')
        other_spec = ce.QuerySpec(intent='rank', metric=spec.metric,
                                   categories=[other_category] if other_category else [],
                                   requested_count=1, sort_direction='desc',
                                   # Mirror the referent's own scope (item vs. location) --
                                   # a location follow-up ("compare it with the top
                                   # location") must compare against the top LOCATION, not
                                   # silently fall back to the top menu item.
                                   rank_by=spec.rank_by)
        gr_other = _compute(other_spec)
        if gr_other is None or not gr_other.rows:
            return {'text': ce.CLARIFY_NO_REFERENT_MSG}
        item_b = gr_other.rows[0]
        if item_b['item_id'] == referent.get('item_id'):
            return {'text': ce.CLARIFY_NO_REFERENT_MSG}
        text = ce.render_compare_narrative(referent, item_b, spec.metric)
        chart = None
        if spec.wants_chart:
            chart = ce.two_slice_doughnut(referent['display_name'], referent['value'],
                                           item_b['display_name'], item_b['value'],
                                           f'{referent["display_name"]} vs. {item_b["display_name"]}')
        return {'text': text, 'chart': chart, 'referenced_item': referent}

    if spec.intent == 'per_location':
        # "most popular meal AT EACH of those locations" -- a per-location breakdown,
        # not the flat "rank_by='both'" pair of independent top-N lists. "those
        # locations" resolves against the previous turn's location ranking (its exact
        # center ids, in that same order) when this message doesn't name explicit
        # centre numbers itself. Reads from last_location_rows specifically (never
        # last_result_rows, which may hold ITEM rows from an intervening 'rank_by=item'
        # or 'both' turn) so meal ids can never be mistaken for center ids.
        center_ids = list(spec.center_ids)
        if not center_ids and prev_state and prev_state.last_location_rows:
            center_ids = [r['item_id'] for r in prev_state.last_location_rows]
        if not center_ids:
            # No explicit centre numbers and nothing to inherit -- e.g. this is the
            # very first message of the conversation. Rather than making the user run
            # a separate "top locations" query first just to unlock this one, fall back
            # to this session's own top locations (same metric/category scope).
            gr_loc = ce.compute_location_ranking(spec, session_id, forecast_df, history_df,
                                                  center_type_map=_center_type_map())
            if gr_loc:
                center_ids = [r['item_id'] for r in gr_loc.rows]
        if not center_ids:
            return {'text': ce.NO_MATCHING_DATA_MSG}
        breakdown = ce.compute_top_item_per_group(spec, session_id, forecast_df, history_df,
                                                    center_ids, center_type_map=_center_type_map(),
                                                    top_n_per_group=spec.requested_count)
        text = ce.render_location_breakdown_narrative(breakdown, spec.metric, spec.sort_direction)
        referenced = next((e['top_items'][0] for e in breakdown if e['top_items']), None)
        return {'text': text, 'referenced_item': referenced}

    # default: 'rank'
    if spec.wants_chart:
        # Resolution order: (1) deterministic keywords in THIS message, (2) a view
        # inherited from the previous turn via structured conversation state (e.g. a
        # bare "show that as a chart" follow-up with no new shape-indicating words),
        # (3) the LLM, only as a last resort for genuinely new, ambiguous phrasing. This
        # keeps simple chart follow-ups answerable purely from state, with zero extra
        # Ollama round-trips or risk of the model re-guessing a shape it has no signal for.
        view = ce.detect_chart_view(spec.raw_message)
        if view is None:
            view = spec.chart_view
        if view is None and not ce.mentions_ranking_language(spec.raw_message):
            view = _plan_chart_view_with_llm(spec.raw_message, messages, _state_summary_for_llm(prev_state))
        spec.chart_view = view  # persisted into conversation state by the caller
        if view and view != 'ranked_items':
            df = forecast_df if spec.metric == 'forecast' else history_df
            as_line = any(kw in spec.raw_message.lower()
                          for kw in ('trend', 'over time', 'progression', 'line', 'time series'))
            chart = ce.build_aggregate_chart(view, spec.metric, df, spec,
                                              center_type_map=_center_type_map(), as_line=as_line)
            if chart is None:
                return {'text': ce.NO_MATCHING_DATA_MSG}
            return {'chart': chart, 'insight_mode': 'llm'}
        spec.chart_view = 'ranked_items'

    # Plain-text (non-chart) ranking: what's being ranked -- menu item(s), location(s),
    # or both -- comes from spec.rank_by (see resolve_query_spec). Without this, a
    # question naming a location/branch/centre ("most popular location") silently fell
    # through to the generic item ranking, with the word "location" entirely ignored.
    if spec.rank_by == 'location':
        gr_loc = ce.compute_location_ranking(spec, session_id, forecast_df, history_df,
                                              center_type_map=_center_type_map())
        if gr_loc is None:
            return {'text': ce.NO_MATCHING_DATA_MSG}
        return {'text': ce.render_ranking_narrative(gr_loc, spec, entity_label='locations'),
                'grounded': gr_loc, 'location_rows': gr_loc.rows, 'referenced_item': gr_loc.rows[0]}

    if spec.rank_by == 'both':
        gr_item = _compute(spec)
        gr_loc  = ce.compute_location_ranking(spec, session_id, forecast_df, history_df,
                                               center_type_map=_center_type_map())
        parts, referenced = [], None
        if gr_item is not None:
            parts.append(ce.render_ranking_narrative(gr_item, spec))
            referenced = gr_item.rows[0]
        if gr_loc is not None:
            parts.append(ce.render_ranking_narrative(gr_loc, spec, entity_label='locations'))
            referenced = referenced or gr_loc.rows[0]
        if not parts:
            return {'text': ce.NO_MATCHING_DATA_MSG}
        # 'grounded' carries the item rows (for a plain ordinal follow-up like "what
        # about the second one" to keep working against the item list), while the
        # location rows are surfaced separately via 'location_rows' -- NOT reused as
        # 'grounded' -- so a later "at each of those locations" can't mistake meal ids
        # for center ids.
        return {'text': '\n\n'.join(parts), 'grounded': gr_item,
                'location_rows': (gr_loc.rows if gr_loc else None), 'referenced_item': referenced}

    gr = _compute(spec)
    if gr is None:
        return {'text': ce.NO_MATCHING_DATA_MSG}
    if spec.wants_chart and gr.chart:
        return {'chart': gr.chart, 'grounded': gr, 'referenced_item': gr.rows[0], 'insight_mode': 'deterministic'}
    return {'text': ce.render_ranking_narrative(gr, spec), 'grounded': gr, 'referenced_item': gr.rows[0]}


@app.route('/chat', methods=['POST'])
@login_required
def chat():
    body     = request.get_json(force=True)
    messages = body.get('messages', [])
    ctx      = body.get('context', {}) or {}
    session_id = body.get('session_id')
    if not messages:
        return jsonify({'error': 'No messages provided'}), 400

    # Prefer rebuilding context straight from the DB for this session over trusting whatever
    # the client sent -- prevents answering from a stale/previous dataset (e.g. old chat tab
    # left open through a new upload). Only overrides when the rebuild actually found data;
    # otherwise keeps the client-supplied context (e.g. a legacy session predating this table).
    if session_id:
        server_ctx = _build_chat_context_from_session(session_id, current_user.id)
        if server_ctx:
            ctx = server_ctx

    last_msg = messages[-1].get('content', '') if messages else ''

    # Block data-specific questions when no file has been uploaded
    if not ctx.get('summary'):
        if any(kw in last_msg.lower() for kw in _DATA_KEYWORDS):
            def _fixed():
                for word in _NO_DATA_REPLY.split(' '):
                    yield f"data: {json.dumps({'delta': word + ' '})}\n\n"
                yield "data: [DONE]\n\n"
            return Response(stream_with_context(_fixed()), mimetype='text/event-stream',
                            headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'})

    # Deterministic item-name lookup takes priority over everything else below -- if the
    # user names a specific uploaded menu item, ground the answer in its real current-session
    # numbers instead of letting the LLM guess (or wrongly claim the item doesn't exist just
    # because it wasn't in the original trained dataset).
    #
    # EXCEPT when the message reads as a category-scoped ranking question ("top soup?",
    # "best sandwich?") -- a category name can score above the fuzzy item-name match
    # threshold against an item whose own name largely consists of that category word
    # (e.g. "top soup?" scores 0.71 against "Tom Yum Soup", above the 0.68 cutoff), which
    # used to silently answer with that one item's stats instead of ranking the category.
    # A recognized category + ranking language together is a stronger, more specific
    # signal than a generic fuzzy text match, so it takes priority.
    #
    # ALSO except a fresh category-vs-category comparison ("compare pizza and pasta") --
    # for the same reason, plus that "pasta" alone can fuzzy-match a specific pasta dish
    # strongly enough to hijack the whole comparison into an unrelated single-item answer
    # (e.g. "compare pizza and pasta" silently became "here's Alfredo Pasta's stats",
    # never mentioning pizza at all).
    #
    # BUT a genuine specific-item question still wins even then, if the message contains
    # something close to that item's FULL name (_has_strong_item_match) -- otherwise
    # "how popular is the Iced Lemon Tea?" would get misrouted into a generic Beverages
    # ranking just because "Tea" (part of the item's own name) is also a category word.
    looks_like_category_ranking = bool(ce.extract_category(last_msg)) and ce.mentions_ranking_language(last_msg)
    looks_like_category_compare = ce.is_category_comparison(last_msg)
    session_meals = ctx.get('session_meals') or []
    skip_item_lookup = (looks_like_category_ranking or looks_like_category_compare) \
        and not _has_strong_item_match(last_msg, session_meals)
    item_answer = (_item_name_answer(last_msg, ctx)
                   if ctx.get('summary') and not skip_item_lookup else None)

    # Structured, server-side conversation state (per user + session -- see chat_engine.
    # ConversationStateStore) resolves contextual follow-ups ("the second one", "show
    # that as a chart", "what about historically") into a concrete QuerySpec, then every
    # ranking/chart/comparison/tie/no-data answer is computed from the full authoritative
    # session dataset in chat_engine, never guessed by the LLM.
    grounded_answer = None
    spec = None
    prev_state = None
    if item_answer is None and ctx.get('summary'):
        prev_state = ce.conversation_store.get(current_user.id, session_id)
        spec = ce.resolve_query_spec(last_msg, prev_state.last_spec if prev_state else None)
        if ce.wants_grounded_ranking(last_msg, spec, has_prior_grounded_state=prev_state is not None):
            grounded_answer = _handle_grounded_query(spec, prev_state, session_id, current_user.id, messages, ctx)
            if grounded_answer is not None and session_id:
                gr = grounded_answer.get('grounded')
                new_state = ce.ConversationState(
                    user_id=current_user.id,
                    session_id=session_id,
                    last_spec=spec.to_state_dict(),
                    last_result_rows=(gr.rows if gr else (prev_state.last_result_rows if prev_state else None)),
                    last_result_metric=spec.metric,
                    last_result_categories=list(spec.categories),
                    last_chart=grounded_answer.get('chart'),
                    last_referenced_item=(grounded_answer.get('referenced_item')
                                           or (prev_state.last_referenced_item if prev_state else None)),
                    last_location_rows=(grounded_answer.get('location_rows')
                                         or (prev_state.last_location_rows if prev_state else None)),
                    updated_at=ce.now_iso(),
                )
                ce.conversation_store.set(current_user.id, session_id, new_state)
                app.logger.info(
                    'chat: session=%s user=%s intent=%s metric=%s categories=%s source=%s',
                    session_id, current_user.id, spec.intent, spec.metric, spec.categories,
                    (gr.source if gr else 'aggregate_chart'),
                )

    def generate():
        if item_answer:
            yield from _stream_text(item_answer)
            yield "data: [DONE]\n\n"
            return

        if grounded_answer is not None:
            chart = grounded_answer.get('chart')
            text  = grounded_answer.get('text')
            if chart:
                yield f"data: {json.dumps({'chart': chart})}\n\n"
                if text is None:
                    if grounded_answer.get('insight_mode') == 'llm':
                        text = _llm_chart_insight(chart, messages, ctx)
                    else:
                        text = ce.render_chart_insight_fallback(chart)
                yield from _stream_text(text)
                yield "data: [DONE]\n\n"
                return
            yield from _stream_text(text or ce.NO_MATCHING_DATA_MSG)
            yield "data: [DONE]\n\n"
            return

        ollama_messages = [{'role': 'system', 'content': _chat_system(ctx)}] + messages
        try:
            r = _requests.post(
                f'{OLLAMA_URL}/api/chat',
                json={'model': OLLAMA_MODEL, 'messages': ollama_messages, 'stream': True,
                      'options': ce.deterministic_ollama_options(num_predict=600)},
                stream=True,
                timeout=120,
            )
            for line in r.iter_lines():
                if not line:
                    continue
                try:
                    obj   = json.loads(line)
                    delta = obj.get('message', {}).get('content', '')
                    if delta:
                        yield f"data: {json.dumps({'delta': delta})}\n\n"
                    if obj.get('done'):
                        break
                except json.JSONDecodeError:
                    pass
        except _requests.exceptions.ConnectionError:
            app.logger.error('Chat assistant unreachable: Ollama connection failed')
            yield f"data: {json.dumps({'error': _ASSISTANT_UNAVAILABLE_MSG})}\n\n"
        except Exception:
            app.logger.exception('Chat assistant error')
            yield f"data: {json.dumps({'error': _ASSISTANT_UNAVAILABLE_MSG})}\n\n"
        yield "data: [DONE]\n\n"

    return Response(
        stream_with_context(generate()),
        mimetype='text/event-stream',
        headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'},
    )


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == '__main__':
    init_db()
    app.run(debug=True, use_reloader=False)
