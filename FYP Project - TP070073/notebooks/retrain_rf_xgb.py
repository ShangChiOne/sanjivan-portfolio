"""
Re-exports rf_model.pkl, xgb_model.pkl, and label_encoders.pkl using the scikit-learn /
xgboost versions pinned in requirements.txt (1.8.0 / 2.0.3), fixing the
InconsistentVersionWarning raised at load time by the originals (trained under
scikit-learn 1.6.1).

Reproduces the exact feature engineering / training steps from
notebooks/demand_forecasting_training.ipynb cells 9, 12-14, 16, 18-19, 25 (merge,
discount_pct, label encoding, lag/rolling features, time-based split, RF + XGBoost
training, artifact save) -- LSTM artifacts are untouched (no version mismatch was
observed for them; see the model-compat report).

Run from the project root:
    python notebooks/retrain_rf_xgb.py
"""
import json
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, '..', 'dataset')
OUTPUT_PATH = os.path.join(BASE_DIR, '..', 'models')

train = pd.read_csv(os.path.join(DATASET_PATH, 'train.csv'))
meal_info = pd.read_csv(os.path.join(DATASET_PATH, 'meal_info.csv'))
center_info = pd.read_csv(os.path.join(DATASET_PATH, 'fulfilment_center_info.csv'))

df = train.merge(meal_info, on='meal_id', how='left')
df = df.merge(center_info, on='center_id', how='left')

df['discount_pct'] = (df['base_price'] - df['checkout_price']) / df['base_price']
df['discount_pct'] = df['discount_pct'].clip(lower=0)

label_encoders = {}
for col in ['category', 'cuisine', 'center_type']:
    le = LabelEncoder()
    df[col + '_enc'] = le.fit_transform(df[col].astype(str))
    label_encoders[col] = le

df = df.sort_values(['center_id', 'meal_id', 'week']).reset_index(drop=True)
grp = df.groupby(['center_id', 'meal_id'])['num_orders']
df['lag_1'] = grp.shift(1)
df['lag_2'] = grp.shift(2)
df['lag_3'] = grp.shift(3)
df['rolling_mean_3'] = grp.shift(1).rolling(window=3, min_periods=1).mean().values
df['rolling_std_3'] = grp.shift(1).rolling(window=3, min_periods=1).std().fillna(0).values
df = df.dropna(subset=['lag_1', 'lag_2', 'lag_3'])

FEATURE_COLS = [
    'week', 'center_id', 'meal_id', 'checkout_price', 'base_price', 'discount_pct',
    'emailer_for_promotion', 'homepage_featured', 'op_area', 'city_code', 'region_code',
    'center_type_enc', 'category_enc', 'cuisine_enc',
    'lag_1', 'lag_2', 'lag_3', 'rolling_mean_3', 'rolling_std_3',
]
TARGET_COL = 'num_orders'

TRAIN_END_WEEK = 130
train_df = df[df['week'] <= TRAIN_END_WEEK]
val_df = df[df['week'] > TRAIN_END_WEEK]
X_train, y_train = train_df[FEATURE_COLS], train_df[TARGET_COL]
X_val, y_val = val_df[FEATURE_COLS], val_df[TARGET_COL]
print(f'Training set : {X_train.shape} (weeks 1-{TRAIN_END_WEEK})')
print(f'Validation set: {X_val.shape} (weeks {TRAIN_END_WEEK + 1}-{int(df["week"].max())})')

print('Training Random Forest...')
rf_model = RandomForestRegressor(
    n_estimators=200, max_depth=15, min_samples_leaf=5, n_jobs=-1, random_state=42,
)
rf_model.fit(X_train, y_train)
rf_preds = rf_model.predict(X_val)
rf_mae = mean_absolute_error(y_val, rf_preds)
rf_rmse = np.sqrt(mean_squared_error(y_val, rf_preds))
print(f'Random Forest  -- MAE: {rf_mae:.2f} | RMSE: {rf_rmse:.2f}')

print('Training XGBoost...')
xgb_model = XGBRegressor(
    n_estimators=500, learning_rate=0.05, max_depth=6, subsample=0.8,
    colsample_bytree=0.8, min_child_weight=5, tree_method='hist',
    random_state=42, n_jobs=-1,
)
xgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=100)
xgb_preds = xgb_model.predict(X_val)
xgb_mae = mean_absolute_error(y_val, xgb_preds)
xgb_rmse = np.sqrt(mean_squared_error(y_val, xgb_preds))
print(f'XGBoost  -- MAE: {xgb_mae:.2f} | RMSE: {xgb_rmse:.2f}')

joblib.dump(xgb_model, os.path.join(OUTPUT_PATH, 'xgb_model.pkl'))
joblib.dump(rf_model, os.path.join(OUTPUT_PATH, 'rf_model.pkl'))
joblib.dump(label_encoders, os.path.join(OUTPUT_PATH, 'label_encoders.pkl'))
with open(os.path.join(OUTPUT_PATH, 'feature_columns.json'), 'w') as f:
    json.dump(FEATURE_COLS, f)

with open(os.path.join(OUTPUT_PATH, 'model_metrics.json')) as f:
    metrics = json.load(f)
metrics['random_forest'] = {'MAE': round(float(rf_mae), 4), 'RMSE': round(float(rf_rmse), 4)}
metrics['xgboost'] = {'MAE': round(float(xgb_mae), 4), 'RMSE': round(float(xgb_rmse), 4)}
with open(os.path.join(OUTPUT_PATH, 'model_metrics.json'), 'w') as f:
    json.dump(metrics, f, indent=2)

print('Artifacts re-saved to', OUTPUT_PATH)
