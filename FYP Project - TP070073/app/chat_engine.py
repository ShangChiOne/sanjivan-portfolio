"""
Deterministic, data-grounded chat engine for the ForecastIQ assistant.

Design principle: Ollama (the local LLM) never computes rankings, comparisons, ties, or
numbers. Every factual claim -- which item is "top", by how much, in what order -- is
computed here in plain Python from the session's authoritative stored data
(`GroundedResult`). A chart and its narrative are always two renderings of the *same*
`GroundedResult`, so they cannot contradict each other the way the reported bug did
(chart showing Iced Lemon Tea highest, prose claiming Thai Milk Tea was top).

Where the LLM is still used (chart-view classification for ambiguous phrasing, and prose
commentary on aggregate/breakdown charts), its output is validated against the exact
chart data before being shown, and discarded in favor of a deterministic template on any
mismatch (see `validate_chart_narrative`).

This module has zero network/DB dependencies -- everything here is pure functions and
dataclasses operating on plain Python/pandas data the caller (app.py) supplies. That
makes it fully unit-testable without Flask, MySQL, or a running Ollama instance.
"""
from __future__ import annotations

import os
import re
import threading
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Optional

import pandas as pd

# ─────────────────────────────────────────────────────────────────────────────
# Ollama call options -- centralized & environment-configurable
#
# Every factual/classification Ollama call in this app (chart-view planning, chart
# insight prose, column/vocab mapping suggestions, and the general chat reply) should
# route its `options` through here, so behavior is controlled from one place and via
# env vars rather than scattered inline dicts. Conservative defaults (temperature 0,
# fixed seed) reduce -- but do NOT eliminate -- run-to-run variance; the backend
# re-validates model output against GroundedResult/chart data regardless of these
# settings (see validate_chart_narrative). Temperature 0 alone is not treated as a
# correctness guarantee anywhere in this module.
# ─────────────────────────────────────────────────────────────────────────────

def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


OLLAMA_TEMPERATURE = _env_float('OLLAMA_TEMPERATURE', 0)
OLLAMA_SEED        = _env_int('OLLAMA_SEED', 42)
OLLAMA_NUM_CTX     = _env_int('OLLAMA_NUM_CTX', 8192)
OLLAMA_NUM_PREDICT = _env_int('OLLAMA_NUM_PREDICT', 400)


def deterministic_ollama_options(**overrides) -> dict:
    """Centralized sampling options for every Ollama call. Pass overrides for a
    call-specific tweak (e.g. a larger num_predict for the free-form chat reply)
    without duplicating the whole options dict at each call site."""
    opts = {
        'temperature': OLLAMA_TEMPERATURE,
        'seed': OLLAMA_SEED,
        'num_ctx': OLLAMA_NUM_CTX,
        'num_predict': OLLAMA_NUM_PREDICT,
    }
    opts.update(overrides)
    return opts


# ─────────────────────────────────────────────────────────────────────────────
# Metric semantics
#
# Documented rule (see CHATBOT_CHARTS.md): "next week" / "forecast" / "will sell" /
# "should stock" -> forecast metric. "historically" / "past" / "actual" / "has sold"
# -> historical metric. A follow-up inherits the previous turn's metric unless the
# user explicitly names the other one. A standalone ambiguous request (no metric
# words, no prior turn to inherit from) defaults to `forecast` -- this app's primary
# use case is forward planning ("what should I stock") -- and the reply always states
# the metric explicitly ("By next week's AI forecast...") rather than silently guessing.
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_METRIC = 'forecast'

METRIC_PHRASES = {
    'forecast':   "By next week's AI forecast",
    'historical': 'By total historical orders in this upload',
}
METRIC_PERIOD_LABEL = {
    'forecast':   'Next-Week AI Forecast',
    'historical': 'Historical Orders',
}

NO_MATCHING_DATA_MSG = (
    "I don't have any data matching that in this upload, so I won't guess. "
    "Try a different item, category, or time scope."
)

CLARIFY_NO_REFERENT_MSG = (
    "I don't have a previous result to reference yet in this conversation -- could you "
    "ask the full question (e.g. which item or category, and forecast or historical)?"
)


# ─────────────────────────────────────────────────────────────────────────────
# Shared keyword/regex vocabulary (single source of truth -- previously duplicated
# across the legacy keyword-rule chart detector and the LLM tool-planner path)
# ─────────────────────────────────────────────────────────────────────────────

CATEGORY_KEYWORDS: dict[str, str] = {
    'drink':      'Beverages',
    'drinks':     'Beverages',
    'beverage':   'Beverages',
    'beverages':  'Beverages',
    'juice':      'Beverages',
    'coffee':     'Beverages',
    'tea':        'Beverages',
    'rice bowl':  'Rice Bowl',
    'pasta':      'Pasta',
    'sandwich':   'Sandwich',
    'salad':      'Salad',
    'pizza':      'Pizza',
    'biryani':    'Biryani',
    'soup':       'Soup',
    'seafood':    'Seafood',
    'fish':       'Fish',
    'dessert':    'Desert',
    'desert':     'Desert',
    'starter':    'Starters',
    'starters':   'Starters',
    'snack':      'Other Snacks',
    'snacks':     'Other Snacks',
}

_COMPARE_WORDS_RE = re.compile(r'\b(vs\.?|versus|compare|comparison|between)\b', re.I)
_GENERIC_FOOD_RE  = re.compile(r'\bfoods?\b', re.I)

_CHART_TRIGGER_KEYWORDS = {
    'chart', 'graph', 'plot', 'visualize', 'visualise',
    'pie', 'doughnut', 'bar chart', 'line chart', 'histogram',
    'statistics', 'stats', 'breakdown', 'distribution',
    'compare', 'comparison', 'trend',
}

_FORECAST_RE = re.compile(
    r'\b(next\s*week|forecast(?:ed|s)?|will\s+sell|should\s+i\s+stock|should\s+stock|'
    r'should\s+i\s+order|should\s+i\s+prepare|upcoming|going\s+to\s+sell|'
    r'expect(?:ed)?\s+to\s+sell|predicted?|prediction)\b',
    re.I,
)
_HISTORICAL_RE = re.compile(
    r'\b(historically|history|historical|in\s+the\s+past|past\s+orders|actual(?:ly)?\s+sold|'
    r'has\s+sold|have\s+sold|already\s+sold|previously|so\s+far|to\s+date|recorded|'
    r'real\s+orders|actual\s+orders)\b',
    re.I,
)
_BOTTOM_RE = re.compile(
    r'\b(bottom|worst|lowest|least\s+popular|least\s+ordered|underperform(?:ing)?|'
    r'weakest|least\s+in\s+demand|least\s+demand|'
    # verb-conjugation coverage ("sells the least", "sold the least", "performs
    # worst") -- without these, "what sells the least?" silently fell through to
    # the *default* (desc/top) direction instead of asc/bottom, returning the best
    # sellers with full confidence when the user asked for the worst ones.
    r'sell(?:s|ing)?\s+(?:the\s+)?(?:least|worst)|sold\s+(?:the\s+)?(?:least|worst)|'
    r'perform(?:s|ing)?\s+(?:the\s+)?worst|worst[- ]?sell(?:s|ing|er)?|'
    # "stock/order/buy/prepare LESS of" -- identifying overstock risk is a distinct,
    # common planning question ("what should I stock less of?"), and previously
    # silently defaulted to the top (best-selling) direction instead of the bottom.
    r'(?:stock|order|buy|prepare|make)\s+less(?:\s+of)?)\b',
    re.I,
)
_TOP_RE = re.compile(
    r'\b(top|best|highest|most\s+popular|most\s+ordered|strongest|most\s+in\s+demand|'
    r'best[- ]?sell(?:s|ing|er)?|'
    r'sell(?:s|ing)?\s+(?:the\s+)?(?:most|best)|sold\s+(?:the\s+)?(?:most|best)|'
    r'perform(?:s|ing)?\s+(?:the\s+)?best|'
    r'(?:stock|order|buy|prepare|make)\s+more(?:\s+of)?)\b',
    re.I,
)
# Same as _TOP_RE minus the bare "top" alternative. Bare "top" is genuinely ambiguous --
# it's used both as a direction word ("top sellers") and as a count prefix ("top 5 ..."),
# so "top 5 least popular meals" must not let it outvote the explicit "least popular".
# Every other alternative here (best/highest/most popular/...) is unambiguous and should
# win even against an explicit bottom word in the rare case both appear.
_TOP_EXPLICIT_RE = re.compile(
    r'\b(best|highest|most\s+popular|most\s+ordered|strongest|most\s+in\s+demand|'
    r'best[- ]?sell(?:s|ing|er)?|'
    r'sell(?:s|ing)?\s+(?:the\s+)?(?:most|best)|sold\s+(?:the\s+)?(?:most|best)|'
    r'perform(?:s|ing)?\s+(?:the\s+)?best|'
    r'(?:stock|order|buy|prepare|make)\s+more(?:\s+of)?)\b',
    re.I,
)
_CENTER_RE = re.compile(
    r'\b(?:centre|center|location|branch|outlet|store)s?\s*#?\s*(\d+)\b', re.I,
)
# "which location should I focus on?" / "what should I prioritize?" ask for a single
# recommendation, not a ranked list -- without this they fell through to the same
# default-5 top-N ranking as a bare "top locations" request, silently ignoring that the
# user asked for ONE specific answer ("should I focus on") rather than a listing.
_SINGLE_ANSWER_RE = re.compile(
    r'\b(?:which\s+\w+\s+should\s+(?:i|we)\s+(?:focus\s+on|prioriti[sz]e|pick|choose|go\s+with)|'
    r'what\s+should\s+(?:i|we)\s+(?:focus\s+on|prioriti[sz]e)|'
    r'where\s+should\s+(?:i|we)\s+focus)\b',
    re.I,
)
_CHART_REQUEST_RE = re.compile(
    r'\b(chart|graph|plot|visuali[sz]e|as a chart|as a graph|as a plot)\b', re.I,
)
_COMPARE_INTENT_RE = re.compile(r'\b(compare|compared to|vs\.?|versus)\b', re.I)
_STILL_RE = re.compile(r'\bstill\b', re.I)
_PER_GROUP_RE = re.compile(r'\b(each|every|per|all)\b', re.I)

LOCATION_KEYWORDS = ('location', 'locations', 'center', 'centers', 'centre', 'centres',
                      'branch', 'branches', 'outlet', 'outlets', 'store', 'stores')
_ITEM_WORD_RE = re.compile(r'\b(item|items|meal|meals|menu|dish|dishes|product|products)\b', re.I)


def mentions_location(msg: str) -> bool:
    lower = msg.lower()
    return any(kw in lower for kw in LOCATION_KEYWORDS)

_ORDINAL_WORDS = {
    'first': 1, '1st': 1, 'second': 2, '2nd': 2, 'third': 3, '3rd': 3,
    'fourth': 4, '4th': 4, 'fifth': 5, '5th': 5, 'sixth': 6, '6th': 6,
    'seventh': 7, '7th': 7, 'eighth': 8, '8th': 8, 'ninth': 9, '9th': 9,
    'tenth': 10, '10th': 10, 'last': -1,
}
_ORDINAL_RE = re.compile(
    r'\b(?:the\s+)?(' + '|'.join(re.escape(w) for w in _ORDINAL_WORDS) + r')\s*(?:one|item|place)?\b',
    re.I,
)
_FOLLOWUP_MARKERS_RE = re.compile(
    r'\b(that|this|it|those|these|what about|show that|same (?:one|thing|item)|again|too|instead)\b',
    re.I,
)


def extract_categories(msg: str) -> list[str]:
    """Detect every real menu category mentioned (e.g. 'beverages', 'pasta'). A term
    that doesn't map to an actual category (e.g. a colloquial 'food') is simply not
    matched -- callers treat an empty result as 'no specific scope requested' rather
    than guessing at a mapping."""
    lower = msg.lower()
    found = []
    for kw, cat in CATEGORY_KEYWORDS.items():
        if kw in lower and cat not in found:
            found.append(cat)
    return found


def extract_category(msg: str) -> Optional[str]:
    cats = extract_categories(msg)
    return cats[0] if cats else None


def mentions_generic_food(msg: str) -> bool:
    """True when 'food' is used as a colloquial catch-all counterpart in a comparison
    (e.g. 'food vs drinks') rather than an incidental mention elsewhere in the message
    (e.g. 'food demand chart'). Requires an explicit comparison word alongside 'food'."""
    lower = msg.lower()
    return bool(_GENERIC_FOOD_RE.search(lower) and _COMPARE_WORDS_RE.search(lower))


def mentions_compare(msg: str) -> bool:
    return bool(_COMPARE_INTENT_RE.search(msg.lower()))


def is_category_comparison(msg: str) -> bool:
    """True for a fresh 'compare CategoryA and CategoryB' / 'CategoryA vs food' style
    request -- naming two real categories, or one real category plus the generic
    'food' catch-all, directly in this message. Distinct from a referent-based
    comparison ('compare IT with the top pizza'), which has no named categories of its
    own and instead needs a prior conversation turn to resolve 'it' against."""
    if not mentions_compare(msg):
        return False
    cats = extract_categories(msg)
    return len(cats) >= 2 or (len(cats) == 1 and mentions_generic_food(msg))


def extract_top_n(msg: str, default=5):
    """Parse a requested count from phrases like 'top 15', 'bottom 5', '20 meals'."""
    lower = msg.lower()
    for pattern in (
        r'top\s+(\d+)',
        r'bottom\s+(\d+)',
        r'worst\s+(\d+)',
        r'best\s+(\d+)',
        r'(\d+)\s+(?:meals?|items?|menu\s*items?|products?|dishes?)',
        r'(\d+)\s+(?:worst|best|top|bottom|popular)',
    ):
        m = re.search(pattern, lower)
        if m:
            return min(max(int(m.group(1)), 1), 30)
    return default


def extract_center_ids(msg: str) -> list[int]:
    return [int(x) for x in _CENTER_RE.findall(msg)]


def looks_like_chart_request(msg: str) -> bool:
    """Cheap keyword pre-filter for whether a message might want a chart at all."""
    lower = msg.lower()
    has_trigger = any(kw in lower for kw in _CHART_TRIGGER_KEYWORDS)
    has_visual_req = (
        any(kw in lower for kw in ('show me', 'show', 'generate', 'create', 'make',
                                    'give me', 'display', 'can i see', 'let me see',
                                    'i want to see', 'visualize', 'visualise'))
        and any(kw in lower for kw in ('meal', 'item', 'food', 'category', 'cuisine', 'demand',
                                        'order', 'week', 'trend', 'forecast', 'popular',
                                        'best', 'worst', 'bottom', 'top', 'performance',
                                        'sales', 'data', 'results', 'chart', 'graph',
                                        'location', 'center', 'centre', 'branch', 'outlet', 'store'))
    )
    return has_trigger or has_visual_req


# ─────────────────────────────────────────────────────────────────────────────
# Chart formatting primitives (pure -- no Flask/DB dependency)
# ─────────────────────────────────────────────────────────────────────────────

CHART_COLORS = [
    'rgba(99,179,237,.85)',  'rgba(154,117,237,.85)', 'rgba(72,199,142,.85)',
    'rgba(252,175,69,.85)',  'rgba(252,114,114,.85)', 'rgba(99,207,237,.85)',
    'rgba(237,150,99,.85)',  'rgba(144,237,99,.85)',  'rgba(237,99,161,.85)',
    'rgba(99,137,237,.85)',  'rgba(237,217,99,.85)',  'rgba(99,237,217,.85)',
    'rgba(217,99,237,.85)',  'rgba(237,130,99,.85)',  'rgba(99,237,137,.85)',
    'rgba(237,180,99,.85)',  'rgba(99,160,237,.85)',  'rgba(180,237,99,.85)',
    'rgba(237,99,130,.85)',  'rgba(130,99,237,.85)',  'rgba(99,237,180,.85)',
    'rgba(237,99,200,.85)',  'rgba(99,200,237,.85)',  'rgba(200,237,99,.85)',
    'rgba(237,115,99,.85)',  'rgba(99,237,155,.85)',  'rgba(155,99,237,.85)',
    'rgba(237,230,99,.85)',  'rgba(115,237,99,.85)',  'rgba(237,99,115,.85)',
]


def _bar_chart(title: str, labels: list, values: list,
               horizontal: bool = True, color_offset: int = 0) -> dict:
    colors = [CHART_COLORS[(i + color_offset) % len(CHART_COLORS)] for i in range(len(values))]
    chart: dict = {
        'type': 'bar',
        'title': title,
        'data': {
            'labels': labels,
            'datasets': [{'label': 'Orders', 'data': values,
                          'backgroundColor': colors,
                          'borderRadius': 6, 'borderSkipped': False}],
        },
        'options': {
            'plugins': {'legend': {'display': False}},
            'scales': {'x': {'beginAtZero': True}} if horizontal else {'y': {'beginAtZero': True}},
        },
    }
    if horizontal:
        chart['options']['indexAxis'] = 'y'
    return chart


def two_slice_doughnut(label_a: str, value_a: int, label_b: str, value_b: int, title: str) -> Optional[dict]:
    if value_a == 0 and value_b == 0:
        return None
    return {
        'type': 'doughnut',
        'title': title,
        'data': {
            'labels': [label_a, label_b],
            'datasets': [{'data': [int(value_a), int(value_b)],
                          'backgroundColor': [CHART_COLORS[0], CHART_COLORS[1]],
                          'borderWidth': 0}],
        },
        'options': {'plugins': {'legend': {'position': 'right'}}, 'cutout': '60%'},
    }


_CODE_FENCE_RE = re.compile(r'```.*?```', re.DOTALL)
_TABLE_ROW_RE  = re.compile(r'^[ \t]*\|.*\|[ \t]*$\n?', re.MULTILINE)


def strip_data_dump(text: str) -> str:
    """Strips markdown code fences and pipe-table rows from LLM chart commentary --
    the data is already visible in the chart, and a raw table restating it reads as a
    bug even when the numbers are correct."""
    text = _CODE_FENCE_RE.sub('', text)
    text = _TABLE_ROW_RE.sub('', text)
    return re.sub(r'\n{3,}', '\n\n', text).strip()


def chart_data_summary(chart_data: dict, limit: int = 15) -> str:
    """Flattens a chart's actual rendered labels/values into a short 'Label: value, ...'
    string, so any LLM prompt asking about it is grounded in exactly what's on screen."""
    data = chart_data.get('data', {})
    labels = data.get('labels', [])
    datasets = data.get('datasets', [])
    if not datasets:
        return ''
    parts = []
    for ds in datasets:
        values = ds.get('data', [])
        prefix = f"{ds['label']} — " if ds.get('label') and len(datasets) > 1 else ''
        pairs = [f'{l}: {v:,}' for l, v in zip(labels, values)]
        parts.append(prefix + ', '.join(pairs[:limit]))
    return '; '.join(parts)


# ─────────────────────────────────────────────────────────────────────────────
# QuerySpec -- the normalized, resolved interpretation of a chat turn
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class QuerySpec:
    intent: str                              # 'rank' | 'ordinal' | 'compare' | 'still_top'
    metric: Optional[str] = None             # 'forecast' | 'historical' | None (unresolved)
    categories: list = field(default_factory=list)
    item_ids: list = field(default_factory=list)
    center_ids: list = field(default_factory=list)
    requested_count: int = 5
    sort_direction: str = 'desc'             # 'desc' (top) | 'asc' (bottom)
    rank_by: str = 'item'                    # 'item' | 'location' | 'both' -- what's being ranked
    wants_chart: bool = False
    chart_view: Optional[str] = None         # resolved view, if any -- set by the caller after
                                              # resolving it, so a later follow-up can inherit it
    ordinal_rank: Optional[int] = None
    compare_item_hint: Optional[str] = None
    raw_message: str = ''
    metric_explicit: bool = False
    inherited_metric: bool = False

    def to_state_dict(self) -> dict:
        return asdict(self)


def resolve_query_spec(last_msg: str, prev_spec: Optional[dict]) -> QuerySpec:
    """Resolves a chat message into a QuerySpec, inheriting metric/categories/filters/
    sort-direction/count from the previous turn's spec when this turn doesn't explicitly
    change them and looks like a follow-up (contextual pronoun, ordinal reference, 'what
    about...', 'show that...', etc). This is what lets 'what about the second one?',
    'show that as a chart', and 'what about historically?' resolve correctly instead of
    being treated as brand-new, unscoped queries."""
    lower = last_msg.lower()
    is_followup = bool(_FOLLOWUP_MARKERS_RE.search(lower)) or bool(_ORDINAL_RE.search(lower))

    # ---- metric ----
    fc_hit   = bool(_FORECAST_RE.search(lower))
    hist_hit = bool(_HISTORICAL_RE.search(lower))
    metric_explicit = fc_hit != hist_hit  # exactly one side matched this turn
    if fc_hit and not hist_hit:
        metric = 'forecast'
    elif hist_hit and not fc_hit:
        metric = 'historical'
    elif prev_spec is not None:
        metric = prev_spec.get('metric')
    else:
        metric = None  # unresolved -- caller applies the documented default

    inherited_metric = (not metric_explicit) and (prev_spec is not None) and (metric is not None)

    # ---- categories / location filters ----
    categories = extract_categories(last_msg)
    if not categories and prev_spec is not None and is_followup:
        categories = list(prev_spec.get('categories') or [])

    center_ids = extract_center_ids(last_msg)
    if not center_ids and prev_spec is not None and is_followup:
        center_ids = list(prev_spec.get('center_ids') or [])

    # ---- sort direction ----
    bottom_hit = bool(_BOTTOM_RE.search(lower))
    top_hit    = bool(_TOP_RE.search(lower))
    # "top" alone is ambiguous (count prefix vs. direction word) -- an explicit bottom
    # word must win over a bare "top N" (e.g. "top 5 least popular meals" is asc, not
    # desc), but a genuinely explicit top word ("best", "most popular", ...) alongside a
    # bottom word is a real contradiction and falls through to the existing tie-break.
    top_explicit_hit = bool(_TOP_EXPLICIT_RE.search(lower))
    if bottom_hit and not top_explicit_hit:
        sort_direction = 'asc'
    elif top_hit and not bottom_hit:
        sort_direction = 'desc'
    elif prev_spec is not None and is_followup:
        sort_direction = prev_spec.get('sort_direction', 'desc')
    else:
        sort_direction = 'desc'

    # ---- what's being ranked: menu item(s), location(s), or both ----
    # A plain ranking question ("most popular X") defaults to ranking menu items --
    # but if the user names locations/branches/centres, that's what must be ranked
    # instead (or in addition, if they also say "item"/"meal"), not silently ignored.
    # Without this, "most popular location" used to fall straight through to the
    # generic item ranking with no acknowledgement "location" was ever in the message.
    wants_location_rank = mentions_location(last_msg)
    # Naming an actual menu category ("drinks", "beverages", "pizza"...) implies asking
    # about items just as much as a generic word like "item"/"meal" does -- "most
    # popular drinks at each location" must trigger the same per-location item
    # breakdown as "most popular meals at each location", scoped to that category.
    wants_item_rank = bool(_ITEM_WORD_RE.search(lower)) or bool(categories)
    # Computed here (ahead of the intent block below, which also needs it) purely to
    # let the requested-count default below distinguish "most popular meal at each
    # location" (no explicit number -- must stay a ONE-item-per-location request) from
    # a flat top-N ranking, which otherwise both fall through the same "no number
    # given" path and would wrongly default to 5.
    is_per_group_request = bool(_PER_GROUP_RE.search(lower)) and wants_location_rank and wants_item_rank

    # ---- requested count ----
    n = extract_top_n(last_msg, default=None)
    if n is None:
        if is_per_group_request:
            n = 1
        elif _SINGLE_ANSWER_RE.search(lower):
            n = 1
        else:
            n = prev_spec.get('requested_count', 5) if (prev_spec is not None and is_followup) else 5

    if wants_location_rank and wants_item_rank:
        rank_by = 'both'
    elif wants_location_rank:
        rank_by = 'location'
    elif prev_spec is not None and is_followup and not wants_item_rank:
        rank_by = prev_spec.get('rank_by', 'item')
    else:
        rank_by = 'item'

    # ---- chart? ----
    wants_chart = bool(_CHART_REQUEST_RE.search(lower)) or looks_like_chart_request(last_msg)

    # If this looks like a bare "redo/show that as a chart" follow-up with nothing in
    # this turn's wording indicating a different chart shape, carry over the *resolved*
    # view from the previous turn (recorded there after it was picked) -- this is what
    # lets a follow-up be answered purely from structured state, without needing to ask
    # the LLM to re-guess a shape it has no strong signal for.
    inherited_chart_view = None
    if wants_chart and is_followup and prev_spec is not None and detect_chart_view(last_msg) is None:
        inherited_chart_view = prev_spec.get('chart_view')

    # ---- ordinal follow-up ("what about the second one") ----
    ordinal_rank = None
    om = _ORDINAL_RE.search(lower)
    if om and is_followup and prev_spec is not None:
        ordinal_rank = _ORDINAL_WORDS.get(om.group(1).lower())

    # ---- intent ----
    intent = 'rank'
    compare_item_hint = None
    if is_per_group_request:
        # "most popular meal AT EACH of those locations" is a per-location breakdown
        # (rank items *within* every named/referenced location), not two independent
        # flat top-N lists -- that's what rank_by='both' answers instead, and answers
        # a different question than the one actually asked.
        intent = 'per_location'
    elif _COMPARE_INTENT_RE.search(lower):
        intent = 'compare'
        compare_item_hint = last_msg
    elif ordinal_rank is not None:
        intent = 'ordinal'
    elif _STILL_RE.search(lower) and is_followup:
        intent = 'still_top'

    return QuerySpec(
        intent=intent,
        metric=metric,
        categories=categories,
        item_ids=[],
        center_ids=center_ids,
        requested_count=n,
        sort_direction=sort_direction,
        rank_by=rank_by,
        wants_chart=wants_chart,
        chart_view=inherited_chart_view,
        ordinal_rank=ordinal_rank,
        compare_item_hint=compare_item_hint,
        raw_message=last_msg,
        metric_explicit=metric_explicit,
        inherited_metric=inherited_metric,
    )


_RANKING_EXTRA_RE = re.compile(
    r'\b(rank(?:ing)?|popular|'
    r'which\s+(?:item|meal|product|dish|location|branch|outlet|store|centre|center)s?|'
    r'should\s+(?:i|we)\s+(?:stock|order|prepare|make)|stock\s+(?:the\s+)?most)\b',
    re.I,
)


def mentions_ranking_language(msg: str) -> bool:
    """True if `msg` contains ranked-item wording. Built from the same _TOP_RE/
    _BOTTOM_RE patterns used for sort-direction detection (plus a few extra
    ranking-adjacent phrases that don't imply a direction, like "rank"/"popular"/
    "should I stock") so there's one source of truth for what counts as a ranking
    request at all, instead of a second, narrower literal-phrase list that could
    (and did) drift out of sync -- it previously missed common verb conjugations
    ("sells"/"sold" vs. a listed "sell") and word orders ("sells best" vs. a listed
    "best sell"), silently sending real questions like "what food sells the most?"
    to the ungrounded general chat reply instead of the grounded pipeline."""
    lower = msg.lower()
    return bool(_TOP_RE.search(lower) or _BOTTOM_RE.search(lower) or _RANKING_EXTRA_RE.search(lower))


def wants_grounded_ranking(last_msg: str, spec: QuerySpec, has_prior_grounded_state: bool = False) -> bool:
    """True when this message needs the authoritative GroundedResult pipeline (a
    ranking, comparison, ordinal follow-up, or chart of order/demand data) rather than
    the general free-form chat reply. Kept as a single gate so ranking/compare/ordinal/
    chart requests never accidentally fall through to two different answer paths.

    `has_prior_grounded_state` covers a bare metric-switch follow-up like "what about
    historically?" -- it names no ranking word at all, but is clearly continuing a
    grounded conversation (there's a previous QuerySpec/result to inherit scope from)
    rather than starting a new, unrelated chit-chat topic."""
    if spec.wants_chart:
        return True
    if spec.intent in ('ordinal', 'compare', 'still_top', 'per_location'):
        return True
    if mentions_ranking_language(last_msg):
        return True
    lower = last_msg.lower()
    if has_prior_grounded_state and (
            _FOLLOWUP_MARKERS_RE.search(lower) or _HISTORICAL_RE.search(lower) or _FORECAST_RE.search(lower)):
        return True
    return False


def detect_chart_view(msg: str) -> Optional[str]:
    """Deterministic chart-view classification. Returns None (meaning: use the default
    ranked-items view, or ask the LLM to disambiguate) when nothing matches. Direction
    (top vs bottom) is never decided here -- it's resolved once, deterministically, in
    QuerySpec.sort_direction, so it can't be flipped by phrasing like 'top 5 historically'
    the way the pre-refactor LLM tool call was observed to do."""
    lower = msg.lower()

    if mentions_location(msg):
        return 'location_breakdown'

    # Trend-over-weeks indicators only -- deliberately does NOT include a bare "week"
    # (e.g. "next week", "this week") since that's just a forecast time-scope mention,
    # not a request to see change across multiple weeks. A message like "category
    # breakdown for next week" must still resolve to a breakdown, not a trend line.
    if 'categor' in lower and any(kw in lower for kw in (
            'trend', 'over time', 'across weeks', 'per week', 'each week', 'week by week',
            'week-over-week', 'progression', 'time series', 'weeks')):
        return 'category_trend'

    if any(kw in lower for kw in ('category', 'breakdown', 'pie', 'doughnut',
                                   'distribution', 'proportion', 'segment', 'share')):
        return 'category_breakdown'

    if any(kw in lower for kw in ('cuisine', 'food type', 'type of food', 'food style')):
        return 'cuisine_breakdown'

    if _COMPARE_WORDS_RE.search(lower) and (extract_categories(msg) or mentions_generic_food(msg)):
        return 'category_breakdown'

    if any(kw in lower for kw in ('all week', 'each week', 'weekly forecast', 'week by week',
                                   'per week', 'forecast week', 'trend', 'weekly',
                                   'over time', 'across weeks', 'progression', 'time series')):
        return 'weekly_totals'

    return None


AGGREGATE_CHART_VIEWS = ('category_breakdown', 'cuisine_breakdown', 'category_trend', 'weekly_totals',
                          'location_breakdown')
CHART_VIEWS = ('ranked_items',) + AGGREGATE_CHART_VIEWS


# ─────────────────────────────────────────────────────────────────────────────
# GroundedResult -- the single authoritative answer object shared by chart + prose
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class GroundedResult:
    session_id: object
    metric: str
    period_label: str
    filters: dict
    rows: list                              # list[dict]: item_id, display_name, category, cuisine, value, rank
    units: str = 'orders'
    source: str = 'session_forecasts'       # or 'session_history'
    ties: list = field(default_factory=list)
    chart: Optional[dict] = None
    query: Optional[dict] = None


def _compute_ties(rows: list) -> list:
    ties = []
    i = 0
    while i < len(rows):
        j = i
        while j + 1 < len(rows) and rows[j + 1]['value'] == rows[i]['value']:
            j += 1
        if j > i:
            ties.append({
                'value': rows[i]['value'],
                'ranks': [rows[k]['rank'] for k in range(i, j + 1)],
                'item_ids': [rows[k]['item_id'] for k in range(i, j + 1)],
                'display_names': [rows[k]['display_name'] for k in range(i, j + 1)],
            })
        i = j + 1
    return ties


def format_tie_note(ties: list, units: str = 'orders') -> Optional[str]:
    if not ties:
        return None
    parts = []
    for t in ties:
        names = t['display_names']
        if len(names) < 2:
            continue
        rank_word = f"rank {t['ranks'][0]}" if len(t['ranks']) == 1 else f"ranks {t['ranks'][0]}–{t['ranks'][-1]}"
        parts.append(f"{', '.join(names[:-1])} and {names[-1]} are tied at {rank_word} "
                     f"with {t['value']:,} {units} each")
    if not parts:
        return None
    return 'Note: ' + '; '.join(parts) + '.'


def build_chart_from_rows(rows: list, metric: str, spec: QuerySpec) -> Optional[dict]:
    """Builds a chart directly from GroundedResult rows -- the exact same ordered
    labels/values the narrative is built from, so chart and prose can never diverge."""
    if not rows:
        return None
    labels = [r['display_name'] for r in rows]
    values = [r['value'] for r in rows]
    scope = ' & '.join(spec.categories) if spec.categories else 'Menu Items'
    direction = 'Bottom' if spec.sort_direction == 'asc' else 'Top'
    period = METRIC_PERIOD_LABEL[metric]
    title = f'{direction} {len(labels)} {scope} — {period}'
    return _bar_chart(title, labels, values, color_offset=(10 if spec.sort_direction == 'asc' else 0))


def compute_grounded_result(spec: QuerySpec, session_id,
                             forecast_df: Optional[pd.DataFrame],
                             history_df: Optional[pd.DataFrame]) -> Optional[GroundedResult]:
    """Computes the authoritative ranking for `spec` straight from the full stored
    per-week dataset (not a trimmed top-20/30 pool). Aggregation, filtering, ranking,
    and tie handling all happen here in Python -- nothing here is decided by an LLM."""
    metric = spec.metric or DEFAULT_METRIC
    df = forecast_df if metric == 'forecast' else history_df
    value_col = 'ai_forecast' if metric == 'forecast' else 'num_orders'

    if df is None or df.empty or value_col not in df.columns or 'meal_id' not in df.columns:
        return None

    df = df.copy()
    if metric == 'forecast' and 'week' in df.columns and not df.empty:
        df = df[df['week'] == df['week'].min()]

    if spec.categories and 'category' in df.columns:
        wanted = {c.lower() for c in spec.categories}
        df = df[df['category'].astype(str).str.lower().isin(wanted)]
    if spec.center_ids and 'center_id' in df.columns:
        df = df[df['center_id'].isin(spec.center_ids)]
    if spec.item_ids:
        df = df[df['meal_id'].isin(spec.item_ids)]

    if df.empty:
        return None

    agg_spec = {value_col: 'sum'}
    for col in ('category', 'cuisine', 'item_name'):
        if col in df.columns:
            agg_spec[col] = 'first'
    grouped = df.groupby('meal_id', dropna=False).agg(agg_spec).reset_index()
    grouped = grouped.rename(columns={value_col: 'value'})
    grouped['value'] = grouped['value'].fillna(0).astype(int)

    ascending = (spec.sort_direction == 'asc')
    # sort by value, then meal_id ascending as a deterministic, stable tie-break --
    # guarantees identical output across repeated identical requests, independent of
    # whatever incidental row order groupby happens to produce.
    grouped = grouped.sort_values(['value', 'meal_id'], ascending=[ascending, True], kind='mergesort')

    n = max(int(spec.requested_count or 5), 1)
    take = grouped.head(max(n, 10))

    records = take.to_dict('records')
    rows_all = []
    for rank, rec in enumerate(records, start=1):
        meal_id = int(rec['meal_id'])
        item_name = rec.get('item_name')
        display_name = item_name if (item_name is not None and pd.notna(item_name)) else f'Meal #{meal_id}'
        category = rec.get('category')
        category = category if (category is not None and pd.notna(category)) else None
        cuisine = rec.get('cuisine')
        cuisine = cuisine if (cuisine is not None and pd.notna(cuisine)) else None
        rows_all.append({
            'item_id': meal_id,
            'display_name': str(display_name),
            'category': category,
            'cuisine': cuisine,
            'value': int(rec['value']),
            'rank': rank,
        })

    # Tie-aware truncation: if the row at the requested cutoff shares its value with the
    # next row(s), keep them too instead of arbitrarily slicing a tie in half.
    if len(rows_all) > n:
        boundary_value = rows_all[n - 1]['value']
        cutoff = n
        while cutoff < len(rows_all) and rows_all[cutoff]['value'] == boundary_value:
            cutoff += 1
        rows_final = rows_all[:cutoff]
    else:
        rows_final = rows_all

    ties = _compute_ties(rows_final)
    filters = {'categories': list(spec.categories), 'center_ids': list(spec.center_ids)}
    chart = build_chart_from_rows(rows_final, metric, spec) if spec.wants_chart else None

    return GroundedResult(
        session_id=session_id,
        metric=metric,
        period_label=METRIC_PERIOD_LABEL[metric],
        filters=filters,
        rows=rows_final,
        units='orders',
        source=('session_forecasts' if metric == 'forecast' else 'session_history'),
        ties=ties,
        chart=chart,
        query=spec.to_state_dict(),
    )


def compute_location_ranking(spec: QuerySpec, session_id,
                              forecast_df: Optional[pd.DataFrame],
                              history_df: Optional[pd.DataFrame],
                              center_type_map: Optional[dict] = None) -> Optional[GroundedResult]:
    """A location-scoped counterpart to compute_grounded_result -- ranks fulfilment
    centres/branches by total demand instead of menu items. Reuses the same
    GroundedResult shape (category/cuisine simply stay None for a location row) so
    text/tie/chart rendering all work unchanged regardless of what's being ranked."""
    metric = spec.metric or DEFAULT_METRIC
    df = forecast_df if metric == 'forecast' else history_df
    value_col = 'ai_forecast' if metric == 'forecast' else 'num_orders'

    if df is None or df.empty or value_col not in df.columns or 'center_id' not in df.columns:
        return None

    df = df.copy()
    if metric == 'forecast' and 'week' in df.columns and not df.empty:
        df = df[df['week'] == df['week'].min()]
    if spec.categories and 'category' in df.columns:
        wanted = {c.lower() for c in spec.categories}
        df = df[df['category'].astype(str).str.lower().isin(wanted)]
    if df.empty:
        return None

    ascending = (spec.sort_direction == 'asc')
    totals = df.groupby('center_id')[value_col].sum().sort_values(ascending=ascending)
    center_type_map = center_type_map or {}

    rows_all = []
    for rank, (cid, val) in enumerate(totals.items(), start=1):
        cid_int = int(cid)
        ctype = center_type_map.get(cid_int)
        rows_all.append({
            'item_id': cid_int,
            'display_name': f'Center #{cid_int}' + (f' ({ctype})' if ctype else ''),
            'category': None,
            'cuisine': None,
            'value': int(val),
            'rank': rank,
        })

    n = max(int(spec.requested_count or 5), 1)
    take = rows_all[:max(n, 10)]
    if len(take) > n:
        boundary_value = take[n - 1]['value']
        cutoff = n
        while cutoff < len(take) and take[cutoff]['value'] == boundary_value:
            cutoff += 1
        rows_final = take[:cutoff]
    else:
        rows_final = take

    ties = _compute_ties(rows_final)
    filters = {'categories': list(spec.categories), 'center_ids': list(spec.center_ids)}
    return GroundedResult(
        session_id=session_id,
        metric=metric,
        period_label=METRIC_PERIOD_LABEL[metric],
        filters=filters,
        rows=rows_final,
        units='orders',
        source=('session_forecasts' if metric == 'forecast' else 'session_history'),
        ties=ties,
        chart=None,
        query=spec.to_state_dict(),
    )


def compute_category_totals(spec: QuerySpec, session_id,
                             forecast_df: Optional[pd.DataFrame],
                             history_df: Optional[pd.DataFrame]) -> Optional[dict]:
    """Total demand per category -- for a category-vs-category or category-vs-'food'
    comparison ('compare pizza and pasta', 'compare beverages and food'). Not a
    GroundedResult (categories aren't individually-ranked items with ids), just a
    plain {category: total} dict. Uses the same grouping as the category_breakdown
    chart, so the chart and the plain-text comparison of the same request agree."""
    metric = spec.metric or DEFAULT_METRIC
    df = forecast_df if metric == 'forecast' else history_df
    value_col = 'ai_forecast' if metric == 'forecast' else 'num_orders'
    if df is None or df.empty or value_col not in df.columns or 'category' not in df.columns:
        return None
    df = df.copy()
    if metric == 'forecast' and 'week' in df.columns and not df.empty:
        df = df[df['week'] == df['week'].min()]
    # A "compare X and Y at center N" question must only total demand actually placed at
    # that center -- without this filter, a category present at a completely different
    # center (never scoped to by this question) silently leaked into the comparison.
    if spec.center_ids and 'center_id' in df.columns:
        df = df[df['center_id'].isin(spec.center_ids)]
    if df.empty:
        return None
    totals = df.groupby('category')[value_col].sum()
    if totals.empty:
        return None
    return {str(k): int(v) for k, v in totals.items()}


def compute_top_item_per_group(spec: QuerySpec, session_id,
                                forecast_df: Optional[pd.DataFrame],
                                history_df: Optional[pd.DataFrame],
                                center_ids: list,
                                center_type_map: Optional[dict] = None,
                                top_n_per_group: int = 1) -> list:
    """Per-location breakdown: the top item(s) WITHIN each of `center_ids`, not a flat
    overall ranking. Answers "most popular meal at each of those locations" -- a
    different question from either a plain item ranking or a plain location ranking,
    and from rank_by='both' (which just answers those two separately). Returns a list
    of {center_id, center_label, top_items: [row, ...]} in the same order as
    `center_ids`, so a prior location ranking's order is preserved in the reply."""
    metric = spec.metric or DEFAULT_METRIC
    df = forecast_df if metric == 'forecast' else history_df
    value_col = 'ai_forecast' if metric == 'forecast' else 'num_orders'
    center_type_map = center_type_map or {}

    if df is None or df.empty or value_col not in df.columns or 'center_id' not in df.columns:
        return [{'center_id': cid, 'center_label': f'Center #{cid}', 'top_items': []} for cid in center_ids]

    df = df.copy()
    if metric == 'forecast' and 'week' in df.columns and not df.empty:
        df = df[df['week'] == df['week'].min()]
    if spec.categories and 'category' in df.columns:
        wanted = {c.lower() for c in spec.categories}
        df = df[df['category'].astype(str).str.lower().isin(wanted)]

    ascending = (spec.sort_direction == 'asc')
    results = []
    for cid in center_ids:
        label = f'Center #{cid}' + (f' ({center_type_map[cid]})' if cid in center_type_map else '')
        sub = df[df['center_id'] == cid]
        if sub.empty:
            results.append({'center_id': cid, 'center_label': label, 'top_items': []})
            continue
        agg_spec = {value_col: 'sum'}
        for col in ('category', 'cuisine', 'item_name'):
            if col in sub.columns:
                agg_spec[col] = 'first'
        grouped = sub.groupby('meal_id', dropna=False).agg(agg_spec).reset_index()
        grouped = grouped.rename(columns={value_col: 'value'})
        grouped = grouped.sort_values(['value', 'meal_id'], ascending=[ascending, True], kind='mergesort')

        n = max(int(top_n_per_group or 1), 1)
        rows_all = []
        for rank, rec in enumerate(grouped.head(max(n, 10)).to_dict('records'), start=1):
            meal_id = int(rec['meal_id'])
            item_name = rec.get('item_name')
            display_name = item_name if (item_name is not None and pd.notna(item_name)) else f'Meal #{meal_id}'
            rows_all.append({
                'item_id': meal_id, 'display_name': str(display_name),
                'value': int(rec['value']), 'rank': rank,
            })

        # Tie-aware truncation at the requested boundary (same rule as the flat
        # item/location rankings) -- a tie for the last requested spot is kept in full
        # rather than arbitrarily cut in half.
        if len(rows_all) > n:
            boundary_value = rows_all[n - 1]['value']
            cutoff = n
            while cutoff < len(rows_all) and rows_all[cutoff]['value'] == boundary_value:
                cutoff += 1
            top_items = rows_all[:cutoff]
        else:
            top_items = rows_all

        results.append({'center_id': cid, 'center_label': label, 'top_items': top_items})
    return results


def render_location_breakdown_narrative(breakdown: list, metric: str, sort_direction: str = 'desc') -> str:
    if not breakdown:
        return NO_MATCHING_DATA_MSG
    metric_phrase = METRIC_PHRASES[metric]
    direction = 'least popular' if sort_direction == 'asc' else 'most popular'
    lines = [f'{metric_phrase}, the {direction} item at each of those locations:', '']
    for entry in breakdown:
        items = entry['top_items']
        if not items:
            lines.append(f"- **{entry['center_label']}**: no data available")
        elif len(items) == 1:
            top = items[0]
            lines.append(f"- **{entry['center_label']}**: {top['display_name']} — {top['value']:,} orders")
        else:
            lines.append(f"- **{entry['center_label']}**:")
            for item in items:
                lines.append(f"  {item['rank']}. {item['display_name']} — {item['value']:,} orders")
    return '\n'.join(lines)


def build_aggregate_chart(view: str, metric: str, df: Optional[pd.DataFrame], spec: QuerySpec,
                           center_type_map: Optional[dict] = None, as_line: bool = False) -> Optional[dict]:
    """Builds category/cuisine/trend/weekly/location charts -- views that aggregate
    across many items rather than ranking individual ones, so they don't fit
    GroundedResult's per-item row shape. Ported from the pre-refactor
    `_build_chart_from_query`, unchanged in behavior."""
    value_col = 'ai_forecast' if metric == 'forecast' else 'num_orders'
    period_label = METRIC_PERIOD_LABEL[metric]
    if df is None or df.empty or value_col not in df.columns:
        return None
    df = df.copy()

    if view == 'category_breakdown':
        if 'category' not in df.columns:
            return None
        generic_food = mentions_generic_food(spec.raw_message)
        if len(spec.categories) == 1 and generic_food:
            named = spec.categories[0]
            is_named = df['category'].astype(str).str.lower() == named.lower()
            named_total = int(df.loc[is_named, value_col].sum())
            other_total = int(df.loc[~is_named, value_col].sum())
            chart = two_slice_doughnut(named, named_total, 'Food (all other categories)', other_total,
                                        f'{named} vs. Food — {period_label}')
            if chart:
                return chart
        if len(spec.categories) >= 2:
            wanted = {c.lower() for c in spec.categories}
            df = df[df['category'].astype(str).str.lower().isin(wanted)]
        totals = df.groupby('category')[value_col].sum().sort_values(ascending=False)
        if totals.empty:
            return None
        cats = totals.index.tolist()
        return {
            'type': 'doughnut',
            'title': f'Demand by Category — {period_label}',
            'data': {
                'labels': cats,
                'datasets': [{'data': [int(v) for v in totals.values],
                              'backgroundColor': [CHART_COLORS[i % len(CHART_COLORS)] for i in range(len(cats))],
                              'borderWidth': 0}],
            },
            'options': {'plugins': {'legend': {'position': 'right'}}, 'cutout': '60%'},
        }

    if view == 'cuisine_breakdown':
        if 'cuisine' not in df.columns:
            return None
        totals = df.groupby('cuisine')[value_col].sum().sort_values(ascending=False)
        if totals.empty:
            return None
        return _bar_chart('Orders by Cuisine Type', totals.index.tolist(),
                          [int(v) for v in totals.values], horizontal=False)

    if view == 'weekly_totals':
        if 'week' not in df.columns:
            return None
        totals = df.groupby('week')[value_col].sum().sort_index()
        if totals.empty:
            return None
        labels = [f'Week {int(w)}' for w in totals.index]
        values = [int(v) for v in totals.values]
        if as_line:
            return {
                'type': 'line',
                'title': f'Weekly {"Forecast" if metric == "forecast" else "Order"} Trend',
                'data': {'labels': labels, 'datasets': [{
                    'label': 'Total Orders', 'data': values,
                    'borderColor': 'rgba(99,179,237,1)', 'backgroundColor': 'rgba(99,179,237,.15)',
                    'fill': True, 'tension': 0.4, 'pointRadius': 4,
                }]},
                'options': {'plugins': {'legend': {'display': False}}, 'scales': {'y': {'beginAtZero': False}}},
            }
        chart_label = 'Weekly Forecast Comparison' if metric == 'forecast' else 'Weekly Order Comparison'
        return _bar_chart(chart_label, labels, values, horizontal=False)

    if view == 'category_trend':
        if 'category' not in df.columns or 'week' not in df.columns:
            return None
        grouped = df.groupby(['category', 'week'])[value_col].sum().reset_index()
        if grouped.empty:
            return None
        weeks = sorted(grouped['week'].unique())
        trend_cats = sorted(grouped['category'].dropna().unique())
        datasets = []
        for idx, cat in enumerate(trend_cats):
            color = CHART_COLORS[idx % len(CHART_COLORS)]
            sub = grouped[grouped['category'] == cat].set_index('week')[value_col]
            datasets.append({
                'label': cat, 'data': [int(sub.get(w, 0)) for w in weeks],
                'borderColor': color, 'backgroundColor': color,
                'fill': False, 'tension': 0.4, 'pointRadius': 3,
            })
        trend_label = 'Forecast Trend by Category' if metric == 'forecast' else 'Historical Trend by Category'
        return {
            'type': 'line', 'title': trend_label,
            'data': {'labels': [f'Week {int(w)}' for w in weeks], 'datasets': datasets},
            'options': {'plugins': {'legend': {'position': 'bottom'}}, 'scales': {'y': {'beginAtZero': True}}},
        }

    if view == 'location_breakdown':
        if 'center_id' not in df.columns:
            return None
        totals = df.groupby('center_id')[value_col].sum().sort_values(ascending=False).head(spec.requested_count or 10)
        if totals.empty:
            return None
        center_type_map = center_type_map or {}
        labels = [f'Center #{int(cid)} ({center_type_map.get(int(cid), "Unknown")})' for cid in totals.index]
        values = [int(v) for v in totals.values]
        return _bar_chart(f'Top {len(labels)} Locations — {period_label}', labels, values)

    return None


# ─────────────────────────────────────────────────────────────────────────────
# Deterministic narrative templates
#
# Per the design requirement "prefer deterministic Python templates for rankings and
# chart insights": plain-text ranking/ordinal/compare/still-top answers NEVER go
# through the LLM at all -- there is no wording-vs-fact validation step needed because
# there is no LLM-authored wording to validate. Only the free-form commentary on
# aggregate/breakdown charts still uses the LLM, and even then only after validation
# (see validate_chart_narrative below).
# ─────────────────────────────────────────────────────────────────────────────

def render_ranking_narrative(gr: GroundedResult, spec: QuerySpec, entity_label: Optional[str] = None) -> str:
    if not gr.rows:
        return NO_MATCHING_DATA_MSG
    scope = entity_label or (' & '.join(spec.categories) if spec.categories else 'menu items')
    metric_phrase = METRIC_PHRASES[gr.metric]
    direction = 'lowest-demand' if spec.sort_direction == 'asc' else 'top'

    # A single-result answer -- either an explicit "top 1" or an implicit single-answer
    # question like "which location should I focus on?" (see _SINGLE_ANSWER_RE) -- reads
    # far more naturally as a direct recommendation than a one-line numbered list, and
    # it's the actual phrasing this class of question is asking for.
    if len(gr.rows) == 1 and not gr.ties:
        row  = gr.rows[0]
        meta = ' — '.join(p for p in (row['category'] or '', row['cuisine'] or '') if p)
        name = f"**{row['display_name']}**{f' ({meta})' if meta else ''}"
        pick = 'lowest-demand pick' if spec.sort_direction == 'asc' else 'top pick'
        return f"{metric_phrase}, your {pick} for {scope} is {name} at {row['value']:,} {gr.units}."

    lines = [f'{metric_phrase}, here are the {direction} {scope}:', '']
    for row in gr.rows:
        meta = ' — '.join(p for p in (row['category'] or '', row['cuisine'] or '') if p)
        lines.append(f"{row['rank']}. **{row['display_name']}**{f' ({meta})' if meta else ''}: "
                     f"{row['value']:,} {gr.units}")
    note = format_tie_note(gr.ties, gr.units)
    if note:
        lines += ['', note]
    return '\n'.join(lines)


def render_ordinal_narrative(gr: GroundedResult, ordinal_rank: int) -> str:
    rows = gr.rows
    if not rows:
        return NO_MATCHING_DATA_MSG
    idx = (len(rows) - 1) if ordinal_rank == -1 else (ordinal_rank - 1)
    if idx < 0 or idx >= len(rows):
        return (f"I only have {len(rows)} ranked item(s) available for that scope -- "
                f"there isn't a #{ordinal_rank} in this list.")
    row = rows[idx]
    metric_phrase = METRIC_PHRASES[gr.metric]
    meta = ' — '.join(p for p in (row['category'] or '', row['cuisine'] or '') if p)
    return (f"{metric_phrase}, rank {row['rank']} is **{row['display_name']}**"
            f"{f' ({meta})' if meta else ''} with {row['value']:,} {gr.units}.")


def render_compare_narrative(item_a: dict, item_b: dict, metric: str) -> str:
    metric_phrase = METRIC_PHRASES[metric]
    a_name, a_val = item_a['display_name'], item_a['value']
    b_name, b_val = item_b['display_name'], item_b['value']
    if a_val == b_val:
        return (f"{metric_phrase}, **{a_name}** and **{b_name}** are tied at "
                f"{a_val:,} orders each.")
    higher, lower = (item_a, item_b) if a_val > b_val else (item_b, item_a)
    diff = higher['value'] - lower['value']
    pct_str = f' ({(diff / lower["value"] * 100):.0f}% more)' if lower['value'] else ''
    return (f"{metric_phrase}, **{higher['display_name']}** ({higher['value']:,}) is ahead of "
            f"**{lower['display_name']}** ({lower['value']:,}) by {diff:,} orders{pct_str}.")


def render_still_top_narrative(item: dict, gr: GroundedResult) -> str:
    metric_phrase = METRIC_PHRASES[gr.metric]
    scope = (' & '.join(gr.filters.get('categories') or [])) or 'menu items'
    if not gr.rows:
        return NO_MATCHING_DATA_MSG
    top = gr.rows[0]
    if top['item_id'] == item['item_id']:
        return (f"Yes — {metric_phrase.lower()}, **{item['display_name']}** is still the top "
                f"{scope} item, with {top['value']:,} {gr.units}.")
    match = next((r for r in gr.rows if r['item_id'] == item['item_id']), None)
    if match:
        rank_str = f"it's currently ranked #{match['rank']} with {match['value']:,} {gr.units}"
    else:
        rank_str = "it doesn't appear in this ranking"
    return (f"No — {metric_phrase.lower()}, **{top['display_name']}** is the top {scope} item "
            f"with {top['value']:,} {gr.units}; **{item['display_name']}** {rank_str}.")


# ─────────────────────────────────────────────────────────────────────────────
# LLM narrative validation
#
# Directly targets the reported failure mode: a chart shows Iced Lemon Tea as the
# tallest bar, but the LLM-written caption claims Thai Milk Tea is the top seller. This
# checks the LLM's prose against the exact rendered chart data and rejects it if it
# contradicts the true maximum, misstates the top value, or claims a different label
# is "the top" than the one that actually has the highest value.
# ─────────────────────────────────────────────────────────────────────────────

_SUPERLATIVE_RE = re.compile(
    r'\b(top|highest|most popular|number one|no\.?\s*1|#1|best[- ]selling|first place|'
    r'most in demand|leads?|leading)\b',
    re.I,
)


def _superlative_contradiction(text: str, true_leader_names: set, other_names: list) -> Optional[str]:
    """Clause-scoped (not fixed-window) superlative check: split on sentence/clause
    boundaries so "X leads at 642, with Y close behind" isn't flagged just because both
    names land within N characters of a superlative word -- only a clause that contains
    a superlative word, names a non-leading entity, and does NOT also name a true
    leader counts as a contradiction. Returns an error message, or None if clean."""
    clauses = re.split(r'[.,;:!?]+|\bwith\b|\bwhile\b|\bwhereas\b|\balthough\b|\bbut\b', text, flags=re.I)
    for clause in clauses:
        if not _SUPERLATIVE_RE.search(clause):
            continue
        clause_lower = clause.lower()
        if any(str(l).lower() in clause_lower for l in true_leader_names):
            continue  # this clause is about a true leader -- not a contradiction
        for name in other_names:
            if str(name).lower() in clause_lower:
                return f'narrative claims "{name}" leads, but the actual leader is {sorted(true_leader_names)}'
    return None


def _validate_single_series_narrative(text: str, labels: list, values: list) -> tuple[bool, str]:
    pairs = list(zip(labels, values))
    if not pairs:
        return True, 'no chart data to validate against'

    max_val = max(v for _, v in pairs)
    tie_labels = [l for l, v in pairs if v == max_val]
    lower_text = text.lower()

    if not any(str(l).lower() in lower_text for l in tie_labels):
        return False, f'narrative never mentions the actual top item(s): {tie_labels}'

    other_labels = [l for l, v in pairs if l not in tie_labels]
    contradiction = _superlative_contradiction(text, set(tie_labels), other_labels)
    if contradiction:
        return False, contradiction

    val_variants = {f'{int(max_val):,}', str(int(max_val))}
    if not any(v in text for v in val_variants):
        return False, f'narrative omits/changes the top chart value {max_val}'

    return True, 'ok'


def _validate_multi_series_narrative(text: str, labels: list, datasets: list) -> tuple[bool, str]:
    """Validates commentary on a multi-series chart (e.g. a category trend line with one
    series per category) against every rendered series -- not an automatic pass. Unlike
    the single-series checks above, ambiguous/unparsable shapes fail CLOSED (reject, so
    the caller falls back to the deterministic template) rather than open, since
    approving unchecked LLM text is exactly the risk this function exists to prevent."""
    series = []
    for ds in datasets:
        name = ds.get('label')
        vals = ds.get('data', [])
        if not name or len(vals) != len(labels):
            return False, 'multi-series chart shape not reliably validatable, using deterministic fallback'
        try:
            series.append((str(name), [float(v) for v in vals]))
        except (TypeError, ValueError):
            return False, 'multi-series chart contains non-numeric data, using deterministic fallback'

    if len(series) < 2 or not labels:
        return False, 'multi-series chart has too few valid series, using deterministic fallback'

    # "the overall maximum" -- the single highest rendered data point, and which
    # series it belongs to.
    overall_max = max(v for _, vals in series for v in vals)
    overall_max_series = {name for name, vals in series if overall_max in vals}

    # "leading series/category" -- the series with the highest total across the chart
    # (e.g. the category with the most cumulative demand across all plotted weeks).
    totals = {name: sum(vals) for name, vals in series}
    max_total = max(totals.values())
    leading_series = {name for name, t in totals.items() if t == max_total}

    true_leaders = leading_series | overall_max_series
    lower_text = text.lower()
    if not any(name.lower() in lower_text for name in true_leaders):
        return False, f'narrative never mentions the actual leading series: {sorted(true_leaders)}'

    other_series = [name for name, _ in series if name not in true_leaders]
    contradiction = _superlative_contradiction(text, true_leaders, other_series)
    if contradiction:
        return False, contradiction

    val_variants = set()
    for v in (overall_max, max_total):
        val_variants.add(f'{int(v):,}')
        val_variants.add(str(int(v)))
    if not any(v in text for v in val_variants):
        return False, f'narrative omits/changes the associated value ({int(overall_max)} / {int(max_total)})'

    return True, 'ok'


def validate_chart_narrative(text: str, chart: dict) -> tuple[bool, str]:
    """Validates LLM-written chart commentary against the exact chart data that was
    rendered. Returns (ok, reason). If validation fails, the caller must discard the
    LLM text and use the deterministic fallback instead."""
    data = chart.get('data', {})
    labels = data.get('labels', [])
    datasets = data.get('datasets', [])
    if not datasets:
        return True, 'no chart data to validate against'

    # Multi-series charts must always go through full validation. The "first dataset"
    # shape check below is a safe early-exit ONLY for the single-series case -- applying
    # it here let a malformed/unreliable multi-series chart (bad labels, a short/missing
    # later dataset, etc.) bypass _validate_multi_series_narrative's fail-closed checks
    # entirely and get approved by default, which is exactly the risk that function
    # exists to prevent.
    if len(datasets) > 1:
        return _validate_multi_series_narrative(text, labels, datasets)

    if not labels:
        return True, 'no chart data to validate against'
    values = datasets[0].get('data', [])
    if not values or len(values) != len(labels):
        return True, 'chart shape unexpected, skipping validation'
    return _validate_single_series_narrative(text, labels, values)


def render_chart_insight_fallback(chart: dict) -> str:
    """Deterministic chart-insight template -- the PRIMARY path for ranked-item charts
    (top_items/bottom_items/location breakdowns), and the safety-net fallback whenever
    an LLM-authored caption fails validate_chart_narrative or Ollama is unreachable."""
    data = chart.get('data', {})
    labels = data.get('labels', [])
    datasets = data.get('datasets', [])
    if not labels or not datasets:
        return "Here's the chart you asked for."
    values = datasets[0].get('data', [])
    if not values or len(values) != len(labels):
        return "Here's the chart you asked for."
    pairs = sorted(zip(labels, values), key=lambda p: p[1], reverse=True)
    top_label, top_value = pairs[0]
    tie = [l for l, v in pairs if v == top_value]
    if len(tie) > 1:
        lead = f"{', '.join(tie)} are tied for the top spot, each at {top_value:,}."
    else:
        lead = f"**{top_label}** leads at {top_value:,}."
    action = 'Use this to prioritize what you stock and prep first.'
    if len(pairs) > 1:
        second_label, second_value = pairs[1]
        return f'{lead} Next is {second_label} at {second_value:,}. {action}'
    return f'{lead} {action}'


# ─────────────────────────────────────────────────────────────────────────────
# Conversation state -- explicit, structured, server-side, per (user, session)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ConversationState:
    user_id: object
    session_id: object
    last_spec: Optional[dict] = None
    last_result_rows: Optional[list] = None
    last_result_metric: Optional[str] = None
    last_result_categories: list = field(default_factory=list)
    last_chart: Optional[dict] = None
    last_referenced_item: Optional[dict] = None
    # Tracked separately from last_result_rows (which can hold either item or location
    # rows depending on the turn) so a later "at each of those locations" can never
    # mistake item rows (meal ids) for location rows (center ids) -- see the bug where
    # a 'rank_by=both' turn's item rows got reused as if they were center ids.
    last_location_rows: Optional[list] = None
    updated_at: str = ''


class ConversationStateStore:
    """In-memory, per-process conversation state, keyed by (user_id, session_id) so one
    user's follow-up context can never leak into another user's chat -- both the key AND
    every accessor require an explicit user_id match, so a caller can never fetch another
    user's state even by guessing/reusing a session_id.

    NOTE: this is intentionally process-local (a dict + lock), matching this app's
    single-process Flask dev-server deployment. If ForecastIQ is ever run behind
    multiple worker processes, this store must move to a shared backend (Redis, or a
    DB table) -- state would otherwise silently fail to persist across requests routed
    to different workers. See CHATBOT_CHARTS.md 'Known limitations'.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._store: dict[tuple, ConversationState] = {}

    def get(self, user_id, session_id) -> Optional[ConversationState]:
        if session_id is None or user_id is None:
            return None
        with self._lock:
            return self._store.get((user_id, session_id))

    def set(self, user_id, session_id, state: ConversationState) -> None:
        if session_id is None or user_id is None:
            return
        with self._lock:
            self._store[(user_id, session_id)] = state

    def clear(self, user_id, session_id) -> None:
        with self._lock:
            self._store.pop((user_id, session_id), None)

    def clear_all_for_user(self, user_id) -> None:
        with self._lock:
            for key in [k for k in self._store if k[0] == user_id]:
                del self._store[key]


conversation_store = ConversationStateStore()


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
