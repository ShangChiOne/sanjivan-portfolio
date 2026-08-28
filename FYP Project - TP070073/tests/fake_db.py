"""
A minimal in-memory stand-in for the pymysql connection app.py uses, built to answer the
exact, small set of SQL queries the chat pipeline issues (see app/app.py: load_user,
_load_session_forecasts_df, _load_session_history_df, _build_chat_context_from_session).

This is intentionally NOT a general SQL engine -- it pattern-matches on distinctive
substrings of each known query string and returns canned rows from an in-memory fixture,
which is enough to exercise the real Flask route (login, ownership checks, streaming)
without a running MySQL server.
"""
import re
from contextlib import contextmanager


def _norm(sql: str) -> str:
    return re.sub(r'\s+', ' ', sql).strip()


class FakeDB:
    def __init__(self):
        self.users = {}           # id -> dict
        self.sessions = {}        # id -> dict (must include 'user_id')
        self.session_meals = {}   # session_id -> list[dict]
        self.forecasts = {}       # session_id -> list[dict] (week, center_id, meal_id, category, cuisine, ai_forecast)


class FakeCursor:
    def __init__(self, db: FakeDB):
        self.db = db
        self._rows = []

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def execute(self, sql, params=None):
        q = _norm(sql)
        params = params or ()

        if 'FROM users WHERE id' in q:
            uid = int(params[0])
            row = self.db.users.get(uid)
            self._rows = [dict(row)] if row else []
            return

        if 'SELECT total_records, weeks, centers, meals FROM sessions' in q:
            sid, uid = int(params[0]), params[1]
            s = self.db.sessions.get(sid)
            if s and s['user_id'] == uid:
                self._rows = [{'total_records': s['total_records'], 'weeks': s['weeks'],
                               'centers': s['centers'], 'meals': s['meals']}]
            else:
                self._rows = []
            return

        if 'SELECT feat_json FROM sessions' in q:
            sid, uid = int(params[0]), params[1]
            s = self.db.sessions.get(sid)
            if s and s['user_id'] == uid:
                self._rows = [{'feat_json': s.get('feat_json')}]
            else:
                self._rows = []
            return

        if 'FROM session_meals WHERE session_id' in q:
            sid = int(params[0])
            self._rows = [dict(m) for m in self.db.session_meals.get(sid, [])]
            return

        if 'GROUP BY week ORDER BY week' in q:
            sid = int(params[0])
            totals = {}
            for r in self.db.forecasts.get(sid, []):
                totals[r['week']] = totals.get(r['week'], 0) + r['ai_forecast']
            self._rows = [{'week': w, 'forecast': v} for w, v in sorted(totals.items())]
            return

        if 'GROUP BY week, category' in q:
            sid = int(params[0])
            totals = {}
            for r in self.db.forecasts.get(sid, []):
                if r.get('category') is None:
                    continue
                key = (r['week'], r['category'])
                totals[key] = totals.get(key, 0) + r['ai_forecast']
            rows = [{'week': w, 'category': c, 'forecast': v} for (w, c), v in totals.items()]
            rows.sort(key=lambda r: (r['week'], -r['forecast']))
            self._rows = rows
            return

        if 'GROUP BY center_id' in q:
            sid = int(params[0])
            totals = {}
            for r in self.db.forecasts.get(sid, []):
                totals[r['center_id']] = totals.get(r['center_id'], 0) + r['ai_forecast']
            rows = [{'center_id': c, 'forecast': v} for c, v in totals.items()]
            rows.sort(key=lambda r: -r['forecast'])
            self._rows = rows[:30]
            return

        if 'JOIN sessions s ON s.id = f.session_id' in q:
            sid, uid = int(params[0]), params[1]
            s = self.db.sessions.get(sid)
            if not s or s['user_id'] != uid:
                self._rows = []
                return
            name_map = {m['meal_id']: m.get('item_name') for m in self.db.session_meals.get(sid, [])}
            self._rows = [
                {'week': r['week'], 'center_id': r['center_id'], 'meal_id': r['meal_id'],
                 'category': r.get('category'), 'cuisine': r.get('cuisine'),
                 'ai_forecast': r['ai_forecast'], 'item_name': name_map.get(r['meal_id'])}
                for r in self.db.forecasts.get(sid, [])
            ]
            return

        raise AssertionError(f'FakeCursor: unrecognized query: {q!r}')

    def fetchone(self):
        return self._rows[0] if self._rows else None

    def fetchall(self):
        return list(self._rows)


class FakeConn:
    def __init__(self, db: FakeDB):
        self.db = db

    def cursor(self):
        return FakeCursor(self.db)

    def close(self):
        pass
