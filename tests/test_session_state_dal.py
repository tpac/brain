"""Unit tests for SessionStateDAL's session-context lifecycle methods.

These three methods were added when the raw session_state SQL in brain.py
(get_or_create_session / present_streams / live_sessions) was migrated onto
the DAL (DAL cleanup Phase 4). The load-bearing semantic is ensure_default's
INSERT-OR-IGNORE: it must NOT clobber a racing thread's already-mutated row,
which is exactly what distinguishes it from set()'s upsert.

Run: BRAIN_ALLOW_ANY_PYTHON=1 python3 -m pytest tests/test_session_state_dal.py -v
"""
import sqlite3
import json
import pytest

from servers.dal import SessionStateDAL

_DDL = """CREATE TABLE session_state (
    session_id TEXT NOT NULL,
    key TEXT NOT NULL,
    node_id TEXT NOT NULL DEFAULT '',
    value TEXT,
    updated_at TEXT,
    PRIMARY KEY (session_id, key, node_id)
)"""


@pytest.fixture
def dal():
    conn = sqlite3.connect(":memory:")
    conn.execute(_DDL)
    conn.commit()
    return SessionStateDAL(conn)


class TestEnsureDefault:
    def test_inserts_when_absent(self, dal):
        dal.ensure_default("s1", "_session_context", '{"stop_counter": 0}')
        assert dal.get("s1", "_session_context") == '{"stop_counter": 0}'

    def test_does_not_clobber_existing(self, dal):
        # A racing thread already wrote live state under this key...
        dal.set("s1", "_session_context", '{"stop_counter": 7}')
        # ...ensure_default must preserve it, not reset to the default.
        dal.ensure_default("s1", "_session_context", '{"stop_counter": 0}')
        assert dal.get("s1", "_session_context") == '{"stop_counter": 7}'

    def test_set_does_clobber(self, dal):
        # Contrast: set() is an upsert — it overwrites. This is why a
        # separate ensure_default method exists.
        dal.set("s1", "_session_context", '{"stop_counter": 7}')
        dal.set("s1", "_session_context", '{"stop_counter": 0}')
        assert dal.get("s1", "_session_context") == '{"stop_counter": 0}'


class TestRecentlyUpdated:
    def test_filters_cutoff_excludes_self_and_orders(self, dal):
        dal.set("old", "_session_context", "{}")          # ts ~ now
        dal.set("mid", "_session_context", "{}")
        dal.set("new", "_session_context", "{}")
        # Manually skew updated_at so ordering is deterministic.
        dal.conn.execute("UPDATE session_state SET updated_at=? WHERE session_id=?",
                         ("2026-05-31T10:00:00+00:00", "old"))
        dal.conn.execute("UPDATE session_state SET updated_at=? WHERE session_id=?",
                         ("2026-05-31T11:00:00+00:00", "mid"))
        dal.conn.execute("UPDATE session_state SET updated_at=? WHERE session_id=?",
                         ("2026-05-31T12:00:00+00:00", "new"))
        dal.conn.commit()

        rows = dal.recently_updated("_session_context",
                                    cutoff_iso="2026-05-31T10:30:00+00:00",
                                    exclude_session="new")
        ids = [r["session_id"] for r in rows]
        assert ids == ["mid"]            # old below cutoff, new excluded
        assert rows[0]["updated_at"] == "2026-05-31T11:00:00+00:00"

    def test_wrong_key_ignored(self, dal):
        dal.set("s1", "fatigue", "{}")   # different key
        assert dal.recently_updated("_session_context",
                                    cutoff_iso="2000-01-01T00:00:00+00:00") == []


class TestSessionsByMessageCount:
    def test_filters_by_json_message_count(self, dal):
        dal.set("busy", "_session_context", json.dumps({"message_count": 12}))
        dal.set("quiet", "_session_context", json.dumps({"message_count": 2}))
        dal.set("nometa", "_session_context", json.dumps({"stop_counter": 1}))  # missing → 0

        ids = dal.sessions_by_message_count("_session_context", min_messages=5)
        assert ids == ["busy"]
