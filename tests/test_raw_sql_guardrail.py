"""Phase 6 guardrail: no NEW raw SQL (DML) outside the DAL layer.

The DAL-cleanup arc's whole point is "raw SQL outside dal*.py is the violation."
This test RATCHETS that: it freezes the current per-file count of raw
INSERT/UPDATE/DELETE/REPLACE sites in `servers/` (excluding the DAL + schema
layers) and fails when a file grows a new one — forcing the author to either
route it through a DAL method or consciously bump the baseline with a why.

It ratchets BOTH ways: when a real migration drops a file's count, the test
fails too, demanding the baseline be lowered — so the allowance can't silently
hide a later re-introduction.

Detection is deliberately simple (a literal DML string right after
`.execute(`/`.executemany(`). It catches the overwhelmingly common shape; SQL
assembled in a variable first is not caught (rare, and a different smell).

Run: BRAIN_ALLOW_ANY_PYTHON=1 python3 -m pytest tests/test_raw_sql_guardrail.py -v
"""
import re
import pathlib
import unittest

SERVERS = pathlib.Path(__file__).resolve().parent.parent / 'servers'
# The DAL + schema layers are where raw SQL is supposed to live.
EXCLUDE = {'dal.py', 'dal_logs.py', 'dal_graph.py', 'dal_metadata.py', 'dal_vector_cached.py', 'schema.py'}
DML_RE = re.compile(
    r"\.execute(?:many)?\(\s*[rfbu]*('''|\"\"\"|'|\")\s*(INSERT|UPDATE|DELETE|REPLACE)\b",
    re.IGNORECASE)

# Frozen baseline (2026-05-31): accepted raw-DML site count per servers-relative
# path. `exception` = legitimately raw forever; `pending` = scheduled for a later
# DAL phase. Lower a number when its sites get migrated; bump (with a why) only
# for a genuine new exception.
ALLOWED = {
    'brain.py': 1,                        # remaining: 1 brain.db INSERT; logs-DB prune DELETEs routed to LogsDAL.prune_oversize (2026-08-18 write-boundary fix)
    'brain_assembly.py': 2,               # exception: brain_meta health-check ping; logs-DB ping + boot_renders routed to LogsDAL (2026-08-18)
    'brain_connections.py': 1,
    'brain_remember.py': 4,               # pending: deferred 3c; enrichment DELETEs consolidated into VectorDAL.delete_for_node
    'daemon_server.py': 1,
    # dispatch_observability.py: 0 — clear_errors DELETEs routed to LogsDAL.clear_errors (2026-08-18)
    'dispatch_ops.py': 1,
    'recall_write_queue.py': 1,           # exception: bg-writer connection (off foreground), batched
    'scales/s2/rejection_table.py': 2,    # exception: owns all s2_rejections SQL — record_rejections INSERT + clear_unplaceable_rejections DELETE (relocated out of community.py)
    'channels/self_channel/signal.py': 4, # exception: parallel-stream file (SelfChannelDAL out of this effort); writes ride logs_conn_w under write_lock (2026-08-18)
    'channels/thalamus/thalamus.py': 9,   # exception: owns all thalamus_items/thalamus_deliveries SQL (the signal.py courier pattern); writes ride logs_conn_w under logs_write_lock. 10→9: file()'s two hand-listed INSERTs became one _insert_item shared by both routes
    # temporal_extraction.py: 0 — entity_dates writes migrated to EntityDatesDAL (Phase 5)
}


def _scan():
    counts = {}
    for p in sorted(SERVERS.rglob('*.py')):
        if p.name in EXCLUDE:
            continue
        n = len(DML_RE.findall(p.read_text()))
        if n:
            counts[str(p.relative_to(SERVERS))] = n
    return counts


# ── Write-connection cursor invariant (dal_logs.py docstring) ──────────────
# A SELECT cursor BOUND to a name on the logs write connection holds a read
# snapshot; the next write on that connection fails INSTANTLY with 'database
# is locked' the moment another process commits in between (SQLITE_BUSY_
# SNAPSHOT — busy_timeout does not apply; brain id:371895a8). The rule:
# statements on logs_conn_w are fully consumed — `conn.execute(...).fetchone()`
# / `.fetchall()`, never `cur = conn.execute('SELECT ...')`. DML bindings are
# fine (the statement completes inside execute(); rowcount is immediate).
# Empirically reproduced 2026-08-29 during the Thalamus review: the shape
# fails whenever the SELECT matches 2+ rows. This scan resolves the common
# alias form (`conn = brain.logs_conn_w`) per file.

_WCONN_ALIAS_RE = re.compile(r'^\s*(\w+)\s*=\s*(?:\w+\.)*logs_conn_w\s*$',
                             re.MULTILINE)


def _bound_wconn_selects(text):
    """Line numbers where a SELECT *cursor* is bound on the logs write conn
    (direct or via a `x = ...logs_conn_w` alias). `x = conn.execute(...)
    .fetchone()/.fetchall()` binds the consumed RESULT, not the cursor —
    that's the safe idiom and is exempt (checked by walking to the balanced
    close of the execute call; SQL text keeps its parens balanced)."""
    names = set(_WCONN_ALIAS_RE.findall(text)) | {'logs_conn_w'}
    pat = re.compile(
        r'\w+\s*=\s*(?:\w+\.)*(%s)\.execute\(' % '|'.join(sorted(names)))
    hits = []
    for m in pat.finditer(text):
        lit = re.match(r'\s*[rfbu]*(?:\'\'\'|"""|\'|")\s*SELECT\b',
                       text[m.end():], re.IGNORECASE)
        if not lit:
            continue  # DML binding — statement completes inside execute()
        depth, i = 1, m.end()
        while i < len(text) and depth:
            if text[i] == '(':
                depth += 1
            elif text[i] == ')':
                depth -= 1
            i += 1
        if text[i:i + 8].lstrip().startswith('.fetch'):
            continue  # fully consumed — the compliant shape
        hits.append(text[:m.start()].count('\n') + 1)
    return hits


class TestRawSqlGuardrail(unittest.TestCase):
    def test_no_new_raw_dml_outside_dal(self):
        current = _scan()

        grown = [f"  {f}: {n} raw-DML (allowed {ALLOWED.get(f, 0)}) — +{n - ALLOWED.get(f, 0)} NEW"
                 for f, n in sorted(current.items()) if n > ALLOWED.get(f, 0)]
        dropped = [f"  {f}: {current.get(f, 0)} now (baseline {a}) — lower ALLOWED to {current.get(f, 0)}"
                   for f, a in sorted(ALLOWED.items()) if current.get(f, 0) < a]

        parts = []
        if grown:
            parts.append(
                "NEW raw SQL (DML) outside the DAL — route it through a DAL method, "
                "or if genuinely exceptional add/bump ALLOWED with a one-line why:\n"
                + "\n".join(grown))
        if dropped:
            parts.append(
                "Raw-DML dropped below baseline (a migration landed — good). Tighten "
                "ALLOWED so the ratchet stays honest:\n" + "\n".join(dropped))
        self.assertEqual(parts, [], "\n\n".join(parts))

    def test_no_bound_select_cursors_on_write_conn(self):
        """No `name = <write-conn>.execute('SELECT ...')` outside dal*.py —
        the snapshot-upgrade hazard the dal_logs.py docstring documents."""
        offenders = []
        for p in sorted(SERVERS.rglob('*.py')):
            if p.name in EXCLUDE:
                continue
            for line in _bound_wconn_selects(p.read_text()):
                offenders.append('  %s:%d' % (p.relative_to(SERVERS), line))
        self.assertEqual(offenders, [], (
            "Bound SELECT cursor on the logs WRITE connection — a held read "
            "snapshot makes the next write fail instantly under a concurrent "
            "commit. Consume it instead: `conn.execute(...).fetchone()` / "
            "`.fetchall()`:\n" + "\n".join(offenders)))

    def test_wconn_detector_has_teeth(self):
        catches = [
            "cur = brain.logs_conn_w.execute('SELECT id FROM t')",
            "conn = brain.logs_conn_w\nrow = conn.execute(\n    'SELECT x FROM t WHERE y = ?', (1,))",
        ]
        for s in catches:
            self.assertTrue(_bound_wconn_selects(s), f"should flag: {s!r}")
        ignores = [
            # consumed result (the compliant shape), incl. multiline + COUNT parens
            "conn = brain.logs_conn_w\nn = conn.execute(\n    'SELECT COUNT(*) FROM t WHERE x = ?',\n    (1,)).fetchone()[0]",
            "conn = brain.logs_conn_w\nconn.execute('SELECT 1').fetchall()",   # unbound
            "conn = brain.logs_conn_w\ncur = conn.execute('UPDATE t SET x=1')",  # DML binding is fine
            "rows = brain.logs_conn.execute('SELECT 1')",       # read conn
        ]
        for s in ignores:
            self.assertFalse(_bound_wconn_selects(s), f"should NOT flag: {s!r}")

    def test_detector_has_teeth(self):
        """The guardrail is only meaningful if DML_RE actually fires — assert it
        catches the shapes it must and ignores reads / DAL-bound calls."""
        catches = [
            "self.conn.execute('INSERT INTO t VALUES (1)')",
            'cur.execute("UPDATE t SET x=1")',
            "c.executemany('DELETE FROM t WHERE id=?', rows)",
            "self.conn.execute(\n    '''REPLACE INTO t VALUES (?)''', (v,))",
            "x.execute(f\"INSERT INTO t VALUES ({v})\")",
        ]
        for s in catches:
            self.assertTrue(DML_RE.search(s), f"should flag: {s!r}")
        ignores = [
            "self.conn.execute('SELECT * FROM t')",
            "self._graph.add_relation(a, b, 'rel')",   # DAL method, not raw execute
            "log = 'we INSERT later'",                  # not an execute call
        ]
        for s in ignores:
            self.assertFalse(DML_RE.search(s), f"should NOT flag: {s!r}")


if __name__ == '__main__':
    unittest.main()
