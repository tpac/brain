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
EXCLUDE = {'dal.py', 'dal_metadata.py', 'dal_vector_cached.py', 'schema.py'}
DML_RE = re.compile(
    r"\.execute(?:many)?\(\s*[rfbu]*('''|\"\"\"|'|\")\s*(INSERT|UPDATE|DELETE|REPLACE)\b",
    re.IGNORECASE)

# Frozen baseline (2026-05-31): accepted raw-DML site count per servers-relative
# path. `exception` = legitimately raw forever; `pending` = scheduled for a later
# DAL phase. Lower a number when its sites get migrated; bump (with a why) only
# for a genuine new exception.
ALLOWED = {
    'brain.py': 6,
    'brain_assembly.py': 5,               # exception: health-check / integrity audit + ping
    'brain_connections.py': 1,
    'brain_remember.py': 11,              # pending: deferred 3c (archive/content_summary/revise/personal multi-field UPDATEs)
    'daemon_server.py': 1,
    'dispatch_observability.py': 4,       # exception: observability writes
    'dispatch_ops.py': 1,
    'recall_write_queue.py': 3,           # exception: bg-writer connection (off foreground), batched
    'scales/s2/rejection_table.py': 1,
    'scales/self_channel/signal.py': 4,   # exception: parallel-stream file (SelfChannelDAL out of this effort)
    'temporal_extraction.py': 3,          # pending: Phase 5 EntityDatesDAL
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
            "self._nodes.update_field(nid, 'x', 1)",   # DAL method, not raw execute
            "log = 'we INSERT later'",                  # not an execute call
        ]
        for s in ignores:
            self.assertFalse(DML_RE.search(s), f"should NOT flag: {s!r}")


if __name__ == '__main__':
    unittest.main()
