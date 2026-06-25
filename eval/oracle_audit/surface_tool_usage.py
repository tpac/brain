#!/usr/bin/env python3
"""What tools does Haiku fire at the surface, with what queries, returning what?
READ-ONLY pass over brain_logs.db trace_events (never brain.db, never a writer).
Reads each s1r K (surface_selected) event's tool_trace + selected set.

Run: ./dev python3 eval/oracle_audit/surface_tool_usage.py [hours]
"""
import json, os, sqlite3, sys
from collections import Counter, defaultdict

HOURS = int(sys.argv[1]) if len(sys.argv) > 1 else 168
DBDIR = os.environ.get("BRAIN_DB_DIR") or os.path.expanduser("~/AgentsContext/brain")
DB = os.path.join(DBDIR, "brain_logs.db")

con = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
con.row_factory = sqlite3.Row
rows = con.execute(
    """SELECT metadata FROM trace_events
       WHERE scale='s1' AND ref_type='surface_selected'
         AND created_at > datetime('now', ?)""",
    (f"-{HOURS} hours",),
).fetchall()
con.close()

n = 0
turns_with_tools = 0
calls_per_turn = []
tool_freq = Counter()
rounds_per_turn = []
sample_queries = defaultdict(list)
result_counts = []
raw_samples = []

def get(d, *keys):
    for k in keys:
        if isinstance(d, dict) and d.get(k) not in (None, ""):
            return d[k]
    return None

for r in rows:
    try:
        md = json.loads(r["metadata"]) if r["metadata"] else {}
    except Exception:
        continue
    tt = md.get("tool_trace")
    if tt is None:
        continue
    n += 1
    n_calls = 0
    rounds_per_turn.append(len(tt))
    for rnd in tt:
        for tc in rnd.get("tool_calls", []) if isinstance(rnd, dict) else []:
            n_calls += 1
            if len(raw_samples) < 3:
                raw_samples.append(tc)
            name = get(tc, "name", "tool", "tool_name") or "?"
            tool_freq[name] += 1
            inp = get(tc, "input", "args", "arguments") or {}
            q = get(inp, "query", "q", "text", "entity") if isinstance(inp, dict) else None
            if q and len(sample_queries[name]) < 6:
                sample_queries[name].append(q[:80])
            res = get(tc, "results", "result_ids", "nodes", "result")
            rc = get(tc, "result_count", "n_results")
            if isinstance(res, list):
                result_counts.append(len(res))
            elif isinstance(rc, int):
                result_counts.append(rc)
    calls_per_turn.append(n_calls)
    if n_calls:
        turns_with_tools += 1

def pct(a, b):
    return f"{100*a/b:.1f}%" if b else "n/a"

print(f"\n=== SURFACE TOOL USAGE  (last {HOURS}h, read-only) ===")
print(f"DB: {DB}")
print(f"K events with a tool_trace: {n}\n")

print("-- did Haiku fire any tool? --")
print(f"  turns with >=1 tool call: {turns_with_tools}  ({pct(turns_with_tools, n)})")
if calls_per_turn:
    print(f"  mean tool calls / turn: {sum(calls_per_turn)/len(calls_per_turn):.2f}")
    print(f"  mean rounds / turn: {sum(rounds_per_turn)/len(rounds_per_turn):.2f}")
cc = Counter(calls_per_turn)
for k in sorted(cc):
    print(f"    {k} calls: {cc[k]:4d} ({pct(cc[k], len(calls_per_turn))})")
print()

print("-- tool frequency --")
tot = sum(tool_freq.values())
for name, c in tool_freq.most_common():
    print(f"  {name:22s} {c:4d}  ({pct(c, tot)})")
print(f"  total tool calls: {tot}\n")

if result_counts:
    print("-- results returned per tool call --")
    print(f"  mean: {sum(result_counts)/len(result_counts):.1f}   max: {max(result_counts)}   (n={len(result_counts)})\n")

print("-- sample queries per tool --")
for name, qs in sample_queries.items():
    print(f"  [{name}]")
    for q in qs:
        print(f"     - {q}")
print()

print("-- raw schema of first tool_calls (for parsing confirmation) --")
for s in raw_samples:
    print("  " + json.dumps(s)[:400])
