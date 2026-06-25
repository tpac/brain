#!/usr/bin/env python3
"""Assemble compact teacher-judgment briefs from traces (READ-ONLY brain_logs.db).
For 'contested' surface turns (the top-1 cosine candidate was NOT selected), print:
the query, the top-N candidates (marked if picked), picks outside top-N, and the
ACTUAL next assistant turn. A human/Opus teacher then judges whether the picks
served the move or a better candidate was dropped.

Run: ./dev python3 eval/oracle_audit/surface_teacher_briefs.py [n_samples] [hours]
"""
import json, os, sqlite3, sys
from collections import defaultdict

N = int(sys.argv[1]) if len(sys.argv) > 1 else 8
HOURS = int(sys.argv[2]) if len(sys.argv) > 2 else 168
DBDIR = os.environ.get("BRAIN_DB_DIR") or os.path.expanduser("~/AgentsContext/brain")
DB = os.path.join(DBDIR, "brain_logs.db")

con = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
con.row_factory = sqlite3.Row

s1 = con.execute(
    """SELECT chain_id, ref_type, ref_id, metadata, session_id, created_at
       FROM trace_events
       WHERE scale='s1' AND ref_type IN ('recall','surface_selected')
         AND created_at > datetime('now', ?) ORDER BY created_at""",
    (f"-{HOURS} hours",),
).fetchall()

chains = defaultdict(dict)
for r in s1:
    md = {}
    try:
        md = json.loads(r["metadata"]) if r["metadata"] else {}
    except Exception:
        pass
    chains[r["chain_id"]][r["ref_type"]] = dict(md=md, ref_id=r["ref_id"],
                                                 sid=r["session_id"], ts=r["created_at"])

def cands(md):
    out = []
    for c in md.get("candidates", []):
        p = c.split("|")
        if len(p) >= 3:
            try:
                sc = float(p[-2])
            except ValueError:
                sc = None
            out.append((p[0], p[1], sc))
    out = [(i, t, s) for i, t, s in out if s is not None]
    out.sort(key=lambda x: x[2], reverse=True)
    return out

def selected(k):
    try:
        ids = json.loads(k["ref_id"])
        if isinstance(ids, list):
            return [str(i).split("|")[0] for i in ids]
    except Exception:
        pass
    return [s.split("|")[0] for s in k["md"].get("selected", [])]

# find contested turns: top-1 cosine dropped, pool present, >=1 pick
contested = []
for cid, ev in chains.items():
    if "recall" not in ev or "surface_selected" not in ev:
        continue
    cc = cands(ev["recall"]["md"])
    sel = selected(ev["surface_selected"])
    if not cc or not sel:
        continue
    if cc[0][0] not in sel:  # top-1 cosine NOT selected
        contested.append((ev["recall"]["ts"], cid, ev, cc, sel))

contested.sort()
# spread the sample evenly across the contested set
step = max(1, len(contested) // N)
sample = contested[::step][:N]

def next_turn(sid, after_ts):
    row = con.execute(
        """SELECT metadata FROM trace_events
           WHERE session_id=? AND ref_type='assistant_message' AND created_at > ?
           ORDER BY created_at LIMIT 1""",
        (sid, after_ts),
    ).fetchone()
    if not row:
        return "(no following assistant turn found)"
    try:
        return (json.loads(row["metadata"]).get("content") or "")[:600]
    except Exception:
        return "(unparseable)"

print(f"\n=== TEACHER BRIEFS — {len(sample)} contested turns (of {len(contested)} total, last {HOURS}h) ===\n")
for ts, cid, ev, cc, sel in sample:
    rank = {i: n + 1 for n, (i, _, _) in enumerate(cc)}
    q = ev["recall"]["md"].get("query", "")[:240]
    print(f"────────────────────────────────────────────────────────")
    print(f"{cid}  @ {ts[:19]}")
    print(f"QUERY: {q}")
    print(f"top-8 candidates (✓=picked):")
    for i, t, s in cc[:8]:
        mark = "✓" if i in sel else " "
        print(f"  {mark} r{rank[i]:<2} {s:.3f}  {t[:62]}")
    offtop = [s for s in sel if s in rank and rank[s] > 8]
    offpool = [s for s in sel if s not in rank]
    if offtop:
        print(f"picks below top-8: " + ", ".join(f"{s}(r{rank[s]})" for s in offtop))
    if offpool:
        print(f"picks off-pool (tool-fetched): {len(offpool)} -> {', '.join(offpool)}")
    print(f"MY NEXT TURN: {next_turn(ev['recall']['sid'], ts)}")
    print()

con.close()
