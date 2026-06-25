#!/usr/bin/env python3
"""Assemble a stratified sample of real surface turns for the teacher-quality pass.
READ-ONLY over brain_logs.db (never brain.db, never a writer). Writes one compact
JSON per turn to a temp dir + a flags.json for join-time breakdowns.

Run: ./dev python3 eval/oracle_audit/surface_quality_sample.py [target_n] [hours] [outdir]
"""
import json, os, sqlite3, sys
from collections import defaultdict

TARGET = int(sys.argv[1]) if len(sys.argv) > 1 else 90
HOURS = int(sys.argv[2]) if len(sys.argv) > 2 else 168
OUT = sys.argv[3] if len(sys.argv) > 3 else "/tmp/teacher_baseline"
DBDIR = os.environ.get("BRAIN_DB_DIR") or os.path.expanduser("~/AgentsContext/brain")
DB = os.path.join(DBDIR, "brain_logs.db")
os.makedirs(OUT, exist_ok=True)

con = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
con.row_factory = sqlite3.Row
s1 = con.execute(
    """SELECT chain_id, ref_type, ref_id, metadata, session_id, created_at
       FROM trace_events WHERE scale='s1' AND ref_type IN ('recall','surface_selected')
         AND created_at > datetime('now', ?) ORDER BY created_at""",
    (f"-{HOURS} hours",),
).fetchall()

chains = defaultdict(dict)
for r in s1:
    try:
        md = json.loads(r["metadata"]) if r["metadata"] else {}
    except Exception:
        md = {}
    chains[r["chain_id"]][r["ref_type"]] = dict(md=md, ref_id=r["ref_id"],
                                                sid=r["session_id"], ts=r["created_at"])

def parse_cands(md):
    out = []
    for c in md.get("candidates", []):
        p = c.split("|")
        if len(p) >= 3:
            try:
                out.append((p[0], p[1], float(p[-2])))
            except ValueError:
                pass
    out.sort(key=lambda x: x[2], reverse=True)
    return out

def parse_sel(k):
    try:
        ids = json.loads(k["ref_id"])
        if isinstance(ids, list):
            return [str(i).split("|")[0] for i in ids]
    except Exception:
        pass
    return [s.split("|")[0] for s in k["md"].get("selected", [])]

def tool_queries(md):
    qs = []
    for rnd in md.get("tool_trace", []) or []:
        for tc in (rnd.get("tool_calls", []) if isinstance(rnd, dict) else []):
            a = tc.get("args") or tc.get("input") or {}
            q = a.get("query") or a.get("phrase") or a.get("entity")
            if q:
                qs.append(f"{tc.get('tool','?')}: {q}")
    return qs

def next_turn(sid, after_ts):
    row = con.execute(
        """SELECT metadata FROM trace_events WHERE session_id=? AND ref_type='assistant_message'
           AND created_at > ? ORDER BY created_at LIMIT 1""", (sid, after_ts)).fetchone()
    if not row:
        return ""
    try:
        return (json.loads(row["metadata"]).get("content") or "")[:900]
    except Exception:
        return ""

valid = []
for cid, ev in chains.items():
    if "recall" not in ev or "surface_selected" not in ev:
        continue
    cc = parse_cands(ev["recall"]["md"])
    sel = parse_sel(ev["surface_selected"])
    if not cc or not sel:
        continue
    valid.append((ev["recall"]["ts"], cid, ev, cc, sel))

valid.sort()
step = max(1, len(valid) // TARGET)
sample = valid[::step][:TARGET]

flags = {}
for i, (ts, cid, ev, cc, sel) in enumerate(sample):
    rank = {cid_: n + 1 for n, (cid_, _, _) in enumerate(cc)}
    pool_ids = set(rank)
    n_off = sum(1 for s in sel if s not in pool_ids)
    prov = ("all_from_pool" if n_off == 0 else
            "all_off_pool" if n_off == len(sel) else "mixed")
    q = ev["recall"]["md"].get("query", "")
    rec = {
        "turn_id": cid,
        "ts": ts,
        "query": q,
        "candidates": [[rank[c2], round(s, 3), c2, t[:70]] for c2, t, s in cc],
        "picks": [{"id": s, "from_pool": s in pool_ids,
                   "pool_rank": rank.get(s)} for s in sel],
        "tool_queries": tool_queries(ev["surface_selected"]["md"]),
        "my_next_turn": next_turn(ev["recall"]["sid"], ts),
    }
    with open(os.path.join(OUT, f"turn_{i:03d}.json"), "w") as f:
        json.dump(rec, f, indent=1)
    flags[cid] = {"idx": i, "provenance": prov, "n_offpool": n_off,
                  "contested": cc[0][0] not in sel, "q_len": len(q)}

with open(os.path.join(OUT, "flags.json"), "w") as f:
    json.dump(flags, f, indent=1)
con.close()

from collections import Counter
pc = Counter(v["provenance"] for v in flags.values())
print(f"wrote {len(sample)} turn files to {OUT}  (of {len(valid)} valid turns, {HOURS}h)")
print(f"provenance: {dict(pc)}")
print(f"contested (top-1 dropped): {sum(v['contested'] for v in flags.values())}")
print(f"short queries (<=40 chars): {sum(v['q_len']<=40 for v in flags.values())}")
print(f"N={len(sample)} DIR={OUT}")
