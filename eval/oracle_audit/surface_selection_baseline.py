#!/usr/bin/env python3
"""Surface-selection baseline: does Haiku's pick diverge from the cosine ranking
it was handed?  READ-ONLY pass over brain_logs.db trace_events (never brain.db,
never a writer).  Pairs each s1r chain's O (recall: 25 candidates w/ cosine) with
its K (surface_selected: Haiku's picks) and reports aggregates only.

Run: ./dev python3 eval/oracle_audit/surface_selection_baseline.py [hours]
"""
import json, os, sqlite3, sys, statistics as st
from collections import Counter, defaultdict

HOURS = int(sys.argv[1]) if len(sys.argv) > 1 else 168
DBDIR = os.environ.get("BRAIN_DB_DIR") or os.path.expanduser("~/AgentsContext/brain")
DB = os.path.join(DBDIR, "brain_logs.db")

con = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)  # read-only, no writes/pragmas
con.row_factory = sqlite3.Row

# cutoff via julianday (numeric — safe vs the TEXT-timestamp lex trap)
rows = con.execute(
    """SELECT chain_id, event_type, ref_type, ref_id, metadata
       FROM trace_events
       WHERE scale='s1' AND ref_type IN ('recall','surface_selected')
         AND created_at > datetime('now', ?) """,  # logs are wall-clock; fine here
    (f"-{HOURS} hours",),
).fetchall()
con.close()

chains = defaultdict(dict)
for r in rows:
    try:
        md = json.loads(r["metadata"]) if r["metadata"] else {}
    except Exception:
        md = {}
    chains[r["chain_id"]][r["ref_type"]] = {"ref_id": r["ref_id"], "md": md}

def parse_candidates(md):
    """O.metadata.candidates = ['id|title|score|type', ...] sorted desc by score."""
    out = []
    for c in md.get("candidates", []):
        parts = c.split("|")
        if len(parts) >= 3:
            cid = parts[0]
            try:
                score = float(parts[-2]) if len(parts) >= 4 else float(parts[2])
            except ValueError:
                # score field position varies; find the float
                score = next((float(p) for p in parts if _isf(p)), None)
            out.append((cid, score))
    # rank by score desc
    out = [(cid, s) for cid, s in out if s is not None]
    out.sort(key=lambda x: x[1], reverse=True)
    return out  # [(id, score)] rank = index+1

def _isf(p):
    try:
        float(p); return True
    except ValueError:
        return False

def parse_selected(k):
    """selected ids: prefer K.ref_id JSON array; fall back to md.selected 'id|title'."""
    rid = k.get("ref_id") or ""
    try:
        ids = json.loads(rid)
        if isinstance(ids, list):
            return [str(i).split("|")[0] for i in ids]
    except Exception:
        pass
    return [s.split("|")[0] for s in k.get("md", {}).get("selected", [])]

# ---- aggregate ----
n_turns = 0
n_surface_dist = Counter()
picks_from25 = 0
picks_offpool = 0          # tool-fetched / not in the 25
sel_ranks = []             # cosine-rank of from-25 picks (1 = top of pool)
top1_selected = 0          # turns where the #1-cosine candidate was selected
top1_dropped = 0           # turns where #1-cosine was NOT selected (pool non-empty, >=1 pick)
top3_coverage = []         # frac of the pool's top-3 cosine that got selected
pool_min, pool_max, pool_spread = [], [], []
turns_with_pool = 0

for cid, ev in chains.items():
    if "recall" not in ev or "surface_selected" not in ev:
        continue
    n_turns += 1
    cands = parse_candidates(ev["recall"]["md"])
    sel = parse_selected(ev["surface_selected"])
    n_surface_dist[len(sel)] += 1
    if not cands:
        continue
    turns_with_pool += 1
    pool_min.append(cands[-1][1]); pool_max.append(cands[0][1])
    pool_spread.append(cands[0][1] - cands[-1][1])
    rank = {cid_: i + 1 for i, (cid_, _) in enumerate(cands)}
    pool_ids = set(rank)
    top1_id = cands[0][0]
    top3_ids = {cid_ for cid_, _ in cands[:3]}

    if sel:
        if top1_id in sel:
            top1_selected += 1
        else:
            top1_dropped += 1
        top3_coverage.append(len(top3_ids & set(sel)) / min(3, len(cands)))

    for s in sel:
        if s in pool_ids:
            picks_from25 += 1
            sel_ranks.append(rank[s])
        else:
            picks_offpool += 1

def pct(a, b):
    return f"{100*a/b:.1f}%" if b else "n/a"

def mean(xs):
    return f"{st.mean(xs):.2f}" if xs else "n/a"

print(f"\n=== SURFACE SELECTION BASELINE  (last {HOURS}h, read-only) ===")
print(f"DB: {DB}")
print(f"paired O+K turns: {n_turns}   (with a candidate pool: {turns_with_pool})\n")

print("-- selection volume (#nodes Haiku surfaced) --")
total = sum(n_surface_dist.values())
for k in sorted(n_surface_dist):
    print(f"  {k} surfaced: {n_surface_dist[k]:4d}  ({pct(n_surface_dist[k], total)})")
abst = n_surface_dist.get(0, 0)
mean_sel = st.mean([k for k in n_surface_dist.elements()]) if total else 0
print(f"  abstention (0): {pct(abst,total)}   mean surfaced: {mean_sel:.2f}\n")

print("-- pick provenance --")
tot_picks = picks_from25 + picks_offpool
print(f"  from the 25-cosine pool: {picks_from25}  ({pct(picks_from25, tot_picks)})")
print(f"  off-pool (tool-fetched): {picks_offpool}  ({pct(picks_offpool, tot_picks)})\n")

print("-- selection vs cosine (for from-pool picks) --")
print(f"  mean cosine-rank of picks: {mean(sel_ranks)}  (1 = top of pool)")
if sel_ranks:
    rc = Counter(sel_ranks)
    for r in range(1, 11):
        if rc.get(r):
            print(f"    rank {r:2d}: {rc[r]:4d}  ({pct(rc[r], len(sel_ranks))})")
    deep = sum(1 for r in sel_ranks if r > 5)
    print(f"    rank >5 (deep inversions): {deep} ({pct(deep, len(sel_ranks))})")
print()

print("-- did the BEST cosine candidate get selected? --")
denom = top1_selected + top1_dropped
print(f"  top-1 cosine SELECTED: {pct(top1_selected, denom)}")
print(f"  top-1 cosine DROPPED:  {pct(top1_dropped, denom)}   <-- selection diverges from cosine")
print(f"  mean top-3 cosine coverage by picks: {mean(top3_coverage)}\n")

print("-- pool cosine band (flat-space check) --")
print(f"  mean pool max: {mean(pool_max)}   mean pool min: {mean(pool_min)}   mean spread: {mean(pool_spread)}")
print()
