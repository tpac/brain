#!/usr/bin/env python3
"""Randomly sample N auditable user turns for the Oracle Audit validation pass.

See docs/ORACLE-AUDIT-SPEC.md. Read-only against brain_logs.db (no writes, no
second writer on the live DB). An "auditable" turn is a user_message that
triggered a recall (has metadata.recall_chain), so its surfaced/candidate
traces exist to diff against.

Usage:  ./dev python3 eval/oracle_audit/sample_turns.py [N=30] [SEED=4] [EXCLUDE_JSON]
  EXCLUDE_JSON: path to a prior sample_seed*.json whose trace_ids to exclude,
                so a second draw is disjoint from the first.
Output: eval/oracle_audit/sample_seed{SEED}.json
"""
import sqlite3, json, os, random, sys

DB = "/Users/tpac/AgentsContext/brain/brain_logs.db"
OUTDIR = "/Users/tpac/brain/.claude/worktrees/frosty-feistel-90c7a9/eval/oracle_audit"
N = int(sys.argv[1]) if len(sys.argv) > 1 else 30
SEED = int(sys.argv[2]) if len(sys.argv) > 2 else 4  # fixed → reproducible sample
EXCLUDE = sys.argv[3] if len(sys.argv) > 3 else None

c = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
rows = c.execute(
    """
    select id, chain_id, session_id, created_at,
           json_extract(metadata,'$.content')      as content,
           json_extract(metadata,'$.recall_chain') as recall_chain,
           summary
    from trace_events
    where ref_type='user_message'
      and json_extract(metadata,'$.recall_chain') is not null
    """
).fetchall()

if EXCLUDE:
    prev = {r["trace_id"] for r in json.load(open(EXCLUDE))}
    rows = [r for r in rows if r[0] not in prev]

random.seed(SEED)
sample = random.sample(rows, min(N, len(rows)))
out = []
for i, (tid, s0, sid, ts, content, rchain, summ) in enumerate(sample):
    out.append({
        "i": i, "trace_id": tid, "s0_chain": s0, "recall_chain": rchain,
        "session_id": sid, "created_at": ts,
        "prompt": (content or summ or "").strip(),
    })

print(f"population={len(rows)} auditable turns | sampled={len(sample)} (seed={SEED})"
      + (f" | excluded prior {len(prev)}" if EXCLUDE else "") + "\n")
for o in out:
    print(f"[{o['i']:>2}] {o['created_at'][:16]}  {o['recall_chain']}")
    print(f"     {o['prompt'].replace(chr(10),' ')[:170]}")

os.makedirs(OUTDIR, exist_ok=True)
out_path = os.path.join(OUTDIR, f"sample_seed{SEED}.json")
with open(out_path, "w") as f:
    json.dump(out, f, indent=2)
print(f"\nwrote {out_path}")
