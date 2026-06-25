#!/usr/bin/env python3
"""Sample ENDO cues: Anchor's own turns (the Stop self-cue), paired with the NEXT
Anchor turn (the move the recognition would change). READ-ONLY over brain_logs.db.
The recall-with-cutoff itself runs later (workflow agents call the recall MCP tool);
this only assembles the cue + next-move + cutoff per turn.

Run: ./dev python3 eval/oracle_audit/surface_endo_sample.py [target_n] [hours] [outdir]
"""
import json, os, sqlite3, sys
from collections import defaultdict

TARGET = int(sys.argv[1]) if len(sys.argv) > 1 else 60
HOURS = int(sys.argv[2]) if len(sys.argv) > 2 else 168
OUT = sys.argv[3] if len(sys.argv) > 3 else "/tmp/endo_baseline"
DBDIR = os.environ.get("BRAIN_DB_DIR") or os.path.expanduser("~/AgentsContext/brain")
DB = os.path.join(DBDIR, "brain_logs.db")
os.makedirs(OUT, exist_ok=True)

con = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
con.row_factory = sqlite3.Row
rows = con.execute(
    """SELECT id, session_id, created_at, metadata FROM trace_events
       WHERE ref_type='assistant_message' AND created_at > datetime('now', ?)
       ORDER BY session_id, created_at""", (f"-{HOURS} hours",)).fetchall()
con.close()

def content(md_str):
    try:
        return json.loads(md_str).get("content") or ""
    except Exception:
        return ""

# pair each Anchor turn with the NEXT Anchor turn in the same session
by_sess = defaultdict(list)
for r in rows:
    by_sess[r["session_id"]].append(r)

pairs = []
for sid, turns in by_sess.items():
    for i in range(len(turns) - 1):
        cue, nxt = turns[i], turns[i + 1]
        ctext, ntext = content(cue["metadata"]), content(nxt["metadata"])
        if len(ctext) < 200:   # skip trivial sign-offs as cues — no real "output" to recognize off
            continue
        pairs.append((cue["created_at"], cue["id"], sid, ctext, ntext))

pairs.sort()
step = max(1, len(pairs) // TARGET)
sample = pairs[::step][:TARGET]

for i, (ts, cid, sid, ctext, ntext) in enumerate(sample):
    rec = {
        "cue_id": cid,
        "ts": ts,
        "cutoff": ts,                      # recall must exclude nodes created at/after the cue
        "session": sid,
        "cue_text": ctext[-1400:],         # the tail of my turn — the freshest output (the Stop cue)
        "next_move": ntext[:1100],         # my actual next turn — what the recognition could change
    }
    with open(os.path.join(OUT, f"cue_{i:03d}.json"), "w") as f:
        json.dump(rec, f, indent=1)

print(f"wrote {len(sample)} endo cues to {OUT}  (of {len(pairs)} candidate Anchor-turn pairs, {HOURS}h)")
print(f"N={len(sample)} DIR={OUT}")
# echo cue 0 so we can validate the loop inline
print("\n--- cue_000 (for inline validation) ---")
r0 = json.load(open(os.path.join(OUT, "cue_000.json")))
print(f"cue_id={r0['cue_id']} cutoff={r0['cutoff']}")
print(f"CUE TAIL: ...{r0['cue_text'][-500:]}")
print(f"NEXT MOVE: {r0['next_move'][:300]}")
