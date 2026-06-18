#!/usr/bin/env python3
"""STAGE 1a — assemble ENDO-WORTHY corpus CANDIDATE cues, read-only over brain_logs.db.

Recognition is hook-agnostic: the recall ENGINE is the same code for every caller, so
an engine fix pays off everywhere. We cue from BOTH attachments so the eval can TEST
that empirically (the engine fix should lift recall equally across sources; if it
doesn't, the gap is hook-specific seed-construction — itself a finding).

Two cue sources, one downstream shape:
  A. anchor_turn  — the Stop self-cue: Anchor turn[i] -> the next Anchor turn[i+1].
  B. operator_msg — the UserPromptSubmit cue: the recall query -> Anchor's next turn.

cutoff = cue timestamp (recall must exclude nodes created at/after the cue — the
cue-era brain is what recognition had to work with).

This stage is assembly ONLY: pure SQL, no embedder, no Anthropic spend. The
dense-prior pre-filter (needs the embedder / IsolatedBrain) and the teacher gold
pass come next. Over-generates on purpose; stratification down-selects later.

Run: ./dev python3 eval/oracle_audit/endo_corpus_build.py [hours] [outdir]
"""
import json, os, sqlite3, sys
from collections import defaultdict, Counter
from datetime import datetime, timezone, timedelta

HOURS = int(sys.argv[1]) if len(sys.argv) > 1 else 336      # 14d default
OUT = sys.argv[2] if len(sys.argv) > 2 else "/tmp/endo_corpus"
DBDIR = os.environ.get("BRAIN_DB_DIR") or os.path.expanduser("~/AgentsContext/brain")
DB = os.path.join(DBDIR, "brain_logs.db")
os.makedirs(OUT, exist_ok=True)

# ISO-T cutoff bound in Python — avoids the SQLite datetime('now') lex bug against
# ISO-T columns ('T' 0x54 > ' ' 0x20 silently over-includes the boundary). brain_logs
# stores ISO-T (TraceDAL routes through clock.iso_now), so a bound ISO-T string compares
# cleanly. (This is the bug flagged in surface_endo_sample.py:23.)
cutoff_iso = (datetime.now(timezone.utc) - timedelta(hours=HOURS)).isoformat()

con = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
con.row_factory = sqlite3.Row

def content(md_str):
    try:
        return json.loads(md_str).get("content") or ""
    except Exception:
        return ""

cues = []

# ---- Source A: anchor-turn cues (endo / Stop attachment) ----
arows = con.execute(
    """SELECT id, session_id, created_at, metadata FROM trace_events
       WHERE ref_type='assistant_message' AND created_at > ?
       ORDER BY session_id, created_at""", (cutoff_iso,)).fetchall()
by_sess = defaultdict(list)
for r in arows:
    by_sess[r["session_id"]].append(r)

for sid, turns in by_sess.items():
    for i in range(len(turns) - 1):
        cue, nxt = turns[i], turns[i + 1]
        ctext, ntext = content(cue["metadata"]), content(nxt["metadata"])
        if len(ctext) < 200:          # skip trivial sign-offs — nothing to recognize off
            continue
        cues.append({
            "source": "anchor_turn",
            "cue_trace_id": cue["id"],
            "ts": cue["created_at"],
            "cutoff": cue["created_at"],
            "session": sid,
            "cue_text": ctext[-1400:],   # tail = freshest output (the Stop cue)
            "next_move": ntext[:1200],   # the move the recognition could change
        })

# ---- Source B: operator-message cues (UserPromptSubmit attachment) ----
# recall (UserPromptSubmit) and assistant_message (Stop) traces do NOT alternate
# 1:1 — a long tool-use turn or a status-heavy session can leave two recalls
# before one logged Stop, so a naive "next assistant after ts" collapses two cues
# onto one next_move (observed). Pair by walking the session timeline: each recall
# stays "open" until the next Stop closes it; a newer recall supersedes an unclosed
# older one (that ambiguous pair is dropped), within a gap bound.
MAX_GAP_MIN = 20

def _dt(s):
    return datetime.fromisoformat(s)

brows = con.execute(
    """SELECT chain_id, session_id, created_at, metadata FROM trace_events
       WHERE scale='s1' AND ref_type='recall' AND created_at > ?
       ORDER BY created_at""", (cutoff_iso,)).fetchall()
rec_by_sess = defaultdict(list)
for r in brows:
    rec_by_sess[r["session_id"]].append(r)

n_drop_ambig = n_drop_gap = 0
for sid, recs in rec_by_sess.items():
    events = [(r["created_at"], "rec", r) for r in recs] + \
             [(a["created_at"], "ass", a) for a in by_sess.get(sid, [])]
    events.sort(key=lambda e: e[0])
    open_rec = None
    for ts, kind, row in events:
        if kind == "rec":
            if open_rec is not None:
                n_drop_ambig += 1          # older recall never closed -> ambiguous
            open_rec = row
            continue
        if open_rec is None:
            continue
        rec, open_rec = open_rec, None      # this Stop closes the open recall
        if (_dt(ts) - _dt(rec["created_at"])).total_seconds() > MAX_GAP_MIN * 60:
            n_drop_gap += 1
            continue
        try:
            q = (json.loads(rec["metadata"]).get("query") or "").strip()
        except Exception:
            q = ""
        ntext = content(row["metadata"])
        if len(q) < 12 or len(ntext) < 120:
            continue
        cues.append({
            "source": "operator_msg",
            "cue_trace_id": rec["chain_id"],
            "ts": rec["created_at"],
            "cutoff": rec["created_at"],
            "session": sid,
            "cue_text": q,
            "next_move": ntext[:1200],
        })
con.close()
print(f"operator_msg pairing: dropped {n_drop_ambig} ambiguous + {n_drop_gap} over-gap")

cues.sort(key=lambda c: c["ts"])
for i, c in enumerate(cues):
    c["cand_id"] = f"{c['source']}_{i:04d}"

with open(os.path.join(OUT, "candidates.json"), "w") as f:
    json.dump(cues, f, indent=1)

bysrc = Counter(c["source"] for c in cues)
print(f"window: {HOURS}h (since {cutoff_iso[:19]}Z)")
print(f"wrote {len(cues)} candidate cues -> {os.path.join(OUT, 'candidates.json')}")
print(f"by source: {dict(bysrc)}")
n_sess = len(set(c["session"] for c in cues))
print(f"distinct sessions: {n_sess}")
for s in ("anchor_turn", "operator_msg"):
    L = sorted(len(c["cue_text"]) for c in cues if c["source"] == s)
    if L:
        print(f"  {s:12s} n={len(L):4d}  cue_text len min/median/max = {L[0]}/{L[len(L)//2]}/{L[-1]}")

# echo one of each source for eyeball sanity
print("\n--- sample anchor_turn cue ---")
ex = next((c for c in cues if c["source"] == "anchor_turn"), None)
if ex:
    print(f"[{ex['cand_id']}] cutoff={ex['cutoff'][:19]}")
    print(f"CUE TAIL: ...{ex['cue_text'][-360:]}")
    print(f"NEXT MOVE: {ex['next_move'][:260]}")
print("\n--- sample operator_msg cue ---")
ex = next((c for c in cues if c["source"] == "operator_msg"), None)
if ex:
    print(f"[{ex['cand_id']}] cutoff={ex['cutoff'][:19]}")
    print(f"CUE (query): {ex['cue_text'][:360]}")
    print(f"NEXT MOVE: {ex['next_move'][:260]}")
