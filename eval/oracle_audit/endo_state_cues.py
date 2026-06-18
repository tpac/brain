#!/usr/bin/env python3
"""STEP 1 of the reverse-engineering regression: assemble, per corpus cue, ALL the
realizable STATE-CUES available at that moment, so STEP 2 can regress which
(state-cue x mechanism) would have surfaced the gold. READ-ONLY over brain_logs.db.

State-cues (all from BEFORE the cutoff = realizable at recall time):
  cue_text       -- the prompt itself (operator msg or Anchor turn)        [have it]
  prev_anchor    -- Anchor's previous turn
  prev_operator  -- the operator's previous message
  recent_context -- concat of the last 4 turns (arc proxy)
  in_context_ids -- node ids surfaced in prior turns this session (S1R surface_selected)
  next_move      -- the ORACLE / future (already in corpus); ceiling, NOT realizable

Run: ./dev python3 eval/oracle_audit/endo_state_cues.py
"""
import json, os, sqlite3
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.join(HERE, "endo_corpus")
corpus = json.load(open(f"{OUT}/endo_gold_corpus.json"))
cmeta = {c["cand_id"]: c for c in json.load(open(f"{OUT}/candidates.json"))}
DBDIR = os.environ.get("BRAIN_DB_DIR") or os.path.expanduser("~/AgentsContext/brain")
con = sqlite3.connect(f"file:{os.path.join(DBDIR,'brain_logs.db')}?mode=ro", uri=True)
con.row_factory = sqlite3.Row

def content(md):
    try:
        return json.loads(md).get("content") or ""
    except Exception:
        return ""

def sel_ids(row):
    out = []
    try:
        rid = row["ref_id"]
        ids = json.loads(rid) if rid else []
        if isinstance(ids, list):
            out = [str(i).split("|")[0] for i in ids]
    except Exception:
        pass
    if not out:
        try:
            out = [s.split("|")[0] for s in (json.loads(row["metadata"]) or {}).get("selected", [])]
        except Exception:
            pass
    return out

out = []
for c in corpus:
    sess = cmeta.get(c["id"], {}).get("session")
    cutoff = c["cutoff"]
    rec = {"cue_id": c["id"], "source": c["source"], "session": sess, "cutoff": cutoff,
           "prev_anchor": "", "prev_operator": "", "recent_context": "", "in_context_ids": []}
    if sess:
        turns = con.execute(
            """SELECT created_at, ref_type, metadata FROM trace_events
               WHERE session_id=? AND ref_type IN ('user_message','assistant_message') AND created_at < ?
               ORDER BY created_at""", (sess, cutoff)).fetchall()
        anchors = [content(t["metadata"]) for t in turns if t["ref_type"] == "assistant_message"]
        users = [content(t["metadata"]) for t in turns if t["ref_type"] == "user_message"]
        rec["prev_anchor"] = anchors[-1][:1400] if anchors else ""
        rec["prev_operator"] = users[-1][:1000] if users else ""
        rec["recent_context"] = " \n ".join(content(t["metadata"]) for t in turns[-4:])[:2000]
        surf = con.execute(
            """SELECT ref_id, metadata FROM trace_events
               WHERE session_id=? AND ref_type='surface_selected' AND created_at < ?
               ORDER BY created_at""", (sess, cutoff)).fetchall()
        seen = []
        for s in surf:
            seen += sel_ids(s)
        rec["in_context_ids"] = sorted(set(seen))
    out.append(rec)
con.close()

json.dump(out, open(f"{OUT}/state_cues.json", "w"), indent=1)
n = len(out)
def pct(k): return f"{sum(1 for r in out if r[k])*100//n}%"
print(f"assembled state-cues for {n} cues -> state_cues.json")
print(f"  have prev_anchor:    {sum(1 for r in out if r['prev_anchor'])}/{n} ({pct('prev_anchor')})")
print(f"  have prev_operator:  {sum(1 for r in out if r['prev_operator'])}/{n} ({pct('prev_operator')})")
print(f"  have recent_context: {sum(1 for r in out if r['recent_context'])}/{n} ({pct('recent_context')})")
nic = [len(r['in_context_ids']) for r in out]
print(f"  have in_context_ids: {sum(1 for x in nic if x)}/{n}  (min/median/max = {min(nic)}/{sorted(nic)[n//2]}/{max(nic)})")
print(f"  by source: {dict((s, sum(1 for r in out if r['source']==s)) for s in ('operator_msg','anchor_turn'))}")