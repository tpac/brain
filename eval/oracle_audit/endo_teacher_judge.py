#!/usr/bin/env python3
"""STAGE 1b-iii — teacher JUDGE: Opus mints endo-worthiness + gold per cue.

Uses the Anthropic API directly via the brain's ANTHROPIC_API_KEY (.env) —
NOT the Claude Code session/5-hour budget, NOT the Workflow tool. Concurrent,
structured-output-enforced. Re-runnable.

The teacher sees CUE + actual NEXT MOVE + a broad candidate union (built by
endo_teacher_prep.py), and is kept BLIND to all retrieval scoring (no lens, no
cosine, no rank) — it judges only the move's actual need, with hindsight.

Run: ./dev python3 eval/oracle_audit/endo_teacher_judge.py [limit]
  limit = max cues to judge (smoke-test with a small number first).
"""
import json, os, sys, concurrent.futures as cf
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from tests.isolated_brain import _load_env
import anthropic

MODEL = "claude-opus-4-8"
WORKERS = 6
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "endo_corpus")
LIMIT = int(sys.argv[1]) if len(sys.argv) > 1 else 10**9

_load_env()
client = anthropic.Anthropic()

SYSTEM = """You are a STRICT evaluator for an endo-recall memory system. The system (the 'brain') stores nodes — decisions, findings, principles, mechanisms — from past work between an operator (Tom) and an AI partner (Anchor). Endo-recall = the brain reflexively surfacing a relevant PRIOR node at the right moment so Anchor doesn't forget what it already knows.

You are given:
- CUE: a moment. Either Anchor's own turn (a reflexive 'Stop' self-cue) or Tom's message.
- NEXT MOVE: what Anchor actually said/did next — the outcome the recall had to serve.
- CANDIDATES: brain nodes, ALL created strictly BEFORE the cue (so each was available to recall at that moment). Each: id, type, creation date, title, content snippet.

Judge WITH hindsight (you can see the next move):

1. endo_worthy (bool): TRUE iff some candidate node existed that the next move genuinely NEEDED but that was NOT already in hand — recognizing it would have CHANGED or materially IMPROVED the move. Qualifying patterns: it answers a question being asked; it prevents re-deriving something already settled; it surfaces a forgotten prior decision/finding the move should build on or contradicts.
   FALSE when: the move is correct silence (a status update, a sign-off, a fresh discovery no prior could hold, a live code/git/file check); OR the relevant knowledge is already evident in the cue or the next move itself (the partner already has it — surfacing would be redundant). Be strict: 'topically related' is NOT enough — the move must have NEEDED it.

2. gold_essential / gold_helpful: node ids FROM THE CANDIDATES ONLY. essential = the move could not be done well without it; helpful = would have helped or compiled the answer. If not endo_worthy, both empty.

3. query_type: factual | compositional | design | action | procedural | other.

4. confidence: high | medium | low. why: 1-2 sentences citing the gold id(s), or the reason for silence.

Judge ONLY by the move's actual need — never by candidate ordering. Gold ids MUST come from the provided candidates."""

SCHEMA = {
    "type": "object",
    "properties": {
        "endo_worthy": {"type": "boolean"},
        "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
        "query_type": {"type": "string",
                       "enum": ["factual", "compositional", "design", "action", "procedural", "other"]},
        "gold_essential": {"type": "array", "items": {"type": "string"}},
        "gold_helpful": {"type": "array", "items": {"type": "string"}},
        "why": {"type": "string"},
    },
    "required": ["endo_worthy", "confidence", "query_type", "gold_essential", "gold_helpful", "why"],
    "additionalProperties": False,
}

def build_user(o):
    lines = [f"CUE [source={o['source']}, date={o['cutoff'][:10]}]:", o["cue_text"], "",
             "NEXT MOVE (what Anchor actually did next):", o["next_move"], "",
             "CANDIDATES (all created before the cue):"]
    for c in o["candidates"]:
        lines.append(f"[{c['id']}] ({c['type']}, {c['created_at']}) {c['title']}")
        lines.append(f"    {c['snippet']}")
    return "\n".join(lines)

def judge(o):
    try:
        resp = client.messages.create(
            model=MODEL, max_tokens=900, system=SYSTEM,
            messages=[{"role": "user", "content": build_user(o)}],
            output_config={"format": {"type": "json_schema", "schema": SCHEMA}},
        )
        text = "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")
        v = json.loads(text)
        # guard: gold must be from the candidate ids
        cand_ids = {c["id"] for c in o["candidates"]}
        v["gold_essential"] = [g for g in v.get("gold_essential", []) if g in cand_ids]
        v["gold_helpful"] = [g for g in v.get("gold_helpful", []) if g in cand_ids]
        v.update(cand_id=o["cand_id"], source=o["source"], cutoff=o["cutoff"],
                 cov_aged=o.get("cov_aged"), n_candidates=len(o["candidates"]))
        return v
    except Exception as e:
        return {"cand_id": o["cand_id"], "source": o["source"], "error": str(e)[:200]}

inp = json.load(open(os.path.join(OUT, "teacher_input.json")))[:LIMIT]
print(f"judging {len(inp)} cues with {MODEL} ({WORKERS} workers)…")

verdicts = []
with cf.ThreadPoolExecutor(max_workers=WORKERS) as ex:
    for v in ex.map(judge, inp):
        verdicts.append(v)
        tag = "ERR" if v.get("error") else ("EW " if v.get("endo_worthy") else "—  ")
        g = "+".join(v.get("gold_essential", []) or []) or "·"
        print(f"  [{tag}] {v['cand_id']:18s} {v.get('query_type','?'):12s} "
              f"conf={v.get('confidence','?'):6s} gold={g}  {v.get('error','')}")

outfile = "teacher_verdicts.json" if LIMIT >= 10**9 else f"teacher_verdicts_smoke{len(inp)}.json"
json.dump(verdicts, open(os.path.join(OUT, outfile), "w"), indent=1)
errs = sum(1 for v in verdicts if v.get("error"))
ew = sum(1 for v in verdicts if v.get("endo_worthy"))
print(f"\nwrote {outfile} — {len(verdicts)} verdicts, {ew} endo_worthy, {errs} errors")
