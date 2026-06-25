#!/usr/bin/env python3
"""REASONING ORACLE — Stage 1 of validating "recall is prediction".

The oracle-gap probe showed recall-with-foresight is a ~90%-different pull, but a cosine
target couldn't tell whether it's BETTER. Tom: "we need reasoning." So: let OPUS judge, with
hindsight (it sees Anchor's actual next response), whether the recalled memories would have
HELPED — and what was needed that wasn't surfaced. Then Opus summarizes the pattern so Anchor
can judge the judge.

Per sampled (user-msg N, assistant-response N) pair from OLD sessions (frozen IsolatedBrain):
  recall top-10 against the user message (what the system surfaces today)
  Opus sees: Tom's message + the 10 recalled memories + Anchor's ACTUAL response
  Opus judges: per-node helped/noise, served/partial/failed, what was MISSING, would foresight help
Then one Opus summary over all judgments. Writes reasoning_oracle_result.json.

Daemon-safe (IsolatedBrain). Usage: ./dev python3 eval/oracle_audit/reasoning_oracle.py
"""
import sys, json, re
ROOT = '/Users/tpac/brain'
sys.path.insert(0, ROOT)
from tests.isolated_brain import IsolatedBrain  # noqa: E402

N_TURNS = 12
TOPK = 10
MODEL = 'claude-opus-4-8'
OUT = f'{ROOT}/eval/oracle_audit/reasoning_oracle_result.json'


def content_of(meta, summary):
    if isinstance(meta, str):
        try:
            meta = json.loads(meta)
        except Exception:
            return summary or ''
    if isinstance(meta, dict):
        return meta.get('content') or summary or ''
    return summary or ''


def parse_json(text):
    if not text:
        return None
    i = text.find('{')
    if i < 0:
        return None
    try:
        return json.JSONDecoder().raw_decode(text[i:])[0]
    except Exception:
        return None


with IsolatedBrain() as env:
    b = env.brain
    client = b._ensure_anthropic_client()
    lc = b.logs_conn

    rows = lc.execute(
        "SELECT session_id, ref_type, summary, metadata, created_at FROM trace_events "
        "WHERE scale='s0' AND ref_type IN ('user_message','assistant_message') "
        "ORDER BY session_id, created_at"
    ).fetchall()
    pairs = []
    for i in range(len(rows) - 1):
        s0, rt0, sm0, m0, _ = rows[i]
        s1, rt1, sm1, m1, _ = rows[i + 1]
        if s0 == s1 and rt0 == 'user_message' and rt1 == 'assistant_message':
            u, a = content_of(m0, sm0), content_of(m1, sm1)
            if len(u) > 12 and len(a) > 60:
                pairs.append((u, a))
    cut = int(len(pairs) * 0.85)              # drop most-recent era (contamination)
    pool = pairs[:cut]
    stride = max(1, len(pool) // N_TURNS)
    sampled = pool[::stride][:N_TURNS]

    SYS = ("You are an expert judge of a memory-recall system for an AI partner named Anchor, "
           "working with operator Tom. You have HINDSIGHT: you see what Anchor actually said next. "
           "Judge whether the recalled memories would have HELPED Anchor's actual response, and "
           "what knowledge was MISSING. Be concrete and skeptical — a memory only 'helped' if it "
           "plausibly shaped or improved the actual response, not merely shares a topic.")

    judgments = []
    for idx, (umsg, aresp) in enumerate(sampled, 1):
        out = b.recall(query=umsg, limit=TOPK)
        res = out.get('results', []) if isinstance(out, dict) else (out or [])
        mem_lines = []
        for n, r in enumerate(res[:TOPK], 1):
            t = (r.get('title') or '')[:90]
            c = (r.get('content') or '')[:240]
            mem_lines.append("[%d] %s\n    %s" % (n, t, c))
        mems = "\n".join(mem_lines) if mem_lines else "(recall returned nothing)"

        user = (
            "TOM SAID (turn N):\n%s\n\n"
            "RECALL SURFACED these %d memories to help Anchor respond:\n%s\n\n"
            "ANCHOR ACTUALLY RESPONDED:\n%s\n\n"
            "Respond ONLY with JSON:\n"
            "{\n"
            '  "per_node": [{"n": 1, "verdict": "helped|neutral|noise", "why": "<8 words>"}, ...],\n'
            '  "served_verdict": "served|partial|failed",\n'
            '  "what_was_missing": "<knowledge that would have helped but was not recalled — describe the kind/topic even if you cannot name it; or: nothing, recall was adequate>",\n'
            '  "key_issue": "<single most important thing recall got wrong or missed this turn>",\n'
            '  "foresight_would_change": "yes|no",\n'
            '  "foresight_why": "<would seeing Anchor\'s actual response in advance have changed which memories to surface? one line>"\n'
            "}"
        ) % (umsg[:1500], len(mem_lines), mems, aresp[:1800])

        try:
            resp = client.messages.create(
                model=MODEL, max_tokens=1600, system=SYS,
                messages=[{"role": "user", "content": user}])
            txt = resp.content[0].text
            j = parse_json(txt)
        except Exception as e:
            j = {"error": str(e)[:200]}
        rec = {"turn": idx, "tom_said": umsg[:160], "anchor_said": aresp[:160],
               "n_recalled": len(mem_lines), "judgment": j}
        judgments.append(rec)
        v = (j or {}).get('served_verdict', j.get('error', '?') if j else '?')
        print("turn %2d: served=%-8s missing=%s" % (idx, v, str((j or {}).get('what_was_missing', ''))[:80]))

    # meta-summary: Opus reads all judgments and summarizes the pattern
    summary = None
    try:
        compact = [{"turn": r["turn"], "judgment": r["judgment"]} for r in judgments]
        sresp = client.messages.create(
            model=MODEL, max_tokens=1800,
            system=("You judged %d turns of a memory-recall system with hindsight. Summarize the "
                    "PATTERN for the engineer who will read this. Be concrete and honest — another "
                    "reviewer (Anchor) will judge your summary." % len(judgments)),
            messages=[{"role": "user", "content":
                "Your %d per-turn judgments (JSON):\n%s\n\n"
                "Summarize in JSON:\n"
                "{\n"
                '  "served_counts": {"served": N, "partial": N, "failed": N},\n'
                '  "dominant_failure_mode": "<is the bottleneck: recall MISSING the right node / RANKING noise above it / the knowledge NOT EXISTING / recall is actually fine? with evidence>",\n'
                '  "foresight_supported": "<did seeing the next move consistently point to DIFFERENT/better memories than the present query? i.e. is recall-as-prediction supported by these judgments? yes/partial/no + why>",\n'
                '  "highest_leverage_fix": "<the single most valuable change>",\n'
                '  "honest_caveats": "<what this sample can and cannot tell us>"\n'
                "}" % (len(judgments), json.dumps(compact, indent=1)[:14000])}])
        summary = parse_json(sresp.content[0].text) or {"raw": sresp.content[0].text[:2000]}
    except Exception as e:
        summary = {"error": str(e)[:200]}

    json.dump({"model": MODEL, "n_turns": len(judgments),
               "per_turn": judgments, "summary": summary}, open(OUT, 'w'), indent=2)
    print("\n=== OPUS SUMMARY ===")
    print(json.dumps(summary, indent=2)[:3000])
    print("\nwrote %s" % OUT)
