"""Failure classifier — diagnose why a wrong answer failed, per-item.

For each failed item, determine which layer dropped the ball:

  ENCODE_MISS  — the gold fact was never encoded (no node covers it)
  RECALL_MISS  — encoded, but not in S1R's top-25 candidates
  SURFACE_MISS — in candidates, but surfacer didn't pick it
  ANSWER_MISS  — selected and in context, but answerer still failed

Classification is trace-driven. Claude is used only for the actionable reason,
not for the bucket (which is computable from O/K/Δ traces + a targeted recall
against the brain using the gold answer as query).

Runs inline per-item so the brain and traces are live. Results get embedded
in each item dict — no separate storage layer.
"""
import json
import os
import sys
from typing import Any, Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


REASON_MODEL = "claude-haiku-4-5-20251001"
REASON_MAX_TOKENS = 180

REASON_PROMPT = """You are analyzing why an AI memory system failed on one question.

Question: {question}
Gold answer: {gold}
System answer: {hypothesis}

Failure bucket (already determined): {bucket}
Evidence:
{evidence}

Given the bucket and evidence, write ONE actionable sentence describing WHAT specifically went wrong and WHERE to invest to fix it. No hedging, no restatement.
Start directly with the finding. 40 words max."""


def _recall_relevant_nodes(brain, gold: str, top_n: int = 15) -> List[Dict[str, Any]]:
    """Use the gold answer as a recall query against the brain.

    Returns the top-N node summaries ranked by the brain's own similarity.
    If the brain can't recall anything semantically close to the gold,
    the encoder likely never captured the fact.
    """
    try:
        results = brain.recall(gold, top_n=top_n) or []
    except Exception:
        return []
    out = []
    for r in results:
        if not isinstance(r, dict):
            continue
        out.append({
            "id": (r.get("id") or "")[:8],
            "title": r.get("title") or "",
            "score": float(r.get("score") or 0.0),
            "type": r.get("type") or "",
        })
    return out


def _read_s1r_trace(brain, query_session_id: str) -> Optional[Dict[str, Any]]:
    """Read the S1R trace for the query session.

    Returns unified view of candidates, selected, dropped, context.
    None if no trace exists (shouldn't happen, but defensive).
    """
    try:
        row = brain.logs_conn.execute(
            "SELECT chain_id, event_type, ref_type, summary, metadata, created_at "
            "FROM trace_events "
            "WHERE session_id = ? AND scale = 's1' "
            "ORDER BY created_at ASC",
            (query_session_id,)).fetchall()
    except Exception:
        return None

    if not row:
        return None

    candidates = []
    selected = []
    dropped = []
    context = ""
    query = ""

    for chain_id, etype, ref_type, summary, meta_json, _ in row:
        try:
            meta = json.loads(meta_json) if meta_json else {}
        except Exception:
            meta = {}
        if etype == "O" and ref_type == "recall":
            # candidates in metadata as ["id|title|score|type", ...]
            for item in meta.get("candidates", []) or []:
                parts = item.split("|", 3)
                if len(parts) >= 3:
                    candidates.append({
                        "id": parts[0],
                        "title": parts[1],
                        "score": float(parts[2]) if parts[2] else 0.0,
                        "type": parts[3] if len(parts) > 3 else "",
                    })
            query = meta.get("query", "")
        elif etype == "delta" and ref_type == "additionalContext":
            selected = meta.get("selected", []) or []
            dropped = meta.get("dropped", []) or []
            context = meta.get("content", "") or ""

    return {
        "query": query,
        "candidates": candidates,
        "selected": selected,
        "dropped": dropped,
        "context": context,
    }


def _bucket(relevant_nodes: List[Dict], trace: Optional[Dict],
            has_context: bool, abstained: bool) -> str:
    """Pick the failure bucket from trace state.

    The trace tells the cleanest story:
      - 0 candidates    → ENCODE_MISS (nothing matched the query at all)
      - candidates, 0 selected → SURFACE_MISS (surfacer rejected them)
      - selected + context     → ANSWER_MISS (answerer had it, didn't use it)

    `relevant_nodes` (from a gold-seeded recall) is a weaker signal —
    gold answers like "190" or "Memrise" don't embed well, so we use it
    only to distinguish ENCODE_MISS from RECALL_MISS in the 0-cand path.
    """
    if not trace:
        return "RECALL_MISS" if relevant_nodes else "ENCODE_MISS"

    n_cand = len(trace["candidates"])
    n_sel = len(trace["selected"])
    ctx_chars = len(trace["context"])

    if n_cand == 0:
        # No candidates matched the query. If a gold-seeded recall also
        # finds nothing, nothing's encoded. If it finds something, recall
        # scoring missed it for the real query.
        return "ENCODE_MISS" if not relevant_nodes else "RECALL_MISS"

    if n_sel == 0:
        return "SURFACE_MISS"

    if ctx_chars > 0:
        return "ANSWER_MISS"

    # Selected but context empty — rare. Treat as surface issue.
    return "SURFACE_MISS"


def _reason(question: str, gold: str, hypothesis: str, bucket: str,
            evidence: Dict[str, Any]) -> str:
    """One-sentence actionable reason from Haiku."""
    import anthropic

    evidence_str = json.dumps(evidence, indent=2)[:1800]
    try:
        client = anthropic.Anthropic()
        resp = client.messages.create(
            model=REASON_MODEL,
            max_tokens=REASON_MAX_TOKENS,
            messages=[{"role": "user", "content": REASON_PROMPT.format(
                question=question, gold=gold, hypothesis=hypothesis or "(empty)",
                bucket=bucket, evidence=evidence_str,
            )}],
        )
        text = "".join(b.text for b in resp.content if hasattr(b, "text")).strip()
        return text[:500]
    except Exception as e:
        return f"(reason call failed: {e})"


def classify_failure(brain, question: str, gold: str, hypothesis: str,
                     query_session_id: str, has_context: bool,
                     abstained: bool) -> Dict[str, Any]:
    """Per-item classification. Called inline after query, before reset.

    Returns:
        {bucket, reason, evidence: {relevant_nodes, candidates_count, selected, ...}}
    """
    gold_str = gold if isinstance(gold, str) else json.dumps(gold)
    relevant = _recall_relevant_nodes(brain, gold_str, top_n=15)
    trace = _read_s1r_trace(brain, query_session_id)

    bucket = _bucket(relevant, trace, has_context, abstained)

    evidence = {
        "relevant_to_gold": [  # top 5 most-relevant nodes from a gold-seeded recall
            {"id": n["id"], "title": n["title"][:80], "score": round(n["score"], 2)}
            for n in relevant[:5]
        ],
        "recall_candidates_count": len(trace["candidates"]) if trace else 0,
        "selected_ids": trace["selected"] if trace else [],
        "context_chars": len(trace["context"]) if trace else 0,
        "query_fired": bool(trace),
    }

    reason = _reason(question, gold_str, hypothesis, bucket, evidence)

    return {
        "failure_bucket": bucket,
        "failure_reason": reason,
        "failure_evidence": evidence,
    }
