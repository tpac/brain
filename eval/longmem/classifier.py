"""Failure classifier — diagnose why a wrong answer failed, per-item.

For each failed item, determine which layer dropped the ball:

  ENCODE_MISS  — the gold fact was never encoded (no node carries it)
  RECALL_MISS  — encoded, but didn't land in the context delivered to the answerer
  SURFACE_MISS — candidates existed, but surfacer selected none (context empty)
  ANSWER_MISS  — selected and in context, but answerer still failed

Classification is trace-driven + a direct brain scan for the gold fact.
The brain scan is the ground-truth for "is the fact in the brain?" — more
reliable than semantic recall for that question. Semantic recall stays as
a diagnostic in evidence (how well the brain RETRIEVES the fact).

Claude is used only for the actionable reason and the PRESENT/MISSING
context sufficiency judgment — never for bucket selection itself.

Runs inline per-item so the brain and traces are live. Results get embedded
in each item dict — no separate storage layer.
"""
import json
import os
import re
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


SUFFICIENCY_PROMPT = """Given a user question, the gold answer, and the context passed to the answerer, judge: was the key fact needed to produce the gold answer PRESENT in the context?

Question: {question}
Gold answer: {gold}
Context delivered to answerer:
{context}

Reply with a single token: PRESENT (context contained the specific fact needed) or MISSING (context lacked the specific fact needed — e.g. mentioned the topic in general but not the specific value/name/quantity). No explanation."""


# Function words and very common tokens — too generic to be recall signals.
_STOPWORDS = frozenset([
    "the", "and", "for", "with", "from", "that", "this", "these", "those",
    "have", "has", "had", "will", "would", "could", "should", "might",
    "about", "into", "than", "when", "where", "what", "which", "while",
    "your", "yours", "their", "there", "them", "they", "been", "being",
    "some", "many", "much", "very", "just", "also", "only", "even",
    "what's", "don't", "didn't", "isn't", "wasn't", "won't",
    "after", "before", "because", "since", "though",
    "said", "says", "tell", "told", "going", "went",
])


def _extract_key_terms(gold: str, limit: int = 10) -> List[str]:
    """Extract recall-signal terms from a gold answer string.

    Keeps:
    - Digit sequences (counts, IDs, amounts)
    - Significant words (≥4 chars, not in stopwords)
    - Two/three-letter uppercase tokens (AM/PM, USD, EU, etc.)

    Lowercased, deduplicated, order-preserved, capped at `limit`.
    """
    if not gold:
        return []

    terms: List[str] = []

    for m in re.finditer(r"\b\d+(?:[.,:]\d+)*\b", gold):
        terms.append(m.group(0))

    for w in re.findall(r"\b[A-Za-z]{2,3}\b", gold):
        if w.isupper() and len(w) >= 2:
            terms.append(w.lower())

    for w in re.findall(r"\b[A-Za-z][A-Za-z'-]{3,}\b", gold):
        wl = w.lower()
        if wl not in _STOPWORDS:
            terms.append(wl)

    seen = set()
    out = []
    for t in terms:
        if t in seen:
            continue
        seen.add(t)
        out.append(t)
        if len(out) >= limit:
            break
    return out


def _scan_brain_for_gold(brain, gold: str) -> Dict[str, Any]:
    """Direct keyword scan of brain nodes for gold-answer terms.

    Ground-truth for "is the fact in any node of this brain?" — distinct
    from semantic recall ranking. Scans:
    - nodes.title, nodes.content (archived nodes excluded)
    - node_metadata_kv.value for high-signal keys (situation, reasoning,
      their_raw_quote, my_raw_quote, event_description, value, entity)

    Match rule: ALL extracted terms must appear within a single node
    (AND across terms, OR across fields). Prevents false positives when
    only one common term from the answer happens to appear in a random
    node. For single-term golds the AND reduces to OR naturally.

    If the gold string as a whole (lowercased, trimmed) is distinctive
    (>3 chars), also run a phrase-match pass — catches short answers like
    "6 PM" that degrade under term extraction.

    Returns:
        {
          found: bool,
          matches: [{node_id, title_snippet, match_source, snippet}],
          terms_used: [str, ...],
          phrase_used: str | None
        }

    Never raises — on DB error returns {found: False, ...} and logs.
    """
    gold_str = (gold or "").strip()
    terms = _extract_key_terms(gold_str)
    phrase = gold_str.lower() if len(gold_str) > 3 else None

    result: Dict[str, Any] = {
        "found": False,
        "matches": [],
        "terms_used": terms,
        "phrase_used": phrase,
    }

    if not terms and not phrase:
        return result

    try:
        conn = brain.conn
    except Exception:
        return result

    matches: List[Dict[str, Any]] = []

    # Pass 1: phrase match on title + content
    if phrase:
        try:
            rows = conn.execute(
                "SELECT id, title, substr(content, 1, 200) "
                "FROM nodes "
                "WHERE archived = 0 "
                "  AND (LOWER(title) LIKE ? OR LOWER(content) LIKE ?) "
                "LIMIT 10",
                (f"%{phrase}%", f"%{phrase}%"),
            ).fetchall()
            for nid, title, snippet in rows:
                matches.append({
                    "node_id": (nid or "")[:8],
                    "title_snippet": (title or "")[:60],
                    "match_source": "phrase:title_or_content",
                    "snippet": (snippet or "")[:160],
                })
        except Exception as e:
            result["db_error"] = f"phrase scan failed: {e}"

    # Pass 2: AND-of-terms match on title + content
    if terms and not matches:
        try:
            conditions = " AND ".join(
                "(LOWER(title) LIKE ? OR LOWER(content) LIKE ?)" for _ in terms
            )
            params = [f"%{t}%" for t in terms for _ in range(2)]
            rows = conn.execute(
                f"SELECT id, title, substr(content, 1, 200) "
                f"FROM nodes "
                f"WHERE archived = 0 AND {conditions} "
                f"LIMIT 10",
                params,
            ).fetchall()
            for nid, title, snippet in rows:
                matches.append({
                    "node_id": (nid or "")[:8],
                    "title_snippet": (title or "")[:60],
                    "match_source": "terms:title_or_content",
                    "snippet": (snippet or "")[:160],
                })
        except Exception as e:
            result["db_error"] = f"terms scan failed: {e}"

    # Pass 3: metadata_kv scan for high-signal keys, still AND across terms
    if (terms or phrase) and not matches:
        try:
            kv_keys = (
                "situation", "reasoning", "their_raw_quote", "my_raw_quote",
                "event_description", "value", "entity", "handle",
            )
            key_placeholders = ",".join("?" * len(kv_keys))
            if phrase:
                rows = conn.execute(
                    f"SELECT DISTINCT kv.node_id, kv.key, substr(kv.value, 1, 200), n.title "
                    f"FROM node_metadata_kv kv "
                    f"JOIN nodes n ON n.id = kv.node_id "
                    f"WHERE n.archived = 0 "
                    f"  AND kv.key IN ({key_placeholders}) "
                    f"  AND LOWER(kv.value) LIKE ? "
                    f"LIMIT 10",
                    (*kv_keys, f"%{phrase}%"),
                ).fetchall()
                for nid, key, snippet, title in rows:
                    matches.append({
                        "node_id": (nid or "")[:8],
                        "title_snippet": (title or "")[:60],
                        "match_source": f"phrase:meta.{key}",
                        "snippet": (snippet or "")[:160],
                    })
            if not matches and terms:
                conditions = " AND ".join("LOWER(kv.value) LIKE ?" for _ in terms)
                rows = conn.execute(
                    f"SELECT DISTINCT kv.node_id, kv.key, substr(kv.value, 1, 200), n.title "
                    f"FROM node_metadata_kv kv "
                    f"JOIN nodes n ON n.id = kv.node_id "
                    f"WHERE n.archived = 0 "
                    f"  AND kv.key IN ({key_placeholders}) "
                    f"  AND {conditions} "
                    f"LIMIT 10",
                    (*kv_keys, *(f"%{t}%" for t in terms)),
                ).fetchall()
                for nid, key, snippet, title in rows:
                    matches.append({
                        "node_id": (nid or "")[:8],
                        "title_snippet": (title or "")[:60],
                        "match_source": f"terms:meta.{key}",
                        "snippet": (snippet or "")[:160],
                    })
        except Exception as e:
            result["db_error"] = f"metadata scan failed: {e}"

    result["matches"] = matches[:5]
    result["found"] = bool(matches)
    return result


def _context_has_gold(question: str, gold: str, context: str) -> bool:
    """Judge whether the context delivered to the answerer actually contained
    the specific fact needed for the gold answer. Distinguishes RECALL_MISS
    (context missed the specific fact) from ANSWER_MISS (context had the
    fact, answerer didn't use it)."""
    import anthropic
    try:
        client = anthropic.Anthropic()
        resp = client.messages.create(
            model=REASON_MODEL,
            max_tokens=10,
            messages=[{"role": "user", "content": SUFFICIENCY_PROMPT.format(
                question=question, gold=gold, context=context[:6000],
            )}],
        )
        text = "".join(b.text for b in resp.content if hasattr(b, "text")).strip().upper()
        return text.startswith("PRESENT")
    except Exception:
        return True  # default to ANSWER_MISS on error — more conservative


def _recall_relevant_nodes(brain, gold: str, top_n: int = 15) -> List[Dict[str, Any]]:
    """Use the gold answer as a recall query against the brain.

    Diagnostic only (NOT ground truth anymore — scan is ground truth).
    Returns the top-N node summaries ranked by the brain's own similarity.
    Useful in evidence to show how the retrieval ranks vs how the scan matches.
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
    tool_trace: List[Any] = []
    surface_variant = ""

    for chain_id, etype, ref_type, summary, meta_json, _ in row:
        try:
            meta = json.loads(meta_json) if meta_json else {}
        except Exception:
            meta = {}
        if etype == "O" and ref_type == "recall":
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
        elif etype == "K" and ref_type == "surface_selected":
            # Agentic surface (v5) writes its tool-use trace into this K event;
            # v4 surface writes an empty list. surface_variant identifies which.
            tool_trace = meta.get("tool_trace", []) or []
            surface_variant = meta.get("surface_variant", "") or ""
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
        "tool_trace": tool_trace,
        "surface_variant": surface_variant,
    }


def _bucket(scan: Dict[str, Any], trace: Optional[Dict],
            has_context: bool, abstained: bool,
            question: str = "", gold: str = "") -> str:
    """Pick the failure bucket from scan + trace state.

    Buckets:
      ENCODE_MISS  — gold fact not in any node (scan.found == False)
      RECALL_MISS  — fact in brain but didn't land in context
                     (covers: 0 candidates, wrong candidates, or diluted context)
      SURFACE_MISS — candidates existed, but surfacer returned nothing
                     (context empty despite candidates — narrow, distinct)
      ANSWER_MISS  — context had the fact, answerer still failed

    Ground truth flow:
      1. scan.found is authoritative for "is the fact encoded somewhere"
      2. If scan.found == False  → ENCODE_MISS, always (regardless of trace)
      3. If scan.found == True:
         - If no trace or 0 candidates → RECALL_MISS (ranking didn't surface it)
         - If candidates > 0 but selected empty / ctx empty → SURFACE_MISS
         - If context delivered → ask Haiku if gold fact is in context:
             PRESENT  → ANSWER_MISS
             MISSING  → RECALL_MISS (wrong nodes selected, fact diluted)
    """
    if not scan.get("found"):
        return "ENCODE_MISS"

    if not trace:
        return "RECALL_MISS"

    n_cand = len(trace["candidates"])
    n_sel = len(trace["selected"])
    ctx_chars = len(trace["context"])

    if n_cand == 0:
        return "RECALL_MISS"

    if n_sel == 0 or ctx_chars == 0:
        return "SURFACE_MISS"

    gold_str = gold if isinstance(gold, str) else json.dumps(gold)
    if _context_has_gold(question, gold_str, trace["context"]):
        return "ANSWER_MISS"
    return "RECALL_MISS"


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
    """Per-item classification. Called inline after query, before cleanup.

    Returns:
        {failure_bucket, failure_reason, failure_evidence: {...}}
    """
    gold_str = gold if isinstance(gold, str) else json.dumps(gold)

    scan = _scan_brain_for_gold(brain, gold_str)
    relevant = _recall_relevant_nodes(brain, gold_str, top_n=15)
    trace = _read_s1r_trace(brain, query_session_id)

    bucket = _bucket(scan, trace, has_context, abstained,
                     question=question, gold=gold_str)

    evidence: Dict[str, Any] = {
        "gold_in_brain": {
            "found": scan.get("found", False),
            "terms_used": scan.get("terms_used", []),
            "phrase_used": scan.get("phrase_used"),
            "matches": scan.get("matches", []),
        },
        "relevant_to_gold": [  # semantic diagnostic — how does recall rank gold?
            {"id": n["id"], "title": n["title"][:80], "score": round(n["score"], 2)}
            for n in relevant[:5]
        ],
        "recall_candidates_count": len(trace["candidates"]) if trace else 0,
        "selected_ids": trace["selected"] if trace else [],
        "context_chars": len(trace["context"]) if trace else 0,
        "query_fired": bool(trace),
    }
    if "db_error" in scan:
        evidence["gold_in_brain"]["db_error"] = scan["db_error"]

    reason = _reason(question, gold_str, hypothesis, bucket, evidence)

    return {
        "failure_bucket": bucket,
        "failure_reason": reason,
        "failure_evidence": evidence,
    }
