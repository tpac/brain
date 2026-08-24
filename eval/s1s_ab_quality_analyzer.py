"""Quality analyzer for S1S A/B smoke runs.

Reads the preserved brains under eval/reports/s1s_ab_smoke/{run}/brains/
and computes quality dimensions per job, then aggregates by (transcript, arm).

Quality dims (what Tom asked for — multi-dimensional, not just volume):

1. Specificity retention — of proper nouns + numeric values in the source
   transcript, how many survive into the encoded nodes' content?
2. Avg content chars per node — node focus proxy
3. Two-register presence — ratio of principle-type nodes with an outgoing
   `grounds` edge (Tom's v13 core claim)
4. Operator voice — % of nodes with populated `their_raw_quote` metadata
5. Edge description quality — avg chars of edge `description` (non-empty)
6. Generic edge regression — count of relations == 'related'/'related_to'
7. Edge type diversity — distinct relation count + Shannon entropy
8. Temporal composition — count of type=time_anchor nodes + event_time
   metadata + Allen-relation edges
9. Brain-presence — for longmem items, does ANY node contain the gold
   answer phrase? (substring check on content+situation+reasoning)

Output:
- Per-job quality table
- Per-(transcript, arm) aggregates
- B vs A delta table per quality dim
- Warnings for anything suspicious
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sqlite3
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


# ─── Source-text specificity extractors ──────────────────────────────────

_PROPER_NOUN_RE = re.compile(
    r"\b([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+){0,3})\b")
_NUMBER_RE = re.compile(r"\b(\d+(?:[.,]\d+)?)\b")
# Stop-words for proper-noun extraction (sentence-initial false positives).
_PROPER_STOP = {
    "I", "The", "A", "An", "This", "That", "These", "Those", "Here",
    "There", "When", "Where", "Why", "How", "What", "Who", "My", "Your",
    "Our", "Their", "His", "Her", "Its", "Just", "So", "But", "And",
    "Or", "If", "Then", "Yes", "No", "Okay", "OK", "Thanks", "Thank",
    "Sure", "Now", "Some", "Any", "All", "Each", "Every", "Many",
    "Most", "Several", "Few", "Some", "Can", "Could", "Would", "Should",
    "Will", "Shall", "May", "Might", "Must", "Do", "Does", "Did",
    "Have", "Has", "Had", "Is", "Are", "Was", "Were", "Be", "Been",
    "Being", "Let", "Go", "See", "Look", "Good", "Great", "Nice",
}


def extract_proper_nouns(text: str) -> set:
    """Extract multi-word capitalized phrases, filtered for sentence-initial
    noise."""
    found = set()
    for m in _PROPER_NOUN_RE.finditer(text or ""):
        phrase = m.group(1)
        first = phrase.split()[0]
        if first in _PROPER_STOP and " " not in phrase:
            continue
        # Keep 2+ word proper nouns OR single-word "rare" capitalized
        if " " in phrase or (len(first) > 2 and first not in _PROPER_STOP):
            found.add(phrase)
    return found


def extract_numbers(text: str) -> set:
    """Extract numeric values (ints or floats)."""
    return {m.group(1) for m in _NUMBER_RE.finditer(text or "")}


def _collect_source_text(transcript: Dict[str, Any]) -> str:
    return "\n".join(m["content"] for m in transcript["messages"])


# ─── Brain reader ─────────────────────────────────────────────────────────

def _load_job_brain(brain_dir: Path) -> Tuple[sqlite3.Connection, sqlite3.Connection]:
    conn = sqlite3.connect(brain_dir / "brain.db")
    logs = sqlite3.connect(brain_dir / "brain_logs.db")
    return conn, logs


def _new_nodes_of_job(conn, session_start_iso: str) -> List[Dict[str, Any]]:
    """Nodes created AFTER the smoke session started (i.e. the encoder's writes).

    `situation` and `reasoning` live in node_metadata_kv, not on the nodes
    table — pull them up as first-class fields while fetching.
    """
    rows = conn.execute(
        "SELECT id, type, title, content, created_at, encoding_source, locked "
        "FROM nodes WHERE created_at >= ? AND archived = 0 "
        "AND encoding_source NOT IN ('anchor:seed','hook:compaction','hook:integrity') "
        "ORDER BY created_at ASC",
        (session_start_iso,)).fetchall()
    out = []
    for r in rows:
        md_rows = conn.execute(
            "SELECT key, value FROM node_metadata_kv WHERE node_id = ?",
            (r[0],)).fetchall()
        metadata = {k: v for k, v in md_rows}
        node = {
            "id": r[0], "type": r[1], "title": r[2] or "",
            "content": r[3] or "",
            "situation": metadata.get("situation", "") or "",
            "reasoning": metadata.get("reasoning", "") or "",
            "created_at": r[4],
            "encoding_source": r[5], "locked": r[6],
            "metadata": metadata,
        }
        out.append(node)
    return out


def _new_edges_of_job(conn, session_start_iso: str) -> List[Dict[str, Any]]:
    """Edges written by THIS run's scribe. Filters out seed-pack edges
    whose created_at timestamp collides with session_start at second
    precision by also excluding encoding_source starting with 'anchor:'
    or 'hook:' — those are not scribe writes."""
    rows = conn.execute(
        "SELECT e.edge_id, e.source_id, e.target_id, e.created_at "
        "FROM edges e WHERE e.created_at >= ? ORDER BY e.created_at ASC",
        (session_start_iso,)).fetchall()
    out = []
    for r in rows:
        rel_rows = conn.execute(
            "SELECT relation, description, weight, encoding_source "
            "FROM edge_relations WHERE edge_id = ?",
            (r[0],)).fetchall()
        for rr in rel_rows:
            enc_src = (rr[3] or "")
            # Exclude seed pack + hook-stamped edges (not scribe output)
            if enc_src.startswith("anchor:") or enc_src.startswith("hook:"):
                continue
            out.append({
                "edge_id": r[0], "src": r[1], "tgt": r[2],
                "relation": rr[0] or "", "description": rr[1] or "",
                "weight": rr[2], "encoding_source": enc_src,
                "created_at": r[3],
            })
    return out


# ─── Quality dimensions ──────────────────────────────────────────────────

def specificity_retention(source_text: str,
                          nodes: List[Dict[str, Any]]) -> Dict[str, float]:
    """% of source proper-nouns + numbers that appear in any new node's
    content/title/situation/reasoning.

    Separates tokens we care about (from source conversation) from tokens
    the encoder paraphrased away.
    """
    source_nouns = extract_proper_nouns(source_text)
    source_numbers = extract_numbers(source_text)
    if not nodes:
        return {
            "source_nouns": len(source_nouns),
            "source_numbers": len(source_numbers),
            "nouns_retained": 0,
            "numbers_retained": 0,
            "noun_retention_pct": 0.0,
            "number_retention_pct": 0.0,
        }
    encoded_blob = " ".join(
        f"{n['title']} {n['content']} {n['situation']} {n['reasoning']} "
        f"{' '.join(str(v) for v in n.get('metadata', {}).values())}"
        for n in nodes)
    nouns_retained = sum(1 for p in source_nouns if p in encoded_blob)
    nums_retained = sum(1 for p in source_numbers if p in encoded_blob)
    return {
        "source_nouns": len(source_nouns),
        "source_numbers": len(source_numbers),
        "nouns_retained": nouns_retained,
        "numbers_retained": nums_retained,
        "noun_retention_pct": 100.0 * nouns_retained / max(len(source_nouns), 1),
        "number_retention_pct": 100.0 * nums_retained / max(len(source_numbers), 1),
    }


def content_sizing(nodes: List[Dict[str, Any]]) -> Dict[str, float]:
    if not nodes:
        return {
            "avg_content_chars": 0, "avg_situation_chars": 0,
            "avg_title_chars": 0, "avg_reasoning_chars": 0,
            "nodes_with_situation_pct": 0.0,
            "nodes_with_reasoning_pct": 0.0,
        }
    contents = [len(n["content"]) for n in nodes]
    situations = [len(n["situation"]) for n in nodes]
    titles = [len(n["title"]) for n in nodes]
    reasonings = [len(n["reasoning"]) for n in nodes]
    with_sit = sum(1 for n in nodes if n["situation"].strip())
    with_reason = sum(1 for n in nodes if n["reasoning"].strip())
    return {
        "avg_content_chars": round(mean(contents)),
        "avg_situation_chars": round(mean(situations)),
        "avg_title_chars": round(mean(titles)),
        "avg_reasoning_chars": round(mean(reasonings)),
        "nodes_with_situation_pct": 100.0 * with_sit / len(nodes),
        "nodes_with_reasoning_pct": 100.0 * with_reason / len(nodes),
    }


def two_register_presence(nodes: List[Dict[str, Any]],
                          edges: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Tom's v13 core claim: principle nodes should have grounds edges to
    concrete facts. Count the ratio of principle-type new nodes with
    outgoing 'grounds' edges.
    """
    node_ids = {n["id"] for n in nodes}
    # Types that typically hold a transferable principle.
    abstract_types = {"principle", "lesson", "mechanism", "pattern",
                      "rule", "architecture", "insight"}
    grounds_edges = [e for e in edges if e["relation"] == "grounds"]
    grounds_src = {e["src"] for e in grounds_edges}
    principle_nodes = [n for n in nodes if n["type"] in abstract_types]
    p_with_grounds = sum(1 for n in principle_nodes if n["id"] in grounds_src)
    return {
        "grounds_edges_count": len(grounds_edges),
        "principle_nodes_count": len(principle_nodes),
        "principles_with_grounds": p_with_grounds,
        "two_register_pct": (100.0 * p_with_grounds / max(len(principle_nodes), 1)
                             if principle_nodes else 0.0),
    }


def operator_voice(nodes: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Fraction of nodes with populated their_raw_quote metadata."""
    if not nodes:
        return {"their_raw_quote_pct": 0.0, "nodes_with_quote": 0}
    with_quote = sum(
        1 for n in nodes
        if (n.get("metadata") or {}).get("their_raw_quote", "").strip())
    return {
        "their_raw_quote_pct": 100.0 * with_quote / len(nodes),
        "nodes_with_quote": with_quote,
    }


def edge_quality(edges: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not edges:
        return {
            "total_edges": 0, "distinct_relations": 0,
            "avg_description_chars": 0, "descriptions_populated_pct": 0.0,
            "generic_related_count": 0,
            "relation_entropy": 0.0,
            "top_relations": [],
        }
    desc_lens = [len(e["description"]) for e in edges]
    desc_pop = sum(1 for e in edges if e["description"].strip())
    rel_counter = Counter(e["relation"] for e in edges)
    generic = rel_counter.get("related", 0) + rel_counter.get("related_to", 0)
    # Shannon entropy across relation distribution (bits)
    total = sum(rel_counter.values())
    probs = [c / total for c in rel_counter.values() if c > 0]
    entropy = -sum(p * math.log2(p) for p in probs)
    return {
        "total_edges": len(edges),
        "distinct_relations": len(rel_counter),
        "avg_description_chars": round(mean(desc_lens)),
        "descriptions_populated_pct": 100.0 * desc_pop / len(edges),
        "generic_related_count": generic,
        "relation_entropy": round(entropy, 2),
        "top_relations": rel_counter.most_common(5),
    }


def temporal_composition(nodes: List[Dict[str, Any]],
                         edges: List[Dict[str, Any]]) -> Dict[str, Any]:
    time_anchors = [n for n in nodes if n["type"] == "time_anchor"]
    events = [n for n in nodes if n["type"] == "event"]
    with_event_time = sum(
        1 for n in nodes if (n.get("metadata") or {}).get("event_time"))
    allen = {"before", "after", "meets", "met_by", "during",
             "overlaps", "contains", "simultaneous_with",
             "anchored_to", "supersedes"}
    allen_edges = [e for e in edges if e["relation"] in allen]
    return {
        "time_anchor_nodes": len(time_anchors),
        "event_nodes": len(events),
        "nodes_with_event_time": with_event_time,
        "temporal_edges": len(allen_edges),
        "allen_relations_used": Counter(e["relation"] for e in allen_edges).most_common(),
    }


def brain_presence(nodes: List[Dict[str, Any]],
                   gold_answer: Optional[str],
                   question: Optional[str] = None) -> Dict[str, Any]:
    """Brain-presence check — does the answer live in ANY node, and where?

    Three levels of strictness:
    1. DIRECT match — gold substring appears verbatim (case-insensitive)
    2. TOKEN match — every non-stopword content-token of the gold appears
       in ONE node's text (paraphrase-tolerant)
    3. TOKEN split — tokens appear but SPLIT across multiple nodes

    Returns per-match detail: node_id, field (title/content/situation/
    reasoning/metadata_key), ±60 chars of context.
    """
    if not gold_answer:
        return {"gold_present": None, "match_node_ids": [], "matches": []}
    gold = gold_answer.strip()
    needle = gold.lower()
    if not needle:
        return {"gold_present": None, "match_node_ids": [], "matches": []}

    # Gold tokens (content-bearing)
    gold_tokens = _content_tokens(gold)

    direct_matches = []
    token_matches = []
    token_fragments: Dict[str, set] = {}  # node_id -> tokens found in that node
    for n in nodes:
        fields = {
            "title": n["title"],
            "content": n["content"],
            "situation": n["situation"],
            "reasoning": n["reasoning"],
        }
        for k, v in (n.get("metadata") or {}).items():
            # Skip internal system fields
            if k.startswith("_sys_"):
                continue
            fields[f"meta.{k}"] = str(v)

        node_tokens_found = set()
        for field_name, text in fields.items():
            low = (text or "").lower()
            # Level 1: direct substring
            idx = low.find(needle)
            if idx >= 0:
                ctx_start = max(0, idx - 60)
                ctx_end = min(len(text), idx + len(needle) + 60)
                direct_matches.append({
                    "node_id": n["id"],
                    "node_type": n["type"],
                    "node_title": n["title"][:60],
                    "field": field_name,
                    "match_kind": "direct",
                    "context": text[ctx_start:ctx_end],
                })
            # Level 2 prep: token matches per node
            for tok in gold_tokens:
                if tok.lower() in low:
                    node_tokens_found.add(tok.lower())

        if node_tokens_found:
            token_fragments[n["id"]] = node_tokens_found
            if node_tokens_found == set(t.lower() for t in gold_tokens):
                token_matches.append({
                    "node_id": n["id"],
                    "node_type": n["type"],
                    "node_title": n["title"][:60],
                    "match_kind": "token_all",
                })

    # Level 3: tokens appear split across multiple nodes
    all_tokens_found_any = set()
    for toks in token_fragments.values():
        all_tokens_found_any.update(toks)
    tokens_missing = {t.lower() for t in gold_tokens} - all_tokens_found_any

    return {
        "gold_present": bool(direct_matches) or bool(token_matches),
        "gold_direct": bool(direct_matches),
        "gold_token_match_single_node": bool(token_matches),
        "gold_token_match_split": (
            not direct_matches and not token_matches
            and not tokens_missing and len(token_fragments) > 1),
        "gold_tokens_missing": sorted(list(tokens_missing)),
        "gold_tokens_total": len(gold_tokens),
        "gold_tokens_found_any": len(all_tokens_found_any),
        "direct_matches": direct_matches[:5],
        "token_matches": token_matches[:5],
        "match_node_ids": (
            [m["node_id"] for m in direct_matches] or
            [m["node_id"] for m in token_matches]),
        "gold_substring": gold[:100],
        "question": (question or "")[:120],
    }


_STOP_WORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "of", "to", "in", "on", "at", "by", "for", "with", "from", "as",
    "that", "which", "who", "whom", "this", "these", "those",
    "and", "or", "but", "if", "then", "than", "so", "such",
    "it", "its", "i", "you", "we", "they", "he", "she",
    "my", "your", "our", "their", "his", "her",
    "some", "any", "all", "no", "not", "do", "does", "did",
    "have", "has", "had", "will", "would", "can", "could", "should",
    "shall", "may", "might", "must", "also", "one", "two",
}


def _content_tokens(text: str) -> List[str]:
    """Tokenize keeping content-bearing words (drop stopwords, single chars,
    pure punctuation). Used for paraphrase-tolerant matching."""
    return [
        w for w in re.findall(r"\w+", text or "")
        if len(w) > 1 and w.lower() not in _STOP_WORDS
    ]


# ─── Gold answers (for longmem) ──────────────────────────────────────────

def _load_longmem_gold(qid: str) -> Tuple[Optional[str], Optional[str]]:
    """Returns (answer, question) from longmem oracle. Both may be None."""
    try:
        data = json.loads((ROOT / "eval" / "longmem" / "data" /
                           "longmemeval_oracle.json").read_text(encoding="utf-8"))
    except Exception:
        return None, None
    for item in data:
        if item["question_id"] == qid:
            ans = item.get("answer")
            q = item.get("question")
            return (str(ans) if ans is not None else None,
                    str(q) if q is not None else None)
    return None, None


# ─── Job analysis ─────────────────────────────────────────────────────────

def analyze_job(brain_dir: Path, job_result: Dict[str, Any],
                transcript: Dict[str, Any]) -> Dict[str, Any]:
    conn, logs = _load_job_brain(brain_dir)
    try:
        # Anchor the "what's new" filter to the session start timestamp
        # recorded on the encoding_prompt trace. The runner suffixes '-{pid}',
        # so probe with '-%' — a bare '%' would let r1 also match r11+.
        session_id = f"smoke-{transcript['slug']}-{job_result['arm']}-r{job_result['run_idx']}"
        row = logs.execute(
            "SELECT MIN(created_at) FROM trace_events WHERE session_id LIKE ?",
            (session_id + "-%",)).fetchone()
        if not row or not row[0]:
            raise RuntimeError(
                "no trace_events for session %s-* in %s — cannot anchor the "
                "what's-new filter (a 1970 fallback would silently score "
                "every non-seed node in the brain as this job's output)"
                % (session_id, brain_dir))
        session_start = row[0]

        nodes = _new_nodes_of_job(conn, session_start)
        edges = _new_edges_of_job(conn, session_start)

        source_text = _collect_source_text(transcript)
        spec = specificity_retention(source_text, nodes)
        sizing = content_sizing(nodes)
        two_reg = two_register_presence(nodes, edges)
        voice = operator_voice(nodes)
        eq = edge_quality(edges)
        temp = temporal_composition(nodes, edges)

        gold, question = None, None
        if transcript.get("source") == "longmem" and transcript.get("question_id"):
            gold, question = _load_longmem_gold(transcript["question_id"])
        bp = brain_presence(nodes, gold, question=question)

        return {
            "job_id": job_result.get("job_id"),
            "slug": transcript["slug"],
            "arm": job_result.get("arm"),
            "run_idx": job_result.get("run_idx"),
            "node_count": len(nodes),
            "edge_count": len(edges),
            "specificity": spec,
            "sizing": sizing,
            "two_register": two_reg,
            "voice": voice,
            "edges": eq,
            "temporal": temp,
            "brain_presence": bp,
            "node_titles": [n["title"][:80] for n in nodes],
        }
    finally:
        conn.close()
        logs.close()


# ─── Aggregation ─────────────────────────────────────────────────────────

def aggregate(rows: List[Dict[str, Any]]) -> Dict[Tuple[str, str], Dict[str, Any]]:
    """Group by (transcript, arm), compute mean of numeric metrics."""
    groups: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        groups[(r["slug"], r["arm"])].append(r)

    agg = {}
    for key, items in groups.items():
        agg[key] = _group_mean(items)
    return agg


def _group_mean(items: List[Dict[str, Any]]) -> Dict[str, Any]:
    def getlist(path):
        out = []
        for i in items:
            v = i
            for p in path:
                v = v.get(p, {}) if isinstance(v, dict) else None
            if isinstance(v, (int, float)):
                out.append(v)
        return out

    def avg(path):
        xs = getlist(path)
        return round(mean(xs), 2) if xs else 0

    # Pick metrics that aggregate as means
    return {
        "n_jobs": len(items),
        "avg_nodes": avg(["node_count"]),
        "avg_edges": avg(["edge_count"]),
        "noun_retention_pct": avg(["specificity", "noun_retention_pct"]),
        "number_retention_pct": avg(["specificity", "number_retention_pct"]),
        "avg_content_chars": avg(["sizing", "avg_content_chars"]),
        "avg_situation_chars": avg(["sizing", "avg_situation_chars"]),
        "nodes_with_situation_pct": avg(["sizing", "nodes_with_situation_pct"]),
        "nodes_with_reasoning_pct": avg(["sizing", "nodes_with_reasoning_pct"]),
        "two_register_pct": avg(["two_register", "two_register_pct"]),
        "grounds_edges": avg(["two_register", "grounds_edges_count"]),
        "their_raw_quote_pct": avg(["voice", "their_raw_quote_pct"]),
        "distinct_relations": avg(["edges", "distinct_relations"]),
        "avg_edge_desc_chars": avg(["edges", "avg_description_chars"]),
        "edges_populated_pct": avg(["edges", "descriptions_populated_pct"]),
        "generic_related_count": avg(["edges", "generic_related_count"]),
        "relation_entropy": avg(["edges", "relation_entropy"]),
        "time_anchors": avg(["temporal", "time_anchor_nodes"]),
        "event_nodes": avg(["temporal", "event_nodes"]),
        "temporal_edges": avg(["temporal", "temporal_edges"]),
        "brain_presence_pct": 100.0 * sum(
            1 for i in items if i["brain_presence"].get("gold_present")
        ) / len(items) if any(
            i["brain_presence"].get("gold_present") is not None for i in items
        ) else None,
    }


# ─── Reporting ───────────────────────────────────────────────────────────

def report(rows: List[Dict[str, Any]], run_dir: Path) -> None:
    agg = aggregate(rows)
    slugs = sorted(set(r["slug"] for r in rows))

    print("=" * 110)
    print("S1S A/B QUALITY ANALYSIS")
    print("=" * 110)
    print()

    # Per-(transcript, arm) quality table
    dims = [
        ("avg_nodes", "Nodes"),
        ("avg_edges", "Edges"),
        ("noun_retention_pct", "Noun ret %"),
        ("number_retention_pct", "Num ret %"),
        ("avg_content_chars", "Content chars"),
        ("avg_situation_chars", "Situ chars"),
        ("nodes_with_situation_pct", "w/Situ %"),
        ("nodes_with_reasoning_pct", "w/Reason %"),
        ("two_register_pct", "Two-reg %"),
        ("grounds_edges", "Grounds"),
        ("their_raw_quote_pct", "Quote %"),
        ("distinct_relations", "Rel types"),
        ("avg_edge_desc_chars", "EdgeDesc"),
        ("relation_entropy", "Rel entropy"),
        ("time_anchors", "TimeAnchors"),
        ("temporal_edges", "TempEdges"),
        ("generic_related_count", "Generic"),
    ]

    for slug in slugs:
        print(f"### {slug}")
        a = agg.get((slug, "A"), {})
        b = agg.get((slug, "B"), {})
        n_a, n_b = a.get("n_jobs", 0), b.get("n_jobs", 0)
        if not n_a or not n_b:
            # An arm with zero completed jobs would render every dimension
            # 0.00 and show the other arm's magnitude as an improvement arrow.
            print(f"⚠ SKIPPED — arm with zero completed jobs (A: {n_a} jobs, "
                  f"B: {n_b} jobs); no comparison exists for this transcript")
            print()
            continue
        print(f"{'DIMENSION':<22}  {f'A (n={n_a})':>10}  {f'B (n={n_b})':>10}  {'Δ (B-A)':>10}")
        print("-" * 60)
        for key, label in dims:
            va = a.get(key, 0)
            vb = b.get(key, 0)
            delta = vb - va
            arrow = "↑" if delta > 0 else ("↓" if delta < 0 else " ")
            print(f"{label:<22}  {va:>10.2f}  {vb:>10.2f}  {delta:>+9.2f}{arrow}")

        # Brain-presence only for longmem items
        bp_a = a.get("brain_presence_pct")
        bp_b = b.get("brain_presence_pct")
        if bp_a is not None or bp_b is not None:
            print(f"{'Brain-presence %':<22}  {(bp_a or 0):>10.2f}  {(bp_b or 0):>10.2f}  "
                  f"{((bp_b or 0) - (bp_a or 0)):>+9.2f}")
        print()

        # Per-run brain-presence detail (only meaningful for longmem items)
        longmem_rows = [r for r in rows if r["slug"] == slug
                        and r["brain_presence"].get("gold_substring")]
        if longmem_rows:
            print(f"   BRAIN-PRESENCE DETAIL — '{longmem_rows[0]['brain_presence']['gold_substring']}'")
            print(f"   Q: {longmem_rows[0]['brain_presence'].get('question', '')}")
            for r in longmem_rows:
                bp = r["brain_presence"]
                status = ("DIRECT" if bp.get("gold_direct") else
                          "TOKEN-same-node" if bp.get("gold_token_match_single_node") else
                          "TOKEN-split" if bp.get("gold_token_match_split") else
                          f"MISSING (have {bp.get('gold_tokens_found_any',0)}/"
                          f"{bp.get('gold_tokens_total',0)} tokens)")
                print(f"    {r['arm']}:r{r['run_idx']}  {status}")
                for m in bp.get("direct_matches", [])[:2]:
                    print(f"        [{m['field']}] on {m['node_id'][:8]} ({m['node_type']}): "
                          f"...{m['context'][:150].strip()}...")
                for m in bp.get("token_matches", [])[:2]:
                    if not bp.get("direct_matches"):
                        print(f"        token-match on {m['node_id'][:8]} "
                              f"(\"{m['node_title']}\")")
                missing = bp.get("gold_tokens_missing", [])
                if missing and not bp.get("gold_direct"):
                    print(f"        missing tokens: {missing[:6]}")
            print()

    # Save full results JSON
    out_path = run_dir / "quality_analysis.json"
    serializable_rows = []
    for r in rows:
        sr = dict(r)
        # top_relations uses tuples — coerce
        if "edges" in sr and "top_relations" in sr["edges"]:
            sr["edges"]["top_relations"] = [list(x) for x in sr["edges"]["top_relations"]]
        if "temporal" in sr and "allen_relations_used" in sr["temporal"]:
            sr["temporal"]["allen_relations_used"] = [list(x) for x in sr["temporal"]["allen_relations_used"]]
        serializable_rows.append(sr)
    out_path.write_text(json.dumps({
        "per_job": serializable_rows,
        "aggregate": {f"{k[0]}:{k[1]}": v for k, v in agg.items()},
    }, indent=2, default=str), encoding="utf-8")
    print(f"Full analysis saved: {out_path}")


# ─── CLI ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_name", help="smoke run name (e.g. smoke_seed_2)")
    args = parser.parse_args()

    run_dir = ROOT / "eval" / "reports" / "s1s_ab_smoke" / args.run_name
    if not run_dir.is_dir():
        print(f"ERROR: run dir not found: {run_dir}")
        sys.exit(2)

    results_path = run_dir / "results.jsonl"
    if not results_path.exists():
        print(f"ERROR: no results.jsonl at {results_path}")
        sys.exit(2)

    # Re-import transcript definitions
    from eval.s1s_ab_wiring_check import load_all_transcripts
    transcripts = {t["slug"]: t for t in load_all_transcripts()}

    job_results = []
    for line in results_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            job_results.append(json.loads(line))

    brains_dir = run_dir / "brains"
    rows = []
    for j in job_results:
        slug = j.get("slug")
        if slug not in transcripts:
            print(f"[skip] unknown slug: {slug}")
            continue
        job_brain_dir = brains_dir / f"{slug}__{j['arm']}__r{j['run_idx']}"
        if not (job_brain_dir / "brain.db").exists():
            print(f"[skip] no brain.db for {j.get('job_id')}")
            continue
        try:
            row = analyze_job(job_brain_dir, j, transcripts[slug])
            rows.append(row)
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"[err] {j.get('job_id')}: {e}")

    report(rows, run_dir)


if __name__ == "__main__":
    main()
