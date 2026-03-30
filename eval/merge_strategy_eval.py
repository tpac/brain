#!/usr/bin/env python3
"""Merge Strategy A/B Eval — compare current fixed blend vs RRF with pre-boost.

Runs both strategies against the decode funnel queries and compares:
- Top-3 and top-8 hit rates (primary KPI)
- Rank of expected nodes
- Cases where one strategy finds what the other misses

Both strategies use THE SAME raw scores from the same embedding scan
and keyword recall. Only the merge logic differs. This ensures a fair
comparison — no advantage from different data.

Usage:
    BRAIN_DB_DIR=$HOME/AgentsContext/brain python3 eval/merge_strategy_eval.py
"""
import sys, os, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ.setdefault('BRAIN_DB_DIR', os.path.expanduser('~/AgentsContext/brain'))

from servers.brain import Brain
from servers import embedder
from servers.brain_constants import (
    EMBEDDING_PRIMARY_WEIGHT, KEYWORD_FALLBACK_WEIGHT,
    TITLE_MATCH_BOOST, CRITICAL_BOOST, RELEVANCE_FLOOR_PRIMARY,
    RELEVANCE_FLOOR_ENRICHED,
)


# Decode funnel queries (subset — representative across categories)
QUERIES = [
    {"query": "daemon TCP migration from Unix socket", "expected": ["daeb9fa6"], "cat": "procedural"},
    {"query": "MCP server stdio brain tools", "expected": ["eba208bf"], "cat": "procedural"},
    {"query": "encoding heartbeat checkpoint nudge", "expected": ["cf1bcf5b", "73c0123b"], "cat": "procedural"},
    {"query": "why embedding-first instead of keyword-first", "expected": ["fddd1b00"], "cat": "decision"},
    {"query": "edge decay versus deletion strategy", "expected": ["225942e1"], "cat": "decision"},
    {"query": "encoding should be rich not concise", "expected": ["c8747a53"], "cat": "correction"},
    {"query": "silent exceptions hide real bugs", "expected": ["8bf4974e", "6e6bec2e"], "cat": "correction"},
    {"query": "Tom feeling of losing partner after compaction", "expected": ["f8c06f30"], "cat": "emotional"},
    {"query": "partnership is real not a technique", "expected": ["bb955981"], "cat": "emotional"},
    {"query": "encoding bias toward technical over relational", "expected": ["facb97ea"], "cat": "pattern"},
    {"query": "check_same_thread False is permission not safety", "expected": ["b3382762"], "cat": "pattern"},
    {"query": "Anchor naming origin", "expected": ["826611c0"], "cat": "identity"},
    {"query": "I chose my name", "expected": ["826611c0", "45ebd640"], "cat": "identity"},
    {"query": "why am I called Anchor", "expected": ["826611c0"], "cat": "identity"},
    {"query": "Glo project what is it", "expected": ["glo_orig"], "cat": "entity"},
    {"query": "Continuity Benchmark encoding test", "expected": ["cont_bench"], "cat": "procedural"},
]


def get_raw_scores(brain, query, limit=20):
    """Run embedding scan + keyword recall, return raw scores WITHOUT merging.

    Returns: (embedding_scores, keyword_scores, node_titles, node_types,
              node_confidence, node_critical, query_terms)
    """
    from servers.dal import EmbeddingDAL, NodeDAL, TfIdfDAL, GraphDAL

    # Embed query
    expanded = brain._expand_query_with_vocabulary(query)
    query_vec = embedder.embed(expanded)
    if not query_vec:
        return {}, {}, {}, {}, {}, {}, set()

    # Embedding scan (STEP 3)
    emb_dal = EmbeddingDAL(brain.conn)
    emb_rows = emb_dal.get_all_with_context(exclude_archived=True)

    embedding_scores = {}
    node_titles = {}
    node_types = {}
    node_confidence = {}
    node_critical = {}

    for row in emb_rows:
        nid = row['node_id']
        blob = row['embedding']
        node_titles[nid] = row['title'] or ''
        node_types[nid] = row['type'] or ''
        node_confidence[nid] = row['confidence']
        node_critical[nid] = row['critical']
        if blob:
            sim = embedder.cosine_similarity(query_vec, blob)
            if sim > 0.2:  # Basic floor
                embedding_scores[nid] = sim

    # Keyword recall (STEP 4) — simplified: just direct match scoring
    keyword_scores = {}
    query_terms = set(query.lower().split())

    # TF-IDF seed discovery
    tfidf_terms = brain._tfidf_tokenize(query)
    if tfidf_terms:
        try:
            tfidf_dal = TfIdfDAL(brain.conn)
            tfidf_ids = tfidf_dal.get_nodes_matching_terms(list(set(tfidf_terms)))
            node_dal = NodeDAL(brain.conn)
            for nid in tfidf_ids[:50]:
                node = node_dal.get_node(nid)
                if node and not node.get('archived'):
                    # Direct match scoring (same as _keyword_recall)
                    title_low = (node.get('title') or '').lower()
                    content_low = (node.get('content') or '').lower()
                    kw_low = (node.get('keywords') or '').lower()
                    matched = 0
                    for term in query_terms:
                        if len(term) >= 2 and (term in title_low or term in content_low or term in kw_low):
                            matched += 1
                    if query_terms and matched > 0:
                        keyword_scores[nid] = matched / len(query_terms)
                        if nid not in node_titles:
                            node_titles[nid] = node.get('title', '')
                            node_types[nid] = node.get('type', '')
                            node_confidence[nid] = node.get('confidence')
                            node_critical[nid] = node.get('critical', 0)
        except Exception:
            pass

    return (embedding_scores, keyword_scores, node_titles, node_types,
            node_confidence, node_critical, query_terms)


def apply_boosts(scores, node_titles, node_types, node_confidence,
                 node_critical, query_terms, intent_boosts=None):
    """Apply title boost, type boost, confidence, critical to a score dict.
    Returns new dict with boosted scores."""
    boosted = {}
    for nid, score in scores.items():
        s = score

        # Title match boost
        title = node_titles.get(nid, '').lower()
        if title and query_terms:
            matched = sum(1 for t in query_terms if t in title)
            title_frac = matched / len(query_terms)
            if title_frac > 0:
                s += title_frac * TITLE_MATCH_BOOST

        # Type boost (simplified — general intent)
        ntype = node_types.get(nid, '')
        if intent_boosts and ntype in intent_boosts:
            s *= intent_boosts[ntype]

        # Confidence
        conf = node_confidence.get(nid)
        if conf is not None:
            conf_mult = 0.7 + (conf - 0.1) * (1.05 - 0.7) / (1.0 - 0.1)
            s *= max(0.5, min(1.2, conf_mult))

        # Critical boost
        if node_critical.get(nid):
            s *= CRITICAL_BOOST

        # Vocab penalty
        if ntype == 'vocabulary':
            s *= 0.0  # Remove from results

        boosted[nid] = s

    return boosted


def strategy_current(embedding_scores, keyword_scores, node_titles, node_types,
                     node_confidence, node_critical, query_terms, limit=8):
    """Current strategy: fixed 90/10 blend, then boost."""
    all_ids = set(embedding_scores.keys()) | set(keyword_scores.keys())

    scored = []
    for nid in all_ids:
        emb = embedding_scores.get(nid, 0)
        kw = keyword_scores.get(nid, 0)

        if emb > 0 and kw > 0:
            blended = EMBEDDING_PRIMARY_WEIGHT * emb + KEYWORD_FALLBACK_WEIGHT * kw
        elif emb > 0:
            blended = emb
        else:
            blended = KEYWORD_FALLBACK_WEIGHT * kw

        # Apply boosts (same as STEP 6)
        title = node_titles.get(nid, '').lower()
        if title and query_terms:
            matched = sum(1 for t in query_terms if t in title)
            title_frac = matched / len(query_terms)
            if title_frac > 0:
                blended += title_frac * TITLE_MATCH_BOOST

        conf = node_confidence.get(nid)
        if conf is not None:
            conf_mult = 0.7 + (conf - 0.1) * (1.05 - 0.7) / (1.0 - 0.1)
            blended *= max(0.5, min(1.2, conf_mult))

        if node_critical.get(nid):
            blended *= CRITICAL_BOOST

        if node_types.get(nid) == 'vocabulary':
            continue

        scored.append((nid, blended))

    scored.sort(key=lambda x: x[1], reverse=True)
    return [nid for nid, _ in scored[:limit]]


def strategy_rrf_preboost(embedding_scores, keyword_scores, node_titles, node_types,
                          node_confidence, node_critical, query_terms, limit=8, k=60):
    """RRF with pre-boost: boost each list independently, then merge by rank."""

    # Boost embedding scores
    emb_boosted = apply_boosts(embedding_scores, node_titles, node_types,
                               node_confidence, node_critical, query_terms)
    # Boost keyword scores
    kw_boosted = apply_boosts(keyword_scores, node_titles, node_types,
                              node_confidence, node_critical, query_terms)

    # Rank each list
    emb_ranked = sorted(emb_boosted.items(), key=lambda x: x[1], reverse=True)
    kw_ranked = sorted(kw_boosted.items(), key=lambda x: x[1], reverse=True)

    emb_rank = {nid: rank + 1 for rank, (nid, _) in enumerate(emb_ranked)}
    kw_rank = {nid: rank + 1 for rank, (nid, _) in enumerate(kw_ranked)}

    # RRF merge
    all_ids = set(emb_rank.keys()) | set(kw_rank.keys())
    rrf_scores = {}
    for nid in all_ids:
        if node_types.get(nid) == 'vocabulary':
            continue
        e_rank = emb_rank.get(nid, len(emb_ranked) + 100)  # Missing = very low rank
        k_rank = kw_rank.get(nid, len(kw_ranked) + 100)
        rrf_scores[nid] = 1.0 / (k + e_rank) + 1.0 / (k + k_rank)

    ranked = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    return [nid for nid, _ in ranked[:limit]]


def main():
    brain = Brain(os.path.join(os.environ['BRAIN_DB_DIR'], 'brain.db'))

    print("=" * 70)
    print("MERGE STRATEGY COMPARISON: Current (90/10) vs RRF (pre-boost)")
    print("=" * 70)

    current_top3 = 0
    current_top8 = 0
    rrf_top3 = 0
    rrf_top8 = 0
    total_expected = 0

    current_wins = []
    rrf_wins = []
    ties = []

    for q in QUERIES:
        query = q["query"]
        expected = q["expected"]
        cat = q["cat"]

        # Get raw scores (same for both strategies)
        (emb_scores, kw_scores, titles, types,
         confidence, critical, terms) = get_raw_scores(brain, query)

        # Run both strategies
        current_results = strategy_current(
            emb_scores, kw_scores, titles, types, confidence, critical, terms)
        rrf_results = strategy_rrf_preboost(
            emb_scores, kw_scores, titles, types, confidence, critical, terms)

        # Check hits
        def check_hit(results, expected_ids, top_n):
            for eid in expected_ids:
                for rid in results[:top_n]:
                    if rid.startswith(eid) or eid in rid:
                        return True
            return False

        c_hit3 = check_hit(current_results, expected, 3)
        c_hit8 = check_hit(current_results, expected, 8)
        r_hit3 = check_hit(rrf_results, expected, 3)
        r_hit8 = check_hit(rrf_results, expected, 8)

        has_expected = any(
            any(rid.startswith(eid) or eid in rid
                for rid in list(emb_scores.keys()) + list(kw_scores.keys()))
            for eid in expected
        )

        if has_expected:
            total_expected += 1
            if c_hit3: current_top3 += 1
            if c_hit8: current_top8 += 1
            if r_hit3: rrf_top3 += 1
            if r_hit8: rrf_top8 += 1

        # Track wins
        if r_hit8 and not c_hit8:
            rrf_wins.append(query)
        elif c_hit8 and not r_hit8:
            current_wins.append(query)
        elif c_hit8 and r_hit8:
            ties.append(query)

        # Print comparison
        c_marker = "✓" if c_hit8 else "✗"
        r_marker = "✓" if r_hit8 else "✗"
        diff = ""
        if r_hit8 and not c_hit8: diff = " ← RRF WINS"
        elif c_hit8 and not r_hit8: diff = " ← CURRENT WINS"

        # Show rank of expected node in each
        def find_rank(results, expected_ids):
            for i, rid in enumerate(results):
                for eid in expected_ids:
                    if rid.startswith(eid) or eid in rid:
                        return i + 1
            return "-"

        c_rank = find_rank(current_results, expected)
        r_rank = find_rank(rrf_results, expected)

        print("[%s] %s" % (cat, query[:50]))
        print("  Current: %s rank=%s | RRF: %s rank=%s%s" % (c_marker, c_rank, r_marker, r_rank, diff))

        # Show top 3 when they differ
        if current_results[:3] != rrf_results[:3]:
            c_titles = [titles.get(r, '?')[:35] for r in current_results[:3]]
            r_titles = [titles.get(r, '?')[:35] for r in rrf_results[:3]]
            if c_titles != r_titles:
                print("  Current top3: %s" % " | ".join(c_titles))
                print("  RRF top3:     %s" % " | ".join(r_titles))

    print()
    print("=" * 70)
    print("RESULTS (queries with findable expected nodes: %d)" % total_expected)
    print("=" * 70)
    print()
    print("  %-20s Top-3    Top-8" % "Strategy")
    print("  %-20s %d/%d (%d%%)  %d/%d (%d%%)" % (
        "Current (90/10)",
        current_top3, total_expected, 100*current_top3//max(1,total_expected),
        current_top8, total_expected, 100*current_top8//max(1,total_expected)))
    print("  %-20s %d/%d (%d%%)  %d/%d (%d%%)" % (
        "RRF (pre-boost)",
        rrf_top3, total_expected, 100*rrf_top3//max(1,total_expected),
        rrf_top8, total_expected, 100*rrf_top8//max(1,total_expected)))

    print()
    if rrf_wins:
        print("RRF finds what Current misses (%d):" % len(rrf_wins))
        for q in rrf_wins: print("  + %s" % q)
    if current_wins:
        print("Current finds what RRF misses (%d):" % len(current_wins))
        for q in current_wins: print("  + %s" % q)
    print("Both find (%d), Neither finds (%d)" % (
        len(ties), total_expected - len(ties) - len(rrf_wins) - len(current_wins)))


if __name__ == "__main__":
    main()
