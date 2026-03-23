#!/usr/bin/env python3
"""
Benchmark: Real Conversation Simulation
Tests how the brain handles messy, topic-jumping, non-engineering usage.

Methodology:
  For each query, embed with Arctic v1.5, brute-force cosine against
  all node_embeddings + node_enrichments, take MAX per node, return top 10.
  Judge relevance harshly: if a brain/engineering result appears for a
  personal query, that's a FAIL (context bleed).

Usage:
  python3 tests/benchmark_real_conversations.py [--verbose]
"""

import json
import os
import shutil
import sqlite3
import struct
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

# Add servers to path so we can import embedder
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'servers'))
import embedder

# ─── Configuration ───────────────────────────────────────────────────────

DB_PATH = '/tmp/brain_conversation_test.db'
MODEL_PATH = os.path.expanduser('~/brain/model-package/brain_embedding/model')
TOP_K = 10  # Results to retrieve per query
JUDGE_K = 5  # Results to judge for precision
CONTEXT_BLEED_THRESHOLD = 0.50  # Scores above this for irrelevant queries = bleed
NOISE_THRESHOLD = 0.30  # Below this = probably noise, above = potential false positive
VERBOSE = '--verbose' in sys.argv or '-v' in sys.argv


# ─── Relevance judgments ─────────────────────────────────────────────────
# For each query, specify whether relevant nodes exist and keywords that
# indicate relevance. 'has_relevant' = False means NO nodes should match.

RELEVANCE_KEYWORDS = {
    # Engineering keywords that indicate a result IS relevant to brain/code queries
    'engineering': [
        'recall', 'pipeline', 'embed', 'brain', 'hook', 'daemon', 'mcp',
        'encoding', 'node', 'edge', 'graph', 'consciousness', 'precision',
        'vocab', 'tfidf', 'cosine', 'vector', 'operator', 'compact',
        'session', 'api', 'architecture', 'sacred', 'benchmark', 'test',
        'refactor', 'dal', 'enrichment', 'constraint', 'convention',
        'lesson', 'decision', 'rule', 'mechanism', 'impact', 'purpose',
        'schema', 'sqlite', 'onnx', 'fastembed', 'arctic', 'server',
        'dream', 'tension', 'hypothesis', 'correction', 'mental_model',
    ],
    # Personal/non-technical keywords
    'personal': [
        'birthday', 'trip', 'travel', 'gift', 'email', 'landlord',
        'pasta', 'cooking', 'recipe', 'mom', 'family', 'car',
        'fishing', 'fashion', 'oil', 'childhood', 'physical',
    ],
}


# ─── Simulation Definitions ─────────────────────────────────────────────

SIMULATION_1_ENGINEERING = {
    'name': 'Simulation 1: Long Engineering Conversation (20 turns)',
    'description': 'Realistic back-and-forth working through a problem',
    'queries': [
        {'turn': 1,  'query': 'I need to fix the recall pipeline', 'has_relevant': True, 'domain': 'engineering'},
        {'turn': 2,  'query': 'what are the main components of recall', 'has_relevant': True, 'domain': 'engineering'},
        {'turn': 3,  'query': 'wait, what did we decide about embeddings last time', 'has_relevant': True, 'domain': 'engineering'},
        {'turn': 4,  'query': 'how does the brute force cosine scan work', 'has_relevant': True, 'domain': 'engineering'},
        {'turn': 5,  'query': 'show me how the hooks work', 'has_relevant': True, 'domain': 'engineering'},
        {'turn': 6,  'query': 'which hooks fire before edits', 'has_relevant': True, 'domain': 'engineering'},
        {'turn': 7,  'query': 'actually let me check if the daemon is healthy', 'has_relevant': True, 'domain': 'engineering'},
        {'turn': 8,  'query': 'what errors has the brain logged recently', 'has_relevant': True, 'domain': 'engineering'},
        {'turn': 9,  'query': 'how does vocabulary expansion affect recall quality', 'has_relevant': True, 'domain': 'engineering'},
        {'turn': 10, 'query': 'ok back to recall — what is the precision score', 'has_relevant': True, 'domain': 'engineering'},
        {'turn': 11, 'query': 'what is NDCG and how do we measure it', 'has_relevant': True, 'domain': 'engineering'},
        {'turn': 12, 'query': 'hmm what about the vocab system, does it help recall', 'has_relevant': True, 'domain': 'engineering'},
        {'turn': 13, 'query': 'what are the enrichment vectors for', 'has_relevant': True, 'domain': 'engineering'},
        {'turn': 14, 'query': 'how does graph augmented recall work', 'has_relevant': True, 'domain': 'engineering'},
        {'turn': 15, 'query': 'let me try a different approach, what are our options for improving recall', 'has_relevant': True, 'domain': 'engineering'},
        {'turn': 16, 'query': 'what is the golden dataset and how many cases does it have', 'has_relevant': True, 'domain': 'engineering'},
        {'turn': 17, 'query': 'how do we handle nodes without embeddings', 'has_relevant': True, 'domain': 'engineering'},
        {'turn': 18, 'query': 'before I commit, any rules about testing', 'has_relevant': True, 'domain': 'engineering'},
        {'turn': 19, 'query': 'what are the sacred systems we cannot change without benchmarks', 'has_relevant': True, 'domain': 'engineering'},
        {'turn': 20, 'query': 'what did we learn from the last time we changed recall', 'has_relevant': True, 'domain': 'engineering'},
    ],
}

SIMULATION_2_TOPIC_JUMPING = {
    'name': 'Simulation 2: Topic Jumping (15 turns)',
    'description': 'Real human jumping between completely different contexts',
    'queries': [
        {'turn': 1,  'query': 'what is our API architecture', 'has_relevant': True, 'domain': 'engineering'},
        {'turn': 2,  'query': 'remind me about mom\'s birthday', 'has_relevant': False, 'domain': 'personal'},
        {'turn': 3,  'query': 'what was that thing about React hooks', 'has_relevant': True, 'domain': 'engineering'},
        {'turn': 4,  'query': 'I need to plan a trip to Japan', 'has_relevant': False, 'domain': 'personal'},
        {'turn': 5,  'query': 'how does the brain handle vocabulary', 'has_relevant': True, 'domain': 'engineering'},
        {'turn': 6,  'query': 'what is a good gift for a 60th birthday', 'has_relevant': False, 'domain': 'personal'},
        {'turn': 7,  'query': 'are there any reminders due', 'has_relevant': True, 'domain': 'engineering'},
        {'turn': 8,  'query': 'what did Tom decide about the operator channel', 'has_relevant': True, 'domain': 'engineering'},
        {'turn': 9,  'query': 'can you help me draft an email to my landlord', 'has_relevant': False, 'domain': 'personal'},
        {'turn': 10, 'query': 'back to the brain — what is the encoding pipeline', 'has_relevant': True, 'domain': 'engineering'},
        {'turn': 11, 'query': 'how do I make homemade pasta', 'has_relevant': False, 'domain': 'personal'},
        {'turn': 12, 'query': 'what are the sacred systems we cannot change', 'has_relevant': True, 'domain': 'engineering'},
        {'turn': 13, 'query': 'I feel stuck, what should I focus on', 'has_relevant': False, 'domain': 'personal'},
        {'turn': 14, 'query': 'what were the benchmark results', 'has_relevant': True, 'domain': 'engineering'},
        {'turn': 15, 'query': 'tell me something interesting the brain dreamed', 'has_relevant': True, 'domain': 'engineering'},
    ],
}

SIMULATION_3_CONTEXT_BLEED = {
    'name': 'Simulation 3: Adversarial Context Bleed',
    'description': 'Ambiguous terms that exist in both engineering and everyday domains',
    'queries': [
        {'turn': 1, 'query': 'birthday', 'has_relevant': False, 'domain': 'personal',
         'note': 'Should NOT return brain/project birthday references'},
        {'turn': 2, 'query': 'model', 'has_relevant': False, 'domain': 'ambiguous',
         'note': 'Ambiguous: embedding model vs fashion model. Brain will return code results = bleed'},
        {'turn': 3, 'query': 'hooks', 'has_relevant': False, 'domain': 'ambiguous',
         'note': 'React hooks vs fishing hooks vs psychological hooks'},
        {'turn': 4, 'query': 'pipeline', 'has_relevant': False, 'domain': 'ambiguous',
         'note': 'Code pipeline vs oil pipeline'},
        {'turn': 5, 'query': 'memory', 'has_relevant': False, 'domain': 'ambiguous',
         'note': 'Brain memory system vs childhood memories'},
        {'turn': 6, 'query': 'crash', 'has_relevant': False, 'domain': 'ambiguous',
         'note': 'API crash vs car crash'},
        {'turn': 7, 'query': 'bridge', 'has_relevant': False, 'domain': 'ambiguous',
         'note': 'Graph bridge edges vs physical bridge'},
        {'turn': 8, 'query': 'my cat is sick', 'has_relevant': False, 'domain': 'personal',
         'note': 'Pure personal, no engineering content'},
        {'turn': 9, 'query': 'weather forecast for tomorrow', 'has_relevant': False, 'domain': 'personal',
         'note': 'Completely unrelated to brain'},
        {'turn': 10, 'query': 'best restaurants near me', 'has_relevant': False, 'domain': 'personal',
         'note': 'Pure personal query'},
    ],
}

SIMULATION_4_EMOTIONAL = {
    'name': 'Simulation 4: Emotional/Vague Queries',
    'description': 'Queries without clear semantic targets',
    'queries': [
        {'turn': 1,  'query': 'I\'m feeling overwhelmed', 'has_relevant': False, 'domain': 'emotional'},
        {'turn': 2,  'query': 'what should I do next', 'has_relevant': False, 'domain': 'vague'},
        {'turn': 3,  'query': 'something feels wrong', 'has_relevant': False, 'domain': 'emotional'},
        {'turn': 4,  'query': 'I had an insight about the project', 'has_relevant': True, 'domain': 'vague',
         'note': 'Might match project/insight nodes'},
        {'turn': 5,  'query': 'let\'s brainstorm', 'has_relevant': False, 'domain': 'vague'},
        {'turn': 6,  'query': 'what is the big picture', 'has_relevant': True, 'domain': 'vague',
         'note': 'Might match architecture/overview nodes'},
        {'turn': 7,  'query': 'am I making progress', 'has_relevant': False, 'domain': 'emotional'},
        {'turn': 8,  'query': 'what have we accomplished', 'has_relevant': True, 'domain': 'vague',
         'note': 'Might match session synthesis or milestone nodes'},
        {'turn': 9,  'query': 'what are we missing', 'has_relevant': True, 'domain': 'vague',
         'note': 'Might match gaps/tensions'},
        {'turn': 10, 'query': 'help me think through this', 'has_relevant': False, 'domain': 'vague'},
    ],
}

SIMULATION_5_SEGMENTS = {
    'name': 'Simulation 5: Long Session with Segment Boundaries',
    'description': '3-hour session with natural topic segments',
    'segments': [
        {'name': 'Segment 1: Deep work on recall', 'domain': 'engineering'},
        {'name': 'Segment 2: Personal task', 'domain': 'personal'},
        {'name': 'Segment 3: Hooks subsystem', 'domain': 'engineering'},
        {'name': 'Segment 4: Future brainstorming', 'domain': 'engineering'},
        {'name': 'Segment 5: Wrapping up', 'domain': 'engineering'},
    ],
    'queries': [
        # Segment 1: Deep work on recall pipeline (turns 1-8)
        {'turn': 1,  'segment': 0, 'query': 'how does the recall pipeline work end to end', 'has_relevant': True, 'domain': 'engineering'},
        {'turn': 2,  'segment': 0, 'query': 'what is the embedding scan step', 'has_relevant': True, 'domain': 'engineering'},
        {'turn': 3,  'segment': 0, 'query': 'how do enrichment vectors improve recall', 'has_relevant': True, 'domain': 'engineering'},
        {'turn': 4,  'segment': 0, 'query': 'what is the blending formula for keyword and embedding scores', 'has_relevant': True, 'domain': 'engineering'},
        {'turn': 5,  'segment': 0, 'query': 'what is the confidence multiplier', 'has_relevant': True, 'domain': 'engineering'},
        {'turn': 6,  'segment': 0, 'query': 'how does graph augmented recall add neighbors', 'has_relevant': True, 'domain': 'engineering'},
        {'turn': 7,  'segment': 0, 'query': 'what recall bugs have we fixed', 'has_relevant': True, 'domain': 'engineering'},
        {'turn': 8,  'segment': 0, 'query': 'what is the current NDCG score on the golden dataset', 'has_relevant': True, 'domain': 'engineering'},
        # Segment 2: Quick personal check (turns 9-12)
        {'turn': 9,  'segment': 1, 'query': 'I need to buy groceries', 'has_relevant': False, 'domain': 'personal'},
        {'turn': 10, 'segment': 1, 'query': 'what time is the dentist appointment', 'has_relevant': False, 'domain': 'personal'},
        {'turn': 11, 'segment': 1, 'query': 'remind me to call the plumber', 'has_relevant': False, 'domain': 'personal'},
        {'turn': 12, 'segment': 1, 'query': 'where did I park my car', 'has_relevant': False, 'domain': 'personal'},
        # Segment 3: Hooks subsystem (turns 13-20)
        {'turn': 13, 'segment': 2, 'query': 'how do the brain hooks work', 'has_relevant': True, 'domain': 'engineering'},
        {'turn': 14, 'segment': 2, 'query': 'what hooks fire on session start', 'has_relevant': True, 'domain': 'engineering'},
        {'turn': 15, 'segment': 2, 'query': 'how does pre-edit suggest work', 'has_relevant': True, 'domain': 'engineering'},
        {'turn': 16, 'segment': 2, 'query': 'what is the operator channel', 'has_relevant': True, 'domain': 'engineering'},
        {'turn': 17, 'segment': 2, 'query': 'how does wrap_for_hook format output', 'has_relevant': True, 'domain': 'engineering'},
        {'turn': 18, 'segment': 2, 'query': 'what happens during compaction', 'has_relevant': True, 'domain': 'engineering'},
        {'turn': 19, 'segment': 2, 'query': 'how does the brain save state', 'has_relevant': True, 'domain': 'engineering'},
        {'turn': 20, 'segment': 2, 'query': 'what is the daemon architecture', 'has_relevant': True, 'domain': 'engineering'},
        # Segment 4: Creative brainstorming (turns 21-25)
        {'turn': 21, 'segment': 3, 'query': 'what is the future of the brain', 'has_relevant': True, 'domain': 'engineering'},
        {'turn': 22, 'segment': 3, 'query': 'how could we make recall faster', 'has_relevant': True, 'domain': 'engineering'},
        {'turn': 23, 'segment': 3, 'query': 'what novel features could the brain have', 'has_relevant': True, 'domain': 'engineering'},
        {'turn': 24, 'segment': 3, 'query': 'how would a second brain instance interact with this one', 'has_relevant': True, 'domain': 'engineering'},
        {'turn': 25, 'segment': 3, 'query': 'what is the dream system and how could it evolve', 'has_relevant': True, 'domain': 'engineering'},
        # Segment 5: Wrapping up (turns 26-30)
        {'turn': 26, 'segment': 4, 'query': 'what should we encode from this session', 'has_relevant': True, 'domain': 'engineering'},
        {'turn': 27, 'segment': 4, 'query': 'what decisions did we make', 'has_relevant': True, 'domain': 'engineering'},
        {'turn': 28, 'segment': 4, 'query': 'any open questions to carry forward', 'has_relevant': True, 'domain': 'engineering'},
        {'turn': 29, 'segment': 4, 'query': 'what lessons did we learn', 'has_relevant': True, 'domain': 'engineering'},
        {'turn': 30, 'segment': 4, 'query': 'summarize what the brain knows about itself', 'has_relevant': True, 'domain': 'engineering'},
    ],
}


# ─── Core Recall Engine (standalone, no Brain class) ─────────────────────

class StandaloneRecall:
    """
    Minimal recall engine that replicates the production embedding scan
    without needing the full Brain class. Reads directly from SQLite.
    """

    def __init__(self, db_path: str):
        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = None  # Use tuples for speed

    def recall(self, query: str, top_k: int = TOP_K) -> List[Dict[str, Any]]:
        """
        Embed query, scan all node_embeddings + node_enrichments,
        take MAX score per node, return top_k.
        """
        query_vec = embedder.embed(query)
        if not query_vec:
            return []

        # Score from primary embeddings
        scores = {}  # node_id -> (best_score, source)
        cursor = self.conn.execute(
            '''SELECT ne.node_id, ne.embedding
               FROM node_embeddings ne
               JOIN nodes n ON n.id = ne.node_id
               WHERE n.archived = 0'''
        )
        for node_id, blob in cursor.fetchall():
            if not blob:
                continue
            sim = embedder.cosine_similarity(query_vec, blob)
            scores[node_id] = (sim, 'primary')

        # Score from enrichment embeddings (take MAX across primary + enrichments)
        try:
            cursor = self.conn.execute(
                '''SELECT en.node_id, en.vector_type, en.embedding
                   FROM node_enrichments en
                   JOIN nodes n ON n.id = en.node_id
                   WHERE n.archived = 0 AND en.embedding IS NOT NULL'''
            )
            for node_id, vec_type, blob in cursor.fetchall():
                if not blob:
                    continue
                sim = embedder.cosine_similarity(query_vec, blob)
                current_best = scores.get(node_id, (0, ''))[0]
                if sim > current_best:
                    scores[node_id] = (sim, f'enrichment:{vec_type}')
        except Exception:
            pass  # Table might not exist

        # Sort by score, get top_k
        ranked = sorted(scores.items(), key=lambda x: -x[1][0])[:top_k]

        # Hydrate with node metadata
        results = []
        for node_id, (score, source) in ranked:
            row = self.conn.execute(
                '''SELECT id, type, title, content, keywords
                   FROM nodes WHERE id = ?''',
                (node_id,)
            ).fetchone()
            if row:
                results.append({
                    'id': row[0],
                    'type': row[1],
                    'title': row[2],
                    'content': row[3][:200] if row[3] else '',
                    'keywords': row[4],
                    'score': round(score, 4),
                    'source': source,
                })

        return results

    def close(self):
        self.conn.close()


# ─── Relevance Judging ──────────────────────────────────────────────────

def judge_relevance(query_info: Dict, result: Dict) -> str:
    """
    Judge whether a result is RELEVANT, NOISE, or AMBIGUOUS for a query.

    Rules:
    - If query has_relevant=False (personal/emotional) and result is
      engineering/brain content: NOISE (context bleed)
    - If query has_relevant=True (engineering) and result matches the
      query topic: RELEVANT
    - Otherwise: AMBIGUOUS
    """
    title = (result.get('title', '') or '').lower()
    content = (result.get('content', '') or '').lower()
    keywords = (result.get('keywords', '') or '').lower()
    node_type = (result.get('type', '') or '').lower()
    combined = f'{title} {content} {keywords} {node_type}'
    query = query_info['query'].lower()
    has_relevant = query_info['has_relevant']
    domain = query_info.get('domain', 'unknown')

    if not has_relevant:
        # This query should NOT have relevant results.
        # Any brain/engineering result is NOISE (context bleed).
        is_brain_content = any(kw in combined for kw in RELEVANCE_KEYWORDS['engineering'])
        if is_brain_content:
            return 'NOISE'
        else:
            # Result doesn't match engineering keywords either — still noise
            # since the query shouldn't match ANYTHING in this brain
            return 'NOISE'

    if has_relevant and domain == 'engineering':
        # Engineering query — check if result is topically related
        query_words = set(query.split()) - {'the', 'a', 'an', 'is', 'are', 'how',
                                             'does', 'what', 'when', 'where', 'why',
                                             'do', 'did', 'we', 'i', 'our', 'to',
                                             'of', 'in', 'for', 'on', 'at', 'by',
                                             'let', 'me', 'ok', 'back', 'actually',
                                             'wait', 'show', 'any', 'about', 'and',
                                             'it', 'this', 'that', 'with', 'from',
                                             'can', 'could', 'should', 'before',
                                             'after', 'not', 'has', 'have', 'had',
                                             'been', 'but', 'or', 'hmm', 'try',
                                             'different', 'approach', 'last', 'time',
                                             'something', 'interesting'}

        # Check if any significant query word appears in the result
        for word in query_words:
            if len(word) > 2 and word in combined:
                return 'RELEVANT'

        # Check semantic topic overlap (looser match)
        topic_keywords = _extract_topic_keywords(query)
        for tk in topic_keywords:
            if tk in combined:
                return 'RELEVANT'

        return 'AMBIGUOUS'

    if has_relevant and domain == 'vague':
        # Vague queries — more lenient. If result is brain-related, it's
        # probably useful context.
        return 'AMBIGUOUS'

    return 'AMBIGUOUS'


def _extract_topic_keywords(query: str) -> List[str]:
    """Extract topic-relevant keywords from a query for matching."""
    topic_map = {
        'recall': ['recall', 'search', 'retrieval', 'query', 'cosine', 'similarity', 'ranking'],
        'pipeline': ['pipeline', 'recall', 'embed', 'score', 'blend', 'hydrate'],
        'embedding': ['embed', 'vector', 'cosine', 'arctic', 'onnx', 'fastembed', 'dimension'],
        'hook': ['hook', 'pre-edit', 'session', 'compact', 'boot', 'daemon'],
        'daemon': ['daemon', 'socket', 'server', 'mcp', 'process', 'boot'],
        'vocab': ['vocab', 'vocabulary', 'expansion', 'term', 'mapping'],
        'precision': ['precision', 'ndcg', 'mrr', 'benchmark', 'golden', 'eval'],
        'enrichment': ['enrichment', 'multi-vector', 'question', 'anchor', 'bridge', 'keywords'],
        'graph': ['graph', 'neighbor', 'edge', 'augment', 'traverse', 'hop'],
        'test': ['test', 'benchmark', 'golden', 'eval', 'sacred', 'harness'],
        'sacred': ['sacred', 'benchmark', 'embed', 'recall', 'encoding', 'precision', 'voice'],
        'confidence': ['confidence', 'multiplier', 'penalty', 'boost', 'weight'],
        'encode': ['encode', 'encoding', 'remember', 'store', 'node', 'checklist'],
        'operator': ['operator', 'channel', 'voice', 'wrap_for_hook', 'relay'],
        'compaction': ['compact', 'save', 'pre-compact', 'post-compact', 'reboot'],
        'dream': ['dream', 'bridge', 'walk', 'novelty', 'consciousness'],
        'architecture': ['architecture', 'daemon', 'mcp', 'stdio', 'socket', 'brain.py'],
        'decision': ['decision', 'decide', 'chose', 'choice', 'pick'],
        'lesson': ['lesson', 'learn', 'mistake', 'correction', 'bug'],
        'bug': ['bug', 'fix', 'error', 'broken', 'crash', 'regression'],
        'rule': ['rule', 'constraint', 'convention', 'locked', 'sacred'],
    }

    keywords = set()
    query_lower = query.lower()
    for topic, kws in topic_map.items():
        if topic in query_lower or any(kw in query_lower for kw in kws):
            keywords.update(kws)

    return list(keywords)


# ─── Metric Computation ─────────────────────────────────────────────────

def compute_metrics(results: List[Dict]) -> Dict[str, Any]:
    """Compute aggregate metrics from simulation results."""

    # Split by has_relevant
    relevant_queries = [r for r in results if r['has_relevant']]
    irrelevant_queries = [r for r in results if not r['has_relevant']]

    # 1. Precision when relevant: % of top-JUDGE_K results that are RELEVANT
    precision_scores = []
    for r in relevant_queries:
        judgments = r.get('judgments', [])[:JUDGE_K]
        if judgments:
            relevant_count = sum(1 for j in judgments if j == 'RELEVANT')
            precision_scores.append(relevant_count / len(judgments))

    avg_precision = sum(precision_scores) / len(precision_scores) if precision_scores else 0

    # 2. False positive rate: for irrelevant queries, % of top-JUDGE_K
    #    results that returned NOISE with high scores
    fp_scores = []
    for r in irrelevant_queries:
        top_results = r.get('results', [])[:JUDGE_K]
        if top_results:
            high_score_count = sum(1 for res in top_results
                                   if res['score'] > CONTEXT_BLEED_THRESHOLD)
            fp_scores.append(high_score_count / len(top_results))

    avg_false_positive = sum(fp_scores) / len(fp_scores) if fp_scores else 0

    # 3. Context bleed score: avg max score for irrelevant queries
    bleed_scores = []
    for r in irrelevant_queries:
        top_results = r.get('results', [])
        if top_results:
            max_score = top_results[0]['score'] if top_results else 0
            bleed_scores.append(max_score)

    avg_bleed = sum(bleed_scores) / len(bleed_scores) if bleed_scores else 0

    # 4. Average top score for relevant queries
    relevant_top_scores = []
    for r in relevant_queries:
        top_results = r.get('results', [])
        if top_results:
            relevant_top_scores.append(top_results[0]['score'])

    avg_relevant_top = sum(relevant_top_scores) / len(relevant_top_scores) if relevant_top_scores else 0

    return {
        'precision_when_relevant': round(avg_precision, 3),
        'false_positive_rate': round(avg_false_positive, 3),
        'context_bleed_score': round(avg_bleed, 4),
        'avg_relevant_top_score': round(avg_relevant_top, 4),
        'total_relevant_queries': len(relevant_queries),
        'total_irrelevant_queries': len(irrelevant_queries),
    }


def compute_segment_metrics(results: List[Dict], simulation: Dict) -> Dict[str, Any]:
    """Compute segment-specific metrics for Simulation 5."""
    segments = simulation.get('segments', [])
    segment_metrics = {}

    for i, seg in enumerate(segments):
        seg_results = [r for r in results if r.get('segment') == i]
        if not seg_results:
            continue

        # Precision within segment
        precisions = []
        for r in seg_results:
            if r['has_relevant']:
                judgments = r.get('judgments', [])[:JUDGE_K]
                if judgments:
                    rel_count = sum(1 for j in judgments if j == 'RELEVANT')
                    precisions.append(rel_count / len(judgments))

        # Bleed score (for personal segments)
        bleeds = []
        for r in seg_results:
            if not r['has_relevant']:
                top_results = r.get('results', [])
                if top_results:
                    bleeds.append(top_results[0]['score'])

        segment_metrics[seg['name']] = {
            'domain': seg['domain'],
            'query_count': len(seg_results),
            'avg_precision': round(sum(precisions) / len(precisions), 3) if precisions else None,
            'avg_bleed_score': round(sum(bleeds) / len(bleeds), 4) if bleeds else None,
        }

    return segment_metrics


def compute_topic_recovery(results: List[Dict]) -> List[Dict]:
    """
    For topic-jumping simulations: after a personal query, does the next
    engineering query recover precision?
    """
    recoveries = []
    for i in range(1, len(results) - 1):
        prev = results[i - 1]
        curr = results[i]
        nxt = results[i + 1] if i + 1 < len(results) else None

        if not curr['has_relevant'] and prev['has_relevant'] and nxt and nxt['has_relevant']:
            # curr is personal, prev is engineering, next is engineering
            prev_prec = _query_precision(prev)
            next_prec = _query_precision(nxt)
            recoveries.append({
                'before_jump': prev['query'],
                'personal_query': curr['query'],
                'after_return': nxt['query'],
                'precision_before': round(prev_prec, 3),
                'precision_after': round(next_prec, 3),
                'recovered': next_prec >= prev_prec * 0.8,  # Within 80%
            })

    return recoveries


def _query_precision(result: Dict) -> float:
    """Compute precision for a single query result."""
    judgments = result.get('judgments', [])[:JUDGE_K]
    if not judgments:
        return 0
    return sum(1 for j in judgments if j == 'RELEVANT') / len(judgments)


# ─── Running Simulations ────────────────────────────────────────────────

def run_simulation(engine: StandaloneRecall, simulation: Dict) -> List[Dict]:
    """Run a simulation and return results with judgments."""
    print(f'\n{"="*70}')
    print(f'  {simulation["name"]}')
    print(f'  {simulation["description"]}')
    print(f'{"="*70}')

    all_results = []
    queries = simulation['queries']

    for q in queries:
        t0 = time.time()
        results = engine.recall(q['query'])
        elapsed = round((time.time() - t0) * 1000)

        # Judge each result
        judgments = [judge_relevance(q, r) for r in results[:JUDGE_K]]

        entry = {
            'turn': q['turn'],
            'query': q['query'],
            'has_relevant': q['has_relevant'],
            'domain': q.get('domain', 'unknown'),
            'segment': q.get('segment'),
            'results': results,
            'judgments': judgments,
            'elapsed_ms': elapsed,
            'top_score': results[0]['score'] if results else 0,
        }
        all_results.append(entry)

        # Print turn summary
        precision = _query_precision(entry)
        status = ''
        if not q['has_relevant']:
            if entry['top_score'] > CONTEXT_BLEED_THRESHOLD:
                status = f' ** BLEED (top={entry["top_score"]:.3f})'
            else:
                status = f' OK (top={entry["top_score"]:.3f})'
        else:
            relevant_count = sum(1 for j in judgments if j == 'RELEVANT')
            status = f' P@{JUDGE_K}={precision:.1%} ({relevant_count}/{len(judgments)} relevant)'

        print(f'  T{q["turn"]:2d} [{elapsed:4d}ms]{status}')
        if VERBOSE:
            print(f'       Q: "{q["query"]}"')
            for i, r in enumerate(results[:JUDGE_K]):
                j = judgments[i] if i < len(judgments) else '?'
                marker = '  ' if j == 'RELEVANT' else 'X ' if j == 'NOISE' else '? '
                print(f'       {marker}{r["score"]:.4f} [{r["type"]:12s}] {r["title"][:60]}')
                if VERBOSE and r.get('source', '').startswith('enrichment:'):
                    print(f'                 (via {r["source"]})')

    return all_results


# ─── Report Generation ──────────────────────────────────────────────────

def print_report(all_sim_results: Dict[str, List[Dict]], simulations: Dict[str, Dict]):
    """Print comprehensive benchmark report."""

    print(f'\n\n{"#"*70}')
    print(f'#  BENCHMARK REPORT: Real Conversation Simulation')
    print(f'#  {time.strftime("%Y-%m-%d %H:%M:%S")}')
    print(f'{"#"*70}')

    # Per-simulation metrics
    for sim_name, results in all_sim_results.items():
        metrics = compute_metrics(results)
        print(f'\n--- {sim_name} ---')
        print(f'  Precision (relevant queries):  {metrics["precision_when_relevant"]:.1%}')
        print(f'  False positive rate:           {metrics["false_positive_rate"]:.1%}')
        print(f'  Context bleed score:           {metrics["context_bleed_score"]:.4f}')
        print(f'  Avg top score (relevant):      {metrics["avg_relevant_top_score"]:.4f}')
        print(f'  Relevant queries: {metrics["total_relevant_queries"]}  |  Irrelevant queries: {metrics["total_irrelevant_queries"]}')

    # Topic recovery analysis (Sim 2)
    sim2_results = all_sim_results.get('sim2', [])
    if sim2_results:
        recoveries = compute_topic_recovery(sim2_results)
        if recoveries:
            print(f'\n--- Topic Recovery Analysis (Sim 2) ---')
            for rec in recoveries:
                status = 'RECOVERED' if rec['recovered'] else 'DEGRADED'
                print(f'  {rec["before_jump"][:40]:40s} -> {rec["personal_query"][:25]:25s} -> {rec["after_return"][:40]}')
                print(f'    Precision: {rec["precision_before"]:.1%} -> {rec["precision_after"]:.1%} [{status}]')

    # Segment analysis (Sim 5)
    sim5_results = all_sim_results.get('sim5', [])
    if sim5_results:
        seg_metrics = compute_segment_metrics(sim5_results, SIMULATION_5_SEGMENTS)
        print(f'\n--- Segment Analysis (Sim 5) ---')
        for seg_name, sm in seg_metrics.items():
            prec_str = f'{sm["avg_precision"]:.1%}' if sm['avg_precision'] is not None else 'N/A'
            bleed_str = f'{sm["avg_bleed_score"]:.4f}' if sm['avg_bleed_score'] is not None else 'N/A'
            print(f'  {seg_name:40s}  P={prec_str:6s}  Bleed={bleed_str}')

    # Context bleed deep dive (Sim 3)
    sim3_results = all_sim_results.get('sim3', [])
    if sim3_results:
        print(f'\n--- Context Bleed Deep Dive (Sim 3) ---')
        for r in sim3_results:
            top = r['results'][0] if r['results'] else None
            score = top['score'] if top else 0
            title = top['title'][:50] if top else 'N/A'
            flag = 'BLEED' if score > CONTEXT_BLEED_THRESHOLD else 'OK'
            note = r.get('note', '') if isinstance(r, dict) else ''
            # Get note from query info
            for q in SIMULATION_3_CONTEXT_BLEED['queries']:
                if q['query'] == r['query']:
                    note = q.get('note', '')
                    break
            print(f'  [{flag:5s}] "{r["query"]:35s}" -> {score:.4f} "{title}"')
            if note:
                print(f'          Note: {note}')

    # Emotional/vague query analysis (Sim 4)
    sim4_results = all_sim_results.get('sim4', [])
    if sim4_results:
        print(f'\n--- Emotional/Vague Query Analysis (Sim 4) ---')
        for r in sim4_results:
            top = r['results'][0] if r['results'] else None
            score = top['score'] if top else 0
            title = top['title'][:50] if top else 'N/A'
            has_rel = 'has_rel' if r['has_relevant'] else 'no_rel'
            print(f'  [{has_rel:7s}] "{r["query"]:35s}" -> {score:.4f} "{title}"')

    # Overall summary
    print(f'\n{"="*70}')
    print(f'  OVERALL SUMMARY')
    print(f'{"="*70}')

    all_results = []
    for v in all_sim_results.values():
        all_results.extend(v)

    overall = compute_metrics(all_results)
    print(f'  Total queries:                 {len(all_results)}')
    print(f'  Overall precision (relevant):  {overall["precision_when_relevant"]:.1%}')
    print(f'  Overall false positive rate:   {overall["false_positive_rate"]:.1%}')
    print(f'  Overall context bleed score:   {overall["context_bleed_score"]:.4f}')

    # Failure modes
    print(f'\n--- Failure Modes ---')

    # High-score noise (context bleed)
    bleed_cases = []
    for r in all_results:
        if not r['has_relevant'] and r['top_score'] > CONTEXT_BLEED_THRESHOLD:
            bleed_cases.append(r)
    print(f'  Context bleed cases (score > {CONTEXT_BLEED_THRESHOLD}): {len(bleed_cases)}')
    for c in bleed_cases[:10]:
        top = c['results'][0]
        print(f'    Q: "{c["query"]}" -> {top["score"]:.4f} [{top["type"]}] {top["title"][:50]}')

    # Low-precision engineering queries
    low_prec = []
    for r in all_results:
        if r['has_relevant'] and _query_precision(r) < 0.4:
            low_prec.append(r)
    print(f'\n  Low-precision engineering queries (P@{JUDGE_K} < 40%): {len(low_prec)}')
    for c in low_prec[:10]:
        prec = _query_precision(c)
        print(f'    Q: "{c["query"]}" -> P@{JUDGE_K}={prec:.1%}, top={c["top_score"]:.4f}')

    return overall


# ─── New Golden Dataset Cases ────────────────────────────────────────────

def generate_golden_cases(all_sim_results: Dict) -> List[Dict]:
    """Generate new golden dataset cases from interesting failures and successes."""
    cases = []

    # From context bleed: queries that SHOULD return nothing but got high scores
    for sim_name, results in all_sim_results.items():
        for r in results:
            if not r['has_relevant'] and r['top_score'] > CONTEXT_BLEED_THRESHOLD:
                cases.append({
                    'query': r['query'],
                    'expected_type': 'negative',
                    'description': f'Context bleed: personal query got high brain score ({r["top_score"]:.3f})',
                    'source_simulation': sim_name,
                    'max_acceptable_score': 0.45,
                })

            # From successful engineering queries
            if r['has_relevant'] and _query_precision(r) >= 0.6:
                top = r['results'][0]
                cases.append({
                    'query': r['query'],
                    'expected_type': 'positive',
                    'expected_node_title': top['title'],
                    'description': f'Engineering query with good precision',
                    'source_simulation': sim_name,
                    'min_expected_score': round(top['score'] * 0.7, 3),
                })

    return cases


# ─── Main ────────────────────────────────────────────────────────────────

def main():
    print('='*70)
    print('  Brain Benchmark: Real Conversation Simulation')
    print('='*70)

    # Load model
    print('\nLoading Arctic v1.5 embedder...')
    t0 = time.time()
    embedder.load_model({
        'model_name': 'Snowflake/snowflake-arctic-embed-m-v1.5',
        'dim': 768,
        'pooling': 'cls',
        'model_file': 'onnx/model.onnx',
        'model_path': MODEL_PATH,
    })
    print(f'  Loaded in {time.time() - t0:.1f}s')

    if not embedder.is_ready():
        print('ERROR: Embedder failed to load. Aborting.')
        sys.exit(1)

    # Initialize recall engine
    engine = StandaloneRecall(DB_PATH)

    # Count nodes
    node_count = engine.conn.execute('SELECT COUNT(*) FROM nodes WHERE archived=0').fetchone()[0]
    embed_count = engine.conn.execute('SELECT COUNT(*) FROM node_embeddings').fetchone()[0]
    enrich_count = engine.conn.execute('SELECT COUNT(*) FROM node_enrichments WHERE embedding IS NOT NULL').fetchone()[0]
    print(f'  Nodes: {node_count}, Embeddings: {embed_count}, Enrichments: {enrich_count}')

    # Run all simulations
    all_sim_results = {}

    print('\n' + '='*70)
    all_sim_results['sim1'] = run_simulation(engine, SIMULATION_1_ENGINEERING)
    all_sim_results['sim2'] = run_simulation(engine, SIMULATION_2_TOPIC_JUMPING)
    all_sim_results['sim3'] = run_simulation(engine, SIMULATION_3_CONTEXT_BLEED)
    all_sim_results['sim4'] = run_simulation(engine, SIMULATION_4_EMOTIONAL)
    all_sim_results['sim5'] = run_simulation(engine, SIMULATION_5_SEGMENTS)

    # Generate report
    overall = print_report(all_sim_results, {
        'sim1': SIMULATION_1_ENGINEERING,
        'sim2': SIMULATION_2_TOPIC_JUMPING,
        'sim3': SIMULATION_3_CONTEXT_BLEED,
        'sim4': SIMULATION_4_EMOTIONAL,
        'sim5': SIMULATION_5_SEGMENTS,
    })

    # Generate new golden dataset cases
    golden_cases = generate_golden_cases(all_sim_results)
    golden_path = os.path.join(os.path.dirname(__file__), 'golden_dataset_conversation_cases.json')
    with open(golden_path, 'w') as f:
        json.dump(golden_cases, f, indent=2)
    print(f'\n  Saved {len(golden_cases)} new golden dataset cases to {golden_path}')

    # Save raw results for analysis
    raw_path = os.path.join(os.path.dirname(__file__), 'benchmark_conversation_raw.json')
    # Strip content from results for smaller file
    slim_results = {}
    for sim_name, results in all_sim_results.items():
        slim = []
        for r in results:
            slim_r = {k: v for k, v in r.items() if k != 'results'}
            slim_r['results'] = [{
                'id': res['id'],
                'type': res['type'],
                'title': res['title'],
                'score': res['score'],
                'source': res['source'],
            } for res in r['results']]
            slim.append(slim_r)
        slim_results[sim_name] = slim

    with open(raw_path, 'w') as f:
        json.dump(slim_results, f, indent=2)
    print(f'  Saved raw results to {raw_path}')

    engine.close()
    print(f'\nDone.')

    return overall


if __name__ == '__main__':
    main()
