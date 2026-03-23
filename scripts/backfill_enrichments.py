#!/usr/bin/env python3 -u
"""
brain — Backfill V5 Enrichment Vectors

Generates Q/A/B/K enrichment vectors for all existing brain nodes using
Gemma 2B via Ollama + Arctic v1.5 embeddings.

Process:
1. Copy brain.db to temp file (validate on copy first)
2. For each node without enrichments:
   a. Find 5 neighbors via edges table
   b. Build V2 structured prompt
   c. Call Ollama gemma2:2b to generate Q/A/B/K
   d. Parse output, embed each text with Arctic v1.5
   e. Store in node_enrichments table
3. Run golden dataset benchmark on enriched copy
4. Optionally apply to live DB

Usage:
    python scripts/backfill_enrichments.py                    # Temp copy only
    python scripts/backfill_enrichments.py --apply            # Also apply to live DB
    python scripts/backfill_enrichments.py --live-only        # Skip temp, go straight to live
    python scripts/backfill_enrichments.py --benchmark-only   # Only run benchmark on existing enrichments
"""

import json
import os
import re
import shutil
import sqlite3
import struct
import subprocess
import sys
import tempfile
import time
import uuid
from datetime import datetime, timezone
from typing import Dict, List, Optional, Set, Tuple, Any

# Force CPU-only ONNX before any imports
os.environ["ORT_DISABLE_ALL_ACCELERATORS"] = "1"
os.environ.setdefault("ONNX_PROVIDERS", "CPUExecutionProvider")

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

OLLAMA_BIN = "/Applications/Ollama.app/Contents/Resources/ollama"
OLLAMA_MODEL = "gemma2:2b"
MODEL_PATH = os.path.expanduser("~/brain/model-package/brain_embedding/model")
LIVE_DB = os.path.expanduser("~/AgentsContext/brain/brain.db")

# V2 structured prompt template (from brain_constants.py)
ENRICHMENT_PROMPT = """The brain found these related memories:
{neighbors}

New node: "{title}"
Content: "{content}"

Generate exactly these lines, no explanations:
Q: [one question a user would naturally ask that leads to this node]
A: [3-5 word phrase using words from the neighbors above]
B: [one sentence connecting this node to its most important neighbor]
K: [5 comma-separated keywords borrowed from neighbors that also describe this node]"""


# ═══════════════════════════════════════════════════════════════
# EMBEDDING ENGINE
# ═══════════════════════════════════════════════════════════════

class Embedder:
    """Thin wrapper around FastEmbed for Arctic v1.5."""

    def __init__(self, model_path: str):
        from fastembed import TextEmbedding
        from fastembed.common.model_description import PoolingType, ModelSource

        model_name = "snowflake/snowflake-arctic-embed-m-v1.5"
        dim = 768

        supported = [m['model'].lower() for m in TextEmbedding.list_supported_models()]
        if model_name.lower() not in supported:
            TextEmbedding.add_custom_model(
                model=model_name,
                pooling=PoolingType.CLS,
                normalization=True,
                sources=ModelSource(hf=model_name),
                dim=dim,
                model_file="onnx/model.onnx",
            )

        self.model = TextEmbedding(
            model_name=model_name,
            specific_model_path=model_path,
            providers=["CPUExecutionProvider"],
        )
        self.dim = dim
        print(f"[embedder] Loaded {model_name} ({dim}d) from {model_path}")

    def embed(self, text: str) -> bytes:
        vecs = list(self.model.embed([text]))
        return vecs[0].astype('float32').tobytes()

    def embed_batch(self, texts: List[str]) -> List[bytes]:
        if not texts:
            return []
        vecs = list(self.model.embed(texts))
        return [v.astype('float32').tobytes() for v in vecs]


# ═══════════════════════════════════════════════════════════════
# OLLAMA LLM
# ═══════════════════════════════════════════════════════════════

def ollama_generate(prompt: str, timeout: int = 30) -> str:
    """Call ollama CLI to generate text."""
    try:
        result = subprocess.run(
            [OLLAMA_BIN, "run", OLLAMA_MODEL],
            input=prompt,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if result.returncode != 0:
            return ""
        return result.stdout.strip()
    except subprocess.TimeoutExpired:
        return ""
    except Exception as e:
        print(f"  [ollama] Exception: {e}", file=sys.stderr)
        return ""


# ═══════════════════════════════════════════════════════════════
# PARSING
# ═══════════════════════════════════════════════════════════════

def parse_enrichment(raw: str) -> Dict[str, str]:
    """Parse Q/A/B/K from LLM output.

    Returns dict with keys 'question', 'anchor', 'bridge', 'keywords'.
    Missing lines are omitted (not None).
    """
    result = {}
    prefix_map = {
        'Q:': 'question',
        'A:': 'anchor',
        'B:': 'bridge',
        'K:': 'keywords',
    }

    for line in raw.split('\n'):
        line = line.strip()
        for prefix, key in prefix_map.items():
            if prefix in line:
                # Extract everything after the prefix
                idx = line.index(prefix)
                text = line[idx + len(prefix):].strip()
                # Strip markdown formatting
                text = re.sub(r'^\*+\s*', '', text)
                text = re.sub(r'\s*\*+$', '', text)
                text = re.sub(r'^\[', '', text)
                text = re.sub(r'\]$', '', text)
                text = text.strip()
                if text and key not in result:  # First match wins
                    result[key] = text[:200]  # Cap at 200 chars
                break

    return result


# ═══════════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════════

def get_nodes_without_enrichments(conn: sqlite3.Connection) -> List[Dict]:
    """Get all non-archived nodes that have no enrichments."""
    rows = conn.execute("""
        SELECT n.id, n.type, n.title, n.content, n.keywords
        FROM nodes n
        WHERE n.archived = 0
          AND n.id NOT IN (SELECT DISTINCT node_id FROM node_enrichments)
        ORDER BY n.created_at
    """).fetchall()
    return [
        {'id': r[0], 'type': r[1], 'title': r[2], 'content': r[3] or '', 'keywords': r[4] or ''}
        for r in rows
    ]


def get_neighbors(conn: sqlite3.Connection, node_id: str, limit: int = 5) -> List[Dict]:
    """Find neighbors via edges table (both directions), sorted by weight DESC."""
    rows = conn.execute("""
        SELECT CASE WHEN e.source_id = ? THEN e.target_id ELSE e.source_id END as other_id,
               e.weight, e.relation
        FROM edges e
        WHERE e.source_id = ? OR e.target_id = ?
        ORDER BY e.weight DESC
        LIMIT ?
    """, (node_id, node_id, node_id, limit)).fetchall()

    neighbors = []
    for other_id, weight, relation in rows:
        node = conn.execute(
            "SELECT id, type, title, keywords FROM nodes WHERE id = ?",
            (other_id,)
        ).fetchone()
        if node:
            neighbors.append({
                'id': node[0],
                'type': node[1],
                'title': node[2],
                'keywords': node[3] or '',
            })
    return neighbors


def build_prompt(node: Dict, neighbors: List[Dict]) -> str:
    """Build V2 structured prompt for enrichment generation."""
    if neighbors:
        neighbor_lines = []
        for n in neighbors:
            kw = f", keywords: {n['keywords']}" if n['keywords'] else ""
            neighbor_lines.append(f"- {n['title']} (type: {n['type']}{kw})")
        neighbor_text = "\n".join(neighbor_lines)
    else:
        neighbor_text = "(no neighbors found)"

    return ENRICHMENT_PROMPT.format(
        neighbors=neighbor_text,
        title=node['title'],
        content=node['content'][:200],
    )


# ═══════════════════════════════════════════════════════════════
# ENRICHMENT STORAGE
# ═══════════════════════════════════════════════════════════════

def store_enrichment(conn: sqlite3.Connection, node_id: str, vector_type: str,
                     text: str, embedding: bytes, model: str = 'snowflake-arctic-embed-m-v1.5'):
    """Store a single enrichment vector in node_enrichments table."""
    eid = uuid.uuid4().hex[:16]
    now = datetime.now(timezone.utc).isoformat()
    conn.execute(
        '''INSERT OR REPLACE INTO node_enrichments
           (id, node_id, vector_type, text, embedding, model, created_at)
           VALUES (?, ?, ?, ?, ?, ?, ?)''',
        (eid, node_id, vector_type, text, embedding, model, now)
    )


# ═══════════════════════════════════════════════════════════════
# MAIN BACKFILL
# ═══════════════════════════════════════════════════════════════

def backfill_enrichments(db_path: str, embedder: Embedder) -> Dict[str, Any]:
    """Backfill enrichment vectors for all nodes without them.

    Returns stats dict.
    """
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=30000")

    # Ensure table exists
    conn.execute("""CREATE TABLE IF NOT EXISTS node_enrichments (
        id TEXT PRIMARY KEY,
        node_id TEXT NOT NULL,
        vector_type TEXT NOT NULL CHECK(vector_type IN ('question', 'anchor', 'bridge', 'keywords')),
        text TEXT NOT NULL,
        embedding BLOB,
        model TEXT DEFAULT 'snowflake-arctic-embed-m',
        created_at TEXT DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (node_id) REFERENCES nodes(id) ON DELETE CASCADE
    )""")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_enrichments_node ON node_enrichments(node_id)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_enrichments_type ON node_enrichments(vector_type)")

    nodes = get_nodes_without_enrichments(conn)
    total = len(nodes)
    print(f"\n[backfill] {total} nodes need enrichments")

    stats = {
        'total_nodes': total,
        'enriched': 0,
        'by_type': {'question': 0, 'anchor': 0, 'bridge': 0, 'keywords': 0},
        'failures': 0,
        'empty_responses': 0,
        'total_time': 0,
        'embed_time': 0,
        'llm_time': 0,
    }

    t_start = time.time()

    for i, node in enumerate(nodes):
        t_node = time.time()

        # Get neighbors
        neighbors = get_neighbors(conn, node['id'], limit=5)

        # Build prompt
        prompt = build_prompt(node, neighbors)

        # Call LLM
        t_llm = time.time()
        raw = ollama_generate(prompt, timeout=30)
        stats['llm_time'] += time.time() - t_llm

        if not raw:
            stats['empty_responses'] += 1
            if (i + 1) % 50 == 0 or i == 0:
                print(f"  [{i+1}/{total}] {node['title'][:40]}... EMPTY RESPONSE")
            continue

        # Parse
        parsed = parse_enrichment(raw)
        if not parsed:
            stats['failures'] += 1
            if (i + 1) % 50 == 0 or i == 0:
                print(f"  [{i+1}/{total}] {node['title'][:40]}... PARSE FAILURE")
            continue

        # Embed and store each vector type
        texts_to_embed = []
        types_to_store = []
        for vtype in ('question', 'anchor', 'bridge', 'keywords'):
            if vtype in parsed:
                texts_to_embed.append(parsed[vtype])
                types_to_store.append(vtype)

        if texts_to_embed:
            t_embed = time.time()
            embeddings = embedder.embed_batch(texts_to_embed)
            stats['embed_time'] += time.time() - t_embed

            for vtype, text, emb in zip(types_to_store, texts_to_embed, embeddings):
                store_enrichment(conn, node['id'], vtype, text, emb)
                stats['by_type'][vtype] += 1

            conn.commit()
            stats['enriched'] += 1

        # Progress
        if (i + 1) % 50 == 0 or i == total - 1:
            elapsed = time.time() - t_start
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            eta = (total - i - 1) / rate if rate > 0 else 0
            print(f"  [{i+1}/{total}] enriched={stats['enriched']} "
                  f"Q={stats['by_type']['question']} A={stats['by_type']['anchor']} "
                  f"B={stats['by_type']['bridge']} K={stats['by_type']['keywords']} "
                  f"fail={stats['failures']} empty={stats['empty_responses']} "
                  f"({rate:.1f} nodes/s, ETA {eta:.0f}s)")

    stats['total_time'] = time.time() - t_start
    conn.close()
    return stats


def print_stats(stats: Dict[str, Any], label: str = ""):
    """Print enrichment stats summary."""
    print(f"\n{'=' * 60}")
    print(f"  ENRICHMENT BACKFILL {label}")
    print(f"{'=' * 60}")
    print(f"  Total nodes:      {stats['total_nodes']}")
    print(f"  Enriched:         {stats['enriched']}")
    print(f"  Failures:         {stats['failures']}")
    print(f"  Empty responses:  {stats['empty_responses']}")
    print(f"  By type:")
    for vtype, count in stats['by_type'].items():
        print(f"    {vtype:>10s}: {count}")
    print(f"  Total time:       {stats['total_time']:.1f}s")
    print(f"  LLM time:         {stats['llm_time']:.1f}s")
    print(f"  Embed time:       {stats['embed_time']:.1f}s")
    if stats['enriched'] > 0:
        avg = stats['total_time'] / stats['total_nodes'] if stats['total_nodes'] > 0 else 0
        print(f"  Avg per node:     {avg:.2f}s")
    print(f"{'=' * 60}\n")


# ═══════════════════════════════════════════════════════════════
# BENCHMARK
# ═══════════════════════════════════════════════════════════════

def run_benchmark(db_path: str) -> Dict[str, Any]:
    """Run golden dataset benchmark against the given DB via subprocess.

    Uses subprocess to avoid FastEmbed model registration conflicts
    between the standalone Embedder and the Brain's embedder module.
    """
    result = subprocess.run(
        [sys.executable, '-u', 'tests/eval_runner.py', db_path],
        capture_output=True,
        text=True,
        timeout=300,
        cwd=os.path.join(os.path.dirname(__file__), '..'),
    )
    print(result.stdout)
    if result.stderr:
        # Filter out just the important lines from stderr
        for line in result.stderr.split('\n'):
            if line.strip() and not line.startswith('/Users') and 'NotOpenSSLWarning' not in line:
                print(line, file=sys.stderr)

    # Parse results from the JSON report
    json_path = os.path.join(os.path.dirname(__file__), '..', 'tests', 'results', 'golden_eval.json')
    if os.path.exists(json_path):
        with open(json_path) as f:
            return json.load(f)
    return {'summary': {'passed': 0, 'total': 0}, 'aggregate': {}}


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    args = sys.argv[1:]
    apply_to_live = '--apply' in args
    live_only = '--live-only' in args
    benchmark_only = '--benchmark-only' in args

    if not os.path.exists(LIVE_DB):
        print(f"ERROR: Brain DB not found at {LIVE_DB}")
        sys.exit(1)

    # Load embedder
    print(f"[setup] Loading embedder from {MODEL_PATH}...")
    embedder = Embedder(MODEL_PATH)

    if benchmark_only:
        print("\n[benchmark] Running golden dataset on live DB...")
        result = run_benchmark(LIVE_DB)
        summary = result['summary']
        agg = result.get('aggregate', {})
        ndcg = agg.get('ndcg@10', {}).get('mean', 0)
        print(f"\nResult: NDCG@10={ndcg:.3f}, passed={summary['passed']}/{summary['total']}")
        return

    if not live_only:
        # Step 1: Work on temp copy
        print(f"\n[step 1] Copying brain.db to temp location...")
        from tests.brain_test_base import copy_brain_for_testing
        tmp_dir, tmp_db = copy_brain_for_testing(LIVE_DB)
        print(f"  Temp copy: {tmp_db}")

        # Step 2: Backfill on temp copy
        print(f"\n[step 2] Backfilling enrichments on temp copy...")
        stats = backfill_enrichments(tmp_db, embedder)
        print_stats(stats, "(TEMP COPY)")

        # Step 3: Benchmark temp copy
        print(f"\n[step 3] Running golden dataset benchmark on enriched temp copy...")
        result = run_benchmark(tmp_db)
        summary = result['summary']
        agg = result.get('aggregate', {})
        ndcg = agg.get('ndcg@10', {}).get('mean', 0)
        mrr_val = agg.get('mrr', {}).get('mean', 0)
        hit_rate = agg.get('hit_rate@10', {}).get('mean', 0)

        print(f"\n  TEMP COPY RESULTS:")
        print(f"    NDCG@10:    {ndcg:.3f}  (baseline: 0.204)")
        print(f"    MRR:        {mrr_val:.3f}")
        print(f"    Hit rate:   {hit_rate:.3f}")
        print(f"    Passed:     {summary['passed']}/{summary['total']}  (baseline: 34/104)")

        # Clean up temp
        shutil.rmtree(tmp_dir, ignore_errors=True)

        if not apply_to_live:
            print(f"\n[done] Temp copy validated. Run with --apply to apply to live DB.")
            return

    # Step 4: Apply to live DB
    print(f"\n[step 4] Backfilling enrichments on LIVE DB: {LIVE_DB}")
    stats = backfill_enrichments(LIVE_DB, embedder)
    print_stats(stats, "(LIVE DB)")

    # Step 5: Benchmark live DB
    print(f"\n[step 5] Running golden dataset benchmark on live DB...")
    result = run_benchmark(LIVE_DB)
    summary = result['summary']
    agg = result.get('aggregate', {})
    ndcg = agg.get('ndcg@10', {}).get('mean', 0)
    mrr_val = agg.get('mrr', {}).get('mean', 0)
    hit_rate = agg.get('hit_rate@10', {}).get('mean', 0)

    print(f"\n  LIVE DB RESULTS:")
    print(f"    NDCG@10:    {ndcg:.3f}  (baseline: 0.204)")
    print(f"    MRR:        {mrr_val:.3f}")
    print(f"    Hit rate:   {hit_rate:.3f}")
    print(f"    Passed:     {summary['passed']}/{summary['total']}  (baseline: 34/104)")


if __name__ == '__main__':
    main()
