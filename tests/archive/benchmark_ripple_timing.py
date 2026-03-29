#!/usr/bin/env python3
"""
Ripple Engine Timing Benchmark
===============================
Measures wall-clock timing of every step in the proposed ripple engine pipeline
to determine whether SYNC, ASYNC, or SPLIT encoding is viable within Claude Code
hook timing constraints.

Usage:
    python3 tests/benchmark_ripple_timing.py

Requirements:
    - /tmp/brain_timing_test.db (copied from live brain)
    - Ollama running with gemma2:2b
    - fastembed + ONNX model at model-package/brain_embedding/model
"""

import json
import os
import sqlite3
import statistics
import struct
import subprocess
import sys
import time
import traceback
from typing import Any, Dict, List, Optional, Tuple

# Force CPU-only ONNX
os.environ["ORT_DISABLE_ALL_ACCELERATORS"] = "1"
os.environ.setdefault("ONNX_PROVIDERS", "CPUExecutionProvider")
os.environ.setdefault("PYTORCH_MPS_DISABLE", "1")

# ─── Configuration ───

DB_PATH = "/tmp/brain_timing_test.db"
MODEL_PATH = os.path.expanduser("~/brain/model-package/brain_embedding/model")
OLLAMA_MODEL = "gemma2:2b"
OLLAMA_URL = "http://localhost:11434/api/generate"
NUM_TEST_NODES = 20
NUM_NEIGHBORS_SMALL = 3
NUM_NEIGHBORS_LARGE = 5
NUM_IMPACTED = 3  # Assume 3 of 5 neighbors need re-enrichment
RUNS_PER_STEP = 3  # Minimum runs for statistics


# ─── Helpers ───

def vec_to_blob(vec) -> bytes:
    """Convert numpy/list vector to bytes."""
    import numpy as np
    if hasattr(vec, 'tolist'):
        vec = vec.tolist()
    return struct.pack(f'{len(vec)}f', *vec)


def blob_to_vec(blob: bytes) -> list:
    """Convert bytes to float list."""
    n = len(blob) // 4
    return list(struct.unpack(f'{n}f', blob))


def cosine_similarity(a_blob: bytes, b_blob: bytes) -> float:
    """Cosine similarity between two embedding blobs."""
    a = blob_to_vec(a_blob)
    b = blob_to_vec(b_blob)
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(x * x for x in b) ** 0.5
    return dot / (na * nb) if na > 0 and nb > 0 else 0.0


def stats_summary(times_ms: List[float]) -> Dict[str, float]:
    """Compute min, max, mean, p50, p95 from a list of times in ms."""
    if not times_ms:
        return {"min": 0, "max": 0, "mean": 0, "p50": 0, "p95": 0}
    s = sorted(times_ms)
    p95_idx = max(0, int(len(s) * 0.95) - 1)
    return {
        "min": round(s[0], 2),
        "max": round(s[-1], 2),
        "mean": round(statistics.mean(s), 2),
        "p50": round(statistics.median(s), 2),
        "p95": round(s[p95_idx], 2),
    }


def ollama_generate(prompt: str, model: str = OLLAMA_MODEL) -> Tuple[str, float]:
    """Call Ollama API, return (response_text, elapsed_ms)."""
    import urllib.request
    t0 = time.perf_counter()
    data = json.dumps({
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"num_predict": 150, "temperature": 0.3}
    }).encode()
    req = urllib.request.Request(OLLAMA_URL, data=data,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        result = json.loads(resp.read())
    elapsed = (time.perf_counter() - t0) * 1000
    return result.get("response", ""), elapsed


# ─── Load Embedder ───

def load_embedder():
    """Load the Arctic v1.5 embedder, return the TextEmbedding instance."""
    from fastembed import TextEmbedding
    from fastembed.common.model_description import PoolingType, ModelSource

    model_name = "Snowflake/snowflake-arctic-embed-m-v1.5"
    dim = 768

    # Register custom model
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

    provider_kwargs = {"providers": ["CPUExecutionProvider"]}
    model = TextEmbedding(model_name=model_name,
                          specific_model_path=MODEL_PATH,
                          **provider_kwargs)
    return model


def embed_text(model, text: str) -> bytes:
    """Embed a single text, return blob."""
    vecs = list(model.embed([text]))
    return vec_to_blob(vecs[0])


# ─── Select Test Nodes ───

def select_test_nodes(conn: sqlite3.Connection, n: int = NUM_TEST_NODES) -> List[Dict]:
    """Pick diverse nodes: mix of types, with edges, with embeddings."""
    rows = conn.execute("""
        SELECT n.id, n.type, n.title, n.content, n.keywords
        FROM nodes n
        JOIN node_embeddings ne ON ne.node_id = n.id
        WHERE n.archived = 0
          AND n.content IS NOT NULL
          AND length(n.content) > 50
        ORDER BY RANDOM()
        LIMIT ?
    """, (n,)).fetchall()
    return [
        {"id": r[0], "type": r[1], "title": r[2], "content": r[3], "keywords": r[4]}
        for r in rows
    ]


# ─── Step Benchmarks ───

class RippleBenchmark:
    def __init__(self, conn: sqlite3.Connection, model):
        self.conn = conn
        self.model = model
        self.timings: Dict[str, List[float]] = {}

    def _record(self, step: str, ms: float):
        self.timings.setdefault(step, []).append(ms)

    # ── Step 1: Find related nodes ──

    def step1_neighbors_edges(self, node_id: str) -> List[Dict]:
        """Find neighbors via edges table (current approach)."""
        t0 = time.perf_counter()
        rows = self.conn.execute("""
            SELECT n.id, n.type, n.title, n.keywords, n.content, e.relation, e.weight
            FROM (
                SELECT target_id AS nid, relation, weight FROM edges WHERE source_id = ?
                UNION
                SELECT source_id AS nid, relation, weight FROM edges WHERE target_id = ?
            ) e
            JOIN nodes n ON n.id = e.nid
            WHERE n.archived = 0
            ORDER BY e.weight DESC
            LIMIT 5
        """, (node_id, node_id)).fetchall()
        elapsed = (time.perf_counter() - t0) * 1000
        self._record("1a_neighbors_edges", elapsed)
        return [{"id": r[0], "type": r[1], "title": r[2], "keywords": r[3],
                 "content": r[4], "relation": r[5], "weight": r[6]} for r in rows]

    def step1_neighbors_embedding(self, node_id: str) -> List[Dict]:
        """Find neighbors via embedding similarity (NEW for ripple)."""
        t0 = time.perf_counter()

        # Get this node's embedding
        row = self.conn.execute(
            "SELECT embedding FROM node_embeddings WHERE node_id = ?", (node_id,)
        ).fetchone()
        if not row:
            self._record("1b_neighbors_embedding", 0)
            return []

        query_vec = row[0]

        # Get all embeddings (brute-force scan — mirrors current recall)
        all_rows = self.conn.execute(
            "SELECT node_id, embedding FROM node_embeddings WHERE node_id != ?",
            (node_id,)
        ).fetchall()

        # Compute similarities
        scored = []
        for nid, emb in all_rows:
            sim = cosine_similarity(query_vec, emb)
            scored.append((nid, sim))

        scored.sort(key=lambda x: x[1], reverse=True)
        top5 = scored[:5]

        # Fetch node details for top 5
        results = []
        for nid, sim in top5:
            r = self.conn.execute(
                "SELECT id, type, title, keywords, content FROM nodes WHERE id = ?",
                (nid,)
            ).fetchone()
            if r:
                results.append({"id": r[0], "type": r[1], "title": r[2],
                                "keywords": r[3], "content": r[4], "sim": sim})

        elapsed = (time.perf_counter() - t0) * 1000
        self._record("1b_neighbors_embedding", elapsed)
        return results

    def step1_neighbors_both(self, node_id: str) -> Tuple[List, float]:
        """Find neighbors via edges + embedding, deduplicated."""
        t0 = time.perf_counter()
        edge_neighbors = self.step1_neighbors_edges(node_id)
        emb_neighbors = self.step1_neighbors_embedding(node_id)

        # Deduplicate
        seen = set(n["id"] for n in edge_neighbors)
        combined = list(edge_neighbors)
        for n in emb_neighbors:
            if n["id"] not in seen:
                combined.append(n)
                seen.add(n["id"])

        elapsed = (time.perf_counter() - t0) * 1000
        self._record("1c_neighbors_both", elapsed)
        return combined[:5], elapsed

    # ── Step 2: Build enrichment prompt ──

    def step2_build_prompt(self, node: Dict, neighbors: List[Dict]) -> str:
        """Format the enrichment prompt (string operations only)."""
        t0 = time.perf_counter()

        neighbor_lines = []
        for nb in neighbors[:5]:
            kw = nb.get("keywords", "") or ""
            kw_short = ", ".join(kw.split()[:5]) if kw else "none"
            neighbor_lines.append(
                f"- {nb['title'][:80]} ({nb['type']}, keywords: {kw_short})"
            )

        content_preview = (node.get("content") or "")[:200]
        prompt = f"""The brain found these related memories:
{chr(10).join(neighbor_lines)}

New node: "{node['title']}"
Content: "{content_preview}"

Generate exactly these lines, no explanations:
Q: [one question a user would naturally ask that leads to this node]
A: [3-5 word phrase using words from the neighbors above]
B: [one sentence connecting this node to its most important neighbor]
K: [5 comma-separated keywords borrowed from neighbors that also describe this node]"""

        elapsed = (time.perf_counter() - t0) * 1000
        self._record("2_build_prompt", elapsed)
        return prompt

    # ── Step 3: LLM generates Q/A/B/K for new node ──

    def step3_llm_enrichment(self, prompt: str) -> Tuple[str, float]:
        """Call Ollama to generate enrichments."""
        t0 = time.perf_counter()
        response, _ = ollama_generate(prompt)
        elapsed = (time.perf_counter() - t0) * 1000
        self._record("3_llm_enrichment", elapsed)
        return response, elapsed

    # ── Step 4: LLM generates impact assessment per neighbor ──

    def step4_impact_assessment(self, node: Dict, neighbor: Dict) -> Tuple[str, float]:
        """Call Ollama for one impact assessment."""
        prompt = f"""Given existing memory:
Title: {neighbor['title'][:80]}
Content: {(neighbor.get('content') or '')[:150]}

A new related memory was just stored:
Title: {node['title'][:80]}
Content: {(node.get('content') or '')[:150]}

Does the new memory VALIDATE, CONTRADICT, or EXTEND the existing one?
Reply with exactly one line: VALIDATES|CONTRADICTS|EXTENDS followed by a 10-word reason."""

        t0 = time.perf_counter()
        response, _ = ollama_generate(prompt)
        elapsed = (time.perf_counter() - t0) * 1000
        self._record("4_impact_assessment", elapsed)
        return response, elapsed

    def step4_batch(self, node: Dict, neighbors: List[Dict], count: int) -> float:
        """Run impact assessments for N neighbors, return total time."""
        t0 = time.perf_counter()
        for nb in neighbors[:count]:
            self.step4_impact_assessment(node, nb)
        elapsed = (time.perf_counter() - t0) * 1000
        self._record(f"4_batch_{count}_neighbors", elapsed)
        return elapsed

    # ── Step 5: LLM re-generates enrichments for impacted neighbors ──

    def step5_re_enrichment(self, neighbor: Dict, all_neighbors: List[Dict]) -> Tuple[str, float]:
        """Re-generate Q/A/B/K for an impacted neighbor."""
        prompt = self.step2_build_prompt(neighbor, all_neighbors)
        t0 = time.perf_counter()
        response, _ = ollama_generate(prompt)
        elapsed = (time.perf_counter() - t0) * 1000
        self._record("5_re_enrichment", elapsed)
        return response, elapsed

    # ── Step 6: Embed all new vectors ──

    def step6_embed_single(self, text: str) -> Tuple[bytes, float]:
        """Embed a single text, return (blob, ms)."""
        t0 = time.perf_counter()
        blob = embed_text(self.model, text)
        elapsed = (time.perf_counter() - t0) * 1000
        self._record("6_embed_single", elapsed)
        return blob, elapsed

    def step6_embed_batch(self, texts: List[str]) -> float:
        """Embed a batch of texts (e.g., 4 Q/A/B/K vectors), return total ms."""
        t0 = time.perf_counter()
        for text in texts:
            embed_text(self.model, text)
        elapsed = (time.perf_counter() - t0) * 1000
        self._record("6_embed_batch_4", elapsed)
        return elapsed

    def step6_new_node_vectors(self) -> float:
        """Embed 4 vectors (Q, A, B, K) for a new node."""
        sample_texts = [
            "What is the brain's approach to memory decay?",
            "memory decay activation stability",
            "This connects to the existing node about Hebbian learning and weight updates",
            "decay, activation, stability, memory, half-life"
        ]
        return self.step6_embed_batch(sample_texts)

    def step6_neighbor_re_embed(self, count: int) -> float:
        """Embed 4 vectors for each re-enriched neighbor."""
        t0 = time.perf_counter()
        sample_texts = [
            "How does this relate to the recall pipeline?",
            "recall pipeline embedding search",
            "This updates the connection to semantic search optimization",
            "recall, embedding, search, pipeline, optimization"
        ]
        for _ in range(count):
            for text in sample_texts:
                embed_text(self.model, text)
        elapsed = (time.perf_counter() - t0) * 1000
        self._record(f"6_re_embed_{count}_neighbors", elapsed)
        return elapsed

    # ── Step 7: Store everything in SQLite ──

    def step7_store_wal(self, node_id: str, num_enrichments: int = 4,
                        num_edges: int = 3) -> float:
        """Measure SQLite write time with WAL mode."""
        # Use a temporary copy to not pollute the test DB
        t0 = time.perf_counter()

        # Simulate writing enrichments
        for i in range(num_enrichments):
            self.conn.execute(
                """INSERT OR REPLACE INTO node_enrichments
                   (id, node_id, vector_type, text, embedding, model, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, datetime('now'))""",
                (f"bench_{node_id}_{i}", node_id, "question",
                 "benchmark test text", b'\x00' * 3072, "test", )
            )

        # Simulate writing edges
        for i in range(num_edges):
            self.conn.execute(
                """INSERT OR IGNORE INTO edges (source_id, target_id, relation, weight, created_at)
                   VALUES (?, ?, ?, ?, datetime('now'))""",
                (node_id, f"bench_target_{i}", "ripple_validates", 0.5)
            )

        # Simulate confidence update
        self.conn.execute(
            "UPDATE nodes SET confidence = confidence * 1.05 WHERE id = ?", (node_id,)
        )

        self.conn.commit()
        elapsed = (time.perf_counter() - t0) * 1000
        self._record("7_store_wal", elapsed)

        # Cleanup benchmark artifacts
        self.conn.execute("DELETE FROM node_enrichments WHERE id LIKE 'bench_%'")
        self.conn.execute("DELETE FROM edges WHERE source_id = ? AND target_id LIKE 'bench_%'",
                          (node_id,))
        self.conn.commit()

        return elapsed

    def step7_store_no_wal(self, node_id: str) -> float:
        """Measure SQLite write time WITHOUT WAL (journal_mode=delete)."""
        self.conn.execute("PRAGMA journal_mode=DELETE")
        elapsed = self.step7_store_wal(node_id)
        self.conn.execute("PRAGMA journal_mode=WAL")
        self._record("7_store_no_wal", elapsed)
        return elapsed

    # ── Full pipeline simulation ──

    def run_full_pipeline(self, node: Dict) -> Dict[str, float]:
        """Run the complete ripple pipeline for one node, return step timings."""
        timings = {}

        # Step 1: Find neighbors (both methods)
        t0 = time.perf_counter()
        neighbors, _ = self.step1_neighbors_both(node["id"])
        timings["step1"] = (time.perf_counter() - t0) * 1000

        if not neighbors:
            return timings

        # Step 2: Build prompt
        t0 = time.perf_counter()
        prompt = self.step2_build_prompt(node, neighbors)
        timings["step2"] = (time.perf_counter() - t0) * 1000

        # Step 3: LLM enrichment for new node
        t0 = time.perf_counter()
        enrichment_response, _ = self.step3_llm_enrichment(prompt)
        timings["step3"] = (time.perf_counter() - t0) * 1000

        # Step 4: Impact assessment for 3 and 5 neighbors
        t0 = time.perf_counter()
        self.step4_batch(node, neighbors, min(3, len(neighbors)))
        timings["step4_3nb"] = (time.perf_counter() - t0) * 1000

        if len(neighbors) >= 5:
            t0 = time.perf_counter()
            self.step4_batch(node, neighbors, 5)
            timings["step4_5nb"] = (time.perf_counter() - t0) * 1000

        # Step 5: Re-enrich impacted neighbors (assume 3)
        t0 = time.perf_counter()
        impacted = min(NUM_IMPACTED, len(neighbors))
        for nb in neighbors[:impacted]:
            self.step5_re_enrichment(nb, neighbors)
        timings["step5"] = (time.perf_counter() - t0) * 1000

        # Step 6: Embed new vectors
        t0 = time.perf_counter()
        self.step6_new_node_vectors()  # 4 vectors for new node
        self.step6_neighbor_re_embed(impacted)  # 4 vectors per impacted neighbor
        timings["step6"] = (time.perf_counter() - t0) * 1000

        # Step 7: Store
        t0 = time.perf_counter()
        self.step7_store_wal(node["id"])
        timings["step7"] = (time.perf_counter() - t0) * 1000

        # Total
        timings["total"] = sum(v for k, v in timings.items()
                               if k.startswith("step") and "_" not in k[4:6])

        return timings


# ─── Memory Measurement ───

def measure_memory():
    """Measure current process memory footprint."""
    import resource
    rusage = resource.getrusage(resource.RUSAGE_SELF)
    return {
        "rss_mb": round(rusage.ru_maxrss / (1024 * 1024), 1),  # macOS reports bytes
    }


# ─── Ollama Cold/Warm Start ───

def measure_ollama_cold_warm():
    """Measure Ollama first call (cold) vs subsequent calls (warm)."""
    print("\n--- Ollama Cold vs Warm Start ---")
    prompt = "Say hello in 5 words."

    # Cold start: unload model first
    try:
        import urllib.request
        data = json.dumps({"model": OLLAMA_MODEL, "keep_alive": 0}).encode()
        req = urllib.request.Request(OLLAMA_URL, data=data,
                                     headers={"Content-Type": "application/json"})
        urllib.request.urlopen(req, timeout=30)
        time.sleep(2)  # Wait for model to unload
    except Exception:
        pass

    # Cold call
    _, cold_ms = ollama_generate(prompt)
    print(f"  Cold start: {cold_ms:.0f} ms")

    # Warm calls
    warm_times = []
    for _ in range(5):
        _, ms = ollama_generate(prompt)
        warm_times.append(ms)

    print(f"  Warm calls (5x): min={min(warm_times):.0f} max={max(warm_times):.0f} "
          f"mean={statistics.mean(warm_times):.0f} ms")

    return {"cold_ms": round(cold_ms, 1), "warm": stats_summary(warm_times)}


# ─── Main ───

def main():
    print("=" * 70)
    print("RIPPLE ENGINE TIMING BENCHMARK")
    print("=" * 70)

    # Verify DB exists
    if not os.path.exists(DB_PATH):
        print(f"ERROR: {DB_PATH} not found. Copy brain.db first:")
        print(f"  cp ~/AgentsContext/brain/brain.db {DB_PATH}")
        sys.exit(1)

    # Connect to DB
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")

    node_count = conn.execute("SELECT COUNT(*) FROM nodes WHERE archived=0").fetchone()[0]
    edge_count = conn.execute("SELECT COUNT(*) FROM edges").fetchone()[0]
    embed_count = conn.execute("SELECT COUNT(*) FROM node_embeddings").fetchone()[0]
    print(f"\nDB: {node_count} nodes, {edge_count} edges, {embed_count} embeddings")

    # Load embedder
    print("\nLoading embedder...")
    t0 = time.perf_counter()
    model = load_embedder()
    load_ms = (time.perf_counter() - t0) * 1000
    print(f"  Embedder loaded in {load_ms:.0f} ms")

    # Warm up embedder (first call is slower due to ONNX session init)
    print("  Warming up embedder...")
    embed_text(model, "warmup text for ONNX session initialization")

    mem_after_embedder = measure_memory()
    print(f"  Memory after embedder: {mem_after_embedder['rss_mb']} MB")

    # Warm up Ollama
    print("\nWarming up Ollama...")
    ollama_generate("Say hello.")

    mem_after_both = measure_memory()
    print(f"  Memory after embedder + Ollama warm: {mem_after_both['rss_mb']} MB")

    # Select test nodes
    test_nodes = select_test_nodes(conn, NUM_TEST_NODES)
    print(f"\nSelected {len(test_nodes)} test nodes:")
    for n in test_nodes[:5]:
        print(f"  [{n['type']}] {n['title'][:60]}")
    if len(test_nodes) > 5:
        print(f"  ... and {len(test_nodes) - 5} more")

    # Create benchmark instance
    bench = RippleBenchmark(conn, model)

    # ── Run individual step benchmarks ──
    print("\n" + "=" * 70)
    print("INDIVIDUAL STEP TIMINGS")
    print("=" * 70)

    pipeline_timings = []

    for i, node in enumerate(test_nodes):
        print(f"\n  Node {i+1}/{len(test_nodes)}: [{node['type']}] {node['title'][:50]}...")
        pt = bench.run_full_pipeline(node)
        pipeline_timings.append(pt)

        total = sum(v for v in pt.values() if isinstance(v, (int, float)))
        print(f"    Total pipeline: {total:.0f} ms")
        for k, v in sorted(pt.items()):
            if k != "total":
                print(f"    {k}: {v:.0f} ms")

    # ── Ollama cold/warm ──
    cold_warm = measure_ollama_cold_warm()

    # ── DB write contention: WAL vs no-WAL ──
    print("\n--- DB Write Contention: WAL vs no-WAL ---")
    sample_node = test_nodes[0]["id"]
    wal_times = []
    no_wal_times = []
    for _ in range(5):
        conn.execute("PRAGMA journal_mode=WAL")
        wal_times.append(bench.step7_store_wal(sample_node))
        conn.execute("PRAGMA journal_mode=DELETE")
        no_wal_times.append(bench.step7_store_wal(sample_node))
    conn.execute("PRAGMA journal_mode=WAL")

    print(f"  WAL mode   (5x): {stats_summary(wal_times)}")
    print(f"  DELETE mode (5x): {stats_summary(no_wal_times)}")

    # ── Aggregate Results ──
    print("\n" + "=" * 70)
    print("AGGREGATE TIMING TABLE (milliseconds)")
    print("=" * 70)

    # Collect all step timings
    step_names = [
        "1a_neighbors_edges",
        "1b_neighbors_embedding",
        "1c_neighbors_both",
        "2_build_prompt",
        "3_llm_enrichment",
        "4_impact_assessment",
        "4_batch_3_neighbors",
        "4_batch_5_neighbors",
        "5_re_enrichment",
        "6_embed_single",
        "6_embed_batch_4",
        "6_re_embed_3_neighbors",
        "7_store_wal",
    ]

    print(f"\n{'Step':<30} {'Min':>8} {'Max':>8} {'Mean':>8} {'P50':>8} {'P95':>8} {'N':>5}")
    print("-" * 77)

    results_table = {}
    for step in step_names:
        times = bench.timings.get(step, [])
        if times:
            s = stats_summary(times)
            results_table[step] = s
            print(f"{step:<30} {s['min']:>8.1f} {s['max']:>8.1f} {s['mean']:>8.1f} "
                  f"{s['p50']:>8.1f} {s['p95']:>8.1f} {len(times):>5}")
        else:
            print(f"{step:<30} {'(no data)':>8}")

    # ── Pipeline Path Estimates ──
    print("\n" + "=" * 70)
    print("PIPELINE PATH ESTIMATES (milliseconds)")
    print("=" * 70)

    def mean_or(step, default=0):
        times = bench.timings.get(step, [])
        return statistics.mean(times) if times else default

    # Path A: SYNCHRONOUS — everything in remember()
    path_a_steps = {
        "1. Find neighbors (edges+emb)": mean_or("1c_neighbors_both"),
        "2. Build prompt": mean_or("2_build_prompt"),
        "3. LLM enrichment (new node)": mean_or("3_llm_enrichment"),
        "4. Impact assessment (3 nb)": mean_or("4_batch_3_neighbors"),
        "5. Re-enrichment (3 nb)": mean_or("5_re_enrichment") * 3,
        "6. Embed new + re-embed (16 vecs)": mean_or("6_embed_batch_4") + mean_or("6_re_embed_3_neighbors"),
        "7. Store all": mean_or("7_store_wal"),
    }
    path_a_total = sum(path_a_steps.values())

    print(f"\n  PATH A: SYNCHRONOUS (full pipeline in remember())")
    for desc, ms in path_a_steps.items():
        print(f"    {desc}: {ms:.0f} ms")
    print(f"    {'─' * 40}")
    print(f"    TOTAL: {path_a_total:.0f} ms ({path_a_total/1000:.1f}s)")

    # Path B: ASYNC — remember() stores node + embedding, rest in background
    # Single embed cost = batch_4 / 4 (embed_single had no separate data)
    single_embed_ms = mean_or("6_embed_batch_4") / 4
    path_b_sync = single_embed_ms + mean_or("7_store_wal") + mean_or("1a_neighbors_edges")  # SQL insert + 1 embed + TF-IDF ~5ms
    path_b_async = path_a_total - path_b_sync

    print(f"\n  PATH B: ASYNC (fire-and-forget)")
    print(f"    Sync part (node + embedding + store): {path_b_sync:.0f} ms")
    print(f"    Async part (ripple in background): {path_b_async:.0f} ms")
    print(f"    Claude waits: {path_b_sync:.0f} ms ({path_b_sync/1000:.1f}s)")

    # Path C: SPLIT — remember() stores node fast, returns prompt, Claude fills in
    path_c_call1 = (mean_or("1c_neighbors_both") + mean_or("2_build_prompt") +
                    mean_or("6_embed_single") + mean_or("7_store_wal"))
    path_c_call2_sync = mean_or("7_store_wal")  # Just store enrichments
    path_c_async = (mean_or("4_batch_3_neighbors") +
                    mean_or("5_re_enrichment") * 3 +
                    mean_or("6_re_embed_3_neighbors"))

    print(f"\n  PATH C: SPLIT (two-call protocol)")
    print(f"    Call 1 — remember() + build prompt: {path_c_call1:.0f} ms")
    print(f"    (Claude generates enrichments in next response — 0ms brain cost)")
    print(f"    Call 2 — store_enrichments(): {path_c_call2_sync:.0f} ms")
    print(f"    Async ripple (impact + re-enrich): {path_c_async:.0f} ms")
    print(f"    Claude waits total: {path_c_call1 + path_c_call2_sync:.0f} ms ({(path_c_call1 + path_c_call2_sync)/1000:.1f}s)")

    # ── Session Overhead Estimate ──
    print(f"\n  SESSION OVERHEAD (10 encodes per session):")
    print(f"    Path A: {path_a_total * 10 / 1000:.1f}s total Claude wait time")
    print(f"    Path B: {path_b_sync * 10 / 1000:.1f}s total Claude wait time")
    print(f"    Path C: {(path_c_call1 + path_c_call2_sync) * 10 / 1000:.1f}s total Claude wait time")

    # ── Memory Snapshot ──
    print(f"\n" + "=" * 70)
    print("MEMORY USAGE")
    print("=" * 70)
    final_mem = measure_memory()
    print(f"  After embedder load: {mem_after_embedder['rss_mb']} MB")
    print(f"  After embedder + Ollama: {mem_after_both['rss_mb']} MB")
    print(f"  After benchmark run: {final_mem['rss_mb']} MB")

    # ── Ollama Cold/Warm ──
    print(f"\n  Ollama cold start: {cold_warm['cold_ms']:.0f} ms")
    print(f"  Ollama warm mean: {cold_warm['warm']['mean']:.0f} ms")

    # ── Recommendation ──
    print(f"\n" + "=" * 70)
    print("RECOMMENDATION")
    print("=" * 70)

    if path_a_total < 3000:
        print(f"""
  Path A (synchronous) is viable at {path_a_total:.0f}ms — under 3s per encode.
  Simple, consistent, no race conditions. Recommended if these numbers hold.""")
    elif path_b_sync < 500:
        print(f"""
  Path B (async) recommended. Sync cost is only {path_b_sync:.0f}ms per encode.
  Ripple runs in daemon background thread — {path_b_async:.0f}ms invisible to Claude.
  Risk: immediate recall after encode may miss ripple results.
  Mitigation: 'ripple_pending' flag on node, cleared when done.""")
    else:
        print(f"""
  Path C (split) recommended. Call 1 costs {path_c_call1:.0f}ms, call 2 costs {path_c_call2_sync:.0f}ms.
  Claude generates enrichments naturally (free). Ripple is async ({path_c_async:.0f}ms).
  Downside: two-call protocol adds complexity.""")

    print(f"""
  Key insight: LLM calls dominate ({mean_or('3_llm_enrichment'):.0f}ms per enrichment,
  {mean_or('4_impact_assessment'):.0f}ms per impact). Embedding is cheap ({mean_or('6_embed_single'):.0f}ms).
  SQLite writes are negligible ({mean_or('7_store_wal'):.0f}ms).
  The bottleneck is Ollama inference — total of {int(1 + 3 + 3)} LLM calls per encode.
""")

    conn.close()
    print("Done.")


if __name__ == "__main__":
    main()
