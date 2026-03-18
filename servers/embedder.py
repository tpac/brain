"""
tmemory — Embedding Engine (v14 Serverless)

Replaces embedder.js (transformers.js + bge-m3) with:
  - FastEmbed (ONNX-based, snowflake-arctic-embed-m, 768d, ~110MB, <1s cold start)
  - sqlite-vec for KNN vector search (replaces brute-force O(n) cosine scan)

WHY FASTEMBED:
  FastEmbed uses ONNX runtime under the hood, same as transformers.js, but:
  - Much smaller model: 110MB vs 560MB
  - Much faster cold start: <1s vs 3-5s
  - Pure Python: no Node.js dependency, no background server to kill
  - snowflake-arctic-embed-m scores well on MTEB benchmarks for retrieval

WHY SQLITE-VEC:
  sqlite-vec is a SQLite extension for vector search. It:
  - Stores vectors in the same .db file (no separate vector DB)
  - Provides KNN search via virtual tables
  - Works with standard sqlite3 module
  - No server process needed

EMBEDDING DIMENSION CHANGE:
  bge-m3 = 1024d, snowflake-arctic-embed-m = 768d
  Existing embeddings must be re-computed on migration (migrate.py handles this).

ARCHITECTURE:
  - embed() is synchronous (FastEmbed is fast enough: ~50ms per text)
  - Store embeddings in node_embeddings table as BLOB (768 × 4 = 3072 bytes each)
  - KNN search via sqlite-vec virtual table for top-k retrieval
  - Graceful degradation: if FastEmbed fails to load, all paths fall back to TF-IDF
"""

import struct
import time
import sys
import os
from typing import Optional, List, Tuple

# ─── Model Configuration ───
MODEL_NAME = "snowflake/snowflake-arctic-embed-m"
EMBEDDING_DIM = 768

# ─── Stats Tracker ───
stats = {
    'model_loaded': False,
    'model_name': MODEL_NAME,
    'embedding_dim': EMBEDDING_DIM,
    'load_time_ms': 0,
    'total_embeddings': 0,
    'total_embed_time_ms': 0,
    'errors': 0,
    'last_embed_ms': 0,
    'peak_embed_ms': 0,
}

# Global model instance
_model = None


def load_model():
    """
    Load the FastEmbed model. Called once at brain init.
    FastEmbed downloads and caches models automatically (~110MB first time).
    Subsequent loads are from local cache (<1s).
    """
    global _model
    t0 = time.time()

    try:
        from fastembed import TextEmbedding
        _model = TextEmbedding(model_name=MODEL_NAME)
        stats['load_time_ms'] = round((time.time() - t0) * 1000)
        stats['model_loaded'] = True
        print(f"[embedder] Model loaded in {stats['load_time_ms']}ms", file=sys.stderr)
    except Exception as e:
        stats['model_loaded'] = False
        stats['errors'] += 1
        print(f"[embedder] FAILED to load model: {e}", file=sys.stderr)
        # Don't re-raise — brain works without embeddings (TF-IDF fallback)


def is_ready() -> bool:
    """Check if model is loaded and ready."""
    return stats['model_loaded'] and _model is not None


def embed(text: str) -> Optional[bytes]:
    """
    Embed a single text → bytes (serialized float32 array).

    Returns None if model not loaded — callers MUST handle None gracefully.
    Returns raw bytes suitable for SQLite BLOB storage and sqlite-vec.
    """
    if not _model:
        stats['errors'] += 1
        return None

    t0 = time.time()
    try:
        # FastEmbed returns a generator of numpy arrays
        embeddings = list(_model.embed([text]))
        vec = embeddings[0]  # numpy array, shape (768,)

        elapsed_ms = round((time.time() - t0) * 1000)
        stats['total_embeddings'] += 1
        stats['total_embed_time_ms'] += elapsed_ms
        stats['last_embed_ms'] = elapsed_ms
        if elapsed_ms > stats['peak_embed_ms']:
            stats['peak_embed_ms'] = elapsed_ms

        # Serialize to bytes: 768 floats × 4 bytes = 3072 bytes
        return _vec_to_blob(vec)
    except Exception as e:
        stats['errors'] += 1
        print(f"[embedder] Embed error: {e}", file=sys.stderr)
        return None


def embed_batch(texts: List[str]) -> List[Optional[bytes]]:
    """
    Batch-embed multiple texts → list of bytes.
    FastEmbed handles batching internally for efficiency.
    """
    if not _model or not texts:
        return []

    t0 = time.time()
    results = []
    try:
        embeddings = list(_model.embed(texts))
        for vec in embeddings:
            results.append(_vec_to_blob(vec))

        elapsed_ms = round((time.time() - t0) * 1000)
        stats['total_embeddings'] += len(texts)
        stats['total_embed_time_ms'] += elapsed_ms
        stats['last_embed_ms'] = round(elapsed_ms / len(texts)) if texts else 0
        if elapsed_ms > stats['peak_embed_ms']:
            stats['peak_embed_ms'] = elapsed_ms
    except Exception as e:
        stats['errors'] += 1
        print(f"[embedder] Batch embed error: {e}", file=sys.stderr)

    return results


def cosine_similarity(a: bytes, b: bytes) -> float:
    """
    Cosine similarity between two embedding blobs.
    For L2-normalized vectors (which snowflake-arctic-embed-m outputs),
    cosine = dot product. No sqrt needed.
    """
    if not a or not b:
        return 0.0
    va = _blob_to_vec(a)
    vb = _blob_to_vec(b)
    if len(va) != len(vb):
        return 0.0
    dot = sum(x * y for x, y in zip(va, vb))
    return dot


def _vec_to_blob(vec) -> bytes:
    """
    Serialize numpy array or list of floats → bytes for SQLite BLOB.
    768 floats × 4 bytes = 3072 bytes per embedding.
    Uses little-endian float32 (compatible with sqlite-vec).
    """
    try:
        # numpy array
        return vec.astype('float32').tobytes()
    except AttributeError:
        # plain list
        return struct.pack(f'<{len(vec)}f', *vec)


def _blob_to_vec(blob: bytes) -> list:
    """
    Deserialize bytes (from SQLite BLOB) → list of floats.
    """
    count = len(blob) // 4
    return list(struct.unpack(f'<{count}f', blob))


def get_stats() -> dict:
    """Snapshot of embedding engine stats for diagnostics."""
    return {
        **stats,
        'avg_embed_ms': (
            round(stats['total_embed_time_ms'] / stats['total_embeddings'])
            if stats['total_embeddings'] > 0 else 0
        ),
    }


def setup_sqlite_vec(conn):
    """
    Try to load the sqlite-vec extension for KNN search.
    If not available, falls back to brute-force cosine similarity.
    Returns True if sqlite-vec is available.
    """
    try:
        conn.enable_load_extension(True)
        # sqlite-vec installs as a loadable extension
        # Try common paths
        for ext_path in [
            'vec0',  # If installed system-wide
            '/usr/lib/sqlite3/vec0',
            '/usr/local/lib/sqlite3/vec0',
            os.path.expanduser('~/.local/lib/sqlite3/vec0'),
        ]:
            try:
                conn.load_extension(ext_path)
                print("[embedder] sqlite-vec loaded successfully", file=sys.stderr)
                return True
            except Exception:
                continue

        # Try loading via Python sqlite_vec package
        try:
            import sqlite_vec
            sqlite_vec.load(conn)
            print("[embedder] sqlite-vec loaded via Python package", file=sys.stderr)
            return True
        except Exception:
            pass

        print("[embedder] sqlite-vec not available — using brute-force cosine similarity", file=sys.stderr)
        return False
    except Exception as e:
        print(f"[embedder] sqlite-vec setup error: {e}", file=sys.stderr)
        return False
