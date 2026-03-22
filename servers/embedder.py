"""
brain — Embedding Engine (Model-Agnostic)

Generic ONNX embedding engine. Doesn't know or care which model it's running.
All model config (name, dimensions, pooling, paths) comes from the caller
via load_model(config). Defaults come from plugin.json.

ARCHITECTURE:
  - load_model(config) initializes FastEmbed with whatever model config is passed
  - embed() is synchronous (~50ms per text)
  - Embeddings stored as BLOB (dim × 4 bytes each)
  - Graceful degradation: if model fails to load, all paths fall back to TF-IDF
  - Phase 0.5A: Loud failure reporting — silent degradation is broken by design
"""

import struct
import time
import sys
import os
from typing import Optional, List, Dict, Any

# ─── Runtime State (set by load_model) ───
_model = None
_config = {}   # Current model config

stats = {
    'model_loaded': False,
    'model_name': None,
    'embedding_dim': None,
    'pooling': None,
    'load_time_ms': 0,
    'load_error': None,
    'total_embeddings': 0,
    'total_embed_time_ms': 0,
    'errors': 0,
    'last_embed_ms': 0,
    'peak_embed_ms': 0,
}


def _discover_pip_model() -> Optional[str]:
    """
    Auto-discover model files from pip-installed brain-embedding package.
    Returns model directory path if found, None otherwise.
    """
    try:
        import brain_embedding
        if brain_embedding.is_available():
            return brain_embedding.get_model_path()
    except ImportError:
        pass
    return None


def load_model(config: Optional[Dict[str, Any]] = None):
    """
    Load an embedding model from config.

    Config keys (all optional — defaults from plugin.json via brain_meta):
      model_name:  HuggingFace model ID, e.g. "Snowflake/snowflake-arctic-embed-m-v1.5"
      dim:         Embedding dimensions, e.g. 768
      pooling:     "cls" or "mean"
      model_file:  ONNX file within the repo, e.g. "onnx/model.onnx"
      model_path:  Local path override — skip download, load from here
      cache_dir:   Where to cache downloaded models

    Phase 0.5A: Loud failures. If this fails, the brain degrades to keyword-only
    recall which is fundamentally broken for semantic understanding.
    """
    global _model, _config

    if config is None:
        config = {}
    _config = config

    model_name = config.get('model_name', 'snowflake/snowflake-arctic-embed-m')
    dim = config.get('dim', 768)
    pooling = config.get('pooling', 'cls')
    model_file = config.get('model_file', 'onnx/model.onnx')
    model_path = config.get('model_path')  # Local path override
    cache_dir = config.get('cache_dir')

    stats['model_name'] = model_name
    stats['embedding_dim'] = dim
    stats['pooling'] = pooling

    # Auto-discover pip-installed model package (brain-embedding)
    if not model_path:
        model_path = _discover_pip_model()

    t0 = time.time()

    try:
        from fastembed import TextEmbedding
        from fastembed.common.model_description import PoolingType, ModelSource

        def _register_if_custom(name):
            """Register model with FastEmbed if not in its built-in list."""
            supported = [m['model'].lower() for m in TextEmbedding.list_supported_models()]
            if name.lower() not in supported:
                pooling_type = PoolingType.CLS if pooling == 'cls' else PoolingType.MEAN
                TextEmbedding.add_custom_model(
                    model=name,
                    pooling=pooling_type,
                    normalization=True,
                    sources=ModelSource(hf=name),
                    dim=dim,
                    model_file=model_file,
                )

        _register_if_custom(model_name)

        # Force CPU-only when ONNX_CPU_ONLY=1 (set by daemon).
        # CoreML/Metal causes SIGABRT in background processes on Apple Silicon.
        cpu_only = os.environ.get("ONNX_CPU_ONLY") == "1"
        provider_kwargs = {"providers": ["CPUExecutionProvider"]} if cpu_only else {}

        # ── Load chain: local path → HuggingFace cache → pip fallback ──

        if model_path:
            # Explicit local path — use directly, no network
            print(f"[embedder] Loading from local path: {model_path}", file=sys.stderr)
            _model = TextEmbedding(model_name=model_name,
                                   specific_model_path=model_path,
                                   **({"cache_dir": cache_dir} if cache_dir else {}),
                                   **provider_kwargs)
        else:
            # Try 1: Let FastEmbed load normally (from cache or HuggingFace)
            try:
                kwargs = {}
                if cache_dir:
                    kwargs['cache_dir'] = cache_dir
                _model = TextEmbedding(model_name=model_name, **kwargs, **provider_kwargs)
            except Exception as hf_err:
                # Try 2: HuggingFace failed — try pip-installed model package
                pip_path = _discover_pip_model()
                if pip_path:
                    print(f"[embedder] HuggingFace unavailable, using pip model: {pip_path}", file=sys.stderr)
                    _model = TextEmbedding(model_name=model_name,
                                           specific_model_path=pip_path,
                                           **provider_kwargs)
                else:
                    # Try 3: pip package not installed — install it and retry
                    # TODO(pypi): Remove auto-install once brain-embedding is on PyPI
                    # and boot.sh handles installation. Currently kept as last-resort
                    # fallback for Cowork environments where HuggingFace is blocked.
                    print(f"[embedder] HuggingFace blocked ({hf_err}). Attempting pip install brain-embedding...", file=sys.stderr)
                    import subprocess
                    result = subprocess.run(
                        [sys.executable, "-m", "pip", "install", "brain-embedding",
                         "--break-system-packages", "--quiet"],
                        capture_output=True, timeout=300)
                    pip_path = _discover_pip_model()
                    if pip_path:
                        print(f"[embedder] Installed model from pip: {pip_path}", file=sys.stderr)
                        _model = TextEmbedding(model_name=model_name,
                                               specific_model_path=pip_path)
                    else:
                        raise hf_err  # All fallbacks exhausted

        stats['load_time_ms'] = round((time.time() - t0) * 1000)
        stats['model_loaded'] = True
        stats['load_error'] = None
        source = "local" if model_path else ("pip" if _discover_pip_model() else "HuggingFace")
        print(f"[embedder] {model_name} ({dim}d, {pooling}) loaded in {stats['load_time_ms']}ms [source: {source}]", file=sys.stderr)

    except ImportError as e:
        stats['model_loaded'] = False
        stats['load_error'] = f"fastembed not installed: {e}"
        stats['errors'] += 1
        print(f"\n{'='*60}", file=sys.stderr)
        print(f"[embedder] CRITICAL: fastembed not installed!", file=sys.stderr)
        print(f"[embedder] Recall will degrade to keyword-only (BROKEN).", file=sys.stderr)
        print(f"[embedder] Fix: pip install fastembed --break-system-packages", file=sys.stderr)
        print(f"[embedder] Error: {e}", file=sys.stderr)
        print(f"{'='*60}\n", file=sys.stderr)

    except Exception as e:
        stats['model_loaded'] = False
        stats['load_error'] = str(e)
        stats['errors'] += 1
        print(f"\n{'='*60}", file=sys.stderr)
        print(f"[embedder] CRITICAL: Model failed to load!", file=sys.stderr)
        print(f"[embedder] Recall will degrade to keyword-only (BROKEN).", file=sys.stderr)
        print(f"[embedder] Error: {e}", file=sys.stderr)
        print(f"[embedder] Model: {model_name}", file=sys.stderr)
        print(f"[embedder] Tried: HuggingFace → pip package → all failed", file=sys.stderr)
        print(f"{'='*60}\n", file=sys.stderr)


def is_ready() -> bool:
    """Check if model is loaded and ready."""
    return stats['model_loaded'] and _model is not None


def get_model_status() -> str:
    """
    Phase 0.5A: Human-readable status with error details.
    """
    name = stats.get('model_name', '?')
    dim = stats.get('embedding_dim', '?')
    if is_ready():
        return f"READY: {name} ({dim}d, loaded in {stats['load_time_ms']}ms)"
    elif stats['load_error']:
        return f"FAILED: {stats['load_error']}"
    else:
        return "NOT LOADED: load_model() not called yet"


def get_config() -> Dict[str, Any]:
    """Return current embedder config."""
    return {**_config}


def get_dim() -> Optional[int]:
    """Return current embedding dimension, or None if not loaded."""
    return stats.get('embedding_dim')


def embed(text: str) -> Optional[bytes]:
    """
    Embed a single text → bytes (serialized float32 array).
    Returns None if model not loaded — callers MUST handle None gracefully.
    """
    if not _model:
        stats['errors'] += 1
        return None

    t0 = time.time()
    try:
        embeddings = list(_model.embed([text]))
        vec = embeddings[0]

        elapsed_ms = round((time.time() - t0) * 1000)
        stats['total_embeddings'] += 1
        stats['total_embed_time_ms'] += elapsed_ms
        stats['last_embed_ms'] = elapsed_ms
        if elapsed_ms > stats['peak_embed_ms']:
            stats['peak_embed_ms'] = elapsed_ms

        return _vec_to_blob(vec)
    except Exception as e:
        stats['errors'] += 1
        print(f"[embedder] Embed error: {e}", file=sys.stderr)
        return None


def embed_batch(texts: List[str]) -> List[Optional[bytes]]:
    """Batch-embed multiple texts → list of bytes."""
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
    For L2-normalized vectors, cosine = dot product.
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
    """Serialize numpy array or list of floats → bytes for SQLite BLOB."""
    try:
        return vec.astype('float32').tobytes()
    except AttributeError:
        return struct.pack(f'<{len(vec)}f', *vec)


def _blob_to_vec(blob: bytes) -> list:
    """Deserialize bytes → list of floats."""
    count = len(blob) // 4
    return list(struct.unpack(f'<{count}f', blob))


def compute_centroid(blobs: List[bytes]) -> Optional[bytes]:
    """Average N embedding blobs into a single centroid blob."""
    if not blobs:
        return None
    vecs = [_blob_to_vec(b) for b in blobs if b]
    if not vecs:
        return None
    dim = len(vecs[0])
    n = len(vecs)
    centroid = [sum(vecs[j][i] for j in range(n)) / n for i in range(dim)]
    return _vec_to_blob(centroid)


def get_stats() -> dict:
    """Snapshot of embedding engine stats."""
    return {
        **stats,
        'avg_embed_ms': (
            round(stats['total_embed_time_ms'] / stats['total_embeddings'])
            if stats['total_embeddings'] > 0 else 0
        ),
    }


def setup_sqlite_vec(conn):
    """
    Try to load sqlite-vec extension for KNN search.
    Falls back to brute-force cosine similarity if not available.
    """
    try:
        conn.enable_load_extension(True)
        for ext_path in [
            'vec0',
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
