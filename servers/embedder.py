"""
brain — Embedding Engine

Thin wrapper around fastembed. No monkey-patches, no thread restrictions, no
custom-model registration. The plugin's isolated runtime pins
onnxruntime >= 1.20, which removes the old ARM64 threadpool race and makes
multi-threaded inference safe by default.

API:
  load_model(config)          — initialize fastembed from plugin.json config
  embed_document(text)        — embed a document (with model's doc prefix)
  embed_query(text)           — embed a query   (with model's query prefix)
  embed_batch(texts, kind)    — batch variant, kind ∈ {"document","query"}
  cosine_similarity(a, b)     — float, for L2-normalized vectors
  compute_centroid(blobs)     — average N blobs into one
  is_ready / get_stats / ...  — introspection

Callers must pick document vs query explicitly — the model needs it to produce
vectors in matched geometry. Mixing them silently collapses recall quality.
"""

import os
import sys
import time
import struct
from typing import Optional, List, Dict, Any


# ─── fastembed session-option polyfill ───────────────────────────
# fastembed 0.8 accepts `extra_session_options={...}` but its
# EXPOSED_SESSION_OPTIONS allowlist is hardcoded to ("enable_cpu_mem_arena",)
# and the only applied key is the same. We need two additional session config
# entries to prevent ORT WorkerLoop threads from busy-waiting idle CPU
# (onnxruntime issue microsoft/onnxruntime#9313):
#
#   session.intra_op.allow_spinning = 0
#   session.inter_op.allow_spinning = 0
#
# Without them, threads>1 pushes idle CPU to 250–290% on macOS ARM64. The
# keys are applied via SessionOptions.add_session_config_entry() — a
# different ORT API than fastembed's attribute-assignment code path.
#
# This polyfill extends EXPOSED_SESSION_OPTIONS and wraps
# add_extra_session_options to route spin keys through the correct ORT call.
# Intended as a temporary bridge — upstream PR against fastembed tracks
# this so the polyfill can be removed once merged.
def _install_fastembed_spin_polyfill() -> None:
    try:
        from fastembed.common.onnx_model import OnnxModel
    except ImportError:
        return

    SPIN_KEYS = (
        "session.intra_op.allow_spinning",
        "session.inter_op.allow_spinning",
    )
    exposed = set(OnnxModel.EXPOSED_SESSION_OPTIONS)
    if all(k in exposed for k in SPIN_KEYS):
        return  # fastembed merged equivalent upstream — no polyfill needed

    OnnxModel.EXPOSED_SESSION_OPTIONS = (
        tuple(OnnxModel.EXPOSED_SESSION_OPTIONS) + SPIN_KEYS
    )
    original_add = OnnxModel.add_extra_session_options  # bound classmethod

    def _patched_add(cls, session_options, extra_options):
        spin = {k: v for k, v in extra_options.items() if k in SPIN_KEYS}
        rest = {k: v for k, v in extra_options.items() if k not in SPIN_KEYS}
        if rest:
            original_add.__func__(cls, session_options, rest)
        for k, v in spin.items():
            session_options.add_session_config_entry(k, str(v))

    OnnxModel.add_extra_session_options = classmethod(_patched_add)


_install_fastembed_spin_polyfill()


# ─── Runtime state ───────────────────────────────────────────────
_model = None
_config: Dict[str, Any] = {}
_doc_prefix: str = ""
_query_prefix: str = ""

stats = {
    'model_loaded': False,
    'model_name': None,
    'embedding_dim': None,
    'load_time_ms': 0,
    'load_error': None,
    'total_embeddings': 0,
    'total_embed_time_ms': 0,
    'errors': 0,
    'last_embed_ms': 0,
    'peak_embed_ms': 0,
}


# ─── Prefix table ────────────────────────────────────────────────
# Models that require task-prefixed inputs. Keys are lowercased — lookup
# normalizes the configured model_name before checking. If your model isn't
# listed, prefixes default to empty strings (bge/e5/etc. don't need them).
_PREFIX_TABLE = {
    "nomic-ai/nomic-embed-text-v1.5":   ("search_document: ", "search_query: "),
    "nomic-ai/nomic-embed-text-v1.5-q": ("search_document: ", "search_query: "),
    "nomic-ai/nomic-embed-text-v1":     ("search_document: ", "search_query: "),
}


def load_model(config: Optional[Dict[str, Any]] = None) -> None:
    """Initialize fastembed. Idempotent on repeat calls with same config.

    Config keys (from plugin.json):
      model_name:  HuggingFace model ID (required)
      dim:         Embedding dimensions (for stats — fastembed knows the real dim)
      cache_dir:   Optional override for fastembed's model cache
    """
    global _model, _config, _doc_prefix, _query_prefix

    if config is None:
        config = {}

    # Singleton guard — don't reload on same config
    if _model is not None and config == _config:
        return
    _config = dict(config)

    model_name = config.get('model_name', 'nomic-ai/nomic-embed-text-v1.5-Q')
    dim = config.get('dim', 768)
    cache_dir = config.get('cache_dir')

    stats['model_name'] = model_name
    stats['embedding_dim'] = dim

    # Resolve prefixes for this model (case-insensitive lookup)
    _doc_prefix, _query_prefix = _PREFIX_TABLE.get(model_name.lower(), ("", ""))

    t0 = time.time()
    try:
        from fastembed import TextEmbedding

        kwargs: Dict[str, Any] = {}
        if cache_dir:
            kwargs['cache_dir'] = cache_dir

        # Disable ORT WorkerLoop spin-wait. Without this, threads>1 burns
        # idle CPU at 250–290% while waiting for work. Values routed through
        # our polyfill (see _install_fastembed_spin_polyfill above) until
        # fastembed exposes these keys upstream.
        kwargs['extra_session_options'] = {
            "session.intra_op.allow_spinning": "0",
            "session.inter_op.allow_spinning": "0",
        }

        _model = TextEmbedding(model_name=model_name, **kwargs)

        stats['load_time_ms'] = round((time.time() - t0) * 1000)
        stats['model_loaded'] = True
        stats['load_error'] = None
        print(
            f"[embedder] {model_name} ({dim}d) loaded in {stats['load_time_ms']}ms"
            + (f" [prefixes: doc={_doc_prefix!r} query={_query_prefix!r}]" if _doc_prefix or _query_prefix else ""),
            file=sys.stderr,
        )

    except ImportError as e:
        stats['model_loaded'] = False
        stats['load_error'] = f"fastembed not installed: {e}"
        stats['errors'] += 1
        print(f"[embedder] CRITICAL: fastembed not installed — recall broken ({e})", file=sys.stderr)

    except Exception as e:
        stats['model_loaded'] = False
        stats['load_error'] = str(e)
        stats['errors'] += 1
        print(f"[embedder] CRITICAL: model load failed — {e}", file=sys.stderr)


def is_ready() -> bool:
    return stats['model_loaded'] and _model is not None


def get_model_status() -> str:
    name = stats.get('model_name', '?')
    dim = stats.get('embedding_dim', '?')
    if is_ready():
        return f"READY: {name} ({dim}d, loaded in {stats['load_time_ms']}ms)"
    elif stats['load_error']:
        return f"FAILED: {stats['load_error']}"
    return "NOT LOADED: load_model() not called yet"


def get_config() -> Dict[str, Any]:
    return dict(_config)


def get_dim() -> Optional[int]:
    return stats.get('embedding_dim')


# ─── Core inference ──────────────────────────────────────────────

def _embed_one(text: str, prefix: str) -> Optional[bytes]:
    if not _model:
        stats['errors'] += 1
        return None

    t0 = time.time()
    try:
        vec = next(iter(_model.embed([prefix + text])))
        elapsed_ms = round((time.time() - t0) * 1000)
        stats['total_embeddings'] += 1
        stats['total_embed_time_ms'] += elapsed_ms
        stats['last_embed_ms'] = elapsed_ms
        if elapsed_ms > stats['peak_embed_ms']:
            stats['peak_embed_ms'] = elapsed_ms
        return _vec_to_blob(vec)
    except Exception as e:
        stats['errors'] += 1
        print(f"[embedder] embed error: {e}", file=sys.stderr)
        return None


def embed_document(text: str) -> Optional[bytes]:
    """Embed text for STORAGE. Applies the model's document prefix."""
    return _embed_one(text, _doc_prefix)


def embed_query(text: str) -> Optional[bytes]:
    """Embed text for SEARCH. Applies the model's query prefix.
    Mismatching document/query prefixes collapses cosine similarity — pick
    the right one for the call site.
    """
    return _embed_one(text, _query_prefix)


def embed_batch(texts: List[str], kind: str = "document") -> List[Optional[bytes]]:
    """Batch embed. `kind` is 'document' or 'query'."""
    if not _model or not texts:
        return []
    prefix = _query_prefix if kind == "query" else _doc_prefix
    prefixed = [prefix + t for t in texts]

    t0 = time.time()
    try:
        vecs = list(_model.embed(prefixed))
        results = [_vec_to_blob(v) for v in vecs]
        elapsed_ms = round((time.time() - t0) * 1000)
        stats['total_embeddings'] += len(texts)
        stats['total_embed_time_ms'] += elapsed_ms
        stats['last_embed_ms'] = round(elapsed_ms / len(texts)) if texts else 0
        if elapsed_ms > stats['peak_embed_ms']:
            stats['peak_embed_ms'] = elapsed_ms
        return results
    except Exception as e:
        stats['errors'] += 1
        print(f"[embedder] batch embed error: {e}", file=sys.stderr)
        return []


# ─── Math / serialization ────────────────────────────────────────

def cosine_similarity(a: bytes, b: bytes) -> float:
    """Cosine similarity between two embedding blobs.
    For L2-normalized vectors, cosine = dot product. fastembed normalizes
    outputs, so dot product is correct and ~100× faster than full cosine.
    """
    if not a or not b:
        return 0.0
    import numpy as np
    va = np.frombuffer(a, dtype=np.float32)
    vb = np.frombuffer(b, dtype=np.float32)
    if len(va) != len(vb):
        return 0.0
    return float(np.dot(va, vb))


def _vec_to_blob(vec) -> bytes:
    """Serialize embedding vector to float32 bytes. L2-normalizes so that
    cosine similarity reduces to a dot product downstream."""
    import numpy as np
    v = np.asarray(vec, dtype=np.float32)
    n = float(np.linalg.norm(v))
    if n > 0:
        v = v / n
    return v.tobytes()


def _blob_to_vec(blob: bytes) -> list:
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
    out = {
        **stats,
        'avg_embed_ms': (
            round(stats['total_embed_time_ms'] / stats['total_embeddings'])
            if stats['total_embeddings'] > 0 else 0
        ),
    }
    # Surface embed_queue drain stats — makes "my write isn't indexed"
    # debuggable without digging through logs.
    try:
        from . import embed_queue
        out['embed_queue'] = embed_queue.get_stats()
    except Exception:
        pass
    return out


def setup_sqlite_vec(conn) -> bool:
    """Try to load sqlite-vec extension for KNN search. Brute-force cosine if absent."""
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
                print("[embedder] sqlite-vec loaded", file=sys.stderr)
                return True
            except Exception:
                continue
        try:
            import sqlite_vec
            sqlite_vec.load(conn)
            print("[embedder] sqlite-vec loaded via python package", file=sys.stderr)
            return True
        except Exception:
            pass
        print("[embedder] sqlite-vec not available — using brute-force cosine", file=sys.stderr)
        return False
    except Exception as e:
        print(f"[embedder] sqlite-vec setup error: {e}", file=sys.stderr)
        return False
