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

import hashlib
import os
import shutil
import sys
import tempfile
import time
import struct
from typing import Optional, List, Dict, Any


# ─── fastembed session-option polyfill ───────────────────────────
# fastembed 0.8 accepts `extra_session_options={...}` but its
# EXPOSED_SESSION_OPTIONS allowlist hardcodes ("enable_cpu_mem_arena",) and
# its add_extra_session_options applies only that single key. We need three
# more session knobs:
#
#   session.intra_op.allow_spinning = 0     (config entry, prevents busy-wait)
#   session.inter_op.allow_spinning = 0     (config entry, prevents busy-wait)
#   enable_mem_pattern              = False (attribute, bounds ORT arena)
#
# Spin keys: without them, threads>1 pushes idle CPU to 250–290% on macOS
# ARM64 (onnxruntime#9313). Applied via add_session_config_entry().
#
# enable_mem_pattern: ORT's mem_pattern optimizer caches allocation patterns
# per distinct input tensor shape. With variable-length text inputs every
# recall produced a new (B, T) shape and the cache grew without bound (RSS
# 450 MB → 5+ GB observed). Disabling drops a small inference optimization
# (designed for fixed-shape models like CNNs) in exchange for a bounded
# arena. Set as a SessionOptions attribute, not a config entry.
#
# The polyfill extends EXPOSED_SESSION_OPTIONS so the new keys pass
# fastembed's allowlist assertion, and routes each key to the correct ORT
# API. Temporary — remove once fastembed exposes these upstream.
def _install_fastembed_spin_polyfill() -> None:
    try:
        from fastembed.common.onnx_model import OnnxModel
    except ImportError:
        return

    CONFIG_KEYS = (
        "session.intra_op.allow_spinning",
        "session.inter_op.allow_spinning",
    )
    ATTR_KEYS = (
        "enable_mem_pattern",
    )
    EXTRA_KEYS = CONFIG_KEYS + ATTR_KEYS
    exposed = set(OnnxModel.EXPOSED_SESSION_OPTIONS)
    if all(k in exposed for k in EXTRA_KEYS):
        return  # fastembed merged equivalent upstream — no polyfill needed

    OnnxModel.EXPOSED_SESSION_OPTIONS = (
        tuple(OnnxModel.EXPOSED_SESSION_OPTIONS) + EXTRA_KEYS
    )
    original_add = OnnxModel.add_extra_session_options  # bound classmethod

    def _patched_add(cls, session_options, extra_options):
        config = {k: v for k, v in extra_options.items() if k in CONFIG_KEYS}
        attrs = {k: v for k, v in extra_options.items() if k in ATTR_KEYS}
        rest = {k: v for k, v in extra_options.items()
                if k not in CONFIG_KEYS and k not in ATTR_KEYS}
        if rest:
            original_add.__func__(cls, session_options, rest)
        for k, v in config.items():
            session_options.add_session_config_entry(k, str(v))
        for k, v in attrs.items():
            setattr(session_options, k, v)

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


# ─── Artifact pinning ────────────────────────────────────────────
# The model is a DEPENDENCY OF THE STORED VECTORS: ~8.4k node embeddings (plus
# edge/trace substrates) are only comparable to fresh query embeddings if the
# artifact that produced them is byte-identical to the one serving queries.
# fastembed follows the HF repo's `main` ref with no pin — a silently
# refreshed artifact splits the vector space in two (operator ruling
# 2026-08-07). load_model therefore verifies the LOADED snapshot against the
# configured pins and refuses to serve on mismatch: a loudly dead embedder
# beats a silently split space.

_HEX = set('0123456789abcdef')


def _looks_hex(s: str, n: int) -> bool:
    return len(s) == n and all(c in _HEX for c in s.lower())


def _verify_artifact_pin(model_obj, config) -> tuple:
    """Verify the loaded snapshot against pinned_revision / pinned_onnx_sha256.

    Returns (revision, onnx_sha256) — recorded in stats either way.
    Raises RuntimeError on mismatch. No-op (returns what it can) when no pin
    is configured.
    """
    pinned_rev = config.get('pinned_revision')
    pinned_sha = config.get('pinned_onnx_sha256')

    inner = getattr(model_obj, 'model', None)
    model_dir = getattr(inner, '_model_dir', None)
    if model_dir is None:
        if pinned_rev or pinned_sha:
            raise RuntimeError(
                'embedder pin check: cannot locate loaded model dir '
                '(fastembed internals changed?) — refusing to serve unverified artifact')
        return None, None
    model_dir = str(model_dir)

    # HF hub cache layout: .../models--org--repo/snapshots/<revision>/...
    revision = os.path.basename(model_dir.rstrip('/'))
    if not _looks_hex(revision, 40):
        revision = None

    onnx_sha = None
    for root, _dirs, files in os.walk(model_dir):
        for f in sorted(files):
            if not f.endswith('.onnx'):
                continue
            target = os.path.realpath(os.path.join(root, f))
            base = os.path.basename(target)
            if _looks_hex(base, 64):        # hub blob filename IS the sha256
                onnx_sha = base
            else:                           # non-hub layout — hash the bytes
                h = hashlib.sha256()
                with open(target, 'rb') as fh:
                    for chunk in iter(lambda: fh.read(1 << 20), b''):
                        h.update(chunk)
                onnx_sha = h.hexdigest()
            break
        if onnx_sha:
            break

    problems = []
    if pinned_rev and revision != pinned_rev:
        problems.append(f'revision {revision!r} != pinned {pinned_rev!r}')
    if pinned_sha and onnx_sha != pinned_sha:
        problems.append(f'onnx sha256 {onnx_sha!r} != pinned {pinned_sha!r}')
    if problems:
        raise RuntimeError(
            'EMBEDDER ARTIFACT PIN MISMATCH — %s. The upstream model changed '
            'or the cache was tampered with. Serving it would embed new '
            'writes in a different space than the stored vectors. Either '
            'restore the pinned artifact or DELIBERATELY update the pins '
            '(set_embedder_config) and re-embed the corpus.' % '; '.join(problems))
    return revision, onnx_sha


def _seed_cache_from_tmp(model_name: str, cache_dir: str) -> None:
    """One-time migration helper: if the durable cache doesn't hold this model
    yet but fastembed's default $TMPDIR cache does, copy the tree over instead
    of re-downloading 137MB (and instead of failing on an offline boot).
    Best-effort — any failure falls through to fastembed's own download."""
    try:
        candidates = {model_name, model_name[:-2] if model_name.lower().endswith('-q') else model_name}
        tmp_cache = os.path.join(tempfile.gettempdir(), 'fastembed_cache')
        for cand in candidates:
            slug = 'models--' + cand.replace('/', '--')
            src = os.path.join(tmp_cache, slug)
            dst = os.path.join(cache_dir, slug)
            if os.path.isdir(src) and not os.path.isdir(dst):
                os.makedirs(cache_dir, exist_ok=True)
                shutil.copytree(src, dst, symlinks=True)
                print(f'[embedder] seeded durable cache from {src}', file=sys.stderr)
    except Exception as e:
        print(f'[embedder] cache seed skipped ({e}) — fastembed will download',
              file=sys.stderr)


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
            _seed_cache_from_tmp(model_name, cache_dir)
            kwargs['cache_dir'] = cache_dir

        # ORT session knobs. fastembed picks these up via
        # _select_exposed_session_options(kwargs) — i.e., flat kwargs
        # whose keys are in EXPOSED_SESSION_OPTIONS. Passing them inside
        # an `extra_session_options={...}` dict does NOT work — fastembed
        # has no such named param; the dict gets filtered out by name.
        # Our polyfill above extends EXPOSED_SESSION_OPTIONS for the spin
        # and mem_pattern keys.
        #
        #  - enable_cpu_mem_arena=False: ORT's arena pre-allocates per-shape
        #    slabs and never releases them — drives multi-GB RSS growth on
        #    long-running daemons with variable input shapes (fastembed
        #    issue #570, ORT issue #11627). This is THE leak fix; the
        #    other two are unrelated CPU-spin/memory hardening.
        #  - allow_spinning=0: prevent WorkerLoop busy-wait (idle CPU sink)
        #  - enable_mem_pattern=False: bound mem_pattern's per-shape cache
        #    (independent from the arena — both must be off)
        session_kwargs = {
            "enable_cpu_mem_arena": False,
            "session.intra_op.allow_spinning": "0",
            "session.inter_op.allow_spinning": "0",
            "enable_mem_pattern": False,
        }

        _model = TextEmbedding(model_name=model_name, **kwargs, **session_kwargs)

        # Artifact pin — raises on mismatch, failing the load LOUDLY. A dead
        # embedder is recoverable; a split vector space is silent corruption.
        revision, onnx_sha = _verify_artifact_pin(_model, config)
        stats['model_revision'] = revision
        stats['onnx_sha256'] = onnx_sha

        stats['load_time_ms'] = round((time.time() - t0) * 1000)
        stats['model_loaded'] = True
        stats['load_error'] = None
        print(
            f"[embedder] {model_name} ({dim}d) loaded in {stats['load_time_ms']}ms"
            + (f" [rev {revision[:8]}]" if revision else "")
            + (f" [prefixes: doc={_doc_prefix!r} query={_query_prefix!r}]" if _doc_prefix or _query_prefix else ""),
            file=sys.stderr,
        )

    except ImportError as e:
        stats['model_loaded'] = False
        stats['load_error'] = f"fastembed not installed: {e}"
        stats['errors'] += 1
        print(f"[embedder] CRITICAL: fastembed not installed — recall broken ({e})", file=sys.stderr)

    except Exception as e:
        _model = None    # never serve a model that failed load OR pin verify
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


# ─── Core inference ──────────────────────────────────────────────
#
# Variable-shape inputs are safe here because we set
# `enable_mem_pattern=False` at session creation (see load_model).
# With mem_pattern off, ORT does not cache allocation plans per
# distinct (batch, seq) shape and the arena stays bounded. No need
# to pad/bucket inputs.

def _embed_one(text: str, prefix: str) -> Optional[bytes]:
    if not _model:
        stats['errors'] += 1
        return None

    t0 = time.time()
    try:
        vecs = list(_model.embed([prefix + text]))
        elapsed_ms = round((time.time() - t0) * 1000)
        stats['total_embeddings'] += 1
        stats['total_embed_time_ms'] += elapsed_ms
        stats['last_embed_ms'] = elapsed_ms
        if elapsed_ms > stats['peak_embed_ms']:
            stats['peak_embed_ms'] = elapsed_ms
        return _vec_to_blob(vecs[0])
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


# Maximum texts per ORT inference call. Defensive cap so callers that
# build large lists (e.g., S1 spreading activation can produce thousands
# of unique enriched-edge texts) do not allocate a single (N, T) tensor
# that ORT must materialize at once. Peak working memory is bounded by
# this × max_seq_len × hidden_dim regardless of how many texts come in.
# 64 keeps the per-inference tensor small (a few hundred KB of input_ids,
# tens of MB of activations) while still amortizing call overhead.
_EMBED_BATCH_CHUNK = 64


def embed_batch(texts: List[str], kind: str = "document") -> List[Optional[bytes]]:
    """Batch embed. `kind` is 'document' or 'query'.

    Chunks the input into sub-batches of `_EMBED_BATCH_CHUNK` so peak
    memory stays bounded regardless of caller batch size.
    """
    if not _model or not texts:
        return []
    prefix = _query_prefix if kind == "query" else _doc_prefix
    n = len(texts)
    results: List[Optional[bytes]] = []

    t0 = time.time()
    try:
        for start in range(0, n, _EMBED_BATCH_CHUNK):
            chunk = [prefix + t for t in texts[start:start + _EMBED_BATCH_CHUNK]]
            vecs = list(_model.embed(chunk))
            results.extend(_vec_to_blob(v) for v in vecs)
        elapsed_ms = round((time.time() - t0) * 1000)
        stats['total_embeddings'] += n
        stats['total_embed_time_ms'] += elapsed_ms
        stats['last_embed_ms'] = round(elapsed_ms / n) if n else 0
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
    # Surface both queue drain stats — makes "my write isn't indexed"
    # and "my access marks aren't landing" debuggable without digging
    # through logs. recall_write_queue is the bg_writer queue for
    # access marks + Hebbian co-access (Phase 5, 2026-05-18).
    try:
        from . import embed_queue
        out['embed_queue'] = embed_queue.get_stats()
    except Exception as e:
        import sys as _sys
        print('[embedder] embed_queue.get_stats failed: %s' % e,
              file=_sys.stderr)
    try:
        from . import recall_write_queue
        out['recall_write_queue'] = recall_write_queue.get_stats()
    except Exception as e:
        import sys as _sys
        print('[embedder] recall_write_queue.get_stats failed: %s' % e,
              file=_sys.stderr)
    return out


