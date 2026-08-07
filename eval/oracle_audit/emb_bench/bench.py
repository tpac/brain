#!/usr/bin/env python3
"""Single-model worker for the embedding head-to-head.

Embeds the WHOLE node set (title+content = the brain's `_primary` blend) and all
73 cues with ONE model, ranks each cue by pure single-vector cosine over the
cutoff-masked candidate pool, and scores against teacher gold. Model-isolating:
nomic and every candidate run this identical pipeline — only the model differs.

Deliberately brain-package-independent (copies score_one rather than importing
servers/*) so the SAME file runs unchanged in a torch-only scratch venv for
Stage-2 candidates (Qwen3-Embedding, EmbeddingGemma, …).

Metrics:
  retrieval     hit@{1,5,10,25}, recall@5, nDCG@5, MRR (on ESSENTIAL gold)
  discrimination top1/top5cut/top25 cosine, spread (top1-top25), and gold
                separation (best-gold cosine, its margin vs the top non-gold,
                its z-score within the candidate cosines) — the flat-space lever
  cost          model load ms, single-query embed latency ms, peak RSS MB

Run (one model):   ./dev python3 eval/oracle_audit/emb_bench/bench.py nomic_q
Run (custom HF):   BENCH_BACKEND=st ./dev python3 .../bench.py <key>   (Stage 2)
"""
import json, os, sys, time, math, resource
import numpy as np


def _install_fastembed_spin_polyfill():
    """Extend fastembed's session-option allowlist so we can pass the same
    ORT knobs the brain's embedder.py uses. WITHOUT enable_mem_pattern=False,
    ORT caches an allocation plan per distinct (batch, seq_len) shape; on
    variable-length text this grows without bound (16 GB observed embedding 6k
    docs). No-op if fastembed isn't installed (Stage-2 torch venv)."""
    try:
        from fastembed.common.onnx_model import OnnxModel
    except ImportError:
        return
    CONFIG_KEYS = ("session.intra_op.allow_spinning", "session.inter_op.allow_spinning")
    ATTR_KEYS = ("enable_mem_pattern",)
    EXTRA = CONFIG_KEYS + ATTR_KEYS
    if all(k in set(OnnxModel.EXPOSED_SESSION_OPTIONS) for k in EXTRA):
        return
    OnnxModel.EXPOSED_SESSION_OPTIONS = tuple(OnnxModel.EXPOSED_SESSION_OPTIONS) + EXTRA
    original_add = OnnxModel.add_extra_session_options

    def _patched_add(cls, session_options, extra_options):
        config = {k: v for k, v in extra_options.items() if k in CONFIG_KEYS}
        attrs = {k: v for k, v in extra_options.items() if k in ATTR_KEYS}
        rest = {k: v for k, v in extra_options.items() if k not in EXTRA}
        if rest:
            original_add.__func__(cls, session_options, rest)
        for k, v in config.items():
            session_options.add_session_config_entry(k, str(v))
        for k, v in attrs.items():
            setattr(session_options, k, v)
    OnnxModel.add_extra_session_options = classmethod(_patched_add)


_install_fastembed_spin_polyfill()

DATA = "/tmp/emb_bench"
HERE = os.path.dirname(os.path.abspath(__file__))
OUTDIR = os.path.join(HERE, "results")
os.makedirs(OUTDIR, exist_ok=True)
KS = (1, 5, 10, 25)

# ── model registry: key -> (hf_name, doc_prefix, query_prefix, backend) ──
# Prefixes are each model's DOCUMENTED asymmetric retrieval prefixes, applied
# MANUALLY over plain .embed() (fastembed's query_embed/passage_embed do NOT
# replicate nomic's search_*: scheme — verified — so we control prefixes here).
BGE_Q = "Represent this sentence for searching relevant passages: "
MODELS = {
    # baseline — exactly the brain's production embedder
    "nomic_q":     ("nomic-ai/nomic-embed-text-v1.5-Q", "search_document: ", "search_query: ", "fastembed"),
    # un-quantized nomic — isolates whether quantization is costing discrimination
    "nomic_full":  ("nomic-ai/nomic-embed-text-v1.5",   "search_document: ", "search_query: ", "fastembed"),
    "bge_base":    ("BAAI/bge-base-en-v1.5",   "", BGE_Q, "fastembed"),
    "bge_large":   ("BAAI/bge-large-en-v1.5",  "", BGE_Q, "fastembed"),
    "gte_large":   ("thenlper/gte-large",      "", "",    "fastembed"),  # gte needs no prefix
    "arctic_l":    ("snowflake/snowflake-arctic-embed-l", "", BGE_Q, "fastembed"),
    "mxbai_large": ("mixedbread-ai/mxbai-embed-large-v1", "", BGE_Q, "fastembed"),
    # 2026 candidate (MTEB v2 Eng retrieval 57.0 vs nomic 48.0) — no prefixes
    "gte_modernbert": ("Alibaba-NLP/gte-modernbert-base", "", "", "fastembed"),
}

# Models absent from fastembed's built-in registry — registered on demand via
# add_custom_model. int8 ONNX to keep the fight fair against nomic-Q (both
# quantized); fp32 is "onnx/model.onnx" if a quantization ablation is wanted.
CUSTOM_MODELS = {
    "gte_modernbert": dict(
        hf="Alibaba-NLP/gte-modernbert-base", dim=768,
        pooling="CLS", model_file="onnx/model_int8.onnx",
    ),
}


def register_custom(key):
    """Idempotently register a CUSTOM_MODELS entry with fastembed. No-op for
    built-in models. Importable by sibling workers (geometry.py)."""
    spec = CUSTOM_MODELS.get(key)
    if not spec:
        return
    from fastembed import TextEmbedding
    from fastembed.common.model_description import ModelSource, PoolingType
    if any(m["model"] == spec["hf"] for m in TextEmbedding.list_supported_models()):
        return
    TextEmbedding.add_custom_model(
        model=spec["hf"],
        pooling=PoolingType[spec["pooling"]],
        normalization=True,
        sources=ModelSource(hf=spec["hf"]),
        dim=spec["dim"],
        model_file=spec["model_file"],
    )


# ── scoring (mirror of eval/oracle_audit/endo_baseline_recall.py score_one;
#    copied to keep this worker free of the brain package) ──
def _dcg(rels):
    return sum(r / math.log2(i + 2) for i, r in enumerate(rels))


def score_one(ranked_ids, ess, helpful):
    pos = {nid: i + 1 for i, nid in enumerate(ranked_ids)}
    ess_ranks = [pos[g] for g in ess if g in pos]
    best = min(ess_ranks) if ess_ranks else None
    top5 = set(ranked_ids[:5])
    m = {
        "best_ess_rank": best,
        "mrr_ess": (1.0 / best) if best else 0.0,
        "recall5_ess": (len(top5 & set(ess)) / len(ess)) if ess else None,
    }
    for k in KS:
        topk = set(ranked_ids[:k])
        m[f"hit{k}_ess"] = 1 if (topk & set(ess)) else 0
        m[f"hit{k}_any"] = 1 if (topk & (set(ess) | set(helpful))) else 0
    rel = {**{g: 2 for g in ess}, **{g: 1 for g in helpful if g not in ess}}
    gains = [rel.get(nid, 0) for nid in ranked_ids[:5]]
    ideal = sorted(rel.values(), reverse=True)[:5]
    m["ndcg5"] = (_dcg(gains) / _dcg(ideal)) if ideal and _dcg(ideal) > 0 else 0.0
    return m


# ── embedding backends — both return L2-normalized float32 (N, d) ──
def _normalize(mat):
    mat = np.asarray(mat, dtype=np.float32)
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return mat / norms


def _clamp_tokenizer_config(hf_name):
    """Some models (ModernBERT family) ship model_max_length as HF's
    VERY_LARGE_INTEGER sentinel; fastembed passes it to enable_truncation and
    Rust overflows. Clamp the cached tokenizer_config to the model's real
    max_position_embeddings. Returns True if anything was patched."""
    import glob, tempfile
    cache = os.environ.get("FASTEMBED_CACHE_PATH") or os.path.join(
        tempfile.gettempdir(), "fastembed_cache")
    slug = "models--" + hf_name.replace("/", "--")
    patched = False
    for tc_path in glob.glob(os.path.join(cache, slug, "snapshots", "*", "tokenizer_config.json")):
        tc = json.load(open(tc_path))
        mml = tc.get("model_max_length")
        if not isinstance(mml, int) or mml <= 1 << 30:
            continue
        cfg_path = os.path.join(os.path.dirname(tc_path), "config.json")
        real = 8192
        if os.path.exists(cfg_path):
            real = json.load(open(cfg_path)).get("max_position_embeddings", real)
        tc["model_max_length"] = real
        json.dump(tc, open(tc_path, "w"), indent=1)
        patched = True
    return patched


class FastEmbedBackend:
    def __init__(self, hf_name):
        from fastembed import TextEmbedding
        kwargs = dict(
            model_name=hf_name,
            threads=int(os.environ.get("BENCH_THREADS", "2")),  # bound ORT CPU — polite to daemon + other streams
            enable_cpu_mem_arena=False,
            enable_mem_pattern=False,   # THE bound — without it ORT's per-shape plan cache blew to 16GB on 6k docs
            **{"session.intra_op.allow_spinning": "0",
               "session.inter_op.allow_spinning": "0"},
        )
        try:
            self.m = TextEmbedding(**kwargs)
        except OverflowError:
            if not _clamp_tokenizer_config(hf_name):
                raise
            self.m = TextEmbedding(**kwargs)

    def embed(self, texts, prefix):
        # Length-aware budget batching. Attention memory is O(batch * max_seq^2),
        # and our docs span 40..7800 chars — a fixed batch of 64 (or even 8) of the
        # LONG docs spiked RSS to 6-45GB. Cap (batch_size * max_chars_in_batch) by a
        # char budget: the longest docs fall to batch=1 (~1GB peak, full content
        # kept), short docs batch large (fast). Identical per-doc embeddings — batch
        # composition never changes a vector. Indices realigned via `vecs[k]`.
        budget = int(os.environ.get("BENCH_CHAR_BUDGET", "10000"))
        maxb = int(os.environ.get("BENCH_MAXBATCH", "48"))
        order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
        vecs = [None] * len(texts)
        i = 0
        while i < len(order):
            j, maxlen, idxs = i, 0, []
            while j < len(order):
                L = len(texts[order[j]])
                if idxs and ((len(idxs) + 1) * max(maxlen, L) > budget or len(idxs) >= maxb):
                    break
                maxlen = max(maxlen, L); idxs.append(order[j]); j += 1
            for k, e in zip(idxs, self.m.embed([prefix + texts[k] for k in idxs])):
                vecs[k] = e
            i = j
        return _normalize(vecs)


class STBackend:
    """sentence-transformers backend for Stage-2 HF models (needs torch)."""
    def __init__(self, hf_name):
        from sentence_transformers import SentenceTransformer
        self.m = SentenceTransformer(hf_name, trust_remote_code=True, device="cpu")

    def embed(self, texts, prefix):
        vecs = self.m.encode([prefix + t for t in texts], batch_size=32,
                             show_progress_bar=False, normalize_embeddings=True)
        return _normalize(vecs)


def run(key):
    register_custom(key)
    hf_name, doc_prefix, query_prefix, backend = MODELS[key]
    nodes = json.load(open(os.path.join(DATA, "nodes.json")))
    cues = json.load(open(os.path.join(DATA, "cues.json")))

    node_ids = [n["id"] for n in nodes]
    created = np.array([n["created_at"] for n in nodes])           # ISO-T, lexically chronological
    docs = ["%s %s" % (n["title"], n["content"]) for n in nodes]   # _primary blend
    queries = [c["query"] for c in cues]

    t0 = time.time()
    Backend = STBackend if backend == "st" or os.environ.get("BENCH_BACKEND") == "st" else FastEmbedBackend
    model = Backend(hf_name)
    load_ms = round((time.time() - t0) * 1000)

    t1 = time.time()
    node_vecs = model.embed(docs, doc_prefix)                      # (N, d)
    doc_embed_s = round(time.time() - t1, 2)
    cue_vecs = model.embed(queries, query_prefix)                  # (M, d)
    dim = node_vecs.shape[1]

    # single-query latency: median over 15 fresh single embeds
    lat = []
    for q in queries[:15]:
        ts = time.time()
        model.embed([q], query_prefix)
        lat.append((time.time() - ts) * 1000)
    latency_ms = round(float(np.median(lat)), 1)

    scored, discrim = [], []
    for i, c in enumerate(cues):
        sims = node_vecs @ cue_vecs[i]                             # (N,)
        mask = created <= c["cutoff"]
        idx = np.where(mask)[0]
        order = idx[np.argsort(-sims[idx])]
        ranked_ids = [node_ids[j] for j in order]
        m = score_one(ranked_ids, c["gold_essential"], c.get("gold_helpful", []))
        m.update(source=c["source"], query_type=c["query_type"], id=c["id"])
        scored.append(m)

        # discrimination on the masked candidate pool
        cand_sims = sims[idx]
        ssort = np.sort(cand_sims)[::-1]
        gold_set = set(c["gold_essential"])
        gold_pos = [k for k, j in enumerate(idx) if node_ids[j] in gold_set]
        best_gold = float(np.max(cand_sims[gold_pos])) if gold_pos else None
        top1 = float(ssort[0])
        top5cut = float(ssort[4]) if len(ssort) > 4 else float(ssort[-1])
        top25 = float(ssort[24]) if len(ssort) > 24 else float(ssort[-1])
        mu, sd = float(np.mean(cand_sims)), float(np.std(cand_sims)) or 1e-9
        discrim.append({
            "top1": top1, "top5cut": top5cut, "top25": top25,
            "spread_1_25": top1 - top25,
            "best_gold": best_gold,
            "gold_margin_vs_top": (best_gold - top1) if best_gold is not None else None,
            "gold_z": ((best_gold - mu) / sd) if best_gold is not None else None,
            "n_cand": len(idx),
        })

    peak_rss_mb = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024), 0)

    def mean(rows, k):
        v = [r[k] for r in rows if r.get(k) is not None]
        return sum(v) / len(v) if v else 0.0

    agg = {
        "key": key, "hf_name": hf_name, "dim": dim, "backend": backend,
        "doc_prefix": doc_prefix, "query_prefix": query_prefix,
        "n_cues": len(cues), "n_nodes": len(nodes),
        "hit1": mean(scored, "hit1_ess"), "hit5": mean(scored, "hit5_ess"),
        "hit10": mean(scored, "hit10_ess"), "hit25": mean(scored, "hit25_ess"),
        "recall5": mean(scored, "recall5_ess"), "ndcg5": mean(scored, "ndcg5"),
        "mrr": mean(scored, "mrr_ess"), "hit5_any": mean(scored, "hit5_any"),
        # discrimination
        "mean_top1": mean(discrim, "top1"), "mean_top5cut": mean(discrim, "top5cut"),
        "mean_top25": mean(discrim, "top25"), "mean_spread_1_25": mean(discrim, "spread_1_25"),
        "mean_best_gold": mean(discrim, "best_gold"),
        "mean_gold_margin_vs_top": mean(discrim, "gold_margin_vs_top"),
        "mean_gold_z": mean(discrim, "gold_z"),
        # cost
        "load_ms": load_ms, "doc_embed_s": doc_embed_s, "latency_ms": latency_ms,
        "peak_rss_mb": peak_rss_mb,
    }
    json.dump({"agg": agg, "scored": scored, "discrim": discrim},
              open(os.path.join(OUTDIR, f"{key}.json"), "w"), indent=1)
    print(f"[{key}] dim={dim} hit@5={agg['hit5']:.0%} recall@5={agg['recall5']:.0%} "
          f"nDCG@5={agg['ndcg5']:.2f} | spread(1-25)={agg['mean_spread_1_25']:.3f} "
          f"gold_z={agg['mean_gold_z']:.2f} | {latency_ms}ms/q {peak_rss_mb:.0f}MB")
    return agg


if __name__ == "__main__":
    run(sys.argv[1] if len(sys.argv) > 1 else "nomic_q")
