#!/usr/bin/env python3
"""
Canary Benchmark — Fast subset of golden dataset for rapid iteration.

91 cases (37 positive + 54 negative):
  - 18 hardest positive failures (1 per category)
  - 19 positive regression canaries (1 per passing category)
  - ALL 54 negative cases (context bleed is P0)

Usage:
  # Against default test DB:
  BRAIN_DB_DIR=$HOME/AgentsContext/brain python3 tests/benchmark_canary.py

  # Against specific DB:
  BRAIN_TEST_DB=/tmp/brain_p0_test.db python3 tests/benchmark_canary.py

  # With custom relevance floor:
  RELEVANCE_FLOOR=0.83 python3 tests/benchmark_canary.py
"""

import os, sys, json, struct, math, time, sqlite3

os.environ["ORT_DISABLE_ALL_ACCELERATORS"] = "1"
os.environ.setdefault("ONNX_PROVIDERS", "CPUExecutionProvider")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
MODEL_PATH = os.path.join(PROJECT_ROOT, "model-package", "brain_embedding", "model")
CANARY_PATH = os.path.join(SCRIPT_DIR, "golden_canary.json")

# DB: explicit test DB > BRAIN_DB_DIR > default
TEST_DB = os.environ.get("BRAIN_TEST_DB")
if not TEST_DB:
    db_dir = os.environ.get("BRAIN_DB_DIR", os.path.join(os.path.expanduser("~"), "AgentsContext", "brain"))
    TEST_DB = os.path.join(db_dir, "brain.db")

RELEVANCE_FLOOR = float(os.environ.get("RELEVANCE_FLOOR", "0.80"))


def load_embedder():
    from fastembed import TextEmbedding
    from fastembed.common.model_description import PoolingType, ModelSource
    model_name = "Snowflake/snowflake-arctic-embed-m-v1.5"
    supported = [m['model'].lower() for m in TextEmbedding.list_supported_models()]
    if model_name.lower() not in supported:
        TextEmbedding.add_custom_model(
            model=model_name, pooling=PoolingType.CLS,
            normalization=True, sources=ModelSource(hf=model_name),
            dim=768, model_file="onnx/model.onnx",
        )
    return TextEmbedding(
        model_name=model_name,
        specific_model_path=MODEL_PATH,
        providers=["CPUExecutionProvider"],
    )


def embed_text(model, text):
    return list(model.embed([text]))[0].tolist()


def blob_to_vec(blob):
    n = len(blob) // 4
    return list(struct.unpack(f'{n}f', blob))


def cosine_sim(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    return dot / (na * nb) if na and nb else 0.0


def run():
    print(f"CANARY BENCHMARK — {CANARY_PATH}")
    print(f"DB: {TEST_DB}")
    print(f"RELEVANCE_FLOOR: {RELEVANCE_FLOOR}")
    print("=" * 70)

    with open(CANARY_PATH) as f:
        cases = json.load(f)
    print(f"Loaded {len(cases)} canary cases")

    t0 = time.time()
    model = load_embedder()
    print(f"Embedder loaded in {(time.time()-t0)*1000:.0f}ms")

    db = sqlite3.connect(TEST_DB)
    primary = {}
    for nid, blob in db.execute("SELECT node_id, embedding FROM node_embeddings").fetchall():
        if blob and len(blob) >= 4:
            primary[nid] = blob_to_vec(blob)

    enrichments = {}
    for eid, nid, vtype, blob in db.execute(
        "SELECT id, node_id, vector_type, embedding FROM node_enrichments WHERE embedding IS NOT NULL"
    ).fetchall():
        if blob and len(blob) >= 4:
            enrichments.setdefault(nid, []).append((eid, vtype, blob_to_vec(blob)))

    titles = {r[0]: r[1] for r in db.execute("SELECT id, title FROM nodes").fetchall()}
    db.close()

    n_primary = len(primary)
    n_enrichments = sum(len(v) for v in enrichments.values())
    print(f"DB: {n_primary} primary, {n_enrichments} enrichment vectors")

    # Run cases
    t_start = time.time()
    pos_pass = 0
    pos_total = 0
    neg_pass = 0
    neg_total = 0
    ndcg_values = []
    failures = []

    for case in cases:
        query = case['query']
        qvec = embed_text(model, query)

        scores = {}
        for nid, vec in primary.items():
            sim = cosine_sim(qvec, vec)
            scores[nid] = {'score': sim, 'won_by': 'primary'}

        for nid, enrich_list in enrichments.items():
            for eid, vtype, vec in enrich_list:
                sim = cosine_sim(qvec, vec)
                if nid not in scores or sim > scores[nid]['score']:
                    scores[nid] = {'score': sim, 'won_by': f'enrichment:{vtype}'}

        ranked = sorted(scores.items(), key=lambda x: -x[1]['score'])[:10]
        top_score = ranked[0][1]['score'] if ranked else 0.0

        # Relevance floor
        if top_score < RELEVANCE_FLOOR:
            ranked = []

        retrieved_ids = [nid for nid, _ in ranked]
        is_negative = 'max_acceptable_top_score' in case

        if is_negative:
            neg_total += 1
            passed = len(ranked) == 0 or top_score < case['max_acceptable_top_score']
            if passed:
                neg_pass += 1
            else:
                top_nid = ranked[0][0] if ranked else None
                failures.append({
                    'id': case['id'],
                    'type': 'neg',
                    'query': query[:40],
                    'score': top_score,
                    'title': titles.get(top_nid, '?')[:45] if top_nid else '?',
                    'won_by': ranked[0][1]['won_by'] if ranked else '?',
                })
        else:
            pos_total += 1
            expected = case.get('expected_relevant', {})
            rel_set = set(expected.keys())

            def dcg(ids, rel, k):
                s = 0.0
                for i, rid in enumerate(ids[:k]):
                    r = rel.get(rid, 0)
                    s += (2 ** r - 1) / math.log2(i + 2)
                return s

            actual = dcg(retrieved_ids, expected, 10)
            ideal = dcg(sorted(expected.keys(), key=lambda x: -expected[x]), expected, 10)
            ndcg = actual / ideal if ideal > 0 else 0.0
            ndcg_values.append(ndcg)

            hr = 1.0 if any(rid in rel_set for rid in retrieved_ids[:10]) else 0.0
            min_hr = case.get('min_hit_rate_at_10', 0.0)
            passed = hr >= min_hr

            if passed:
                pos_pass += 1
            else:
                failures.append({
                    'id': case['id'],
                    'type': 'pos',
                    'query': query[:40],
                    'score': top_score,
                    'title': titles.get(retrieved_ids[0], '?')[:45] if retrieved_ids else '(empty)',
                })

    elapsed = time.time() - t_start
    ndcg_mean = sum(ndcg_values) / len(ndcg_values) if ndcg_values else 0

    # Print results
    print(f"\n{'='*70}")
    print(f"RESULTS (FLOOR={RELEVANCE_FLOOR})")
    print(f"{'='*70}")
    print(f"Positive: {pos_pass}/{pos_total}")
    print(f"Negative: {neg_pass}/{neg_total} ({neg_pass/neg_total:.0%})")
    print(f"NDCG@10:  {ndcg_mean:.4f}")
    print(f"Total:    {pos_pass + neg_pass}/{pos_total + neg_total}")
    print(f"Time:     {elapsed:.1f}s ({elapsed/len(cases)*1000:.0f}ms/case)")

    if failures:
        print(f"\n--- Failures ({len(failures)}) ---")
        neg_f = [f for f in failures if f['type'] == 'neg']
        pos_f = [f for f in failures if f['type'] == 'pos']

        if neg_f:
            print(f"\nNegative ({len(neg_f)} context bleed):")
            for f in sorted(neg_f, key=lambda x: -x['score']):
                print(f"  {f['score']:.3f} {f['won_by']:<22} {f['title']:<45} {f['query']}")

        if pos_f:
            print(f"\nPositive ({len(pos_f)} missed):")
            for f in pos_f:
                print(f"  {f['id']:<35} {f['query']}")

    return {
        'pos_pass': pos_pass, 'pos_total': pos_total,
        'neg_pass': neg_pass, 'neg_total': neg_total,
        'ndcg': round(ndcg_mean, 4),
        'elapsed': round(elapsed, 1),
        'floor': RELEVANCE_FLOOR,
    }


if __name__ == '__main__':
    run()
