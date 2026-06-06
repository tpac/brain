#!/usr/bin/env python3
"""Burial-fix probe suite — empirically test 3 candidate fixes on an ISOLATED brain.

ALL ON THE SIDE: IsolatedBrain copies brain.db/brain_logs.db to a temp dir; z-score
stats are computed and applied on the COPY; no product code is edited and nothing is
written to the live brain. Phase-2 of docs/HANDOFF-RECALL-NORMALIZATION.md.

Arms (each scored on the SAME gate — surface ex.co AND don't move the controls):
  baseline   — current brain.recall() top-30 (what the surfacer sees today)
  bm25_resv  — ADDITIVE: reserve top-5 bm25-ranked fts5 hits into the candidate set
               (the bm25 signal already exists in Fts5DAL; recall flattens it to 0.20
                and cuts it — this stops discarding it). Dense top-25 + 5 lexical.
  zscore     — per-node contrastive re-rank z=(cos-mean)/std, stats computed on the COPY
               (tests whether the reverted "z-score is inert" verdict was a stale-stats artifact)
  decompose  — split a compositional query into anchors, retrieve each, merge by max-cosine
               (tests the representation axis without changing the embedding stack)

FALSIFICATION: an arm that surfaces ex.co but MOVES a brain-dev control's baseline top-5
is a FAIL (that is what killed the global RRF). We report where each arm LOSES.

Usage: ./dev python3 eval/oracle_audit/probe_suite.py
"""
import os, sys, json
ROOT = '/Users/tpac/brain'
sys.path.insert(0, ROOT)
from tests.isolated_brain import IsolatedBrain  # noqa: E402

CORPUS = json.load(open(f'{ROOT}/eval/oracle_audit/meshed_top10.json'))
KNOWN_EXCO = {'e62cc595', 'dabb3078', 'af92b2cb', '30d88dd0', 'b3bda662', '5fe121db',
              '8359cf1d', '5410f4be', 'ef2f3276', '41d31ca5', '671d1f22', '598d78a8'}
EXCO_SRCS = {'EXCO-recall', 'B1-20'}
PROD_LIMIT = 30   # surfacer candidate window (surface_contract max_candidates)

# De-primed: novel ex.co-domain phrasings NOT in the #11/#12/B1-20 wording.
# (Partial de-prime; fuller = random trace sampling — flagged in the report.)
DEPRIMED = [
    {'rank': 'D1', 'src': 'deprimed', 'prompt': 'how should we handle ad pods in our CTV product'},
    {'rank': 'D2', 'src': 'deprimed', 'prompt': 'what did we figure out about kevel'},
    {'rank': 'D3', 'src': 'deprimed', 'prompt': 'remind me how pacing logic is supposed to work'},
]
ALL_ITEMS = CORPUS + DEPRIMED
EXCO_RANKS = {it['rank'] for it in ALL_ITEMS if it['src'] in EXCO_SRCS or it['src'] == 'deprimed'}


def bust(b):
    if hasattr(b, '_recall_cache'):
        try: b._recall_cache.clear()
        except Exception: pass


def recall_full(brain, q, limit):
    """Return list of dicts {id, cos, title} in rank order."""
    bust(brain)
    out = brain.recall(query=q, limit=limit)
    res = out.get('results', []) if isinstance(out, dict) else (out or [])
    rows = []
    for r in res:
        nid = (r.get('id') or r.get('node_id') or '')[:8]
        rows.append({'id': nid,
                     'cos': r.get('embedding_similarity'),
                     'title': (r.get('title') or '')[:44]})
    return rows


def best_exco(ids):
    return next((i for i, x in enumerate(ids, 1) if x in KNOWN_EXCO), None)


def n_exco(ids):
    return sum(1 for x in ids if x in KNOWN_EXCO)


with IsolatedBrain() as env:
    brain = env.brain
    brain._ensure_structural_degree_cache()
    print("\n=== PROBE SUITE (isolated copy, %d nodes) ===" % env.node_count())

    # ---- precompute: baseline recalls (limit 100 so we have a pool to re-rank) ----
    pools = {}     # rank -> [rows]
    baseline30 = {}  # rank -> [ids] top-30
    for it in ALL_ITEMS:
        rows = recall_full(brain, it['prompt'], 100)
        pools[it['rank']] = rows
        baseline30[it['rank']] = [r['id'] for r in rows[:PROD_LIMIT]]

    # ---- z-score stats computed INLINE on the COPY ----
    # FINDING: compute_zscore_stats.py is STALE — it imports EmbeddingDAL and queries a
    # node_embeddings table, both removed in the refactor; it cannot run against the
    # current schema. That staleness is itself evidence per-node contrastive has been
    # untested since April. We recompute inline via the live VectorDAL.
    sys.path.insert(0, f'{ROOT}/scripts')
    from compute_zscore_stats import CALIBRATION_QUERIES
    from servers import embedder
    import statistics as _st
    print("\n--- computing per-node z-stats INLINE on the COPY (stale script bypassed) ---")
    _active_model = embedder.stats.get('model_name') or ''
    _qvecs = [embedder.embed_query(q) for q in CALIBRATION_QUERIES]
    _qvecs = [v for v in _qvecs if v]
    _vrows = brain._vec_dal.get_all_vectors(vector_types=['_primary'], model=_active_model or None)
    zstats = {}
    for r in _vrows:
        nid, blob = r['node_id'], r['embedding']
        if not blob:
            continue
        cos = [embedder.cosine_similarity(qv, blob) for qv in _qvecs]
        if cos:
            zstats[nid] = (_st.fmean(cos), max(_st.pstdev(cos), 0.01))
    print("  computed stats for %d nodes across %d calibration queries" % (len(zstats), len(_qvecs)))

    def zscore_rerank(rows):
        scored = []
        for r in rows:
            c = r['cos']
            if c is None:
                continue
            m, s = zstats.get(_full_id(r['id']), (0.50, 0.05))
            scored.append((((c - m) / max(s, 0.01)), r['id']))
        scored.sort(reverse=True)
        return [nid for _, nid in scored]

    # zstats keys are full ids; our rows carry 8-char. Build a prefix map.
    _full = {k[:8]: k for k in zstats.keys()}
    def _full_id(short): return _full.get(short, short)

    # ---- bm25 lexical reservation (additive) ----
    def bm25_resv(item):
        dense = baseline30[item['rank']][:25]
        try:
            lex = brain._fts.search(item['prompt'], 12) or []
        except Exception:
            lex = []
        lex = [h[:8] for h in lex]
        reserved = [x for x in lex if x not in dense][:5]
        return dense + reserved, reserved

    # ---- decomposition (ex.co-relevant queries only): {full, "ex.co"} merge by max cos ----
    def decompose(item):
        anchors = [item['prompt'], 'ex.co']
        best = {}  # id -> max cos
        title = {}
        for a in anchors:
            for r in recall_full(brain, a, 60):
                if r['cos'] is None:
                    continue
                if r['id'] not in best or r['cos'] > best[r['id']]:
                    best[r['id']] = r['cos']
                    title[r['id']] = r['title']
        ranked = sorted(best, key=lambda n: -best[n])
        return ranked[:PROD_LIMIT]

    # ============ RUN ALL ARMS ============
    print("\n%-5s %-9s %-5s | base | bm25 | zscr | deco  (best EX.CO rank in top-%d; '-'=miss)" %
          ("rank", "src", "kind", PROD_LIMIT))
    print("-" * 78)
    fails = []
    for it in ALL_ITEMS:
        k = it['rank']
        exco_q = k in EXCO_RANKS
        kind = 'EXCO' if exco_q else 'ctrl'
        b_ids = baseline30[k]
        be_base = best_exco(b_ids)

        # bm25 arm
        bm_ids, reserved = bm25_resv(it)
        be_bm = best_exco(bm_ids)

        # zscore arm
        zr = zscore_rerank(pools[k])[:PROD_LIMIT]
        be_z = best_exco(zr)

        # decompose arm (ex.co-relevant only)
        be_d = best_exco(decompose(it)) if exco_q else None

        # control regression gate: did bm25 / zscore move the baseline top-5?
        base5 = set(b_ids[:5])
        bm_ov = len(base5 & set(bm_ids[:5]))
        z_ov = len(base5 & set(zr[:5]))
        if kind == 'ctrl':
            if bm_ov < 5: fails.append(("bm25", k, "ctrl top5 overlap %d/5" % bm_ov))
            if z_ov < 5: fails.append(("zscore", k, "ctrl top5 overlap %d/5" % z_ov))

        def s(x): return str(x) if x else '-'
        print("%-5s %-9s %-5s | %-4s | %-4s | %-4s | %-4s" %
              (str(k), it['src'][:9], kind, s(be_base), s(be_bm), s(be_z), s(be_d)))

    # ============ FALSIFICATION + JUNK REPORT ============
    print("\n=== CONTROL REGRESSIONS (a fix that moves a control top-5 FAILS) ===")
    if not fails:
        print("  none — bm25 & zscore preserved all control top-5")
    else:
        for arm, k, why in fails:
            print("  FAIL %-7s turn %-4s : %s" % (arm, str(k), why))

    print("\n=== bm25 reservation — what it INJECTS on controls (junk check) ===")
    for it in ALL_ITEMS:
        if it['rank'] in EXCO_RANKS:
            continue
        _, reserved = bm25_resv(it)
        if reserved:
            titles = []
            for nid in reserved[:5]:
                row = next((r for r in pools[it['rank']] if r['id'] == nid), None)
                titles.append(nid)
            print("  %-4s injects: %s" % (str(it['rank']), titles))
