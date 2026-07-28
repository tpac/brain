"""Profile spread_activation in isolation — find where the 12s actually goes.

Runs spread_activation against the live brain (read path only — daemon
keeps writing) with timing on the four sub-operations inside
_build_edge_coeffs:

  1. SQL bulk fetch (`gdal.get_neighbors_bulk`)
  2. Per-row enriched text composition
  3. Cache lookup pass
  4. `_desc_vecs_batched` call (fastembed model inference)

Plus per-hop totals so we see whether one hop dominates.

Seed selection: take the 5 highest-activation node IDs from the most
recent surface-selected file (mirrors what surface.py would feed
_graph_expand). Falls back to a hand-picked set if not found.

Run:
    ./dev python3 scripts/profile_spread.py
        # uses live brain.db, latest seed set

    ./dev python3 scripts/profile_spread.py --runs 3
        # 3 cold runs to see variance
"""
from __future__ import annotations
import argparse
import glob
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from servers.scales.dispatch import load_env  # noqa: E402
load_env()
from servers.daemon_config import brain_tmp_dir  # noqa: E402


def _latest_surface_selected_seeds() -> list[str]:
    """Most recent surface-selected file → list of full node IDs."""
    # Honor BRAIN_TMP_DIR via brain_tmp_dir() so we read where the daemon WROTE
    # (matches the WRITER; default /tmp).
    paths = glob.glob(os.path.join(brain_tmp_dir(), 'brain-*-surface-selected.json'))
    if not paths:
        return []
    paths.sort(key=os.path.getmtime, reverse=True)
    with open(paths[0]) as f:
        d = json.load(f)
    short_ids = d.get('selected_ids', [])
    if not short_ids:
        return []
    # Resolve short IDs to full UUIDs via brain
    from servers.brain import Brain
    from servers.dal import NodeDAL
    brain = Brain.get_instance(
        os.path.join(os.environ.get('BRAIN_DB_DIR') or
                     os.path.expanduser('~/AgentsContext/brain'),
                     'brain.db'))
    ndal = NodeDAL(brain.conn)
    return [ndal.resolve_id(sid) for sid in short_ids if ndal.resolve_id(sid)]


def _instrumented_build_edge_coeffs(brain, brain_conn, activated_nodes,
                                     query_vec, rel_to_family,
                                     meaning_by_family, cached_edge_coeffs,
                                     timings):
    """Copy of _build_edge_coeffs with per-step timers feeding `timings`."""
    import os as _os
    import numpy as np
    from servers.dal_graph import GraphDAL
    from servers.scales.s1.surface_contract import (
        _compose_enriched_edge_text,
        _desc_vecs_batched,
        _cosine_nonneg,
        SPREAD_NEIGHBOR_LIMIT_DEFAULT,
    )

    gdal = GraphDAL(brain_conn)
    excluded = set(brain.aspects.traversal_exclusions)
    _SPREAD_LIMIT = int(_os.environ.get(
        'BRAIN_SPREAD_NEIGHBOR_LIMIT', str(SPREAD_NEIGHBOR_LIMIT_DEFAULT)))

    # Phase 1: SQL bulk fetch
    t = time.monotonic()
    bulk = gdal.get_neighbors_bulk(activated_nodes,
                                   exclude_relations=excluded)
    timings['sql_ms'] += int((time.monotonic() - t) * 1000)
    timings['sql_rows'] += sum(len(rs) for rs in bulk.values())

    # Phase 2: per-row enriched text composition + cache check
    edges_out = []
    enriched_to_embed = []
    enriched_keys = []
    t = time.monotonic()
    for source_id in activated_nodes:
        rows = bulk.get(source_id, [])
        if len(rows) > _SPREAD_LIMIT:
            rows = sorted(rows, key=lambda r: r.get('weight') or 0,
                          reverse=True)[:_SPREAD_LIMIT]
        for r in rows:
            target_id = r.get('id', '')
            enriched = _compose_enriched_edge_text(
                {'title': r.get('title', ''),
                 'relation': r.get('relation', ''),
                 'description': r.get('edge_description') or ''},
                rel_to_family, meaning_by_family)
            cached = cached_edge_coeffs.get(enriched)
            if cached is not None:
                edges_out.append((source_id, target_id, cached, r))
                timings['cache_hits'] += 1
            else:
                enriched_to_embed.append(enriched)
                enriched_keys.append((source_id, target_id, r, enriched))
                timings['cache_misses'] += 1
    timings['compose_ms'] += int((time.monotonic() - t) * 1000)
    timings['edges_total'] += len(edges_out) + len(enriched_to_embed)

    # Phase 3: batch embed (the suspected hot path)
    if enriched_to_embed:
        t = time.monotonic()
        blobs = _desc_vecs_batched(enriched_to_embed)
        timings['embed_ms'] += int((time.monotonic() - t) * 1000)
        timings['embed_n'] += len(enriched_to_embed)

        # Phase 4: cosine + cache fill
        t = time.monotonic()
        norm_q = float(np.linalg.norm(query_vec))
        for (src, tgt, edge, text), blob in zip(enriched_keys, blobs):
            if blob is None:
                coeff = 0.0
            else:
                vec = np.frombuffer(blob, dtype=np.float32)
                coeff = _cosine_nonneg(query_vec, vec, norm_a=norm_q)
            cached_edge_coeffs[text] = coeff
            edges_out.append((src, tgt, coeff, edge))
        timings['cosine_ms'] += int((time.monotonic() - t) * 1000)

    return edges_out


def run_one(brain, query_vec, seeds):
    """Run spread_activation with instrumented _build_edge_coeffs."""
    import numpy as np
    from servers.scales.s1.surface_contract import (
        _batch_load_field_vectors, _field_cosines_for_node,
        _SPREAD_MAX_STEPS, _SPREAD_NOISE_FLOOR, HOP_SCRUTINY_DEFAULT,
    )
    from servers.scales.s1 import surface_contract as sc

    timings = {
        'sql_ms': 0, 'sql_rows': 0,
        'compose_ms': 0, 'cache_hits': 0, 'cache_misses': 0, 'edges_total': 0,
        'embed_ms': 0, 'embed_n': 0,
        'cosine_ms': 0,
        'hops': 0,
    }

    # Build aspect map (small one-time cost)
    rel_to_family = {}
    meaning_by_family = {}
    for name, aspect in brain.aspects.all().items():
        if aspect.meaning:
            meaning_by_family[name] = aspect.meaning
        for r in aspect.edge_relations:
            rel_to_family[r] = name

    # Seed activations
    blended = query_vec
    norm_q = float(np.linalg.norm(blended))
    node_vectors = _batch_load_field_vectors(brain.conn, list(seeds))
    node_activation = {}
    for nid in seeds:
        vecs = node_vectors.get(nid, {})
        field_cos = _field_cosines_for_node(blended, vecs, norm_q=norm_q)
        node_activation[nid] = max(field_cos.values()) if field_cos else 0.0

    # Spread loop
    cached_edge_coeffs = {}
    t_total = time.monotonic()
    for step in range(_SPREAD_MAX_STEPS):
        active = [n for n, a in node_activation.items() if a > 0]
        if not active:
            break
        if step >= 2 and HOP_SCRUTINY_DEFAULT and len(active) > 4:
            sorted_acts = sorted(node_activation.values(), reverse=True)
            floor = sorted_acts[len(sorted_acts) // 2]
            active = [n for n in active if node_activation[n] >= floor]

        edges = _instrumented_build_edge_coeffs(
            brain, brain.conn, active, blended,
            rel_to_family, meaning_by_family, cached_edge_coeffs,
            timings)
        timings['hops'] += 1

        if not edges:
            break
        max_coeff = max(e[2] for e in edges)
        if max_coeff < _SPREAD_NOISE_FLOOR:
            break

        # Apply transmission (simplified — just to drive activation propagation)
        threshold = float(np.percentile([e[2] for e in edges], 50))
        for src, tgt, coeff, _edge in edges:
            if coeff >= threshold:
                cur = node_activation.get(tgt, 0.0)
                # simple accumulation; production uses tanh
                new = max(cur, node_activation[src] * coeff)
                node_activation[tgt] = new

    timings['total_ms'] = int((time.monotonic() - t_total) * 1000)
    timings['final_active'] = len(node_activation)
    return timings


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--runs', type=int, default=2)
    args = parser.parse_args()

    seeds = _latest_surface_selected_seeds()
    if not seeds:
        raise SystemExit('No surface-selected file. Trigger a recall first.')
    print(f'Seeds: {len(seeds)} — {[s[:8] for s in seeds]}')

    from servers.brain import Brain
    from servers.embedder import embed_query
    import numpy as np
    brain = Brain.get_instance(
        os.path.join(os.environ.get('BRAIN_DB_DIR') or
                     os.path.expanduser('~/AgentsContext/brain'),
                     'brain.db'))
    # Use a representative query vector — same shape as production
    blob = embed_query('what is the spread activation doing')
    query_vec = np.frombuffer(blob, dtype=np.float32)

    print('\nRunning spread_activation under the profiler...')
    for i in range(args.runs):
        print(f'\n=== run {i+1}/{args.runs} ===')
        t = run_one(brain, query_vec, seeds)
        print(f'  total:                 {t["total_ms"]:6d} ms')
        print(f'  hops:                  {t["hops"]:6d}')
        print(f'  final activated nodes: {t["final_active"]:6d}')
        print(f'  ─── per-hop sums ───')
        print(f'  SQL bulk fetch:        {t["sql_ms"]:6d} ms  ({t["sql_rows"]:5d} rows)')
        print(f'  compose+cache check:   {t["compose_ms"]:6d} ms  '
              f'(hits={t["cache_hits"]:5d} misses={t["cache_misses"]:5d})')
        print(f'  fastembed batch:       {t["embed_ms"]:6d} ms  '
              f'({t["embed_n"]:5d} edge texts)')
        print(f'  cosine+cache fill:     {t["cosine_ms"]:6d} ms')
        if t['total_ms'] > 0:
            sql_pct = t['sql_ms'] / t['total_ms'] * 100
            embed_pct = t['embed_ms'] / t['total_ms'] * 100
            print(f'  → SQL: {sql_pct:.1f}%   embed: {embed_pct:.1f}%')


if __name__ == '__main__':
    main()
