#!/usr/bin/env python3
"""DUAL-STORE VALIDATION — four experiments validating the seed-and-spread simulation.

Validates the on-paper simulation in docs/RECALL-DUAL-STORE-DESIGN.md against the REAL
isolated brain. Each experiment has a PASS defined BEFORE running (the discipline that
caught a wrong causal claim last session). Never touches live (IsolatedBrain copies the DBs).

  A. PROVENANCE — for the 3 EX.CO queries, what does the episodic lane actually surface?
     The original episode (April) or a recent self-echo (within the 30-day embed window)?
     PASS: identify the provenance of the top hit. Prediction: recent echo (April is out of window).
  B. SEPARABILITY — top-trace cosine + sharpness for all 12 queries, EX.CO vs controls.
     PASS: a threshold τ separates EX.CO top-1 cosine from controls'. If overlapping, the
     cosine-only seed valve FAILS and needs a query-intent signal (not match-strength alone).
  C. EMBED CENSUS — what's in trace_embeddings: scale, date range, in-window count, April count.
     PASS: confirm only s0 embedded + the 30-day horizon + ~0 EX.CO-era (April) embeddings.
  D. SOURCE_REF COVERAGE NOW — coverage by created_at (pre/post v22 2026-05-26); reverse
     trace->node reachability; one structural traversal.
     PASS: post-v22 coverage >> pre-v22, and >=some in-window traces have node links.

Usage: ./dev python3 eval/oracle_audit/dual_store_validation_probe.py
"""
import sys, json
ROOT = '/Users/tpac/brain'
sys.path.insert(0, ROOT)
from tests.isolated_brain import IsolatedBrain  # noqa: E402

CORPUS = json.load(open(f'{ROOT}/eval/oracle_audit/meshed_top10.json'))
EXCO_RANKS = {2, 11, 12}            # cross-project + cross-session + entity EX.CO
WINDOW_CUTOFF = '2026-05-07'        # ~30 days before 2026-06-06 (TRACE_EMBED_WINDOW_DAYS=30)
V22_CUTOFF = '2026-05-26'           # source_refs teaching activated

with IsolatedBrain() as env:
    brain = env.brain
    from servers import embedder
    lc = brain.logs_conn
    bc = brain.conn

    # preload trace embeddings joined to their event metadata (one pass)
    trows = lc.execute("""
        SELECT te.trace_id, te.vector, te.text, te.created_at,
               ev.ref_type, ev.session_id, ev.scale
        FROM trace_embeddings te
        LEFT JOIN trace_events ev ON ev.id = te.trace_id
    """).fetchall()
    traces = [dict(trace_id=r[0], vec=r[1], text=r[2], created=r[3] or '',
                   ref_type=r[4], session=r[5], scale=r[6]) for r in trows if r[1]]

    def top_traces(query, k=10):
        qv = embedder.embed_query(query)
        scored = sorted(((embedder.cosine_similarity(qv, t['vec']), t) for t in traces),
                        key=lambda x: -x[0])
        return scored[:k]

    def q(rank):
        return next(it['prompt'] for it in CORPUS if it['rank'] == rank)

    # ============ EXPERIMENT A — provenance of episodic-lane hits ============
    print("\n" + "=" * 78)
    print("EXPERIMENT A — what does the episodic lane surface for EX.CO queries?")
    print("  (original April episode vs recent self-echo; window cutoff %s)" % WINDOW_CUTOFF)
    print("=" * 78)
    for rank in sorted(EXCO_RANKS):
        print("\n  #%d: %s" % (rank, q(rank)[:70]))
        for cos, t in top_traces(q(rank), 5):
            inwin = 'IN-WIN ' if t['created'][:10] >= WINDOW_CUTOFF else 'OUT-WIN'
            print("    cos=%.3f %s %s %-10s sess=%s  %s"
                  % (cos, inwin, t['created'][:10], (t['ref_type'] or '?')[:10],
                     (t['session'] or '?')[:8], (t['text'] or '').replace('\n', ' ')[:60]))

    # ============ EXPERIMENT B — seed separability ============
    print("\n" + "=" * 78)
    print("EXPERIMENT B — seed separability: top-trace cosine, EX.CO vs controls")
    print("  PASS: threshold separates EX.CO top-1 from controls' top-1")
    print("=" * 78)
    print("  %-5s %-7s %-8s %-8s %-8s %s" % ("rank", "grp", "top1", "top5", "gap", "prompt"))
    exco_top1, ctrl_top1 = [], []
    for it in CORPUS:
        rank = it['rank']
        scored = top_traces(it['prompt'], 5)
        top1 = scored[0][0]
        top5 = scored[-1][0]
        gap = top1 - top5
        grp = 'EXCO' if rank in EXCO_RANKS else 'ctrl'
        (exco_top1 if rank in EXCO_RANKS else ctrl_top1).append(top1)
        print("  #%-4d %-7s %-8.3f %-8.3f %-8.3f %s" % (rank, grp, top1, top5, gap, it['prompt'][:42]))
    import statistics as st
    print("\n  EX.CO  top1: min=%.3f mean=%.3f max=%.3f" %
          (min(exco_top1), st.mean(exco_top1), max(exco_top1)))
    print("  ctrl   top1: min=%.3f mean=%.3f max=%.3f" %
          (min(ctrl_top1), st.mean(ctrl_top1), max(ctrl_top1)))
    sep = min(exco_top1) > max(ctrl_top1)
    print("  SEPARABLE on top-1 cosine alone? %s  (EX.CO min %.3f vs ctrl max %.3f)"
          % ("YES" if sep else "NO", min(exco_top1), max(ctrl_top1)))

    # ============ EXPERIMENT C — embed census ============
    print("\n" + "=" * 78)
    print("EXPERIMENT C — embedded-trace census")
    print("=" * 78)
    total_ev = lc.execute("SELECT COUNT(*) FROM trace_events").fetchone()[0]
    total_emb = len(traces)
    print("  trace_events total: %d   |   trace_embeddings total: %d (%.1f%%)"
          % (total_ev, total_emb, 100.0 * total_emb / max(total_ev, 1)))
    from collections import Counter
    by_scale = Counter(t['scale'] for t in traces)
    print("  embedded by scale: %s" % dict(by_scale))
    by_ref = Counter(t['ref_type'] for t in traces)
    print("  embedded by ref_type: %s" % dict(by_ref))
    dates = sorted(t['created'][:10] for t in traces if t['created'])
    print("  embedded date range: %s .. %s" % (dates[0] if dates else '?', dates[-1] if dates else '?'))
    inwin = sum(1 for t in traces if t['created'][:10] >= WINDOW_CUTOFF)
    april = sum(1 for t in traces if t['created'][:10].startswith('2026-04'))
    print("  embedded in-window (>=%s): %d (%.1f%%)   |   embedded from April (EX.CO era): %d"
          % (WINDOW_CUTOFF, inwin, 100.0 * inwin / max(total_emb, 1), april))

    # ============ EXPERIMENT D — source_ref coverage NOW ============
    print("\n" + "=" * 78)
    print("EXPERIMENT D — source_ref coverage NOW + structural reachability")
    print("=" * 78)
    n_total = bc.execute("SELECT COUNT(*) FROM nodes WHERE archived=0").fetchone()[0]
    n_ref = bc.execute("""SELECT COUNT(DISTINCT node_id) FROM node_source_refs""").fetchone()[0]
    print("  nodes (active): %d   |   nodes with >=1 source_ref: %d (%.1f%%)"
          % (n_total, n_ref, 100.0 * n_ref / max(n_total, 1)))
    pre = bc.execute("""SELECT COUNT(DISTINCT nsr.node_id) FROM node_source_refs nsr
                        JOIN nodes n ON n.id = nsr.node_id WHERE n.created_at < ?""", (V22_CUTOFF,)).fetchone()[0]
    post = bc.execute("""SELECT COUNT(DISTINCT nsr.node_id) FROM node_source_refs nsr
                         JOIN nodes n ON n.id = nsr.node_id WHERE n.created_at >= ?""", (V22_CUTOFF,)).fetchone()[0]
    n_pre = bc.execute("SELECT COUNT(*) FROM nodes WHERE archived=0 AND created_at < ?", (V22_CUTOFF,)).fetchone()[0]
    n_post = bc.execute("SELECT COUNT(*) FROM nodes WHERE archived=0 AND created_at >= ?", (V22_CUTOFF,)).fetchone()[0]
    print("  pre-v22  (<%s): %d/%d nodes with refs (%.1f%%)"
          % (V22_CUTOFF, pre, n_pre, 100.0 * pre / max(n_pre, 1)))
    print("  post-v22 (>=%s): %d/%d nodes with refs (%.1f%%)"
          % (V22_CUTOFF, post, n_post, 100.0 * post / max(n_post, 1)))
    # reverse reachability: of embedded in-window traces, how many are referenced by a node?
    ref_trace_ids = set(r[0] for r in bc.execute("SELECT DISTINCT trace_id FROM node_source_refs").fetchall())
    emb_ids = set(t['trace_id'] for t in traces)
    reachable = emb_ids & ref_trace_ids
    print("  embedded traces that a node points at (structural chain LIVE): %d / %d embedded"
          % (len(reachable), len(emb_ids)))
    # one structural traversal
    if reachable:
        tid = next(iter(reachable))
        nodes = bc.execute("SELECT node_id FROM node_source_refs WHERE trace_id=?", (tid,)).fetchall()
        print("  sample: trace %s -> %d node(s): %s" % (tid, len(nodes), [n[0][:8] for n in nodes][:5]))
    else:
        print("  NO embedded trace is referenced by any node -> structural chain DEAD today (semantic chain only)")

    print("\n" + "=" * 78)
    print("DONE. Read each EXPERIMENT's PASS line against the header.")
    print("=" * 78)
