#!/usr/bin/env python3
"""DUAL-STORE MERGE PROBE — Tier-1 test of the trace-chain as a parallel candidate source.

WHAT THIS IS (and is NOT): an OFFLINE simulation of the proposed merged pipeline, assembled in a
probe. It does NOT modify brain_recall.py. It answers: "if we run the trace-chain (query->trace->node)
in parallel with the real recall and merge into a reserved tail, does it rescue the buried EX.CO nodes
WITHOUT disturbing the controls?" — and produces the spec numbers (reserved-tail size, embedding mode,
how many traces to chain from) so we can spec the build from measurement, not assumption.

THE TOTAL PLAN this sits in (docs/RECALL-DUAL-STORE-DESIGN.md):
  recall = [direct semantic] + [fts5] + [trace-chain]  -> merge/dedup -> spread -> Haiku surfaces.
  This probe tests the RETRIEVAL tier only (does the right node enter the pool). The SELECTION tier
  (does Haiku pick it; do controls stay stable at selection level) is Tier 2 via frame_replay.py.
  The OUTCOME tier (does Anchor answer #11) is Tier 3 via the oracle/longmem answerer. Cheapest first.

THE TRAP WE MUST NOT REPEAT (burial session): the BASELINE must be the REAL brain.recall() pool, NOT
raw cosine. Raw cosine ranks the best EX.CO node at 3 (no burial); the real pipeline buries it via the
z-step/title-boost. Baselining against raw cosine would test the rescue against a pool that never
buried -> a fake win. So: baseline = brain.recall()'s actual returned results, in its own order.

THE SUBTLETY THE PAPER SIM HID: for #11 the rank-1 trace is the QUERY echoed back (a user_message whose
vector ~= the query -> chaining from it just re-runs the buried query = inert). The ANSWER trace
("your last EX.CO session was...") is rank-2. So we chain from TOP-T traces, not top-1, and report WHICH
trace drives each rescued node -> that measurement specs T.

DISCIPLINE: this is a reimplementation of the MERGE, but the BASELINE is the real pipeline. Trustworthy
for DIRECTION (spec), not for ship. Ship-gate is the real pipeline + control eval. Never touches live
(IsolatedBrain copies the DBs). Usage: ./dev python3 eval/oracle_audit/dual_store_merge_probe.py
"""
import sys, json
ROOT = '/Users/tpac/brain'
sys.path.insert(0, ROOT)
from tests.isolated_brain import IsolatedBrain  # noqa: E402

CORPUS = json.load(open(f'{ROOT}/eval/oracle_audit/meshed_top10.json'))
EXCO_RANKS = {2, 11, 12}                       # cross-project + cross-session + entity EX.CO queries
KNOWN_EXCO = {'e62cc595', 'dabb3078', 'af92b2cb', '30d88dd0', 'b3bda662', '5fe121db',
              '8359cf1d', '5410f4be', 'ef2f3276', '41d31ca5', '671d1f22', '598d78a8'}
POOL = 25            # the real recall candidate pool size (what the surfacer sees today)
T_TRACES = 5         # how many top traces we chain FROM (>1 because the answer trace may not be rank-1)
N_NODES = 25         # how many nodes each trace pulls (its own little ranked list)
K_SWEEP = [0, 3, 5, 8]   # reserved-tail sizes to sweep -> specs how many rescue slots to reserve

with IsolatedBrain() as env:
    brain = env.brain
    from servers import embedder
    lc, bc = brain.logs_conn, brain.conn

    # ---- preload node _primary vectors (the target of the trace->node hop) ----
    model = embedder.stats.get('model_name') or ''
    nrows = brain._vec_dal.get_all_vectors(vector_types=['_primary'], model=model or None)
    node_vecs = [(r['node_id'], r['embedding']) for r in nrows if r['embedding']]

    # ---- preload trace embeddings + their event metadata (for query->trace + HYGIENE) ----
    trows = lc.execute("""
        SELECT te.trace_id, te.vector, te.text, ev.ref_type, ev.session_id, ev.scale, te.model
        FROM trace_embeddings te LEFT JOIN trace_events ev ON ev.id = te.trace_id
    """).fetchall()
    # HYGIENE (decided): drop tool_result (the 82% poison incl. recall-echoes); keep only s0 dialogue.
    # We do NOT session-filter offline ("this-session" is undefined here) but we REPORT each driving
    # trace's session so eval-artifact sessions are visible rather than silently trusted.
    # column order: 0 trace_id, 1 vector, 2 text, 3 ref_type, 4 session_id, 5 scale, 6 model
    traces = [dict(tid=r[0], vec=r[1], text=r[2], ref=r[3], sess=r[4], model=r[6])
              for r in trows if r[1] and r[5] == 's0' and r[3] in ('user_message', 'assistant_message')]
    # GEOMETRY DIAGNOSTIC (verifier finding): node vectors are model-filtered; surface whether the trace
    # vectors share that model. If they diverge, cosine compares across geometries (silent 0.0 on dim
    # mismatch) and the rescue numbers are distorted. We print rather than hard-filter (a hard filter on a
    # mismatched model string would silently empty the set → a worse, invisible failure).
    _tmodels = set(t['model'] for t in traces)
    print("[geometry] node model=%r | distinct trace models=%r | aligned=%s"
          % (model, _tmodels, _tmodels <= {model}))

    def title(nid):
        t = bc.execute("SELECT title FROM nodes WHERE id=?", (nid,)).fetchone()
        return (t[0] if t else '?')[:38]

    def baseline_pool(query):
        """THE REAL PIPELINE. brain.recall() -> the genuinely buried pool, in its own order."""
        if hasattr(brain, '_recall_cache'):
            try: brain._recall_cache.clear()
            except Exception: pass
        out = brain.recall(query=query, limit=POOL)
        res = out.get('results', []) if isinstance(out, dict) else (out or [])
        return [(r.get('id') or r.get('node_id')) for r in res]

    def trace_chain(query, mode='stored'):
        """query -> top-T dialogue traces -> each trace's vector -> top-N nodes.
        mode='stored': use trace's stored doc-vector (free, symmetric doc<->doc) — the §8.4 decided path.
        mode='reembed': embed_query(trace_text) — the asymmetric path trace_as_vector_probe used.
        Returns (scored_traces, rescue_list). rescue_list = (nid, trace, tcos, ncos) ordered by the
        design's combined rescue score tcos*ncos (§4; a 1-hop chain has no hop_decay term). A node pulled
        by several of the top-T traces is attributed to its STRONGEST puller (argmax tcos*ncos), not the
        first trace that happened to reach it — so the driving-trace readout that specs T is honest."""
        qv = embedder.embed_query(query)
        scored_tr = sorted(((embedder.cosine_similarity(qv, t['vec']), t) for t in traces),
                           key=lambda x: -x[0])[:T_TRACES]
        best = {}   # nid -> (nid, trace, tcos, ncos), keeping the pairing with max tcos*ncos
        for tcos, t in scored_tr:
            tvec = t['vec'] if mode == 'stored' else embedder.embed_query((t['text'] or '')[:500])
            nn = sorted(((embedder.cosine_similarity(tvec, b), nid) for nid, b in node_vecs),
                        key=lambda x: -x[0])[:N_NODES]
            for ncos, nid in nn:
                if nid not in best or tcos * ncos > best[nid][2] * best[nid][3]:
                    best[nid] = (nid, t, tcos, ncos)
        out = sorted(best.values(), key=lambda x: -(x[2] * x[3]))   # design §4 combined-score order
        return scored_tr, out

    def q(rank):
        return next(it['prompt'] for it in CORPUS if it['rank'] == rank)

    # =================== EX.CO queries: does the chain RESCUE buried nodes? ===================
    print("\n" + "=" * 80)
    print("PART 1 — EX.CO RESCUE (does the trace-chain surface EX.CO nodes the real pipeline buried?)")
    print("=" * 80)
    rescue_by_k = {k: 0 for k in K_SWEEP}   # total EX.CO rescued across the 3 queries, per reserve size
    total_rescuable = 0                      # ceiling = EX.CO nodes not already in each baseline, summed
    for rank in sorted(EXCO_RANKS):
        base = baseline_pool(q(rank))
        base_set = set(base)
        base_exco = sorted(set(n[:8] for n in base if n[:8] in KNOWN_EXCO))
        rescuable = len(KNOWN_EXCO) - len(base_exco)   # how many EX.CO nodes the chain could still add
        total_rescuable += rescuable
        print("\n  #%d: %s" % (rank, q(rank)[:64]))
        print("    BASELINE (real recall) top-%d: %d/%d EX.CO in pool %s  (rescuable ceiling: %d)"
              % (POOL, len(base_exco), len(KNOWN_EXCO), base_exco, rescuable))
        scored_tr, chain = trace_chain(q(rank), 'stored')
        print("    driving traces (after hygiene):")
        for tcos, t in scored_tr:
            print("      tcos=%.3f %-10s sess=%s  %s" % (tcos, t['ref'], (t['sess'] or '?')[:8],
                                                         (t['text'] or '').replace('\n', ' ')[:52]))
        # rescue = chain nodes that are EX.CO AND not already in baseline (ordered by tcos*ncos)
        rescues = [(nid, t, tcos, ncos) for nid, t, tcos, ncos in chain
                   if nid[:8] in KNOWN_EXCO and nid not in base_set]
        print("    RESCUED EX.CO (not in baseline), by combined tcos*ncos:")
        for i, (nid, t, tcos, ncos) in enumerate(rescues[:8], 1):
            print("      r%-2d tcos*ncos=%.3f (t=%.2f n=%.2f) %s driven-by[%s] %s"
                  % (i, tcos * ncos, tcos, ncos, nid[:8], t['ref'], title(nid)))
        # re-embed comparison (does the asymmetric mode rescue more?)
        _, chain_re = trace_chain(q(rank), 'reembed')
        rescues_re = [nid for nid, t, tcos, ncos in chain_re if nid[:8] in KNOWN_EXCO and nid not in base_set]
        print("    rescued ids (stored): %s" % [nid[:8] for nid, t, tcos, ncos in rescues])
        print("    rescue count: stored-vec=%d  re-embed=%d  (of %d rescuable)"
              % (len(rescues), len(rescues_re), rescuable))
        for k in K_SWEEP:
            rescue_by_k[k] += len(rescues[:k])

    # =================== controls: does reserving K tail slots DISPLACE anything important? ===================
    print("\n" + "=" * 80)
    print("PART 2 — CONTROL SAFETY (reserving K=5 tail slots: what gets displaced, what gets inserted?)")
    print("=" * 80)
    K = 5
    ctrl_inserts = []   # available non-baseline inserts per control — feeds the MEASURED sweep in PART 3
    for it in CORPUS:
        if it['rank'] in EXCO_RANKS:
            continue
        base = baseline_pool(it['prompt'])
        base_set = set(base)
        _, chain = trace_chain(it['prompt'], 'stored')
        avail = [(nid, tcos * ncos) for nid, t, tcos, ncos in chain if nid not in base_set]
        ctrl_inserts.append(len(avail))
        inserts = avail[:K]
        # reserved tail FILLS ON DEMAND (like the fts5_only lane): only as many baseline-tail nodes are
        # displaced as there are inserts to place — empty reserved slots fall back to the semantic tail.
        n_disp = min(K, len(inserts), len(base))
        start = len(base) - n_disp
        displaced = base[start:] if n_disp else []
        print("\n  #%d: %s" % (it['rank'], it['prompt'][:52]))
        print("    DISPLACED (baseline ranks %d-%d, =min(K,inserts)=%d): %s"
              % (start + 1, len(base), n_disp, [(n[:8], title(n)) for n in displaced]))
        print("    INSERTED (trace-chain rescue): %s" % [(n[:8], title(n)) for n, _ in inserts])

    # =================== sweep summary: the spec falls out here ===================
    print("\n" + "=" * 80)
    print("PART 3 — RESERVED-TAIL SWEEP (the spec: rescue gained vs displacement cost)")
    print("=" * 80)
    print("  %-6s %-26s %s" % ("K", "EX.CO rescued (of %d)" % total_rescuable,
                                "ctrl nodes displaced (MEASURED, 9 controls)"))
    for k in K_SWEEP:
        disp = sum(min(k, ic) for ic in ctrl_inserts)   # fill-on-demand: real displacement, not k*9
        print("  %-6d %-26d %d" % (k, rescue_by_k[k], disp))
    print("\n  READ: pick the smallest K where EX.CO rescue saturates (ceiling %d); the MEASURED" % total_rescuable)
    print("  displacement at that K is the real control cost to inspect at Tier-2 (does Haiku still")
    print("  pick the controls' true answers given those tail swaps?).")
    print("=" * 80)
