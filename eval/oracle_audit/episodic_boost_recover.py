#!/usr/bin/env python3
"""EPISODIC MATRIX-BOOST recovery test — does Tom's trace->node weight-boost recover the misses?

Tom's idea: the most relevant nodes were written when the operator said similar things, so
cosine the query against the OPERATOR's past utterances (trace_embeddings, ref_type=user_message),
then read out node weights through a trace->node association matrix M, and let that boost node
scores. Attention over episodic trace memory reading out node values: softmax(Q.Tu^T).M.

FAITHFUL recovery test on the 6 NEVER-RECALLED essentials. For each:

  1. Run the REAL production recall() (limit=200 -> 100-row pool). Confirm gold is absent.
  2. Read gold's REAL production relevance_score via a type-filtered recall (forces it into the
     scored output WITHOUT changing how it is scored) — this is the score it WOULD carry.
  3. Build the episodic boost from operator user_message traces:
        s[i]     = cosine(Q, operator_trace_i)         top-K seed traces
        boost[j] = sum_i s[i] * M[i,j]                  readout through assoc matrix M
     M[i,j] from (A) node_source_refs and (B) +/-W-day temporal association.
  4. Re-rank {pool candidates + gold}, all with REAL relevance_score, blended:
        score[j] = relevance_score[j] + lambda * boost_norm[j]
     Report whether gold enters top-K, WITH and WITHOUT IDF-normalization of M (the
     popularity-bias trap: without IDF, nodes near MANY traces dominate).

Hygiene: operator user_message traces only; this-session not present in isolated copy.
Daemon-safe: IsolatedBrain. Usage: ./dev python3 eval/oracle_audit/episodic_boost_recover.py
"""
import sys, math
ROOT = '/Users/tpac/brain'
sys.path.insert(0, ROOT)
from tests.isolated_brain import IsolatedBrain  # noqa: E402

SIX = [
    ('TR6', 'f58e9b12', "let's clean up the old backups"),
    ('TO4', '4b35293c', "how does the fatigue thing work?"),
    ('TO5', '51ff0a92', "what's the real value you add over plain Claude?"),
    ('TO6', 'e49766ac', "how are the scouts built — do their prompts have examples?"),
    ('HV6', '94f6e01a', "what have we already tried and ruled out on recall burial?"),
    ('EP4', '90e27c77', "what did we conclude early on about similar_to edges and old artifacts"),
]

TOPK_TRACES = 25       # top operator traces seeding the readout
WINDOW_DAYS = 2.0      # temporal-association half-window (Tom's ±W)
NODE_TOPK   = 25       # production surface horizon (~25 candidates per turn)
LAMBDAS     = [0.25, 0.5, 1.0]


def daysecs(iso):
    if not iso:
        return None
    s = str(iso)[:10]
    try:
        return int(s[:4]) * 365.0 + int(s[5:7]) * 30.4375 + int(s[8:10])
    except Exception:
        return None


with IsolatedBrain() as env:
    brain = env.brain
    from servers import embedder
    lc, bc = brain.logs_conn, brain.conn

    # operator user_message traces only
    rows = lc.execute(
        "SELECT te.trace_id, te.vector, te.text, ev.created_at "
        "FROM trace_embeddings te JOIN trace_events ev ON ev.id = te.trace_id "
        "WHERE ev.ref_type = 'user_message'").fetchall()
    op_traces = [(tid, vec, txt, ca) for (tid, vec, txt, ca) in rows]
    print("operator user_message traces (embedded): %d\n" % len(op_traces))

    # node created/revised days for temporal association (ACTIVE nodes only — the live pool)
    node_days = {}
    for nid, ca, rev in bc.execute(
            "SELECT id, substr(created_at,1,10), substr(revised_at,1,10) "
            "FROM nodes WHERE archived = 0").fetchall():
        ds = [d for d in (daysecs(ca), daysecs(rev)) if d is not None]
        node_days[nid] = ds

    # source_refs reverse index
    trace_to_nodes = {}
    for tid, nid in bc.execute("SELECT trace_id, node_id FROM node_source_refs").fetchall():
        trace_to_nodes.setdefault(tid, set()).add(nid)

    def resolve(prefix):
        r = bc.execute("SELECT id, type, archived FROM nodes WHERE id LIKE ?",
                       (prefix + '%',)).fetchone()
        return r  # (id, type, archived) or None

    def episodic_boost(query, idf_norm):
        """node_id -> boost over top-K operator-trace readout."""
        qv = embedder.embed_query(query)
        scored = sorted(((embedder.cosine_similarity(qv, vec), tid, txt, ca)
                         for tid, vec, txt, ca in op_traces),
                        key=lambda x: -x[0])[:TOPK_TRACES]
        per_trace = []
        deg = {}
        for cos, tid, txt, ca in scored:
            assoc = set(trace_to_nodes.get(tid, set()) & node_days.keys())
            td = daysecs(ca)
            if td is not None:
                for nid, ds in node_days.items():
                    if ds and min(abs(td - d) for d in ds) <= WINDOW_DAYS:
                        assoc.add(nid)
            per_trace.append((cos, assoc))
            for nid in assoc:
                deg[nid] = deg.get(nid, 0) + 1
        n_seed = max(1, len(scored))
        boost = {}
        for cos, assoc in per_trace:
            for nid in assoc:
                w = 1.0
                if idf_norm:
                    w = max(0.0, math.log(n_seed / (1.0 + deg[nid])))
                boost[nid] = boost.get(nid, 0.0) + cos * w
        return boost, scored

    print("=" * 96)
    print("FAITHFUL RECOVERY  | TOPK_TRACES=%d  WINDOW=±%gd  NODE_TOPK=%d  λ=%s"
          % (TOPK_TRACES, WINDOW_DAYS, NODE_TOPK, LAMBDAS))
    print("=" * 96)

    summary = {idf: {lam: 0 for lam in LAMBDAS} for idf in (False, True)}
    testable = 0

    for qid, gold8, query in SIX:
        g = resolve(gold8)
        print("\n%-4s gold=%s  query=%r" % (qid, gold8, query))
        if not g:
            print("     [gold not found] -> skip")
            continue
        gid, gtype, garch = g
        if garch:
            print("     [ARCHIVED — no node embedding exists; cannot be a recall candidate")
            print("      under ANY embedding-cosine mechanism, boost included] -> untestable")
            continue

        # real production pool (unfiltered)
        out = brain.recall(query=query, limit=200)
        pool = out.get('results', [])
        pool_score = {r['id']: r.get('relevance_score', 0.0) for r in pool}
        in_pool = gid in pool_score
        # real gold score via type-filtered recall (does not change scoring, only admits gold)
        gout = brain.recall(query=query, limit=200, filter={'type': {'in': [gtype]}})
        grow = next((r for r in gout.get('results', []) if r['id'] == gid), None)
        gold_rel = grow.get('relevance_score', 0.0) if grow else 0.0
        print("     production pool=%d  gold-in-pool=%s  gold relevance_score=%.4f"
              % (len(pool), in_pool, gold_rel))
        testable += 1

        for idf in (False, True):
            boost, seed = episodic_boost(query, idf_norm=idf)
            bmax = max(boost.values()) if boost else 0.0
            gb = boost.get(gid, 0.0)
            tag = "IDF" if idf else "raw"
            if gid not in boost or bmax == 0:
                print("       [%s] gold has NO episodic association (boost=0)" % tag)
                continue
            # Universe = real pool candidates UNION every boosted node (so the
            # co-temporal crowd actually competes — otherwise the popularity trap
            # is hidden). Boosted nodes outside the pool get relevance_score 0
            # (they were below the floor; that is why they need the boost).
            universe = dict(pool_score)
            for nid in boost:
                universe.setdefault(nid, 0.0)
            universe[gid] = gold_rel
            lines = []
            for lam in LAMBDAS:
                def blended(nid):
                    return universe[nid] + lam * (boost.get(nid, 0.0) / bmax)
                gold_b = blended(gid)
                rank = 1 + sum(1 for nid in universe if blended(nid) > gold_b)
                hit = rank <= NODE_TOPK
                if hit:
                    summary[idf][lam] += 1
                lines.append("λ=%.2f→#%d%s" % (lam, rank, "*" if hit else ""))
            # popularity diagnostic
            top3 = sorted(boost.items(), key=lambda x: -x[1])[:3]
            td = []
            for nid, bv in top3:
                t = bc.execute("SELECT title FROM nodes WHERE id=?", (nid,)).fetchone()
                td.append("%s=%.2f%s" % (nid[:6], bv, "«GOLD" if nid == gid else ""))
            print("       [%s] gold_boost=%.3f / max=%.3f   %s"
                  % (tag, gb, bmax, "  ".join(lines)))
            print("              top-boosted: %s" % "  ".join(td))

    print("\n" + "=" * 96)
    print("SUMMARY — recovered into production top-%d (of %d TESTABLE active golds):"
          % (NODE_TOPK, testable))
    print("=" * 96)
    print("  %-10s %s" % ("", "  ".join("λ=%.2f" % l for l in LAMBDAS)))
    for idf in (False, True):
        print("  %-10s %s" % ("IDF-norm" if idf else "raw-M",
                              "  ".join("%5d" % summary[idf][l] for l in LAMBDAS)))
    print("\n  (* = recovered into top-%d; archived golds are untestable — no node vector exists)"
          % NODE_TOPK)
