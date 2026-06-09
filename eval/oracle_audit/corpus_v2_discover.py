#!/usr/bin/env python3
"""CORPUS-V2 DISCOVERY SCOUT — dual-gold (node + trace) candidate finder.

For each corpus query, pre-fetches BOTH lanes' candidates so the dual-store eval has ground truth
for each store:
  • node candidates   — brain.recall(rich, limit=K)  → the SEMANTIC lane's job
  • trace candidates  — the REAL trace->node chain: cosine(query, s0 dialogue trace vectors) → top-T
                        traces, and for each, the top-3 nodes that trace pulls (the trace->node hop).
                        This is exactly what the BRAIN_TRACE_CHAIN lane retrieves (servers/brain_recall.py
                        _trace_chain_candidates), so trace-gold scores the EPISODIC lane honestly.
  • coverage          — embedded vs total s0 dialogue traces + per-entity hit counts. Documents the
                        ~2-week embed-window / ~3,500 un-embedded backfill gap so the corpus does NOT
                        hide that the trace-chain may be starved for the very queries it serves.

Daemon-safe: IsolatedBrain copies both DBs and runs the embedder in-process — never touches the live
daemon. All candidate TEXT is pre-fetched so the downstream judge workflow can score from text alone
(zero daemon calls). Dumps eval/oracle_audit/corpus_v2_candidates.json + prints a readable summary.

Usage: ./dev python3 eval/oracle_audit/corpus_v2_discover.py
"""
import sys, json
import numpy as np
ROOT = '/Users/tpac/brain'
sys.path.insert(0, ROOT)
from tests.isolated_brain import IsolatedBrain          # noqa: E402
from servers import embedder                            # noqa: E402

# ── the corpus queries (dual-gold targets). kind drives which lane SHOULD carry it. ──
QUERIES = [
    {"id": "T1",  "kind": "thick-concept",     "terse": "MCP facts",
     "rich": "important facts about MCP — model context protocol, tools, servers, restart constraints"},
    {"id": "T2",  "kind": "thick-relational",  "terse": "adCP protocol",
     "rich": "what is adCP — the ad context protocol, the buying flow and what the protocol means"},
    {"id": "E1a", "kind": "entity-rich",       "terse": "ex.co leadership",
     "rich": "who is in EX.CO leadership — the co-founders, names and titles of people at the company"},
    {"id": "E1b", "kind": "entity-terse",      "terse": "names of people at ex.co",
     "rich": "Do you remember any name from people working at ex.co?"},
    {"id": "E2",  "kind": "bare-token",        "terse": "Shachar",
     "rich": "Shachar"},
    {"id": "L1",  "kind": "literal-match",     "terse": "SpringServe multicall dynamic pricing",
     "rich": "SpringServe multicall dynamic pricing — what's the ad server setting"},
    {"id": "L2",  "kind": "literal-match",     "terse": "turn off multicall ex.co",
     "rich": "should multicall be turned off when connecting an ad server to EX.CO?"},
    {"id": "EP1", "kind": "episodic",          "terse": "last session on ex.co",
     "rich": "what did we do on the last session we worked on ex.co?"},
    {"id": "C1",  "kind": "control",           "terse": "error logs on hooks",
     "rich": "Do we have some standard for error logs on all hooks?"},
    {"id": "C2",  "kind": "control",           "terse": "fatigue pillar nodes",
     "rich": "Tom on fatigue: should general pillar nodes pop repeatedly in recall?"},
]
ENTITY_TOKENS = ["ex.co", "adcp", "adcp", "springserve", "shachar", "kevel", "multicall", "vast"]
NODE_K, TRACE_T, HOP_N = 15, 8, 3


def _norm_rows(blobs):
    """bytes[] -> (M,768) L2-normalized float32 matrix + index map of which survived."""
    vecs, idx = [], []
    for i, b in enumerate(blobs):
        if not b:
            continue
        v = np.frombuffer(b, dtype=np.float32)
        n = np.linalg.norm(v)
        if n == 0 or v.shape[0] == 0:
            continue
        vecs.append(v / n)
        idx.append(i)
    return (np.vstack(vecs) if vecs else np.zeros((0, 768), np.float32)), idx


with IsolatedBrain() as env:
    brain = env.brain
    model = embedder.stats.get('model_name') or None

    # ---- node vectors (the trace->node hop target + a title/type/snippet lookup) ----
    node_rows = brain._vec_dal.get_all_vectors(vector_types=['_primary'], model=model)
    node_mat, nidx = _norm_rows([nr['embedding'] for nr in node_rows])
    node_ids = [node_rows[i]['node_id'] for i in nidx]

    def node_meta(nid):
        r = brain.conn.execute("SELECT title, type, substr(content,1,200) FROM nodes WHERE id=?",
                               (nid,)).fetchone()
        return {"id": nid, "title": (r[0] if r else '?'), "type": (r[1] if r else '?'),
                "snippet": (r[2] if r else '')}

    # ---- s0 dialogue trace vectors + text (the lane's exact hygiene filter) ----
    trows = brain.logs_conn.execute(
        """SELECT te.trace_id, te.vector, te.text, ev.ref_type, ev.scale, ev.session_id, ev.created_at
           FROM trace_embeddings te LEFT JOIN trace_events ev ON ev.id = te.trace_id""").fetchall()
    s0 = [r for r in trows if r[1] and r[4] == 's0' and r[3] in ('user_message', 'assistant_message')]
    trace_mat, tidx = _norm_rows([r[1] for r in s0])
    trace_meta = [{"trace_id": s0[i][0], "text": (s0[i][2] or '')[:300], "ref_type": s0[i][3],
                   "session": (s0[i][5] or '?')[:8], "created_at": s0[i][6]} for i in tidx]

    # ---- coverage census ----
    tot_s0_events = brain.logs_conn.execute(
        "SELECT COUNT(*) FROM trace_events WHERE scale='s0' AND ref_type IN ('user_message','assistant_message')"
    ).fetchone()[0]
    emb_s0 = len(trace_meta)
    tok_hits = {}
    for tok in set(ENTITY_TOKENS):
        tok_hits[tok] = sum(1 for m in trace_meta if tok in m['text'].lower())

    print("=" * 90)
    print("TRACE COVERAGE CENSUS")
    print("  embedded s0 dialogue traces : %d" % emb_s0)
    print("  total s0 dialogue events    : %d   (un-embedded gap: %d, %.0f%%)"
          % (tot_s0_events, tot_s0_events - emb_s0,
             100.0 * (tot_s0_events - emb_s0) / max(tot_s0_events, 1)))
    print("  embedded-trace token hits   : " + ", ".join("%s=%d" % (t, tok_hits[t]) for t in sorted(tok_hits)))
    print("=" * 90)

    out = {"model": model, "coverage": {"embedded_s0": emb_s0, "total_s0_events": tot_s0_events,
                                        "token_hits": tok_hits}, "queries": []}

    for q in QUERIES:
        qv = embedder.embed_query(q['rich'])
        qa = np.frombuffer(qv, dtype=np.float32)
        qa = qa / (np.linalg.norm(qa) or 1.0)

        # NODE candidates (semantic lane)
        res = brain.recall(query=q['rich'], limit=NODE_K)
        nres = res.get('results', []) if isinstance(res, dict) else (res or [])
        node_cands = [{"id": (r.get('id') or r.get('node_id')), "title": (r.get('title') or '')[:55],
                       "type": r.get('type'), "discovery": r.get('_discovery')} for r in nres]

        # TRACE candidates (episodic lane) — query->trace cosine, then trace->node hop
        trace_cands = []
        if trace_mat.shape[0]:
            tscore = trace_mat @ qa
            top = np.argsort(-tscore)[:TRACE_T]
            for j in top:
                tv = trace_mat[j]
                hop = node_mat @ tv if node_mat.shape[0] else np.zeros(0)
                hop_top = np.argsort(-hop)[:HOP_N] if hop.shape[0] else []
                trace_cands.append({
                    "trace_id": trace_meta[j]['trace_id'], "tcos": round(float(tscore[j]), 3),
                    "session": trace_meta[j]['session'], "ref_type": trace_meta[j]['ref_type'],
                    "created_at": trace_meta[j]['created_at'], "text": trace_meta[j]['text'],
                    "pulls_nodes": [{"id": node_ids[k], "ncos": round(float(hop[k]), 3),
                                     **{kk: vv for kk, vv in node_meta(node_ids[k]).items() if kk in ('title', 'type')}}
                                    for k in hop_top],
                })

        out["queries"].append({**q, "node_candidates": node_cands, "trace_candidates": trace_cands})

        print("\n#%-4s [%s] %s" % (q['id'], q['kind'], q['rich'][:60]))
        print("  NODES: " + " ".join("%s(%s)" % ((c['id'] or '?')[:8], (c['discovery'] or '?')[:3]) for c in node_cands[:8]))
        print("  TRACES (query->trace cosine, then what each trace pulls):")
        for tc in trace_cands[:5]:
            pulls = " ".join("%s/%.2f" % (p['id'][:8], p['ncos']) for p in tc['pulls_nodes'])
            print("    %.3f [%s %s] %-46s -> %s"
                  % (tc['tcos'], tc['session'], (tc['created_at'] or '?')[:10], tc['text'][:46].replace('\n', ' '), pulls))

    with open(f'{ROOT}/eval/oracle_audit/corpus_v2_candidates.json', 'w') as f:
        json.dump(out, f, indent=2)
    print("\n" + "=" * 90)
    print("wrote eval/oracle_audit/corpus_v2_candidates.json (%d queries, full text pre-fetched for judges)"
          % len(out['queries']))
    print("=" * 90)
