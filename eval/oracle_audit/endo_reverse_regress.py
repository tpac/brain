#!/usr/bin/env python3
"""STEP 2 -- reverse-engineering regression: for every gold, what (state-cue x mechanism)
would have surfaced it? Find the best REALIZABLE path and compare to the oracle.

State-cues (from state_cues.json): cue, prev_anchor, prev_operator, recent_context (all
realizable) + next_move (ORACLE / future). Mechanisms: cosine on each node FIELD vector
(_primary/content/situation/question/title/reasoning), FTS (lexical), episodic (cue ->
similar past trace -> node), graph (1-hop from in-context nodes via a semantic edge).

For each cue we take the BEST essential-gold rank under each feature. Outputs:
  (1) feature matrix: state-cue x field -> hit@5 / hit@25 (which cue+field is the lever)
  (2) the extra mechanisms (FTS / episodic / graph) hit@5/@25
  (3) partition: realizably-reachable vs oracle-only vs unreachable (encode-gap)
  (4) by source.

Run (daemon maintenance-locked): ./dev python3 eval/oracle_audit/endo_reverse_regress.py
"""
import os, sys, json
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from tests.isolated_brain import IsolatedBrain
from servers import embedder

HERE = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.join(HERE, "endo_corpus")
corpus = {c["id"]: c for c in json.load(open(f"{OUT}/endo_gold_corpus.json"))}
state = {s["cue_id"]: s for s in json.load(open(f"{OUT}/state_cues.json"))}
FIELDS = ["_primary", "content", "situation", "question", "title", "reasoning"]
CUES = ["cue", "prev_anchor", "prev_operator", "recent_context", "next_move"]  # next_move = ORACLE
REALIZABLE = {"cue", "prev_anchor", "prev_operator", "recent_context"}
NOISE = {"co_accessed", "community_member", "co_anchored", "related", "related_to", "emergent_bridge", "co_member"}

with IsolatedBrain() as env:
    if not embedder.is_ready():
        embedder.load_model()
    conn, logs = env.brain.conn, env.brain.logs_conn
    j = " ".join(f"JOIN node_enrichments {a} ON {a}.node_id=n.id AND {a}.vector_type='{f}'"
                 for a, f in zip("pcsqtr", FIELDS))
    rows = conn.execute(f"SELECT n.id, n.created_at, p.embedding, c.embedding, s.embedding, "
                        f"q.embedding, t.embedding, r.embedding FROM nodes n {j} WHERE n.archived=0").fetchall()
    ids = [r[0] for r in rows]; pos = {nid: i for i, nid in enumerate(ids)}
    created = np.array([r[1] or "" for r in rows]); N = len(ids)
    Vf = {f: np.vstack([np.frombuffer(r[2 + k], dtype=np.float32) for r in rows]) for k, f in enumerate(FIELDS)}
    print(f"nodes with all {len(FIELDS)} field-vectors: {N}")

    # episodic substrate
    trows = logs.execute("""SELECT te.vector, ev.created_at FROM trace_embeddings te
        JOIN trace_events ev ON ev.id=te.trace_id
        WHERE ev.scale='s0' AND ev.ref_type IN ('user_message','assistant_message') AND te.vector IS NOT NULL""").fetchall()
    TV = np.vstack([np.frombuffer(r[0], dtype=np.float32) for r in trows])
    tcreated = np.array([r[1] or "" for r in trows])

    # graph adjacency (semantic edges only)
    adj = {}
    for sid, tid, rel in conn.execute("""SELECT e.source_id, e.target_id, er.relation
            FROM edge_relations er JOIN edges e ON e.edge_id=er.edge_id"""):
        if rel in NOISE:
            continue
        adj.setdefault(sid, set()).add(tid); adj.setdefault(tid, set()).add(sid)

    # embed all state-cue texts
    flat = []
    for cid, st in state.items():
        for S in CUES:
            txt = corpus[cid]["query"] if S == "cue" else corpus[cid]["next_move"] if S == "next_move" else st.get(S, "")
            if txt:
                flat.append((cid, S, txt))
    blobs = embedder.embed_batch([t[2] for t in flat], kind="query")
    vec = {(cid, S): np.frombuffer(b, dtype=np.float32) for (cid, S, _), b in zip(flat, blobs) if b}

    def rank_of(score, gold, elig):
        rs = [int((np.where(elig, score, -np.inf) > score[pos[g]]).sum()) + 1 for g in gold if g in pos]
        return min(rs) if rs else 10**9

    feat = {}   # cue_id -> {feature_name: best-essential-gold rank}
    for cid in corpus:
        c = corpus[cid]; st = state[cid]; gold = c["gold_essential"]
        elig = created < c["cutoff"]; f = {}
        for S in CUES:
            sv = vec.get((cid, S))
            if sv is None:
                continue
            for F in FIELDS:
                f[f"{S}|{F}"] = rank_of(Vf[F] @ sv, gold, elig)
        # FTS (lexical) on cue + recent_context
        for S in ("cue", "recent_context"):
            txt = c["query"] if S == "cue" else st.get("recent_context", "")
            hits = {nid: r + 1 for r, nid in enumerate(env.brain._fts.search(txt, limit=200))} if txt else {}
            f[f"{S}|FTS"] = min((hits[g] for g in gold if g in hits), default=10**9)
        # episodic: cue -> top past traces -> node
        sv = vec.get((cid, "cue"))
        if sv is not None:
            tc = np.where(tcreated < c["cutoff"], TV @ sv, -np.inf)
            top = np.argsort(-tc)[:20]; top = top[np.isfinite(tc[top])]
            epi = (Vf["_primary"] @ TV[top].T * tc[top]).sum(axis=1) if len(top) else np.zeros(N)
            f["cue|episodic"] = rank_of(epi, gold, elig)
        # graph: gold 1-hop from an in-context node via a semantic edge
        ctx = set(st.get("in_context_ids", []))
        f["incontext|graph1hop"] = 1 if any(g in adj and adj[g] & ctx for g in gold) else 10**9
        feat[cid] = f

# ---- report ----
def hit(fname, k, subset=None):
    rows = [feat[cid] for cid in feat if subset is None or corpus[cid]["source"] == subset]
    vals = [r[fname] for r in rows if fname in r]
    return f"{100*sum(1 for v in vals if v <= k)//len(vals):>3d}%" if vals else "  -"

print("\n=== (state-cue x field) -> hit@5 / hit@25  (best essential-gold rank) ===")
print(f"  {'':14s}" + "".join(f"{F[:9]:>11s}" for F in FIELDS))
for S in CUES:
    tag = "ORACLE " if S not in REALIZABLE else ""
    print(f"  {tag+S:14s}" + "".join(f"  {hit(f'{S}|{F}',5)}/{hit(f'{S}|{F}',25)}" for F in FIELDS))
print("\n=== extra mechanisms -> hit@5 / hit@25 ===")
for fn in ("cue|FTS", "recent_context|FTS", "cue|episodic", "incontext|graph1hop"):
    print(f"  {fn:22s} {hit(fn,5)} / {hit(fn,25)}")

# partition: best realizable vs oracle
print("\n=== partition (per cue: best gold rank) ===")
def best(cid, feats):
    return min((feat[cid][k] for k in feat[cid] if k.split('|')[0] in feats or k == 'incontext|graph1hop'),
               default=10**9)
for subset in (None, "operator_msg", "anchor_turn"):
    cids = [c for c in feat if subset is None or corpus[c]["source"] == subset]
    real = [min((feat[c][k] for k in feat[c]
                 if k.split('|')[0] in REALIZABLE or k.startswith('incontext')), default=10**9) for c in cids]
    orac = [min((feat[c][k] for k in feat[c] if k.startswith('next_move')), default=10**9) for c in cids]
    rr5 = sum(1 for r in real if r <= 5); rr25 = sum(1 for r in real if r <= 25)
    only_oracle = sum(1 for r, o in zip(real, orac) if r > 25 and o <= 25)
    unreach = sum(1 for r, o in zip(real, orac) if r > 25 and o > 25)
    lab = subset or "ALL"
    print(f"  {lab:13s} n={len(cids):2d} | realizable-reach @5 {100*rr5//len(cids)}%  @25 {100*rr25//len(cids)}% "
          f"| oracle-only(real>25,orac<=25) {only_oracle} | unreachable {unreach}")
