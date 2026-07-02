#!/usr/bin/env python3
"""LAF operators — each maps (state, brain) → per-node activation, verifiable in isolation.

The edge_context lesson (2026-06-25): a configured component can be 100% dead and
invisible to the top-level metric. So every operator here is written to be checked by
verify_substrate.py — liveness, input-dependence, and a math invariant — before any
fusion result built on it is trusted.

Operator #1 (this file): MaxSim-cosine over the live field-groups — ColBERT-style late
interaction. Each node's cosine = the MAX over its field-groups (best-matching field
wins, never the average). This is the realizable form of the §18.12 best-field oracle.

Next operators (separate build, after reading their substrate APIs): typed-graph-spread,
temporal-distinctiveness ÷norm.
"""
from collections import namedtuple

import numpy as np

from servers import embedder
from servers.pipeline_contract import EMBEDDING_GROUPS

# KNOWN_DEAD_GROUPS: groups with 0 production coverage, excluded from MaxSim so a dead
# group can't silently contribute nothing (the failure class this whole gate guards).
# edge_context WAS here (0 rows) but was REVIVED 2026-06-25 (main ebb58ad: backfill handler
# implemented; now 4490 vectors / 71% coverage = 100% of nodes with a described edge), so it
# is promoted into MAXSIM_GROUPS below. Empty now — the mechanism stays for the next dead one.
KNOWN_DEAD_GROUPS = set()

MAXSIM_GROUPS = []
for _g in EMBEDDING_GROUPS.values():
    _vt = _g.get("vector_type")
    if _g.get("weight", 0) > 0 and _vt and _vt not in MAXSIM_GROUPS and _vt not in KNOWN_DEAD_GROUPS:
        MAXSIM_GROUPS.append(_vt)


def unit(blob):
    """Decode an embedding blob → unit vector (None if absent/zero).

    Alias of the PRODUCTION normalizer (servers/recall_laf.py:_unit) so every
    probe measures the exact function the shipped engine runs — single source,
    no eval↔production drift (code-review 2026-07-02)."""
    from servers.recall_laf import _unit
    return _unit(blob)


def query_vec(query):
    """Unit query vector (None if embedder not ready / empty)."""
    return unit(embedder.embed_query(query))


def load_group_vectors(brain, vt, model):
    """{node_id: unit_vec} for one vector_type, via the same DAL recall uses."""
    out = {}
    for r in brain._vec_dal.get_all_vectors(vector_types=[vt], model=model or None):
        uv = unit(r.get("embedding"))
        if uv is not None:
            out[r["node_id"]] = uv
    return out


def build_field_matrices(brain, model, groups):
    """Stack each group's vectors into aligned matrices for fast full-field scoring.

    Returns (master_ids, idx, mats) where master_ids is the ordered list of nodes
    with ≥1 live group vector, idx maps node_id→row, and mats[vt] is an
    [N × dim] float32 matrix with NaN rows where that group is absent for a node.
    NaN propagates through `mats[vt] @ qv` so np.nanmax cleanly ignores absent groups.
    """
    gv = {vt: load_group_vectors(brain, vt, model) for vt in groups}
    master = sorted(set().union(*[set(gv[vt]) for vt in groups])) if gv else []
    idx = {nid: i for i, nid in enumerate(master)}
    dim = next((len(v) for vecs in gv.values() for v in vecs.values()), 0)
    mats = {}
    for vt in groups:
        M = np.full((len(master), dim), np.nan, dtype=np.float32)
        for nid, v in gv[vt].items():
            M[idx[nid]] = v
        mats[vt] = M
    return master, idx, mats


def maxsim_field(qv, mats, groups):
    """[N] MaxSim cosine — max over groups, NaN-ignoring (absent groups don't count)."""
    stack = np.stack([mats[vt] @ qv for vt in groups])     # [G, N], NaN where group absent
    return np.nanmax(stack, axis=0)                          # [N]


def primary_field(qv, mats):
    """[N] _primary-only cosine — the A/B reference and the MaxSim≥primary invariant base."""
    return mats["_primary"] @ qv                             # NaN where _primary absent (rare)


# ───────────────────── operator #2: typed-graph-spread ─────────────────────
# Undirected, degree-normalized activation flow over the noise-excluded edges.
# Degree-normalization IS the ÷norm built into the operator (a hub can't dominate).
# Undirected + degree-normalized keeps the spread step a symmetric, bounded transform
# — the settling-friendly form (Hopfield read: asymmetry in similarity, symmetric readout).

def build_adjacency(brain, idx):
    """Typed (noise-excluded) undirected weighted edges over the master index.

    Returns (src, dst, w, degree) numpy arrays for scatter-add spread. Edges whose
    endpoints lack an embedding (not in `idx`) are dropped — the field is the embedded
    node set. Multiple relations on one pair are summed (more relations = stronger link).
    """
    na = brain.aspects.by_name("noise")
    noise = list(na.edge_relations) if na else []
    ph = ",".join("?" * len(noise)) if noise else "''"
    rows = brain.conn.execute(
        "SELECT e.source_id, e.target_id, SUM(COALESCE(er.weight, 1.0)) "
        "FROM edge_relations er JOIN edges e ON e.edge_id = er.edge_id "
        "WHERE (er.archived IS NULL OR er.archived = 0) "
        "  AND er.relation NOT IN (%s) "
        "GROUP BY e.source_id, e.target_id" % ph, noise).fetchall()
    src, dst, w = [], [], []
    for s, t, wt in rows:
        if s in idx and t in idx and s != t:
            src.append(idx[s]); dst.append(idx[t]); w.append(float(wt or 1.0))
    src = np.asarray(src, dtype=np.int64)
    dst = np.asarray(dst, dtype=np.int64)
    w = np.asarray(w, dtype=np.float32)
    degree = np.zeros(len(idx), dtype=np.float32)
    if src.size:
        np.add.at(degree, src, w)
        np.add.at(degree, dst, w)
    return src, dst, w, degree


def graph_spread(a, adj, hops=1):
    """[N] one-or-more hops of undirected, degree-normalized spread of activation `a`."""
    src, dst, w, degree = adj
    cur = np.asarray(a, dtype=np.float32).copy()
    nz = degree > 0
    for _ in range(hops):
        nxt = np.zeros_like(cur)
        if src.size:
            np.add.at(nxt, dst, cur[src] * w)
            np.add.at(nxt, src, cur[dst] * w)
        nxt[nz] = nxt[nz] / degree[nz]
        cur = nxt
    return cur


# ─────────────────── operator #3: temporal-distinctiveness ───────────────────
# von-Restorff / SIMPLE (b4733c4e, §18.3): salience = temporal ISOLATION, not recency.
# A node alone in its time-neighbourhood rises; a co-temporal crowd divides down — recency
# "activates AND inhibits in parallel". QUERY-INDEPENDENT: a node-prior modulated by the
# field's temporal density, NOT a query↔node similarity. This is the operator that was dead
# as recency_score=1.000 (37021fd1) — so its gate checks isolated≠crowd, not query-sensitivity.

def parse_days(created_at_list):
    """ISO created_at strings → float days-since-epoch (NaN if unparseable)."""
    import datetime
    out = []
    for s in created_at_list:
        try:
            out.append(datetime.datetime.fromisoformat(s).timestamp() / 86400.0)
        except Exception:
            out.append(np.nan)
    return np.asarray(out, dtype=np.float64)


def temporal_distinctiveness(days, eligible_mask, window_days=7.0):
    """[N] 1/(1 + neighbours within ±window_days), over the eligible set. Isolated→~1,
    dense-crowd→~0. Non-eligible / unparseable nodes get 0 (no contribution to the field)."""
    out = np.zeros(len(days), dtype=np.float32)
    elig = eligible_mask & ~np.isnan(days)
    idxs = np.where(elig)[0]
    if idxs.size == 0:
        return out
    t = days[idxs]
    order = np.argsort(t)
    ts = t[order]
    lo = np.searchsorted(ts, ts - window_days, side="left")
    hi = np.searchsorted(ts, ts + window_days, side="right")
    neighbours = (hi - lo - 1).astype(np.float32)        # exclude self
    out[idxs[order]] = 1.0 / (1.0 + neighbours)
    return out


# ──────────────── operator: relational reinstatement (the graph rebuild) ────────────────
# The old graph_spread weighted edges by the stored `weight` — which is an uncalibrated 0.5
# default (5a58ea33): SUM(weight) ≈ 0.5 × relation-multiplicity, no relevance signal. Decay +
# Hebbian only touch the noise/co-access relations we already exclude. So the rebuild throws the
# stored weight away and sets each edge's CONDUCTANCE from MEANING: cos(cue, edge.why), using the
# per-edge description embedding (edge_relations.embedding, v26+, ~91% of semantic edges). Activation
# flows from cosine-reachable seeds along the edges whose description matches the cue — the realizable
# form of what the judges did (follow a typed edge whose `why` IS the need, from a node they found).

EdgeIndex = namedtuple("EdgeIndex", "src dst emat rels created")


def build_edge_conductance(brain, idx):
    """Per-edge substrate for the graph operators — an EdgeIndex namedtuple
    (src, dst, emat, rels, created) over noise-excluded edges that HAVE a description
    embedding. Built once, reused per cue. NAMED shape, not a bare tuple (arity drifts
    silently; a named field extends without breaking consumers).

    emat[e] is the unit vector of edge e's description (compose_edge_text(relation,
    description)); conductance = emat @ qv at recall time. created[e] is the relation's
    ISO created_at — REQUIRED for eval integrity: an edge written after a cue's cutoff
    did not exist at that moment and must not conduct (the future-leak fix; 1/11 beam
    rescues rode a hindsight edge before this mask existed). Edges whose endpoints lack
    a node embedding (not in idx), self-loops, or with no description vector are dropped."""
    na = brain.aspects.by_name("noise")
    noise = list(na.edge_relations) if na else []
    ph = ",".join("?" * len(noise)) if noise else "''"
    rows = brain.conn.execute(
        "SELECT e.source_id, e.target_id, er.relation, er.embedding, er.created_at "
        "FROM edge_relations er JOIN edges e ON e.edge_id = er.edge_id "
        "WHERE (er.archived IS NULL OR er.archived = 0) "
        "  AND er.relation NOT IN (%s) AND er.embedding IS NOT NULL" % ph, noise).fetchall()
    src, dst, vecs, rels, created = [], [], [], [], []
    for s, t, rel, emb, ca in rows:
        if s in idx and t in idx and s != t:
            uv = unit(emb)
            if uv is not None:
                src.append(idx[s]); dst.append(idx[t]); vecs.append(uv)
                rels.append(rel); created.append(ca or "")
    src = np.asarray(src, dtype=np.int64)
    dst = np.asarray(dst, dtype=np.int64)
    emat = np.asarray(vecs, dtype=np.float32) if vecs else np.zeros((0, len(next(iter(idx), "")) or 768))
    return EdgeIndex(src, dst, emat, rels, np.asarray(created, dtype=object))


def edge_cos(edges, qv, cutoff=None):
    """[E] cue↔edge-why conductance: clipped cos(qv, edge description vector).

    THE one definition all graph probes/operators share (was copy-pasted np.clip in
    five files). `cutoff` (ISO string) zeroes edges created after it — a zeroed edge
    can never conduct (every follow-threshold is > 0), so post-cutoff hindsight edges
    are dead for that cue. Empty created_at (pre-provenance edges) is treated as OLD
    (conducts): those are early-build edges, not future leaks."""
    if edges.emat.shape[0] == 0:
        return np.zeros(0)
    c = np.clip(edges.emat @ qv, 0.0, None)
    if cutoff:
        c = np.where((edges.created == "") | (edges.created <= cutoff), c, 0.0)
    return c


def created_at_array(brain, master):
    """[N] ISO created_at strings aligned to the master index — ONE query (was a
    full-table scan per master id inside a list comprehension in five probe files)."""
    created = dict(brain.conn.execute("SELECT id, created_at FROM nodes").fetchall())
    return np.array([created.get(nm, "") for nm in master])


def relational_reinstatement(qv, seed, edges, n, hops=1, cutoff=None):
    """[N] activation spread from `seed` (a per-node cosine field) along edges, each edge weighted
    by its cue-MEANING conductance cos(qv, edge.why) — NOT the stored weight. ÷norm by conductance-
    degree so a node fed by many weakly-matching edges can't dominate. Undirected v1.
    `cutoff` masks post-cutoff edges (eval integrity — see edge_cos)."""
    out = np.zeros(n, dtype=np.float64)
    if edges.src.size == 0 or edges.emat.shape[0] == 0:
        return out
    src, dst = edges.src, edges.dst
    cond = edge_cos(edges, qv, cutoff)                   # [E] cue↔edge-why match, no negatives
    condeg = np.zeros(n, dtype=np.float64)
    np.add.at(condeg, dst, cond)
    np.add.at(condeg, src, cond)
    cur = np.clip(np.asarray(seed, dtype=np.float64), 0.0, None).copy()
    for _ in range(hops):
        nxt = np.zeros(n, dtype=np.float64)
        np.add.at(nxt, dst, cur[src] * cond)
        np.add.at(nxt, src, cur[dst] * cond)
        nz = condeg > 0
        nxt[nz] /= (1.0 + condeg[nz])                     # ÷norm: conductance-degree hub damp
        cur = nxt
    return cur
