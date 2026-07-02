"""LAF v1 recall variant — the flag-gated challenger scorer (§19 P1).

Activated by BRAIN_RECALL_VARIANT=laf_v1 (exported from hooks/scripts/brain-env.sh,
sourced by start-daemon.sh). Flag off → this module is never imported and the champion
path in brain_recall._recall_impl runs unchanged. Rollback = unset the flag + restart.

The composition (measured on the 24-cue lens-independent gold, eval/laf/composition_probe.py
+ maxsim_decomp.py — 18% need@5 / 28% need@25 vs production ~10% need@5):

    score(n) = sigmoid( z(maxsim) + g_pick·z(pick) + g_enc·z(enc) + g_idf·z(idf) + g_sit·z(sit) )

    maxsim — best raw cosine(query, node) across the 6 embedding views (nanmax; the
             decomposed sum(z)/max(z) variants measured worse IN-STACK at static gains —
             they become fitted columns at P3, see eval/laf/maxsim_decomp.md)
    pick   — nodes Haiku surfaced AND selected at past moments similar to the query
             (uncapped full-history trace scan — the newest-500 recall_episodes cap is
             deliberately NOT inherited; Tom 2026-07-02: structurally wrong for a field)
    enc    — nodes created/revised at similar past moments
    idf    — production's idf2 rare-token title boost re-expressed as a lane
    sit    — cosine(query, node _situation vector) as its own lane (NOT folded into
             maxsim's nanmax: separate gain so P4 can re-cue it from situation state)

Gains are static v1 constants overridable via the interactions K-store
(get_interaction_config('recall_laf')) — P3 fits them, P4 makes them g(query); the
runtime shape here never changes for that.

The sigmoid maps the unbounded z-sum into (0,1) monotonically so the champion's
downstream machinery (noise/relevance floors, effective_activation consumers) sees
cosine-like magnitudes; ranking is unaffected.

How it integrates: brain_recall._recall_impl calls LafV1Engine.scores() right after the
query embed and injects {node_id: score01} as each node's `sim` in the STEP-3 loop —
so archived/type/project filters, synaptic fatigue, critical boost, mismatch penalty,
hydration and tracing are all inherited from the champion loop rather than duplicated.
The channels the field REPLACES (z-weighted groups, situation scan, FTS5 net, keyword
blend, idf2 title boost, trace-chain lane) are gated off under the flag at their sites.

Caches (daemon-resident, staleness-checked per call, lock-guarded):
  field matrices  [N×768]×6 views + _situation — rebuilt when node_enrichments changes
  title idf       token sets + df counts       — rebuilt when nodes change
  trace matrix    [T×768] conversational-s0    — incremental append by created_at
"""
import math
import re
import threading
from collections import defaultdict

import numpy as np

try:
    from . import embedder
    from .brain_constants import _TITLE_BOOST_STOPWORDS
    from .pipeline_contract import EMBEDDING_GROUPS
    from .trace_contract import CONVERSATIONAL_REF_TYPES
    from .scales.s1.trace_links import gather, nodes_for_traces, _stop_of
except ImportError:                                    # direct-script import shape
    import embedder
    from brain_constants import _TITLE_BOOST_STOPWORDS
    from pipeline_contract import EMBEDDING_GROUPS
    from trace_contract import CONVERSATIONAL_REF_TYPES
    from scales.s1.trace_links import gather, nodes_for_traces, _stop_of

# v1 gains — the measured static composition. Overridable via the interactions
# K-store (name='recall_laf', config JSON with these keys); P3 replaces the values,
# not the shape.
DEFAULT_CONFIG = {
    'gain_pick': 0.5,
    'gain_enc': 0.3,
    'gain_idf': 0.5,
    'gain_sit': 0.5,
    'top_moments': 15,        # similar past moments to seed episodic roles from
    'window_turns': 1,        # ±N-turn moment window (the measured winner)
    'sigmoid_scale': 3.0,     # z-sum → (0,1) squash temperature
    'session_trace_pull': 2000,
}

# The 6 maxsim views: every live embedding group (weight>0) by vector_type.
MAXSIM_VIEWS = []
for _g in EMBEDDING_GROUPS.values():
    _vt = _g.get('vector_type')
    if _g.get('weight', 0) > 0 and _vt and _vt not in MAXSIM_VIEWS:
        MAXSIM_VIEWS.append(_vt)

_IDF_TOK = re.compile(r"[a-z0-9]+(?:[._][a-z0-9]+)*")   # idf2's tokenizer, verbatim


def _unit(blob):
    """Embedding blob → unit float32 vector (None if absent/zero)."""
    if blob is None:
        return None
    v = (np.frombuffer(blob, dtype=np.float32)
         if isinstance(blob, (bytes, bytearray)) else np.asarray(blob, dtype=np.float32))
    n = float(np.linalg.norm(v))
    return (v / n) if n > 1e-9 else None


def _zscore(x, n):
    """Standardize over finite entries → unit variance; 0 elsewhere (the LAF
    fusion normalizer — gains stay pure influence dials only through this form)."""
    m = np.isfinite(x)
    o = np.zeros(n)
    if int(m.sum()) > 2 and np.std(x[m]) > 1e-9:
        o[m] = (x[m] - x[m].mean()) / x[m].std()
    return o


class LafV1Engine:
    """Daemon-resident scorer: caches the matrices, computes per-query field scores.

    One instance per Brain (lazily attached as brain._laf_engine). All cache
    refreshes are guarded by one lock; concurrent recalls of different queries
    share the refreshed caches.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._mats = None            # {view: [N×768] float32, NaN where absent}
        self._sit_mat = None         # [N×768] float32, NaN where absent
        self._master = []            # ordered node ids
        self._idx = {}               # node_id → row
        self._short = {}             # 8-char short id → row (unambiguous only)
        self._mats_key = None        # (count, max_rowid) of node_enrichments
        self._title_tok = {}         # row → frozenset(title tokens)
        self._title_df = {}          # token → doc frequency over titles
        self._titles_key = None      # (count, max_rowid) of nodes
        self._tr_meta = []           # [(chain_id, session_id)]
        self._tr_mat = None          # [T×768] float32
        self._tr_last = ''           # max created_at ingested (incremental append)

    # ── config ──
    def config(self, brain):
        cfg = dict(DEFAULT_CONFIG)
        try:
            stored = brain.get_interaction_config('recall_laf') or {}
            cfg.update({k: stored[k] for k in DEFAULT_CONFIG if k in stored})
        except Exception:
            pass                      # K-store empty → defaults
        return cfg

    # ── caches ──
    def _refresh_matrices(self, brain, model):
        key = brain.conn.execute(
            'SELECT COUNT(*), COALESCE(MAX(rowid),0) FROM node_enrichments').fetchone()
        if self._mats is not None and key == self._mats_key:
            return
        views = list(MAXSIM_VIEWS) + ['_situation']
        gv = {vt: {} for vt in views}
        for r in brain._vec_dal.get_all_vectors(vector_types=views, model=model or None):
            uv = _unit(r.get('embedding'))
            if uv is not None:
                gv[r['vector_type']][r['node_id']] = uv
        master = sorted(set().union(*[set(d) for d in gv.values()]) if gv else set())
        idx = {nid: i for i, nid in enumerate(master)}
        dim = next((len(v) for d in gv.values() for v in d.values()), 768)
        mats = {}
        for vt in views:
            m = np.full((len(master), dim), np.nan, dtype=np.float32)
            for nid, v in gv[vt].items():
                m[idx[nid]] = v
            mats[vt] = m
        by_short = defaultdict(list)
        for nid in master:
            by_short[nid[:8]].append(nid)
        self._short = {s: idx[fulls[0]] for s, fulls in by_short.items()
                       if len(fulls) == 1}
        self._sit_mat = mats.pop('_situation')
        self._mats, self._master, self._idx, self._mats_key = mats, master, idx, key

    def _refresh_titles(self, brain):
        key = brain.conn.execute(
            'SELECT COUNT(*), COALESCE(MAX(rowid),0) FROM nodes').fetchone()
        if self._titles_key == key:
            return
        tok, df = {}, defaultdict(int)
        for nid, title in brain.conn.execute(
                'SELECT id, title FROM nodes WHERE archived = 0').fetchall():
            i = self._idx.get(nid)
            if i is None or not title:
                continue
            ts = frozenset(_IDF_TOK.findall(title.lower()))
            tok[i] = ts
            for t in ts:
                df[t] += 1
        self._title_tok, self._title_df, self._titles_key = tok, dict(df), key

    def _refresh_traces(self, brain):
        rows = brain._trace_dal.event_vector_rows(
            scale='s0', ref_types=list(CONVERSATIONAL_REF_TYPES),
            since=self._tr_last or None)
        if not rows:
            return
        meta, vecs, last = [], [], self._tr_last
        for chain_id, session_id, created_at, blob in rows:
            uv = _unit(blob)
            if uv is None:
                continue
            meta.append((chain_id, session_id))
            vecs.append(uv)
            if created_at > last:
                last = created_at
        if not vecs:
            self._tr_last = last
            return
        new = np.stack(vecs)
        self._tr_mat = new if self._tr_mat is None else np.vstack([self._tr_mat, new])
        self._tr_meta.extend(meta)
        self._tr_last = last

    # ── episodic roles (production port of eval/laf/episodic_ops.py, pick+enc only) ──
    def _episodic_vectors(self, brain, qv, cfg, n):
        """(pick, enc) [n] activation vectors from similar past surface-moments.

        Full-history matrixized scan (no newest-500 cap), top `top_moments`
        (session, stop) moments deduped at max similarity, roles joined via the
        canonical trace_links substrate, ±window unioned, picked-wins.
        """
        pick = np.zeros(n)
        enc = np.zeros(n)
        if self._tr_mat is None or not len(self._tr_meta):
            return pick, enc
        sims = self._tr_mat @ qv
        top = np.argsort(-sims)[:max(cfg['top_moments'] * 3, cfg['top_moments'])]
        moments = {}                          # (session, short, stop) → score
        for i in top:
            chain = self._tr_meta[i][0] or ''
            stop = _stop_of(chain)
            parts = chain.split('-')          # s0-{short}-{stop}
            sess = self._tr_meta[i][1]
            if stop is None or not sess or len(parts) < 3:
                continue
            key = (sess, parts[1], stop)
            s = float(sims[i])
            if s > moments.get(key, 0.0):
                moments[key] = s
            if len(moments) >= cfg['top_moments']:
                break
        w = int(cfg['window_turns'])
        by_sess = defaultdict(list)
        for (sess, short, stop), s in moments.items():
            by_sess[sess].append((short, stop, s))
        for sess, moms in by_sess.items():
            st = gather(brain, sess, streams=('surface', 'encode', 'recall'),
                        limit=cfg['session_trace_pull'])
            targets = [{'id': '%d@%d' % (mi, ws), 'chain_id': 's0-%s-%d' % (short, ws)}
                       for mi, (short, stop, _s) in enumerate(moms)
                       for ws in range(max(stop - w, 0), stop + w + 1)]
            links = nodes_for_traces(st['surface'], st['encode'], targets,
                                     recall_traces=st['recall'])
            for mi, (short, stop, s) in enumerate(moms):
                picked, encoded = set(), set()
                for ws in range(max(stop - w, 0), stop + w + 1):
                    link = links.get('%d@%d' % (mi, ws)) or {}
                    picked.update(link.get('surfaced', []))
                    encoded.update(link.get('encoded', []))
                for node in picked:
                    i = self._resolve(node)
                    if i is not None and s > pick[i]:
                        pick[i] = s
                for node in encoded:
                    i = self._resolve(node)
                    if i is not None and s > enc[i]:
                        enc[i] = s
        return pick, enc

    def _resolve(self, node_id):
        """Role node id (8-char short OR full) → master row, or None."""
        i = self._idx.get(node_id)
        return i if i is not None else self._short.get(node_id)

    # ── the lanes ──
    def _idf_vector(self, query, n):
        vec = np.zeros(n)
        q_tokens = {t for t in _IDF_TOK.findall(query.lower())
                    if len(t) >= 2 and t not in _TITLE_BOOST_STOPWORDS}
        if not q_tokens or not self._title_tok:
            return vec
        n_titles = max(len(self._title_tok), 1)
        idf = {t: math.log((n_titles + 1) / (self._title_df.get(t, 0) + 1))
               for t in q_tokens}
        total = sum(idf.values()) or 1.0
        for i, ts in self._title_tok.items():
            m = sum(idf[t] for t in q_tokens if t in ts)
            if m > 0:
                vec[i] = m / total
        return vec

    # ── the scorer ──
    def scores(self, brain, query, query_vec, model=None):
        """{node_id: score01} over every node with ≥1 embedding view.

        Monotonic sigmoid of the z-scored gain-weighted field sum; injected as
        `sim` per node in the champion's STEP-3 loop (which then applies its own
        fatigue / filters / floors / boosts downstream).
        """
        qv = _unit(query_vec)
        if qv is None:
            return {}
        cfg = self.config(brain)
        with self._lock:
            self._refresh_matrices(brain, model)
            self._refresh_titles(brain)
            self._refresh_traces(brain)
            n = len(self._master)
            if n == 0:
                return {}
            stack = np.stack([self._mats[vt] @ qv for vt in MAXSIM_VIEWS])
            with np.errstate(all='ignore'):
                maxsim = np.nanmax(stack, axis=0)
            sit_raw = self._sit_mat @ qv
            sit = np.where(np.isfinite(sit_raw), sit_raw, 0.0)
            pick, enc = self._episodic_vectors(brain, qv, cfg, n)
            idfv = self._idf_vector(query, n)
            zsum = (_zscore(maxsim, n)
                    + cfg['gain_pick'] * _zscore(pick, n)
                    + cfg['gain_enc'] * _zscore(enc, n)
                    + cfg['gain_idf'] * _zscore(idfv, n)
                    + cfg['gain_sit'] * _zscore(sit, n))
            s01 = 1.0 / (1.0 + np.exp(-zsum / float(cfg['sigmoid_scale'])))
            return {nid: float(s01[i]) for i, nid in enumerate(self._master)}


def get_engine(brain):
    """The per-Brain engine singleton (lazily attached)."""
    eng = getattr(brain, '_laf_engine', None)
    if eng is None:
        eng = LafV1Engine()
        brain._laf_engine = eng
    return eng
