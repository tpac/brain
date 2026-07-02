"""LAF v1 recall variant — the flag-gated challenger scorer (§19 P1).

Activated by BRAIN_RECALL_VARIANT=laf_v1 (exported from hooks/scripts/brain-env.sh,
sourced by start-daemon.sh). Flag off → this module is never imported and the champion
path in brain_recall._recall_impl runs unchanged. Rollback = unset the flag + restart.

The composition (measured on the 24-cue lens-independent gold, eval/laf/composition_probe.py
+ maxsim_decomp.py — 18% need@5 / 28% need@25 vs production ~10% need@5):

    score(n) = sigmoid( Σ_field gain_field · z(field(n)) )

The FIELD REGISTRY (LafV1Engine._fields) is the extension seam: adding a lane = one
registry entry + a 'gain_<name>' config key. v1 fields:

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

Gains resolve through ONE seam (LafV1Engine.config: module defaults ⊕ the interactions
K-store entry 'recall_laf', TTL-cached) — P3 fits the values, P4 swaps the resolver to
g(query); the runtime shape here never changes for that.

RANGE CONTRACT: scores() returns values in the OPEN interval (0,1) — brain_recall's
champion machinery (fatigue multiply, noise/relevance floors, critical boost) assumes
cosine-like magnitudes. The sigmoid guarantees it mathematically; scores() validates
finiteness and raises rather than ship an out-of-contract vector (the caller's
try/except falls back to champion and logs).

PER-QUERY TELEMETRY: scores() also returns a compact per-field z-score record for the
top-scoring nodes ({node_id: {field: z}}); brain_recall attaches it to the result as
`_laf_fields` so the S1R trace substrate accretes (query, per-field scores, outcome)
rows in production — the P2 dataset walker's training feed.

Known accepted gap (code-review 2026-07-02, finding 10): a node with NO embedding rows
is invisible to the field; under the flag it is reachable only via the champion's
keyword fallback (which still runs), not the FTS5 stem net (gated off). The class is
narrow and transient (vector backfill closes it); lexical returns via the P4 gate.

Caches (daemon-resident, staleness-checked per call, lock-guarded):
  field matrices  growable [N×768] per view — INCREMENTAL rowid-watermark upserts via
                  VectorDAL.vectors_since; full rebuild only on deletion (count shrink)
  title idf       full rebuild, but only when nodes' change_key moved AND a TTL expired
  trace matrix    block list (no per-append copy), incremental by created_at
  config          TTL-cached K-store overlay
"""
import math
import re
import threading
import time
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

# v1 gains + knobs — the measured static composition. Overridable via the interactions
# K-store (name='recall_laf', config JSON with these keys); P3 replaces the values,
# not the shape. One 'gain_<name>' per registry field.
DEFAULT_CONFIG = {
    'gain_maxsim': 1.0,
    'gain_pick': 0.5,
    'gain_enc': 0.3,
    'gain_idf': 0.5,
    'gain_sit': 0.5,
    'top_moments': 15,        # similar past moments to seed episodic roles from
    'window_turns': 1,        # ±N-turn moment window (the measured winner)
    'sigmoid_scale': 3.0,     # z-sum → (0,1) squash temperature
    'session_trace_pull': 2000,
    'telemetry_top_n': 50,    # per-field z-scores recorded for this many top nodes
}

CONFIG_TTL_S = 60.0           # K-store overlay refresh cadence
TITLES_TTL_S = 60.0           # min seconds between title-idf rebuilds
TRACE_BLOCK_CONSOLIDATE = 32  # merge trace blocks when the list grows past this
_GROW = 1.3                   # matrix capacity growth factor

# The 6 maxsim views: every live embedding group (weight>0) by vector_type.
MAXSIM_VIEWS = []
for _g in EMBEDDING_GROUPS.values():
    _vt = _g.get('vector_type')
    if _g.get('weight', 0) > 0 and _vt and _vt not in MAXSIM_VIEWS:
        MAXSIM_VIEWS.append(_vt)
_ALL_VIEWS = MAXSIM_VIEWS + ['_situation']

_IDF_TOK = re.compile(r"[a-z0-9]+(?:[._][a-z0-9]+)*")   # idf2's tokenizer, verbatim


def _unit(blob):
    """Embedding blob → unit float32 vector (None if absent/zero).

    THE single normalizer for LAF vectors — eval/laf/operators.py aliases this
    so probes measure the exact function production runs."""
    if blob is None:
        return None
    v = (np.frombuffer(blob, dtype=np.float32)
         if isinstance(blob, (bytes, bytearray)) else np.asarray(blob, dtype=np.float32))
    n = float(np.linalg.norm(v))
    return (v / n) if n > 1e-9 else None


def _zscore(x, n, mask=None):
    """Standardize over finite (optionally masked) entries → unit variance; 0
    elsewhere. THE LAF fusion normalizer — gains stay pure influence dials only
    through this exact form; eval/laf/laf_metrics.py delegates here."""
    m = np.isfinite(x)
    if mask is not None:
        m = m & mask
    o = np.zeros(n)
    if int(m.sum()) > 2 and np.std(x[m]) > 1e-9:
        o[m] = (x[m] - x[m].mean()) / x[m].std()
    return o


def idf_scores(query, title_tok, title_df, n):
    """Production idf2 title boost as a lane: per node-row, Σ log-idf of the
    query's rare tokens found in its title, normalized by the query's total idf
    mass. Pure function so eval probes score the identical formula.

    title_tok: {row: frozenset(tokens)}; title_df: {token: doc frequency}.
    """
    vec = np.zeros(n)
    q_tokens = {t for t in _IDF_TOK.findall(query.lower())
                if len(t) >= 2 and t not in _TITLE_BOOST_STOPWORDS}
    if not q_tokens or not title_tok:
        return vec
    n_titles = max(len(title_tok), 1)
    idf = {t: math.log((n_titles + 1) / (title_df.get(t, 0) + 1))
           for t in q_tokens}
    total = sum(idf.values()) or 1.0
    for i, ts in title_tok.items():
        m = sum(idf[t] for t in q_tokens if t in ts)
        if m > 0:
            vec[i] = m / total
    return vec


def roles_for_moments(brain, moments, window_turns, pull_limit):
    """(session, short, stop)-keyed similar-moment scores → per-moment role records.

    THE shared episodic role-join (production engine AND eval/laf/episodic_ops.py
    consume this, so the measured join semantics are the shipped ones): per session,
    gather the surface/encode/recall streams once, join by stop via nodes_for_traces,
    union roles across the ±window stops, picked-wins-over-dropped within a moment.

    moments: {(session_id, short, stop): score}
    Returns [{'score', 'picked', 'encoded', 'dropped'}] — sets of node ids.
    """
    w = int(window_turns)
    by_sess = defaultdict(list)
    for (sess, short, stop), s in moments.items():
        by_sess[sess].append((short, stop, s))
    records = []
    for sess, moms in by_sess.items():
        st = gather(brain, sess, streams=('surface', 'encode', 'recall'),
                    limit=pull_limit)
        targets = [{'id': '%d@%d' % (mi, ws), 'chain_id': 's0-%s-%d' % (short, ws)}
                   for mi, (short, stop, _s) in enumerate(moms)
                   for ws in range(max(stop - w, 0), stop + w + 1)]
        links = nodes_for_traces(st['surface'], st['encode'], targets,
                                 recall_traces=st['recall'])
        for mi, (short, stop, s) in enumerate(moms):
            picked, encoded, dropped = set(), set(), set()
            for ws in range(max(stop - w, 0), stop + w + 1):
                link = links.get('%d@%d' % (mi, ws)) or {}
                picked.update(link.get('surfaced', []))
                encoded.update(link.get('encoded', []))
                dropped.update(link.get('dropped', []))
            dropped -= picked          # picked-wins within the whole moment
            records.append({'score': s, 'picked': picked,
                            'encoded': encoded, 'dropped': dropped})
    return records


class LafV1Engine:
    """Daemon-resident scorer: caches the matrices, computes per-query field scores.

    One instance per Brain (lazily attached as brain._laf_engine). All cache
    refreshes are guarded by one lock; concurrent recalls of different queries
    share the refreshed caches.
    """

    def __init__(self):
        self._lock = threading.Lock()
        # field matrices — growable, row-aligned across views
        self._mats = None            # {view: [cap×768] float32, NaN where absent}
        self._master = []            # ordered node ids (rows 0.._n)
        self._idx = {}               # node_id → row
        self._short = {}             # 8-char short id → row (unambiguous only)
        self._ambig = set()          # shorts that collided (never resolved)
        self._n = 0                  # rows in use
        self._cap = 0                # allocated rows
        self._dim = 768
        self._vec_key = None         # VectorDAL.change_key() at last sync
        self._vec_watermark = 0      # max node_enrichments rowid ingested
        # title idf
        self._title_tok = {}         # row → frozenset(title tokens)
        self._title_df = {}          # token → doc frequency over titles
        self._titles_key = None
        self._titles_ts = 0.0
        # trace matrix — block list, no per-append copy
        self._tr_blocks = []         # [ndarray [k×768]]
        self._tr_meta = []           # [(chain_id, session_id)] aligned to blocks' rows
        self._tr_last = ''           # max created_at ingested
        # config overlay
        self._cfg = None
        self._cfg_ts = 0.0

    # ── config: the ONE gain-resolution seam (P4 swaps this body to g(φ(q))) ──
    def config(self, brain):
        now = time.monotonic()
        if self._cfg is not None and now - self._cfg_ts < CONFIG_TTL_S:
            return self._cfg
        cfg = dict(DEFAULT_CONFIG)
        try:
            stored = brain.get_interaction_config('recall_laf') or {}
            cfg.update({k: stored[k] for k in DEFAULT_CONFIG if k in stored})
        except Exception as e:
            # Loud: a broken K-store must be distinguishable from an empty one.
            # TTL-bounded, so this can't spam (≤1 log per CONFIG_TTL_S).
            try:
                brain._log_error('recall_laf_config', e,
                                 'get_interaction_config failed — running on '
                                 'module defaults')
            except Exception:
                pass
        self._cfg, self._cfg_ts = cfg, now
        return cfg

    # ── field matrices: growable buffers + rowid-watermark incremental upserts ──
    def _ensure_capacity(self, need):
        if need <= self._cap:
            return
        new_cap = max(int(need * _GROW), 64)
        for vt, m in self._mats.items():
            grown = np.full((new_cap, self._dim), np.nan, dtype=np.float32)
            grown[:self._n] = m[:self._n]
            self._mats[vt] = grown
        self._cap = new_cap

    def _row_for(self, node_id):
        """Row for node_id, appending a fresh NaN row (and short-id entry) if new."""
        i = self._idx.get(node_id)
        if i is not None:
            return i
        i = self._n
        self._ensure_capacity(self._n + 1)
        self._n += 1
        self._master.append(node_id)
        self._idx[node_id] = i
        s = node_id[:8]
        if s in self._short:
            del self._short[s]        # collision → ambiguous, never resolve
            self._ambig.add(s)
        elif s not in self._ambig:
            self._short[s] = i
        return i

    def _full_matrix_build(self, brain, model):
        gv = {vt: {} for vt in _ALL_VIEWS}
        for r in brain._vec_dal.get_all_vectors(vector_types=_ALL_VIEWS,
                                                model=model or None):
            uv = _unit(r.get('embedding'))
            if uv is not None:
                gv[r['vector_type']][r['node_id']] = uv
        master = sorted(set().union(*[set(d) for d in gv.values()]) if gv else set())
        self._dim = next((len(v) for d in gv.values() for v in d.values()), 768)
        self._n = len(master)
        self._cap = max(int(self._n * _GROW), 64)
        self._master = list(master)
        self._idx = {nid: i for i, nid in enumerate(master)}
        by_short = defaultdict(list)
        for nid in master:
            by_short[nid[:8]].append(nid)
        self._short = {s: self._idx[f[0]] for s, f in by_short.items() if len(f) == 1}
        self._ambig = {s for s, f in by_short.items() if len(f) > 1}
        self._mats = {}
        for vt in _ALL_VIEWS:
            m = np.full((self._cap, self._dim), np.nan, dtype=np.float32)
            for nid, v in gv[vt].items():
                m[self._idx[nid]] = v
            self._mats[vt] = m
        # titles are row-keyed — force a rebuild against the new row space
        self._title_tok, self._titles_key, self._titles_ts = {}, None, 0.0

    def _refresh_matrices(self, brain, model):
        key = brain._vec_dal.change_key()
        if self._mats is not None and key == self._vec_key:
            return
        if self._mats is None or key[0] < (self._vec_key or (0, 0))[0]:
            # first build, or rows were deleted — the watermark can't see
            # deletions, so rebuild from scratch (rare: revise/archive cleanup)
            self._full_matrix_build(brain, model)
        else:
            for rowid, nid, vt, blob in brain._vec_dal.vectors_since(
                    self._vec_watermark, vector_types=_ALL_VIEWS,
                    model=model or None):
                uv = _unit(blob)
                if uv is None or vt not in self._mats:
                    continue
                self._mats[vt][self._row_for(nid)] = uv
        self._vec_key = key
        self._vec_watermark = key[1]

    # ── title idf: full rebuild, TTL-throttled (cheap but not free) ──
    def _refresh_titles(self, brain):
        now = time.monotonic()
        if self._title_tok and now - self._titles_ts < TITLES_TTL_S:
            return
        key = brain._nodes.change_key()
        if key == self._titles_key and self._title_tok:
            self._titles_ts = now
            return
        tok, df = {}, defaultdict(int)
        for nid, title in brain._nodes.title_rows():
            i = self._idx.get(nid)
            if i is None:
                continue
            ts = frozenset(_IDF_TOK.findall(title.lower()))
            tok[i] = ts
            for t in ts:
                df[t] += 1
        self._title_tok, self._title_df = tok, dict(df)
        self._titles_key, self._titles_ts = key, now

    # ── trace matrix: block list — appends never copy the resident rows ──
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
        self._tr_last = last
        if not vecs:
            return
        self._tr_blocks.append(np.stack(vecs))
        self._tr_meta.extend(meta)
        if len(self._tr_blocks) > TRACE_BLOCK_CONSOLIDATE:
            self._tr_blocks = [np.vstack(self._tr_blocks)]   # one copy, occasional

    # ── episodic field (shared join semantics via roles_for_moments) ──
    def _episodic_vectors(self, brain, qv, cfg, n):
        """(pick, enc) [n] activation vectors from similar past surface-moments.

        Full-history block scan (no newest-500 cap), top `top_moments`
        (session, stop) moments deduped at max similarity, roles joined via the
        canonical shared roles_for_moments, ±window unioned, picked-wins.
        """
        pick = np.zeros(n)
        enc = np.zeros(n)
        if not self._tr_blocks:
            return pick, enc
        sims = np.concatenate([b @ qv for b in self._tr_blocks])
        want = int(cfg['top_moments'])
        moments = {}                          # (session, short, stop) → score
        for i in np.argsort(-sims)[:want * 3]:
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
            if len(moments) >= want:
                break
        records = roles_for_moments(brain, moments, cfg['window_turns'],
                                    cfg['session_trace_pull'])
        for r in records:
            s = r['score']
            for node in r['picked']:
                i = self._resolve(node)
                if i is not None and s > pick[i]:
                    pick[i] = s
            for node in r['encoded']:
                i = self._resolve(node)
                if i is not None and s > enc[i]:
                    enc[i] = s
        return pick, enc

    def _resolve(self, node_id):
        """Role node id (8-char short OR full) → master row, or None.
        Surface traces store 8-char shorts; encode deltas store full ids."""
        i = self._idx.get(node_id)
        return i if i is not None else self._short.get(node_id)

    # ── the field registry: name → per-node raw activation [n] ──
    def _fields(self, brain, query, qv, cfg, n):
        """Adding a lane = one entry here + a 'gain_<name>' DEFAULT_CONFIG key.
        Every vector is z-scored by the caller — return RAW activations."""
        with np.errstate(all='ignore'):
            maxsim = np.nanmax(
                np.stack([self._mats[vt][:n] @ qv for vt in MAXSIM_VIEWS]), axis=0)
        sit_raw = self._mats['_situation'][:n] @ qv
        pick, enc = self._episodic_vectors(brain, qv, cfg, n)
        return {
            'maxsim': maxsim,
            'pick': pick,
            'enc': enc,
            'idf': idf_scores(query, self._title_tok, self._title_df, n),
            'sit': np.where(np.isfinite(sit_raw), sit_raw, 0.0),
        }

    # ── the scorer ──
    def scores(self, brain, query, query_vec, model=None):
        """({node_id: score01}, telemetry) over every node with ≥1 embedding view.

        score01 is a monotonic sigmoid of the z-scored gain-weighted field sum,
        guaranteed ∈ (0,1) (the range contract brain_recall's floors/boosts
        assume); telemetry is {node_id: {field: z}} for the top nodes — the
        production feed for the P2 dataset walker.
        """
        qv = _unit(query_vec)
        if qv is None or not MAXSIM_VIEWS:
            return {}, None
        cfg = self.config(brain)
        with self._lock:
            self._refresh_matrices(brain, model)
            self._refresh_titles(brain)
            self._refresh_traces(brain)
            n = self._n
            if n == 0:
                return {}, None
            fields = self._fields(brain, query, qv, cfg, n)
            zf = {name: _zscore(vec, n) for name, vec in fields.items()}
            zsum = np.zeros(n)
            for name, z in zf.items():
                zsum = zsum + float(cfg['gain_' + name]) * z
            s01 = 1.0 / (1.0 + np.exp(-zsum / float(cfg['sigmoid_scale'])))
            if not np.all(np.isfinite(s01)):
                # out-of-contract vector — refuse to ship it; the caller's
                # try/except falls back to champion and logs the error
                raise ValueError('laf_v1 produced non-finite scores')
            score_map = {nid: float(s01[i])
                         for i, nid in enumerate(self._master[:n])}
            top = np.argsort(-s01)[:int(cfg['telemetry_top_n'])]
            telemetry = {self._master[i]: {name: round(float(z[i]), 3)
                                           for name, z in zf.items()}
                         for i in top}
            return score_map, telemetry


def get_engine(brain):
    """The per-Brain engine singleton (lazily attached)."""
    eng = getattr(brain, '_laf_engine', None)
    if eng is None:
        eng = LafV1Engine()
        brain._laf_engine = eng
    return eng
