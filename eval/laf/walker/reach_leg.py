"""Q1 reach leg — gold-24 through the engine's as_of path (§20.5).

Scores each gold cue's NEED coverage @5/@25 under a moment-shape config,
full-field (never pool-restricted), with as_of=cutoff doing eligibility
natively — no rank-compression workaround, the engine time-travels.

Message stack per cue (24 cues, manifest-matched to (session, stop)):
  j=0 op      = embedder.embed_query(cue text)  — the production query side
  j≥1 op      = doc-side trace vector of the LAST user_message at stop-j
  j≥1 anchor  = doc-side trace vector of the LAST assistant_message at
                stop-j (j=0 anchor excluded — the temporal-leak rule)
  texts       = trace_embeddings.text (500-cap column — which IS idf's
                production query cap)
Lanes per message, all through production code: maxsim/sit = engine
matrices @ vec; idf = eng._idf_asof(text, n, cutoff); pick/enc =
episodic_from_sims (the table build's engine-parity function) at
as_of=cutoff. Composition = q1_sweep.compose — ONE implementation shared
with the rank leg, no cross-leg drift.

SELF-CHECKS (must pass before any grid number is trusted):
  base-parity  K0 composition must rank the field IDENTICALLY to the
               production engine's scores(as_of=cutoff) (top-25 sequence
               equality) — proves this harness IS the engine at the base
               point, not a reimplementation.
  positive     pick-only / enc-only rankings must reproduce the measured
               ±1-turn episodic result (node 9634cce9: pick 8%/16%,
               enc 6%/14% need-reach @5/@25) within the gold-24 ±4pp
               noise band (§20.5).

Run:  ./dev python3 eval/laf/walker/reach_leg.py            (self-checks + K0)
      grid configs run via q1_sweep --grid (gated on the final H2 look).
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

from walker_db import WALKER_DIR, GOLD_DIR

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from servers.recall_laf import (LafV1Engine, MAXSIM_VIEWS, _unit,  # noqa: E402
                                DEFAULT_CONFIG)
from q1_sweep import compose, stack_messages, configs, weights, GAINS  # noqa: E402
from episodic_roles import K_MAX                                       # noqa: E402

TEXT_CAP = 500
POSITIVE_EXPECT = {'pick': (0.08, 0.16), 'enc': (0.06, 0.14)}   # 9634cce9
NOISE_PP = 0.042                       # gold-24 ±4pp band (§20.5) + slack
REPORT = WALKER_DIR / 'reach_leg.md'


def load_cues():
    """[{cue_id, session, stop, cutoff, text, needs: {need: {shorts}}}]"""
    manifest = json.loads((WALKER_DIR / 'gold_manifest.json').read_text())
    gold = json.loads((GOLD_DIR / 'frozen_gold_24.json').read_text())
    moments = {m['cue_id']: m for m in
               json.loads((GOLD_DIR / 'moments.json').read_text())}
    cues = []
    for cue in manifest['cues']:
        if not cue.get('matched'):
            continue
        cid = cue['cue_id']
        needs = defaultdict(set)
        for t in ('gold_plus', 'gold'):
            for it in gold[cid]['tiers'].get(t, []):
                needs[(it.get('need') or it['node_id']).strip()].add(
                    it['node_id'])
        cues.append({'cue_id': cid, 'session': cue['session_id'],
                     'stop': cue['stop'], 'cutoff': gold[cid]['cutoff'],
                     # production caps recall queries at 500 — grid legs use
                     # the capped text; the positive control replays the
                     # ORIGINAL audit's query, which was uncapped
                     'text': ((moments[cid]['cue'] or {}).get('text')
                              or '')[:TEXT_CAP],
                     'text_full': (moments[cid]['cue'] or {}).get('text')
                     or '',
                     'needs': dict(needs)})
    return cues


def stack_rows(brain, session, stop):
    """{(j, kind): (text, unit_vec)} from the trace substrate — the LAST
    user/assistant message per stop, stops S..S-K_MAX."""
    lo = max(stop - K_MAX, 0)
    rows = brain._trace_dal.conn.execute(
        "SELECT te.chain_id, te.ref_type, tem.text, tem.vector "
        "FROM trace_events te JOIN trace_embeddings tem ON tem.trace_id=te.id "
        "WHERE te.session_id=? AND te.scale='s0' AND te.ref_type IN "
        " ('user_message','assistant_message') AND tem.vector IS NOT NULL "
        "ORDER BY te.created_at ASC", (session,)).fetchall()
    out = {}
    for chain, ref_type, text, blob in rows:
        tail = str(chain).rsplit('-', 1)[-1]
        if not tail.isdigit():
            continue
        s = int(tail)
        if not (lo <= s <= stop):
            continue
        j = stop - s
        kind = 'op' if ref_type == 'user_message' else 'anchor'
        uv = _unit(blob)
        if uv is not None:
            out[(j, kind)] = ((text or '')[:TEXT_CAP], uv)   # last wins (ASC)
    return out


def cue_fields(eng, trace_mask, cue, q0):
    """{lane: [n × M_slots]} message-lane matrices for one cue.
    Slot layout: op columns j=0..K_MAX then anchor columns j=0..K_MAX
    (anchor j=0 stays NaN — stack_messages slices it away).

    pick/enc call the PRODUCTION _episodic_vectors directly — at 24 cues ×
    ≤17 messages the per-call gather cost is affordable, and it removes the
    argsort-tie divergence the role-map shortcut carries (the shortcut
    stays only in the 40k-call walker table build, where its own
    self-check gates it)."""
    n = eng._n
    cutoff = cue['cutoff']
    cfg = dict(DEFAULT_CONFIG)
    slots = {}
    slots[(0, 'op')] = (cue['text'], q0)
    for (j, kind), v in stack_rows(eng._brain_ref, cue['session'],
                                   cue['stop']).items():
        if (j, kind) == (0, 'op'):
            continue                       # query side owns j=0 op
        slots[(j, kind)] = v
    op = {ln: np.full((n, K_MAX + 1), np.nan) for ln in GAINS}
    an = {ln: np.full((n, K_MAX + 1), np.nan) for ln in GAINS}
    for (j, kind), (text, vec) in slots.items():
        tgt = op if kind == 'op' else an
        with np.errstate(all='ignore'):
            tgt['maxsim'][:, j] = np.nanmax(
                np.stack([eng._mats[vt][:n] @ vec for vt in MAXSIM_VIEWS]),
                axis=0)
        tgt['sit'][:, j] = eng._mats['_situation'][:n] @ vec
        if text:
            tgt['idf'][:, j] = eng._idf_asof(text, n, cutoff)
        pick, enc = eng._episodic_vectors(eng._brain_ref, vec, cfg, n,
                                          as_of=cutoff,
                                          trace_mask=trace_mask)
        tgt['pick'][:, j] = pick
        tgt['enc'][:, j] = enc
    return op, an


def original_recipe_roles(eng, brain, q0, cutoff):
    """The 9634cce9 measurement's OWN moment selection, reproduced on this
    substrate: newest-500 traces before cutoff (the recall_episodes cap),
    top-15 ROWS by cosine, deduped to (session, short, stop) at max score,
    ±1-turn roles via the production roles_for_moments. The engine
    deliberately dropped the 500-cap (P1: 'a coverage ceiling, not a
    feature') — so the positive control must re-create the cap to compare
    against the number measured under it."""
    from servers.recall_laf import roles_for_moments
    created = np.asarray(eng._tr_created, dtype='<U40')
    elig = np.flatnonzero(created < cutoff)
    elig = elig[np.argsort(created[elig])][-500:]          # newest 500
    mat = np.vstack(eng._tr_blocks)[elig]
    sims = mat @ q0
    moments = {}
    for pos in np.argsort(-sims)[:15]:                     # top-15 ROWS
        i = elig[pos]
        chain = eng._tr_meta[i][0] or ''
        parts = chain.split('-')
        tail = parts[-1]
        sess = eng._tr_meta[i][1]
        if not tail.isdigit() or not sess or len(parts) < 3:
            continue
        key = (sess, parts[1], int(tail))
        s = float(sims[pos])
        moments[key] = max(moments.get(key, 0.0), s)
    return roles_for_moments(brain, moments, 1, 2000)


def rank_rows(score, node_mask):
    """Descending row order over eligible rows only."""
    s = np.where(node_mask, score, -np.inf)
    s = np.where(np.isfinite(s), s, -np.inf)
    return np.argsort(-s)


def needs_reach(eng, order, needs, at):
    top = set()
    for r in order[:at]:
        nid = eng._master[r]
        top.add(nid[:8])
    return sum(1 for _n, shorts in needs.items() if shorts & top), len(needs)


def cue_need_fraction(eng, order, needs, at):
    """The ORIGINAL audit's per-cue metric (gold24_episodic_audit.need_hit):
    fraction of this cue's needs with any node in the top-at. The positive
    control macro-averages this over cues WITH episodes — metric parity,
    not just recipe parity."""
    if not needs:
        return None
    top = {eng._master[r][:8] for r in order[:at]}
    return sum(1 for shorts in needs.values() if shorts & top) / len(needs)


def main():
    import servers.embedder as embedder
    from tests.isolated_brain import IsolatedBrain
    cues = load_cues()
    lines = ['# reach_leg — gold-24 self-checks + K0 (§20.5 reach leg)', '']
    fails = []
    with IsolatedBrain() as env:
        eng = LafV1Engine()
        with eng._lock:
            eng._refresh_matrices(env.brain, None)
            eng._refresh_titles(env.brain)
            eng._refresh_traces(env.brain)
        eng._brain_ref = env.brain
        n = eng._n
        k0 = configs()[0]
        w0 = weights(k0)
        cov = {5: [0, 0], 25: [0, 0]}
        sub = {ln: {5: [], 25: []} for ln in ('pick', 'enc')}
        eng_recipe = {ln: {5: [], 25: []} for ln in ('pick', 'enc')}
        parity_ok = 0
        for cue in cues:
            q0 = _unit(embedder.embed_query(cue['text']))
            node_mask, trace_mask = eng._asof_masks(cue['cutoff'], n)
            op, an = cue_fields(eng, trace_mask, cue, q0)
            # base-parity: K0 composition ≡ production scores(as_of)
            mats = {}
            ww = None
            for ln in GAINS:
                mats[ln], ww = stack_messages(op[ln], an[ln], w0, k0)
            s_k0 = compose(mats, ww, k0, n, mask=node_mask)
            order = rank_rows(s_k0, node_mask)
            smap, _tel = eng.scores(env.brain, cue['text'], q0,
                                    as_of=cue['cutoff'])
            eng_order = [nid for nid, _s in sorted(smap.items(),
                                                   key=lambda kv: -kv[1])][:25]
            mine = [eng._master[r] for r in order[:25]]
            if mine == eng_order:
                parity_ok += 1
            else:
                fails.append('base-parity MISMATCH on %s' % cue['cue_id'])
            for at in (5, 25):
                hit, tot = needs_reach(eng, order, cue['needs'], at)
                cov[at][0] += hit
                cov[at][1] += tot
            # engine-recipe single-lane numbers (informational baseline;
            # same per-cue metric as the control for comparability)
            for ln in ('pick', 'enc'):
                s_ln = op[ln][:, 0]         # j=0 message, lane alone
                order_ln = rank_rows(np.where(np.isfinite(s_ln), s_ln,
                                              -np.inf), node_mask)
                for at in (5, 25):
                    f = cue_need_fraction(eng, order_ln, cue['needs'], at)
                    if f is not None:
                        eng_recipe[ln][at].append(f)
            # positive control: the ORIGINAL 9634cce9 recipe (newest-500
            # capped scan, top-15 rows, UNCAPPED query text) AND the
            # original METRIC (per-cue need fraction, macro-averaged,
            # empty-feed cues excluded)
            q_full = _unit(embedder.embed_query(cue['text_full']))
            records = original_recipe_roles(eng, env.brain, q_full,
                                            cue['cutoff'])
            if records:
                for ln, role in (('pick', 'picked'), ('enc', 'encoded')):
                    act = np.zeros(n)
                    for r in records:
                        for nid in r[role]:
                            row = eng._resolve(nid)
                            if row is not None and r['score'] > act[row]:
                                act[row] = r['score']
                    order_ln = rank_rows(np.where(act > 0, act, -np.inf),
                                         node_mask)
                    for at in (5, 25):
                        f = cue_need_fraction(eng, order_ln, cue['needs'],
                                              at)
                        if f is not None:
                            sub[ln][at].append(f)
    lines.append('- cues: %d; base-parity (K0 ≡ engine.scores(as_of), '
                 'top-25 sequence): %d/%d %s'
                 % (len(cues), parity_ok, len(cues),
                    'PASS' if parity_ok == len(cues) else 'FAIL'))
    if parity_ok != len(cues):
        fails.append('base-parity %d/%d' % (parity_ok, len(cues)))
    k0_reach = {at: cov[at][0] / max(cov[at][1], 1) for at in (5, 25)}
    lines.append('- K0 need-reach: @5 %.1f%% (%d/%d) · @25 %.1f%% (%d/%d)'
                 % (100 * k0_reach[5], cov[5][0], cov[5][1],
                    100 * k0_reach[25], cov[25][0], cov[25][1]))
    lines.append('')
    lines.append('## positive control — ±1-turn episodic reproduction '
                 '(9634cce9, ORIGINAL newest-500 recipe, envelope ±%.1fpp)'
                 % (100 * NOISE_PP))
    for ln in ('pick', 'enc'):
        got = {at: float(np.mean(sub[ln][at])) if sub[ln][at] else 0.0
               for at in (5, 25)}
        eg = {at: (float(np.mean(eng_recipe[ln][at]))
                   if eng_recipe[ln][at] else 0.0) for at in (5, 25)}
        exp5, exp25 = POSITIVE_EXPECT[ln]
        ok = (abs(got[5] - exp5) <= NOISE_PP
              and abs(got[25] - exp25) <= NOISE_PP)
        lines.append('- %s-only: @5 %.1f%% (expect %.0f%%) · @25 %.1f%% '
                     '(expect %.0f%%) → %s   [engine recipe, uncapped: '
                     '@5 %.1f%% · @25 %.1f%%]'
                     % (ln, 100 * got[5], 100 * exp5, 100 * got[25],
                        100 * exp25, 'PASS' if ok else 'FAIL',
                        100 * eg[5], 100 * eg[25]))
        if not ok:
            fails.append('positive control %s' % ln)
    lines.append('')
    lines.append('**Overall: %s**' % ('PASS — reach harness trusted'
                                      if not fails else
                                      'FAIL: ' + '; '.join(fails)))
    REPORT.write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines))
    return 0 if not fails else 1


if __name__ == '__main__':
    sys.exit(main())
