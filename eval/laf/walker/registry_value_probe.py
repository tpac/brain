"""Is the relation registry worth building? — filter/ranking curve on the best walk.

Tom's critique of T2 (2026-07-29): the per-lane reach probe walked EVERY
complementary edge from every seed — the crudest possible traversal. Efficiency
(rescues per 100 fan-out nodes) is precisely what better edge information is
supposed to raise, so a BLIND walk understates a registry-informed one.

This probes each of docs/RELATION-REGISTRY-DESIGN.md's dimensions as a WALK
POLICY, against the cheap data-quality filters, on the best seed lane (`sit`,
2.3× maxsim per T2) over the miss population:

  sign        (§1.3) reinforcing → walk as reach; inhibitory → do NOT walk for
              reach (it is a suppression signal); gating → conditional.
              NOTE the doc defines sign FUNCTIONALLY (recall-polarity), not
              logically — so `resolves`/`addresses` are reinforcing even though
              the correction_improvement ASPECT claims them. This probe follows
              the doc, and the aspect-vs-sign divergence is reported.
  symmetry+   (§1.1/1.2) symmetric verbs walk both ways; asymmetric verbs walk
  direction   ONLY along the actor→recipient frame. The earlier global
              direction test (all-out vs all-in) found no discrimination; this
              is the per-verb-conditioned policy it could not test.
  transitivity(§1.4) "the guardrail for multi-hop spread — only propagate along
              composable chains." T2 showed hop2 efficiency collapsing
              0.35→0.08; if transitive-gating recovers it, the dimension pays.

Baselines it must beat to justify the build:
  B0 all verbs · B1 complementary-only (T2's walk) · B2 desc_len≥80 (the
  cheapest known prior: rescue median 118 chars vs noise 22) · B3 non-hub.
VERDICT RULE: if the registry policies do not beat B2/B3, the registry is not
the priority for THIS consumer — data quality is.

Verb dimensions are HAND-classified here by the doc's own derivation heuristic
(voice→direction, valence→sign) — a proxy for the S2 classifier's future
output, not its result. Coverage is printed; unclassified verbs are excluded
from policy arms and counted.

Read-only. Run: VECLIB_MAXIMUM_THREADS=3 nice -n 19 \
                ./dev python3 eval/laf/walker/registry_value_probe.py
"""
import json
import sys
from collections import Counter, defaultdict

import numpy as np

from walker_db import OUT_DIR, open_brain_ro

sys.path.append(str(OUT_DIR))
import laf_doors as D                                               # noqa: E402
import laf_lane_audit as A                                          # noqa: E402
from lambda_probe import zn                                         # noqa: E402
from edge_fusion_census import LAM, iso                             # noqa: E402

REPORT = OUT_DIR / 'registry_value_probe.md'
SEED_LANE = 'sit'
SEED_K = 25
DESC_MIN = 80
HUB_MAX = 60

# ── hand-classified registry proxy: verb → (sign, symmetry, transitive) ──
# sign is FUNCTIONAL recall-polarity per §1.3, not aspect membership.
REIN_ASYM = ('extends', 'grounds', 'implements', 'validates', 'refines',
             'instantiates', 'specifies', 'produces', 'strengthens',
             'formalizes', 'synthesizes', 'abstracts', 'operationalizes',
             'contextualizes', 'scopes', 'motivates', 'targets', 'maps',
             'advances', 'resolves', 'addresses', 'fixes', 'resolved_by',
             'addressed_by', 'improves', 'protects', 'explains_cause_of',
             'proves_equivalence_of', 'restates', 'illustrates', 'applies',
             'confirms', 'complements', 'informs', 'reframes', 'revises')
REIN_SYM = ('similar_to', 'related', 'related_to', 'parallels',
            'same_domain_as', 'co_accessed', 'co_anchored',
            'community_member', 'differs_from', 'parallels_design_of')
INHIB = ('corrects', 'supersedes', 'overrides', 'redefines', 'updates',
         'changes', 'modifies', 'rejected_for', 'preferred_over',
         'could_replace', 'weakens', 'challenges', 'contradicts',
         'corrected_by', 'superseded_by', 'absorbed_into',
         'consolidated_into', 'prevents')
GATING = ('enables', 'depends_on', 'requires', 'constrains',
          'resolves_constraint', 'configures', 'prerequisite_for',
          'triggers')
TRANSITIVE = ('depends_on', 'before', 'after', 'during', 'part_of',
              'supersedes', 'enables', 'community_member', 'absorbed_into',
              'consolidated_into', 'anchored_to')
TEMPORAL_REIN = ('before', 'after', 'during', 'simultaneous_with',
                 'anchored_to')


def classify(rel):
    """(sign, symmetric, transitive) or None when unclassified."""
    sym = rel in REIN_SYM
    trans = rel in TRANSITIVE
    if rel in INHIB:
        return ('inhibitory', rel in ('contradicts',), trans)
    if rel in GATING:
        return ('gating', False, trans)
    if rel in REIN_SYM:
        return ('reinforcing', True, trans)
    if rel in REIN_ASYM or rel in TEMPORAL_REIN:
        return ('reinforcing', False, trans)
    return None


def main():
    idx = json.loads((OUT_DIR / 'field_cache_index.json').read_text())
    m2i = {sid: i for i, sid in enumerate(idx['master'])}
    n_nodes = idx['n_nodes']

    b = open_brain_ro()
    # directed adjacency with the edge payload the policies need
    out_adj = defaultdict(list)   # actor-frame: source -> (target, ...)
    in_adj = defaultdict(list)
    partners = defaultdict(set)
    verb_count = Counter()
    for src, tgt, rel, dlen, created in b.execute(
            "SELECT e.source_id, e.target_id, r.relation, "
            "LENGTH(COALESCE(r.description,'')), e.created_at "
            "FROM edges e JOIN edge_relations r ON r.edge_id=e.edge_id "
            "WHERE (r.archived IS NULL OR r.archived=0)"):
        si, ti = m2i.get(src), m2i.get(tgt)
        if si is None or ti is None:
            continue
        rec = (rel, dlen or 0, created)
        out_adj[si].append((ti,) + rec)
        in_adj[ti].append((si,) + rec)
        partners[si].add(ti)
        partners[ti].add(si)
        verb_count[rel] += 1
    b.close()
    deg = {k: len(v) for k, v in partners.items()}

    covered = sum(c for v, c in verb_count.items() if classify(v))
    total = sum(verb_count.values())
    unclassified = [(v, c) for v, c in verb_count.most_common()
                    if not classify(v)]
    print('verb coverage: %d/%d edge-relations classified (%.1f%%); '
          'top unclassified: %s' % (covered, total, 100.0 * covered / total,
                                    unclassified[:8]))

    turns, _enr, n = D.build_corpus('2026-05-11')
    gains = np.array([A.GAINS[ln] for ln in A.LANES])

    # POLICIES: name -> (edge_ok(rel,dlen,tgt) , walk_both_ways)
    def pol_all(rel, dlen, tgt):
        return True                     # walk everything (B0 baseline)

    def pol_comp(rel, dlen, tgt):       # T2's walk (complementary class)
        c = classify(rel)
        return bool(c) and c[0] == 'reinforcing' and not c[1]

    def pol_desc(rel, dlen, tgt):
        return dlen >= DESC_MIN

    def pol_nonhub(rel, dlen, tgt):
        return deg.get(tgt, 0) <= HUB_MAX

    def pol_sign(rel, dlen, tgt):       # registry §1.3
        c = classify(rel)
        return bool(c) and c[0] == 'reinforcing'

    def pol_sign_desc(rel, dlen, tgt):
        return pol_sign(rel, dlen, tgt) and dlen >= DESC_MIN

    def pol_sign_desc_hub(rel, dlen, tgt):
        return pol_sign_desc(rel, dlen, tgt) and pol_nonhub(rel, dlen, tgt)

    POLICIES = [
        ('B0 all verbs, both ways', pol_all, True),
        ('B1 complementary-only (T2)', pol_comp, True),
        ('B2 desc≥%d only' % DESC_MIN, pol_desc, True),
        ('B3 non-hub only (deg≤%d)' % HUB_MAX, pol_nonhub, True),
        ('R1 sign=reinforcing', pol_sign, True),
        ('R2 sign + symmetry-aware direction', pol_sign, False),
        ('R3 sign + desc≥%d' % DESC_MIN, pol_sign_desc, True),
        ('R4 sign + desc + non-hub', pol_sign_desc_hub, True),
        ('R5 sign + desc + non-hub + sym-dir', pol_sign_desc_hub, False),
    ]

    res = {nm: defaultdict(int) for nm, _f, _s in POLICIES}
    fan = {nm: [] for nm, _f, _s in POLICIES}
    # hop2: transitive-gated vs ungated, on the best static policy (R4)
    h2 = defaultdict(int)
    h2fan = defaultdict(list)
    n_miss = 0

    for t in turns:
        U = np.flatnonzero(t['alive'])
        if U.size < 20:
            continue
        Z = np.column_stack([t['zl'][ln][U] for ln in A.LANES])
        f0 = Z @ gains
        if not np.isfinite(f0).any() or f0.std() <= 1e-9:
            continue
        zf0 = (f0 - f0.mean()) / f0.std()
        zmh = zn(t['mh'])[U]
        mix = LAM * zf0 + (1.0 - LAM) * zmh
        fin = np.where(np.isfinite(mix), mix, -np.inf)
        order_local = np.argsort(-fin)
        gpos = np.flatnonzero(U[order_local] == t['gr'])
        if not gpos.size or int(gpos[0]) + 1 <= 5:
            continue                                   # misses only
        gold_rank = int(gpos[0]) + 1
        n_miss += 1
        gr = int(t['gr'])
        tdt = t.get('turn_dt')
        arr = np.full(n_nodes, -np.inf)
        arr[U] = t['zl'][SEED_LANE][U]
        k = min(SEED_K, U.size)
        seeds = set(int(x) for x in U[np.argsort(-arr[U])[:k]])
        if gr in seeds:
            for nm, _f, _s in POLICIES:
                res[nm]['organic'] += 1
            continue

        def step(sources, ok, both, exclude):
            front, hit = set(), False
            for si in sources:
                walks = [(out_adj.get(si, ()), True)]
                if both:
                    walks.append((in_adj.get(si, ()), False))
                else:
                    # symmetry-aware: symmetric verbs also walk the in-edge
                    walks.append((
                        [e for e in in_adj.get(si, ())
                         if (classify(e[1]) or (None, False))[1]], False))
                for edges, _isout in walks:
                    for (oi, rel, dlen, created) in edges:
                        if oi in exclude:
                            continue
                        if not ok(rel, dlen, oi):
                            continue
                        edt = iso(created)
                        if tdt and edt and edt > tdt:
                            continue
                        front.add(oi)
                        if oi == gr:
                            hit = True
            return front, hit

        base_front, base_hit = set(), False   # bound unconditionally: the
        # hop2 stage below reads these, and keying them off a display STRING
        # made a rename silently raise NameError instead of failing loudly.
        for nm, ok, both in POLICIES:
            front, hit = step(seeds, ok, both, seeds)
            res[nm]['hop1'] += 1 if hit else 0
            if hit:
                res[nm]['new' if gold_rank > 25 else 'sort'] += 1
            fan[nm].append(len(front))
            if nm == 'R4 sign + desc + non-hub':
                base_front = front
                base_hit = hit
        # hop2 from R4's frontier: transitive-gated vs ungated
        if not base_hit:
            for tag, gate in (('ungated', False), ('transitive-only', True)):
                def ok2(rel, dlen, tgt, gate=gate):
                    if not pol_sign_desc_hub(rel, dlen, tgt):
                        return False
                    return (rel in TRANSITIVE) if gate else True
                f2, hit2 = step(base_front, ok2, True,
                                seeds | base_front)
                h2['%s_hit' % tag] += 1 if hit2 else 0
                h2fan['%s' % tag].append(len(f2))

    L = ['# Is the relation registry worth building? — walk-policy curve', '',
         'Seed lane **%s** top-%d (T2\'s best seed) · miss population n=%d · '
         'verb coverage %.1f%% of edge-relations' % (SEED_LANE, SEED_K, n_miss,
                                                     100.0 * covered / total),
         '', 'EFF = rescues per 100 nodes of fan-out. Registry arms (R*) must '
         'beat the cheap data-quality baselines (B2/B3) to justify the build.',
         '', '| policy | hop1 rescues | new reach | sorting | mean fanout | EFF |',
         '|---|---|---|---|---|---|']
    for nm, _f, _s in POLICIES:
        s = res[nm]
        fm = float(np.mean(fan[nm])) if fan[nm] else 0.0
        eff = (100.0 * s['hop1'] / max(n_miss, 1) / fm) if fm > 0 else 0.0
        L.append('| %s | %d | **%d** | %d | %.0f | **%.2f** |'
                 % (nm, s['hop1'], s['new'], s['sort'], fm, eff))
    L += ['', '## Transitivity as the multi-hop guardrail (§1.4)', '',
          'hop2 from the R4 frontier, on turns R4 did not already rescue.', '',
          '| hop2 policy | extra rescues | mean fanout | EFF |',
          '|---|---|---|---|']
    for tag in ('ungated', 'transitive-only'):
        fm = float(np.mean(h2fan[tag])) if h2fan[tag] else 0.0
        eff = (100.0 * h2['%s_hit' % tag] / max(n_miss, 1) / fm) if fm else 0.0
        L.append('| %s | %d | %.0f | **%.2f** |'
                 % (tag, h2['%s_hit' % tag], fm, eff))
    L += ['', '## Verb coverage', '',
          'classified %d/%d edge-relations (%.1f%%). Top unclassified: %s'
          % (covered, total, 100.0 * covered / total, unclassified[:10]), '']
    REPORT.write_text('\n'.join(L) + '\n')
    print('\n'.join(L))
    return 0


if __name__ == '__main__':
    sys.exit(main())
