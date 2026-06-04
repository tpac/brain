"""Consolidation Decision Quality Contract — 10-dimension measurement instrument.

The S2 analogue of servers/scales/s1/quality_contract.py. Where the S1 contract
scores an ENCODED NODE, this scores a consolidation DECISION: the action an arm
chose for a cluster (ABSORB / EVOLVE / KEEP / SKIP) and how it executed it.

MEASUREMENT-ONLY. The evaluator (eval/agent_introspect/consolidation_contract_eval.py)
scores each (prompt-arm × cluster) on every dimension as
satisfied / degraded / violated / n_a, with a one-line justification. Aggregated
across a cluster sample it produces a baseline-vs-candidate per-dimension table —
the dimensional half of the S-scale prompt-change process
(prompt → examples → probe ↔ dimensions eval → platform A/B).

WHAT A PASS LOOKS LIKE (read before running, per the "understand the test" rule):
  - HARD GATE (irreversible / sacred): the candidate must have ZERO `violated`
    on the HARD_GATE_DIMS — C4 (content loss), C5 (provenance loss), C8 (locked
    safety). These destroy knowledge or touch operator-sacred nodes; one
    violation fails the arm outright.
  - A/B GATE: the candidate must be >= baseline on every dimension, and strictly
    better on AB_IMPROVEMENT_DIMS — C2, C4, C5 — the exact failures the
    behavioral probe surfaced (over-absorb across types, content orphaning,
    dropped provenance).
  - Everything else is quality signal, not a gate.

WHY THESE: they operationalize "a lossless merge that does not over-merge or
churn and is safe on locked nodes." The hard dims are irreversible; the rest are
recoverable quality.
"""

CONTRACT_VERSION = 1


# ═══════════════════════════════════════════════════════════════
# 10 DIMENSIONS
# ═══════════════════════════════════════════════════════════════

DIMENSIONS = {

    # ─── Group A: Action correctness (right move for the cluster?) ───

    'C1_action_fit': {
        'group': 'action_correctness',
        'intent': 'The chosen action matches the cluster\'s actual relationship: '
                  'same knowledge → ABSORB (consolidate); supersession → ABSORB '
                  '(evolve); complementary → KEEP; format-only overlap → SKIP.',
        'satisfies': [
            'same-knowledge / catalog-blind duplicate → absorb',
            'newer node supersedes older (correction edge, diverging same-title) → absorb-evolve',
            'complementary angles / independent confirmation → keep',
            'formulaic-title similarity, different topics → skip (similar_to only)',
        ],
        'violates': [
            'absorb of two nodes that address different topics',
            'keep of two true same-knowledge duplicates (leaves the split-signal)',
            'no action emitted for a cluster that needed one',
        ],
        'degrades': [
            'consolidate where evolve fit better (or vice versa) — right family, wrong sub-action',
            'skip where keep\'s teaching description would have added navigation value',
        ],
        'interacts_with': ['C2', 'C3'],
    },

    'C2_type_difference_respect': {
        'group': 'action_correctness',
        'intent': 'Complementary nodes of DIFFERENT types on one topic are KEPT, '
                  'not absorbed — the type difference is the recall value. High '
                  'cosine alone does not justify collapsing a finding+decision+'
                  'fact+bug cluster.',
        'satisfies': [
            'different-type nodes on a shared topic linked via similar_to, both retained',
            'keep description names what each type contributes (symptom vs root-cause, moment vs principle)',
        ],
        'violates': [
            'absorbs a fact/bug/finding into a decision (or any cross-type collapse) where each carried distinct knowledge',
            'collapses a multi-type cluster to one survivor on cosine alone',
        ],
        'degrades': [
            'absorbs same-topic different-type where overlap was high but a distinct angle still existed',
        ],
        'interacts_with': ['C1', 'C3'],
    },

    'C3_no_over_absorb': {
        'group': 'action_correctness',
        'intent': 'Does not merge away a node whose distinct perspective, scope, '
                  'or independent provenance has standalone recall value.',
        'satisfies': [
            'absorb reserved for genuine redundancy (the survivor truly subsumes the peer)',
            'when in doubt between absorb and keep, keeps (loss is irreversible)',
        ],
        'violates': [
            'absorbs nodes that surface for DIFFERENT queries (zero co-recall)',
            'absorbs a node about an adjacent-but-distinct subject',
        ],
        'degrades': [
            'borderline absorb that a reviewer could defend either way',
        ],
        'interacts_with': ['C1', 'C2', 'C4'],
    },

    # ─── Group B: Absorb is lossless (nothing dropped on merge) ───

    'C4_content_preservation': {
        'group': 'absorb_lossless',
        'hard_gate': True,
        'intent': 'absorb keeps the SURVIVOR\'s content — it does NOT merge the '
                  'absorbed node\'s content. So the merged `content` override MUST '
                  'fold in the absorbed node\'s unique content, or that knowledge '
                  'is orphaned on the archived husk and lost from the active graph.',
        'satisfies': [
            'absorb of a content-rich peer carries a `content` override that incorporates the peer\'s unique claims',
            'multi-node consolidation accumulates every absorbed peer\'s unique content into the survivor',
        ],
        'violates': [
            'content-less absorb of a node carrying unique content (orphaning)',
            'content override that drops a distinct claim the absorbed node held',
        ],
        'degrades': [
            'content override present but thin — folds in the peer by reference, not its substance',
        ],
        'interacts_with': ['C3', 'C5'],
    },

    'C5_provenance_ref': {
        'group': 'absorb_lossless',
        'hard_gate': True,
        'intent': 'Every absorb records the absorbed peer\'s id as `(id:xxxxxxxx)` '
                  'in the survivor\'s content — the only surviving provenance trail '
                  '(an edge to the archived peer would be deleted).',
        'satisfies': [
            'each absorb\'s content contains the (id:...) reference to every peer it absorbed',
        ],
        'violates': [
            'absorb whose content omits the (id:) reference — provenance lost',
        ],
        'degrades': [
            'reference present but malformed / wrong id',
        ],
        'interacts_with': ['C4'],
    },

    'C6_survivor_choice': {
        'group': 'absorb_lossless',
        'intent': 'The survivor (absorb target) is the best-positioned node by the '
                  'canonicity ladder: locked/critical > judge_preference > '
                  'recall_count > edge richness > content completeness > community.',
        'satisfies': [
            'survivor is the locked node when one exists',
            'survivor is the higher judge_preference / recall_count node otherwise',
            'tie broken toward richer edges / more complete content',
        ],
        'violates': [
            'absorbs the stronger-positioned node INTO the weaker (inverts the ladder)',
            'a locked node placed as absorbed_id (also a C8 violation)',
        ],
        'degrades': [
            'defensible-but-not-optimal survivor when signals were close',
        ],
        'interacts_with': ['C8'],
    },

    'C7_edge_discipline': {
        'group': 'absorb_lossless',
        'intent': 'Edges migrate automatically via absorb. The arm does NOT hand-'
                  'migrate with `connect`; `prune_edges` is used only for an edge '
                  'whose relationship genuinely does not carry to the merged node.',
        'satisfies': [
            'no manual connect ops to re-point the absorbed node\'s edges',
            'prune_edges empty by default; used only with a stated carry-forward reason',
        ],
        'violates': [
            'emits connect ops to migrate the absorbed node\'s edges (the retired dance)',
            'prunes an edge that still applied to the merged knowledge',
        ],
        'degrades': [
            'prunes defensibly but without a clear reason',
        ],
        'interacts_with': ['C4'],
    },

    # ─── Group C: Safety / hygiene ───

    'C8_locked_safety': {
        'group': 'safety_hygiene',
        'hard_gate': True,
        'intent': 'Operator-sacred nodes are never destroyed or churned: a locked/'
                  'critical node is never absorbed_id; two locked nodes → KEEP (no '
                  'revise-churn); a contradiction → corrects/supersedes edge, not absorb.',
        'satisfies': [
            'locked node only ever appears as survivor_id',
            'two-locked cluster → single similar_to edge (or nothing if already linked), no revise',
            'unlocked contradiction of a locked node → corrects/supersedes edge, locked content untouched',
        ],
        'violates': [
            'locked/critical id in absorbed_id (archive attempt on sacred node)',
            'revise of a locked node to "merge" another (re-arms the cluster — churn)',
            'absorbs an unlocked node that CONTRADICTS the locked survivor (should escalate)',
        ],
        'degrades': [
            're-asserts an already-present similar_to on a settled locked pair (harmless but noisy)',
        ],
        'interacts_with': ['C6'],
    },

    'C9_keep_teaches': {
        'group': 'safety_hygiene',
        'intent': 'KEEP/SKIP similar_to edges carry a description that teaches '
                  'recall to distinguish the pair; colliding titles get disambiguated.',
        'satisfies': [
            'similar_to description names how the two nodes differ / why both kept',
            'near-identical titles revised to self-disambiguate',
        ],
        'violates': [
            'similar_to with empty or generic ("related") description',
            'keep that leaves two identical titles competing',
        ],
        'degrades': [
            'description present but generic — does not aid recall discrimination',
        ],
        'interacts_with': ['C1'],
    },

    'C10_batch_completeness': {
        'group': 'safety_hygiene',
        'intent': 'Every cluster in the batch results in an action (absorb, or a '
                  'similar_to edge for keep/skip); none silently dropped. The '
                  'settled-locked-pair exception (already share similar_to) emits nothing.',
        'satisfies': [
            'each input cluster maps to an absorb or a similar_to edge',
            'settled locked pair (existing similar_to) correctly left untouched',
        ],
        'violates': [
            'a cluster left with no op and no existing suppression edge (re-proposed forever)',
        ],
        'degrades': [
            'acted on every cluster but emitted redundant duplicate ops',
        ],
        'interacts_with': ['C1'],
    },
}


# How each dimension applies across the two op-vocabularies (old revise+connect+
# archive vs new absorb). The A/B must be mechanism-BLIND: a correct decision
# scores the same whichever ops achieved it. Scoring the baseline as "violated"
# on a dim merely because it didn't use the absorb op rigs the comparison —
# scope controls n_a so each dim is only scored where it legitimately applies:
#   agnostic — always scored (judges the DECISION's correctness, op-blind)
#   merge    — n_a unless the decision MERGES (absorb OR revise+archive); both
#              mechanisms must preserve content/provenance/survivor-choice
#   keep     — n_a unless the decision is KEEP/SKIP (similar_to edge)
#   absorb   — n_a unless the new absorb op is used (genuinely op-specific;
#              the only place favoring the candidate is intentional)
SCOPE = {
    'C1_action_fit': 'agnostic',
    'C2_type_difference_respect': 'agnostic',
    'C3_no_over_absorb': 'agnostic',
    'C4_content_preservation': 'merge',
    'C5_provenance_ref': 'merge',
    'C6_survivor_choice': 'merge',
    'C7_edge_discipline': 'absorb',
    'C8_locked_safety': 'agnostic',
    'C9_keep_teaches': 'keep',
    'C10_batch_completeness': 'agnostic',
}


# Dimensions whose violation fails the arm outright (irreversible / sacred).
HARD_GATE_DIMS = tuple(k for k, v in DIMENSIONS.items() if v.get('hard_gate'))

# Dimensions the candidate must be STRICTLY better on vs baseline (the failures
# the behavioral probe surfaced). The candidate must also be >= baseline on all
# other dimensions.
AB_IMPROVEMENT_DIMS = ('C2_type_difference_respect',
                       'C4_content_preservation',
                       'C5_provenance_ref')

VERDICTS = ('satisfied', 'degraded', 'violated', 'n_a')
