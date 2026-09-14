# Aspect exclusion policies — one table, one owner

**Status:** proposal, 2026-09-13. Not started. Raised while landing edge_context invalidation, when
the noise exclusion reached its sixth consumer through its fifth delivery shape.

## The problem

"Which relations does consumer X ignore" is answered in many places, in many shapes. Each answer is
individually defensible; together they are the drift class the aspect registry was built to end:
a policy stated per consumer, at the consumer, in whatever form the consumer found convenient.

Inventory (read from the code, 2026-09-13):

| consumer | policy | where it is stated | delivery shape |
|---|---|---|---|
| node pulls (`get_node` connections, encoder catalog, healer, consolidation loader) | noise | `aspects.structural_exclusions`, derived at adopt | registry attribute, read directly |
| graph dynamics (traverse, spread, `graph_expand`) | noise − `community_member` | `aspects.traversal_exclusions` | registry attribute, read directly (3 sites) |
| spread ride-along | lineage − traversal set | `aspects.lineage_relations` | registry attribute |
| edge_context text and its invalidation | noise | `aspects.structural_exclusions` | attribute copied onto `GraphDAL.edge_context_excluded` at Brain construction, plus a `find_missing(exclude_relations=…)` parameter |
| encoder relation vocabulary (`render_edge_aspects_block`) | generic_relation, noise, survivor_lineage | `EDGE_ASPECT_PROMPT_SKIP` in `aspects.py` | module tuple of aspect names |
| community typed adjacency (decoder and structural stamper) | generic_relation, noise **plus** literal `community_member` | `ADJACENCY_SKIP_ASPECTS` and `ADJACENCY_EXCLUDED_RELATIONS` in `community_contract.py` | one tuple of aspect names, one tuple of verbs, side by side |
| community neighbourhood fingerprint | noise, generic_relation | `relations_in([...])` inline in `community_decoder.py` | inline registry call |
| community cohesion check | 5 literal verbs | `non_cohesion_relations` in `COMMUNITY_DETECTION` config | interaction-config verb list |
| consolidation suppression | settlement aspect, fallback 8 literal verbs | `suppression_relations(brain)` in `consolidation_contract.py` | derived function with a hardcoded fallback |
| encoder catalog noise strip, correction dedup | noise; correction_improvement | `relations_in([...])` inline in `encode_contract.py` | inline registry call |
| absorb edge migration | `community_member` only, by endpoint types | `absorb_migrates_relation` in `dal_graph.py` | literal in a rule function (deliberate, tested) |
| archive exemption | survivor_lineage | `relations_in(['survivor_lineage'])` in `brain_remember.py` | inline registry call |

Twelve consumers. Three of them still carry literal verbs (`non_cohesion_relations`,
`ADJACENCY_EXCLUDED_RELATIONS`, the suppression fallback); the rest resolve through the registry but
each in its own way. Nothing lists them together, so the question Tom asked in July, "is there one
JSON that owns everything", still has the answer: the *verbs* are owned, the *policies* are not.

## The proposal

Give the registry a **policy table**: named exclusion sets, each declared as aspect names and
resolved to verbs once at adopt time, next to where `structural_exclusions` and
`traversal_exclusions` are already computed.

```
EXCLUSION_POLICIES = {
    'reads':              {'skip': ['noise']},
    'traversal':          {'skip': ['noise'], 'keep': ['community_member']},
    'edge_context':       {'skip': ['noise']},
    'encoder_vocabulary': {'skip': ['generic_relation', 'noise', 'survivor_lineage']},
    'community_adjacency':{'skip': ['generic_relation', 'noise']},
    'community_cohesion': {'skip': ['noise', 'generic_relation']},   # today: 5 literal verbs
}
```

Consumers read `brain.aspects.excluded('edge_context')` and get a frozenset of verbs. The two
existing attributes become entries (`structural_exclusions` is `reads`, `traversal_exclusions` is
`traversal`); `lineage_relations` keeps its own derivation but subtracts `excluded('traversal')`.

**Where the table lives.** In `aspects_v1.json` if a policy is data the S2 aspect encoder may one day
touch; in `aspects.py` beside the adopt-time derivations if it is code's to own. My take: `aspects.py`.
The policies name *consumers*, which are code, and the July finding already records that four aspect
metadata facts live in Python on purpose. A JSON policy that names a Python consumer drifts the other
way.

**What each consumer does.**

| consumer | change |
|---|---|
| node pulls, graph dynamics, spread | rename the attribute read to the policy read; no behaviour change |
| edge_context | `Brain.__init__` assigns `excluded('edge_context')` to the DAL; `find_missing` callers pass the same |
| encoder vocabulary | `render_edge_aspects_block` reads the policy; delete `EDGE_ASPECT_PROMPT_SKIP` |
| community adjacency | decoder and stamper read `excluded('community_adjacency')`; delete both tuples. `community_member` is in noise, so the literal tuple was redundant with the aspect tuple beside it |
| community cohesion | `non_cohesion_relations` becomes a policy read plus `related`/`related_to` from `generic_relation`; the two dream verbs are legacy (dreams are paused) and should be checked for live rows before dropping |
| community fingerprint, encoder strip, correction dedup, archive exemption | inline `relations_in` calls become policy reads only where the set is an *exclusion*; `relations_in(['correction_improvement'])` and `['survivor_lineage']` are selections, not exclusions, and stay |
| consolidation suppression | stays a selection from the settlement aspect; the literal fallback is the one to question separately |
| absorb migration | stays. It is a rule over endpoint types, not an exclusion set |

**Enforcement.** One test in `tests/test_aspects_*.py`: every `relations_in([...])` or verb tuple
in `servers/` whose contents are a subset of a policy's resolved set must read the policy instead
(a source grep, like the raw-SQL ratchet). The table's `keep` entries must name a verb that is in the
`skip` set, or the carve-out is dead.

## Why this shape

- **One answer to "what does X ignore".** Today it takes a grep across seven files.
- **New verbs propagate.** The three literal lists are frozen at time of typing; the taxonomy grows
  every S2 cycle. Policies resolve at adopt, so a verb the classifier adds to noise is excluded
  everywhere the next boot.
- **Delivery stops varying.** Attribute, tuple, inline call, parameter, config list collapse to one
  read. Tonight's `GraphDAL.edge_context_excluded` copy stays a copy (the DAL cannot hold a brain),
  but it is assigned from the table like everything else.

## Cost and risk

- Six consumers, three of them S2 units with their own parity tests (decoder ↔ stamper adjacency is
  parity-critical and already guarded).
- `non_cohesion_relations` and the suppression fallback may be literal on purpose. Read their
  history before folding them in; the plan should list them as "decide", not "convert".
- No recall or encoding behaviour changes if the resolved sets are byte-identical before and after.
  Pin that with a one-shot equality check per policy at the first boot after the change.

## Sequencing

Dependency-ordered, each step cold-runnable in its own session:

1. Add the table and `excluded()` to `AspectRegistry`; alias the two existing attributes to it. Tests.
2. Move edge_context, traversal, reads consumers to policy reads. Delete nothing yet.
3. Community adjacency: both tuples → one policy read; parity test unchanged.
4. Encoder vocabulary: `EDGE_ASPECT_PROMPT_SKIP` → policy.
5. Decide `non_cohesion_relations` and the suppression fallback with Tom; convert or document.
6. Ratchet test; delete the aliases.

Run this through the architecture-review skill before step 1: it recalls the June DAL ruling
(49d734ad), the July inventory (cf731a70) and the aspect-ownership plan, and will catch any policy
the brain already decided differently.
