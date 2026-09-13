# S1E V3.5 — authored, frozen, cells pinned (2026-09-12)

Candidate: `eval/fixtures/s1e_guide_v3_5_2026-09-12/` (arm `v3_5_titles`
746fed199eb0…, parent V3.4 `v3_4_titles` e3e22515…, live tool schemas as V3.4 froze them).
Template 106,775 → 109,209 chars (+2,434, +2.3%; production 111,323); gist 5,703 → 5,894;
strategy and closure are the parent's. Static checks pass. Nothing is merged, deployed or
committed; no model call has been made on the candidate.

## Where it came from

1. **Tom's call after the 12-pack blind read:** spend on the weave, not on repeat 3, and
   implement my take on the production-over-V3.4 overlay
   (`eval/fixtures/s1e_guide_v3_5_overlay_2026-09-12/OVERLAY-REVIEW.md`): its seven EDITs
   taken, its one ADD (A1, "I am the source — the graph's shape this turn is my call") cut —
   the only change with no measured carrier, and it pushes on the register where V3.4 holds
   its clearest blind advantage (evidence, ownership and scope 21:4 over twelve packs).
2. **The three defect classes every arm showed on the fresh cell** — a revise leaves the
   value where the encoder was not looking; firming up past the evidence; the assistant's
   own words missing from its nodes — woven at the carriers the full read of V3.4 located.
3. **The full read itself** (`READ-AUDIT-V3-4.md`, Tom's standing method id:71eeff20) found
   that the walk-every-surface RULE already exists in V3.4 (L258, gist targets); what is
   missing is a worked op that depicts an old value in a trigger or edge why on a plain count
   node, and a roll-call that names event_time, type and the quotes. It also surfaced two
   ledger-conformance defects V3.4 inherited (E17 title in the sweep; two whys arguing from
   "I persist" after the node was retitled).

## The changes (author.py is the record; every edit is an exact-once replacement)

| Id | Where (V3.4 line) | Change | Ledger boxes | Chars |
|---|---|---|---|---:|
| X1 | Fields, five-surfaces sentence (87) | a value that IS the claim rides title, content and quote — "three paths to one fact, and three surfaces a later revise must walk" | D8, E11, T6; pairs with W6 | +240 |
| X2 | Detail and meaning (708) | the generative half restored beside the restraint half, gated by retrieval divergence and scope; production's "principle each one points to" and "half the rate" NOT imported | D1, D7 | +271 |
| X3 | Traps sentence in Actions (268) | "one batch carries as many nodes as the window earned rather than the two that feel tidy, and a half-populated node free-rides on a title match" | D1, B6/B7 | +169 |
| X4 | Nadia reading (167) | "my own paraphrase of what she experienced never outranks her own wording for it" | D10, D3 exception | +82 |
| X5 | gist, `new` roll-call | "and a `thought` when I have a hunch or a connection of my own worth keeping beside it" | E21, E23 | +83 |
| X6 | Gate sentence (264) | "— it rides in `thought`" | T8, E18 | +24 |
| X7 | Timeline sample (21) | an encoded="true" turn with provenance and trimmed-actions stub depicted | E12, C6 | +394 |
| W6a | Yoga excerpt + op + ladder (720–763) | the excerpt gains `Situation: … her twice-a-week yoga is a fixed slot` and an Edges line whose why carries the count; the op swaps that why; the comment and the ladder sentence name the trigger and the edge | E12, A10 (+1 why-swap op), D11, E13 | +772 |
| W6b | gist, targets roll-call | "`thought` and the quotes when present, `event_time` and `type` when the window moved when a thing happened or what the node now is" | E21, E23, A10 | +108 |
| W6c | Sweep, a45c88f1 title (875) | "Rollout order ruling — superseded 2024-03-02 when auth-rewrite was scrapped" (the dead sequence no longer embedded) | E17 conviction 1 | −23 |
| W7 | Evidence sentence (56) + traps (268) | "Progress, completion and confirmation need their own evidence; a related success, a later mention or a passed date does not supply it"; "a leaning, a passing mention or a target date" | fidelity rows 1, 10; D4 | +50 |
| W8 | Quote derivation, my side (93) | "and the sentence that carried the knowledge when my turn is what the node is derived from: a diagnosis, an explanation, a recommendation the other side took up" | D3; carriers exist (Mira plan, fusion mechanism) | +161 |
| W9 | Continuity whys (955–956) | both argue from the node's current claim | A3, E9 | +54 |
| W10 | Correction form 4 (71) | across windows, two statements of theirs that differ with no word of correction keep both dated values — "a habit that moved is dated knowledge, a slip corrected is a repair" | fidelity row 11, D9; **rule-only, A1 risk recorded** | +240 |

Diffs: `template.md.diff`, `gist.md.diff` in the fixture. Static checks (`static_checks.py`,
V3.4's checker with PARENT repointed): fences parse, ops and keys valid, swap `old` strings
depicted before their ops, ids depicted before use, no agent-name literal, sizes and hedge
census (23 hedge clauses, unchanged), A10 census unchanged in kind plus the yoga why swap
(the checker does not parse the pseudo-JSON `revise_batch` block; the yoga before-state was
verified by hand: the old why appears once in the excerpt at line 733 and once in the op at
line 773, `c8d13e05` first at 733).

## Anti-overfitting record

`transfer_split.json` was written before author.py (22:02 UTC): fresh items by the V3.3
sha256 rule, first unexposed feasible per type — knowledge-update affe2881, 07741c45;
multi-session gpt4_e05b82a6 (four sessions); temporal gpt4_68e94288, gpt4_0b2f1d21; synthetic
conv_003_philosophy. No content, question or answer read. The mention scan excludes the kept
brains' binary embedding caches (final_brain/, fastembed_cache/), which the V3.3/V3.4 scans
predate; everything else in the rule is unchanged. Four candidates rejected as exposed
(c6853660, 71315a70, 60bf93ed_abs, gpt4_65aabe59). All twelve inspected corpora are
regression data.

## Cells pinned (zero model calls so far)

| Cell | Arms | Encodes | Compared with |
|---|---|---:|---|
| `eval/results/s1e_v35_transfer_2026-09-12` | production, V3.4, V3.5 | 135 | each other; blind packs; downstream |
| `eval/results/s1e_v35_regression_v34_2026-09-12` | V3.5 | 42 | saved production / V3.3-live / V3.4 brains of the V3.4 fresh cell |
| `eval/results/s1e_v35_regression_v33_2026-09-12` | V3.5 | 42 | saved production / V3.2 / V3.3 brains of the V3.3 cell; V3.4 from its regression cell |
| `eval/results/s1e_v35_sanity_2026-09-12` | V3.5 | 9 | saved production / V3.2 / V3.3 / V3.4 on creative_design |

Preflight passed on every cell: identical factual sections across arms, exact match to the
saved baselines' preflights. Encoding cost at the V3.4 cell's recorded rate
($0.107 per encode, Sonnet 4.6 list prices): about $24 for 228 encodes; the per-node judge
and the blind Opus packs come on top.

## The analysis that runs on it

`ANALYSIS.md` in the fixture: the V3.3/V3.4 instruments (census, revise sweep, coverage
targets, uncovered nodes, downstream, blind review, tally) plus four new ones — a stale-surface
detector, a surface-redundancy census, a firming-marker census and a per-node content and
field quality judge — the census extended with voice derivation and reasoning restatement, the
blind rubric extended with an eighth dimension (content and field quality) and four probes.
Each change above names its readout there before the run.
