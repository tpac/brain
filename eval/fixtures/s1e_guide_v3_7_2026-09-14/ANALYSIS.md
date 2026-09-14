# V3.7 analysis plan — fixed before the cell runs (2026-09-14)

The V3.7 round (Tom, 2026-09-14: "Start the S1E V3.7 round"). Production runs V3.6 full (`s1e` 641a0d269583,
`s1e_gist` 7617804462cd, main 6406a72); the V3.6 cell's frozen `v3_6_full` arm is byte for byte what the code
default assembles (`deploy_defaults.py --arm v3_6_full --check`, re-proved in `arms.py build()`), so that arm
is the baseline every carrier is read against. Three carriers target V3.6 full's three largest measured faults
(brain id:632ec982, from the fault taxonomy id:0705fdfd and the revise read id:1b63e6fb). One carrier per arm
(id:4c43742b): the ladder attributes each alone; nothing is stacked until each has passed alone. Instruments
are the V3.6 set unchanged; this file adds only what this round pre-registers — arms, targets, guards, the
stack rule, the regression — and the per-repeat spread as the guard scale in place of V3.6's 55–65 band.

## Arms and what each attributes

| Arm | What it is | The step it isolates |
|---|---|---|
| `v3_4_live` | the V3.6 cell's frozen V3.4-on-branch-runtime arm | continuity: the fresh six against a second known arm, so a corpus effect reads apart from a carrier effect |
| `v3_6_full` | the V3.6 cell's frozen full arm == production | the baseline; all three carriers are exact-once replacements on its template |
| `v3_7_advice` | + carrier 1 (example): the thin window's `<me>` turn gains one general tip Wren does not take up; the Bad paragraph names why it earns no node; the closing sentence states the scope — a method or threshold I delivered AND they took up | baseline → advice: does the scope guard cut the "My … advice" method nodes without cutting the person's facts |
| `v3_7_quote` | + carrier 2 (example): the Priya excerpt renders `Their Raw Quote` with the old value; the revise replaces the quote whole with her new words and says why (a quote is evidence for the claim beside it; never a span swap inside someone's sentence) | baseline → quote: does a depicted before-state stop the stale-quote-beside-new-claim class |
| `v3_7_event` | + carrier 3 (example): a second Wren window after the thin one — the swatch is done, the back cast on; the plan's title, content, situation, reasoning and `event_time` move (`event_time stale` on the targets line, the planning date kept in reasoning), the quote stays because nothing in it went false | baseline → event: does a depicted planned → done transition move `event_time` with the state |

Fresh corpora (recorded before authoring, `transfer_split.json`): 07741c44, 89941a94 (knowledge-update);
88432d0a, e56a43b9 (multi-session); gpt4_e414231f, gpt4_2655b836 (temporal-reasoning). 15 windows per (arm,
repeat). Three repeats. 5 arms × 3 × 15 = 225 encodes, ≈ $25 at V3.6's $0.11 per window. The gist is not
touched by any carrier (`gist_full.md` is the frozen V3.6 gist; `arms.py` asserts it).

## The spread rule (in place of the 55–65 band)

For a per-repeat readout `m` (question fill, nodes per window, isolates, method nodes per window …), an arm's
spread is `max − min` over its three repeats. A difference between two arms counts only when it exceeds the
larger of the two spreads; inside it, the arms read the same. V3.6 measured spreads of 9 (full) and 31 (the
unstable layer arm) on question fill; both baselines sat outside the old band on those corpora, which is why
the band is dropped. Per-repeat values come from `quality_census_<L>.json` `per_sequence` and
`analysis_inventory.json`.

## Targets — one per carrier, named before the run

| Carrier arm | Target readout | Instrument | Passes alone when |
|---|---|---|---|
| `v3_7_advice` | generic-advice value class share; method nodes per window; owner-not-kept on `method` nodes | content_quality (Sonnet judge, value + owner_ok); quality census types | generic-advice share and method nodes per window both below `v3_6_full` by more than the spread, on the same corpora; the hand read of every remaining "My … advice" title says each carries a threshold, method or diagnosis the person took up |
| `v3_7_quote` | revise ops whose replaced value survives in `their_raw_quote` / `my_raw_quote`; judge `fidelity_evidence` naming a quote/claim mismatch | stale_surfaces (quote column); content_quality queue | the quote column below `v3_6_full`'s (ops, not tokens) and the hand read of every revised node's quote finds no quote asserting the superseded value beside the new claim; a quote replaced with words that are not the speaker's counts as a loss |
| `v3_7_event` | judged `event_time` wrong share; the hand-read count of class (a) — a node whose content records a completed state while `event_time` stays at the first disclosure | content_quality (fields.event_time); the judge queue read by hand, class (a) apart from class (b) bare-month dating, which is the contract's | class (a) count below `v3_6_full`'s across the three repeats, and the wrong share not above; class (b) is reported, not attributed |

## Guards — a target win with a loss here is a trade Tom rules on, never a ship

Baseline for every guard: `v3_6_full` on the same corpora and repeats. "Above/below" means beyond the spread
rule unless a fixed margin is named.

| Guard | Instrument | Loss means |
|---|---|---|
| question fill, per repeat | quality census | any arm's per-repeat spread over 20 points, or its mean below the baseline's by more than the larger spread |
| nodes per window | quality census | below the baseline by more than 10% — for `v3_7_advice`, counted without `method`-type nodes (fewer advice nodes is the target, fewer of the person's facts is the loss) |
| first-disclosure facts kept | uncovered_nodes (`--a v3_6_full --b <arm>`), read by hand | facts the baseline kept and the arm dropped, beyond the pairs the baseline also drops against the arm |
| judged overclaimed share | content_quality | above the baseline by more than 3 points |
| fabricated element share | content_quality | above the baseline by more than 2 points |
| generic-advice share | content_quality | above the baseline (quote and event arms; the advice arm's target) |
| event_time wrong share | content_quality | above the baseline (advice and quote arms; the event arm's target) |
| stale trigger surfaces on revise (title / situation / question) | stale_surfaces, hand read | ops leaving a superseded value on a trigger surface above the baseline's count |
| isolates, near-twin pairs | quality census | above the baseline by more than the spread |
| rounds per window, refusals, write-path errors | revise_sweep, cost_census | rounds up by more than one on average; any refused or failed op the baseline did not have |
| empty final memories | analyze | any sequence with zero nodes (baseline expectation 0 of 18) |
| downstream, top-5, 6 items × 3 repeats | downstream (Sonnet answer + judge) | correct count below the baseline's; every miss read from the kept brain (E24 fragmentation named apart from a dropped fact) |
| blind sum, repeat 1, sealed | blind_pack + tally, Opus 4.8 | below the baseline; the dimension profile read, not only the sum — a loss on "revision and preservation" or "facts" is a loss even with the sum ahead |

## The ship rule

1. A carrier **passes alone** when its target moves as registered and it loses no guard.
2. **Cell 2, the stack**: every passing carrier applied together on the same six corpora, three repeats, beside
   `v3_6_full` (and the passing single arms are not re-run). The stack ships only if it too loses no guard
   and its downstream count is not below the baseline's. If exactly one carrier passes, that arm is the
   stack and Cell 2 is not run.
3. **Regression** of the shipping rung on V3.6's six fresh corpora (development data now), 3 × 13 windows,
   against the saved `v3_6_full` and `v3_4_live` brains of `eval/results/s1e_v36_refine_2026-09-13`; the same
   guards, the downstream on the kept brains. A guard lost there is a trade for Tom, as above.
4. If no carrier passes, nothing ships; V3.6 full stays production and this document says which fault each
   carrier failed to move and what the failure looked like in the queue.
5. No prompt edits after outputs are seen. A carrier that fails is re-authored, if at all, in a new round with
   fresh corpora.

## Per-change readouts (the attribution rows)

| Change | Arm step | Readout |
|---|---|---|
| general tip in the `<me>` turn + Bad paragraph + scoped closing sentence | baseline → advice | method nodes per window and generic-advice share on the advice-heavy corpora; the "My … advice" titles read by hand; nodes-without-method per window as the guard |
| `Their Raw Quote` in the Priya excerpt + whole replacement in the revise | baseline → quote | quote column of stale_surfaces; every revised node's quote read against its title; question/situation staleness unchanged (the carrier does not touch them) |
| second Wren window: planned → done, `event_time stale`, the quote left clean | baseline → event | class (a) event_time count; judged event_time wrong; the plan nodes on the multi-session items read by hand for the date they carry after a completion |
| nothing (the gist is unchanged) | all | gist-position readouts (thought, question fill) expected flat; a move is a corpus effect, read against `v3_4_live` |

## How the readout is written

As V3.6's: runs (crashes, resumes, write path); whole sequences that encoded nothing; census with the ladder
read step by step, per repeat; class R with the instrument's share and the hand reading of its queue; class F;
function (downstream, misses read from the kept brains); the blind read with each pack's decisive defect
attributed through the key; the targets and guards applied as registered, per carrier; the stack decision;
regression; what the round found beyond its question; cost and tools. Numbers from the instrument files, never
from memory. The result document names the shipping rung and its deploy steps: `deploy_defaults.py --arm
<rung> --write`, the assembly diff (must be empty), the wide tier composed from layers, `./dev
check-overrides`, merge on Tom's yes, `./redeploy.sh`, daemon restart, `get_interaction_effective('s1e')` and
one real encoding trace.

## Known limits, on the ledger now

- The V3.6 cell's `seed_baseline` and kept `final_brain/` databases were not found in the saved copy
  (`~/AgentsContext/s1e-v3x-eval-results/results/s1e_v36_refine_2026-09-13/` holds `final_brain.json`
  records, not the databases). The regression cell needs the seed; if it cannot be recovered, the regression
  re-seeds the six V3.6 corpora from the oracle by id (the same traces, the same clock) and compares against the
  saved v3_6_full *outputs* (nodes_after.json) rather than re-running its downstream on kept brains.
- One judge reading per node; its counts are distributions, its queue is where to read.
- Carrier 3's example marks the plan's quote `clean` (Wren's "so I have time to swatch" is not falsified by
  the swatch being done); carrier 2 replaces a quote that asserts a superseded value. The two are consistent
  and are read together in Cell 2 if both pass; a rise in stale quotes on the event arm alone is a finding
  against carrier 3, not against carrier 2.
