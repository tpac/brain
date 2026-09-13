# V3.6 analysis plan — fixed before the cell runs (2026-09-13)

One refinement round before deployment (Tom). The round measures the deploy path: every V3.6 arm is
assembled by `encode._build_system_prompt` on the reconciled branch (main merged at 9a77e73; contract
changes at 36b106b), so what the cell measures is what a merge would run. Instruments are the V3.5 set,
unchanged (`ANALYSIS.md` in `s1e_guide_v3_5_2026-09-12` names what each reads and cannot decide); this
file adds only what this round pre-registers: the arms, the win, the guards, and the attribution ladder.

## Arms and what each step of the ladder attributes

| Arm | What it is | The step it isolates |
|---|---|---|
| `v3_4_titles` | frozen V3.4 exactly as measured on 2026-09-12 (its own tail, its own tools) | continuity with the V3.4 and V3.5 cells |
| `v3_4_live` | V3.4's template and gist on today's runtime, contract and tools | frozen → live: the reconcile, the contract reword, the hidden fields, the runtime closure in place of the strategy |
| `v3_6_layer` | + the text pass (W6c, W9, source_refs stated once, two whys in band, Nadia correction dated, L93 names delivered method, gist drift) | live → layer: the text pass |
| `v3_6_full` | + facts-first `new` procedure, the thin worked window, thought named with question at the tail | layer → full: the two measured carriers |
| `v3_6_full_v34tail` | full, with V3.4's strategy and closure in place of the runtime closure; strategy_subsample only | full → tail: the strategy section, on a small sample (Tom) |

Fresh corpora (recorded before authoring, `transfer_split.json`): 7e974930, 0e4e4c46 (knowledge-update);
ef9cf60a, gpt4_2f91af09 (multi-session); 4dfccbf7, gpt4_4cd9eba1 (temporal-reasoning). 13 windows. Three
repeats. 174 encodes. Regression: the winners on the six V3.4 corpora against the saved V3.4 brains.

## The win, named before the run

The candidate that ships is the highest arm on the ladder that meets all three and loses no guard:

1. **Downstream** correct of 6 items (three repeats, 18 answers) at or above `v3_4_live`.
2. **Empty final memories** 0 of 18 sequences (V3.4 fresh cell: 1; production: 3).
3. **Stale-surface share** of value-replacing revises (`STALE-SURFACES`) not above `v3_4_live`, and the
   dates-moved-with-event_time-untouched count not above it.

If only `v3_6_layer` meets them, the layer ships. If `v3_6_full` meets them, full ships. If neither,
`v3_4_live` ships as measured and this file says so.

## Guards — a win on the target with a loss here is a trade Tom rules on, never a ship

| Guard | Instrument | Loss means |
|---|---|---|
| question fill, per repeat | quality census | any repeat outside V3.4's fresh band (55–65%), or a per-repeat spread over 20 points |
| judged overclaimed share | content_quality (Sonnet judge) | above `v3_4_live` by more than 3 points |
| generic-advice value class | content_quality | above `v3_4_live` (the guide arm over-minted my own advice here) |
| isolates, near-twin pairs | quality census | above `v3_4_live` by more than the V3.4–V3.5 spread (10 → 20 isolates was the loss) |
| event_time wrong | content_quality | above `v3_4_live` (the reworded contract line could over-date) |
| nodes per window | quality census | below `v3_4_live` by more than 10% |
| rounds per window, refusals | revise_sweep | rounds up by more than one on average |
| blind sum, repeat 1, sealed | blind_pack + tally, Opus | below `v3_4_live`; dimension profile read, not only the sum |

## Per-change readouts (the attribution rows)

| Change | Arm step | Readout |
|---|---|---|
| contract event_time reword | frozen → live | event_time fill and judged event_time-correct on the temporal-reasoning items; the theme-park refusal class must not recur ("cannot be inferred from conversation date") |
| hidden fields | frozen → live | confidence constant gone (census: confidence never written by the encoder) |
| runtime closure in place of the strategy | frozen → live, and full → tail | inspect-and-repair still happens (repair ops after a write in revise_sweep) |
| text pass | live → layer | no readout expected; guards only |
| facts-first `new` (P1) + thin window (P2) | layer → full | empty sequences; nodes per window on knowledge-update items; generic-advice guard |
| thought at the gist with question at the tail (P3) | layer → full | thought fill up from 2; question fill inside the band with per-repeat spread under 20 |

## How the readout is written

As V3.5's: infrastructure first (revise sweep clean), then shape per repeat, then the judge's queues read by
hand for the shipping arm, then function (downstream), then the blind profile, then the ladder table with
every step's readout. No prompt edits after outputs are seen. The result doc names the shipping arm and the
deploy steps: wide tier, `./dev check-overrides` (two permanent pointers), `redeploy.sh` (the contract
description reaches the MCP schemas through the plugin copy), daemon restart, Tom's yes before the merge.
