# S1E V3.4 — one more weave, authored 2026-09-12

Fixture: `eval/fixtures/s1e_guide_v3_4_2026-09-12/` (`author.py` is the record;
`manifest.json` pins parent V3.3 `7c5730206c7c…`, candidate `e3e2251524…`,
attribution arm `8e3ec27a06…`). Static checks passed; sizes: template
105,384 → 106,775 chars (+1,391), strategy 1,348, hedge clauses unchanged at 23.

## The five weaves (all inside existing carriers)

| # | Shape class | Evidence in the V3.3 cells | Where it lives now |
|---|---|---|---|
| W1 | The quote moves with the fact | superseded number left in `their_raw_quote` 3/3 repeats (b6019101), stale start-quote on a completion node (2ebe6c92); quote refreshed in 1 of 31 revision events | second Mira window: plan revise gains a `their_raw_quote` swap to Mira's October 14 words; `targets:` names the quote stale; the receiver's-view close names the quote as a surface seen beside the title |
| W2 | Question lane | question on 51% of V3.3 transfer nodes vs V3.2 67%; the worked window's remembers carried none | both window remembers carry a one-line `question` |
| W3 | A Q&A window is not routine | a whole Kauai session gated as "pure travel Q&A"; recommendation turns dropped on three corpora; 15 of 42 windows at ≤2 nodes | the Skip sentence gains the contrast case and production's own under-encoding understanding, in the same sentence |
| W4 | A refused op changed nothing | the unapplied-swap window certified as applied; 8 of 9 refusals were another field's sentence sent as a content swap | one sentence in the Working strategy (tail position) |
| W5 | Thought shape | V3.3's example thoughts were all epistemic-status statements on interpretation nodes; window 1 said the board fact stands "without … thought to justify it"; production's 23 real thoughts sit on facts, decisions, events | the plan revise carries a hunch-shaped thought (a dependency the encoder noticed), the targets line lists it, the board sentence now says a fact may carry a hunch when there is one |

## Tools: a confound found while ranking the thought gap

Every V3.x eval arm since V3.1 has run with the 2026-09-08 *generic description
candidate* (`eval/fixtures/s1e_tool_descriptions_2026-09-08/`), never wired into
code. The branch's live encoder schemas (`brain_mcp.TOOLS` through
`encode._get_tool_schemas`, the deploy path) have identical shapes and different
descriptions on 73 of 117 description fields — for `thought`, "Optional
interpretation, hypothesis or connection beyond the stored account and its
evidence…" against the contract's "My own read on the memory — a hunch, a
connection, a take the content doesn't carry. Delivered…". Production ran on
the contract text. V3.4 is frozen with the **live** tools (deploy path) and the
cell carries a fourth arm, V3.3 prompt + live tools, so prompt and tools are
attributed separately. Re-freezing with the candidate tools is one command if
Tom decides those descriptions ship.

## Thought field — the three production/V3.3 differences, by likely impact

1. **The description read at write time.** The arms' tool schema says
   "Optional interpretation … beyond the stored account and its evidence";
   production's says "My own read … a hunch, a connection … Delivered". V3.2 and
   V3.3 share the flat text and near-identical counts (3–4 vs production 23
   across both cells) although V3.3 restored the template paragraph — the
   template change moved nothing while the tool text stayed constant. Quality
   follows the same line: V3.x thoughts are epistemic caveats; production's are
   hunches and connections. Addressed by the tools choice above.
2. **What the worked examples make thought for.** V3.3's five example thoughts
   all manage a read's status on interpretation nodes and window 1 denies the
   fact a thought; production's canonical example is a hunch on a plain node
   ("unverified hunch; worth a look next time we touch it"). Addressed by W5.
3. **Salience and position.** Production gives thought a headed subsection with
   its own Bad/Good pair at 30% depth and two late-prompt mentions ("recover
   what was thought, not just what was decided"; the finishing checklist); V3.3
   folds it into the Types section at 10% depth and its strategy and closure
   never name it. Not changed in V3.4 — position moves are a separate lever.

## Fresh material (recorded before authoring)

`transfer_split.json`: LongMemEval reserves 69fee5aa, f685340e (knowledge-update),
2b8f3739 (multi-session), eac54adc, gpt4_1d4ab0c9 (temporal), plus the synthetic
`conv_002_debugging` (zero prior mentions). The six V3.3 transfer corpora and
creative_design are development data and run as regression sets only. Cells
prepared and preflighted with zero model calls: fresh transfer 168 encodes
(4 arms × 3 repeats × 14 windows), regression 42, sanity 9.
