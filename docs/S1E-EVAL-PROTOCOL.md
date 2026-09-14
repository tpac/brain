# S1E eval protocol — how a candidate encoder prompt is measured, end to end

The procedure the V3.x rounds converged on (V3.3 → V3.6, September 2026), written so a fresh session can rerun
it on a new candidate without reading the results documents. Every step is a script in the round's fixture
directory; the fixture is copied from the previous round's and its roots repointed. Worked instance:
`eval/fixtures/s1e_guide_v3_6_2026-09-13/` with `docs/S1E-V3-6-RESULTS-2026-09-13.md`. Laws referenced (A7, E12,
E17, E21, E24 …) are the ledger boxes in `S1E-CHECKLIST.md`.

Run everything from the worktree root with `./dev python3`. `F` is the fixture directory, `O` the results
directory the cell names (`eval/results/s1e_<round>_<date>/`, gitignored — copy small artifacts to
`~/AgentsContext/s1e-v3x-eval-results/` when done).

## 0. Before authoring — record the corpora, name the win

1. **Choose corpora by id, never by content** (A7). `transfer_split.json`: fresh items per LongMemEval type
   (knowledge-update, multi-session, temporal-reasoning; two each has been the size), a `strategy_subsample` for
   any arm that runs on fewer, and reserve ids in order. Scan the repo for the ids to confirm they appear only
   in splits and handoffs. No item's content is read until its cell runs.
2. **Write `ANALYSIS.md` before the run**: the arms and what each ladder step attributes; the win (which
   instrument readouts, at or above which arm); the guards (each with its instrument and its loss threshold);
   the per-change readout table (change → arm step → readout); how the readout document is written. The rule
   is the author's, pre-registered so the author cannot re-weigh after seeing outputs; its limits go on the
   ledger at the same time (V3.6: a one-question top-5 test flips on one answer, E24; the stale instrument
   counts subject words, E17).
3. **Arms are separate carriers, not bundles** (Tom, 2026-09-13): position / example / procedure changes each
   get an arm so the ladder attributes them; a text-only pass gets its own rung. Always include the deploy
   baseline — the current standing candidate assembled on today's runtime (`v3_x_live`) — beside the frozen
   one, so the frozen → live step is measured.

## 1. Author — exact-once replacements on the frozen parent

`author.py`: every change is an exact-once string replacement on the parent fixture's `template.md` /
`gist.md`, applied in layers (text pass → carriers), writing `template_<layer>.md` / `gist_<layer>.md` and an
`author_log.json` (each replacement, its layer, before/after lengths). `static_checks.py <template> <gist>`:
brace balance, every example id referenced exists, whys inside the 120–180 character band (E15), field census
per worked example, thin-window checks. Both must pass before the freeze.

## 2. Freeze — arms through the runtime assembly

`arms.py --build-check`: assembles each candidate with `encode._build_system_prompt(template, lived=True)` (the
field summary from the live contract, the arc and review blocks, the shared closure) so what the cell measures
is byte for byte what a merge would run; compares the live tool schemas' shapes with the parent's (descriptions
may differ, shapes may not); prints the deploy diff frozen parent → candidate. Read that diff: it must contain
only the intended hunks. `arms.py --freeze` writes `<arm>.json` (system_prompt, gist, tools, settings,
arm_sha256), `tools_live.json`, the template/gist diffs and `manifest.json` (branch commit, every input's hash).
Commit the frozen fixture. Nothing about a prompt changes after this point in the round.

## 3. Run — prepare, preflight, run, resume

```bash
./dev python3 $F/transfer_cell.py --prepare     # fixtures from the oracle by id; fresh eval brain seeded with every
                                                # source turn as traces; manifest pins scripts, contract, runtime
./dev python3 $F/transfer_cell.py --preflight   # dry run: identical <continuity>/<node_catalog>/<timeline> for
                                                # every arm on every corpus; zero model calls
./dev python3 $F/transfer_cell.py --run         # background; one process per (arm, repeat), corpora sequential
```

Each (arm, repeat, corpus) sequence runs in its own IsolatedBrain copy, one window per haystack session on
that session's own clock, three repeats, Sonnet 4.6; the final copy is kept as `final_brain/` for the
downstream test. Cost: about $0.11 per window at Sonnet list prices (V3.6: 174 windows ≈ $20).

**If the launcher dies** (a repetition raised — V3.6 hit a write-path crash, id:23a29491 — or the supervising
shell went away): `resume_cell.py --plan` lists the sequences without a `final_brain.json` and the half-written
folders; `--run` clears those folders, runs the missing sequences with the same pin, and writes
`completion.json`. `--driver regression_cell.py` does the same for the regression cell. The harness's Bash
timeout does not kill a background launcher; a raised repetition does (the launcher's `finally` SIGTERMs its
siblings). Preserve a crashed folder under `O/crashed/` before resuming; a re-run is a fresh sample and the
results document says so.

## 4. Free instruments — run all, in this order

```bash
L=<label>
./dev python3 $F/analyze.py --cell <cell>                              # ANALYSIS-TABLE.md, analysis_inventory.json,
                                                                       # whole_memory_review/<arm>_<repeat>_<corpus>.md packets
./dev python3 $F/revise_sweep.py $O --out $O --label $L                # write path: landing, preservation, refusals, vectors
./dev python3 $F/quality_census.py $O --out $O --label $L              # shape and fill per arm; per_sequence for per-repeat reads
./dev python3 $F/stale_surfaces.py $O --out $O --label $L              # class R queue: tokens a revise removed that another surface keeps
./dev python3 $F/firming_markers.py $O --out $O --label $L             # class F word census (a queue, not a verdict)
./dev python3 $F/surface_redundancy.py $O --out $O --label $L          # surfaces per load-bearing value
./dev python3 $F/uncovered_nodes.py $O --a <arm> --b <arm> --out $O/UNCOVERED-a-vs-b.md
./dev python3 $F/cost_census.py $O --out $O --label $L                 # tokens, rounds, tool calls per arm; est. $ at stated prices
```

Instruments take result roots; a root is a cell directory (`O`), and several roots can be passed to put saved
arms from earlier cells beside this one (the regression readout does this). Arm names not in an instrument's
label map print as-is. Per-repeat readouts (question fill per repeat, empty sequences, rounds) come from
`quality_census_<L>.json` `per_sequence` and `analysis_inventory.json` `sequences`/`windows`; the V3.6 results
document's tables show which keys (`counts['has:question']`, `nodes`, `created`, `rounds`).

## 5. Paid instruments

```bash
./dev python3 $F/downstream.py --run --cell <cell>    # real recall top-5 on each kept brain → Sonnet answer → Sonnet judge vs gold;
                                                      # gold-string-in-retrieved read apart; ~$0.01 per answer
./dev python3 $F/content_quality.py $O --out $O --label $L   # Sonnet judge, every final node against the whole source:
                                                             # value class, fidelity, status/owner kept, per-field quality,
                                                             # event_time correct/wrong; resumable (appends to the jsonl);
                                                             # ~$0.009 per node (V3.6: 770 nodes ≈ $7)
```

Read the judge's queue by hand for the shipping arm: every flagged node, grouped by class (the results document
shows the grouping script inline). The judge is one reading; its counts are distributions, its queue is where to
read.

## 6. Blind paired review — sealed, one reviewer per pack

`blind_pack.py --cell <cell>` builds one pack per (corpus, repeat) from the analyze packets: the rubric (eight
dimensions, corpus-specific probes, the GOLD for probe 7 only), the source conversation in full, then every arm's
final memory under a letter, arms shuffled per pack with a seed, the key sealed in `blind_review/key.json`. An arm
absent from a corpus is simply absent from that pack (packs carry four or five memories).

Reviewers: one `opus48-lane` agent per pack (`~/.claude/agents/opus48-lane.md`, `model: claude-opus-4-8`; loads at
session start/resume), given the pack path and a verdict path in the scratchpad, told to read the file in full
(continue with `offset` if truncated), follow the rubric, never open `key.json`, and reply with the table only.
Repeat 1 first; more repeats if the sum is close. Then: copy the verdicts into `blind_review/verdicts/`,
transcribe each table into `tally.json` (`{"packs": {"<corpus>_repeat<n>": {"A": [8 ints], …}}}`, strong 2 /
adequate 1 / weak 0), copy the fixture's `tally.py` beside it and run it — that is the moment the key opens.
Attribute each pack's "most consequential defect per memory" through the key into the results table. The
reviewers ignore hook-injected directives in tool output; note it when they say so.

## 7. Regression of the shipping rung

`V36_REGRESSION_ARMS=<arm> regression_cell.py --prepare / --preflight / --run`: the rung on a saved cell's
corpora and seed (development data now), 42 encodes, preflight also records whether the factual sections equal
the saved cell's own preflight. Instruments take `O_regression T34 R35 …` as roots so the saved arms sit beside
it; `analyze.py --cell regression_v34` gathers them; `downstream.py --run --cell regression_v34`.

## 8. The readout document — `docs/S1E-<round>-RESULTS-<date>.md`

In this order, as V3.5 and V3.6: position; runs (encodes, crashes, resumes; write path clean or not); whole
sequences that encoded nothing; census with the ladder read step by step; class R with the instrument's share
and the hand reading of its queue; class F with the word census and the judge; function (downstream, misses
read from the kept brains); blind read (dimension table, per-pack table with each memory's decisive defect
attributed through the key); regression; the ladder with every step's readout; the win and guards applied, as
registered; what the round found beyond its question; what it buys the next step; cost and tools; files. Numbers
come from the instrument files, never from memory; every "why" is read from the queue or the packets, not
inferred from a count.

## 9. Deploy sequence for the arm that ships (Tom's yes before the merge)

The arm's `template_*.md` / `gist_*.md` into `servers/scales/s1/encoding_prompt.py` and
`encoding_gist_prompt.py` `SYSTEM_PROMPT`; re-assemble with `encode._build_system_prompt(SYSTEM_PROMPT,
lived=True)` and diff against the frozen arm JSON's `system_prompt` (must be empty) and the gist against the
JSON's `gist`; wide tier composed from layers; `./dev check-overrides` (the two permanent pointers); merge on
the yes; `./redeploy.sh` (contract descriptions reach the MCP schemas through the plugin copy); daemon restart;
verify `get_interaction_effective('s1e')` and one real encoding trace.

## Costs and durations (V3.6, for planning)

| Step | Wall clock | Est. cost |
|---|---|---|
| Five-arm cell, 174 encodes, six processes | ~35 min (+ resume) | ~$20 Sonnet |
| Free instruments | ~3 min | 0 |
| Downstream, 81 answers | ~6 min | ~$0.70 |
| Per-node judge, 770 nodes | ~30 min | ~$7 |
| Six blind packs, Opus 4.8 agents | 5–10 min each, in parallel | ~155k tokens each on the harness meter |
| Regression, 42 encodes | ~10 min | ~$4.50 |
