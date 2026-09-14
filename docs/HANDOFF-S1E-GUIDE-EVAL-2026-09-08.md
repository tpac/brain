# HANDOFF — s1e encoder prompt eval + operating-guide arm (session 34bd6c81, 2026-09-07 15:30 → 2026-09-08 06:00 UTC)

Tom continues the refining from here. Nothing is deployed: the daemon runs main; the eval branch stays
unmerged until Tom promotes parts. Read this, then the ledger, then the brain nodes; the dumps are the
ground truth for any number you doubt.

## Where the work lives

| what | where |
|---|---|
| eval branch (code + candidate prompts) | `claude/sweet-lichterman-ba9854`, worktree `/Users/tpac/brain/.claude/worktrees/s1e-revise-shape-review-6cb242`, head **069f22c** (all of last night committed; main merged in through adbbf97). Do not edit `/Users/tpac/brain` (main = live daemon). |
| hand ledger (every run read by hand) | `ab_2026-09-01_03/ops9/ADJUDICATION.md` — sections in run order: evening thread, per-item cells, Cell A, Longmem 9, Folded, deep-dive, Pass 2/1 summaries, Why we miss, probe results, Guide design/probes/cell/longmem |
| raw ops per run (`--dump-ops`) | `ab_2026-09-01_03/ops9/<item>-<arm>/*.json` — arms: prod, control, candidate, verbscan, folded, guide; the guide dumps carry `round_texts` (the lists) |
| probes | `ops9/probes/` (miss interviews, 9 transcripts), `probes_guide/live/` (6 v1.0 live encodes + interviews), `probes_guide/coldread_{coherence,adversarial,v1_1}.md` (3 Opus cold reads), `S1E-PROMPT-AUDIT-2026-09-07.md` (full-read audit), `EXAMPLE-SHAPE-INVENTORY-2026-09-07.md` (9 shapes × 12 worked calls, hand-counted) |
| tools (all in `ab_2026-09-01_03/tools/`) | runners `run_cell9.sh run_cellA.sh run_folded.sh run_guide.sh run_longmem9.sh run_longmem9b.sh run_longmem_guide.sh run_guide_all.sh`; generators `make_verbscan_candidate.py make_getnodes_candidate.py make_folded_candidate.py make_guide_candidate.py`; readers `adjudicate.py adjudicate_reads.py worklist_kpi.py`; probes `edge_interview_probe.py miss_interview_probe.py guide_live_probe.py` |
| patched captures | `payloads_patched/` (d034485c, loosened preamble), `payloads_patched_guide/` (six captures, lists-first preamble) |
| longmem corpora / sweeps | corpora under `/Users/tpac/AgentsContext/eval-corpus/<hash>/` — prod **e16d35**, candidate **eaaeb9**, guide **f4897d**; sweeps `lm9_prod_sweep2`, `lm9_candidate_sweep2`, `lm_guide_sweep`; compares `eval/longmem/reports/ab_compare_lm9b`, `ab_compare_lm_guide_vs_{prod,candidate}` |
| artifact (Tom-facing) | https://claude.ai/code/artifact/238bf011-7f05-496b-a9a9-a807f66af95c |
| brain nodes | handoff d91e12b2 (origin); 4973d72f e091429d 89bceb81 a0224457 000287eb 51ec2f62 224a3027 f3f86cf1 (cells); a15a9411 6e6133d4 c1213e42 (miss probes); 0ffb2d50 e9679ff9 06fef26a (guide design / gold / longmem); 818cc0a0 (6-pass plan, superseded) |

## Every arm we ran (same substrate: IsolatedBrain from production, frozen captures, writes intercepted, reads served from the live copy; K = interaction fingerprints)

| arm | what it is | K (s1e / s1e_gist) | gold /38 (hand) | longmem /30 | one-line verdict |
|---|---|---|---|---|---|
| prod | 2026-09-03 production template, no gist (`eval/candidate_prompts/s1e_production_2026-09-03.md` via `--s1e-template`, capture untouched) | 3817564d21c4 / — | **15** | **22** (e16d35) | the daemon today; canonical node never revised; zero reads |
| control | prod template + branch gist v2 | 3817564d21c4 / b4a8b12c1af5 | **22** | — | the gist alone is worth +7 |
| candidate | branch code default + gist v2 (what the branch merge deploys) | f625390ea5aa / b4a8b12c1af5 | **29** | **26** (eaaeb9) | best balance; refs 37% is its regression; memrise 0/3; twin 2/2 |
| verbscan (cell A) | candidate with 19 English verbs → op names (`s1e_verbscan_2026-09-07.md`) | 3a067c9c05c2 / b4a8b12c1af5 | **23** | — | −6, not promoted |
| loosen: prompt / both | two read-restriction swaps (+ assembler preamble) on the d034485c replay only | — | — | — | 0 reads in 12/12; moved nothing |
| folded | loosening + fact-atom sentence + gist v2.2 (`s1e_folded_2026-09-07.md`, `s1e_gist_v2_2_2026-09-07.md`) | 9e8612037724 / 7410a59f1d7b | **19/28 scoreable** (3 VOID reads) | — | refs 94%, first reads (3/14), first-ever 78983ba6 revise (1/2), memrise 2/3; bb5b1ef4 2/6 |
| guide v1.0 | four lists (fetch/changes/targets/new) — probes only | 924e33d495c4 / 379b6d9e8b0a | — (6 live probes) | — | lists 6/6, fetch: none on both unrendered cases, situation 1/2 |
| **guide v1.2** | lists in dependency order (changes → targets roll-call → fetch unconditional → new), template 9 swaps, lists-first preamble (flag), closure fix (`s1e_guide_v1_2026-09-08.md`, `s1e_gist_guide_v1_2026-09-08.md`) | b8ecf4a6ec1c / 5269c57748b3 | **25/30 scoreable** (2 VOID) | **20** (f4897d) | revise half best of any arm (bb5b1ef4 6/6, run-44 8/8, refs 100%); create half regresses longmem −6 net |

Per-item gold (guide / candidate / folded / prod): d827d22f 3/4 on 1 run + 2 VOID / 9/12 / 5/8 / 0/12 · 86af52d1 4/6 / 4/6 / 5/6 / 3/6 · a85d5fb5 4/6 / 4/6 / 4/4 / 3/6 · bb5b1ef4 **6/6** / 5/6 / 2/6 / 5/6 · run-44 **8/8** / 7/8 / 3/4 / 4/8 (twin minted in every arm; folded revised 78983ba6 once).
Longmem per item guide vs candidate: fca762bc 0→3, 71017276 3→0, 59524333 3→0, 09ba9854_abs 3→0, six unchanged.

## The ten failure patterns (evidence in the ledger; this is the map for any refinement)

1. Node is the unit, not the field — content fixed, situation/title/edge left. Fixed by the roll-call on situation; edge why still 0/48 (no run wrote a `why→` verdict).
2. Keyed on strings, not claims — siblings carrying the event's tokens swept, the differently-worded target skipped (a85d5fb5 4/12 + guide run 1; bb5b1ef4 folded). Unmoved.
3. Rendered = exists — residue-named id fetched 1/24, edge-line-only node revised 1/~65; unconditional fetch sources fired 0/5. Unmoved by prose; both interviews and the cell say render it.
4. Pending clause read as trigger — "after X merges" kept 5/9 (+2/3 guide); "revise when it surfaces" = wait.
5. Rule read ≠ rule run — every added sentence quoted back, ranked last 9/9; the lists (procedure with output) fired 23/23.
6. Worthiness judgment swallows the instruction — "nodes I will have to touch" → fetch: none ×5; "single session, no continuity signal" / "lacks specificity" → zero nodes on two longmem items.
7. Residue = memory of verdicts — d034485c ×4 runs; gym "not minting now" cited by runs 2–3. Header "(not a to-do list)" + "don't re-assert" reinforce; age IS rendered (`open ×N since MM-DD`, nudge at ×5) but says resolve/hand up, never act.
8. Voice see-saw — my-voice clause: memrise 0→3, abstention 3→0 (my Tokyo fares encoded), five care-tip nodes planned instead of the chandelier; user dates 79→40%, user quotes 64→36% on longmem.
9. Plan then stop — text-only closes 5–8/23 in every arm (legit); once the guide planned five nodes and made no call (71017276 run 2).
10. Measurement bends around reads — reading a gold node voids the run (live copy already fixed: d827d22f); scorer token gates ≠ hand rule (86af52d1, bb5b1ef4); interviews confabulated once.

## Gates (all Tom's)

- Promote from the guide: targets roll-call + its exemplar, round contract sentence, fetch scope (v1.2 minus my-turns emphasis), refs line. NOT the `new` list as written.
- `new` needs: fact-first ("a fact the other side states about their life/schedule/possessions/plans is an atom the first time it is said — the anchor count is for patterns, never facts") ahead of the my-voice clause; "a no-mint verdict never goes to residue".
- Mechanisms: (a) assembler renders residue-named + edge-line ids into the catalog — `encode._build_catalog` already unions surfaced ∪ {encoded, authored, recalled}; two more id sources of the same shape + a tag legend + a test (~20 lines); (b) `trace_contract` header "(not a to-do list)" / nudge wording (shared with S2); (c) runner single continuation when a reply carries a non-empty `new`/`targets` and no tool call; (d) closure sentence already changed on the branch (shared with S2) — gate before merge; (e) production traces don't store `round_texts` (runner returns them; `build_delta_metadata` doesn't carry them) — needed if lists go live.
- Edge repair 0/48: mechanism vs example still open; not probed.
- Contract step: confidence → label (from the origin handoff, untouched this session).

## How to run the next cell (copy of what worked)

```bash
# gold cell for a new arm (template file T, gist file G, captures dir C) — ~4 min/run, ~1.3h for 17 runs
cd /Users/tpac/brain/.claude/worktrees/s1e-revise-shape-review-6cb242
./dev python3 eval/encoder_prompt_ab.py <capture> --arms F --s1e-template T --gist-file G --behavior --repeat 3 --gold eval/ground_truth/s1e_fieldcov_<id>.json --dump-ops ops9/<item>-<arm>
# then: tools/adjudicate.py 'ops9/<item>-<arm>/*.json' (hand read), tools/worklist_kpi.py (lists vs ops), eval/encoder_ops_shape.py <ab dir> (add the arm to ARMS)
# longmem: build_corpus.py --qids <gate-4 ten> --s1e T --interaction-template s1e_gist=G --label X ; sweep.py --corpus <hash> --variance 3 --force-preflight --label X_sweep ; compare_arms.py lm9_candidate_sweep2 X_sweep --labels candidate,X --out-dir …
# BRAIN_S1E_LISTS_PREAMBLE=1 in the environment for any arm that wants the lists-first preamble in the longmem build
```

Costs observed: gold run ~4 min; 17-run cell ~50 min; longmem build ~52 min + sweep ~8 min. One IsolatedBrain at a time. Never sqlite3 the live DB; read eval brains via the brain API on scratch copies (`Brain(db_path=copy)`, `query_traces(ref_type='encoding_run')` → `events[].metadata.final_text`).

## Perishables / traps

- d827d22f's live node is already fixed → any arm that fetches it is VOID on that item; score the item on non-reading runs or accept the read as the KPI.
- The scorer's token gates: 86af52d1 PASS with the pending tense kept; bb5b1ef4 FAIL with 47 kept as an upper bound — the hand rule wins.
- `make_guide_candidate.py` swaps must each match exactly once against the CURRENT code default; a main merge that touches the prompt breaks anchors loudly (NOT UNIQUE).
- `payloads_patched_guide/` embeds the preamble order `changes, targets, fetch, new` — regenerate if the order changes.
- Interviews are the encoder's plausible account; the ops log is the truth.
