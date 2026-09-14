# V3.3 working set: design, evidence and execution

Read [the handoff](HANDOFF-S1E-V3-3-2026-09-11.md) first. Paths below are relative to `/Users/tpac/brain/.claude/worktrees/s1e-revise-shape-review-6cb242` unless absolute. This is a durable local working set, uncommitted in this existing worktree. It is not an executable new candidate, and contains no newly run experiment. `S1E-V3-3-SALVAGE-INVENTORY-2026-09-11.json` records critical files and their hashes; evidence is in the original artifacts, not optional scratch space.

## The understanding to carry

Tom wants an encoder that serves future reasoning, with better revision and preservation. Avoid narrowing this to source refs, field population, commitment wording, or prompt length. His latest direction is to combine many shapes in a single example, using the challenges and probes to keep improving the teaching. A coherent episode should reveal choices, not merely display valid JSON.

The design proposal is mostly revisions: restore intellectual initiative and consequential choice preservation while keeping fidelity and the four-list revise procedure. The production audit and future-memory-needs audit were performed by two inline agents; their observations were checked against the prompts. The proposed causal explanation—compressed rationale plus repeated restraint teaching shifts attention toward audit—is not isolated experimentally. Production uses the same Sonnet 4.6 and sometimes develops more useful understanding, so a model ceiling is not established.

For a future agent, recovering only “reduce by 30%” would be less useful than recovering why: reduce competition for attention while preserving understanding. Rationale enables adaptation. The proposed receiver test asks whether a retrieved subset conveys the governing purpose, supported next move, actual commitment, open alternatives and circumstances for reconsideration. Do not turn this into mandatory fields or ritual doubt on every fact.

## Read the actual prompt carriers

| Carrier | Where / use |
|---|---|
| Frozen V3.2 | `eval/fixtures/s1e_guide_v3_2_2026-09-10/{template,gist,strategy,closure}.md`, their `.diff` files, `v3_2_titles.json`, frozen `CHALLENGES.md` |
| Verified September 11 production | `eval/fixtures/s1e_production_comparison_2026-09-11/effective_s1e.json` contains the full template; `production_deployed.json` contains native system/tools/preamble/settings; `export_metadata.json` records identity |
| V3.1 predecessor | `eval/fixtures/s1e_guide_v3_1_reviewed_2026-09-08/` |
| Original compression | `docs/S1E-GUIDE-V2-COMPRESSION-2026-09-08.md`; `eval/candidate_prompts/s1e_guide_v2_compact_shapes_2026-09-08.md`, compact reference and compact episode alternatives; Astra authored those options |
| Reverse-pass architecture | `docs/S1E-STRUCTURE-REVERSE-PASS-2026-09-08.md` |
| Generic tools | `eval/fixtures/s1e_tool_descriptions_2026-09-08/`; descriptions are not yet integrated into the real MCP defaults |
| Release owners/gates | `docs/S1E-PRODUCTION-COMPARISON-AND-RELEASE-SCOPE-2026-09-11.md`; apply reviewed diffs to current owners, never overwrite defaults with a stale full snapshot |

Particularly useful V3.2 locations: Reading lines 41–73; fields/thought 81–123; edges 125–141; capture/defaults 262–266; central mixed episode 288–460; later thought-only revision 462–492; bounded negative trial 500–524; detail/meaning pair 580–610; unspoken restoration pattern 915–958. These examples already teach uncertainty, inference without endorsement and evolving thought: do not claim those are absent. Rebalance their demonstrated value and integrate complementary shapes.

Production template excerpts (line numbers inside the JSON `template` string): emerging patterns 190–206, “my own read” 339–343, capture 867–880, six default countermeasures 882–914. The opening detail/meaning paragraph is unchanged in V3.2, and much of edge craft survives. Production's more expansive motivation is a candidate teaching function to recover, not a mandate to restore every word or “always” trait claim.

## Authoring and review sequence

1. Map desired behaviors and known failures to existing challenge IDs, example decisions and probe evidence. Distinguish a shape named in prose from one demonstrated in the source → choice → write → resulting-memory sequence. Keep one owner for each teaching concern.
2. Draft V3.3 as a separate artifact. Revise the existing mixed episode so several shapes interact naturally: incidental fact, contribution in my voice, decision/priority and reason, a supported emerging interpretation, a narrowed or strengthened read, an existing-node correction and a relationship. Cover other shapes only where the episode cannot honestly carry them. Use fictional, unrelated source material; don't put inspected eval answers into the prompt.
3. Preserve decision-bearing meaning through consolidation. “Same topic” is not “same claim”; putting new details in an old node must not swallow a choice, reason, order or separate story. Preserve still-true details during revisions.
4. Review both success and failure paths. A strong interpretation can be appropriate; an uncertainty qualifier should not replace substance. Show firm evidence remaining firm. A rejected idea is different from an unchosen one; certainty that a plan exists does not mean it is a commitment or accomplished event.
5. Run the existing appropriate static/example probes, inspect exact assembled requests—including tool descriptions and final runtime text—and freeze before model calls. Keep author-side coverage metadata richer than the text sent to Sonnet.
6. Evaluate a bounded sanity, then broaden by unresolved dimension. Analyze source → worklist → operation → stored fields → next-window verdict → retrieved use. Report both gains and regressions, not a scalar score invented after seeing outputs.

The finite procedure should balance productive discovery and truthful integration. Preserve four-list field comparison; do not add a fifth list simply because the coverage matrix grew. Arc should express meaningful movement, while durable rationale belongs in nodes. Its shared text remains a distinct gate.

## Current challenges and example probes — verified, not recalled paths

The inline infrastructure audit was checked against the current files and git history. Start with the checklist's authoring procedure around line 381, **A1–A11** and **E1–E24**, plus the weave method around line 700: reverse-derive what each example actually shapes, find missing shapes, and weave them into existing examples. Its historical heads are evidence, not current runtime state.

- `docs/challenges/semantic-fidelity.md`: current whole-memory calibration and semantic status/scope checks.
- `docs/challenges/{encode-notice,encode-absent,encode-write}.md`: known failure shapes.
- `docs/challenges/{prompt-genealogy,process,eval-method}.md`: prior outcomes and review/eval discipline.
- `eval/agent_introspect/quality_contract.py`: surviving 36-dimension contract, cross-dimension rules, authoring conventions and `validate_example_authoring`.
- `eval/agent_introspect/encoder_contract_eval.py`: independent evaluator framing and `load_example_for_eval`, separating self-claimed scores. Its CLI prints contract statistics; it does not run model evaluation.

**Important historical correction:** the old `servers/scales/s1/examples/` library, its renderers and `eval/agent_introspect/v20_assembly.py` were intentionally deleted in `654b2e2` because private conversations were shipping to installs. Do not restore that library. Current teaching examples are inline in `encoding_prompt.py` and frozen candidate templates. The old six-part authoring concept (register cues, decisions, counterexample, voice annotations, dimensions, valid alternatives) remains useful as a design idea. It does not prove those old files still exist or require hidden-chain-of-thought output; use concise observable choice explanations.

The surviving validator is stale for modern examples: it requires `<placeholder>` trace IDs and reads `encoder_output.nodes`/`revisions`, not `brain_batch.operations`. It can reject grounded example IDs while silently missing operations. Some contract scoring assumes old source-ref/detail rules. Reconcile it with current schemas and Tom's selective-ref and quote-credit rulings before using it as a gate. No modern generic structured-example renderer/validator was built in this session. Extend what the candidate actually needs; don't spend the successor session recreating the deleted library.

| Probe / entry | What to reuse | Caveat before use |
|---|---|---|
| `eval/identity_exemplar_probe.py` | Behavioral **TEACH** on unrelated material plus independent rotated **RANK**; supports `--trials 3 --raters 5 --out <new-prefix>` | Fixed identity candidates; imports this branch's SYSTEM_PROMPT, despite “live” wording. Paid API calls, no tool execution. Adapt its design for the woven example rather than running unrelated hardcoded candidates. |
| `eval/candidate_prompts/probe_scenarios/{README,closure_settled,closure_unsettled}.md` | Positive/negative pair for answered, partly answered and still-open claims | Old scout legend and stateless simulated writes; no dedicated runner there. Teaching evidence only, not persisted-state evidence. |
| `eval/encoder_prompt_probe.py <assembled-prompt> --out <new-report.md>` | Five independent prompt-reading lenses; `--no-parallel` supported | Hardcodes Sonnet 4.5 and has no model flag. Must adapt deliberately for 4.6; rule-reading is not demonstrated execution. |
| `eval/agent_introspect/encoder_replay.py` | Single-call tool-emission probe on oracle qids | Empty/approximate catalog and whole haystack; no executed writes or modern sequential continuity. |
| `eval/agent_introspect/quality_probe.py` | Prompt comparisons with `--qids`, `--prompts label=path,...`, `--trials`, `--model`, `--parallel`, `--out` | `_extract_nodes` counts only remember_batch, ignoring brain_batch remembers. Unchanged reuse could falsely report zero nodes. Paid model work and scratch eval brain. |
| `eval/fixtures/s1e_guide_v3_2_2026-09-10/arms.py` | Frozen arm assembly checks model/tools/closure; with no arguments validates existing hashes | Do not `--freeze` existing V3.2 or edit its pins. Copy/adapt for a new sibling V3.3 fixture. |
| `eval/encoder_eval/{runner,harness,quality_probes}.py` | Isolated LongMem encode + quality + downstream answer framework | Registered-version inputs and older stop conditions need adaptation for full packages. Stage ranges are comma-separated, despite README's semicolons. Ref/atomization counts are not current quality quotas. |

`eval/encoder_eval/version_pinner.py` was removed in `618e983`. The current harness reads templates and passes an isolated `s1e_override`; old `__init__.py` activation text is stale. `tests/interaction_override.py` is the supported override owner, applied only to isolated eval brains here. Never activate a live version to test a candidate.

One safe existing validation command, run from this worktree, is:

```bash
./dev python3 eval/fixtures/s1e_guide_v3_2_2026-09-10/arms.py
```

The successor should verify copied V3.3 assembly similarly. New model-probe commands above are a capability map, **not already-run validation or a request to launch every probe**. Select independent evidence for the decisions the new example is supposed to teach. Have a cold reader infer shapes from the example without its author-side matrix, then compare that reading with the intended matrix and behavioral transfer.

## Evidence already paid for

| Evidence | Finding / limit | Primary path |
|---|---|---|
| Current production vs V3.2 | 9 new + 9 reused; one synthetic creative-design source, 3 windows × 3 repeats; no overall winner | `docs/S1E-CURRENT-PRODUCTION-VS-V3-2-RESULTS-2026-09-11.md`; `eval/results/s1e_production_comparison_2026-09-11/` |
| V3.2 vs V3.1 whole memory | 9 + 9; mixed improvement; priority loss and no post-write semantic repair | `docs/S1E-V3-2-WHOLE-MEMORY-REVIEW-2026-09-11.md`; `eval/results/s1e_v32_semantic_sanity_2026-09-10/` |
| V3.1 cross-corpus | 54 encodes: 3 arms × 3 repeats × 3 windows × 2 sources | `docs/S1E-V3-1-CROSS-CORPUS-RESULTS-2026-09-08.md` |
| Content/lifecycle trace | Correct operations can preserve wrong claim identity, speaker, status or scope; both useful and harmful revisions | `docs/S1E-CONTENT-LIFECYCLE-REVIEW-2026-09-09.md`; brain `id:7d36279a`, `id:180f9bc0`, `id:62a71622` |
| Compression / initial sanity | Bounded comparison; static preservation is not behavioral equivalence | `docs/S1E-GUIDE-COMPRESSION-EVAL-2026-09-08.md`, `docs/S1E-GUIDE-SANITY-2026-09-08.md` |
| V3.3 sanity (development source) | 9 new V3.3 encodes beside the saved production and V3.2 nines; blind paired review 36 / 22 / 17 of 42 (V3.3 / V3.2 / production); the priority miss closed 3/3 | `docs/S1E-V3-3-RESULTS-2026-09-11.md`; `eval/results/s1e_v33_sanity_2026-09-11/` |
| V3.3 transfer (untouched material) | 126 encodes: 6 corpora × 3 arms × 3 repeats; census, 18 blind packs, downstream answer test; V3.3 strongest on ownership, revision reach and receiver's view, weakest on coverage of assistant-heavy Q&A; one revise-infra interaction lost an update | `docs/S1E-V3-3-RESULTS-2026-09-11.md`; `eval/results/s1e_v33_transfer_2026-09-11/`; brain `id:12098c8a`, `id:9f3c21de` |
| V3.4 one more weave (fresh material) | 126 fresh encodes (5 LongMemEval reserves + debugging synthetic; production / V3.3 on live tools / V3.4) + 42 regression + 9 sanity; question lane and thought shape moved, quote-refresh and coverage did not; all V3.x arms had run on never-deployed tool descriptions; one dispatcher crash on a string revision item | `docs/S1E-V3-4-RESULTS-2026-09-12.md`; `docs/S1E-V3-4-AUTHORING-2026-09-12.md`; `eval/results/s1e_v34_*_2026-09-12/` |
| Original seven arms | Other stream's hand-adjudicated gold and LongMem; use item rows and dumps | `/Users/tpac/AgentsContext/s1e-field-coverage-gold/ab_2026-09-01_03/ops9/ADJUDICATION.md`; original `docs/HANDOFF-S1E-GUIDE-EVAL-2026-09-08.md` |

Cross-corpus correction: the two V3.1 sources were synthetic `conv_004_art_design_extended` and the LongMemEval oracle `gpt4_f49edff3` (baby/sibling gifts). Oren was not rerun. “Zero standalone advice nodes” did not mean a zero-node run. No API/truncation failure explains those reported zeros. Wider source claims in stale memories were corrected.

Current-production comparison details, already recorded (do not recompute to restate): 56 vs 52 final nodes; 0 vs 6 distinct text-revised existing nodes; priority retained 3/3 vs 2/3; populated thoughts 9 vs 1; semantic relations 106 vs 69; all Arc phases retained 2/3 vs 3/3 (shared 800-character cap). Node+relation words 15,440 vs 11,993; generated output tokens 37,470 vs 38,930. These are descriptive, not quality quotas or the objective. Sparse thoughts also occur in earlier V3 (1), V3 + tools (0), V3.1 (1), each over the same three creative repeats.

In the production result folder, `whole_memory_review/*.md` contains all final fields and semantic edges for six graphs. `comparison_inventory.json`, `quality_inventory.json`, and `whole_memory_review/summary.json` retain measured inventories. Each `production_deployed/repeatN/creative_design/windowN/` has saved requests/results, snapshots and continuity. Candidate runs remain in their original V3.2 folder.

Useful concrete handles: V3.2 R1 graph `492707b8` drops priority after its `changes` list noticed it; R2 `8df39434` adds priority and proposed D3; R3 `97020aa0` preserves implementation additions. Production R2 `d7d7a8df` offers useful meta-observation synthesis; `a187e1d4` identifies the vision→features→beauty arc; `69dab691` overstates a behavioral “always”; `1ab0d719` distinguishes cold from bad. These are eval nodes in saved isolated graphs, NOT live brain IDs to get through MCP. The synthetic speaker named Tom is not evidence about the real operator.

Original ledger cautions: guide 25/30 scoreable versus candidate 29/38 use different denominators. Runs fetching live, already-fixed `d827d22f` cannot score that historical item. Residue-named target reads remained zero across historical arms; another prose fetch rule is not the remaining lever—rendering is. Guide v1.2 revise was strong, but create rejected first-disclosure facts and carried no-mint judgments into residue. The later audit also found its LongMem capture lacked the claimed lists-first preamble. Do not restore that arm wholesale or silently blend its baseline with the September 11 baseline.

## Evaluation design and scope

Protect facts, temporal detail, corrections and propagation, independent arcs, commitments/alternatives, behaviors, both voices, intellectual contributions, useful doubt, emotional texture, meaningful relationships and actual downstream use. Assess title/content/situation/question/reasoning/thought/quotes/edges for their contribution, not fill rates. Credit a quote that preserves sentiment. Distinguish adequate preservation, preservation with practical ambiguity, and materially wrong/missing knowledge. Inspect the retrieved subset separately; it can be misleading even when the whole graph holds a corrective quote.

Keep production, V3.2 and the new V3.3 as clear references. Reuse saved runs only when substrate, source, settings, assembled package and read behavior are comparable. Earlier candidates are archival references, not an obligation to rerun every arm. Pair per-item gains/losses; maintain three repetitions and sequential state within each repeat. Parallel arms/repeats use separate OS processes and independent closed-seed IsolatedBrain copies.

The exact next sanity matrix—arms, source and resulting call count—is not selected or frozen yet. The repetition/window policy is settled; selecting a discriminating bounded matrix belongs after authoring review. The authoring corpus must not become the only evaluation corpus.

Transfer-source entry points are `eval/corpus/loader.py` and `eval/corpus/conv_*.json`, plus `eval/longmem/corpus.py` and `eval/longmem/data/longmemeval_oracle.json`. `eval/longmem/realchat_corpus.py` describes the realchat source, whose local oracle availability needs checking. These are inventories, not certified held-out splits. Exclude the inspected creative-design and gift material, consult prior result manifests/reports for exposure, and record a split before reading selected transfer source/answers for authoring. No untouched subset was chosen during salvage; don't label an item untouched simply because this letter doesn't mention it.

Use distinct instruments deliberately: an example probe asks whether teaching transfers; an encoding arm comparison measures stored knowledge; a read-only diagnostic examines a known output; a downstream recall/answer test measures what future use recovers. An interview where Sonnet says it understood a rule does not establish execution. The nine historical interviews repeatedly quoted the rule; the visible procedure was the productive intervention. Do not conflate probe scores with arm KPIs.

The prepared six-call diagnostic in `eval/results/s1e_v32_review_probe_2026-09-11/` is held/unrun, not a missing mandatory evaluation. It disables tools and compares review with/without saved worklists; it is not another encoding or an introspective interview. An automatic approval rejection preceded any calls. It was held when Tom redirected to broad release quality. Existing Sonnet eval transmission was explicitly approved; do not relabel completed runs as blocked or ask the same approval again.

## Harness and production traps

The completed cells' `corpus_cell.py` files accept `--prepare`, `--preflight`, `--repeat {1,2,3}`, `--run`. They are hardwired to their frozen result folders; **do not rerun prepare/run in those directories**. `save(..., 'x')` and pins deliberately refuse overwrites. Build a new separately named wrapper for V3.3 using existing runner APIs, then inspect it and freeze it. Read `eval/fixtures/s1e_guide_v3_2_2026-09-10/corpus_cell.py`, the production wrapper beside its package, and `eval/s1e_guide_v2_sequence_probe.py`.

Both recent arms use a common branch replay/catalog/storage/journal and local dispatch lacking full S1Scribe provenance. The production package exposes only its native older tool shapes; it emitted zero candidate-only revisions. Neither comparison runs S1R/S2/answerer. A stronger answerer experiment must be wired and validated; it was proposed, not already built here.

September 11 deployed snapshot: daemon source `/Users/tpac/brain`, fingerprint `8c33e3f67dc54c06`; effective S1E default `fd28b7b6321f`, no override, Sonnet 4.6 medium, 12,288 max tokens, five rounds, no gist. Source was branch `codex/contract-host` at `c58f533`, while main was elsewhere at `e35575c`. Recheck through daemon/effective-interaction APIs before calling a later snapshot current. Frozen production arm SHA-256 `6e5a73832c453a4639ca91a5bd83540e53e14c01e2f03c22132b37e022b8250a`; V3.2 `15a9770ef2b21aa4939b127e3dcb51792c861ca0850130d0e03dfe7135821031`.

Use `./dev python3`, not arbitrary Python. Optional `jsonschema` was unavailable in the completed production preflight; the wrapper removed that unnecessary added dependency before checks or model calls, with prior manifest retained. This is not an invitation to weaken required schema gates. Do not edit manifests to bless changed model inputs. Original external `make_guide_candidate.py` anchors to the default as of `069f22c`; default changes can fail NOT UNIQUE. Keep old payloads and their changes/targets/fetch/new order.

The old external tools live at `/Users/tpac/AgentsContext/s1e-field-coverage-gold/ab_2026-09-01_03/tools/`. Historical approximate durations: one gold run ~4 minutes, 17-run cell ~50 minutes, LongMem build ~52 + sweep ~8. Use current bounded cost estimates for broader work; these are not promises for a new package.

## Local state and delivery

At salvage, HEAD remains `9a1727f`; runtime sources untouched. Existing tracked change: ten added lines in `docs/S1E-CHECKLIST.md` before this salvage. Many candidate/report/fixture files are untracked, and result folders are local/ignored. `:memory:.ses` is pre-existing; leave it alone. The salvage only adds documents/inventory and updates navigation in existing living docs. It does not commit or clean the worktree.

All known paid jobs owned by this work are complete according to their completion artifacts; no new model call was launched in salvage. The system process list was unavailable in the sandbox, so no claim is made about unrelated OS jobs. Read-only inline agents finish before delivery. No new user task is automatically started: the user receives a launch prompt and exact handoff address for a fresh session in this worktree.

Tom's outstanding gates: merge/deploy; shared trace closure/header/nudge; residue/edge-id catalog rendering; runner continuation after nonempty lists without a tool; production round-text traces; edge-repair mechanism vs teaching; confidence→label. Complete-package intent does not automatically select every parked experiment. Do the authorized authoring, review and bounded isolated evaluation; leave deployment behind Tom's explicit call.
