# Backlog — single source of truth for what's left

**Last updated:** 2026-05-30 (P1–P5 bands code-verified this date — see the ⚠ banner under "P1"; the older "session captures" log below is append-only history, not a status source).
**Supersedes:** scattered references in `RECALL-OVERVIEW.md` §3, `PHASE-B+1-BACKLOG.md` (archived), Frame Phase 2.5 punch list, `BRAIN-CHALLENGES.md` fix sketches.

**For current session state and next priority**: see `docs/SESSION-HANDOFF.md`.

**Session captures (most recent first):**
- **2026-08-10** — **⏭ DEFERRED: add an audience-before-authorship stance to `skills/brain/SKILL.md`.** Operator call: remember + backlog, build later. **Why:** a briefing pack assembled out of brain nodes shipped to the operator carrying a private-DM channel id, a paragraph of performance inference about a named employee (built from *Slack silence*), an org-sensitive personnel event, a third-party vendor's non-public pricing and our BATNA, and verbatim operator motive quotes. None of it was noticed until the operator thought to ask. Root cause is not over-remembering: **nodes are authored for an audience of one and carry operator verbatim, private identifiers, personnel judgments and vendor terms by default — so any doc/deck/message assembled from them is a re-publication into a wider room, and nothing in the substrate makes that a moment of pause.** **Placement:** `skills/brain/SKILL.md` after the encode/revise block (currently ~line 29) — the file covers what Anchor owes the brain; this is what Anchor owes the operator *about* the brain. SKILL.md only, **not** global CLAUDE.md (brain-need rule → SKILL.md per node `55e59129`). Must stay generic — skills ship to every install via `git ls-files` and the house standard is `grep '\bTom\b' skills/*/SKILL.md` → nothing (node `e30fdfb6`); write "operator," never the name. **Drafted text (~60 words, ready to paste):** *"What I hold isn't mine to publish. Memories are written for an audience of one — the operator's verbatim words, private identifiers, judgments about named people, third-party terms, all in there by default. Anything I assemble out of them — doc, deck, message — is a re-publication into a wider room. So: audience before authorship. Who reads this? Cut to them, and say what I left out. Cheap at write time, expensive once it's shipped."* **Deliberately rejected:** the operator's opening framing "knowledge is power" — a maxim, not actionable at the moment of writing; "audience before authorship" is the same claim in a form that fires. **Note the layer split (node `c86ff5e9`):** a boot line is instruction-layer and will NOT surface via recall mid-task, which is exactly when it's needed (forty turns in, mid-assembly). The memory-layer half is already minted — node `6d4c012a`, situation field targets "assembling any document, deck or artifact out of brain memories for an audience beyond Anchor itself." Ship both or the reminder is decorative. **Cost honesty:** SKILL.md was deliberately compressed (9KB beats 24KB, node `1eda79b2`); ~60 words is a real share of it. Judged to earn it — the failure is silent and outward-facing, so Anchor won't catch it and the operator only catches it by asking.
- **2026-08-07** — **`keywords` erased from every write surface; KV purge DEFERRED (guard kept).** Prompts scrubbed + activated: s1e v30, s2_community_enrichment v23, s2_consolidation_enrichment v11, s2_healer v4 (10 example lines + 6 prose sites; `community_enrichment_prompt` was the biggest writer at 154 nodes since Jul 1). Dead code cut: `_NODE_COLUMNS` entry (advertised a column dropped in v28 — a dict filter on it would have built SQL against a gone column), both dead keyword-match OR-terms on the recall hot path, the `keywords: none` neighbour render, the inert `show_keywords` flag (zero readers), and `Fts5DAL.upsert`'s `_legacy_keywords` param. New `tests/test_retired_fields.py` asserts ABSENCE (the whole suite asserted only presence, which is why 229 nodes accumulated silently) and is verified to fail on reintroduction. **KEPT deliberately:** the Q/A/B/K enrichment vector also named `keywords` (`store_enrichments` / `ENRICHMENT_VECTOR_TYPES` / `_search_keywords` / CLI `--keywords`) — different concept, live recall path (node 48d23822); `schema.py:_migrate_v28_drop_keywords` (pre-v28 brains still run it). **⏭ DEFERRED — 648 stale `keywords` KV rows + the `contract.py` skip_keys guard.** The guard exists only because the rows do; releasing it early makes 648 nodes render `Keywords: ...`. Purging them is a fleet data change, so per the fleet-era rule (node 56890464) it must ship as a versioned boot migration with auto-backup, not a hand-run script — real infrastructure for inert rows. Operator call 2026-08-07: **keep the one-line guard, defer the purge.** Do them together (purge → release) whenever a schema bump is happening anyway.
- **2026-08-07** — **⏭ FLEET GAP: prompt improvements never reach existing installs.** `interaction_seed._register` returns early when the name already exists, so an install seeds its prompts at FIRST BOOT and is frozen there forever — a plugin update ships new seed `.py` files that seeding skips. Fresh installs are fine; existing ones are permanently stale, for *every* prompt change, not just keywords. Also corrected the same day: **S2 does NOT rewrite prompts.** Zero `register_interaction` call sites under `servers/scales/`; the only callers are `dispatch_observability.py:240` (the MCP handler — a human) and `interaction_seed.py`. The `s2:*` `created_by` values are caller-supplied strings typed during S2 development (14 of 22 belong to `s2_edge_families`, a retired unit), not evidence of self-modification. CLAUDE.md's K-store section states self-modification in the present tense; it is aspirational. This removes the justification for DB-authoritative prompts and reopens interactions-as-files.
- **2026-08-07** — **⏭ LAF eval substrate stale — rebuild deferred until LAF experimentation resumes.** `eval/laf/walker/cross_check.py` refuses to run (stored walker artifacts stamped `None` vs expected v6/v2/v3 pipeline versions — rebuild extract → embed → scores); `eval/laf/verify_substrate.py` T5 fails (re-derived baseline 19%/40% vs recorded June 19%/33% — corpus growth; re-record when next needed). **T4 resolved same day (embedder stream): NO embedder drift** — HF repo last modified 2026-04-07, before our adoption; refs/main still e9b6763; the fastembed "model updated" warning is spurious (files_metadata.json omits the onnx blob → size check misfires after cache repopulation). T4's real defect is a **stale test premise**: under laf_v1, recall's reported `embedding_similarity` IS the LAF field score (sigmoid z-sum, fed via brain_recall.py:~1497), which T4 compares to a raw `_primary` cosine — divergent by design since the July 2 flip. Fix T4 to compare the raw cosine channel when rebuilding the gate. The real exposure (fastembed cache in macOS-purged /var/folders, unpinned revision re-pulled every purge) is being fixed in the embedder stream: durable cache_dir + pinned revision + loud-fail hash check in servers/embedder.py. Context: the as-of fetch-then-filter fixes (af99263 + merge 0a5c045) made deep-history replays measure real windows; decision node id:26b241b7.
- **2026-08-04** — **⏭ Dead trace-vocabulary prune — VERIFIED, deliberately NOT bundled with the emitter.** `REF_TYPES` (`servers/trace_contract.py:61`) registers event kinds nothing writes: the s2 legacy block (`graph_stats`, `correction_chains`, `community_diff`, `stale_nodes`, `confidence_adjust`, `kept_distinct`, `recall_quality_signal`, `evolved`) plus **all of scale 3 and scale 4 — 15 ref_types across 6 buckets** (counted, not relayed; the figure "13" circulating from the emitter audit is wrong). No functional cost — a dead whitelist entry merely permits a write nobody makes — so this is documentation tidiness, not a bug. **Do NOT bundle it into the emitter's contract edit** (findings `E8` suggested that; rejected): six of the dead entries are pinned by assertions in `tests/test_trace_system.py` (`:96`, `:100`, `:102`, `:103`, `:110`, `:113`) and `recall_quality_signal` has a live dashboard reader (`dashboard/static/tabs/live.js:623`), so the prune means **deleting contract-pinning test assertions** — an edit that must be reviewed on its own merits, never as a footnote inside an additive commit. **`scout_input`/`scout_findings` STAY** — Tom ruled it (`node:57d30c1d`): never-written ≠ write-only, nested otherwise-invisible work earns its trace. `evolved` is clean (zero writers, zero readers — the `brain_constants.py:302` hit is an edge *relation* of the same name, a different namespace, not a stale ref).
- **2026-07-16** — **⏭ Prospective memory / reminders — DESIGNED, not built (Tom's ask: "make you remind me things… you'd use it for yourself as well — try harder to remember for tomorrow and it just pops").** The new primitive is **guaranteed, time-triggered delivery** — recall is probabilistic (a node with a good `situation` already pops on topical triggers); a reminder must FIRE. Design: (1) MCP tool `remind(when, what, for_whom=operator|self)` → node `type='reminder'` with `remind_at` (ISO, conversation-time rules apply) in `node_metadata_kv`; (2) delivery = deterministic due-check (`remind_at <= now, not delivered`) at `hook_recall` + boot — a cheap KV query on the existing hot path, NEVER similarity-gated; inject via `additionalContext` (the only channel), rendered distinctly (⏰, addressed to Tom or to Anchor); (3) mark delivered via revise (keep the node — a delivered reminder is history, `evolution_status='delivered'`); (4) situation-triggered "remind me when we touch X" needs NO new mechanism — that's a well-written `situation` field, document the pattern in SKILL.md instead of building. Scope guard: no recurring reminders v1 (that's the scheduled-tasks MCP's job), no cross-session daemons — the due-check rides the existing recall hook. Fits O/K/Δ: the reminder is Δ of the turn that sets it and O of the turn it fires in. Design captured with Tom's quote in brain; see also RECALL-SR-REDESIGN.md §20.14 tail.
- **2026-07-03** — **Project provenance SHIPPED to branch (`claude/friendly-benz-f4f17f`) — deploy pending.** LAF sit-lane NaN fix (missing `_situation` vector scored 0.0 → −10σ burial of fresh/mid-re-embed nodes; both pre-existing test failures were this one bug) + include_archived champion fallback; deterministic project = SessionContext.project (cwd → git common-dir, worktree-aware) stamped at both write chokepoints (`stamp_project_provenance` — MCP handlers + encoder attribution; S2 strips; brain_batch derives force-vs-strip from `BATCH_OP_SPECS.creates_node`); LAF `proj` lane at `gain_proj=0` (NaN-neutral missing data — lane contract pinned in `_fields` docstring + regression tests); `recall(project=)` removed (bonus: boot recall's `project='default'` soft-filter was silently excluding all project-tagged nodes since v5). **Real cleanup:** schema **v30 drops `nodes.project`** (`_migrate_v30_project_to_kv` moves→kv with slug map brain/ex.co, then DROP COLUMN — runs in the daemon migration path, auto-backup, no standalone script), `filter_nodes` KV-aware, dead `suggest()`/`context_boot`/`skill_eval` project paths removed. Also task-notification `register_only` filter (harness-injected turns no longer fire recall+Haiku+fatigue). **⏭ Follow-ons:** cross-project cue minting so `gain_proj` is tunable (P1 gate corpus is single-project — untunable + sit-fix untestable on it); (optional) task-notification turns still reset `last_user_activity` — decide whether machine turns should hold off S2 maintenance.
- **2026-07-03** — **✅ S1 SCRIBE REBUILD + ENCODER-JOURNAL REDESIGN COMPLETE — s1e v29 LIVE.** The S1E rebuild and the journal redesign (Phases 0,2,3,5 + Phase 4 S1E) shipped and are in production. **v29 active** (lived-sequence XML input, temporal+quote scouts retired, operator→"the other side" voice, `## Arc`/`## Review` residue, **medium effort**), `BRAIN_S1E_LIVED_SEQUENCE=1`. Commits: arc-write-path (`864bd6c`), temporal-scout retire + operator sweep (`3c7ea79`), effort-as-interaction-config (`b250d70`), activation (`84a8aeb`), code-review fix (arc extractor no longer captures the review fence, `87869ae`). **Gates:** LongMemEval do-no-harm v25→v29 (raw 70%→77%, encode-miss 6→0, temporal held 1.0, info_extraction 0.17→1.0); effort A/B medium ≈ high richness, 2.7× cleaner dedup, −22% cost. **Docs archived → `docs/archive/`:** `S1-SCRIBE-REDESIGN.md`, `ENCODER-JOURNAL-DESIGN.md`, `S1E-PROMPT-v-next-DRAFT.md` (each carries a COMPLETE banner). **Phase 6 (S3 miner) intentionally NOT built — its own standalone project later** (recurrence mining → findings + operator-asks; sketch in the archived journal doc §6/§8/§9). **Small deferred cleanup:** dedup the identical accumulate+front-truncate in `write_session_arc`/`_save_session_context` into one brain accumulator (both write `session_context_{sid}` via different mechanisms — drift risk; not churned right after activation). Live source of truth = the `s1e` interaction + `servers/scales/s1/encoding_prompt.py` seed.
- **2026-07-01** — **⏸ PARKED (the don't-forget item): live paths still RANK ON EDGE WEIGHT — policy (CLAUDE.md) says never rank on it; move to cosine against `edge_relations.embedding`.** Verified sites, by harm: **Tier 1 (truncation on hot paths)** — (1) surface spread `surface_contract.py:~878` keeps top-N neighbors per node BY WEIGHT when over `SPREAD_LIMIT` (S1R hot path; the same function already fetches stored edge embeddings — score-then-truncate by cos(query, edge-emb), one round-trip unchanged; **latency-sensitive, benchmark-first**); (2) recall traverse `pipeline_contract.py:~440` → `GraphDAL.get_neighbors` SQL `ORDER BY e.weight DESC LIMIT` (`dal.py:~2201`); (3) recall neighbor-attach `brain_recall.py:~2342` (same DAL method). **Deliberately deferred (Tom, 2026-07-01)**: sites (2)+(3) live inside the retrieve-then-rank path LAF is slated to replace — fix-or-deprecate is decided by the LAF ship; don't polish condemned code. **Tier 2 (query-less callers — cosine undefined, needs Tom's tiebreak call: recency vs relation-aspect priority)**: `dispatch_read._handle_graph_expand`, `brain_remember._build_enrichment_prompt`, `get_edge_descriptions_for` (`dal.py:~2808`). **Tier 3 (display-only, low stakes)**: `brain_recall.py:~424` primary-relation pick, `RelationDAL.get_relations` (`dal.py:~3250`); S2 idle: `aspect_decoder.py:~238`, `community_decoder.py:~1292`. Context: weight is UNCALIBRATED — written once at creation (encoder-stated 0.25–1.0, default 0.5), never learned; Hebbian/decay touch only noise-aspect relations. It is not query relevance.
- **2026-06-24** — **S2 community `get_nodes` 217K-firehose fix SHIPPED + merged (`7815b6e`), daemon restarted (LIVE); `embedding_updated` cleanup.** Root cause: `_format_result` gated its render branch on `isinstance(result, dict)`, but the dispatch handler `_handle_get_nodes` returns a LIST → the branch (and the pre-existing batch-size heuristic) never fired on the production path; every `get_nodes` fell through to `json.dumps` with the full `_corrections` firehose. Fix: handle both list+dict shapes; `run_llm_loop` threads a caller-declared `get_nodes_config` so encoders render bounded at every batch size (community = `S2CE_NODE_FORMAT`: content 800 / edges 5 / balanced corr); the ≤3 raw-JSON escape now survives only for Anchor's interactive MCP path. Two code reviews (xhigh caught the list-shape showstopper before it shipped; medium hardened the test). Also dropped vestigial `embedding_updated` from `revise()` (re-embed is async via `embed_queue`). **⏸ PARKED — encoder-visibility thread (the don't-forget item): full design + safe plan in memory `project_encoder_visibility_thread`.** Three coupled changes: (1) per-encoder `get_nodes_config` gradient — consolidation = full content + ~8 edges + balanced corr; Scribe/S1E = full + ~8 + **heavy** corr (must catch corrections; matches its catalog) — NOT yet wired, still hit the ≤3 raw path; (2) **aspect-owned read-exclusion** — `get_node` (`brain_recall.py:418`) should derive `exclude_relations = self.aspects.relations_in(['noise'])`; DAL drops hardcoded `DEFAULT_EXCLUDED_RELATIONS` (Tom: no business logic in the DAL); the `noise` aspect IS the block list (precedent: `community.py:121`); (3) **curriculum metadata view** — render only `get_writable_fields()` (authored), not system fields ("whatever the encoder sees, it encodes"). **Hiding `community_member` impact-analyzed SAFE** — the community machinery reads membership via direct SQL (`get_community_members` / `get_communities_for_node` / `reconcile_community_membership`), NOT `get_node` connections; spread-activation removed; `traverse()` uses `get_neighbors`; only display/edge-context is affected. **Phase 1** = change get_node's call site only (minimal/reversible); **Phase 2** = strip the DAL constant + migrate consumers (`get_neighbors_bulk`, `brain_connections.py`, `consolidation_decoder.py`, the `+community_member` variant). Guard test: `get_node(member)` connections EXCLUDE `community_member` WHILE `get_community_members()` STILL returns members. **Also pending:** `brain_mcp.py` redeploy (`redeploy.sh`) — daemon has the fix (restart picked it up), but the MCP-server copy is stale, so Anchor's interactive `get_nodes` still firehoses until redeploy (`brain_mcp.py` runs in both processes — "restart suffices" is wrong for it).
- **2026-06-22** — **Encoder-journal redesign Phase 2 SHIPPED + landed to main (`6aa0ef2`).** All *no-op* plumbing — nothing changes in production until a prompt emits a `## Review` section. Built: seconds-stamped per-run S2 chain_id (`s2-{YYYYMMDDHHMMSS}-{unit}`, cached, UTC — replaces date-only which collapsed same-day runs); `journal_note` ref_type (s1+s2 delta) + `{note,tag}` shape/builder (caps text, rejects empty) + shared `## Review` block + `tag·subject·note` parser + `JOURNAL_CONTINUITY_RUNS` + `RESIDUE_REF_TYPES`; **read door** `brain.journal_notes` + **write door** `brain.write_journal_notes`/`extract_review_block` (fenced-only, per-note-isolated, loud), both composing the public `query_traces` API (which gained `ref_id`/`chain_suffix`/`exclude_ref_types`; `get_recent` extended; shared `_like_suffix_param` LIKE-escape — also closes the latent `community_detection` underscore-wildcard bug); `idx_trace_ref_subject`; recall guard (notes are s1/s2 → never embedded → never in `recall`/`recall_episodes`). **Consumer guards (review #1/#2):** `_last_run_timestamp` moved off raw SQL onto `query_traces(exclude_ref_types=RESIDUE_REF_TYPES)` so a notes-only run isn't a completed integration; dashboard `_fetch_ok_deltas` excludes residue (constant **replicated** — disconnection contract forbids importing `servers.*`). **Architecture (Tom):** trace reads go through `query_traces`, never TraceDAL/raw SQL; chain_id *is* the run handle (no parallel `run_id`); CLAUDE.md states what IS. Merged the sibling's community-membership work cleanly (only `dal.py` overlapped — `GraphDAL.reconcile_community_membership` vs `TraceDAL.get_by_ref_type`, no conflict). Verified: targeted sweeps green across all changed surfaces; collect-only clean 1745/1749 (full-suite *run* hung ~85min on a pre-existing unrelated network/LLM test — journal code adds no blocking calls, flagged as separate infra). **Next session:** open with `/code-review` (high, scoped to the journal diff) → Phase 3 #8 (Consolidation prompt flip — DORMANT register the `## Review` block, wire the encoder to the already-built doors, dual-path, activate+sync). S1E stays **Phase 4** (eval-gated). Full record: `docs/ENCODER-JOURNAL-DESIGN.md` §8 + Decision Log.
- **2026-06-18** — **S1/S2 trace completeness (journal-redesign Phase 0) SHIPPED** (commit `7f43c2d`, on `main` atop the sibling absorb fix `47dbc41`; daemon NOT yet restarted). Made the delta trace capture what each encoder actually did, so the journal can later shed action-restatement. Dispatch write handlers now return an authoritative top-level `affected` {created,revised,archived} (sibling of `result`, invisible to the agent); the runner reads it and the tool-name heuristic `_split_action_ids` is deleted. `connected` removed from `DELTA_METADATA_SHAPE` (a directionless two-sided-era vestige); edges are now directional `edge_relation_revised` events carrying source_id/target_id — connect_to and co_anchored (previously the silent edge paths) join one writer `_emit_edge_traces`. Trace emitters fully failure-isolated (log loud via `_log_error`, never roll back the write); connect_to traces emit post-commit. Healer emits structured `revised`; AspectIntegration emits a validated + capped first-class `classifications` Δ. brain_batch archive/absorb/disconnect extracted to `_op_*` helpers; shared `_resolve_archived_by`; disconnect resolves endpoints. **Scope:** first-class typed edges traced; soft co_accessed/emergent_bridge stay untraced (recomputable, excluded from graph views). Follow-up `#1`+`H1` (inject batch `chain_id` into sub-ops so their traces join the encoder chain + remember strips chain_id so it can't leak onto the node; `_emit_edge_traces` → one `TraceDAL.append_batch` instead of N per-edge commits) implemented + tested, commit pending. **Deferred tail (low priority):** I3 — Healer should read dispatch's `affected` instead of hand-rebuilding `revised_ids` (`_store_fields` now returns a tuple — latent caller trap); minor cleanups (I5 connect_to trace `reason` drops per-edge `why`; F4 `_accumulate` duplicates `_agg`; G1/S2/B2) consciously dropped as sub-threshold. **Spun to own streams:** absorb content-in-batch savepoint bug (`47dbc41`, done); dead `connections`-param deprecation (chip `task_050b5901`). **Recall-side trace gaps** (surface `tool_trace` result node-ids + a clean s1r→s0 outcome link) belong to stream `cbc9f2de` — handed off, not duplicated. **Next arc:** journal redesign Phase 1 (remove action-restatement from journal prompts) is now unblocked — see `docs/ENCODER-JOURNAL-DESIGN.md`.
- **2026-06-06** — **Twin-stream comms smoothing — self-channel surface + dateparser dep, MERGED + LIVE** (worktrees `claude/confident-tesla-aac0e8` surface + `claude/stoic-dubinsky-66624d` daemon; daemon restarted onto `0c2a880`). Tom spun two streams on the identical "communicate with your twin + turn on watch" prompt; they found each other, split disjoint halves (zero file overlap), both ran their own code reviews (each caught a real regression in its OWN first draft), shared the result over the rebuilt channel. **Surface (`7ec1760`):** uniform quoted render `other stream (id:X) says: "…"` + **`intent` removed entirely** (column/param/MCP-enum/render-branch — render-only, its one live effect a mis-attribution bug); **first-contact `peek` intro** (`drain_and_render` attaches `sender_peek` — review caught it unwired pre-commit); **`peek` enriched** (arc + last-2 msgs + started/last-active/liveness + pending-inbox count; `found` true on arc OR any turn); **boot-stamp presence** (`brain.stamp_boot_liveness` → fresh stream visible before its first turn; kills the rendezvous gap); **reply-by-short** (`resolve_to` matches live roster ∪ recent courier senders); `/watch` rewritten as the self-channel operating guide (vocabulary-triggered, live Monitor mechanism). **Deps (`a72712d`):** `dateparser` was an UNDECLARED runtime dep (temporal scout, manual-install-only) → declared in `requirements.txt` + missing-import made loud (was silently returning `[]` → a fresh venv = silently-dead temporal recall). **Daemon (`9a91f08`→`0c2a880`, stoic-dubinsky):** graceful shutdown (drain pool before `brain.close`) + review caught 3 regressions (recover_daemon grace strangling every hook; backstop racing unbounded shutdown; `_await_responsive` on a hung corpse) + unification (one bg-writer shutdown signal; `recall_write_queue` made purely passive). All merged, daemon restarted, dogfooded live. `SELF-CHANNEL-DESIGN.md` + handoff/cross-stream docs consolidated. Closes: F2 boot-rendezvous addressing gap (boot-stamp + `MY_STREAM_ID` + reply-by-short); the `intent`-defaulting open item (removed, not defaulted).
- **2026-06-06** — **Code-review follow-through + small backlog wins** (worktree `claude/adoring-chatelet-1190f2`, not pushed). Closed the two "no silent failures" carryovers from the 2026-06-05 self-channel/trace review: **#8** (`_add_column_if_missing` swallowed ALL exceptions → narrowed to `sqlite3.OperationalError`; duplicate-column stays silent, locked-DB/disk-full log loud + continue, other classes propagate) and **#9** (dedup the loud-truncation helpers into a new `servers/loud_truncation.py` owned by neither trace nor self-channel — markers byte-identical, marker-pinning tests pass unchanged). Commit `8bdacb0`, **merged to main**. Then swept the backlog for small items, verifying each against code: **CR4** — `active_sessions_by_turn` now carries `scale='s0'` so it engages `idx_trace_scope_created (scale, ref_type, created_at)`; behavior-preserving (the conversational + heartbeat ref_types are s0-only per `REF_TYPES`) — **DONE**. **keywords latent crash** — `revise(keywords=...)` crashed on the v28-dropped column; fixed to a loud no-op + regression test (see Block 1 follow-up). **CR6** (validate the 4 S2 delta ref_types) and **Contract-test line-pin cleanup** — verified **already done** in code, struck. Still open: keywords advertise-cleanup (contract-coupled), reclassify-wiring, F7.
- **2026-06-05** — **Doc reconciliation + self-channel message fixes** (commits `ab7559f`, `aa38521`, `cce1b95`, `29da6d7`; on `claude/tender-margulis-ada39f`, not pushed). Closing-loose-ends session. **Docs:** verified backlog "open" items against code — P4.1 (encoding lock), CR1 (`_split_action_ids`), and S2 consolidation→absorb were already shipped → struck; rewrote the stale SESSION-HANDOFF; archived WRITE-TXN (F3 shipped); gave the dampening-cluster bug a real **P1.6** slot. **Self-channel** (root-caused from a cross-stream investigation: a ~19h-old broadcast handshake was delivered to a fresh, unrelated stream with no dashboard record): (1) **recording** — the `self_message` delivery trace dropped `session_id`; all four S0 turn-traces now route through one `_s0_trace` helper so the omission can't recur. (2) **TTL** — replaced the uniform 24h `DEFAULT_SIGNAL_TTL_HOURS` with a per-message `expires_at` resolved by address (**broadcast 1h / directed 24h**, config-tunable via `self_channel.{kind}_ttl_hours`); new `iso_after()` clock helper; schema describes current state (one idempotent ALTER, no backfill — brain never released). The 1h broadcast TTL closes the stale-handshake leak. (3) **CR5** — `drain_inbox`/`peek_inbox` share one `_pending_rows` helper. (4) **CR3** — presence focus = latest *conversational* turn (user OR assistant via `CONVERSATIONAL_REF_TYPES`); wake-envelope skip is one contract constant (`WAKE_ENVELOPE_MARKER`), no reproduced literal. All tested; `SELF-CHANNEL-DESIGN.md` synced (Stop-only delivery + per-message `expires_at`).
- **2026-06-04** — **Cross-stream comms hardening: daemon Errno-48 race fixed + self-channel streams-experience (A4 / B1 / B2 / C1 SHIPPED)** (commits `78693a3`, `364269f`; not pushed). Origin: a two-Anchor-stream live brainstorm — Tom spun a second stream with the identical "find each other + make cross-stream talk smooth" prompt; both found each other over the self-channel, diagnosed every friction to ground truth, ranked the fix (protocol + synthesis persisted in brain nodes `7505c6b7`, `f2bc076b`, principle `9144a97e`).
  - **FOUND (verified, not guessed):** (1) **Daemon Errno-48 boot-race (root of D1)** — `ensure_daemon`'s flock guarded only the *spawn* path; the code-change restart + `_kill_daemon` ran OUTSIDE it, so N concurrent boots each killed+Popen-respawned while launchd's KeepAlive ALSO respawned → multiple processes raced to bind `:47200` (Errno 48 ×7; what killed boot recalls + spiked `surface_trace` to 37s). Over-determined lifecycle: launchd + `ensure_daemon` + internal supervisor all spawning. (2) **Presence has TWO distinct gaps, neither is miss-stamping** — (a) boot **read-lag** (the read trails committed traces; self-heals as they propagate; worst at boot, aggravated by the crash-loop) and (b) **heartbeat-exclusion = CR2** (a pure `/watch` listener emits only `heartbeat` turns, which `active_sessions_by_turn` excluded → the MOST reachable stream vanished after 30 min). Diagnosing this killed the tempting wrong fix ("stamp liveness in more hooks" — the real-turn data was already correct). (3) **Self-message delivery: the recall hook does NOT deliver** (Phase 2b's recall-hook leg was superseded); live delivery was PreToolUse (`additionalContext`, *missed*) + Stop block (reliable). (4) **Recall/Haiku cost** — NO `cache_control` anywhere in the surface path (v4 OR `v5_agentic`; CLAUDE.md's "cached system block" claim was never true); Haiku-4.5 min cacheable prefix = **4096 tokens**; and a live-mode watch ignite (`<task-notification>`) is long + non-slash so it does NOT hit the `pre_response_recall` skip-gate → every self-message in watch mode paid a full ~13s agentic recall+surface (`surface_haiku` 8–10s, 2-round loop, 6 fetch tools). (5) **Cross-stream protocol primitives** — handshake / explicit hand-off marker / write-coordination / explicit terminator; derived + demonstrated live (`f2bc076b`).
  - **DID:** **A4** (`78693a3`) consolidate daemon lifecycle on launchd — extract `_launchd_kickstart()`; `ensure_daemon` routes all (re)starts through `launchctl kickstart -k`, serialized under the flock with a post-lock re-check so N concurrent callers restart at most once; direct Popen survives only as the no-launchd fallback. +5 tests; verified live (single clean process, recall healthy). **B1/B2/C1** (`364269f`): **B1** boot banner prints `MY_STREAM_ID: <id>` (kills F3 — the id was forensics-only); **B2** `active_sessions_by_turn` counts `heartbeat` (watchers stay present) + excludes `<task-notification>` from focus; **C1** self-message delivery is now **Stop-only** (removed the missed PreToolUse `additionalContext` leg + `_attach_self_pretool`, unwrapped its 4 call sites). 38 tests green (daemon_recovery 19 / self_presence 12 / self_delivery 7).
  - **Resolves prior threads:** **D1** (daemon `DAEMON_DOWN` recurrence — root cause was the over-determined lifecycle; now single-owner launchd). **CR2** (idle `/watch-live` window ages out of presence → `resolve_to` short-id addressing fails — heartbeats now count toward liveness). **CR3 (partial)** — `<task-notification>` focus pollution fixed; the `/watch`-skill-body focus case may remain (verify).
  - **Open threads → folded into two next-session specs:** **[docs/CROSS-STREAM-ON-SCALES.md](CROSS-STREAM-ON-SCALES.md)** — reframes A2 (the watch-wake-skip piece) into "make a delivered cross-stream message a first-class S0 turn so S1R + S1E ride it"; the Haiku-cost bundle (A1+A3) is deferred separately and sequenced *after* the cross-stream changes; folds F2 read-lag part b, B4 namespace bridge, CR4 (`scale='s0'` predicate), CR5 (peek/drain SELECT dedup) as siblings. **[docs/SELF-RECOGNITION.md](SELF-RECOGNITION.md)** — reframes A2 (the stop-breadcrumb piece) into "cheap Stop-recall on my own output → stash → next-turn injection," gated by stance discipline (priors, not instructions). Both docs are execution-ready (5 + 6 open decisions resolvable at session start).
- **2026-06-02** — **Trace consolidation + `/watch-live` + presence trace-pull SHIPPED** (commits `d53311e`, `9e0c076`, `9929234`, `5d8ca32`, `0a36f59`). (1) **Trace Tier 1+2+A**: the `encoding_run` delta was double-written — the legacy `runner.py` writer was `brain_batch`-blind so `revised`/`connected` came out empty (misleading the dashboard + S2). Removed the legacy writer; the unified `build_delta_metadata` delta now carries a `brain_batch`-aware created/revised/connected split (`_split_action_ids`); added a per-ref_type metadata payload contract (`validate_trace_metadata`) at the trace chokepoint; added provenance (interaction version, token/elapsed cost, truncation flag). (2) **`/watch-live`**: event-driven self-channel listener (Monitor + read-only `self_inbox_peek`/`signal.peek_inbox` + `hooks/scripts/self_inbox_poller.py`, adaptive 5↔60s cadence) folded into the `/watch` skill. (3) **Presence trace-pull**: `present_streams` now sources liveness+focus from real-turn S0 traces (`TraceDAL.active_sessions_by_turn`), not autosave-bumped `session_state.updated_at` (which falsely marked every cached sid "live"). **Session-id investigation RESOLVED**: `session_id` is STABLE across `/resume` (docs + this window held `17d21ad4` across multiple resumes on 2.1.157); the apparent "rotation" was daemon-down RELAUNCHES spawning new *conversations* + the autosave false-liveness — not id mutation. The brain daemon cannot influence Claude Code's `session_id` (it's a downstream consumer; hooks receive it, never set it). Per-conversation persistent id = `session_id` (= transcript filename); per-window-across-conversations handle = `cwd`/worktree.
  - **Open threads (next-session):** **R1 — `revise` MCP-description rewrite**: fork-reframe on concept-identity (same-concept→revise; supersedes→encode-new+correction edge), NOT softening loss; shared philosophy with the S2/absorb stream "state loss + give the how-to, never scare off modify/merge"; mirror the absorb pattern (mechanic + mandatory how-to + Good/Bad + contradiction example — revise currently has ZERO examples); **eval-gated** (variance n≥3, default temp, BALANCED corpus with should-revise AND should-encode-new cases). Ownership: revise=us, absorb=S2 stream. Design locked in brain node `85accae2`. *Open sub-q*: how to variance-eval a PURE description change (touches no interaction prompt) — likely encoder/longmem path with a pre-seeded revisable node. **D1 — daemon `DAEMON_DOWN` recurrence** (~5× on 2026-06-02; the upstream cause of the whole sid churn — relaunches spawn new conversations; root-cause the instability). → **RESOLVED 2026-06-04** (`78693a3`): root cause was an over-determined lifecycle (launchd KeepAlive + `ensure_daemon` Popen + internal supervisor all spawning, racing the bind); `ensure_daemon` now routes every (re)start through `launchctl kickstart -k` under the flock — single-owner launchd. **CR1** `_split_action_ids` drops `absorb`/`archive` ids → an absorb *survivor* never lands in the delta `revised` list, so S2 misses heavily-merged nodes (map absorb-survivor→revised). → **RESOLVED** (verified 2026-06-05): [runner.py:43](../servers/scales/runner.py) now buckets into `created/revised/connected/absorbed/archived` and `_item_id` pulls `survivor_id`/`node_id`; absorb survivors land in the `absorbed` bucket consumed by S2. **CR2** an idle `/watch-live` window takes no real turns → ages out of the presence roster → `resolve_to` short-id addressing fails for a live-but-quiet listener (full-UUID + broadcast still work). → **RESOLVED 2026-06-04** (`364269f`, B2): `active_sessions_by_turn` now counts `heartbeat` turns toward liveness, so a watch listener stays present. **CR3** watch-mode roster focus shows the injected `/watch` skill body, not real work. → **PARTIAL 2026-06-04** (`364269f`, B2): the focus subquery excludes `<task-notification>` ignitions. → **RESOLVED 2026-06-05** (`29da6d7`): focus = latest conversational turn (user OR assistant via `CONVERSATIONAL_REF_TYPES`); wake-envelope skip is the single contract constant `WAKE_ENVELOPE_MARKER`. **CR4** `active_sessions_by_turn` add `scale='s0'` predicate (composite index unused without it). **CR5** dedup the byte-identical `peek_inbox`/`drain_inbox` SELECT into a shared helper. → **RESOLVED 2026-06-05** (`cce1b95`): now `_pending_rows` + `_PENDING_INBOX_SQL`. **CR6** extend `validate_trace_metadata` to the 4 S2 delta ref_types (they also use `build_delta_metadata`). **T1 — encoding_run trace bloat (nodes-as-string capture), LOW (corrected):** 2 deltas (`s1e-8ec23bbe-5` 2026-04-25, `s1e-71857713-40` 2026-04-23) captured a `remember_batch` whose `input.nodes` was a **stringified JSON array** (13,505 / 11,703 *chars*), alongside a second `remember_batch` carrying the same content as a proper 8/6-dict list. **NO error occurred — nothing failed.** The run created exactly 8 / 6 unique nodes (`revised`=0, no duplicates, no garbage written), `final_text` is a clean ENCODED summary, and `errors`=[] because there was genuinely nothing to report. (Anchor's initial "failure / junk nodes" read was a probe artifact — running `len()` / iteration over a *string* yields a char-count and per-char items; it was never a real failure.) Real issue is only **trace verbosity**: the model emitted `nodes` stringified in one of two redundant remember_batch calls (the dispatch tolerated it), captured verbatim → large trace. Minor hardening if ever revisited: (a) normalize/reject `nodes`-as-string at the remember_batch boundary; (b) cap large stringified inputs stored in `action_details`. Low priority — no integrity impact. **L1** lineage / `cwd`-keying for cross-conversation continuity — PARKED, gated on D1 (only needed if relaunches keep spawning new conversations).
- **2026-05-30** — **Frozen Corpus eval platform SHIPPED + v24/v7/v4 ACTIVATED.** Two-stage longmem harness: `eval/longmem/build_corpus.py` (encode once, content-addressed by s1e/ingest-surface/s2-cadence/oracle/qids + interaction-overrides; incremental manifest) → `sweep.py` (recall cheaply, many times; ~100× under a full run). Caught + fixed a `brain_batch` leaked-transaction crash in the S2 community encoder — interim guard in `dispatch_write.py`, root fix tracked at **F3 / [docs/WRITE-TXN-ISOLATION-ROOTFIX.md]**. Answerability gate stopped hard-excluding composed/multi-session answers (its single-node keyword-AND gold-scan false-negatives them — a 13-node on-topic item scanned "unanswerable") → sweep now **scores every item** with a `recall_conditional` rate + `ABSTENTION_FAIL` bucket. Also fixed harness `_snapshot_error_count` (queried a non-existent `brain_errors` table → now `debug_log`). 20-item baseline (corpus `a300d2`): **94.4% recall-conditional (17/18)**, 85% raw — the 2 clean misses are `ENCODE_MISS` (encoder wrote 0 nodes on a preference + a rejected-suggestion exchange), recall reads strong. Added `build_corpus --interaction-override` (fetches DORMANT versions from the live daemon); 2-item v24+v7 A/B showed facts-scout v7 0→6 candidates + v24 encoding nodes v22 dropped → **activated s1e v24 + s1_scout_facts v7 + s1_scout_quote v4** (`d0fea6d`, `47f7018`; seeds synced, `--check` clean). Commits: `18ac427`, `beb38ff`, `d0fea6d`, `47f7018`, `9243600` (CLAUDE.md). Refs: [docs/EVAL-PLATFORM.md], [docs/WRITE-TXN-ISOLATION-ROOTFIX.md]. Residual encode-coverage gap (encoder filters low-stake exchanges even when they hold a future answer) is a design-philosophy call, deliberately not patched.
- **2026-05-29** — S2 idle-gating + test-suite redundancy cleanup (separate thread from the encoder/episodic-refs work). S2 **Community** + **Consolidation** were doing full O(graph) scans every ~15 min 24/7 (87% / ~88% zero-work, never converging); both now gate on graph-change + skip idle cycles (Consolidation also activates its dead incremental-scan path, stamping the cutoff only after the encoder completes). AspectIntegration + Healer audited clean. Test-redundancy hunt over ~70 files: 13 inert/duplicate tests removed-or-consolidated + 1 stale RED test (`test_same_relation_strengthens`) fixed to the Stage 1B contract. 5 commits (`47ed457`, `ba89bb9`, `e37c8e5`, `694b339`, `5ab2dde`), not pushed; daemon picks up gates on next restart. **Full handoff + remaining test-cleanup map (buckets A–E): [docs/archive/session-handoffs/S2-GATING-AND-TEST-CLEANUP-HANDOFF.md](S2-GATING-AND-TEST-CLEANUP-HANDOFF.md).** Community Phase 2 (delta-decode) parked → memory `project_s2_community_phase2_parked`; dampening cluster (4 red tests) parked with recall.
- **2026-05-26** — Phase B Step 7 (co_anchored auto-edge) + Layer 1 validator + sync-prompts active-version fix SHIPPED. v22 active in production at 2026-05-26T00:46:54Z (commit `c144ddf`). 50-cell longmem real-test result: v22 23/25 (92%) vs v19 22/25 (88%); v22 100% source_refs coverage uniformly; +1 axis win on info_extraction where v19 zero-encoded `cc539528`. `eval/encoder_eval/` infrastructure shipped — 6 substrate-aware probes, parallel + stratified runner, scout-override capability via `apply_interaction_override` pattern. v23/v24 + scout v6/v7 + quote v3/v4 registered DORMANT after two iteration cycles (initial draft → probe-driven refinement). Targeted A/B + follow-up evals (15 cells, 10 distinct items): v24 candidate 9/10 — qualitative wins on abstention discipline (`gpt4_93159ced_abs` v22+v19 ✗ → v24 ✓) + hobby cohort retention (`5025383b` v22-this-run ✗ → v24 ✓), one regression direction on temporal hedging (`gpt4_d31cdae3` v22 ✓ → v24 ✗). Methodological finding: c2ac3c61 wasn't deterministic — v22 succeeded on re-run, confirming N=1 stochasticity floor. Eval-decision call open. 5 commits on main. 165 tests pass.
- **2026-05-25 (mid-day)** — Schema v29 trace_id INTEGER→hex migration + Phase B Steps 0+0.5+1+2+3+4 SHIPPED. Substrate complete; encoder behavior unchanged until v22 prompt drafted. 60,634 trace_events + 12,933 trace_embeddings migrated cleanly via auto-backup framework. MCP schemas, brain.remember/revise, dispatch validation all wired for source_refs. Quality contract v3 (36 dims, Group 9 example_authoring). All 7 §7.6 examples placeholder-compliant. Reviewer pass: F1 critical (revise REPLACE not APPEND) + F2/F4/F8 fixed; F3/F6/F7 deferred. 145 v29-touching tests pass. Next: Step 6 prompt v22 + Step 8 3-way eval (v22 vs v21 vs v19) gate before production flip.
- **2026-05-25 (overnight)** — Encoder quality contract v1 (32 dims) + s1e v20 SHIPPED. Fixed 4 canonical example bugs (Example 1 D31, Example 4 D23, Example 5 D7, revise ghi789 keywords drop) + appended §7.6 wave-1 block with 6 new examples (Anchor self-reference triad A6+A7+A4, correction-with-affect A2, trust formation A3, methodology/principle split B1). Built encoder eval platform at `eval/agent_introspect/encoder_contract_eval.py`. A/B-validated v20 vs v19 on 4 corpuses; D31 fix landed cleanly on structurally-different clean retest (C4). v20 active 2026-05-25 03:12 UTC.
- **2026-05-24 (late)** — Block 1 substrate cleanup SHIPPED (4 commits on top of Phase A). MCP `get_trace`/`get_traces`, `query_traces` session_id + session_ids, killed `encode_cluster` + `auto_connect` (related_to pollution source), Schema v28 dropping `nodes.keywords` + auto-extractor. Plus 327-node encoder-quality scan; see `docs/ENCODER-QUALITY-FINDINGS.md`.
- **2026-05-24 (mid)** — Time-window architecture: `iso_now`/`iso_cutoff` with `at=` parameter (conversation-anchored cutoffs); contract test banning direct `datetime.now()` in S1/S2. 12-site SQL `datetime()` fix.
- **2026-05-24 (early)** — Phase A substrate SHIPPED end-to-end (10 commits). Schema v27, DAL methods, identity stamping, embed worker, historical identity migration.
- 2026-05-23 — eval system v2 + episodic-references design complete; §15.1 closed via biology research; decision 19 revised
- 2026-05-18 — bg_writer migration + recall latency diagnosis (next-session packages A + B BOTH SHIPPED, see commits below)
- 2026-05-15 — surface v6 redesign + full A/B comparison + tooling
  - [docs/AGENT-PROBES.md](AGENT-PROBES.md) — probe family
- 2026-05-11 — temporal arc + agent introspection (ARCHIVED)
  - [docs/TEMPORAL-ARCHITECTURE.md](TEMPORAL-ARCHITECTURE.md) — full temporal handling reference

---

## ⚡ Top of queue — what's next

### 🔴 Migration runner rebuild (attempt 2) — unowned, and three cleanups queue behind it

Attempt 1 (`dfc74ee` + `5cead71`) was **reverted** (`813b7c2`, `58581ff`): three
independent reviewers found `MAIN_MIGRATIONS` dead by construction — `ensure_schema`
stamped `BRAIN_VERSION` at step 7, then the runner re-read that key at step 8, saw
itself already current, and ran nothing. Latent only because the list was empty; it
detonates on the first real migration. **Fix: the runner owns the stamp.** The logs-side
integration was correct and is the model. Full mechanism + repro: brain `b5b72b74`.
`MAIN_MIGRATIONS` appears in zero tests — the layer's only exercise is with an empty
list, which is exactly what hides the bug.

What's live right now is the older `_backfill_data(conn, current_version)` ladder, which
works because it takes the pre-stamp version as an argument.

**Queued behind it** (each wants a version bump, and one bump should serve all):

| Item | Detail |
|---|---|
| Dead-table drops | 12 undeclared in `brain.db`, 15 in `brain_logs.db` — full inventory + row counts in brain `2b49ac02`. `brain_logs.db` has **no** migration mechanism at all after the revert, so its half is blocked, not merely deferred. "Undeclared" is not "drop-safe": ~12 names still return code hits that need disambiguating from a real reader first. |
| `bridge_proposals` | Undeclared from `TABLES`/`INDEXES` 2026-08-11 (fresh brains skip it); 0 rows, empty table still present on existing brains. Joins the bulk drop. |
| 648 stale `keywords` KV rows | Deferred 2026-08-07 with the `contract.py` skip_keys guard held in place; purge and guard-release ship together. |

**Why it's not free to bump:** `_backup_before_migration` copies the whole DB — 675 MB
today — on every DB below the new version, at boot, before the port opens. Per the fleet
rule (brain `56890464`) a migration must be sub-second or the watchdog can `kickstart -k`
mid-run, since a closed port reads as *dead*, not busy. And the fleet is larger than
production installs: 606 of 657 `brain.db` files on this machine are below v30, counting
backups, eval corpora and clones (brain `50d41c27`).

### 🔴 Generic-edge pollution: 18.1% of the live graph carries no relation signal (found 2026-08-07)

```
39,975 live edge_relations
 7,243 'related' (2,527) + 'related_to' (4,716)   →  18.1%
```

**Why it matters.** Edge weight is dead (static `0.5`); edge relevance is cosine
against `edge_relations.embedding`. The `brain_batch` schema already forbids
these verbs at the write door — *"generic relations pollute the activation
kernel and match no query"* — so the system blocks new ones at the front while
nearly one in five existing edges contributes nothing to activation. This is a
recall-quality problem wearing a data-hygiene costume.

**The tool already exists and never finished the job.**
`servers/scales/s2/archive/reclassify.py` (`RelationReclassifier`) is a working
`IntegrationUnit`: reads generic edges that *have descriptions*, batches to
Sonnet, assigns a specific verb. Built for the v22 "related_to crisis"
(id:524670fe), archived mid-migration. **Do not delete it** — it is the fix.
(It no longer ships in the plugin as of 2026-08-07; in-repo, out-of-package.)

**OPEN QUESTION — legacy or leak?** Tom's read: *"sounds to me like there is a
leak from somewhere."* The write door forbids these verbs, so either all 7,243
predate the ban, or something is still minting them. **Distinguishable by
dating the edges** — histogram `edge_relations.created_at` for the generic set
against the date the ban landed. If any are post-ban, find the writer first;
reclassifying while a leak runs just refills the pool.

**Sequence:** (1) date the edges → legacy vs leak; (2) if leak, plug the writer;
(3) run the reclassifier over the backlog (Sonnet spend — batched, chunked);
(4) re-measure. Note ~how many of the 7,243 lack descriptions — the
reclassifier needs one to work from, so that subset needs a different answer.

**Two next-session specs ready (2026-06-04):**
- [**docs/CROSS-STREAM-ON-SCALES.md**](CROSS-STREAM-ON-SCALES.md) — make a delivered cross-stream message a first-class S0 turn so S1R surfaces against the *body* (not the `<task-notification>` envelope) and S1E encodes the exchange. Subsumes the A2 watch-wake-skip piece. Folds the 2026-06-04 leftovers (A1+A3 Haiku bundle pointer, F2 read-lag part b, B4 namespace bridge, CR4, CR5). 5 open decisions (OD1–OD5) resolvable in ~5 min at session start.
- [**docs/SELF-RECOGNITION.md**](SELF-RECOGNITION.md) — cheap Stop-recall keyed on my own output, stashed for the next prompt. Subsumes the A2 stop-breadcrumb piece. **Recognition is anchoring** — the echo-chamber risk is the value, gated by stance discipline ("priors, not instructions"). 6 open decisions (RD1–RD6); SKILL.md stance section is the safety lever.

**Phase B substrate SHIPPED end-to-end. v22 active in production since 2026-05-26T00:46:54Z.** Step 7 (co_anchored auto-edge), Layer 1 validator (hex format soft warn + sparsity >5), and sync-prompts active-version fix all shipped. **s1e v24 + scout-facts v7 + scout-quote v4 ACTIVATED in production 2026-05-30** (commits `d0fea6d`, `47f7018`); v23 + scout v6 + quote v3 remain dormant.

**Next-session priority order**:

1. **Render expansion at SURFACE_FORMAT** (~0.5-1 day, pending). The recall-side joint reactivation read shape — when a source-anchored node surfaces, expand its source_refs inline. Designed in `docs/EPISODIC-REFERENCES.md §8`; not built. Depends on enough v22-encoded nodes in production to measure the surface impact (start accumulating now). **[Verified 2026-05-30: the code path is entirely unbuilt — zero `source_ref` handling in `surface_contract.py` render code. This is ground-up implementation, NOT "almost done, waiting for data." The v24 accumulation only gates *measuring* impact after the code exists.]**

2. **source_summary parallel-pathway recall scoring** (~0.5 day, pending). `docs/EPISODIC-REFERENCES.md §9.5` + decision 22. Add `source_summary` cohort to recall scoring as `max(legacy_weighted_sum, source_summary_score)`. Backwards compat by design.

3. **S2Healer source_refs cleanup** (`docs/EPISODIC-REFERENCES.md §10.6`, pending). Periodic scan for invalid trace_ids; archive orphan `co_anchored` edges when no shared trace remains.

4. **Path A ground-truth authoring** (~1.75h Tom-time, pending). 7 conversation templates scaffolded at `eval/ground_truth/` covering 5 strata (2 identity-bearing + 2 partnership voice + 1 technical correction + 1 methodology + 1 temporal). Each file has fillable YAML for ideal-node authoring. Once filled: targeted eval against ground truth (structural delta) joins the longmem oracle path (recall delta).

5. **Phase B+ quote_fidelity substring validation** (~0.5 day, deferred) — bigger fix than the identical-strings check that shipped commit `d3f6307`. Requires threading conversation context to `brain_remember.remember()` so `user_raw_quote` can be substring-matched against user_messages window and `anchor_raw_quote` against agent_messages window.

6. **S2Healer stale-node extension** (~30min) — aspect-resolved detection for status-as-fact / plan-as-executed nodes.

### Reviewer follow-ups (deferred from 2026-05-25 Phase B review)

| ID | What | Effort | Status |
|---|---|---|---|
| **F7** | Move `_SOURCE_REFS_SCHEMA` from `brain_mcp.py` to `contract.py` under new `JOIN_TABLE_FIELDS` category (parallel to STRUCTURAL / PROMOTED). Makes contract-sync test implicitly cover field registration. | ~45 min | open |
| **F9** | Shared module-level review-block/closure inject helpers (Piece 4, 2026-06-30). Resolved by the journal component (2026-07-28): `scales/journal.py` `JournalBinding.decorate_system` is the one assembly for S1E and the S2 units; the base inject wrappers were deleted. | — | done |
| **F8** | Unify Anchor's per-action S0 traces under the per-turn `anchor_touched` aggregate (Piece 3a, 2026-06-29). Today `revise`/`connect` emit their own `node_revised`/`edge_relation_revised` rows on the date chain (`s0-{YYYYMMDD}-revise`) while `remember`/`get_node` emit nothing — an accident of history. Piece 3a adds a per-turn `anchor_touched` delta (stop chain) capturing all of created/revised/connected/recalled, so the same node id now lives in *two* rows (legacy per-action + new per-turn aggregate). Left as-is to avoid scope creep, but it's ugly now that the per-turn aggregate is the better convention: consider migrating the legacy per-action emissions onto the aggregate (or dropping them if the dashboard/audit consumers can read the aggregate instead). Verify no consumer depends on the date-chain granularity first. | ~1–2h | open |
| **F10** | Enable Strict Tool Use on the Sonnet call sites (S2 reclassify, S2 base, S1 Scribe encoder). Unblocked: `brain_batch`'s nested schema is already a `oneOf` discriminated union per op-type (derived from `contract.BATCH_OP_SPECS`). Before flipping, verify Anthropic's current strict-mode JSON Schema keyword subset (`oneOf` vs `anyOf`, `const`) against live docs. | ~0.5 day | open |
| **F11** | **Legacy-journal dead-code sweep (map made 2026-07-03, post-v29-activation).** Three tiers by deadness. **(a) Inert NOW in prod — cut when convenient:** the Frame `## Recent moves` slot (`frame.py:161-166` `_render_recent_moves` + `:189` wire + `:12` doc) reads the `encoding_journal_{sid}` blob that v29 never writes → renders `(fresh session)` every turn forever; its only reader `brain.get_recent_encoding_journal` (`brain.py:699-716`); and the blob *writer* `encode.py:1132-1167` `_save_journal` (called only on the control arm, `encode.py:255`). This is the "remove Recent-moves at activation" cut the design called for and the activation skipped — the one worth doing now (kills an always-empty boot-context section). **(b) Flag-off / v25-rollback path — retire only when `BRAIN_S1E_LIVED_SEQUENCE` itself is retired (keep as rollback + ab_encode control for now):** `_save_session_context` (`encode.py:1170-1195`, incl. the truncation dup'd into `write_session_arc` — the deferred dedup); the control-arm branches in `_build_user_content` (`~740-749`, `~769-777`, `:718-720` `### Encoding Journal`/`### Session Context`); `journal_max_chars`/`journal_entry_limit` in `encode_contract.py:91,98`. **(c) Vestigial text:** `healer_prompt.py:198-200` `SKIPPED:`/`WATCHING:` journal instructions in the journal-EXEMPT Healer (verify active `s2_healer` v3 before cutting); stale `'encoding_journal'` label in `scribe.py:42` `K_SOURCES`. NOTE: base.py S2 journal legacy already retired (`6cb8047`, node `bfdd5330` resolved) — not in scope. | (a) ~30 min · (b) with flag-retire · (c) ~15 min | open |

### v24 experimentation thread (active)

Surfaced from forensic analysis of c2ac3c61 (multi_session precision-refinement) in 50-cell longmem run:

| Thread | Status |
|---|---|
| s1e v24 DORMANT — multi-ref anchoring sentence in §7.4 | registered, awaiting activation decision |
| s1_scout_facts v7 DORMANT — supersession-scope clarifier + Cap ranking refinement + Example 4 tightening + NEW Example 5 (parallel-entity + same-axis refinement, languages-domain) | registered, awaiting activation decision |
| s1_scout_quote v4 DORMANT — Skip-list addition for routine factual claims (facts-scout territory boundary) + mixed-content handling | registered, awaiting activation decision |
| Eval-decision call (option B: re-run gpt4_d31cdae3 ×3) | pending |
| Diff files preserved at `/tmp/{s1e_v24,s1_scout_facts_v7,s1_scout_quote_v4}_proposed.txt` and `/tmp/{*}_pre.txt` | gitignored — for inspection |

**Key methodological finding from this session**: **LLM encoders are stochastic at N=1.** c2ac3c61 failed v22 in the 50-cell run but succeeded on re-run with the same prompt. 5025383b succeeded in the 50-cell run but failed v22 on re-run. Future eval design should account for this — multi-sample per cell (N≥3) is needed to distinguish deterministic failure modes from tail outcomes.

Detailed scope per thread: see SESSION-HANDOFF.md.

---

## Open questions surfaced 2026-05-25 (post-v20 ship)

These need judgment calls before deeper work. Captured here so they don't get lost between sessions.

### Q1 — B1's generalization range
The clean C4 retest showed B1's mechanism-vs-principle teaching landed on a structurally different corpus, but N=1 corpus for "different shape." Does B1 land across the actual range of methodology-shaped conversations Tom and Anchor encounter, or only the two corpus shapes we've tested? **No way to know without production exposure.** Schedule: rerun A/B against new corpus types post-v20 (~1 week).

### Q2 — Phase B prompt rule vs example reliance
When source_refs writes ship, do we teach via:
- (a) NEW canonical examples demonstrating source_refs alongside the §7.6 wave-1 that already have them
- (b) An explicit prompt rule paragraph (the §7.4 "Anchoring nodes in the substrate" prose from EPISODIC-REFERENCES.md)
- (c) Both

Evaluator findings on Era E (D7 voice symmetry, 0/6 on identity-bearing nodes despite the rule being in v16+) suggest **examples alone are insufficient for D5/D7-level discipline.** Suggests prompt rules earn their place. Going with (c).

### Q3 — §7.6 vs canonical hierarchy in Sonnet attention
v20.1 §7.6 examples include source_refs; canonical body doesn't. A/B outputs showed Sonnet didn't write source_refs (correctly — schema doesn't accept). But this signals canonical pattern dominates §7.6 attention-wise. Should we **move §7.6 ABOVE canonical** in the prompt structure? Or keep current order? Currently §7.6 lives after canonical (line 983 vs 869). Architectural choice — defer until Phase B forces a prompt restructure anyway.

### Q4 — D22 axiom-layer carveout
Identity-bearing axioms (A7, A4 shape) naturally produce monochromatic edges (all in identity_bearing-adjacent aspects). The contract's D22 marks this as "degraded" but the encoder evaluator surfaced it as a contract gap — the rule needs an explicit carveout, OR D22 needs a new CR12 ("axiom-layer monochromatism is by design"). Defer to contract v2 alongside wave 2 examples.

### Q5 — §7.6 wave 2 priority vs Phase B priority
Both are next-session candidates. Phase B unlocks substrate value (source_refs become functional); wave 2 fills contract dimension gaps (D11 revise audit, D24 multi-aspect-pair). **Instinct: Phase B first** — substrate value compounds, wave 2 examples can use source_refs once they're functional.

### Q6 — Speaker misattribution rate in production
A/B sample showed 1/3 corpuses had the C2 shape (33%). Real production rate unknown until v20 runs for ~1 week and we measure (via the new `voice_fidelity_identical_strings` error log entries). Could be much lower — C2 had a specific operator-asks-question + anchor-articulates-principle shape that triggered it. **Schedule measurement: query brain_logs.db.brain_errors for `voice_fidelity_identical_strings` after 1 week of v20.**

### Q7 — Source_refs silent-drop pragmatics
§7.6 examples have `source_refs: [...]` scaffolding. When Sonnet writes a node matching the pattern in production today, the source_refs get silently dropped (schema doesn't accept). Should the dispatcher add a runtime warning log entry "source_refs received but write path not active until Phase B," OR is silent-drop fine until Phase B ships? **Pure pragmatics call.** Silent-drop avoids alarm-fatigue on every encoding cycle until Phase B; warning would surface premature scaffolding noise. Going silent until Phase B unless production behavior changes our read.

---

## Contract refinements identified by evaluator (10 items, defer to v2)

After 1-2 weeks of v20 production exposure, refine the contract:

1. **D22 axiom-layer carveout** — explicit rule for identity_bearing monochromatism
2. **D1 title length cap clarification** — pick 60c vs 80c, operationalize
3. **CR4 extension to novel RELATIONS** — currently covers types only; extend to verbs
4. **D24 multi-aspect verb list** — operationalize the `note` field as scoring discipline
5. **D25 vs D26 trade-off explicit rule** — when verbatim quote field is set, must its source turn be in source_refs?
6. **D31 lock-worthy N=1 criterion** — operator-named recurrence ("every time", "always") counts as multi-instance evidence
7. **D23 correction_improvement aspect strictness** — type=correction implies at least one correction_improvement aspect edge
8. **D18 conversation-time vs wall-clock check** — explicit format requirement for event_time
9. **D7 scope clarification** — technical pattern-naming reframes also earn anchor_raw_quote
10. **D8 type-aspect coherence with reasoning** — flag type-in-aspect-X-while-reasoning-invokes-aspect-Y

---

## Episodic-references execution status (re-sliced 2026-05-25)

**Phase A — substrate** ✅ **SHIPPED 2026-05-24 (early)** (10 commits). Schema v27, DAL methods, identity stamping, embed worker, historical identity migration.

**Block 1 — substrate cleanup** ✅ **SHIPPED 2026-05-24 (late)** (4 commits, see commits `24e83bc`, `c015d1b`, `fea0fef`, `8d41c8c`).

**Encoder quality contract + canonical fixes + §7.6 wave-1** ✅ **SHIPPED 2026-05-25 (overnight)** — s1e v20 registered + synced. Contract at `servers/scales/s1/quality_contract.py`. Examples at `servers/scales/s1/examples/`. Eval platform at `eval/agent_introspect/encoder_contract_eval.py`.

**Schema v29 trace_id hex migration** ✅ **SHIPPED 2026-05-25 (mid-day)** — `trace_events.id`, `trace_embeddings.trace_id`, `node_source_refs.trace_id` migrated INTEGER → TEXT (8-char hex). Auto-backup `brain.db.v28.bak` exists. DAL coercion removed (reject ints loudly per reviewer F2). MCP `get_trace`/`get_traces` string-typed.

**Quality contract v3 (Group 9 example_authoring)** ✅ **SHIPPED 2026-05-25** — 36 dims, 12 CR. D33 placeholder_syntax + D34 ref_internal_consistency + D35 voice_annotation_coverage (mechanical) + D36 turn_node_divergence (LLM-judged). All 7 §7.6 examples placeholder-compliant.

**Encoder source_refs write path** ✅ **SHIPPED 2026-05-25 (mid-day)** — MCP schemas declare `source_refs: array[string]` on remember/remember_batch/revise_batch (brain_batch inherits). `brain.remember()` accepts kwarg + persists via `GraphDAL.add_source_refs`. `brain.revise()` accepts kwarg + persists via `GraphDAL.replace_source_refs` (REPLACE semantics per unified 2-class revise contract id:`995ffeb1`). Dispatch validates list-of-strings + sparseness warns via `brain._log_warning`. `tests/test_remember_source_refs.py` covers the integration path. **Encoder prompt teaching (Step 6) still pending.**

**Render expansion at SURFACE_FORMAT** — pending (depends on Phase B substrate, which is now done).

**§7.6 wave 2 + 3** — pending. Wave 2 = shape diversity (~4-5 examples including brain_batch mix). Wave 3 = domain breadth (math/poetry/psychology/research, parallel-agent dispatched).

**Recall + eval block** — Phase B onward. `source_summary` parallel-pathway scoring, `co_anchored` writes at encode + S2Healer cleanup, weight-tuning eval (§13.6), quality_probe + source_fidelity_probe runs. Now unblocked by the MCP trace API.

---

## Identity-architecture gap-to-function items (added 2026-05-24)

From the competitive-landscape research (see `docs/IDENTITY-RESEARCH-2026-05-24.md` for the full write-up). The substrate exists; these are concrete missing functions on top of it. Listed by effort, not priority — each is independent.

| Gap | Function we can't perform | Effort |
|---|---|---|
| **Identity-eval scaffolding** | Drift detection across sessions, non-contradiction check, persistence-through-change measurement, post-model-upgrade identity verification | ~1-2 days — borrow Agent Identity Evals (arxiv:2507.17257) shape: ~30 self-reference questions, run pre/post change, judge for stylistic consistency + semantic similarity |
| **Partner-minting flow** | "Hi I'm Alice" → new-partner recognition + binding; multi-partner-over-time arc | ~0.5 day — minting writes a `partner` node, refreshes `brain.operator_name`, identity stamp picks up the new value at next trace write |
| **Identity-filter query** | "Show me everything Tom said about X"; speaker-filter in recall | ~0.5 day — extend `query_traces` with `human_identity` param; promote metadata field to indexed column if perf demands |
| **Self-narrative generation** | "Tell me about yourself" → coherent autobiography across sessions; onboarding pitch for new partner | ~1 day — Sonnet over curated subset (locked identity nodes + recent partnership traces); render-time, not encode-time |
| **Damage resilience** | Survive partial brain corruption; redundancy-backed identity (quorum of env + locked partner node + recent traces) | ~1 day — formalize identity-source quorum, detect mismatch loudly via the existing `_maybe_warn_identity_unset` write-boundary signal |
| **Plug-and-play install** | New user can install without dev setup | Larger arc — `pyproject.toml`, multi-platform daemon adapter, deferred until there's a real second user |
| **Multi-tenant SaaS** | Multiple operators on one daemon with isolation | Larger arc — auth + isolation + architectural shift; only when warranted by an actual second-user case |

Plus measurement scaffolds unique to our substrate (require no new architecture, just eval code):

- **Identity-neighborhood stability** — centroid of all "Tom said..." trace embeddings tracked over months; centroid drift signals identity-token decay
- **Source attribution accuracy** — after render-expansion ships: pick a recalled node, ask "where did this come from?", judge against encoded source_refs (cryptomnesia analog)
- **Engram cohort recall** — pick a trace, find all co_anchored nodes, run a recall query semantically near that trace, measure how many surface together
- **Bidirectional partnership impact** — the target function. Hardest to measure, most important. A/B against pre-Anchor task baselines.

Full measurement detail: `docs/IDENTITY-RESEARCH-2026-05-24.md` Part 4.

---

## Borrowing items from competitor library deep-dive (added 2026-05-24, Part 7)

Three concrete techniques from other systems that translate cleanly into Anchor's architecture. Each is small (~hours, not days). Listed by leverage.

| Borrow | Source | What it does | Effort | Leverage |
|---|---|---|---|---|
| **A.U.D.N. dedup decision** (ADD / UPDATE / DELETE / NOOP as LLM tool calls, gated by InformationContent) | Mem0 (arxiv:2504.19413) | Replaces ad-hoc consolidation prompts with a principled four-tool LLM decision. Already aligned with our `similar_to`/`supersedes` aspect edges. | ~0.5 day in S2 Consolidation encoder | High — turns a heuristic into a structured judgment; works without architectural change |
| **Node specificity weighting** (`s_i = |P_i|^-1` as IDF surrogate for spread-activation seeds) | HippoRAG (arxiv:2405.14831) | Rare nodes get more probability mass; high-degree hub nodes don't dominate. Addresses the "93% never recalled, hubs dominate" finding (node 0591813f). | ~0.5 day in spread-activation kernel | High — directly addresses an known recall problem we already measured |
| **Nucleus expansion** (after semantic nucleus match, expand ±N adjacent trace events from same session by graph adjacency) | MemMachine (arxiv:2604.04853) | Recovers context that spans turn boundaries when only one turn is embedding-similar. Natural fit for source_refs render expansion. | ~0.5 day in render path / surface_contract | Medium-high — strengthens joint reactivation (§8) without changing the encode side |

**Plus revisitable ideas (not immediate, worth tracking)**:

- **A-MEM's K/G/X mutation on link** — when a new node links to an existing one, update the existing node's structured fields. We have Healer for missing-field fill; don't mutate filled fields. The §16.1 labile-reconsolidation direction matches this.
- **Generative Agents' explicit `importance` field (LLM-rated 1-10)** — we approximate via aspect classification + locking; an explicit numeric field could complement.
- **Letta's `sleep-time agent`** — first-class concept for the same role our S2 maintenance + S1 Scribe play. Their naming is cleaner; our split across S1 Scribe and S2 Coordinator is functionally similar.
- **Letta's shared memory blocks across agents** — directly applicable when we get a second partner or a second agent personality (sub-agent flow).
- **MIRIX's procedural/resource memory split** — we don't have procedural memory (§16.5 future direction). When we do, the split into "how to do things" vs "what was looked at" is worth borrowing.

Full technical detail per library: `docs/IDENTITY-RESEARCH-2026-05-24.md` Part 7.

---

## Open questions to address over time (from 2026-05-24 research close-out)

Eight questions that emerged from the synthesis — not blocking next session's work, but worth keeping visible until each has a measured answer. Full context in `docs/IDENTITY-RESEARCH-2026-05-24.md` Part 8.

| Q | Question | Why it matters | Rough effort |
|---|---|---|---|
| **Q1** | What's the actual extraction-loss bound? (Node-alone vs trace-alone vs node + nucleus-expanded) | Tests whether our extraction earns its keep against MemMachine's verbatim-keeping (93.0% LongMemEvalS). The architectural gut check. | ~1 day for a 20-node probe |
| **Q2** | How often does source attribution actually fail? (Cryptomnesia rate) | Johnson-Hashtroudi-Lindsay's source-monitoring is uncited anywhere in LLM memory research; we have the substrate to measure it. | ~0.5 day, 50-node probe |
| **Q3** | Does identity survive a model upgrade? (Claude 4.7 → 4.8) | The killer test for the concrete-token biology-grounding claim (decision 19). | ~0.5 day to baseline, re-run on next model swap |
| **Q4** | Do identity-load-bearing nodes survive S2 consolidation? | Aspect taxonomy says they should be protected; untested. | ~0.5 day probe + S2 archive log walk |
| **Q5** | Is the spreading-activation kernel earning its complexity vs PPR + node specificity? | If PPR ties or beats, retire the kernel complexity. Decisive A/B. | ~1 day |
| **Q6** | What's the right labile-reconsolidation design? (§16.1 named but unspec'd) | Recall opens an update window in biology; our graph drifts toward original framings without this. | Design conversation; then ~1-2 day build |
| **Q7** | Does aspect-taxonomy dual-role tension cause real bugs? (Structural routing vs semantic classification) | Determines whether to split aspects into two taxonomies. | Surface-via-observation, not a probe |
| **Q8** | When the first non-Tom partner appears, what breaks? | Per-utterance binding + render reconstructive frame are designed for this; never exercised. | Synthetic Alice-session test, ~0.5 day |

**The discipline lesson** (also captured as a memory): when a substrate change ships specifically to enable a measurement, the measurement gets a task on the next session's plan automatically. Don't let substrate sit un-measured. The architecture isn't real until the eval runs.

---

## ✅ Completed since 2026-05-18

| What | Commit | When |
|---|---|---|
| Package A — Signal producer cleanup (collapse signals phase, delete reminders/encoding_gap/hook_errors/etc.) | `02f5c32` | 2026-05-19 |
| Package B — `db_maintenance.py` module (5-min checkpoint, 30-min optimize, integrity at boot, pragmas helper) | `9ce056b` | 2026-05-19 |
| Daemon: host-suspend detection (macOS sleep) + Anthropic timeout caps | `47c5907` | 2026-05-19 |
| Daemon health: BRAIN_DEV_MODE opt-out + 20s ping threshold | `6a4fe6e` | 2026-05-19 |
| brain_batch: single transaction per batch, rollback on failure | `a0434c2` | 2026-05-19 |
| bg_writer: empty-drain ticks refresh last_drain_at — kill false stalls | `0a91b43` | 2026-05-19 |
| CLAUDE.md: write topology section refresh | `b10260b` | 2026-05-19 |
| Runner: per-round diagnostic stats (ttft + per-round tokens/cache) | `a28fbc3` | 2026-05-23 |
| Eval system v2: judge reasoning + comparison enum + variance + 3 agent_introspect probes | `0b1115e` | 2026-05-23 |
| **Schema v27** — `node_source_refs` + `trace_embeddings` tables, composite index | `9015636` | 2026-05-24 |
| **DAL methods** — TraceDAL embeddings + GraphDAL source_refs primitives | `8a52164` | 2026-05-24 |
| **Identity stamping** wired at trace write (TraceDAL.set_identity + _stamp_identity) | `75075eb` | 2026-05-24 |
| **Identity activated end-to-end** — brain-env sources user config; dispatch decodes JSON-string metadata (also fixes pre-existing double-encode bug) | `65bf483` | 2026-05-24 |
| **Embed worker** — pull-reconciliation trace embedding phase, concrete-identity render | `7b5b845` | 2026-05-24 |
| **Worker review fixes** — 30-day window (architectural correctness), composite index, observability, defensive double-decode | `669ecee` | 2026-05-24 |
| **Migration**: backfill identity on all 57,546 historical trace_events; clean 22,416 double-encoded legacy rows | `5cff407` | 2026-05-24 |
| **Loud-by-default** — embedder-not-ready / vector-count-mismatch silent paths surfaced | `4288ec8` | 2026-05-24 |
| **Identity-unset signal at write boundary** — TraceDAL._maybe_warn_identity_unset (replaces boot-only check) | `987587f` | 2026-05-24 |
| **Substrate cleanups** — _decode_metadata rolled out to all TraceDAL readers, point-lookup API (`brain.get_trace`/`get_traces`), dead test removed | `d68bddc` | 2026-05-24 |
| **SQL datetime() trip-hazard** — 12 broken time-window queries fixed via `iso_cutoff` helper | `255b9de` | 2026-05-24 |
| **Time-window architecture** — `iso_now`/`iso_cutoff` with `at=` for conversation anchoring; contract test bans direct `datetime.now()` in S1/S2 | `3dd37d4` | 2026-05-24 |
| **query_traces session_id fix** — singular session_id authoritative, ignores hours, loud on empty | `24e83bc` | 2026-05-24 |
| **MCP get_trace + get_traces + auto_connect kill** — wired MCP tools (was decision 23, unstarted); deleted encode_cluster (dead code, 121 lines, 0 callers); removed auto_connect from remember_batch (source of related_to pollution) | `c015d1b` | 2026-05-24 |
| **query_traces cross-session** — `session_ids: List[str]` plural authoritative, mutually exclusive with singular | `fea0fef` | 2026-05-24 |
| **Schema v28 — keywords kill** — drop `nodes.keywords` column + rebuild `nodes_fts` without keywords + delete `_extract_keywords`/`enrich_keywords`. Auto-extractor produced near-duplicate tokenizer noise; porter stemming on title+content is cleaner | `8d41c8c` | 2026-05-24 |
| **Encoder prompt s1e v19** — example cleanup (remove `auto_connect: true` from canonical remember_batch example; no functional change) | (DB-only registration) | 2026-05-24 |
| **S2 community prompt v17** — same example cleanup + remove "`auto_connect: false` always" rule | (DB-only registration) | 2026-05-24 |
| **absorb op primitive** — lossless merge (folds one node into another transfer-by-default, then archives the absorbed). See `docs/S2-ABSORB-OP-DESIGN.md`. Consolidation wiring shipped 2026-06-04 (see below). | `d3a0fa1` | 2026-05-28 |
| **S2 consolidation v6** — stopped locked-node churn | `714ee68` | 2026-05-28 |
| **F3 root fix — write-path transaction discipline** — `BatchAwareConnection.in_batch` + `commit_unless_batched()` gate, `commit` kwarg removed from 6 GraphDAL writers, `brain._batch_mode` deleted. See `docs/WRITE-TXN-ISOLATION-ROOTFIX.md`. | (dal-cleanup-2) | 2026-05-30 |
| **s1e v24 + s1_scout_facts v7 + s1_scout_quote v4 activated** in production | `d0fea6d`, `47f7018` | 2026-05-30 |
| **Frozen Corpus eval platform** — two-stage harness (`build_corpus.py` → `sweep.py`), content-addressed, `--interaction-override` for DORMANT-version A/B | `18ac427`, `beb38ff`, `9243600` | 2026-05-30 |
| **S2 consolidation emits `absorb`** — prompt v7 + decoder lever A (`_pre_classify` cross-type → `needs_judgment`); merge-recall fix, eval-tested (correct 10→15, under-merge 8→3). Completes the absorb-primitive wiring. | `abe98df`, `e2fb44f` | 2026-06-04 |
| **Daemon Errno-48 boot-race fix (D1)** — all (re)starts route through launchd (`_launchd_kickstart`), serialized under flock; single-owner lifecycle | `78693a3` | 2026-06-04 |
| **Self-channel streams-experience (A4/B1/B2/C1)** — self-id at boot, watchers count as present, Stop-only delivery; resolved CR2 | `364269f` | 2026-06-04 |

Phase A of episodic references is fully shipped: substrate live, identity stamped on every historical trace, embed worker auto-populating.
Block 1 substrate cleanup also shipped: MCP trace API live, related_to pollution source closed, keywords column retired, 327-node encoder-quality scan documented in `docs/ENCODER-QUALITY-FINDINGS.md`.

---

## ⚡ Block 1 follow-ups (deferred from 2026-05-24 late)

### Taxonomy lockdown → v19 rubric
8 emerging-quality clusters + 60+ named qualities ready in `docs/ENCODER-QUALITY-FINDINGS.md`. Collaborative session with Tom to prune/merge/rename, lock the rubric axes, decide preserve-vs-fix on each. ~1-2h. **Gates v19 examples authoring (§7.6 in EPISODIC-REFERENCES.md).** This is the highest-leverage next-session item.

### Conversation-time backdating — consumer wiring + recall feature
The helper exists (`servers/clock.py:conversation_now(at=...)`); the consumers (scouts, encoder, recall) still call `iso_now()`/`iso_cutoff()` without `at=`. Plus a new `recall(query, as_of=...)` parameter for eval/replay backdating. Strategic for evals. ~3-4h once scoped. Tom called this "very important for Evals" 2026-05-24.

### Wider quote-fidelity audit (200 nodes)
Scale the 50-node probe (`/tmp/encoder-scan/probe_quote_fidelity.py`) to 200 nodes. Hand-classify suspicious cases into {paraphrase / cross-session / fabricated / pre-trace}. Output: confident drift rate informing encode-time validation rule. ~1h.

### Reclassify scheduling check
`servers/scales/s2/reclassify.py` exists for legacy `related/related_to`-with-descriptions cleanup. Verify it's wired in the S2 coordinator's unit list and run once against the corpus. ~30min.

### Empty-description `related/related_to` archive sweep
Pre-2026-05-24 auto_connect accumulated empty-description `related_to` edges. Reclassify can't fix these (no description to read). One-off archive script — `archive_dangling_edges` pattern. ~30min after Reclassify verified.

### keywords API surface cleanup ✅ DONE (2026-06-06)
After schema v28 dropped the column, the MCP/CLI/seed_pack surfaces still advertised a `keywords` parameter (silently ignored by remember()). **Fully removed** — keywords no longer appears on any node-writing surface.
> **2026-06-06:** the trigger was a latent bug — `keywords` was still in
> `revise()`'s `NODES_TABLE_FIELDS`, so `revise(keywords=...)` (advertised in
> revise's own docstring) **crashed** on a `SELECT`/`UPDATE` of the dropped column.
> The full purge, done after Tom flagged that a dead param has no business on the
> pillar (and definitely not kept alive to satisfy a stale test):
> - `remember()` signature — `keywords` param removed.
> - MCP — dead `elif name == "keywords"` branch removed from `_generate_remember_schema`;
>   `keywords` removed from the revise_batch schema. Verified: no node-writing tool
>   (remember / remember_batch / brain_batch / revise / revise_batch) advertises it.
> - CLI — `--keywords` flag removed from the `remember` subcommand (enrich's stays — live).
> - `revise()` — `keywords` now a loud deprecated no-op (`DEPRECATED_FIELDS`), not a crash.
> - Tests — stripped `keywords=` from ~28 `test_core.py` calls + the `test_s2_community`
>   remember call; updated the stale `test_remember_has_required_params` (it pinned
>   `keywords` to the signature — the exact stale test that shouldn't dictate dead API).
> - `seed_pack` — removed 16 dead `"keywords"` node-dict entries + the node-craft seed
>   content that taught a `keywords` field.
>
> **NOT removed (correctly):** the `store_enrichments`/enrich/healer **keywords
> enrichment vector** (Q/A/B/**K**) — a separate, live recall path.
>
> **Follow-ups surfaced (separate, out of scope):** (1) the s1e **encoder prompt**
> still shows `keywords:` in examples — cosmetic now (the tool schema rejects it, so
> the encoder can't emit it), but DB-authoritative + eval-gated to fix cleanly.
> (2) `brain_recall.py` keyword-recall still reads the now-always-empty
> `node['keywords']` — dead-ish scoring branch on the recall hot path; its own
> investigation.

### brain_dashboard.db write removal
The daemon-down INSERT into hook_log uses `datetime('now')` — marked `# sql-datetime-ok` as mid-deprecation. Per existing brain memory: `brain_dashboard.db deprecation: stop writing from log_hook_output()`. Full removal when the dashboard's deprecation actually lands.

### Contract-test line-pin cleanup ✅ DONE (verified in code 2026-06-06)
`tests/test_time_window_contract.py` already standardizes on the inline
`# sql-datetime-ok` marker (`SQL_EXEMPT_MARKER`); there is no `BRAIN_MCP_EXPECTED`
line-pin dict in the file. Already resolved — struck.

### 29 pre-existing test failures from session_context signature drift
From May 2 commit `1cdb2b8` (Frame Phase 2.5 — session_context leak fix changed `_save_session_context` signature; scout_muster/trace_system/s1_data_assembly tests still use stale signatures). Not architecture work — pure test maintenance. Address as separate cleanup pass.

### `get_node_lineage(node_id)` — proposed encoder read API
Agent 3 from the quality scan named this wish: single call returning `{creation_chain, revision_chains, related_traces}` for a node. Hold as a candidate when designing the encoder's read surface (alongside `get_traces`/`get_trace` which shipped).

---

## ⚡ Still-open items (older sessions)

### Future — `surface_haiku` 7.5s warm floor
Single Anthropic API call is the architectural floor on hook_recall latency. Options to investigate (no commitment): intent classifier to skip surface for simple queries; async surface (background Haiku while rendering recall); smaller/local model. Larger arc, separate session.

> **2026-06-04 diagnosis (feed the Haiku turn analysis — A1+A3 bundle):** the premise above is stale. It is NOT a single call — `v5_agentic` runs a **2-round** loop (`surface.py:_call_surface_agentic`, `max_rounds=2`) with 6 fetch tools and a brain-recall tool-exec between rounds; steady-state `surface_haiku` measured **8–10s**. And there is **NO `cache_control` anywhere** in the surface path — neither v4 nor `v5_agentic` (the "cached system block" CLAUDE.md and this item imply was never wired). Concrete levers: **(A1)** add `cache_control` — Haiku-4.5 min cacheable prefix is **4096 tokens**, so first measure the instructions-vs-candidates token split; the reliable win is caching round-1's prefix so the 2-round loop's round 2 hits cache. **(A3)** drop the network roundtrip (single-shot, or expand candidates locally). Also: a live-mode `/watch` ignite pays this full cost per self-message because `<task-notification>` isn't caught by the `pre_response_recall` skip-gate — tracked as **A2** in the 2026-06-04 capture.

### Future — Auto-restart hung-daemon handling
`47c5907` capped Anthropic timeouts which addresses one trigger. Force-kill-then-respawn behavior for "process exists but not responding" still untouched.

### Future — Historical co_accessed trim
Pre-Phase-5 `co_accessed` edges still pollute the graph (`integrity_audit.py` flags this). Cleanup task documented as §16.8 in EPISODIC-REFERENCES.md; can run any time after episodic-refs ships.

---

## ⚡ Open items from 2026-05-11 temporal session

### Generic kv field promotion in render
Today the render hard-codes `event_time` as a promoted structured line ([surface_contract.py `_event_time_line()`](../servers/scales/s1/surface_contract.py)). Future generalization: **query-aware kv field promotion** — a temporal query promotes `event_time` / `created_at`; a "what did X say" query promotes `user_raw_quote`; a "why" promotes `reasoning`. Generalize via the existing `field_activation` scoring (the "cousin filtering" mechanism Tom referenced) so any kv field can promote when query-relevant. Tom's note: *"we have cousin filtering fields i think"*.

### UTC-internal clock refactor
Currently `brain.now()` returns operator's local TZ. UTC-internal storage + operator-TZ render at display time is the standard architecture for long-running multi-timezone systems. Today operator-TZ-default ships first since the daemon runs on the operator's machine, but if Anchor ever runs in a managed environment or multi-operator setting, UTC-internal becomes required. Brain memory `dcb5b951` notes this.

### Dispatcher enforcement for mandatory metadata fields
Encoder compliance asymmetry (brain memory `92b890e7`): restraint rules 100%, generative rules ~0-20%. v15.8 raised event_time compliance from 0% to ~5-8% — clear improvement but ceiling visible. Path B: dispatcher-level enforcement (precedent: `related/related_to` ban via dispatcher, brain memory `c39b8cc8` / `5e27a23f`). Detect at `remember_batch` dispatch: node has dated content but missing event_time → auto-extract via regex OR log loud to brain_errors. Higher reliability than prompt iteration.

### S2 Healer temporal enrichment
Healer currently fills `question` / `situation` / `reasoning`. Designed but unbuilt: dangling-anchor resolution (resolve "before the move" once "the move" is dated), implicit-sequence-edge creation (link co-occurring events with Allen relations), date propagation through sequence graph, cross-session temporal consolidation. Clean architectural slot — same idle cycle, same `Haiku + revise()` machinery.

### Agent introspection — remaining probes
4 of 6 modes built (aspect, compliance, coherence, counterfactual). Coverage probe was the open slot for "given THIS conversation, what would the agent do?" — **built 2026-05-15 as two domain-specific tools:** [`eval/agent_introspect/encoder_replay.py`](../eval/agent_introspect/encoder_replay.py) for encoder, [`eval/agent_introspect/surface_replay.py`](../eval/agent_introspect/surface_replay.py) for surface. Both replay the actual agent call against a candidate prompt without paying eval-pipeline cost (~$0.001, ~2-13s per replay). Unbuilt: **edge-case probe** (corner-case scenarios), **priority probe** (when rules conflict, which wins?). Build any when a specific iteration arc demands it.

---

## ⚡ Open items from 2026-05-15 surface arc

### v14 + v6 surface eval (the right ship-test for surface v6)
2026-05-15 12-item diverse eval showed v15.11+v6 = 8/12 (67%) vs v15.11+v5 = 6/12 (50%) — surface v6 is +2 items on the same encoder. But v14+v4 = 10/12 (83%) on the same items. Conclusion: v6 surface is a real win, v15.11 encoder is not. **Right next step: run v14 + v6 on the 24-item cohort to isolate the surface-only ship.** If v14+v6 ≥ v14+v4 by any margin, ship v6 surface alone. Cost ~$15, ~50min.

### Surface `75832dbd` render-size bug
Item `75832dbd` ("Can you recommend some recent publications or conferences") hits `stop=max_tokens` at Haiku's 8192-token output ceiling. Pre-Haiku rendered content too dense for the model to emit JSON after thinking. Likely fix in `surface_contract.py` candidate rendering — trim candidate `content` field for surface input, or use compressed rendering. Not v6-prompt issue; this is a rendering-volume issue. Failing on 1/24 items in the A/B but reproducible.

### S1S Quality Rubric — implement or drop?
Apr 24 design notes at (former) `docs/S1S-QUALITY-RUBRIC-NOTES.md` (now archived) proposed a multi-dimensional per-node + per-run Haiku-judge rubric: structural (0/1), semantic (0/1/2), atomization (0/1), correction-axis specifics. Never built. **Question for Tom: still wanted? If yes, P3 validation infrastructure; if no, drop the design intent.**

### SKILL.md Encoding Craft section reframe
2026-05-15 SKILL.md surgical edits flipped 4-5 active-encoding bullets but did not touch the "## Encoding Craft" section (title + line 109 "when you encode lessons" + Encoding Richness subsection). Decision deferred — wider reframe scope. Either delete the section (encoder owns this craft) or reframe it as "Node Craft — for revising and reading nodes" with verb flips.

### connect_to ID-shape pattern in encoder prompt
2026-05-15 bug fix (commit `08156ee`) at `_resolve_connect_to_entry` recovers when the encoder passes an 8+ hex-char ID in the `title` field. Real root cause is the encoder prompt not having a clean schema for "connect to this specific known node by id" — the `connect_to` spec is title-only. A future improvement: support `connect_to: {id: "...", relation: ..., why: ...}` shape so the encoder can express ID-based connection intent directly.

---

## Recent ship log (older — moved to archive review)

The May 8, May 10, and May 15 ship logs (encoder prompt versions v14/v15.3/v15.6/v15.8/v15.11, AspectIntegration rewire, eval artifacts subsystem, surface v6 prompt, 24-item A/B eval) are preserved in the historical context but trimmed from this backlog. Current encoder iteration moved past v15.11 to the v18-era work and is now superseded by the v19 + episodic-references arc (see EPISODIC-REFERENCES.md §7).

Key carry-forward items from those sessions (still relevant):
- **v15.11 encoder was not recommended for ship** — atomic-fact substrate didn't aggregate as well as v14 narrative bundles for multi-session synthesis. The episodic-references arc resolves this differently (source_refs preserve substrate while the encoder writes atomized framing).
- **Surface v6 candidate registration HELD** pending v14+v6 isolation test. Still held; will be re-evaluated once episodic-references ship.
- **Eval artifacts subsystem + refined-bucket analyzer + run-diff** all in production (eval/longmem/ + eval/agent_introspect/).

---

## The mission

Recall — the moment relevant memories rise into Anchor's awareness when the operator speaks. Everything in this backlog either *directly* improves that moment (P0–P2), *validates* it (P3), or is operational hygiene that prevents regressions (P4–P5).

Priority bands:
- **P0** — blocking right now
- **P1** — high-leverage recall improvements (direct user-felt impact, designed-or-cheap)
- **P2** — recall arc (bigger pieces, designed-not-built)
- **P3** — validation infrastructure
- **P4** — operational backlog (post-launch items)
- **P5** — backburner / long-tail

---

---

## P1 — High-leverage recall improvements (design-done or cheap)

> **⚠ Code-verified 2026-05-30 (Anchor).** These bands predate the Frame (Phase 2)
> + episodic-refs ships and carried stale framing. Per-item check against code:
> **P1.2–P1.5, P2.2, P2.4, P3.x confirmed NOT-BUILT (accurate as written).**
> **P1.1, P2.1, P2.5 had stale claims — corrected inline below.** Don't trust a
> band's "design-done" label cold; verify against code before picking up.

### P1.1 — Frame as filter (recency bias fix)

- **Why:** BRAIN-CHALLENGES.md entry #2. Recall consistently returns aged-but-topical clusters when the user names a recent specific arc ("Aspect encoder", "Frame", "where we left off"). Frame holds the recent arc as a structured prior but isn't biasing candidate selection. Direct user-felt failure — Tom hit this multiple times this session.
- **What:** Frame's `Active threads` and `Recent moves` sections carry node IDs. Pass that ID set into the surface scoring step. Boost candidates whose IDs match (or are 1-hop neighbors). Apply BEFORE Haiku selects from the 25-candidate pool.
- **Files:** `servers/scales/s1/frame.py` (expose `frame.frontier_ids()` returning the union), `servers/scales/s1/surface_contract.py` (the scoring function — add a `frame_match_boost` step), `servers/scales/s1/surface.py` (pass frame to scoring).
- **Effort:** ~2 h.
- **Acceptance:** capture the two failing prompts from BRAIN-CHALLENGES #2 (`aspect_encoder_pickup`, `frame_recall_resume`) as labeled queries in `eval/frame_replay.py`. Snapshot before/after. Both should surface fresh-arc nodes (not 1-month-old encoder-optimization history).

> **Verified 2026-05-30:** premise partly stale, acceptance wrong.
> (1) Frame **did** ship as a prompt-prior — injected as a "Partnership context
> (your prior)" block ([surface_contract.py:369](../servers/scales/s1/surface_contract.py)).
> That sways Haiku's choice *among* the 25 candidates. P1.1's distinct,
> still-**NOT-BUILT** lever is a candidate-**scoring** boost on Frame frontier IDs
> (no `frontier_ids()` / `frame_match_boost` in code) — it changes *which* 25 make
> the pool. The recency miss is most likely pool-composition, so P1.1 is **not**
> subsumed by the Phase-2 prior.
> (2) Acceptance is broken: `aspect_encoder_pickup` / `frame_recall_resume` do
> **not** exist in `eval/frame_replay.py` (corpus is `exco_cold`, `self_intro`,
> `exco_pivot`, `where_were_we`, `open_last_week`). First step is to ADD the
> failure queries, then re-confirm the miss still reproduces post-Phase-2 before
> building.

### P1.2 — Phrase-anchored title boost

- **Why:** Same failure family as P1.1. When user says "Aspect encoder" by name, candidates whose titles contain that exact phrase should pin to the top — not get embedded into a cosine score that buries them under broader topic matches.
- **What:** Extract proper-noun-ish phrases from query (capitalized multi-word, 2–4 tokens). FTS5 search the title field. Boost matches.
- **Files:** `servers/scales/s1/surface_contract.py`.
- **Effort:** ~1 h.
- **Acceptance:** same regression queries as P1.1.

### P1.3 — Connection scoring (Step 3.5)

- **Why:** Designed in `RECALL-OVERVIEW.md` §3, never built. Localizes hub bias — true hubs only dominate when they connect to other relevant candidates for THIS query, not always. Spec exists.
- **What:** After candidate enrichment scoring, score each candidate by connectivity to OTHER high-scoring candidates in the pool. Edge type weights via `brain.aspects` (`correction_improvement`/`extension_refinement` strong; `generic_relation`/`noise` weak; `hierarchical_structure`/`temporal_sequence` moderate). Cluster detection: 3+ interconnected candidates score together; isolated high-cosine nodes get lower priority.
- **Files:** `servers/scales/s1/surface_contract.py` (new `_connection_score()` step), `servers/brain_recall.py` (rerank pipeline integration).
- **Effort:** ~3–4 h.
- **Acceptance:** `eval/frame_replay.py` shows different ranking on `where_were_we` corpus vs baseline; hub nodes ranked lower when query doesn't match cluster.

### P1.4 — Posture detection (recent-vs-historic bias)

- **Why:** When user says "where we left off / yesterday / what we just did," surface should bias hard toward recent. Today it doesn't — same scoring regardless of intent.
- **What:** Lightweight regex/heuristic on query that detects "recency intent" → boost recency in scoring. Could be a single boolean knob passed into scoring.
- **Files:** `servers/scales/s1/surface_contract.py`.
- **Effort:** ~1 h.
- **Acceptance:** same regression queries; recent nodes (≤7d) outrank older equivalents on recency-flavored queries.

> **Verified 2026-05-30 — mostly pre-built; this is a wire-up, not a from-scratch build, and it has a regression scar.**
>
> **The math already exists.** `servers/recall_scoring.py` (Apr 2) defines
> `unified_score(semantic_score, created_at, emotion, access_count, confidence)`
> = `base * (1 + recency_boost + emotion_boost + frequency_penalty + confidence_boost)`,
> bounded ~0.81–1.55×. `freshness_from_created(created_at)` is the recency term
> (fresh nodes lifted; old nodes get boost=0, *no penalty*). The inputs are
> already collected into scope at recall time — `node_created_at` /
> `node_emotion` / `node_access_count` at [brain_recall.py:1346-1348](../servers/brain_recall.py).
>
> **Why it's OFF: it regressed R@8 by −10pts.** The deferred-work comment at
> [brain_recall.py:1712](../servers/brain_recall.py) records that wiring
> `unified_score` in *unconditionally* dampened scores that were previously
> passing the relevance floor. Its own next-step: *"likely need per-query-type
> adaptive weights rather than one fixed formula."* **That sentence IS this
> item** — the recency-intent boost = apply the existing freshness modulator
> ONLY when the query has recency intent (and relax `frequency_penalty` then),
> instead of always. So the two remaining pieces are: (1) the intent detector
> (the genuinely-new part — what phrasings count, false-positive cost), and
> (2) the conditional gate at line 1712 where `query` + `node_created_at[nid]`
> are both already in scope.
>
> **Revised effort & file:** ~half-day, and the real insertion is
> `brain_recall.py:1712` + `recall_scoring.py` — NOT (only) `surface_contract.py`.
> **Mandatory eval gate** (`decode_funnel` / `frame_replay`) — the naive
> always-on version already burned −10pts; cannot just flip on.
>
> **Orthogonal to fatigue.** Fatigue ([brain_recall.py:1562](../servers/brain_recall.py),
> `score *= 1-fatigue`) is a per-session *anti-repeat* dampener; freshness is
> node-age. They don't fight.
>
> **Caveat — freshness is a proxy.** "Where we left off" really means *this
> session's arc* (→ P1.1 Frame-frontier), not raw node age. A 3-day-old
> unrelated node also gets the freshness boost. P1.4 is the blunt lever; P1.1
> is the precise one. They stack.
>
> **Live evidence (2026-05-30 experiment):** recency intent carries zero weight
> today — query "where we left off *today* — the brain_batch invalid op fix"
> ranked today's fix node #4 behind three ~6-week-old April nodes; another
> today-node fell out of the top-25 entirely on the topical query. Reproduced
> as a **ranking/pool-composition** problem (not Haiku selection), N=2 / one
> topic — a real verdict needs the recency-query eval corpus that P1.1's
> acceptance also lacks.

### P1.5 — Cadence split: brain-level vs session-level Frame caching

- **Why:** Today every recall re-injects the full Frame (~1900 tokens). 60% of it (Operator + Partnership-integrated + Permanent) is slow-changing — wasted re-injection most turns. Cost + latency.
- **What:** Split Frame build into brain-level (cacheable, refreshed on S2 cycles or encoder writes) and session-level (current_focus + recent_moves, refreshed per turn). Two cache breakpoints in surface system block.
- **Files:** `servers/scales/s1/frame.py`, `servers/scales/s1/surface.py` (cache_control structure).
- **Effort:** ~2–3 h.

### P1.6 — Dampening cluster: synaptic-fatigue + hub-dampening regression (KNOWN BUG)

- **Why:** Post spread-activation migration, the per-session anti-repeat **fatigue** dampener and **hub-dampening** are broken. Four tests reproduce it and are **correctly RED** (code wrong, not the tests): `test_fatigue_accumulates`, `test_fatigue_dampens_scores`, `test_fatigue_increments` (in `tests/integration/test_recall_pipeline.py` / `tests/integration/test_session_mechanisms.py`) and `test_hub_dampening` ([test_recall_quality.py:172](../tests/test_recall_quality.py)). Parked **with the recall work** since 2026-05-29.
- **Detail:** `docs/archive/session-handoffs/S2-GATING-AND-TEST-CLEANUP-HANDOFF.md` → "Parked work" + "Intentionally left alone" (`test_high_confidence_ranks_higher` is entangled — "fixing" it could surface this same bug).
- **Verify on pickup:** confirm the four are still red before assuming — they've sat a week.
- **Effort:** unknown until diagnosed; natural bundle with the recall-side episodic-refs arc (both are recall-quality).

---

## P2 — Recall arc (designed, bigger builds)

### P2.1 — Agentic Haiku-first recall (7-tool `fetch_batch`)

- **Why:** Design finalized in `docs/archive/FRAME-DESIGN.md` §4. Replaces single-cosine candidate pull with Haiku planning the fetch per-turn. Variable cost, sample-then-deepen, frame-shaped output. The next major recall capability. (Note: `BRAIN_SURFACE_VARIANT=v5_agentic` is the live surface variant — that's a different tool surface than this 7-tool `fetch_batch`; don't conflate.)
- **The 7 tools:** `search(query, mode, limit)`, `find_about(entity, limit)`, `find_open_loops(topic?, limit)`, `trace_lineage(node_id, direction, max_steps)`, `get_community(community_id, query?)`, `find_temporal(when, query?, limit)`, `get_full(node_ids)`. All wrapped in single `fetch_batch` for parallel-op single Haiku turn.
- **Files:** new `servers/scales/s1/fetch_batch.py`, surface prompt v5, `servers/daemon_dispatch.py` (new commands), tool descriptions.
- **Effort:** ~2 days.
- **Depends on:** Q13 decision (does spread activation survive Phase 4?).

> **Verified 2026-05-30:** `servers/scales/s1/fetch_batch.py` does **not** exist —
> unbuilt as described, a ground-up build. Caveat: `BRAIN_SURFACE_VARIANT=v5_agentic`
> *is* active (`brain-env.sh`), but that's the surface *rendering/selection*
> variant — NOT this 7-tool fetch module. Don't conflate them.

### P2.2 — Multi-anchor query decomposition

- **Why:** Implements the query-multiplicity principle. Messages contain 2–4 distinct concepts; collapsing into one cosine vector loses the structure. Decompose, run multiple anchors in parallel, convergence is the strongest signal.
- **What:** Query-decomposition step (heuristic or Haiku) → multi-spread orchestration → convergence scoring on overlapping nodes.
- **Files:** new `servers/brain_recall_multi.py`, surface integration.
- **Effort:** ~1–2 days.

### P2.3 — Hybrid retrieval — FTS5 + embeddings full integration

- **Why:** Partially shipped — recall pipeline has both channels but they don't combine cleanly. Full integration with connection scoring (P1.3) lets us catch both "what does this concept mean" (embeddings) and "what was that exact phrase" (FTS5).
- **Files:** `servers/brain_recall.py` (channel combiner), `servers/scales/s1/surface_contract.py`.
- **Effort:** ~1 day. Depends on P1.3 (connection scoring) to score the union.

### P2.4 — Wire Frame into S1 Scribe (gap-aware encoding)

- **Why:** Encoder doesn't currently see Frame; doesn't know what's already in awareness. Could yield complementary encoding instead of restating.
- **Files:** `servers/scales/s1/encode.py`, `servers/scales/s1/encode_contract.py` (Frame in user content).
- **Effort:** ~3 h.

### P2.5 — Q13 decision: kill spread activation or keep it?

- **Why:** Today's `_traverse_graph()` 3–4 s baseline cost eats the latency budget that Phase 4 (agentic) tools need. Anchor's lean: retire spread, keep kernel as tool-internal helper. **Decision needed before P2.1 ships.**
- **What:** This isn't a build, it's a 30-minute conversation + decision document.

> **Verified 2026-05-30:** largely **resolved in practice.** `_traverse_graph` was
> removed from the recall path 2026-04-14 (dead, 0 callers — see `brain_recall.py`
> comments); `spread_activation` still lives and runs in `surface.py` (post-selection
> expansion). The de-facto state already matches Anchor's lean — retired from recall,
> retained in surface. What's left is to *document* the decision and confirm P2.1
> doesn't still treat this as a blocking gate.

---

## P3 — Validation infrastructure

### P3.1 — Fresh-Claude vs Anchor calibration test (Frame punch list #11)

- **Why:** Only path to empirically validating SKILL.md / boot changes. `eval/frame_replay.py` bypasses Claude Code; `eval/longmem` deliberately avoids Anchor's voice. Today Tom is the only sensor for "does Anchor wake up as Anchor."
- **What:** Spawn fresh Claude Code session with brain skill loaded; identical wakeup probes ("Who am I working with? What's open? Where were we?"); compare to fresh Claude WITHOUT brain. The delta IS what the brain buys at the wakeup moment.
- **Files:** new `eval/calibration_fresh_vs_anchor.py`.
- **Effort:** ~4 h.

### P3.2 — Fix `eval_runner.py` bypass of enrichment scoring

- **Why:** `RECALL-OVERVIEW.md` §4 tension #2. Eval bypasses the enrichment scoring step that production uses. Backfill / scoring improvements are invisible to eval.
- **Files:** `eval/eval_runner.py` (wire enrichments into evaluator) OR switch to production recall method.
- **Effort:** ~2 h.

---

## P4 — Operational backlog (was PHASE-B+1)

### 🟡 MEDIUM

#### P4.2 — Build `s2_vector_healer` unit (was B+1.4)
Detects + repairs stale vectors that escaped `revise()` invalidation (kv text updated AFTER `node_enrichments.created_at`). Backstop for paths that bypass revise. **2–3 days.**

#### P4.3 — Encoder activation visibility into S1R (was B+1.5)
S1R discards activation metadata before encoder. Encoder makes revise decisions blind to which fields fired and how strongly. Pass activation through trace metadata. **4–6 h.**

#### P4.4 — Multiple format configs consolidation (was B+1.6)
Three subtly-different configs for `render_rich_node` (HAIKU_FORMAT / SURFACE_FORMAT / S1_NODE_CONFIG). Merge into a single config family with named modes. **1 h.**

#### P4.5 — Edge selection called twice per recall (was B+1.7)
`daemon_hooks.py:229` calls `select_edges()` per candidate; `surface_contract.py:1129` calls AGAIN during activation render. Cache first call's result. **30 min.**

#### P4.6 — Catalog from rendered strings, not activation results (was B+1.9)
`build_node_catalog()` regex-extracts node IDs from rendered surface text strings. Inefficient and fragile. Track surfaced node IDs in S0/S1 traces directly. **1 h.**

#### P4.7 — Healer unsolicited fields (was B+1.14)
S2 Healer asks for specific missing fields; Haiku returns ALL three. System rejects + logs loudly. Strengthen prompt (move single-field example first) OR move to `tool_choice` JSON schema. **30 min prompt / 2 h schema.**

#### P4.8 — Haiku selects IDs not in candidate menu (was B+1.15)
Surfacer Haiku given top-25 menu, sometimes returns out-of-menu IDs. Could be feature (Haiku knows IDs from training) or bug. Diagnostic first: log out-of-menu IDs, check if real brain nodes. **30 min.**

#### P4.9 — Encoder uses two tool families when one would do (was B+1.16)
First production cycle used `remember_batch` + `brain_batch` = 3 rounds when one `brain_batch` would do. Strengthen prompt's MIX rule. **20 min.**

### 🟢 LOW (cleanup, do in batches when convenient)

- **P4.10** — `find_missing()` filter naming + `source_kv_keys` semantics doc (B+1.3) — 10 min
- **P4.11** — Truncation has no ellipsis (B+1.8) — 15 min
- **P4.12** — Remaining silent-`pass` excepts (B+1.10), 4 sites — 5 min each
- **P4.13** — Hardcoded constants → interaction config (B+1.11) — 30 min
- **P4.14** — `failed_connect_to_count` in batch result (B+1.12) — 10 min
- **P4.15** — Sibling-map case-sensitivity docstring (B+1.13) — 5 min
- **P4.16** — Trace metadata bloat (B+1.17) — 30 min after a week of accumulation
- **P4.17** — Rename `judge_output` → `surface_output` across the trace metadata contract — 1–2h. The S1 surface step was renamed from "judge" → "surface" in commit `620fb4f` (2026-05-03), and the user-facing/code path has been cleaned (commit `b126d98`, 2026-05-09 — `surface.py` no longer falls back to 'judge', orphan 'judge' interaction row deleted). What remains is the trace metadata field name still carrying the legacy "judge_output" — written by `dal.py:get_user_turns` (lines 687–729) into the `judge_output` key of trace dicts, read/asserted on by `tests/test_s1_data_assembly.py`, `tests/test_okd_cycle.py`, `tests/test_scout_muster.py`, `tests/test_trace_system.py`, plus the `pipeline_contract.py` legacy aliases (`format_candidate_for_judge`, `build_judge_prompt`, `format_judge_output` — lines 509–511). One commit: rename the field, drop the aliases, update tests. No data migration needed (it's a derived field assembled from `additionalContext` traces, not stored). Defer until there's another reason to touch dal.py to keep the diff focused.
- **P4.18** — Dashboard Frame view (from archived FRAME-DESIGN.md §0 punch-list #5) — display the current Frame for any session as observability. Pure read-only; useful for debugging "why did this turn surface what it did." ~1h.
- **P4.19** — Rename `haiku_id_outside_candidates` → `haiku_id_from_prior_context` + downgrade `_log_error` → debug log (from archived FRAME-DESIGN.md §12.1). Code-verified still firing at `surface.py:701` 2026-05-31. Investigation confirmed it's not a bug — Haiku correctly using multi-turn context, picks IDs from prior turns that resolve to real nodes. Error log is noise; downgrade so real new errors aren't hidden under it. ~20 min.
- **P4.20** — `spread_seed_no_vectors` archived-node race (from archived FRAME-DESIGN.md §12.1 / Q12). Code-verified still firing at `surface_contract.py:962`. Haiku picks a node from prior-turn context that was archived since; vectors cascade-deleted on archive → spread crashes gracefully but loudly. Two options: (a) vector grace period (don't cascade-delete on archive immediately, keep ~24h), (b) validate Haiku picks against current archived state, classify as `haiku_id_now_archived`. Recommend both. ~1-2h.
- **P4.21** — Agentic surface trace observability (from archived AGENTIC-SURFACE-CONTRACT.md §5, never shipped). Add 3 new `ref_type` values under `scale='s1'`, `event_type='K'` to `servers/trace_contract.py`: `tool_call` (per tool invocation — `{tool, args, result_count, latency_ms, round_idx}`), `tool_round` (per round boundary — `{round_idx, tools_called, total_tools, elapsed_ms}`), `surface_variant` (per surface call — `{variant, prompt_version}`). Wire emission in `_call_surface_agentic`. Without this, we have no per-turn record of which fetch tools Haiku invoked or how many rounds it ran — opaque agentic loop. ~2-3h.
- **P4.22** — PostToolUseFailure → failure memory recall (from archived HOOK-BRAIN-INTEGRATION.md #1). When a tool fails, recall lessons about similar failures before Claude retries blindly. Brain data: `lesson` / `failure_mode` / `bug_lesson` nodes matching error context. Handler: command hook (daemon recall with error as query). What Claude sees: "this file failed before because X. The fix was Y." Brain has the answers; nobody asks today. ~1-2h.
- **P4.23** — SubagentStart → brain context injection (from archived HOOK-BRAIN-INTEGRATION.md #2). Subagents (Explore, Plan, claude-code-guide, general-purpose) currently spawn brain-blind. They repeat mistakes the brain already corrected. Inject `engineering_context()` — conventions, constraints, mechanisms, locked rules — at SubagentStart. Handler: command hook, existing daemon endpoint, no new code. Every subagent we spawn today wastes tokens rediscovering things the brain knows. ~1h.
- **P4.24** — Dead-guard cleanup (Bucket E from `docs/archive/session-handoffs/S2-GATING-AND-TEST-CLEANUP-HANDOFF.md`). `build_delta_metadata(*, actions=0, …, **extras)` declares reserved keys as keyword-only params, so they can never arrive via `**extras` — the `if k not in metadata` guard is dead code. Remove the dead guard in `servers/trace_contract.py` (a source edit, not guardian-blocked) + retire `test_extras_do_not_overwrite_reserved_keys`. ~10 min. (Was blocked when trace_contract.py was under parallel edit; that self-channel edit has since landed.)

---

## P5 — Backburner

### Old standing threads (open in graph, low recent activity)

- **Telemetry / brain proprioception** (overdue since 2026-03-29, reminder `49b33c19`) — comprehensive observability layer. Likely subsumed by P0.1's memory_watchdog work + dashboard.
- **Temporal reasoning ideas** (`095cc070`) — relative time display already shipped (`format_judge_output`). Time-range retrieval is separate, harder, deferred.
- **Encoder vs Stop hook 10s timeout** (`38bc9a6f`) — addressed by background-encoding architecture; the tension node may be stale.
- **Emergent types ignored** (`c8c773b4`) — partially addressed by today's aspects work; the original "auto-promote when N nodes accumulate" hypothesis isn't built.
- **Host environment awareness** (`891f9a53`) — never worked. Low priority.
- **Brain proactively surface prior art** (`dd7b4d20`) — aspirational. Overlaps with agentic recall (P2.1).
- **Write-path autocommit (Option B / `isolation_level=None`)** — deferred defense-in-depth hardening. The F3 root fix (Option A) shipped and removed the user-visible failure, so there is no urgency; Option B has a bigger blast radius and is **benchmark-gated** (recall + encode latency before/after). Full record + rationale: `docs/archive/WRITE-TXN-ISOLATION-ROOTFIX.md` → "Option B".

### Stage 1C — explicitly deferred

Keywords→KV migration. Audit confirmed 0 dual-state. Pick up only if natural.

### Latency tuning (from archived FRAME-DESIGN.md §14, eval-gated)

- **P5.1 — Lighter candidate format** (was §14.3). Reduce per-candidate token cost in surface prompt. Today each candidate is ~250-400 tokens (metadata, situation, edges). Strip what Haiku doesn't actually use for selection. Investigation: what does Haiku actually look at? Could test by ablating each field and checking selection quality. Expected gain: 30 candidates × 100 token savings = 3K tokens per call. Marginal latency, real cost reduction. ~half-day with eval.
- **P5.2 — Reduce candidate count A/B** (was §14.4). Currently 30 candidates per surface call. A/B `max_candidates ∈ {15, 20, 25, 30, 35}` on the test corpus. Quality vs latency curve. ~1-2h eval.

### The "irresolvable" tensions

- **SKILL.md tension** (`RECALL-OVERVIEW` §4 #7) — instructions to a stateless thing about how to behave as if continuous. Built-in contradiction. Don't try to dissolve.

---

## Decisions needed (open)

These aren't builds — they're choices that gate other work.

| # | Question | Gates | Lean |
|---|---|---|---|
| Q-A | Daemon memory_watchdog: enable now or after next leak? | P0.1 | enable now |
| Q-B | AspectIntegration auto-merge or operator-review gate in production? | P0.2 | auto-merge for now |

**Resolved 2026-05-31** (doc-cleaning session):
- **Q13** (kill spread activation?) → resolved in practice: `_traverse_graph` removed from recall path 2026-04-14; `spread_activation` retained as post-selection expansion in `surface.py`. De-facto matches the lean (retire from recall, keep kernel). No further decision needed.
- **Q-C** (wire Frame into encoder?) → resolved: encoder stays **per-session view only** (its own arc + journal). Brain-wide Frame is for Anchor's recall, not the encoder's catalog — the encoder needs distance, not more context (brain `eaf833c5`). Not wiring it.

---

## What this replaces

- `docs/PHASE-B+1-BACKLOG.md` — moved to `docs/archive/`. All items folded into P4 here.
- `docs/RECALL-OVERVIEW.md` §3 — short list pointing here; the recall arc + Frame punch list inline there will be replaced with a pointer.
- `docs/BRAIN-CHALLENGES.md` — kept (different purpose: cognitive-gap log). Items #1 + #2 fixes now live in P1 here.

## How to update this doc

- Item ships → strikethrough + `→ shipped {commit}`
- Item promoted/demoted → move between bands, note date
- New item → numbered slot in target band, brief why + acceptance
- Don't delete items — keep history visible
