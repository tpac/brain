# Trace Modes — One Recorder, One Switch, Files for Fat Payloads

Status: PROPOSED v2 (2026-08-02) — awaiting Tom's approval before implementation.
v2 incorporates a three-lens panel review (implementer / simplifier / consumer);
every code claim below was verified against the repo by the panel.
Thread: trace-debug-unification (handoff node e25d60f3; motivating finding 74dfb59c —
the 6M-char tool result whose diagnosis required stitching four capture sources).

## Rulings this design implements (Tom, 2026-08-02)

1. **Modes, not a dial.** Trace extensiveness is discrete: `normal` = bare minimum
   forever, `debug` = opens up all the information. No per-field verbosity creep.
2. **The traces layer owns trace structures.** Consumers call layer functions
   (`build_delta_metadata` / `build_failed_run_metadata` pattern); no caller ever
   builds or parses a trace record, payload path, or cap.
3. **Unify, don't hack.** All ad-hoc capture (tmp files, capture dirs) is deprecated
   into the one mechanism. All S1 and S2 agents record the same way.
4. **Fat payloads live in FILES, not the DB** (supersedes the June sidecar-table
   decision 8851f5a1). Files are human-legible, visually separate from the DBs,
   and the user can delete them "from outside" — deletability is a designed
   property, never a failure mode.
5. **daemon.log stays its own layer** — it's the channel that works when the brain
   isn't accessible. Operational prints are not capture; they don't migrate.
6. **Per-kind on/off gates in config**, runtime-updatable (K-store).
7. **Must work on IsolatedBrain** — that's where debugging happens.

## Architecture

### The recorder lives in the traces layer

The centralization point is NOT any LLM loop — `run_llm_loop` covers the Scribe and
S2 encoders, but the surface's agentic loop, the scouts, recall query-expansion, and
the healer encoder all call the API directly. The one layer they already report to
is the traces layer. It grows a recorder API (methods on `BrainTracesMixin`,
contract in `trace_contract.py`):

```python
brain.record_payload(chain_id, kind, content, *, seq=None)
    # → pointer (path RELATIVE to db_dir) or None (kind gated off / empty).
    # Gate-aware. Writes the file, returns the pointer the caller stores on
    # its trace row metadata. Never raises into the caller's run (loud-logs
    # on failure, returns None). Unknown kind → gated off + ONE loud errors-
    # table entry per kind (a typo'd kind must be visible, not silent).

brain.read_payload(pointer)  # → Optional[str]; None = pruned/missing/gated
```

Call sites hold zero knowledge of gates, caps, paths, or formats. A new payload
kind is a new `kind` string + a config entry — no new mechanism. (LAF recall-field
dumps can ride this later exactly that way; deliberately NOT pre-registered.)

**The runner seam** (panel finding — `run_llm_loop` has no brain by design, and the
per-round capture moment is inside the loop at `_create_message`): `run_llm_loop`
grows a `record_round_fn(round_idx, payload_dict)` callback parameter that
**replaces** `capture_label`. Each caller builds the closure over its own
`(brain, chain_id)` — encode.py has `brain` + the s1e chain; the S2 encoders have
`self.brain` + the unit's run chain. Same boundary pattern as error logging
(encoder-side, never runner-side). Do NOT route recording through `dispatch_fn` —
dispatch closures are write-classified and attribution-stamped, and the
surface/scout paths don't use dispatch at all.

**Payload root:** `{db_dir}/payloads/` where db_dir is the directory of the brain
INSTANCE the caller holds — never a global env var (the a88343d6 lesson:
env-resolved writer/reader path seams split-brain silently). IsolatedBrain and
eval's per-item fresh brains land payloads inside their own dirs by construction.
The Scribe runs in-process on the daemon's brain (no separate read_brain — that
premise is outdated), so there is exactly one production db_dir. File writes touch
no SQLite; recording from encoder worker threads has no contention.

### What stays in traces (DB) vs. what goes to files

| | trace_events (brain_logs.db) | payload files |
|---|---|---|
| role | permanent index + bounded forensics | fat payloads, human-first |
| content | chain skeleton, action records with bounded heads, token/size counts, payload **pointers** | full prompts, per-round request/response payloads, full tool results, judge output |
| lifetime | bounded by `trace_detail()` caps (row pruning: future work — trace_events is never pruned today) | age-pruned by the S2 maintenance cycle; user-deletable anytime |

The trace row is authoritative and self-sufficient: bounded forensics survive
payload deletion. The pointer is enrichment; `read_payload` on a missing file
returns None and consumers render "payload pruned".

### File layout

```
{db_dir}/payloads/                      # sibling of brain.db — obvious, separate
  2026-08-02/                           # date-first: retention/deletion = one glob
    s1e-cca75b43-48/                    # chain-second: one run = one ls
      000-prompt.md
      001-round_payload.json
      002-round_payload.json
      003-failed_run.json
    s1r-cca75b43-12/
      000-prompt.md
      000-judge.json
    s2-20260802143000-consolidation/    # S2 chains: identical shape, no special case
      000-prompt.md
```

- Filename: `{seq:03d}-{kind}.{ext}` — `kind` is the gate-config key **verbatim**
  (filenames grep against config; a kind that needs two formats is two kinds),
  `seq` is the caller's round/stop ordinal (3 digits: numeric sort survives
  100+ rounds), extension is fixed per kind in the contract (`prompt`→`.md`,
  `judge`/`round_payload`/`failed_run`→`.json`).
- **Chain directories are append-only** (panel finding: a Scribe retry on the idle
  tail reuses the SAME chain_id — same stop_counter — and would silently overwrite
  the failed attempt's files, destroying exactly the forensics you kept them for).
  The recorder opens with `O_EXCL`; on collision it suffixes an attempt ordinal
  (`000-prompt.2.md`). A chain dir only ever grows; the retention pass is the only
  deleter.
- Date dir = wall-clock day of the chain's FIRST payload; the recorder reuses an
  existing chain dir found under yesterday's date before creating today's (a run
  straddling midnight stays in one dir).

### Gates: one runtime switch, K-store-resident

The panel killed the two-switch design: `BRAIN_TRACE_MODE` is set nowhere, a
launchd daemon's env is fixed at spawn, so the env half was restart-only — the
switch you'd reach for mid-incident is the one that doesn't turn, and the restart
destroys the wedged state you wanted to inspect. File capture therefore has ONE
switch: the `trace_recording` K-store interaction (config-only, like `recall_laf`
gains), per-kind **on/off**:

```json
{"kinds": {"prompt": true, "judge": true, "failed_run": true,
           "round_payload": false, "tool_result": false},
 "retention_days": 14}
```

**Modes survive as named config versions**, preserving the modes-not-dial ruling:
`normal` (the default above — bare minimum forever) and `debug` (everything true)
are both pre-registered; "entering debug" = `set_interaction_active` — one MCP
call, instant, reversible, no restart, works from a live conversation while the
daemon stays up. Discrete shapes, no dial, one switch.

The recorder reads the config per call (1-2 SELECTs on logs_conn — negligible at
per-round frequency; if it ever matters, copy the `LAFEngine.config` TTL-cache
pattern, nothing heavier).

**`trace_detail()` (env) keeps only what it already ships: trace-ROW caps.** It
stays restart-bound and the doc says so plainly — that's acceptable for row-size
tuning, which is a deploy-time decision. One recommended amendment to shipped
code: `tool_result_cap` should become mode-INVARIANT (one value in both modes).
It truncates what the LLM SEES — raising it in debug mode means a debug-mode
retry runs a *different conversation* than the run it's trying to reproduce.
Capture must be observation-neutral: debug changes what's recorded, never what
the model experiences. (Row-head caps varying by mode are fine — those are
trace-side only.)

### Failure-triggered capture — the 2AM story

The panel's core finding: normal mode as originally designed left a failed run
with 500-char heads — "the interesting run is always the one you weren't
capturing." The fix costs nothing and is NOT a dial: at the moment `RunLoopError`
is raised, the full `msgs` list — the conversation exactly as the model saw it,
tool results already capped — is in memory. The encoder-side failure handler
records it:

```
failed_run kind: on run failure, the full msgs JSON, gated ON in the normal
config. Event-triggered forensics (build_failed_run_metadata's sibling), not a
verbosity level. Size is bounded by tool_result_cap × rounds.
```

The next 6M-char-class incident is diagnosable from the trace row + one file,
in normal mode, without reproduction.

## Migration table — the six existing capture mechanisms

| # | Mechanism | Writer | Reader | Fate |
|---|---|---|---|---|
| a | `/tmp/brain-encoding-prompt-*.json` | `encode.py` :153/:168 (ref_id at :1148) | dashboard `/api/encoding-prompt` | → `record_payload(chain, 'prompt')`; pointer carried in trace **metadata**; ref_id swaps only in the same release as the reader migration (see rollout) |
| b | `BRAIN_PROMPT_CAPTURE_DIR` per-round dumps | `runner.py:446 _capture_payload` | `eval/longmem/{build_corpus,ab_encode}.py` | → `record_round_fn` → `record_payload(chain, 'round_payload')`; eval reworked (own subsection below); env var + `capture_label` deleted |
| c | daemon.log `[s1e]` prints | `encode.py` (~10 sites) | grep | **KEEP** — operational log, its own layer |
| d | bounded action-record forensics | runner / trace layer | traces | already trace-native — unchanged; extended by `failed_run` |
| e | `/tmp/brain-consolidation-prompt-*.json` | S2 consolidation encoder (50KB-truncated today) | `dashboard/server.py:410` (keyed by batch number) | → `record_payload(chain, 'prompt')` — now full content (deliberate improvement); dashboard endpoint rekeys batch → chain_id |
| f | `/tmp/brain-judge-result-*.json` | S1 surface, daemon-side (`surface.py:934 _write_surface_result_file` — NOT a hook; brain in hand) | `dashboard/queries/recalls.py:19` | → `record_payload(chain, 'judge')` — the easiest of the six |

Killing (a)(e)(f) retires the hardcoded-`/tmp` dashboard readers and the
BRAIN_TMP_DIR writer/reader seam class for capture paths. The stale comment at
`trace_contract.py:329` ("full-payload capture belongs to the future unified
debug mechanism") is deleted when this ships. Operational tmp files that are
STATE, not capture — `brain-{session}-current-stop.txt`, the surface-selected
S1R→S1E handoff file — survive and are allowlisted in the contract test.

### Readers: dashboard reads files directly; daemon API for everyone else

The dashboard is a deliberately servers-decoupled passive observer — its charter
is reading the substrate **when the daemon is broken**, which is exactly when
you're staring at it. Routing payload reads through daemon TCP would make every
payload pane go blank in the primary debugging scenario. So: pointers are
relative to db_dir, and the dashboard resolves them against `BRAIN_DB_DIR`
exactly as `dashboard/db.py` already locates the DBs. Lazy stays: payloads load
on card expand, never on list render (the de3f5525 half that survives).

Sanctioned direct-file readers (named, like brain_traces.py names its DAL
exceptions): the dashboard and the eval harness (frozen-corpus brains run
daemonless). Everything in-process goes through `read_payload`; a daemon
`read_payload` command (one read-only `CmdEntry`) exists for TCP consumers that
don't share the filesystem.

**Legacy rows:** pre-migration `encoding_prompt` traces carry an absolute `/tmp`
path in ref_id. Readers branch on pointer shape: absolute → attempt the legacy
read, else "payload pruned"; relative → recorder pointer. The branch is
time-bounded — delete it once default views (24-48h windows) can no longer reach
pre-migration rows.

### Eval harness rework (migration row (b), expanded)

Capture is a hard validity gate for eval, not a nicety (`prompt_captured` is in
ab_encode's HARD_CHECKS; soft checks parse captured round bodies). Specifics:

- **Eval flips the per-kind gate, never a mode**: the harness registers the
  `trace_recording` config with `round_payload: true` on its fresh brains (it
  already has `_apply_interaction_override` machinery). Encoder behavior is
  untouched — an A/B run measures the production configuration
  (probe-input-fidelity rule).
- **Arm disambiguation moves from filename labels to brain dirs**: both A/B arms
  share a session id → identical chain_ids; today `BRAIN_PROMPT_CAPTURE_ARM`
  disambiguates filenames. Under the design each arm's brain has its own
  `payloads/` root — cleaner. `capture_files_for` and the HARD_CHECK re-target
  `{arm_db_dir}/payloads/**`.
- **`round_payload` content shape is pinned in the contract** = today's
  `_capture_payload` dict (`label, round, seq, model, effort, system, messages,
  tools`) so ab_encode's body-parsing checks port unchanged.
- `build_corpus`'s manifest `prompts_dir` field becomes per-item (payloads ship
  frozen inside each corpus item's brain dir). ab_encode's `wipe=True` now wipes
  payloads with the brain — correct for A/B (fresh run = fresh evidence).

## Failed-run residue (Tom, via the S1E-reliability stream, 2026-08-02)

Two gaps the sibling investigation surfaced; both belong to step 2:

1. **Per-op errors inside an ok=True batch are invisible.** A brain_batch can
   return ok=True while individual operations carry ok=False results — a giant
   per-op error string never reaches the errors table, so error scans miss it.
   Fix rides the existing encoder-side loud-scan pattern (the same pass that
   logs `s1e_oversized_tool_result`): scan action records for per-op failures
   and log them loud.
2. **A failed run leaves no reflective residue.** No journal note is written on
   failure, and the next encode's timeline doesn't surface
   `encoding_run_failed` traces — the retry doesn't know its predecessor died.
   Fix: the failure path writes a journal note (existing journal-note
   contract), and the next run's prompt assembly includes the chain's
   `encoding_run_failed` trace (with its `payload_pointer`) in the timeline, so
   the retry encodes with knowledge of the failure instead of amnesia.

## Rollout (two steps, no double-write)

The original five-step plan protected external readers that don't exist — every
reader lives in this repo. Two steps, each shipping alone:

1. **Recorder** (additive): `record_payload`/`read_payload`, gate config +
   normal/debug versions seeded, `failed_run` capture wired at the RunLoopError
   handler, retention on the S2 maintenance cycle (own `brain_meta` last-run
   stamp per S2 gating convention — NOT the logs-size check, which is
   size-triggered, on the error-write path, and never fires for files), tests.
   Existing writers untouched.
2. **Atomic migration**: wire all call sites (a,b,e,f) + `record_round_fn`
   replaces `capture_label` + migrate dashboard endpoints and eval scripts +
   delete old writers, `BRAIN_PROMPT_CAPTURE_DIR`, the three `/tmp` dashboard
   readers + flip ref_id semantics + enable the grep-pin contract test
   (allowlist: the operational state files above). Transition tests are the
   safety net; old `/tmp`-pointing rows degrade to the legacy-read branch.

## Test surface

- Contract: recorder is the only capture writer (grep-pin + allowlist), pointer
  format, `read_payload` None semantics, `round_payload` dict shape,
  gate-config schema (kinds map + retention_days), unknown-kind loud-log.
- Component: gate resolution (kind × config version), path derivation from
  db_dir (IsolatedBrain), append-only collision handling (O_EXCL + attempt
  ordinal), midnight chain-dir reuse, seq formatting.
- Transition: dashboard endpoints against pointer-carrying traces (+ legacy
  ref_id branch, + pruned render distinguishable from daemon-down at the
  endpoint); eval by-chain read equivalence with old capture-dir content.
- Cycle: replay a failed run on an IsolatedBrain copy — the 6M-char scenario's
  forensics must be reconstructible from trace row + `failed_run` payload alone,
  in the normal config.
