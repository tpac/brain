# Aspect Taxonomy Ownership — Architecture Plan

## Scope

The aspect taxonomy read/write surface: `aspects_v1.json`, the `AspectRegistry` that reads it, the
three code paths that write it, and the ~21 places that hold copies of taxonomy data outside the
taxonomy.

**Boundary traced:** `servers/aspects.py`, `servers/scales/s2/aspect_{contract,decoder,encoder,integration}.py`,
`dashboard/queries/aspects.py`, 18 consumer files reading `brain.aspects.*`, plus the eval/script
readers. Five parallel angle reviews (placement, unification, cohesion, coupling, altitude).

**Coverage caveats.** Reads/writes were enumerated by grepping the filename and the path helpers, so
a reader that resolves the path by some other means would have been missed (none found). The
`min_count_threshold`/rest-gate and `supersedes` multi-homing questions were excluded as settled and
not re-examined. Numbers below were measured against the live brain and the shipped seed on
2026-07-25 and are perishable — re-derive before acting on any of them.

**A coverage failure worth inheriting.** The first draft of this plan presented
`INTENTIONAL_EDGE_TYPES` as an urgent discovery. It is a deliberate operator deferral from a month
earlier (`id:28f7fe69`), and Step 0A now says so. The cause: the review recalled the aspect *storage*
history and passed those constraints to the review agents, but never recalled the aspect *filter*
history — so the agents found real code, correctly, and the synthesis mistook a parked decision for a
new one. **Any future review of this surface must recall both halves.** The brain prevents
re-litigation only for the questions you actually ask it about.

**The governing constraint.** Three storage architectures were weighed (brain node `id:9edc4cf5`)
and Option B won: the taxonomy is a **JSON config file**, sole source of truth. The aspects-as-nodes
path is retired. Nothing here relocates the data. "DAL-first" in this repo is a rule about the
SQLite databases; it is deliberately **not** extended into "the JSON must become a table." What is
in scope is giving the file an owner, and changing its *schema*.

**The evaluation criterion is legibility.** Tom does not read the code. This surface confused its
only maintainer repeatedly inside one session — the write race was misread in the wrong direction,
the file was hand-patched twice, and a new writer was added without anyone noticing. Prefer target
states that make a hazard structurally impossible over ones that document it.

---

## Execution posture — read this before starting any step

**Each step is: show the code → short discussion → agree → implement → verify.** Not "hand the step
to a session and let it run." Tom does not read the code, so the walk-through *is* his review — and a
plan is exactly the artifact that can bake in a wrong premise (`id:6a1af480`), which is why the
premise gets checked against the real code before anything is written.

Concretely, per step: open the actual files and confirm the Problem section still describes what is
there; bring the measurement (re-derive any number quoted here — they are all perishable); state what
changes and what could break; get the nod; then implement, verify, and commit.

"Executable cold" in this doc means each step **carries enough context to be understood** without
this conversation. It does **not** mean unattended execution. A step whose Problem section no longer
matches the code should stop and report, not adapt silently.

## Dependency summary

**Step 0A is not work — it is a parked operator decision. Do not schedule it.** (Read its section.)
**Step 0B is the one genuinely urgent, genuinely new item**, and it needs a word with the LAF stream
first: it changes numbers that stream is actively producing.

```
0B   (urgent — but coordinate with the live LAF stream before touching it)

1 ──► 2 ──► 3 ──► 5
│                 ▲
└──► 4 ───────────┘
     └──► 6   ◄── highest-value substantive step

7, 8, 9, 10  (independent — any order, any session)
```

- **Step 1** is the keystone: it dissolves five defects at once and is the prerequisite for 2, 3, 5.
- **Step 6** is the highest-value *substantive* step: eight filter lists frozen at whenever someone
  typed them, which never see a verb the classifier adds. It is the real version of what Step 0A
  looked like.
- **Steps 1 and 4** are independent of each other and together make Steps 2/3/5
  unnecessary-rather-than-fixed.
- **Steps 7–10** are standalone cleanups; 9 is the cheapest legibility-per-line on the list.

**Cross-thread collision warning.** Step 7 touches `consolidation_contract.suppression_relations`,
which is the mechanism behind journal-audit finding #4 (brain node `id:2ace76f3`), owned by a
different session (opener `id:2d85f6b2`). Do not do both. Whichever session gets there first should
note it on the node.

---

## Step 0A — `INTENTIONAL_EDGE_TYPES` — ALREADY PARKED BY THE OPERATOR. Do not schedule this.

**Status: deferred on purpose, one month ago, by Tom. Not a new finding.** Brain node `id:28f7fe69`
records a full audit of exactly this class — three hardcoded edge-filter lists predating the aspect
registry. Two were fixed (`LINEAGE_FAMILIES` → aspects union, commit `5be197be`;
`EXCLUDED_EDGE_TYPES` → noise guard, commit `4d190406`). This was the third, and Tom's words on the
node are explicit:

> *"intentional edge types shouldnt be done here. I've taken the idea and also you'll remember it
> for the right time and or a different stream. let's do the fixes."*

**Why it was parked, and why that still holds:** the swap is conceptually ready — the noise guard
removed the only technical blocker — but it **changes production recall behavior** and needs its own
eval stream with recall-quality benchmarks. It is not a side-fix, and it is not this plan's business.

**Scope, stated accurately** (the first draft of this doc overstated it): the constant has exactly
**one** use site — `brain_recall._enrich_results` (`servers/brain_recall.py:2243`), which attaches
`_neighbors` for display. Called on the top 3 of a recall and in `recall_node`; `_neighbors` is read
only by `daemon_hooks.py:388`, which promotes it to context. It does **not** touch candidate
selection, scoring, ranking, or the LAF field. The measured 25,566-of-36,219-rows figure is real and
is about that neighbor-context list — not about recall being 70% broken.

**If and when it is picked up:** it wants its own session with a recall-quality benchmark, and Step 6
should land first so `structural_exclusions(brain)` already exists to swap in.

**Lesson recorded.** This item came back as a "discovery" because this review recalled the aspect
*storage* history and fed it to the review agents, but never recalled the aspect *filter* history.
Any future review of this surface must recall BOTH. See `id:28f7fe69` and `id:40e7125a` first.

---

## Step 0B — Point the eval walkers at the live taxonomy, not the repo seed

**Problem.** `eval/laf/walker/influence_ops.py:63,152` and `eval/laf/walker/mesh_forensics2.py:39,69`
hardcode `REPO / 'servers' / 'scales' / 's2' / 'aspects_v1.json'` — the **seed** — and derive the
`corr` influence operator's verb set from it, while reading their edges from the live
`$BRAIN_DB_DIR/brain.db`. One probe, two provenances.

Measured seed vs. live: **411 members diverge across 17 aspects.** On the lane these scripts use,
`correction_improvement` is 26 in the seed against **58** live — the correction lane is scored
against under half its verbs. The gap is one-directional and grows every S2 cycle;
`min_count_threshold = 1` guarantees it.

This is live ground: the last five commits on `main` are LAF eval work, one of which already
retracted a +2.55pp result as a fold artifact. A taxonomy-source skew of this size is a plausible
contributor to that class of unreproducible delta.

**Target state.** Both scripts import the path resolver (`aspects_json_path()`, or the Step 1 owner)
instead of hardcoding. They already `sys.path.insert(0, REPO)` and import six sibling probe modules,
so this costs nothing. Contract tests that read the seed (`tests/test_aspects_contract.py`,
`tests/test_absorbed_into_edge.py:60`) are **correct as-is** — locking the shipped baseline is their
job. Do not "fix" those.

**Files & call sites.** The four lines above, plus `docs/anchor-viz-prototypes/build_mind.py:16`
(hardcodes an absolute `/Users/tpac/...` path — prototype, same fix, low priority).

**Verification.** Re-run whichever walker arm last produced a quoted number and diff it. **Expect
the numbers to move** — that is the point, and any conclusion drawn from the `corr` operator before
this fix should be re-derived rather than trusted.

**Blast radius.** Eval-only, no production code. But it invalidates prior `corr`-operator readings,
so it needs a dated note wherever those numbers are cited.

**Depends on.** None. Independent and urgent.

**Respects.** Settled #6 (the dashboard's duplication stays; this is about eval scripts, which have
no disconnection contract and can import freely).

---

## Step 1 — Make `AspectRegistry` the single write door — **SHIPPED 2026-07-27 (`9381593`)**

**Landed as planned with one deviation:** the boot reconcile runs inside the registry
constructor (public as `brain.aspects.reconcile_with_seed()`), not as a separate
`brain.py` call — every construction site (daemon, IsolatedBrain, test brains, fresh
eval brains) materializes correctly by default, and a construct-then-reconcile split
would log spurious empty-registry warnings on every fresh boot. Also folded in from
the deferred list: atomic first-boot seed copy + corrupt-working-copy
quarantine-and-reseed. `invalidate()`/`_refresh_if_dirty()` deleted, not wired.

**Problem.** The registry is the read object for `aspects_v1.json` and writes nothing. Both writers
reach around it: `AspectEncoder._write_aspects` (`aspect_encoder.py:381`) and
`ensure_aspects_user_copy` (`aspect_contract.py:73`) — the latter called *from inside*
`AspectRegistry._load` (`aspects.py:202`), so the documented read path writes the file. A third
writer is a human hand-editing it.

Every known defect on this surface is downstream of that inversion:
- `invalidate()` exists because the writer cannot update the in-memory maps → **zero callers** → the
  daemon serves a boot-time snapshot for its whole process life. Relations classified during a 3am S2
  cycle are invisible to `correction_enrich` until the next restart.
- `_refresh_if_dirty()` is threaded through 14 read paths and can never fire.
- There is no write-boundary validation because there is no write boundary.
- The read-modify-write race needed an 8-line "accepted, narrow" comment
  (`aspect_contract.py:161-166`) that exists *only* because two writers share a file with no lock.
- The same atomic temp-file + `os.replace` block is hand-rolled **three times** across two files
  (`aspect_contract.py:182-195`, `aspect_encoder.py:385-396`, `aspect_encoder.py:413-417`), with
  *different* `json.dump` kwargs. Observable artifact: the shipped seed contains 49 `—` escapes
  **and** literal em-dashes, interleaved — both writers' output in one file.
- A fourth writer (the designed-but-unbuilt provenance sidecar, `id:bc31c184`) has no home, which is
  why it was refused.

**Target state.** One public door on the registry:

```python
brain.aspects.add_members(classifications, source=...)   # what the encoder calls
brain.aspects.reconcile_with_seed()                      # what boot calls
```

Both route through one private path that validates (Step 3), writes atomically, and re-derives the
in-memory maps from the dict just written. Then **`invalidate()` and `_refresh_if_dirty()` get
deleted, not wired** — the staleness they were built to manage cannot occur when the writer owns the
cache. A grep for writes to `aspects_v1.json` returns exactly one file.

Byte-level helpers (`_atomic_json_write`, path resolution, `SEED_ASPECTS_JSON_PATH`) move to a small
core module `servers/aspect_store.py`; the registry uses it. That keeps "touches bytes" separate from
"interprets contents" while leaving one public door.

**The import cycle is the seam.** `aspects.py:190` and `aspect_contract.py:119` both carry
function-local imports commented "avoid import cycle" — the cycle *is* the placement error
announcing itself. Moving path resolution and `REQUIRED_ASPECTS` into core makes the dependency
one-directional (S2 → core) and both imports become module-level. Re-export `REQUIRED_ASPECTS` from
`servers/aspects.py` for its 5 existing importers.

**Files & call sites.** New `servers/aspect_store.py`. `servers/aspects.py` (registry gains the
door; `_load` reverts to a pure read). `servers/scales/s2/aspect_contract.py` — loses ~168 of its 207
lines, ending at ~40 lines of pure config, the shape `healer_contract.py` already has.
`aspect_encoder.py:370-396` (`_load_aspects`/`_write_aspects` become door calls).
`aspect_decoder.py:116-133` (`_load_classified_strings` becomes a registry call — removes a third
production loader). `servers/brain.py:333` (call `reconcile_with_seed()` explicitly before
constructing the registry, adjacent to `seed_interactions` at `:302`, so boot materialization is
visible in the init sequence where its two siblings already are).

**Verification.** `tests/test_aspects_contract.py`, `tests/test_aspects.py`,
`tests/test_aspect_registry_wired.py`, `tests/test_absorbed_into_edge.py` (the self-heal tests),
`tests/test_aspects_path_isolation.py` (**critical** — it guards the live file against test writes,
by content and mtime), `tests/test_prompt_sync.py`. Full suite before merge: the taxonomy feeds
Frame, surface, and every S2 unit.

**Blast radius.** Large but mechanical — ~10 files, mostly moves. The risk is
`test_aspects_path_isolation`: if the door resolves paths differently from `aspects_json_path()`,
tests start writing the operator's live taxonomy. Land the path move first and confirm that test
green before touching the writers.

**Respects.** Settled #1 (relocates file plumbing, not data). Settled #5 (the heal stays
additive-only — `reconcile_with_seed` is the same logic behind the door). Brain `id:fdc6991a` — this
is the ownership decision the `fcntl` patch was deferred for; locking, if still needed, now has one
place to live.

---

## Step 2 — Collapse two registry constructors into one

**Problem.** `_load` (`aspects.py:262`) builds reverse-lookup maps with `setdefault` — **first**
claimant wins. `from_dict` (`aspects.py:534`) uses plain assignment — **last** claimant wins. Two
constructors of one object, disagreeing on a documented contract (CLAUDE.md: "Reverse lookups return
the FIRST claimant in JSON order").

Measured on the current seed: **29 strings resolve differently**, including `supersedes`
(`correction_improvement` vs `hierarchical_structure` — the exact verb from the 2026-07-24 recall
bug), `absorbed_into`, and **all 12** `wisdom` node types (`insight`, `principle`, `vision`, …),
because `wisdom` is appended last in JSON order by design.

All 14 `from_dict` call sites are tests. So `tests/test_aspects_contract.py:261`
`test_reverse_lookup_resolves_deterministically` documents first-claimant, exercises the constructor
that violates it, and asserts only `assertIsNotNone` — it cannot catch the disagreement. Production
has zero `by_node_type`/`by_edge_relation` callers today (3 eval files only), so live blast radius is
nil — but the tests are validating the wrong semantics, and the moment a consumer adopts the reverse
lookup that becomes a real bug with green tests.

**Target state.** One `_adopt(data)` body builds aspects + indexes. `_load` becomes
`self._adopt(load_taxonomy())`; `from_dict` becomes `instance._adopt(data)`. Deletes ~25 duplicated
lines and makes the disagreement unrepresentable. Then strengthen
`test_reverse_lookup_resolves_deterministically` to pin an actual multi-homed verb's primary.

**Files & call sites.** `servers/aspects.py:200-266` and `:508-538`.
`tests/test_aspects_contract.py:261`.

**Verification.** `tests/test_aspects.py`, `tests/test_aspects_contract.py`,
`tests/test_aspect_registry_wired.py`. The new assertion should fail before the fix and pass after.

**Blast radius.** Small. Tests are the only `from_dict` consumers.

**Depends on.** Step 1 (both constructors should route through the store's reader).

**Respects.** Settled #2 — `supersedes` stays multi-homed; this fixes which aspect is reported
*primary*, not membership.

---

## Step 3 — Validate at the write door, not at process start

**Problem.** Validation is bound to a process lifecycle event instead of to data change. Five
enforcement points with overlapping rules and none covering the whole:
`tests/test_aspects_contract.py` pins 12 invariants **against the seed only**;
`AspectRegistry._validate` (`aspects.py:268-309`) re-implements 2 of those 12 against the working
copy and only *logs*; `AspectEncoder._validate_classifications` checks category legality per
classification; `ensure_aspects_user_copy` has type guards; `_load` silently coerces a malformed
`node_types` to `[]` with no log at all. `from_dict` skips validation entirely.

The 2026-07-24 `supersedes` defect is prevented on the seed and nowhere else. The working copy passes
all 12 invariants today — by the encoder's per-classification strip, not by enforcement. Nothing
would catch a hand-edit breaking noise-exclusivity; `_validate` logs it and the daemon keeps the
broken snapshot for its process life. This is the repo's own rule (`feedback_loud_at_write_boundary`)
inverted: loud checks belong where every gap trips them, not at one bypassable moment.

**Target state.** `validate_taxonomy(data) -> list[Violation]` — the full invariant set, once.
Called by the Step 1 door as a pre-write gate that **refuses** the write, by `_validate` (delete its
two reimplementations), and by the seed contract test, which becomes
`assert validate_taxonomy(load_taxonomy('seed')) == []`. Remove the silent shape coercion in `_load`
in favour of a loud violation.

**Files & call sites.** New function in `servers/aspect_store.py`. `servers/aspects.py:268-309`,
`:244-247`. `aspect_encoder.py:253-352` (keep the per-classification filter — it correctly drops a
bad member and keeps survivors; add the whole-file gate at the door).
`tests/test_aspects_contract.py` (the 12 invariants become one call).

**Verification.** `tests/test_aspects_contract.py`; add a test that a taxonomy breaking
noise-exclusivity is **refused** at the door rather than written and logged.

**Blast radius.** Moderate. A refusing gate can block an S2 cycle's write — that is the intent, but
it must log loudly and leave the prior file intact rather than half-writing.

**Depends on.** Step 1 (needs the door to gate).

**Respects.** Settled #5. The `feedback_loud_at_write_boundary` rule directly.

---

## Step 4 — Move per-aspect facts into the JSON entry

**Problem.** The JSON is the declared source of truth for aspect membership, but four per-aspect
*facts* live in Python name-keyed literals, and the aspect name list is enumerated **six** times:

| copy | count | test-pinned? |
|---|---|---|
| `REQUIRED_ASPECTS` (`aspects.py:37`) | 16 | yes |
| `aspects_v1.json` keys | 16 | yes |
| `ASPECT_ACCEPTS` (`aspect_encoder.py:34`) | 15 | **no** |
| `order` (`aspect_encoder.py:188`) | 15 | **no** |
| `EDGE_ASPECT_PROMPT_SKIP` (`aspects.py:66`) | 3 | freezes the duplication |
| `LINEAGE_FAMILIES` (`surface_contract.py:1437`) | 4 | partial |

`ASPECT_ACCEPTS` fuses two unrelated facts: *which categories an aspect is defined over* (a property
of the aspect, same kind as `dimension`/`locked`, which live in the JSON) and *whether the LLM may
route to it* (S2 policy — `survivor_lineage` is absent for the second reason, not the first).
`order` is literally `list(ASPECT_ACCEPTS)`, re-typed by hand 140 lines away.

CLAUDE.md advertises adding an aspect as "a deliberate human edit to the JSON." Do that today and
`_validate_classifications:296` **rejects every classification routed to the new aspect, forever**,
with a log line about encoder hallucination — while every contract test passes. Miss `order` instead
and the aspect is silently never offered to the classifier. Both failure modes are silent and both
look like "the classifier just doesn't route to it."

This class has already fired in production: `surface_contract.py:1428-1435` records that
`LINEAGE_FAMILIES` held **five aspect names that no longer existed**, so `dependency_flow` and the
supersedes lineage silently stopped riding along in spread activation until 2026-06-08. That comment
names the correct fix and parks it: *"The drift-proof end state is a first-class structural flag on
the aspect itself."* The precedent is already shipped — `metadata.display_label` is a per-aspect fact
that lives in the JSON entry and is contract-tested.

**Target state.** Per-aspect fields in `aspects_v1.json` beside `locked`/`dimension`/`metadata`:
`accepts: ["node_types"|"edge_relations"]`, `routable: true|false`, `prompt_visible: true|false`,
`structural_lineage: true|false`. `Aspect` grows the fields. `ASPECT_ACCEPTS`, `order`,
`EDGE_ASPECT_PROMPT_SKIP`, `LINEAGE_FAMILIES` are all **deleted** and derived from
`brain.aspects.all()`. Python keeps only `REQUIRED_ASPECTS` (a presence contract, not a fact table).
Adding an aspect becomes one JSON edit plus one `REQUIRED_ASPECTS` line, both test-pinned, and
cannot half-land. The ASPECT MENU header count derives instead of drifting — it is currently stated
as **14** in four files, **15** in one, and **16** in two.

**Files & call sites.** `servers/scales/s2/aspects_v1.json` (all 16 entries),
`servers/aspects.py:37,66` + the `Aspect` dataclass, `aspect_encoder.py:34-50,181,188-196`,
`surface_contract.py:1437`, `tests/test_prompt_closers.py:65` (currently asserts the frozen
duplication — must flip to asserting derivation).

**Verification.** `tests/test_aspects_contract.py::TestSeedShape` (add the new required keys),
`tests/test_prompt_closers.py`, `tests/test_aspect_registry_wired.py`. Add a test that every aspect
declares a non-empty `accepts ⊆ {node_types, edge_relations}` — that assertion is what makes a
JSON-only aspect addition safe.

**Blast radius.** Schema change to all 16 entries, so the Step 1 reconcile and `TestSeedShape` both
need the new key. Not free. Highest legibility payoff on the list: it moves K out of six scattered
Python literals into the file the maintainer is already told is authoritative.

**Depends on.** None strictly, but landing after Step 1 avoids editing files mid-move.

**Respects.** Settled #1 — schema change, not relocation. Explicitly allowed.

---

## Step 5 — Members become records: provenance and retirement in one change

**Problem.** A member is a bare string in a list. `_merge_into_aspects` appends and moves on;
`rationale` is captured in `aspects_proposed.json`, which is **overwritten every cycle**. So a
classification is permanent, evidence-free, and unrevisitable — and with `min_count_threshold = 1`,
essentially every future classification is a single-example permanent decision whose reasoning
evaporates within the hour.

Two consequences measurable right now:

1. **The working copy has diverged into a different artifact than the seed** — roughly 400+ members
   the seed lacks, zero seed-only entries. The seed is not a baseline; it is a stale minority. The
   reconcile is one-way and additive, so the seed can never correct a bad classification.
2. **`noise.edge_relations` contains `temporal_sequence`, `extension_refinement`,
   `validation_evidence` — three *aspect names* filed as relation verbs.** The file cannot tell you
   whether real edges carry those literal strings or the classifier echoed the menu it was shown —
   which is the provenance gap in miniature. **The brain can, and did:** node `id:40e7125a` audited
   this a month ago and confirmed no real edge uses those strings as relation values, so it is
   harmless today and a one-line cleanup. Keep it as the worked example of *why* records matter —
   answering it required a human-run audit that a `count_at` field would have answered for free —
   but do not treat it as an open question, and fix the three entries whenever this step lands.

And because the heal is additive-only *and* a member is a bare string, a manual retirement is
reverted on the next boot with no way to express intent. Removal is not awkward — it is
**unrepresentable in the protocol**.

**Target state.** Members become records in the same file:

```json
"edge_relations": [
  {"name": "corrects", "by": "seed",                  "at": "2026-05-08", "count_at": 0},
  {"name": "revises",  "by": "s2:aspect_integration", "at": "2026-06-14", "count_at": 7,
   "rationale": "...", "retired": false}
]
```

The registry normalizes both shapes on load (bare string → `{name}`), so consumers keep receiving
tuples of strings and **no consumer changes**. This lands four things at once: provenance with **no
new writer** (the Step 1 door stamps it); `retired: true` makes retirement an *additive* fact, so the
additive-only reconcile propagates it correctly and the unwinnable-removal defect dissolves; the
seed-vs-working divergence becomes legible (`by: "seed"` vs `by: "s2:aspect_integration"`); and a
future audit pass can rank suspect classifications by `count_at` — which is how you would catch the
three aspect-names-in-noise above.

This supersedes the sidecar design in brain node `id:bc31c184`. Revise that node when this lands.

**Files & call sites.** `servers/scales/s2/aspects_v1.json`, the `Aspect` dataclass +
normalization in `servers/aspects.py`, `aspect_encoder.py:354-366` (`_merge_into_aspects` stamps
records), the Step 1 door, `aspect_encoder.py:410-421` (`_write_audit_trail` may become redundant).

**Verification.** `tests/test_aspects_contract.py`, `tests/test_aspects.py`. Add: a bare-string
member normalizes to a record; a `retired: true` member is excluded from `relations_in`/`types_in`;
the reconcile propagates a retirement rather than reverting it. **Then** re-run the full suite —
every consumer reads through the normalizer.

**Blast radius.** Largest schema change here. Mitigated by normalizing on load so no consumer
changes, but the normalizer is now on every read path — verify it is not a hot-path cost
(`correction_enrich` runs on every recall pull).

**Depends on.** Step 1 (the door stamps records). Step 4 is a natural companion — same file, same
schema-change review.

**Respects.** Settled #1 (schema, not location) and #5 (retirement becomes additive, so
additive-only stays correct — it *widens* what additive can express).

---

## Step 6 — Name the second concept inside `noise`; unify nine exclusion literals

**Problem.** Nine independent hardcoded "noise exclusion" sets coexist with
`relations_in(['noise'])`, and no two agree — cardinalities 1, 2, 2, 2, 3, 5, 7 against a live
taxonomy of 10. Sites: `dal_graph.py:65`, `pipeline_contract.py:412` (a byte-identical duplicate of
the previous under a different name in a different module), `brain_recall.py:340,344`,
`brain_constants.py:309`, `community_contract.py:85,146`, `community_decoder.py:1289`. The most
legible instance: `community_decoder.py:222` reads the set from the registry, and **the same file**
at `:1289` inlines it as a SQL literal, 1,000 lines apart.

**This is not lazy duplication, and that is the finding.** `noise` holds two different kinds of
member: **code-owned plumbing** (`co_accessed`, `emergent_bridge` — written by `recall_write_queue`,
defined in `brain_constants.EDGE_TYPES`, fixed by code) and **classifier verdicts**
(`community_member`, `dreamed_from`, `member`, `co_member`, `test_marker`, and the three
aspect-names — judgments that grow and can be wrong). The literals contain only the plumbing subset.
So the two sets were never supposed to be equal, which is why nobody unified them — and why
`dal_graph.py:63-64` states in a comment that `community_member` is **NOT** in its default because
it is "real thematic context," while `aspects_v1.json` files `community_member` under `noise`. The
DAL and the taxonomy openly contradict each other about one string, and the only way to learn that
is to read both.

**The trap:** a maintainer told to "source the noise set from the taxonomy" will do it, silently pull
`community_member` (7,237 edges) into `DEFAULT_EXCLUDED_RELATIONS`, and drop community context out
of every `get_connections_bulk` read. `consolidation_decoder.py:783-786` depends on the current
behavior.

**Target state.** Name the second concept — a non-routable `structural_plumbing` aspect
(`routable: false` per Step 4, following the shipped `survivor_lineage` pattern), or derive the
plumbing set from `EDGE_TYPES` keys flagged system-written. Then three named policies replace nine
literals, each a registry union minus an explicitly named carve-out:

- `structural_exclusions(brain)` = `relations_in(['noise']) − {community_member}` → replaces
  `dal_graph.py:65`, `pipeline_contract.py:412`, `brain_recall.py:340,344`,
  `brain_constants.py:309`, and Step 0A's whitelist
- `cohesion_exclusions(brain)` = `relations_in(['noise','generic_relation'])` → replaces
  `community_contract.py:85`, `community_decoder.py:1289`
- `adjacency_exclusions(brain)` → replaces `community_contract.py:146`

Put the `community_member` carve-out in the JSON as aspect metadata, not a code literal, so the one
real policy decision lives in the config file and the accessors stay pure derivation. Model it on
`ABSORB_EXCLUDED_RELATIONS` (`dal_graph.py:82`) — one deliberate, documented, tested carve-out.

Residual migration delta beyond `community_member` is small and measured: `dreamed_from` 20 +
`dream_observation` 19 + `temporal_sequence` 9 + `member` 8 + `co_member` 3 = **59 edges**.

**Files & call sites.** The nine sites above, plus a home for the three accessors (`servers/aspects.py`
or `pipeline_contract.py`). `dashboard/queries/encoding.py:64,331` and `s2_runs.py:153` are forced by
the disconnection contract — **leave them.**

**Verification.** `tests/test_community_detection.py`, `tests/test_consolidation*.py`,
`tests/integration/test_recall_pipeline.py`, `tests/test_raw_sql_guardrail.py`. Full suite — this
touches the DAL default and the recall path.

**Blast radius.** Widest of any step; nine call sites across four subsystems, each with a slightly
different current set. Land it as one step so the sets converge together rather than drifting mid-migration.

**Depends on.** Step 4 (the flags mechanism). Step 0A should land first and can inline its exclusion
until this arrives.

**Respects.** Settled #6. Preserves the deliberate `community_member` carve-out rather than
"unifying" it away.

---

## Step 7 — Delete the shadow taxonomies

**Problem.** Three consumers hold private forks of taxonomy data:

| file | literal | vs. live |
|---|---|---|
| `consolidation_decoder.py:836` | `{corrects, corrected_by, supersedes, superseded_by}` | **4 of 58** |
| `consolidation_decoder.py:847` | `{contradicts, challenges, conflicts_with, contrasts, undermines, violates}` | **6 of 32** |
| `consolidation_contract.py:131` | `suppression_relations = {similar_to, consolidated_into, corrects, supersedes}` | **never reads the registry at all** |

The first two read the registry correctly, then fall back to a literal `if not correction_rels` — a
branch that fires exactly when the registry loaded empty, i.e. the corrupt-file scenario. When it
fires it silently narrows the correction walk to ~7% of its real membership and lets consolidation
merge nodes that already corrected each other. A registry failure that should be loud becomes a quiet
6× narrowing. The `contradiction_conflict` fallback degrades the "NEVER consolidate a tension" guard.

`suppression_relations` is worse and has a live consequence: `resolves` (455 edges), `addresses`
(240), `reframes` (101), `updates`, `fixes`, `revises`, `overrides`, `absorbed_into` are all live
`correction_improvement` members that **will never suppress** — so pairs the encoder resolved with
those verbs re-propose every cycle, which is the churn the comment above the constant exists to
prevent.

**Target state.** Delete both fallbacks — a corrupt taxonomy should fail loudly, not quietly
consolidate corrected pairs (Step 10 makes the degraded mode explicit). Point `suppression_relations`
at `relations_in(['correction_improvement'])` plus its genuinely-local additions (`similar_to` and
`consolidated_into` are consolidation's own outcome markers — legitimately local).

**Files & call sites.** `servers/scales/s2/consolidation_decoder.py:834-838,846-850`,
`servers/scales/s2/consolidation_contract.py:131`.

**Verification.** `tests/test_consolidation_fingerprint.py`, `tests/test_absorbed_into_edge.py`,
`tests/test_aspects_contract.py`. Add a test that a pair resolved via `resolves` does not re-propose.

**Blast radius.** Changes which clusters consolidation considers settled — expect *fewer*
re-proposals, which is the fix. Watch one S2 cycle's journal after landing.

**Depends on.** None. **COLLISION:** overlaps journal-audit finding #4 (`id:2ace76f3`), owned by
another session. Coordinate before starting.

**Respects.** Settled #2.

---

## Step 8 — Make `relations_in` / `types_in` raise on an unknown aspect name

**Problem.** `aspects.py:378-402` does `a = self._aspects.get(n); if not a: continue` — a typo or a
renamed aspect returns `()`, indistinguishable from "that aspect is empty." The single-name form
(`__getattr__`) raises `AspectContractError`; the union form fails silently. Every caller passes a
code-level literal, never user input.

This is the 2026-06-08 bug's mechanism: 5 of 8 hardcoded family names no longer existed, so
`dependency_flow` and the supersedes lineage silently stopped riding along in spread activation.
`LINEAGE_FAMILIES` now has a contract test, but that covers one of six call sites and does not
generalize.

**Target state.** Raise `AspectContractError` on an unknown name, matching `__getattr__`. Callers
with genuinely optional names use `by_name()`, which already returns `Optional`. Replaces a
per-caller test with a structural guarantee.

**Files & call sites.** `servers/aspects.py:378-402`. Then audit the six union call sites:
`surface_contract.py:1437`, `community_contract.py`, `encode_contract.py:165`,
`brain_remember.py:228`, `aspects.py:66`, `consolidation_decoder.py`.

**Verification.** `tests/test_aspects.py`, `tests/test_aspects_contract.py`; add a test that an
unknown name raises. Full suite — a latent typo anywhere becomes a loud failure, which is the point,
but it will surface at import/first-call rather than silently.

**Blast radius.** Small if the six sites are clean, loud if not — either outcome is the desired one.

**Depends on.** None. Best after Step 4 (which deletes several of those literals outright).

**Respects.** The Loud-by-Default principle.

---

## Step 9 — Delete the dead surface and correct the module docstring

**Problem.** `servers/aspects.py:18-23` advertises five consumer groups. Checked against production:
`list_aspects` (MCP tool) **does not exist** — the only two mentions in the repo are inside
`aspects.py` itself; `filter_nodes(field='aspect')` and `recall(filter={'aspect':...})` have no
implementation; the Healer's `metadata.display_label` usage was removed (`healer_contract.py:41-47`
documents it, and 9 aspects carry `display_label` with zero readers); `relation_meaning_map()` has 0
production hits (its sole consumer was removed by `c7538a2`).

Zero-production-consumer surface: `invalidate`, `_refresh_if_dirty`, `by_dimension`, `dimensions`
(only one dimension value exists), `all_with_counts`, `required`, `emergent`, `relation_meaning_map`,
`type_meaning_map`, and four `Aspect` predicates — roughly 55 lines.

CLAUDE.md repeats the same three false MCP claims.

**Cost, and why it is on this list at all.** This is the legibility tax, and it is the plausible root
cause of the misread that started this review: the `_dirty`/`invalidate` machinery *describes* a
coherent refresh protocol, so it reads as live. A maintainer reasoning about staleness from this
docstring is reasoning about a system that is not there.

**Target state.** Rewrite the Consumers block from an actual grep (Frame via `by_name('wisdom')`; S2
via `relations_in`/`types_in`; `correction_enrich` via the `correction_improvement` member list;
`base._inject_edge_aspects` via `all()`). Fix the CLAUDE.md claims. For the dead methods: **do not
delete blind** — brain node history warns that a zero-caller method is sometimes a migration target,
and `all_with_counts` was built for the unbuilt `list_aspects` tool. Either wire `list_aspects` (the
"measure what you built" rule applies) or annotate `# test-only surface`. `invalidate` and
`_refresh_if_dirty` are the exception: Step 1 makes them genuinely deletable.

Also relocate `compose_edge_text` (`aspects.py:463`) — it touches no `self`, its audience is the
edge-embedding pipeline (`brain_connections.py:85`, `surface_contract.py:1081,1838`), and
`tests/test_spread_activation.py:104` already calls it as
`AspectRegistry.compose_edge_text(object(), ...)` to prove the taxonomy is not involved. Its
residency costs `brain.py:325-333` an 8-line comment explaining an init-ordering constraint that a
pure string formatter should not impose, and it has already caused a false claim at
`surface_contract.py:1714` ("family meaning is composed into the enriched text… contributes
automatically") which is provably false against the function's own contract. Move to a small
`servers/edge_text.py` (repo idiom: `loud_truncation.py` is 31 lines) or `pipeline_contract.py`, keep
a delegating shim for one pass, delete the stale claim.

**Files & call sites.** `servers/aspects.py:18-23,311-321,411-450,454-504,463-495`, `CLAUDE.md`,
`servers/brain.py:325-333`, `servers/scales/s1/surface_contract.py:1714`, plus the 4 production +
1 script + 2 test call sites of `compose_edge_text`.

**Verification.** `tests/test_aspects.py`, `tests/test_spread_activation.py`,
`tests/test_contract_sync.py`, `tests/test_prompt_closers.py`.

**Blast radius.** Small, mostly docs and a move. The `compose_edge_text` relocation touches the edge
write path — verify `tests/test_spread_activation.py` and one live `connect` round-trip.

**Depends on.** None (the `invalidate` deletion half depends on Step 1).

---

## Step 10 — Decide what a failed registry does, and stop swallowing it

**Problem.** `brain.py:333-340` wraps registry construction in `try/except`, prints a warning, and
leaves `self.aspects` **unset** — deliberately, so consumers get a loud `AttributeError`. Two
consumers then defeat that: `fetch_tools.py:729` guards with `hasattr(brain, 'aspects')`, and
`encode_contract._filter_noise_relations` (`:163-166`) catches `AttributeError` and returns
unfiltered. Both are commented as test-stub tolerance.

So a real registry failure degrades to "aspect lookup returns None" and "the S1 encoder sees
unfiltered plumbing edges and learns to imitate them" — with no error anywhere. The intended
loudness is designed in at `brain.py` and cancelled downstream.

**Target state.** Pick one and make it consistent. Either assign an empty registry on failure (then
the `AttributeError` contract is real and both guards can go), or make the test stubs carry a real
`AspectRegistry.from_dict(...)`, which exists for exactly this purpose. The second is better: it
removes the production tolerance rather than legitimizing it.

**Files & call sites.** `servers/brain.py:333-340`, `servers/scales/s1/fetch_tools.py:729`,
`servers/scales/s1/encode_contract.py:163-166`, plus whichever test stubs need a real registry.

**Verification.** `tests/test_s1_data_assembly.py`, `tests/test_fetch_tools.py`,
`tests/test_aspect_registry_wired.py`. Add a test that a failed registry surfaces rather than
degrading silently.

**Blast radius.** Small. Makes a currently-silent failure loud, which may surface pre-existing
test-stub shortcuts.

**Depends on.** None. Step 7's fallback deletion is safer once this lands.

---

## Dropped — checked against settled constraints and rejected

- **Moving the taxonomy into `brain.db` (as nodes or a table).** Every angle was told this was
  settled and none proposed it. Recorded here because it is the obvious move for anyone who sees
  "a file behaving like a database" and it was deliberately decided against (`id:9edc4cf5`,
  principle `id:ec405b7c`).
- **A per-unit last-run gate for the aspect decoder.** The classified-string filter is the rest gate.
  Settled #4.
- **Adding `fcntl` locking to the two current writers.** Superseded by Step 1 — with one writer, the
  lock either becomes trivial or unnecessary (`id:fdc6991a`).
- **The provenance sidecar as its own file** (`id:bc31c184`). Superseded by Step 5, which achieves it
  with no new writer.
- **"Unify all the noise-exclusion literals against `relations_in(['noise'])`."** Rejected as stated
  — it would silently drop `community_member` (7,237 edges) out of `get_connections_bulk`. Step 6 is
  the version that survives.
- **Splitting `render_edge_aspects_block` out of `servers/aspects.py`.** Two angles examined it; it
  mirrors `render_journal_review_block` in `trace_contract.py` — the vocabulary owner holds its own
  prompt block, `base.py` holds the injection. Convention, deliberately resolved this way. Not a
  finding.
- **`aspect_decoder`'s raw SELECTs.** All four S2 decoders do this and
  `tests/test_raw_sql_guardrail.py` is DML-only by design. House pattern.
- **Call-time path resolution via env vars.** Correct — it fixed a real IsolatedBrain heal-leak
  (2026-06-16) and is pinned by `tests/test_aspects_path_isolation.py`. `IsolatedBrain` and
  `BrainTestBase` must redirect *after* import, which no import-time constant supports.

## Small deferred items — real, low-cost, would otherwise evaporate

These came out of the pre-commit review of the 2026-07-25 aspect fix and are too small to be steps,
but each is a genuine defect. Fold them into whichever step touches the same file.

- **`ensure_aspects_user_copy` can break noise-exclusivity.** It is now a second writer of semantic
  member lists and the only one that doesn't enforce `noise ∩ semantic = ∅` (the invariant
  `_validate` logs an error on, per `id:4d190406`). Latent today — verified no overlap in either the
  seed or the working copy — but the live `noise` already carries classifier-grown members the seed
  lacks (`test_marker`, `system_note`, `co_member`, `member`), so a future curated seed edit filing
  one of those under a semantic aspect would re-break it every boot with nothing to repair it. Step 3
  (validate at the door) covers this for free.
- **First-boot seed copy is not atomic.** `aspect_contract.py:114` uses `shutil.copy2` while the heal
  write one screen below goes to real trouble with tempfile + `os.replace` and documents why. Two
  processes constructing a Brain concurrently on first boot can leave a partial file, after which
  every subsequent boot takes the `except (OSError, json.JSONDecodeError): return False` path and
  never re-seeds — permanently the empty-registry state the function's own comment calls
  catastrophic. Fix: copy into a tempfile + `os.replace`, and on `JSONDecodeError` move the corrupt
  file aside and re-seed rather than returning False. Belongs with Step 1.
- **`_render_py`'s `constant` parameter is accepted and never used** (`sync_prompts.py`) — the
  template hardcodes `SYSTEM_PROMPT`. A `SEED_PROMPTS` entry declaring any other constant name would
  be permanently out of sync and have its constant renamed on every sync. Latent; one-line fix.
- **Three aspect names sitting in `noise.edge_relations`** (`temporal_sequence`,
  `extension_refinement`, `validation_evidence`) — audited harmless (`id:40e7125a`), one-line
  cleanup, do it whenever Step 5 or Step 6 opens the JSON.
- **`run_aspect_cycles_on_clone.py --wipe-members` is broken since `2109376`** (pre-dates
  Step 1): the member-level seed heal re-heals wiped members back at Brain construction, so
  the "harder eval" starting state gets un-wiped. Default (keep-seeds) mode unaffected.
  Eval-script-only; fix is to make the script strip members AFTER Brain construction, or
  seed its work file from a non-seed baseline.

## Also noted, outside this boundary

`ASPECT['model'] = 'claude-sonnet-4-6'` and `max_tokens` in `aspect_contract.py:19-20` are **dead for
this unit**: `base._call_llm` takes both only from the interactions table (`base.py:572`), never
`self.config`. The sibling units chain to `self.config`, so the identical-looking key is live for
them and dead here, and a third hand-copy sits at `interaction_seed.py:179`. If the `s2_aspects` row
ever lacks `model`, classification silently downgrades to `claude-haiku-4-5` while the most
authoritative-looking source says Sonnet. Fix: drop the two keys from `ASPECT`, and make
`base._call_llm`'s hardcoded fallback log when the row carries no model.
