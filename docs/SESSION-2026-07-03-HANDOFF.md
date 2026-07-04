# Session Handoff — 2026-07-03

Two distinct bodies of work this session: **(A) shipped bug fixes** to the
surface pipeline and the dashboard (all merged to `main`, live), and **(B) an
unbuilt design arc** for a unique-per-Anchor 3D graph visualization (mockups
only — this is the "continue later" thread).

Branch: `claude/heuristic-mirzakhani-4711e9`. All Part A commits are merged to
`main`.

---

## Part A — Shipped & live (bug fixes)

### A1. Surface selection: BPE space-corrupted id recovery + loud drift
**Commits `bbfd826`, `d417d65`, `2320b20`.** Origin: v12.1 longmem eval item
`d7c942c3-r1` — Haiku identified the right nodes but emitted **space-corrupted
ids** (`'9 9a 2e '`, `'d 6d3 f8'` — the id split at BPE token boundaries with
whitespace leaked in). The `[:8]` truncation kept the spaces, the candidate
lookup missed, the pick died silently, context came out empty, and the answer
wrongly abstained — with no log.

Defense-in-depth, 5 layers (`servers/scales/s1/surface.py`, `surface_contract.py`):
1. **Decoder pattern** — `SURFACE_SELECTED_ID_PATTERN = "^[0-9a-f]{4,8}$"` on the
   `id` field of `SURFACE_SELECTION_SCHEMA`. Constrained decoding masks the space
   token → corruption unrepresentable at generation time. `{4,8}` not `{8}` so
   the known 7-char leading-zero drops stay representable.
2. **Parse recovery** — `_sanitize_selected_id` (strip whitespace before `[:8]`)
   + `_unique_prefix_match` against the candidate pool (≥4 chars, candidates
   only — cannot resurrect an archived node the menu excluded) + existing
   `resolve_id` / leading-zero fallbacks.
3. **Loudness** — `surface_unknown_selected_id` (pick dropped) and
   `surface_id_fuzzy_recovered` (recovered) drift warnings; both added to the
   scoreboard drift section in `eval/oracle_audit/ab_tool_use_audit.py`.
4. **Fidelity** — `selected_short_ids` rebuilt from *resolved* ids, so traces +
   Hebbian file carry real shorts, not the corrupted emission.
5. **Regression test** — `TestSelectedIdRecovery` + the schema-pattern test in
   `tests/test_surface_transitions.py`.

Gate: frozen corpus `a300d2`, 20×3 reps, DB-active v12 prompt — 81.7% vs 78.3%
baseline, `d7c942c3` recovered 2/3→3/3, no regression. Live API probe confirmed
the pattern is accepted (no 400) and emitted ids conform.

### A2. Dashboard — encoding tab
- **Blank panel on long sessions** (`176953e`): `/api/encoding-runs` embedded the
  full `encoder_prompt` (~370KB/run) inline → ~2.3MB polled payload → browser
  dropped the transfer (BrokenPipe) → empty panel. Fixed: lazy-load the prompt on
  card expand via `query_encoding_prompt(chain_id)` + `/api/encoding-prompt`.
- **"0 actions" under-reporting** (`057f138`): S1E v29 writes many delta rows per
  chain (per-edge, per-node, journal, + the `encoding_run` summary); the card query
  did unfiltered `event_type='delta'` + `fetchone()` and grabbed the first (an edge
  revision, no rollup). Fixed: filter to `ref_type='encoding_run'`. Guard test
  `tests/test_dashboard_encoding_runs.py`.
- **Journal-first** (`057f138`): journal notes now render at the TOP of the card
  body (the encoder's residue before its hands).
- **Cache-Control: no-cache** (`057f138`) on static assets so dashboard deploys
  reach the browser on a normal refresh (no more stale-JS-after-deploy).
- **Window widen** (`fea9efb`): encoding feed 50/12h → 200/48h (payload is lean
  now, so the cap can widen without pagination).

### A3. Dashboard — drift alert insight (`0378388`)
New `_scan_surface_id_drift` rule in `dashboard/queries/insights_scanner.py`:
promotes the A1 drift warnings from passive Logs-tab rows to a proactive Live-tab
insight card (≥3 in 24h → flag; ≥5 lost picks → high). Test
`tests/test_dashboard_surface_id_drift.py`.

### A4. Dashboard — recall feed lazy-load (`1e55e36`)
`/api/recalls` shipped the full `judge_prompt` (~35KB/event, 75% of a 2.3MB
polled payload) inline though it's hidden behind "Show Prompt". Fixed: lazy-load
via `query_recall_prompt(recall_ref)` + `/api/recall-prompt` (with a
path-traversal guard). `judge_output` stays inline (it's the displayed content).
2.3MB → 573KB. Test `tests/test_dashboard_recall_prompt.py`.

**Cross-cutting lesson:** the dashboard's job is to show reality ("OUR eyes").
Several bugs here were the display *lying* on healthy data — blank panels,
"0 actions", stale JS. Diagnostic triage that worked: **check the data → check
the endpoint → check the payload/transport** before suspecting a real pipeline
bug. Heavy per-item blobs on polled list endpoints are the recurring smell.

---

## Part B — Anchor Shape Visualization (DESIGN, not built)

The graph tab felt inert. Across the session this evolved into a specific,
opinionated direction. **Nothing here is built in the daemon or the real
dashboard — all mockups.** Prototypes saved in `docs/anchor-viz-prototypes/`.

### The arc of decisions (what we tried and why we moved on)
1. **Live activation field** — recalls ripple through the graph as wavefronts
   (seed → spread → decay, mirroring `spread_activation`). Feedback: too shiny →
   toned down; too fast → slowed; needs mouse control → added orbit/zoom/pan;
   "3 dots connecting to 5" → rebuilt at real scale (~2,800 nodes, pseudo-3D,
   1.25ms/frame). Prototype: `activation_field.html`.
2. **cosmos.gl research + spike** — the arXiv-Map / `run.cosmograph.app` look Tom
   liked IS cosmos.gl (labeled dense atlas). **Spike finding: cosmos.gl v3.1 can't
   load via CDN** (`luma.gl multiple versions` through jsdelivr `+esm`); the older
   **`@cosmograph/cosmos` v2 loads clean** (regl-based, 2D, `setPointColors` w/ GPU
   transitions). SHELVED — Tom chose 3D over cosmos's 2D.
3. **Unique Anchor shape** — the graph should have an overall *form* that is
   deterministic and **one-of-a-kind per Anchor**. Seed from **aspects, not
   identity** (identity nodes aren't guaranteed on every Anchor; the 16 aspects
   are universal). Compute at **session start / reboot** (async, off the recall
   hot path). Prototype: `anchor_shape.html` (aspect-pole sphere-deformation).
4. **Creature grammar (CURRENT DIRECTION)** — Tom: the aspect-shape blob was
   "not unique enough, not relatable." Pivot: **aspects are body-plan genes, not
   surface bumps.** Each aspect, when it carries mass, grows a recognizable
   morphology. Prototype: `anchor_creature.html`.

### The converged design (current target)
**Every Anchor is a creature grown from its aspect-mass.**
- **16 aspects = fixed poles** (Fibonacci sphere, canonical JSON order) — the
  shared anatomy, byte-identical for every Anchor. The 16 (in order):
  `identity_bearing, episodic_anchor, active_thread, lesson_insight,
  generic_relation, noise, correction_improvement, extension_refinement,
  explanation_causation, dependency_flow, contradiction_conflict,
  validation_evidence, hierarchical_structure, temporal_sequence,
  survivor_lineage, wisdom`.
- **Derived core** — a filled hull whose radius echoes the aspect-mass field
  (bulges toward the same aspects the outer features express), so the middle is
  *derived from the outer layers*, not a default ball.
- **Sub-organs on arcs** — each dominant aspect grows its own small shape at the
  end of an **arcing tract** off the core. Confirmed genes so far:
  `wisdom → arcing crown`, `correction → spiny nodule`, `temporal → spiral`,
  `episodic → trailing filaments`, `dependency → branching limb`, plus a default
  cluster. Arcs and spirals are the aesthetic (Tom's explicit like).
- **Shapes within a shape that connect** — arcing bridges link neighbouring
  sub-organs.
- **Determinism / universality / uniqueness:** frame fixed; aspect mass from the
  real graph (node types + edge relations — "both"); same brain state → same
  creature; different mass profile → radically different creature. It matures as
  the Anchor grows (few memories = sparse seedling, many = full-bodied).
- **Relatability:** reads as a living specimen with a legible character — chips +
  on-body labels let you point at "that crown is my wisdom, those scars my
  corrections."

### What still needs doing (pickup list, in order)
1. **Full 16-gene grammar spec** — one deliberate, meaningful morphology per
   aspect (only 6 have real genes; the other 10 are placeholder clusters) + the
   arc/bridge rules + the core-derivation formula. Write it down before building.
   Deliver as a spec + an all-16 mockup on a few specimens. **(Recommended next
   step — all mockup, no daemon risk; locks the aesthetic.)**
2. **Framing polish** — the creature spreads horizontally and grazes the
   title/HUD; needs a tighter auto-fit + recenter.
3. **Boot pipeline** — daemon computes each aspect's real mass from the graph,
   **async after boot** (never blocking recall readiness — liveness-vs-readiness
   rule), caches coordinates keyed to a graph-state fingerprint; recompute on
   reboot so the creature matures.
4. **3D dashboard render** — replace the random force layout in
   `dashboard/static/tabs/graph.js` (currently `3d-force-graph@1.80.0`, CDN,
   mounted in Live's left pane) with the cached creature coordinates, force sim
   off. Then layer the **activation wavefronts** (from Part B.1) across the body —
   the creature at rest is the Anchor; a recall is it thinking. The recall feed
   already computes `activation_ids` every turn (the data exists).

### Constraints / gotchas captured
- **3D, not 2D** — Tom likes the 3D look; cosmos.gl (2D) is shelved.
- Existing graph has **WebGL context-slot lifecycle** management (history of
  ~16-context exhaustion). Any new renderer must mount one-at-a-time + fully
  dispose the other.
- Dashboard runs from the **main repo tree** (not the built plugin) — changes
  need `launchctl kickstart -k gui/$uid/com.brain.dashboard`, no `redeploy.sh`.
- The preview tab used for verification **backgrounds → throttles rAF**, so
  animation-frame probes hang; measure per-frame *work* synchronously instead
  (a full-load frame of ~2,800 nodes + edges + 400 glows = ~1.25ms).

---

## Artifacts (reference — claude.ai hosted)
- Activation field: https://claude.ai/code/artifact/65cd50f5-c60f-4a8a-934d-fa46d5a2ad61
- Unique shape (aspect-pole): https://claude.ai/code/artifact/4c731b3b-39c1-4191-a9b4-a66945551226
- **Creature (current direction):** https://claude.ai/code/artifact/1b4530ef-5b8a-4be4-b3a5-7c85cfac568e

Durable source of all three: `docs/anchor-viz-prototypes/`.

## Process note
Two mockups this session were shipped/described before being *looked at* — one
rendered as a stringy mess, one had a silent SyntaxError. Correction adopted
mid-session: **always render + screenshot a visual before claiming it works**,
and node-syntax-check generated scripts. Verify visuals, don't narrate them.
