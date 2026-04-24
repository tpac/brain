# S1 Architecture

*Working reference. Settled tonight (2026-04-20). Captures decisions,
vocabulary, and next-step POC so work survives session breaks.*

## Principles

1. **O/K/Δ recursive across scales.** S1 has multiple integration units,
   each running integrate(O, K) → Δ at turn-or-gate cadence. Fractal
   mirrors S2 — same shape, faster cadence.

2. **Decoder + Encoder pattern within each integration unit** (borrowed
   from S2). Decoder produces proposals (cheap, often algorithmic).
   Encoder reviews proposals and commits writes (agentic when needed).

3. **Providers feed decoders.** Reusable components — algorithmic (SQL,
   regex, graph traversal) or Haiku agents with focused prompts.
   Providers don't own an integrate function of their own; they
   contribute structured proposals to a decoder.

4. **Make Anchor stateful.** LLMs are natively forward (generating,
   continuing). They don't natively pause, consult memory, verify,
   or match against prior commitments. Each decoder/provider is a
   forced backward operation the forward-only LLM wouldn't do otherwise.

5. **Haiku-first for non-deterministic extraction.** If correctness
   requires semantic reading (entities, features, patterns, gaps),
   use a small Haiku call. Don't waste engineer time tuning
   algorithmic approximations (TF-IDF, regex) of semantic judgments —
   they fail in weird ways, and Haiku is cheap and correct by default.

6. **Encoder is a judge, not a Swiss-army knife.** Pre-digestion in
   providers. Encoder reads pre-chewed, categorized proposals and
   decides what to commit. Leaner prompt, sharper decisions.
   Provider outputs replace encoder-prompt directives; as more
   providers land, the encoder prompt shrinks.

## Vocabulary

- **Integration Unit** — one O/K/Δ loop at a scale. Has a decoder phase
  and an encoder phase. S2 already runs 4 units (edge_families,
  consolidation, community, healer).
- **Decoder** — the proposal-producing phase. Reads O, emits structured
  proposals. Can be mostly algorithmic, orchestrates providers.
- **Encoder** — the proposal-commit phase. Reviews proposals, selects,
  commits writes.
- **Provider** — reusable sub-component that contributes proposals.
  Algorithmic or Haiku. Cross-integration, cross-scale — same
  `rules_applicable` provider can feed S1Surface AND S1Gate.
- **Proposal** — structured datum passed decoder → encoder. Standard
  shape: `{type, content, confidence, source_provider, evidence}`.
  Traceable — every commit has a provenance chain.

## S1 Integration Units

| Unit | Decoder assembles | Encoder does | Cadence |
|---|---|---|---|
| **S1Surface** | candidate memories (recall) + rules + terms + dates + corrections | Haiku renders categorized `additionalContext` | every user turn |
| **S1Scribe** *(rename from S1Encoder)* | entity-feature pairs + resolved dates + gaps + update candidates + pattern candidates + existing node catalog | Sonnet reviews, commits node creates/revises/connects | every 5 stops |
| **S1Gate** *(later)* | rules applicable to pending tool call | algorithmic block / warn / allow | per PreToolUse |

Each gets its own trace chain. Each gets its own entry in the
`interactions` table (learnable K for that boundary).

### Naming note — S1Encoder → S1Scribe

"Encoder" is technically accurate but understates the role. The unit
is agentic, the ONLY scale that creates new nodes from new input, and
the committer that turns ephemeral conversation into durable graph
state. "Scribe" conveys: listens, decides what's worth recording,
commits to permanent record. Historical weight (scribes as keepers
of records; specialists).

## Providers

### Algorithmic (deterministic, always-on, near-zero cost)

- `recall` — existing `brain.recall()`, top-25 candidates with scores
- `rules_applicable` — SQL over locked rules × relevance filter
- `session_continuity` — session state + recent trace query
- `corrections_chain` — graph walk from correction edges
- `relative_time_expressions_finder` — regex: "2 weeks ago",
  "last Tuesday" (finding is lexical; resolving is Haiku)

### Haiku (anything requiring semantic judgment)

- `entity_extractor` — entities + attributes + quantities + spans,
  returns `[{entity, attributes: [...], quantities: [...], span}]`.
  Replaces TF-IDF "distinctive terms" approximation.
- `date_resolver` — identified phrases + reference date → ISO-8601
- `gap_detector` — "conversation references X, brain has no node for X"
- `temporal_edge_inferrer` — event-to-event temporal relations
- `pattern_candidate_detector` — patterns the conversation names
- `update_detector` — "this supersedes a prior value on existing entity"

### Cost & caching

Haiku providers share a base prefix (instruction + conversation turns +
current date + reference context). Mark that prefix `cache_control`.
First provider call writes cache (~1.25× cost), subsequent provider
calls in the 5-min TTL window hit cache (~0.1× cost).

For 5 Haiku providers running per gate, that's effectively one
base-cost call amortized across five.

**Also: `run_llm_loop` should use `cache_control` on system prompt +
early user content.** Current code doesn't. Multi-round Sonnet calls
re-send full context each round. Adding cache markers is a small
change (~30 min), zero architectural risk, ~90% input-token savings
on rounds 1+ of any multi-round call.

## Tools Available to S1Scribe (on-demand providers)

Scribe doesn't need every provider run upfront. Cheap providers run
via the decoder; expensive semantic probes can be TOOLS the Scribe
invokes when it identifies a gap in the proposals it received:

- `deeper_recall(refined_query)` — second-pass narrower query
- `resolve_entity(entity_name)` — deeper research on ambiguous entity
- `check_contradiction(new_fact)` — does this conflict with existing
- `propose_edge_type(a, b, context)` — best relation for this edge

Keeps upfront provider cost bounded. On-demand depth only where
Scribe identifies it's needed. Same provider code either way;
two invocation paths (upfront batch vs tool call).

## Proposals: current vs target

### Today (S1E receives all raw, unstructured)

- Node catalog: 25 surfaced nodes, raw content, Scribe interprets relevance
- Conversation timeline: 10 raw messages
- Encoding journal: prior decisions as prose
- Session context: accumulated phrases

Scribe does extraction + decision + commit. Overloaded.

### Target (structured, categorized, pre-chewed)

```
{type: 'entity_feature',
 content: {entity: 'Memrise', feature: 'mnemonics-based vocab learning'},
 span: 'turn 3', confidence: 0.9,
 source_provider: 'entity_extractor'}

{type: 'temporal_anchor',
 content: {raw_phrase: '2 weeks ago', resolved_iso: '2024-04-05'},
 span: 'turn 5', confidence: 0.95,
 source_provider: 'date_resolver'}

{type: 'gap',
 content: {entity_mentioned: 'Anki', brain_has_node: false, suggested_type: 'concept'},
 span: 'turn 2',
 source_provider: 'gap_detector'}

{type: 'update',
 content: {existing_node_id: 'abc123', new_value: '6pm', superseded_value: '7am'},
 span: 'turn 7',
 source_provider: 'update_detector'}
```

Scribe reads categorized stream, judges per-category, commits.

### What drops from Scribe's prompt when providers land

- "Two registers of knowledge" paragraph (~150 chars)
- Temporal references rule + inline example (~680 chars)
- "Preserve distinctive terms" guidance
- "Don't abstract away specifics" directives
- Entity-feature association rules

Scribe prompt becomes about: how to judge proposals (commit vs skip),
node shape + edge type vocabulary, journal writing. Shorter, sharper.

## First Test

**Success criterion** (Tom's bar): encoder enhanced (higher entity
coverage, higher gold-match rate) AND effort reduced (shorter prompt,
lower tokens, tighter variance) → architecture validated.

**Control A**: current system (s1e v6), our 10-item LongMemEval subset.
Measure:
- Entity coverage: of distinctive entities in conversation, how many
  appear in encoded nodes
- Gold-match rate: does gold-answer-matching content appear in any
  encoded node
- Scribe prompt length (chars, tokens)
- Sonnet tokens per gate
- Variance across 3 identical runs

**Treatment**: one provider — `entity_extractor` (Haiku).
- Input: current date + last 10 conversation turns
- Prompt: "Extract named entities (products, places, people, books,
  apps, specifics). For each, list claims, features, quantities,
  distinguishing properties. Return JSON: [{entity, attributes,
  quantities, span}]. Skip generics. Include specifics even in lists."
- Output flows into Scribe's K as a structured block
- Scribe prompt stripped of specifics-preservation directives (~1000
  chars shorter)
- Same 10 items, 3 runs for variance

**Measure both, compare.** If treatment improves entity coverage AND
gold-match while shortening prompt and lowering total tokens → thesis
validated. Extend pattern to next provider (`date_resolver` probably).
If not → we learn which part of the hypothesis breaks and iterate
targeted.

## Work Sequence

1. **`cache_control` in `run_llm_loop`** — ~30 min, pure plumbing,
   zero architectural risk, ~90% input-token savings on multi-round
   calls. Worth doing independent of the S1 decomposition.
2. **`entity_extractor` provider** — ~45 min, Haiku, cached base
   prefix.
3. **Wire into Scribe's context** — ~15 min.
4. **Shortened Scribe prompt** — register v7 (or start S1Scribe v1
   under new interaction name), strip specifics-preservation
   directives.
5. **Run control A + 3 treatment runs** — ~90 min eval + wait.
6. **Analyze** — entity coverage, gold-match, prompt/token deltas,
   variance. Decide continue vs iterate.

~3 hours end-to-end to a validated data point.

## Carried forward / open

- Exact proposal JSON shape — start loose, firm up after iteration.
- Rename timing — do we rename `s1e` interaction to `s1_scribe` when
  we register v1 of the shortened prompt, or keep the name `s1e` and
  only use "Scribe" in docs/conversation? Lean toward renaming when
  we write the first real decomposed version.
- Shared caching across S1Surface + S1Scribe — possible but adds
  coupling. Start with within-unit caching.
- `S1Gate` design — deferred. Comes after Scribe decomposition proves
  the pattern.
- Closing the outcome loop (S3) — named as a strategic gap earlier
  tonight. Still the biggest missing piece architecturally. Out of
  scope for this POC.

## Related files

- `servers/scales/s1/encode.py` — current S1E (will rename to scribe)
- `servers/scales/s1/encoding_prompt.py` — current Scribe prompt seed
- `servers/scales/s1/surface.py` — current S1Surface entry point
- `servers/scales/s1/surface_contract.py` — surface config + helpers
- `servers/scales/runner.py` — `run_llm_loop` (target for cache_control)
- `eval/longmem/` — eval harness for validating architecture changes


## Scout architecture roadmap (2026-04-23)

### Phase 1 — Muster-and-Scouts (current)

Replace the monolithic S1 Scribe detection work (6 concurrent detection
patterns embedded in the encoder prompt) with four parallel scouts, each
with a bounded dimension. The scribe stays the integrator and only writer;
scouts inspire via category-statement + evidence candidates.

Scouts:
- `s1_scout_quote` — phrases worth atomizing as quote nodes (Haiku)
- `s1_scout_temporal` — date anchors → time_anchor bridges (algorithmic)
- `s1_scout_facts` — entity-feature-value tuples (Haiku)
- `s1_scout_synthesis` — cross-turn patterns no single turn names (Sonnet)

Shared input (user content, 5m cache, byte-identical across scouts):
orientation + session context + current date + node catalog + surfaced
nodes + conversation window.

Per-scout task (system prompt, 1h cache, per-scout) comes from the
interaction template. Each scout has its own interaction entry so S3 can
evolve them independently.

The muster runs all four scouts in parallel at the start of `run_encoding`,
collects envelope outputs, formats a SCOUT_REPORTS block that feeds into
S1Scribe's prompt. Orientation is unified via `servers/scales/s1/orientation.py`.

Event relations (`before`/`after` between events) are NOT scouted in
Phase 1. Temporal emits date anchors with event_description sentences;
S1S's prompt gets guidance to spot relational markers ("just before",
"right after") and, when the reference event is in its catalog, compose
an edge between them. Without enrichment (Phase 2) the catalog often
lacks the reference event, so relations fire rarely in Phase 1 — that's
expected; Phase 2 unlocks them.

### Phase 2 — Scribe enrichment path (CONDITIONAL, not committed)

**This only happens if the scribe proves insufficient after P1.** Symmetry
between what S1R surfaces and what S1Scribe sees is valuable by default —
same catalog, same framing, one source of truth. Splitting the paths is
additional complexity we only pay for if data forces it.

The condition: after P1's baseline run, if ENCODE_MISS failures persist
specifically because the scribe lacked context that SHOULD have been in
the catalog (entities mentioned but not surfaced, referenced events not
picked for Anchor's reply), then Phase 2 is warranted. Otherwise skip.

If we do activate Phase 2, the shape below is the starting proposal —
don't treat as decided architecture:

Today the scribe's catalog is exactly what the surfacer selected for
Anchor's replies across the N-turn window. The surfacer is tuned for
response-context ("what helps the operator's reply be sharp"), which is
narrower than what the scribe might need to place new nodes well.

The second retrieval path, if added, would fire once per encoding cycle
and be deliberately broader:

- Higher top_k per turn (e.g. 15-20 vs surface's 5-8)
- Per-turn recall seeded from proper-noun mentions (entity-aware)
- Lower noise floor — loose matches OK; scribe uses them as anchor
  candidates, not as Anchor's context
- Different fatigue policy — scribe WANTS to see repeated nodes to
  notice repetition
- Merged with the surface-selected catalog → scribe gets a richer
  neighborhood

If activated, this is also what would make event-relation encoding
tractable. A turn saying "just before my trip to Portland" needs the
prior "trip to Portland" node in the scribe's catalog to compose a
`before` edge. Surface alone may not pick it up (peripheral to the
reply); enrichment would.

Expected lift if activated: multi_session and temporal axes on the
longmem benchmark — these would benefit most from event+reference-event
connectivity.

**Decision rule:** wait for P1 baseline. If multi_session lift is
materially lower than info_extraction and knowledge_update lifts, AND
failure analysis points to scribe-side catalog gaps, consider P2.
Otherwise: keep the symmetry, the simplicity wins.

### Phase 3 — Tune (after P1 baseline, regardless of P2 status)

After P1 baseline we read the data and iterate. Independent of whether
P2 is activated:

- Entity extraction quality — is proper-noun regex enough, or do we
  need Haiku-extracted entity lists?
- Scribe-specific scoring — different z-weights, different fatigue?
- Catalog size caps — when does richer hurt vs help?
- Temporal scout Haiku fallback — wire for phrases dateparser can't
  resolve (liturgical dates, seasons with context-dependent meaning)
- Temporal scout relation-marker detection — optionally emit
  `has_relational_marker: true` when "just before / right after /
  during" appears near a date phrase, so S1S knows this turn is a
  candidate for a cross-event edge. Still algo; still not a scout
  output kind.

### What we deliberately rejected

- **Option B (temporal scout owns event-relations with Haiku fallback).**
  Adds multi-kind candidates to one scout, brings LLM flakiness into
  the deterministic-by-design temporal path, and the reasoning needed
  to parse "before X" into a node-to-node edge is exactly what S1S
  does naturally given both endpoints. See 2026-04-23 scout design
  discussion.
- **Storing explicit `before`/`after` edges for every relational
  marker.** Most temporal queries fall out of ANCHOR structure: two
  events each with a date → subtract at query time. Only rare explicit
  chaining ("it happened just before X") benefits from an edge, and
  Phase 2 + S1S prompt guidance handles those without a dedicated scout.
