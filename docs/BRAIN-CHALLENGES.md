# Brain Challenges

Two layers live here:

1. **The challenge ledger** (below) — the indexed record of every failure,
   miss, and hard-won lesson the memory system has accumulated about itself,
   mined from the brain's own record (communities, corrections, encoder
   journals, prompt-version diffs, eval reports). One row per distinct
   failure; full tables in [docs/challenges/](challenges/).
2. **Narrative entries** (bottom) — long-form cognitive-gap write-ups where a
   single case deserves the full story. The ledger indexes these too.

The ledger's target function — what every row is scored against:
**memory exists to change the next action.**

## The challenge ledger

Built 2026-08-20 by a 24-miner fan-out over: 299 encoder/recall-family
communities + 3,375 member nodes, 553 loose corrections/bugs/diagnoses/opens,
555 out-of-community lessons/principles/findings/insights, 2,552 encoder
journal notes (s1e + S2, 2026-07-22→08-20), all 37 s1e prompt-version diffs
with adjacent decision nodes, and the longmem eval reports + EVAL-PLATFORM
docs. Merged and deduplicated by shared-evidence clustering (386 folded).
Coverage note: ~600 journal notes from mid-June→July were never mined by any
pass (operator ruling: recurring patterns will resurface; completeness not
required).

**Row semantics**

| column | meaning |
|---|---|
| claim | one self-contained sentence naming the mechanism |
| evidence | brain node ids (8-hex), chain ids, or file paths |
| fix | what shipped and when — `none stated` = known, never fixed |
| pinned-by | the test/eval/gold item protecting it — `NONE` = unprotected |
| status | `unverified` (default) / `superseded-by:<id>` / `verified-current` (only when the text itself proves currency) |
| corpus? | could become a gold eval item for the encode/recall corpus |

Old rows carry superseded conclusions **by design** — treat `unverified` as
"was true when written," never as current truth.

**The ledger** (1,813 rows after dedup, from 2,199 raw across 24 miners)

| class | rows | meaning |
|---|---|---|
| [encode-write](challenges/encode-write.md) | 191 | the encoder wrote a wrong or poor node |
| [encode-notice](challenges/encode-notice.md) | 50 | the moment was noticed but encoded badly |
| [encode-absent](challenges/encode-absent.md) | 85 | an expected node was never written — catalog gap |
| [recall-surface](challenges/recall-surface.md) | 185 | the knowledge existed but was never surfaced |
| [recall-rank](challenges/recall-rank.md) | 165 | surfaced but ranked or selected wrong |
| [answer](challenges/answer.md) | 55 | recalled but misused in the reply |
| [structure](challenges/structure.md) | 312 | graph, edges, or community layer wrong |
| [process](challenges/process.md) | 231 | working-discipline failure around memory |
| [eval-method](challenges/eval-method.md) | 309 | benchmark or eval methodology mistake |
| [prompt-genealogy](challenges/prompt-genealogy.md) | 230 | prompt-design decision and its origin |

Plus the version-by-version prompt history:
[prompt-genealogy-versions.md](challenges/prompt-genealogy-versions.md) —
all 37 s1e registrations, what changed and the recorded why.

**Headline numbers (2026-08-20)**

- **943 rows (52%) are `pinned-by: NONE`** — past victories protected by no
  test, gold item, or eval. This is the regression-gap map for Q3/Q7.
- **609 rows (34%) have `fix: none stated`** — diagnosed, never shipped.
- **934 rows are corpus candidates** — the target list for time-travel
  gold-item harvesting (Track A1).
- 101 rows explicitly superseded; only 91 claim verified-current (a 12-row
  spot-check of named pins found all 12 present in the repo).

**Cross-cutting findings the merge surfaced** (each recurs across
independent miners):

- The `outcome` element of the O/K/Δ trace schema was specified at every
  scale and never implemented — nothing in the record can currently *prove*
  a memory changed the next action. The target function is unmeasurable by
  design, not by accident.
- The most re-learned lessons: *recall-before-re-deriving* (≥6 independent
  occurrences), *build-the-ruler-first* (≥5), *examples-beat-instructions
  for Sonnet* (≥4, same edit applied to a different prompt each time — no
  standing cross-prompt example audit exists), *n=1 is not a measurement*
  (≥3), *measurement circularity* (≥4 in the eval record alone).
- The catalog holds uncorrected contradiction pairs (opposite claims about
  the same state, no correction edge) in every large batch — including S2
  community narratives that launder debunked findings ("+12pt z-score")
  into the summary layer recall reads first.
- Silent-plumbing failures recur: signals computed then dropped
  (`edge_context` at weight 0.55 with 0 rows ever; 41 healer question
  embeddings dropped; auto-encode dead from birth behind a 3s timeout).
  The class survives because nothing pins wiring end-to-end.

## Narrative entries

Long-form gap write-ups. When an entry's fix becomes a real work item, it
migrates to [docs/BACKLOG.md](BACKLOG.md); entries stay here as the
cognitive-gap record. Entry #2's fix family is now the Frame-frontier
scoring lane and the cue-side temporal lane in BACKLOG.md's *Now* section.

---

## 1. State-dependent memories surfaced as if stable

**What happened (2026-04-22).** A memory recalled during a prompt-edit task
said *"s1e v2→v3 (+307 chars): Added question to both example nodes;
reasoning to 2nd node"* (memory id:d3dc26d7, 2 days old). Anchor treated
the version number as current and proposed a v3→v4 edit. The live
interaction was actually at v9 — six versions had landed since the memory
was written. Tom caught it.

**Why the brain fell short.** The memory was factually correct at write
time but claimed a **state-dependent fact** (a version number that
advances every time someone registers a new prompt). Recall pool treats
it identically to stable knowledge like *"don't use `related_to`
edges"* — no signal that one rots and one doesn't.

**Pattern.** Other recalled memories carry the same shape without flags:
- counts (*"135 clusters proposals"*)
- inventories (*"Brain DB backup inventory — pre-consolidation through Apr 2026"*)
- taxonomies pinned to a specific version (*"S2Consolidate encoder actions — complete taxonomy"*)
- *"was last set to X"* facts about anything mutable

When the underlying thing evolves, the memory keeps asserting the old
value in recall with full confidence.

**What would help.**
- **Volatility tag at encode.** Encoder classifies a node as `stable`
  (principles, architecture choices) vs `snapshot` (numbers, versions,
  counts at a moment) vs `mutable` (entity state, config values).
  Surface annotates or downweights volatile candidates.
- **Surface-time live check for cited entities.** If a memory mentions a
  specific interaction name / node id / version string, the surface
  could fetch the current state and attach it. *"Memory said v3 → live
  now shows v9."*
- **Operator default to verify when acting on recalled claims.** Cultural,
  not structural: memory gives the *question*, live DB gives the
  *answer*. Applies to Anchor and to any agent that reads brain state.

None of these is free. The volatility classifier adds an encoder
decision; live-check at surface costs a DB round-trip per candidate. But
the failure mode is real and silent — Anchor confidently built a plan on
stale facts.

2026-08-20 ledger note: none of the three proposals was ever built
(`grep -rn volatility servers/` returns zero hits). The staleness family
in the ledger carries five documented production misses and zero shipped
fixes.

---

## 2. Recall surfaces topical-but-stale, misses recent named arcs

**What happened (2026-05-05).** Two prompts in one session, both naming
arcs the brain knows well, both got irrelevant surfaces.

*Prompt 1:* "rebuild a fresh plan. First thing is to make sure Aspect
encoder is populated well, yesterday we worked on that and sent into a
spiral to revise a lot of our core functions. Aspect encoder should be
able to produce in a json format..."
Surfaced: `04ff3d58` (MCP-vs-prompt split, 1w ago), `822325f9` (A2 Fix
1 brain_batch result, 1w ago), `d8846a75` (Fix 1.1 verification rule,
1w ago), `705b742b` (new encoder adopted, 4w ago).
Did NOT surface: any of the ~25 aspect-encoder decisions from
2026-05-04 (cycle counts, routing asymmetry, locked-required guards,
testing strategy) — the literal subject of the prompt, encoded
yesterday.

*Prompt 2:* "revisit the Frame and see where we left it off, and then
go back to recall which was the original task."
Surfaced: `32ab5545` (target function, 1mo ago), `66b0d6f5` (Recall
Philosophy community, 3d ago — partially relevant), `f50db1f2` (build
order, 1mo ago).
Did NOT surface: Frame Phase 2/2.5 work nodes from May 1–3, the agentic
recall 7-tool design, connection scoring spec, Q13 spread-activation
decision — the active edges of both arcs.

**Why the brain fell short.** Token-level overlap pulled
topically-adjacent old material ("encoder" → encoder optimization
history; "recall" → recall philosophy axioms) and missed the
specifically-named recent arcs. Frame should have served as the prior
that biased toward "what we worked on yesterday" and "what's open in
recall right now," but either it didn't fire, didn't reach Haiku, or
Haiku didn't weight it. The selected nodes have high
content/confidence/access — strong general gravity — but no signal
that ties them to *this conversation's frontier*.

**Pattern.** Aged-but-topical beats fresh-and-named. Specific keywords
that should be high-precision anchors ("Aspect encoder", "Frame")
behave like general topic queries, returning the dominant historical
cluster on that topic rather than the live work. The brain is acting
like a topic encyclopedia, not a partner that knows what we're doing
this week.

**What would help.**
- **Frame-as-filter, not just prior.** If Frame's "Active threads" and
  "Recent moves" carry node IDs, the surface can boost candidates that
  match those IDs (or are 1-hop neighbors) before Haiku selects.
  Recency + Frame-membership as a hard re-rank signal.
- **Phrase-anchored recall for exact named entities.** "Aspect encoder"
  in the query should pin candidates whose *title* contains the exact
  phrase, not just candidates with cosine-similar embeddings to the
  whole sentence. FTS5 exists; this is a weighting question, not new
  infra.
- **Fresh-arc preference at the surface.** Selected nodes carry their
  own age. The surface prompt could explicitly bias Haiku toward
  recent work when the user message is itself about "what we're doing
  now" / "where we left off" / "yesterday." This is a posture detection
  layer on top of recall, not a recall change.
- **Diagnostic.** Capture this exact case in `eval/frame_replay.py` —
  two queries (`aspect_encoder_pickup`, `frame_recall_resume`) that
  fail today, and use them as the regression test for the fix.

The deeper read: recall is currently optimized for "find similar
memories." We've named the principle (recognition over retrieval), we
haven't operationalized it. This is one of the cases where the
operational gap is visible from outside.

---

## 3. Recall can't find facts encoded in node content (2026-05-10)

**What happened (2026-05-10).** Two regressions in the Eval A v15.3 run share a single pattern:

- `2318644b` (multi_session): Question — "How much more did I spend on accommodations per night in Hawaii compared to Tokyo?" Brain has node `139695c7` ("Tom's Maui trip — resort context and budget-balancing approach") with content explicitly: *"Tom has booked a luxury resort in Maui costing over $300/night."* Recall returned 12 candidates, top score 0.62. The Maui-$300-bearing node was NOT in the top-12. Answerer correctly reported "memories don't contain Hawaii accommodation cost info" — because the relevant node wasn't surfaced.
- `8e91e7d9` (multi_session): Question — "What is the total number of siblings I have?" Brain has node `ea478a37` ("Operator's weekly book club — 10F/4M/1NB composition") with content saying *"He has a brother... and 3 sisters."* Plus `956a1495` ("Tom's professional network is male-dominated despite female-dominant home environment") with *"Tom grew up with 3 sisters."* Recall returned 6 candidates; the most sibling-direct node (book club) was NOT in them. Surface selected 0 (correct abstention — none of the 6 candidates were about siblings).

**Why the brain fell short.** The facts ARE encoded — in node `content`. They're not in node `title` though. Titles describe meta-context ("budget-balancing approach," "professional network male-dominated") rather than the underlying facts ("Maui $300/night," "siblings: 1 brother + 3 sisters = 4"). Multi-vector MAX recall scoring includes content embeddings, but in the flat-cosine space (top scores 0.37–0.62, per `dea1a002`), the discrimination isn't sharp enough to surface content-only matches over title-stronger candidates.

This is the inversion of entry #1's volatility problem: there, recall surfaced wrong facts; here, recall fails to surface right facts that exist.

**Why it's a recall issue, not an encoder issue (Tom's reframe).** The structure of title/content/emergent fields (situation, reasoning, their_raw_quote, my_raw_quote, entity, etc.) is intentional — different containers carry different concepts. The encoder doing its job means the information is captured in the appropriate container. If the fact is in `content` and recall can't find it, **that's recall failing to use the structure well**, not the encoder failing.

**Pattern.** v15.3 encoder is producing richer, more narrative nodes that bundle facts inside context. This is good encoding from a curation/preservation perspective (the *why* is preserved with the *what*) — but bad for atomic-fact retrieval. The pattern shows up when:
- Query terms describe the bare fact (Hawaii cost, siblings count)
- Title describes the meta-frame (budget approach, social network)
- Content has the fact verbatim
- Recall ranker doesn't reach the content-only match

**What would help.**
- **Title-vs-content rescoring at the recall ranker.** Currently MAX across vector types; could add a "content match without title match" boost when the query's distinctive terms appear in content but not title.
- **z-score normalization + RRF** (`951f3ac8`, designed not built) — sharpens flat-cosine discrimination so a 0.55-on-content match doesn't drown in 0.60-on-title matches.
- **Multi-anchor query decomposition** (`87bb8718`) — a 4-anchor query ("Hawaii", "Tokyo", "accommodation", "per night") with convergence scoring would find nodes that match a subset of anchors strongly, even if no single anchor matches the title.
- **Title-improvement at the encoder side as a softer countermeasure** — title patterns like `{entity}: {value}` for fact-bundled nodes (e.g. an alternate node titled "Tom's Maui resort: $300/night" sibling to the context node). But this would inflate node count; better to fix recall.

This entry stays here as the cognitive-gap record. The fixes are recall-pipeline work — already known from prior research (`951f3ac8`, `87bb8718`, `6b07b072`), now confirmed urgent by today's eval evidence.

---

## 4. Names that fight the model they carry (2026-08-21)

**What happened.** Mid-way through the s1e prompt co-review, Tom eyed the
canonical examples and caught what 37 prompt versions, a 1,813-row ledger,
and eleven probes had all read past: the voice-quote field NAMES are wrong.
`user_raw_quote` encodes the assistant-serves-user frame that three separate
fix-waves had been un-teaching at the prompt layer (the D3 voice-equality
work), and `anchor_raw_quote` is internal jargon a stateless encoder only
understands through a gloss. Every run paid a mapping tax ("theirs in
user_raw_quote, mine in anchor_raw_quote") to translate the schema's frame
into the prompt's frame. Ruling: rename to `their_raw_quote` /
`my_raw_quote` via a standalone migration stream (decision id:52c8eb6d).

**The class.** A name is a teaching channel that never stops transmitting
(T2 applies to identifiers, not just descriptions). When a name encodes a
frame the system has since rejected, every consumer re-learns the wrong
frame on every read — and prompt-layer corrections fight a schema-layer
signal, which is why three fix-waves moved the behavior only partway.
Instances already on record: `user_raw_quote`/`anchor_raw_quote` (this
entry); `connect_to_unresolved` claimed for mis-copied ids after the real
source became `connect_to_bad_id` (fixed, walk Stop 8); the retired
`tom_quotes_limit` config key surviving rename by four months (18cd056c).

**The commissioned sweep (not yet run).** A scout pass over every
agent-facing identifier — field names, error-source names, type tags,
relation verbs, config keys, tool parameter names — scoring each against
the question: *does this name teach the model we currently hold, or a
frame we've since rejected?* Output shape: the recall-to-encode pointer
table (one row per mislabel, evidence, blast radius, stamp). Run it the way
the R-rows were mined — a single Opus scout with the current contracts and
the identity docs side by side.

