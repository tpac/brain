# V3.5 analysis plan — fixed before the cell runs (2026-09-12)

Tom's ask: before running, make the analysis robust; keep everything the V3.3/V3.4 passes
measured and add the quality of content and of every other field. This plan names each
instrument, what it reads, what it can and cannot decide, and where past passes went wrong
(brain: id:9f1bbac1 field fill / word count / value base; id:f612f309 fresh corpora by id;
id:079d9736 run-to-run variance dwarfs arm deltas — sample, never read one run; id:d3a944ed
defensible plurality — discovery, not a single score; id:3cdb33b3 automated staleness scores
are a READING QUEUE, not a verdict; E22 read every A/B on the full shape; E24 a gold set
scores one purpose).

## Design constraints carried forward

- **Fresh material chosen by id before authoring** (`transfer_split.json`, recorded first):
  the next feasible LongMemEval reserve items in the recorded sha256 order with zero mentions
  under eval/, docs/ and ~/AgentsContext, plus one repository synthetic with zero prior
  mentions. All twelve V3.3/V3.4 corpora are development data and run as regression only.
- **Live tool schemas** as V3.4 froze them (the V3.1–V3.3 confound is closed; do not reopen).
- **Three repeats per (arm, corpus)**; sequential windows with journal continuity; one process
  per (arm, repeat); preflight proves identical factual sections across arms; every input
  pinned by hash before a model call; the gist is an arm axis (X5 lives there) and is
  fingerprinted in the arm identity.
- **Arms on fresh material:** production (deployed), V3.4 (frozen parent), V3.5 (candidate).
  V3.3-on-live-tools is retired from the fresh cell: the attribution question it answered is
  closed. **Regression:** V3.5 alone on the six V3.4 fresh corpora against the saved
  production / V3.3-live / V3.4 brains, and on the six V3.3 corpora against the saved
  production / V3.2 / V3.3 / V3.4 brains. **Sanity:** V3.5 on creative_design beside the saved
  arms.
- **No prompt edits after outputs are seen** in this cell. Findings feed the next weave.

## Instruments — what each reads, what it can decide

| # | Instrument | Reads | Measures | Decides | Cannot decide |
|---|---|---|---|---|---|
| 1 | `quality_census.py` (V3.3 instrument + V3.5 counters) | final nodes, source | types/aspects/entropy; field fill; words per field; specificity per 100 words; situation trigger register / title restatement; question register / encoder-side; reasoning provenance / **restating content**; provenance leaks (turn coordinates, encoder clock); event_time precision and clock; quotes verbatim / style-only / splice / not-in-source; edges: relations, generic, restating, duplicate pairs, isolates, components, degree; near-twins; recall lanes per node; **nodes derived from MY turns vs THEIR turns and the quote each side carries** | shape and fill differences between arms; the voice-derivation gap | whether any field's content is TRUE or useful |
| 2 | `revise_sweep.py` (V3.3) | every revise op, before/after | write path: landing, preservation, deltas, revised_at, vectors; refusals by class; connect_to on revise | infrastructure correctness; op shapes the encoder uses | semantic quality of the revision |
| 3 | `stale_surfaces.py` (new) | every successful revise op + end-of-window node | old-value tokens the op replaced that still sit in title / situation / question / thought / quotes / edge descriptions (record fields content/reasoning reported apart); dates moved in text with event_time untouched; every event_time change beside the node's quote | a QUEUE of half-revised nodes per arm and per surface; the class the 12-pack blind found in every arm (retro on V3.4: production 10/14 ops, V3.3-live 9/12, V3.4 8/12 leave a value behind; edges and quotes carry most) | whether a hit is a dead claim or a dead subject (E17) — the reader decides |
| 4 | `surface_redundancy.py` (new) | final nodes | load-bearing values (numbers, dates, proper nouns) and how many of title / content / quote / situation / question carry each; content-only share; titles carrying a value | X1's target (retro on V3.4: mean surfaces per value 1.56 / 1.54 / 1.69; content-only 65 / 67 / 60%) | whether a value belongs on more surfaces |
| 5 | `firming_markers.py` (new) | final nodes, source | certainty markers per 100 nodes; markers whose stem the source never used (UNSOURCED), on retrieval surfaces vs in reasoning; hedges per 100 nodes | a QUEUE for the scope reviewer; the register balance per arm (retro on V3.4: unsourced surface markers on 31 / 34 / 41% of nodes; hedges 46 / 76 / 91 per 100 — V3.4 carries both registers most) | whether a marker is an overclaim — the reviewer decides |
| 6 | `content_quality.py` (new, Sonnet 4.6 judge, per node against the whole source) | final nodes + source | value class (specific knowledge / useful synthesis / supporting context / generic advice / redundant / unsupported); fidelity (supported / overclaimed / weakened / fabricated element / mixed) with the deciding phrases; status kept; owner kept; stale surface; per-field quality — situation (trigger / restated / narration / overbroad), question (real / formulaic), reasoning (basis / encoder justification / generic / restated), thought (hunch-or-connection / caveat / restatement / noise), quotes (verbatim load-bearing / incidental / altered / fabricated), edges (insightful / restating / generic), event_time (correct / wrong / missing where supported); the single most consequential defect | distributions per arm at full coverage (every node, every repeat); a queue of flagged nodes for the author's read | a verdict — one judge, one reading; calibrate by reading the queue and a random sample of "supported" |
| 7 | `coverage_targets.py` (V3.4) | synthetic corpora with ground truth | encode targets OWN / FOLDED / ABSENT via judge; decode queries through real recall + Sonnet answer | the folding stance (X2's primary readout) | anything on LongMemEval items (no targets) |
| 8 | `uncovered_nodes.py` (V3.4) | two arms' final nodes | nodes one arm wrote with no counterpart in the other (token Jaccard) | what the smaller memory dropped, by type | value of the dropped nodes (read them) |
| 9 | `downstream.py` (V3.4) | kept final brains | real recall top-5 → Sonnet answer → Sonnet judge against gold; gold reach in the retrieved set | recall-conditional answerability on the LongMemEval question | quality beyond the one gold question (E24) |
| 10 | `analyze.py` + `blind_pack.py` (V3.4, rubric extended) | packets | blind paired review by Opus: 7 dimensions + **8 Content and field quality** (per-node value density: reasoning as basis not restatement; thought as own read; edge whys carrying insight; quotes verbatim and load-bearing; situation and question in their registers) + corpus probes (**stale surfaces after a value changed; my voice where my turn carried the knowledge; a mention read as confirmation; two statements that differ with no word of correction**) | the qualitative comparison, sealed, per pack | a score — ratings are a reading aid; read the profile, not the sum |
| 11 | Per-pack tally (`tally.py`) | verdict tables | strong/adequate/weak per dimension per arm; pack winners; never-weak packs | the dimension PROFILE and its stability across repeats | anything the reviewers did not read |

## How the readout is written

1. Infrastructure first (2): the write path must be clean before any quality claim.
2. Shape (1, 4, 5, 8): what each arm writes — with run-to-run spread shown per repeat, not a
   mean that hides it (id:079d9736).
3. Content (6, 3): the judge's distributions, then the QUEUES read by hand — every flagged
   node on the fresh cell for the candidate arm, a sample for the others; the author's read
   confirms or overturns the judge, and the doc says which.
4. Function (7, 9): does the memory answer, does the folding stance move.
5. Blind (10, 11): the qualitative profile, dimension by dimension; a level sum is not a tie.
6. Weave attribution: each V3.5 change names its readout in advance —
   X1 → 4 (surfaces per value) and 3 (stale surfaces must not rise); X2 → 7 (FOLDED→OWN on the
   synthetic) and node count / uncovered by type (8); X3 → node count and 6 value classes
   (watch generic_advice); X4 → 6 event_time + 3 event_time changes; X5+X6 (one lane) → thought
   fill (1) and thought class (6); X7 → no expected number (depiction law); W6 → 3 (stale
   surfaces per op, per surface) and blind probe; W7 → 5 unsourced markers and 6 overclaimed;
   W8 → 1 my-voice quote on me-derived nodes, blind dim 5; W9 → static only; W10 → blind
   probe on any item with two differing statements (none guaranteed on fresh material —
   say "silent" if none).
7. Regression: V3.5 against V3.4 on the twelve development corpora on every instrument; any
   loss is a trade to rule on, not noise.

## Known limits, stated now

- One judge model (Sonnet 4.6) reads production, V3.4 and V3.5 alike; its biases are shared
  across arms but not absent. The blind reviewers are Opus; the author reads the queues.
- Token heuristics (3, 4, 5) over-fire on prose swaps and proper-noun noise; they are queues.
- Three repeats bound, but do not remove, run-to-run variance; per-repeat spread is reported.
- LongMemEval items carry one gold question each; content quality beyond it comes from 6 and
  the blind read, not from 9.
