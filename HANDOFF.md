# Session Handoff — 2026-03-22

## What Was Done This Session

### Phase 1: Text Processing Pipeline — COMPLETE AND DEPLOYED

Four integrated systems built in `servers/text_processing.py`, tested (65 new tests, 207/208 passing), benchmarked, committed, merged to main, plugin rebuilt (408K), and cache replaced.

**Commits on main:**
- `a5d155a` — Identifier splitting: single-pass `re.findall` replaces fragile two-pass `re.sub`
- `596af51` — Common-word filter: 10K word list from Google Trillion Word Corpus for domain-specificity detection
- `03214df` — 7-strategy vocabulary extraction with domain filter (extractions doubled: 14→32)
- `8d45d07` — Sentence splitting: pySBD with code reference protection (accuracy: 31%→94%)
- `6b19eb1` — Build manifest updated to include `text_processing.py` + `data/common_words_10k.txt`

**Benchmark results:**
| System | Accuracy/Impact | Performance |
|--------|----------------|-------------|
| Identifier splitting | 16 cases improved, 0 regressed | 0.003ms/call (35% faster) |
| Common-word filter | 100% on 57 test cases | 0.0004ms/call |
| Vocab extraction | Extractions doubled (14→32) | 0.037ms/call |
| Sentence splitting | 31% → 94% accuracy | 0.198ms/call |

**Key files:**
- `servers/text_processing.py` — All 4 systems (346 lines)
- `servers/daemon_hooks.py` — 7-strategy extraction integrated (lines 594-686)
- `servers/brain_surface.py` — `split_identifier()` integrated in suggest() (lines 50-70)
- `servers/recall_scorer.py` — `split_sentences()` replaces 3 naive `re.split` calls
- `data/common_words_10k.txt` — 9,894 common English words
- `tests/test_core.py` — 65 new tests across 8 test classes
- `tests/bench_*.py` — 4 benchmark scripts

**Brain nodes about this work:**
- `8acd81b1` (locked) — Full mechanism doc of all 4 systems
- `40b34f96` (locked) — Session summary
- `93e20a72` — Vocabulary pipeline gap lesson
- `6c75f6ed` — Context continuation vs compaction lesson

---

## Critical Gap: Vocabulary Loop is Open

The 7-strategy extraction detects domain-specific terms and stores them as `vocabulary_gaps` config entries (capped at 20). These surface as consciousness signals. **But nothing auto-creates `vocabulary` type nodes.** There are **0 vocabulary nodes** in the brain (1,056 nodes across 30+ types).

The method `brain.learn_vocabulary(term, meaning)` exists in `brain_vocabulary.py` but only fires when Claude explicitly calls it. The automatic pipeline stops at gap detection and waits.

**This is the most important gap to close** — the entire Phase 1 extraction improvement is wasted if detected terms never get stored.

---

## What Tom Wants Next

### Priority 1: Close the vocabulary loop
Make extracted terms actually get stored. The extraction and filtering are solid — the storage step is missing.

### Priority 2: Phase 2 — Entity Extraction (spaCy)
See `docs/ENTITY-GRAPH-DESIGN.md` for full design. Tom's entity model is broad:
- People, products, companies
- System components (daemon, recall scorer, precision loop)
- Architecture (screens, pages, API endpoints)
- Concepts-as-things ("the supply adapter pattern")

Key dependency: `spacy` + `en_core_web_sm` (15MB, MIT licensed).

### Priority 3: Knowledge graph + ATS engineering graph
Tom mentioned wanting to work on this "on a different thread" but never specified details. ATS = Applicant Tracking System? Ask him. This may relate to Phase 3 (typed edges + graph traversal) in the entity graph design.

### Phases 3-4 (future):
- Phase 3: Typed edges, LLM relationship classification, graph-augmented recall
- Phase 4: Entity-aware consciousness signals, timelines, relationship-based dreaming

---

## Known Issues

1. **DAL ratchet test failing** — `TestDALPatternEnforcement::test_violation_counts_not_increasing` expects <=40 violations, finds 49. Pre-existing, not from our changes. Don't touch.

2. **Stale "TEST NODE DELETE ME" node** — Shows up in recall results. Should be deleted: `brain.conn.execute("DELETE FROM nodes WHERE title='TEST NODE DELETE ME'")`

3. **GitHub push pending** — Main is 53+ commits ahead of origin. SSH key issue may need Tom to resolve.

4. **Worktree cleanup** — `claude/keen-lederberg` worktree branch still exists. Can be cleaned up after confirming main has everything.

5. **pySBD performance** — 0.198ms/call is ~100x slower than old regex. Acceptable for hook pipeline but worth noting if latency becomes an issue.

---

## Brain State

- **1,056+ nodes** across 30+ types (decision: 355, rule: 173, lesson: 75, concept: 64, context: 46, ...)
- **0 vocabulary nodes** (the gap)
- **1,735 recalls logged, 0 evaluated** (precision loop fix was committed in earlier session as `98dbaf9` but may not be live yet)
- **Key brain nodes to recall:** `8acd81b1` (text processing systems), `d5d89d` (typed edges decision), `7713d6` (three extraction layers), `6cd874` (Tom's entity model), `c0b404` (Tom's encoding definition)

---

## Session Meta

- Context continuation event happened mid-session (not normal compaction). This is a newer Claude Code behavior when context is completely exhausted — starts a new conversation with a summary, rather than trimming within the same session. Tom was confused by it. Brain hooks (PreCompact/PostCompact) don't fire during continuation.
- Tom expects parallel execution — launch agents on separate threads when tasks are independent.
- Tom expects proactive next-step suggestions after completing work.
- Broad `find` on home directory triggers macOS permission dialogs — avoid.
