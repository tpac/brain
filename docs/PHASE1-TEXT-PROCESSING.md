# Phase 1: Text Processing Upgrades — Implementation Plan

**Status:** Ready to implement
**Created:** 2026-03-22
**Dependencies:** pySBD (MIT), 10K common word list (public domain)
**All other changes:** Zero new dependencies

---

## Implementation Order

**Change 1 → Change 4 → Change 3 → Change 2**

This ordering matters: Change 4 (common-word list) is consumed by Change 3 (vocab regex), and Change 2 (pySBD) is the only external dependency so it goes last.

---

## New Module: `servers/text_processing.py`

All four features live in one shared module:
- `split_identifier(name) -> list[str]`
- `split_sentences(text) -> list[str]`
- `load_common_words() -> set[str]`
- `is_domain_specific(term) -> bool`
- `filter_domain_terms(candidates) -> list[str]`

---

## Change 1: Identifier Splitting

**Current** (`brain_surface.py:58-65`): Two-pass `re.sub` then `re.split` — fragile, breaks on acronyms.

**New**: Single `re.findall` pattern in `split_identifier()`:

```python
re.findall(r'[A-Z]+(?=[A-Z][a-z])|[A-Z][a-z]+|[A-Z]+|[a-z]+|\d+', segment)
```

**Algorithm:**
1. Strip known file extensions (`.py`, `.js`, `.ts`, `.sh`, etc.)
2. Protect version numbers (`v1.2.3`) with placeholders
3. Split on `_`, `-`, `.`, `/`, `\`
4. Tokenize each segment with the findall pattern
5. Restore version tokens, lowercase, filter single-char tokens

**Benchmark examples:**

| Input | Current (broken) | New (fixed) |
|-------|-----------------|-------------|
| `parseHTMLDoc` | `parse HTMLD oc` | `parse html doc` |
| `getURLParser` | `get URLP arser` | `get url parser` |
| `HTTPSConnection` | `HTTPSC onnection` | `https connection` |
| `sha256Hash` | `sha256 Hash` | `sha256 hash` |
| `v2.3.1-beta` | `v2 3 1 beta` | `v2.3.1 beta` |
| `pre_response_recall.py` | `pre response recall` | `pre response recall` |
| `XMLParser` | `XMLP arser` | `xml parser` |
| `MyHTTPSURLSession` | garbled | `my https url session` |

**Files:**
- Create `servers/text_processing.py` with `split_identifier()`
- Modify `servers/brain_surface.py:58-65` — replace re.sub calls
- Create `tests/bench_identifier_split.py` — before/after comparison
- Add `TestIdentifierSplitting` to `tests/test_core.py` (8 tests)

---

## Change 4: Common-Word Filter

**Current**: Small hardcoded `skip_words` set (~40 words) in daemon_hooks.py.

**New**: 10K common English word list. Any extracted term whose head word is NOT in the list = domain-specific.

**Logic:**
- Head word = last word (English head-final NPs: "supply adapter" → head is "adapter")
- All-uppercase terms (acronyms) always pass as domain-specific
- Multi-word: if ANY non-head word is also not in 10K, pass it through

**Examples:**

| Term | Head word | In 10K? | Result |
|------|-----------|---------|--------|
| `recall scorer` | scorer | No | ✅ domain-specific |
| `supply adapter` | adapter | No | ✅ domain-specific |
| `the file` | file | Yes | ❌ common |
| `DAL` | DAL | — | ✅ acronym pass-through |
| `webhook` | webhook | No | ✅ domain-specific |
| `brain daemon` | daemon | No | ✅ domain-specific |
| `magic links` | links | Yes | ⚠️ Known limitation — head word is common |

**Files:**
- Create `data/common_words_10k.txt` (MIT/public domain source)
- Add filter functions to `servers/text_processing.py`
- Create `tests/bench_common_word_filter.py`
- Add `TestCommonWordFilter` to `tests/test_core.py` (6 tests)

---

## Change 3: Improved Vocabulary Regex

**Current** (`daemon_hooks.py:594-685`): 4 regex strategies (quoted, "the X", verb+object, hyphenated).

**Add 5 new strategies:**

| # | Strategy | Pattern | Catches |
|---|----------|---------|---------|
| 5 | Capitalized mid-sentence | `(?<=[a-z]\s)([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)` | Clerk, Valinor, Supabase |
| 6 | Backtick-wrapped | `` `([^`]{2,40})` `` | `brain.recall()`, `daemon_hooks` |
| 7 | Acronyms | `\b([A-Z]{2,6})\b` | DAL, API, NLP, BART, GAM |
| 8 | Expanded "the/a/this X" | Wider suffix list (60+ tech suffixes) | the supply adapter, a middleware |
| 9 | Expanded verb-object | 50+ verbs (implement, configure, deploy...) | implement the strategy pattern |

**Integration:** After collecting all candidates, pass through `filter_domain_terms()` from Change 4.

**Benchmark examples:**

| Input | Current | New |
|-------|---------|-----|
| `"Check if Clerk webhook fires"` | maybe `clerk webhook` | `Clerk` + `clerk webhook` |
| `"The DAL needs refactoring"` | nothing | `DAL` (acronym) |
| `` "Look at `brain.recall()`" `` | nothing | `brain.recall()` (backtick) |
| `"Implement the supply adapter for GAM"` | maybe `supply adapter` | `supply adapter` + `GAM` |
| `"Can you check something?"` | nothing | nothing (correctly filtered) |

**Files:**
- Modify `servers/daemon_hooks.py:594-685` — add 5 strategies + filter integration
- Create `tests/bench_vocab_extraction.py`
- Add `TestVocabExtraction` to `tests/test_core.py` (7 tests)

---

## Change 2: Sentence Splitting with pySBD

**Current** (`recall_scorer.py:190,269-270`): `re.split(r'[.!?]+', text)` — 50% accuracy.

**New**: pySBD (MIT, pure `re`, 97.9%) with code reference protection.

**Algorithm:**
1. Protect code refs (`brain.recall()`, `self.conn.execute()`), version numbers, URLs, file paths with placeholders
2. Split with `pysbd.Segmenter(language="en", clean=False)`
3. Restore placeholders
4. Fallback to `re.split` if pySBD not installed

**Benchmark examples:**

| Input | Current (broken) | New (fixed) |
|-------|-----------------|-------------|
| `"I used brain.recall() to find it. It worked."` | 3 fragments | 2 sentences |
| `"We're on v2.3.1 now. The update fixed it."` | 4 fragments | 2 sentences |
| `"Mr. Smith said fine. Dr. Jones agreed."` | 4 fragments | 2 sentences |
| `"See https://example.com/path.html for details."` | split on URL dots | 1 sentence |
| `"Fix bug in daemon_hooks.py. Vocab detection fails."` | split on `.py` | 2 sentences |

**Files:**
- Add `split_sentences()` to `servers/text_processing.py`
- Modify `servers/recall_scorer.py:190,269-270` — replace re.split calls
- Create `tests/bench_sentence_split.py`
- Add `TestSentenceSplitting` to `tests/test_core.py` (7 tests)

---

## Commit Sequence

| # | Commit | Tests | Benchmark |
|---|--------|-------|-----------|
| 1 | `split_identifier()` + wire into brain_surface.py | 8 unit tests | bench_identifier_split.py |
| 2 | Common-word filter + `data/common_words_10k.txt` | 6 unit tests | bench_common_word_filter.py |
| 3 | 5 new vocab regex strategies + filter integration | 7 unit tests | bench_vocab_extraction.py |
| 4 | pySBD sentence splitting + fallback | 7 unit tests | bench_sentence_split.py |

**Total: 28 new unit tests, 4 benchmark scripts**

---

## Risks & Mitigations

| Risk | Mitigation |
|------|-----------|
| pySBD not installed on some systems | Fallback to `re.split` — system still works |
| spaCy cold start 200ms | Not in Phase 1 — deferred to Phase 2 |
| Common-word filter misses "magic links" (head word common) | Accept limitation; other strategies (capitalized, backtick) catch many |
| 5 new regex patterns add latency | ~2-3ms total on typical message — acceptable |
| Common-word list too aggressive/permissive | Benchmark script allows empirical tuning |

---

## License Summary

| Package | License | Commercial OK |
|---------|---------|--------------|
| pySBD | MIT | ✅ |
| spaCy (Phase 2) | MIT | ✅ |
| en_core_web_sm (Phase 2) | MIT | ✅ |
| YAKE | AGPL v3 | ❌ Excluded |
| 10K word list | Public domain / MIT | ✅ |
