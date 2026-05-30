# Recall-Side Temporal Anchor — Spec

**Status:** designed, not implemented
**Filed:** 2026-05-24
**Companion:** [`TEMPORAL-ARCHITECTURE.md`](TEMPORAL-ARCHITECTURE.md) (broader temporal arc), commit [`3dd37d4`](../) (the iso_cutoff/iso_now infrastructure this builds on)

## The gap in one sentence

`brain_constants.TEMPORAL_PATTERNS` resolves "today / yesterday / last week / N days ago" in user queries against **host wall-clock** with no path for a caller to override — so eval replays of historical conversations silently mis-window every temporal recall.

## Why it didn't bite before

Until eval became a real workflow, every recall ran in production where wall-clock IS the right anchor. The bug 6d5b789e was caught and fixed on the **encoder** side ([`encode.py:104`](../servers/scales/s1/encode.py)) — but the recall classifier was never threaded with conversation_now, because no caller in production needed it.

It bites the moment eval calls `brain.recall("what happened last week")` against a 2023 haystack: "last week" resolves to 2026 wall-clock and the temporal filter returns empty.

## Today's data flow (the problem)

```
brain.recall(query)
    └─→ _recall_impl(query, ctx=None)              ← ctx available, never threaded
        └─→ _keyword_recall(query, ...)
            └─→ self._classify_intent(query)        ← query string only
                └─→ for tp in TEMPORAL_PATTERNS:
                        range_fn()                   ← zero-arg lambda
                            datetime.now(timezone.utc) ← HARDCODED wall-clock
```

The classifier never sees:
- the conversation messages
- the session's notional now
- the `[Current date: ...]` eval prefix

Even if the caller had `conversation_now()` resolved, there's no path to pass it down.

## Proposed shape

Thread an optional `current_now: Optional[datetime]` parameter through the recall stack, defaulting to wall-clock (production stays unchanged).

### API surface

```python
# servers/brain_recall.py
def recall(self, query, ..., current_now: Optional[datetime] = None, ...):
    ...

def _recall_impl(self, query, ..., current_now=None, ctx=None):
    ...

def _keyword_recall(self, query, ..., current_now=None, ...):
    ...

# servers/brain.py
def _classify_intent(self, query: str,
                    current_now: Optional[datetime] = None) -> Dict[str, Any]:
    ...
```

### Pattern builders accept the anchor

```python
# servers/brain_constants.py
def _start_of_today(now: Optional[datetime] = None):
    base = now or datetime.now(timezone.utc)
    return base.replace(hour=0, minute=0, second=0, microsecond=0)

TEMPORAL_PATTERNS = [
    {
        'pattern': re.compile(r'\btoday\b', re.IGNORECASE),
        'range_fn': lambda now=None: {
            'after': iso_now(at=_start_of_today(now))
        }
    },
    # ... 8 more, all gaining an optional `now=None` arg
]
```

### Classifier passes it through

```python
# servers/brain.py:_classify_intent
for temporal in TEMPORAL_PATTERNS:
    pattern = temporal['pattern']
    match = pattern.search(lower_query)
    if match:
        range_fn = temporal['range_fn']
        try:
            temporal_filter = range_fn(match, now=current_now)
        except TypeError:
            temporal_filter = range_fn(now=current_now)
```

### Caller resolution

Production callers (hooks, MCP) don't pass `current_now` → wall-clock default → no behavior change.

Eval callers pass:

```python
from servers.clock import conversation_now
now = conversation_now(messages=eval_msgs, session_started_at=haystack_date)
result = brain.recall("what happened last week", current_now=now)
```

The eval harness becomes the single integration point. All the threading lives inside brain_recall — eval just hands in the anchor.

## What MUST NOT change

1. **Production behavior** — every existing caller defaults to wall-clock. The default-None signature is non-breaking; commit fingerprint shifts but no behavior does.
2. **`Brain.now()` and stored `created_at`** — those stay wall-clock bookkeeping. Eval still writes "encoded today" stamps; only the **query-resolved window** moves with `current_now`.
3. **The fetch_tools' `recall_recent` / `recall_by_time` API** — those already take explicit `window` strings; they don't run through `_classify_intent`. If eval uses those, they need the same `current_now` plumbing in [`servers/scales/s1/fetch_tools.py`](../servers/scales/s1/fetch_tools.py:221) (`brain_now()` calls at lines 221, 290). Check both call paths before locking the API.

## Touch list (small)

| File | What changes |
|---|---|
| `servers/brain_constants.py` | 9 pattern lambdas + 3 `_start_of_*` helpers accept `now=` |
| `servers/brain.py` | `_classify_intent(query, current_now=None)`; pass to `range_fn` |
| `servers/brain_recall.py` | `recall()`, `_recall_impl()`, `_keyword_recall()` plumb `current_now` |
| `servers/scales/s1/fetch_tools.py` | `recall_recent`/`recall_by_time` accept `current_now=`, route through `brain_now`/`conversation_now` |
| `eval/<harness>.py` | resolve `conversation_now()` once per session, pass to every recall call |
| `tests/test_temporal_anchor_recall.py` | new — see below |

**Not touched:** the encoder write path (already correct), the S2 healer path, dashboard.

## Test strategy

1. **Unit:** mock a brain, call `_classify_intent("last week", current_now=datetime(2023,3,19,tzinfo=utc))`, assert the returned `temporalFilter['after']` is `2023-03-12T00:00:00+00:00`-anchored.
2. **Unit:** same call with `current_now=None`, assert filter anchored to wall-clock (within 2s drift).
3. **Integration:** spin up an `IsolatedBrain` from `tests/isolated_brain.py`, seed three nodes dated 2023-03-{05,12,18}, run `recall("last week", current_now=datetime(2023,3,19))`, assert only the 2023-03-12 node lands inside the window.
4. **Regression:** call `recall("last week")` with no `current_now` against the same haystack and confirm zero results (wall-clock = 2026 → no 2023 nodes in "last week of 2026"). Locks in the gap-by-default semantic so it can't accidentally start anchoring to encoding time.
5. **Contract:** extend `tests/test_time_window_contract.py` to flag any `range_fn` lambda in `brain_constants` that hardcodes `datetime.now` instead of accepting `now=`.

## Sequencing

1. Land the parameter-threading change behind a default that preserves production behavior. Zero callers update.
2. Update the eval harness to resolve `conversation_now()` and pass it.
3. Confirm via eval suite: temporal queries against historical haystacks now return non-empty.
4. After it's been alive a week, consider tightening: add a kwarg-only `current_now` on the public `Brain.recall` MCP and have the hook layer pass `brain_now(brain=brain)` explicitly. Forces every caller to declare their anchor — removes the silent wall-clock fallback. Optional hardening, not required for the eval fix.

## Out of scope (deliberately)

- **Storage-side conversation anchoring.** `created_at` stays wall-clock per the Brain.now contract. If eval ever needs nodes stamped with the historical date for ordering reasons, that's a separate spec — likely a new `event_at` column rather than overloading `created_at`, since the two questions ("when did we write the row?" vs "when did the conversation happen?") have different consumers.
- **Recall-side resolution of conversation-relative phrases in answers.** When recall renders "X days ago" deltas at result time ([`encoding_prompt.py:451`](../servers/scales/s1/encoding_prompt.py:451) territory), it already uses `brain_now()`. In eval that's wall-clock-wrong but only affects display strings, not which nodes match. Defer.
- **`TEMPORAL_PATTERNS` taxonomy expansion** ("next week", "yesterday morning", named days). Same plumbing once it exists; not a blocker.

## Risk surface

**Low.** The default-None signature is non-breaking. The classifier runs synchronously, no concurrency surface. The lambdas in `TEMPORAL_PATTERNS` are not pickled / cached / serialized anywhere I traced — they're called inline from `_classify_intent`, so adding a kwarg is safe.

One thing to watch: the `try/except TypeError` at [`brain.py:512`](../servers/brain.py:512) currently distinguishes `range_fn(match)` from `range_fn()` based on whether the pattern captured groups. After adding `now=`, the `except` branch needs `range_fn(now=current_now)` — bare `range_fn()` would still work via the default-None signature, so the worst case is a silent wall-clock fallback in the exception path. Make the kwarg explicit on both branches.

## Definition of done

- All 9 patterns and 3 helpers accept `now=None`
- `_classify_intent` accepts and forwards `current_now`
- `recall()`, `_recall_impl()`, `_keyword_recall()` plumb it end-to-end
- `fetch_tools.recall_recent` / `recall_by_time` accept `current_now=`
- Eval harness resolves `conversation_now()` per session and passes it
- 5 tests above pass
- `test_time_window_contract.py` extended (or a sibling test) catches future drift
- One commit. ~150 LOC of diff. No daemon restart needed at API level — only when changing brain.py internals.
