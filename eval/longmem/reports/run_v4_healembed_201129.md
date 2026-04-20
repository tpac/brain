# LongMemEval — v4_healembed_201129

**Overall: 4/5 = 80%** (wall clock 610.9s)

## By axis

| Axis | Pass | Rate |
|---|---|---|
| ✓ abstention | 1/1 | 100% |
| ✓ info_extraction | 1/1 | 100% |
| ✓ knowledge_update | 1/1 | 100% |
| ✓ multi_session | 1/1 | 100% |
| ✗ temporal | 0/1 | 0% |

## Where we're losing

### ENCODE_MISS (1) — encode — the answer never made it into the brain

- `71017276` [temporal] — How many weeks ago did I meet up with my aunt and receive the crystal chandelier?
    → The memory system never encoded the aunt meetup and crystal chandelier event, so no data existed to retrieve; invest in improving event capture during user interactions to ensure personal moments are actually stored in memory.

## Where to invest (ranked by impact)

**1. ENCODE_MISS** — 1 failure (temporal×1)

    Look at the S1E encoder — was the detail in the conversation timeline, or did the encoder skip the turn? If detail was present, tighten the prompt's attention to that question type. If absent, the encoder gate (every 5 turns) may be missing context — consider wider message windows.

## Perf

- Ingest total: 587.1s (117.4s/item)
- Query S1R:    11.4s (2.3s/item)
- Answerer:     6.9s (1.4s/item)

