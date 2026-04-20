# LongMemEval — v8_temporal_225007

**Overall: 7/10 = 70%** (wall clock 1603.1s)

## By axis

| Axis | Pass | Rate |
|---|---|---|
| ✓ abstention | 2/2 | 100% |
| ✗ info_extraction | 0/2 | 0% |
| ~ knowledge_update | 1/2 | 50% |
| ✓ multi_session | 2/2 | 100% |
| ✓ temporal | 2/2 | 100% |

## Where we're losing

### ANSWER_MISS (3) — answer — the fact was in context, but the answerer didn't use it

- `54026fce` [info_extraction] — I've been thinking about ways to stay connected with my colleagues. Any suggestions?
    → The retrieval system found 8 relevant candidates but selected only 3, missing context about the user's previous remote work experiences and failed connection attempts that should have personalized the response.
- `fca762bc` [info_extraction] — I wanted to follow up on our previous conversation about language learning apps. You menti
    → The system retrieved general language app lists but failed to retrieve the specific memory linking Memrise to mnemonic-based learning; improve memory indexing to capture feature-to-app associations, not just app names.
- `59524333` [knowledge_update] — What time do I usually go to the gym?
    → The system retrieved only 2 out of 8 relevant memories and failed to surface the most recent 6pm conversation, indicating the retrieval ranking mechanism prioritizes older or less relevant gym time memories over recent updates.

## Where to invest (ranked by impact)

**1. ANSWER_MISS** — 3 failures (info_extraction×2, knowledge_update×1)

    Context contained the fact, answerer still failed. Abstention threshold too aggressive OR context too noisy (too many neighbors diluting signal). Review the answerer prompt.

## Perf

- Ingest total: 1530.7s (153.1s/item)
- Query S1R:    39.6s (4.0s/item)
- Answerer:     12.3s (1.2s/item)

