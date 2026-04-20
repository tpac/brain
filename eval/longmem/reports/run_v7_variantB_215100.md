# LongMemEval — v7_variantB_215100

**Overall: 6/10 = 60%** (wall clock 710.1s)

## By axis

| Axis | Pass | Rate |
|---|---|---|
| ✓ abstention | 2/2 | 100% |
| ✓ info_extraction | 2/2 | 100% |
| ~ knowledge_update | 1/2 | 50% |
| ~ multi_session | 1/2 | 50% |
| ✗ temporal | 0/2 | 0% |

## Where we're losing

### ENCODE_MISS (2) — encode — the answer never made it into the brain

- `71017276` [temporal] — How many weeks ago did I meet up with my aunt and receive the crystal chandelier?
    → The memory system never stored or indexed the aunt meetup and chandelier gift event, so no retrieval query could find it; invest in improving the encoding pipeline to capture personal events from initial user messages.
- `59524333` [knowledge_update] — What time do I usually go to the gym?
    → The memory system never executed a query despite receiving a retrievable question, indicating a retrieval trigger failure—invest in fixing the query execution logic that determines when to search memory.

### PARTIAL_RECALL (2) — partial recall — context delivered but the specific fact wasn't in it

- `2311e44b` [multi_session] — How many pages do I have left to read in 'The Nightingale'?
    → The system retrieved only 1 of 2 relevant memories; the missing memory likely contained the current page number needed to calculate remaining pages, requiring improved ranking to surface all relevant context.
- `gpt4_b086369` [temporal] — How many days ago did I participate in the 5K charity run?
    → The memory record for the 5K event stored the completion time but omitted the date; storage procedures must capture all event metadata fields, not just performance metrics.

## Where to invest (ranked by impact)

**1. PARTIAL_RECALL** — 2 failures (multi_session×1, temporal×1)

    The surfacer delivered adjacent/general nodes but not the one carrying the specific fact — either the specific fact wasn't encoded as its own node (encoder abstraction bias — tune S1E to keep specifics), OR the specific node exists but didn't score into the top candidates (recall scoring gap for fact-oriented queries).

**2. ENCODE_MISS** — 2 failures (temporal×1, knowledge_update×1)

    Look at the S1E encoder — was the detail in the conversation timeline, or did the encoder skip the turn? If detail was present, tighten the prompt's attention to that question type. If absent, the encoder gate (every 5 turns) may be missing context — consider wider message windows.

## Perf

- Ingest total: 661.8s (66.2s/item)
- Query S1R:    17.5s (1.7s/item)
- Answerer:     10.9s (1.1s/item)

