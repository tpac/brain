# LongMemEval — baseline_pre_scouts

**Overall: 11/20 = 55%** (wall clock 775.6s)

## By axis

| Axis | Pass | Rate |
|---|---|---|
| ~ abstention | 2/4 | 50% |
| ~ info_extraction | 3/4 | 75% |
| ~ knowledge_update | 2/4 | 50% |
| ~ multi_session | 1/4 | 25% |
| ~ temporal | 3/4 | 75% |

## Where we're losing

### PARTIAL_RECALL (8) — partial recall — context delivered but the specific fact wasn't in it

- `fca762bc` [info_extraction] — I wanted to follow up on our previous conversation about language learning apps. You menti
    → The retrieval system failed to surface the correct memory about Memrise despite having 13 relevant candidates available, indicating the ranking/scoring mechanism prioritizes irrelevant memories over the gold answer.
- `2311e44b` [multi_session] — How many pages do I have left to read in 'The Nightingale'?
    → The system retrieved 2 of 6 relevant memories but neither contained the specific page count; improve retrieval ranking to surface memories with numerical progress data about books being read.
- `bc149d6b` [multi_session] — What is the total weight of the new feed I purchased in the past two months?
    → The system retrieved only 1 of 7 relevant memories; invest in improving retrieval scoring to surface all feed purchase records within the two-month timeframe for proper aggregation.
- `2318644b` [multi_session] — How much more did I spend on accommodations per night in Hawaii compared to Tokyo?
    → The retrieval system failed to surface the Hawaii accommodation cost memory despite 8 candidate memories existing; improve the semantic matching or expand the query expansion strategy to capture travel-destination-specific expense comparisons.
- `gpt4_65aabe5` [temporal] — Which device did I set up first, the smart thermostat or the mesh network system?
    → The system retrieved 2 memories from 6 candidates but neither contained the setup order information, indicating the retrieval ranking failed to surface the relevant memory despite it existing in the system.
- `cc5ded98` [knowledge_update] — How much time do I dedicate to coding exercises each day?
    → The memory stores coding activity logs but lacks explicit time-tracking metadata; indexing should capture duration fields or add a dedicated "daily routine" category to surface time-commitment facts.
- `59524333` [knowledge_update] — What time do I usually go to the gym?
    → The system retrieved only 1 of 2 relevant memories and selected the more specific but incorrect one (7 PM Mon/Wed/Fri) instead of the general baseline time (6 PM), indicating retrieval ranking needs to prioritize frequency-based defaults over detailed variants.
- `bc8a6e93_abs` [abstention] — What did I bake for my uncle's birthday party?
    → The retrieval system retrieved 2 memories but missed the niece's birthday baking memory that should have triggered a contrastive response distinguishing it from the uncle's party, indicating the embedding similarity threshold needs lowering.

### ANSWER_MISS (1) — answer — the fact was in context, but the answerer didn't use it

- `09ba9854_abs` [abstention] — How much will I save by taking the bus from the airport to my hotel instead of a taxi?
    → The system retrieved only 2 of 3 relevant memory candidates and those selected lack user-specific context (hotel location, trip date, currency); invest in filtering logic to prioritize memories containing destination and temporal specifics.

## Where to invest (ranked by impact)

**1. PARTIAL_RECALL** — 8 failures (multi_session×3, knowledge_update×2, info_extraction×1, temporal×1, abstention×1)

    The surfacer delivered adjacent/general nodes but not the one carrying the specific fact — either the specific fact wasn't encoded as its own node (encoder abstraction bias — tune S1E to keep specifics), OR the specific node exists but didn't score into the top candidates (recall scoring gap for fact-oriented queries).

**2. ANSWER_MISS** — 1 failure (abstention×1)

    Context contained the fact, answerer still failed. Abstention threshold too aggressive OR context too noisy (too many neighbors diluting signal). Review the answerer prompt.

## Perf

- Ingest total: 3203.7s (160.2s/item)
- Query S1R:    72.8s (3.6s/item)
- Answerer:     21.7s (1.1s/item)

