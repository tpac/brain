# LongMemEval — v5_hookshared_204056

**Overall: 5/10 = 50%** (wall clock 921.5s)

## By axis

| Axis | Pass | Rate |
|---|---|---|
| ✓ abstention | 2/2 | 100% |
| ~ info_extraction | 1/2 | 50% |
| ~ knowledge_update | 1/2 | 50% |
| ~ multi_session | 1/2 | 50% |
| ✗ temporal | 0/2 | 0% |

## Where we're losing

### ENCODE_MISS (1) — encode — the answer never made it into the brain

- `71017276` [temporal] — How many weeks ago did I meet up with my aunt and receive the crystal chandelier?
    → The memory system never stored the chandelier gift event, so retrieval attempts found zero candidates; invest in encoding completeness to ensure all personal events and gifts are captured during memory formation.

### SURFACE_MISS (1) — surface — in candidates, but the surfacer passed it over

- `bc149d6b` [multi_session] — What is the total weight of the new feed I purchased in the past two months?
    → The retrieval system failed to surface any of the 3 relevant feed purchase records during recall, indicating the vector search or ranking logic cannot match purchase-intent queries to stored transaction data.

### ANSWER_MISS (3) — answer — selected and in context, but the answerer didn't use it

- `fca762bc` [info_extraction] — I wanted to follow up on our previous conversation about language learning apps. You menti
    → The memory system retrieved a general note about language learning app usage but failed to retrieve the specific previous conversation storing which app (Memrise) uses mnemonics, indicating the retrieval index needs finer semantic matching for feature-specific app comparisons.
- `gpt4_b086369` [temporal] — How many days ago did I participate in the 5K charity run?
    → The system retrieved only one of two relevant memory records (selected 1 of 2 candidates), selecting an outdated entry instead of the most recent one—invest in improving retrieval ranking to prioritize temporally recent memories.
- `59524333` [knowledge_update] — What time do I usually go to the gym?
    → The system retrieved only one of two relevant memory entries (c8e3045b), missing the entry containing the 6:00 PM time; improve retrieval ranking to surface all temporally-relevant gym memories equally.

## Where to invest (ranked by impact)

**1. ANSWER_MISS** — 3 failures (info_extraction×1, temporal×1, knowledge_update×1)

    Context was delivered, answerer failed. Either the context was too noisy (too many neighbors diluting the signal) or the answerer's abstention threshold is too aggressive. Review the answerer prompt.

**2. SURFACE_MISS** — 1 failure (multi_session×1)

    The surfacer (Haiku) had the node in candidates and rejected it. That's a surfacer prompt / judgment issue. Review the surfacer interaction prompt for this axis — it's dropping signal it should keep.

**3. ENCODE_MISS** — 1 failure (temporal×1)

    Look at the S1E encoder — was the detail in the conversation timeline, or did the encoder skip the turn? If detail was present, tighten the prompt's attention to that question type. If absent, the encoder gate (every 5 turns) may be missing context — consider wider message windows.

## Perf

- Ingest total: 866.2s (86.6s/item)
- Query S1R:    21.8s (2.2s/item)
- Answerer:     11.9s (1.2s/item)

