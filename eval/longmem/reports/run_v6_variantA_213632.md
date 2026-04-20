# LongMemEval — v6_variantA_213632

**Overall: 3/10 = 30%** (wall clock 776.4s)

## By axis

| Axis | Pass | Rate |
|---|---|---|
| ✓ abstention | 2/2 | 100% |
| ✗ info_extraction | 0/2 | 0% |
| ✗ knowledge_update | 0/2 | 0% |
| ~ multi_session | 1/2 | 50% |
| ✗ temporal | 0/2 | 0% |

## Where we're losing

### ENCODE_MISS (2) — encode — the answer never made it into the brain

- `71017276` [temporal] — How many weeks ago did I meet up with my aunt and receive the crystal chandelier?
    → The memory system never stored or indexed the chandelier event, so no retrieval query was fired; invest in improving the encoding pipeline to capture and persist personal events mentioned in conversations.
- `cc5ded98` [knowledge_update] — How much time do I dedicate to coding exercises each day?
    → The user's daily coding exercise duration was never encoded into memory, so the retrieval system had no stored information to recall; invest in improving the initial encoding pipeline to capture lifestyle and habit details.

### SURFACE_MISS (2) — surface — in candidates, but the surfacer passed it over

- `54026fce` [info_extraction] — I've been thinking about ways to stay connected with my colleagues. Any suggestions?
    → The memory retrieval system failed to surface any of the 3 available relevant memories about the user's remote work situation, team collaboration history, or previous connection attempts, indicating a retrieval ranking or relevance matching defect.
- `fca762bc` [info_extraction] — I wanted to follow up on our previous conversation about language learning apps. You menti
    → The retrieval system failed to match the query's key semantic signals (mnemonics, word/phrase memorization) to stored information about Memrise, indicating the embedding model needs retraining on language-learning app terminology.

### PARTIAL_RECALL (1) — partial recall — context delivered but the specific fact wasn't in it

- `gpt4_b086369` [temporal] — How many days ago did I participate in the 5K charity run?
    → The system retrieved only 1 of 3 relevant memories (c565c719) and selected the wrong one; invest in improving retrieval ranking to surface memories with accurate temporal information over outdated event references.

### ANSWER_MISS (2) — answer — the fact was in context, but the answerer didn't use it

- `bc149d6b` [multi_session] — What is the total weight of the new feed I purchased in the past two months?
    → Memory records lacked timestamps, preventing the system from filtering purchases by the "past two months" temporal constraint despite retrieving the correct purchase items.
- `59524333` [knowledge_update] — What time do I usually go to the gym?
    → The system retrieved only one relevant memory (73424af4) when two candidates existed, missing the memory containing the primary 6:00 pm time—invest in recall ranking to prioritize the most frequently occurring or recently confirmed gym time.

## Where to invest (ranked by impact)

**1. SURFACE_MISS** — 2 failures (info_extraction×2)

    The surfacer (Haiku) had the node in candidates and rejected it. That's a surfacer prompt / judgment issue. Review the surfacer interaction prompt for this axis — it's dropping signal it should keep.

**2. ANSWER_MISS** — 2 failures (multi_session×1, knowledge_update×1)

    Context contained the fact, answerer still failed. Abstention threshold too aggressive OR context too noisy (too many neighbors diluting signal). Review the answerer prompt.

**3. ENCODE_MISS** — 2 failures (temporal×1, knowledge_update×1)

    Look at the S1E encoder — was the detail in the conversation timeline, or did the encoder skip the turn? If detail was present, tighten the prompt's attention to that question type. If absent, the encoder gate (every 5 turns) may be missing context — consider wider message windows.

**4. PARTIAL_RECALL** — 1 failure (temporal×1)

    The surfacer delivered adjacent/general nodes but not the one carrying the specific fact — either the specific fact wasn't encoded as its own node (encoder abstraction bias — tune S1E to keep specifics), OR the specific node exists but didn't score into the top candidates (recall scoring gap for fact-oriented queries).

## Perf

- Ingest total: 721.8s (72.2s/item)
- Query S1R:    17.4s (1.7s/item)
- Answerer:     8.2s (0.8s/item)

