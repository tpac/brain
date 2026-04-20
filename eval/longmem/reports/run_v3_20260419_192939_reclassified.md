# LongMemEval — v3_20260419_192939

**Overall: 4/10 = 40%** (wall clock 1200.0s)

## By axis

| Axis | Pass | Rate |
|---|---|---|
| ✓ abstention | 2/2 | 100% |
| ✗ info_extraction | 0/2 | 0% |
| ~ knowledge_update | 1/2 | 50% |
| ~ multi_session | 1/2 | 50% |
| ✗ temporal | 0/2 | 0% |

## Where we're losing

### ENCODE_MISS (1) — encode — the answer never made it into the brain

- `71017276` [temporal] — How many weeks ago did I meet up with my aunt and receive the crystal chandelier?
    → The memory system never encoded the chandelier meeting event, so no retrieval query could find it; invest in improving the encoding pipeline to capture personal events mentioned in conversations before they're needed for recall.

### SURFACE_MISS (3) — surface — in candidates, but the surfacer passed it over

- `fca762bc` [info_extraction] — I wanted to follow up on our previous conversation about language learning apps. You menti
    → The retrieval system failed to surface the stored conversation about Memrise despite a query firing, indicating the embedding similarity between "mnemonics to help learners memorize words" and previous Memrise discussion was below the retrieval threshold.
- `2311e44b` [multi_session] — How many pages do I have left to read in 'The Nightingale'?
    → The memory retrieval system found 1 relevant candidate but failed to select it, indicating the ranking or filtering mechanism incorrectly rejected valid memories about your reading progress in 'The Nightingale'.
- `59524333` [knowledge_update] — What time do I usually go to the gym?
    → The retrieval system found 2 candidate memories but failed to select any of them for context, indicating the ranking or filtering mechanism incorrectly rejected relevant gym schedule information.

### ANSWER_MISS (2) — answer — selected and in context, but the answerer didn't use it

- `54026fce` [info_extraction] — I've been thinking about ways to stay connected with my colleagues. Any suggestions?
    → The memory system failed to retrieve or synthesize general colleague connection strategies from existing memories (virtual coffee shops, team collaboration patterns), indicating the retrieval mechanism lacks sufficient abstraction to generalize specific examples into broader applicable guidance.
- `gpt4_b086369` [temporal] — How many days ago did I participate in the 5K charity run?
    → The memory stored "just now" instead of the actual date (2023/03/19), so the system couldn't calculate the elapsed time—invest in enforcing timestamp capture at memory creation rather than relative temporal expressions.

## Where to invest (ranked by impact)

**1. SURFACE_MISS** — 3 failures (info_extraction×1, multi_session×1, knowledge_update×1)

    The surfacer (Haiku) had the node in candidates and rejected it. That's a surfacer prompt / judgment issue. Review the surfacer interaction prompt for this axis — it's dropping signal it should keep.

**2. ANSWER_MISS** — 2 failures (info_extraction×1, temporal×1)

    Context was delivered, answerer failed. Either the context was too noisy (too many neighbors diluting the signal) or the answerer's abstention threshold is too aggressive. Review the answerer prompt.

**3. ENCODE_MISS** — 1 failure (temporal×1)

    Look at the S1E encoder — was the detail in the conversation timeline, or did the encoder skip the turn? If detail was present, tighten the prompt's attention to that question type. If absent, the encoder gate (every 5 turns) may be missing context — consider wider message windows.

## Perf

- Ingest total: 1149.5s (115.0s/item)
- Query S1R:    22.5s (2.2s/item)
- Answerer:     9.1s (0.9s/item)

