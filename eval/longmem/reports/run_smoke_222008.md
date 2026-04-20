# LongMemEval — smoke_222008

**Overall: 3/5 = 60%** (wall clock 851.6s)

## By axis

| Axis | Pass | Rate |
|---|---|---|
| ✓ abstention | 1/1 | 100% |
| ✓ info_extraction | 1/1 | 100% |
| ✗ knowledge_update | 0/1 | 0% |
| ✓ multi_session | 1/1 | 100% |
| ✗ temporal | 0/1 | 0% |

## Where we're losing

### SURFACE_MISS (1) — surface — in candidates, but the surfacer passed it over

- `71017276` [temporal] — How many weeks ago did I meet up with my aunt and receive the crystal chandelier?
    → The retrieval system failed to return any of 4 available recall candidates matching the query about meeting an aunt and receiving a chandelier, indicating the semantic matching or ranking logic between query and stored memories needs improvement.

### PARTIAL_RECALL (1) — partial recall — context delivered but the specific fact wasn't in it

- `cc5ded98` [knowledge_update] — How much time do I dedicate to coding exercises each day?
    → The memory system retrieved a related but indirect document about progress tracking instead of the specific daily time allocation fact, indicating the indexing strategy needs to capture explicit time-commitment statements separately from general goal-setting records.

## Where to invest (ranked by impact)

**1. SURFACE_MISS** — 1 failure (temporal×1)

    The surfacer (Haiku) had the node in candidates and rejected it. That's a surfacer prompt / judgment issue. Review the surfacer interaction prompt for this axis — it's dropping signal it should keep.

**2. PARTIAL_RECALL** — 1 failure (knowledge_update×1)

    The surfacer delivered adjacent/general nodes but not the one carrying the specific fact — either the specific fact wasn't encoded as its own node (encoder abstraction bias — tune S1E to keep specifics), OR the specific node exists but didn't score into the top candidates (recall scoring gap for fact-oriented queries).

## Perf

- Ingest total: 819.3s (163.9s/item)
- Query S1R:    17.3s (3.5s/item)
- Answerer:     5.2s (1.0s/item)

