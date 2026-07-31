# Per-view substrate census — what nanmax aggregates over

turns scored: 707 · engine nodes: 8014 · views: title, _primary, high_meta, other_meta, edge_context, question

## A. Coverage — which nodes even have the view

| view | weight (inert in nanmax) | nodes with vector |
|---|---|---|
| title | 1.00 | 100.0% |
| _primary | 0.85 | 100.0% |
| high_meta | 0.70 | 100.0% |
| other_meta | 0.40 | 100.0% |
| edge_context | 0.55 | 75.9% |
| question | 0.90 | 99.9% |

views present per node: 5 views: 1937 nodes · 6 views: 6077 nodes

## B. Scale — per-view cosine distribution vs real queries

| view | mean cos | std | mean+1σ (what it brings to a max) |
|---|---|---|---|
| title | 0.5274 | 0.0410 | 0.5684 |
| _primary | 0.5916 | 0.0359 | 0.6274 |
| high_meta | 0.5724 | 0.0410 | 0.6134 |
| other_meta | 0.5896 | 0.0343 | 0.6240 |
| edge_context | 0.5853 | 0.0383 | 0.6236 |
| question | 0.5054 | 0.0456 | 0.5511 |

## C. Which view SUPPLIES the max — gold vs top-5 non-gold

| view | wins for GOLD | wins for non-gold | gold share | non share |
|---|---|---|---|---|
| title | 28 | 49 | 4.0% | 1.5% |
| _primary | 435 | 1522 | 61.5% | 46.0% |
| high_meta | 136 | 717 | 19.2% | 21.7% |
| other_meta | 52 | 604 | 7.4% | 18.2% |
| edge_context | 41 | 351 | 5.8% | 10.6% |
| question | 15 | 67 | 2.1% | 2.0% |

## D. Headroom — gold rank per view alone vs the aggregate

| scorer | median gold rank | gold in top-5 | top-25 |
|---|---|---|---|
| title | 24 | 28.3% | 51.2% |
| _primary | 11 | 35.6% | 71.9% |
| high_meta | 38 | 23.9% | 44.6% |
| other_meta | 253 | 10.6% | 21.8% |
| edge_context | 105 | 17.1% | 34.3% |
| question | 133 | 16.4% | 31.8% |
| BEST SINGLE VIEW (oracle) | 3 | 60.4% | 88.5% |
| **maxsim (shipped, unweighted)** | 13 | 31.8% | 68.5% |
| weighted max (existing weights) | 23 | 28.4% | 51.9% |

## E. Convergence — how many views rank the gold top-25

| views agreeing | turns | share |
|---|---|---|
| 0 | 81 | 11.5% |
| 1 | 133 | 18.8% |
| 2 | 168 | 23.8% |
| 3 | 123 | 17.4% |
| 4 | 107 | 15.1% |
| 5 | 68 | 9.6% |
| 6 | 27 | 3.8% |

## F. Field-richness bias — views present, gold vs non-gold

- GOLD nodes: mean 5.84 views present (median 6)
- top-5 NON-GOLD nodes: mean 5.81 views present (median 6)
- corpus-wide: mean 5.76 views present

(non-golds richer than golds ⇒ the max is partly rewarding field-richness rather than relevance.)
