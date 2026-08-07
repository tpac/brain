# Vector-behavior scorecard — nomic_q vs gte_modernbert

Pack: 2026-08-07T10:27:13 · 2152 Door-1 cues · 2151 with gold

## A · Space geometry (label-free)

**node_primary**

| model | n | aniso(cos-rand) | σ raw | σ centred | ‖mean‖ ratio | PC1 | PR(eff dims) |
|---|---|---|---|---|---|---|---|
| nomic_q | 8393 | 0.694 | 0.0431 | 0.0987 | 0.8335 | 0.0452 | 100.5 |
| gte_modernbert | 8393 | 0.6044 | 0.0535 | 0.0976 | 0.7781 | 0.0416 | 103.2 |

**node_title**

| model | n | aniso(cos-rand) | σ raw | σ centred | ‖mean‖ ratio | PC1 | PR(eff dims) |
|---|---|---|---|---|---|---|---|
| nomic_q | 8393 | 0.6127 | 0.0446 | 0.0831 | 0.7827 | 0.0305 | 139.8 |
| gte_modernbert | 8393 | 0.5679 | 0.0542 | 0.0937 | 0.7538 | 0.0392 | 110.4 |

**node_situation**

| model | n | aniso(cos-rand) | σ raw | σ centred | ‖mean‖ ratio | PC1 | PR(eff dims) |
|---|---|---|---|---|---|---|---|
| nomic_q | 8393 | 0.6815 | 0.0418 | 0.0938 | 0.8255 | 0.0385 | 111.4 |
| gte_modernbert | 8393 | 0.5816 | 0.0535 | 0.099 | 0.7629 | 0.0441 | 99.2 |

**node_question**

| model | n | aniso(cos-rand) | σ raw | σ centred | ‖mean‖ ratio | PC1 | PR(eff dims) |
|---|---|---|---|---|---|---|---|
| nomic_q | 8385 | 0.5833 | 0.0496 | 0.0878 | 0.7638 | 0.0298 | 123.9 |
| gte_modernbert | 8385 | 0.5457 | 0.0555 | 0.1003 | 0.7386 | 0.0413 | 97.4 |

**edge_why**

| model | n | aniso(cos-rand) | σ raw | σ centred | ‖mean‖ ratio | PC1 | PR(eff dims) |
|---|---|---|---|---|---|---|---|
| nomic_q | 8000 | 0.6162 | 0.1436 | 0.4068 | 0.784 | 0.3323 | 8.4 |
| gte_modernbert | 8000 | 0.5857 | 0.147 | 0.3895 | 0.7642 | 0.2896 | 10.6 |

**episodic**

| model | n | aniso(cos-rand) | σ raw | σ centred | ‖mean‖ ratio | PC1 | PR(eff dims) |
|---|---|---|---|---|---|---|---|
| nomic_q | 6000 | 0.7302 | 0.0646 | 0.1377 | 0.8543 | 0.0756 | 53.8 |
| gte_modernbert | 6000 | 0.6023 | 0.0749 | 0.1337 | 0.7764 | 0.0719 | 60.0 |

**door1_queries**

| model | n | aniso(cos-rand) | σ raw | σ centred | ‖mean‖ ratio | PC1 | PR(eff dims) |
|---|---|---|---|---|---|---|---|
| nomic_q | 2152 | 0.5398 | 0.071 | 0.0947 | 0.7351 | 0.0461 | 107.3 |
| gte_modernbert | 2152 | 0.5595 | 0.0585 | 0.1012 | 0.7481 | 0.047 | 94.4 |


## B · Ranking behavior + E · Door-1 gold (node_primary, pure cosine)

| model | arm | Q | spread1-25 | margin@5 | margin@25 | top25σ | hub1%share | gold med | hit@5 | hit@25 | g-margin | g-z |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| nomic_q | raw | 2152 | 0.0503 | 0.00277 | 0.00062 | 0.013 | 0.1245 | 20 | 24% | 55% | -0.0459 | 2.872 |
| nomic_q | centred | 2152 | 0.1318 | 0.00717 | 0.00161 | 0.0341 | 0.074 | 27 | 22% | 48% | -0.1303 | 2.991 |
| gte_modernbert | raw | 2152 | 0.0542 | 0.00298 | 0.00069 | 0.0139 | 0.1133 | 72 | 16% | 34% | -0.0713 | 2.294 |
| gte_modernbert | centred | 2152 | 0.1276 | 0.00689 | 0.00165 | 0.033 | 0.0708 | 67 | 17% | 35% | -0.1659 | 2.471 |

## C · Proposition bands (same-topic discrimination)

| model | arm | correction pairs | topic siblings | random | topic−random gap | within-topic σ |
|---|---|---|---|---|---|---|
| nomic_q | raw | 0.8045 ±0.0635 | 0.7628 ±0.0603 | 0.694 ±0.0434 | 0.0688 | 0.0603 |
| nomic_q | centred | 0.3546 ±0.2033 | 0.2189 ±0.1858 | -0.0019 ±0.099 | 0.2208 | 0.1858 |
| gte_modernbert | raw | 0.7519 ±0.0766 | 0.6987 ±0.0762 | 0.6053 ±0.0535 | 0.0934 | 0.0762 |
| gte_modernbert | centred | 0.3569 ±0.193 | 0.2304 ±0.1833 | -0.0013 ±0.0985 | 0.2317 | 0.1833 |

## D · Multi-view redundancy (MaxSim degeneration check)

| model | title↔primary | situation↔primary | question↔primary |
|---|---|---|---|
| nomic_q | 0.8734 ±0.0333 | 0.8016 ±0.0542 | 0.7363 ±0.0614 |
| gte_modernbert | 0.84 ±0.0388 | 0.7726 ±0.0557 | 0.7446 ±0.0603 |

## F · Edge-lane conductance (query → edge-why cosine)

| model | arm | mean | σ | >0.6 |
|---|---|---|---|---|
| nomic_q | raw | 0.5039 | 0.0776 | 11.3% |
| nomic_q | centred | -0.0002 | 0.0776 | 0.0% |
| gte_modernbert | raw | 0.5252 | 0.055 | 9.3% |
| gte_modernbert | centred | 0.0002 | 0.0785 | 0.0% |

## Cross-model rank agreement (how differently they see the corpus)

| arm | Q | Jaccard@25 | Spearman(top-100 ∪) |
|---|---|---|---|
| raw | 600 | 0.1859 | -0.0907 |
| centred | 600 | 0.2088 | -0.0281 |

## G · Cost

| model | load ms | primary embed s | 1-query ms | peak RSS MB |
|---|---|---|---|---|
| nomic_q | 170 | 1591.3 | 19.7 | 5056 |
| gte_modernbert | 234 | 738.3 | 14.3 | 5441 |
