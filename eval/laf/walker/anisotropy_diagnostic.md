# Anisotropy diagnostic — can whitening un-flatten the space?

View `_primary` · 8028 nodes · 3818 queries · **label-free readouts first**

## 1. How anisotropic is it?

| measure | value | isotropic reference |
|---|---|---|
| mean cosine, random node pairs | **0.6933** | ~0.00 |
| sigma of that | 0.0434 | — |
| \|\|mean vector\|\| / mean \|\|v\|\| | **0.8325** | ~0.00 |
| PC1 variance share | **4.5%** | ~0.13% |
| PC1-8 variance share | 20.8% | ~1.04% |
| participation ratio (effective dims) | **100** | 768 |

## 2. What does the dominant direction encode?

- PC1 projection vs log(content length): r = **-0.114**

| node type | n | mean PC1 projection |
|---|---|---|
| decision | 1193 | +0.000 |
| finding | 1100 | -0.031 |
| principle | 630 | +0.090 |
| community | 620 | -0.001 |
| fact | 525 | -0.058 |
| lesson | 521 | -0.004 |
| architecture | 397 | -0.014 |
| insight | 344 | +0.093 |

## 3. Dynamic range (label-free) + 4. churn vs raw

| arm | random-pair cos | query→node cos mean | sigma | head spread (cos@1−cos@25) | top-25 overlap w/ raw |
|---|---|---|---|---|---|
| raw | 0.6933 | 0.5833 | **0.0619** | **0.0483** | 100% |
| centre | 0.0001 | 0.0000 | **0.0805** | **0.1274** | 67% |
| centre+PC1 | -0.0000 | 0.0000 | **0.0776** | **0.1272** | 62% |
| centre+PC2 | 0.0001 | 0.0000 | **0.0758** | **0.1289** | 59% |
| centre+PC4 | 0.0000 | 0.0000 | **0.0734** | **0.1287** | 55% |
| centre+PC8 | -0.0000 | 0.0000 | **0.0702** | **0.1273** | 48% |

## 5. LABEL-DEPENDENT (secondary — depends on corpus-v2 gold)

n=707 turns with a gold in this view

| arm | median gold rank | @5 | @25 |
|---|---|---|---|
| raw | 17 | 27.0% | 61.8% |
| centre | 20 | 26.4% | 56.3% |
| centre+PC1 | 22 | 25.3% | 53.2% |
| centre+PC2 | 23 | 25.3% | 52.2% |
| centre+PC4 | 23 | 24.2% | 52.2% |
| centre+PC8 | 27 | 23.5% | 49.1% |
