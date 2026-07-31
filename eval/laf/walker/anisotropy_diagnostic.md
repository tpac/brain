# Anisotropy diagnostic — can whitening un-flatten the space?

View `_primary` · 8062 nodes · 3818 queries · **label-free readouts first**

## 1. How anisotropic is it?

| measure | value | isotropic reference |
|---|---|---|
| mean cosine, random node pairs | **0.6930** | ~0.00 |
| sigma of that | 0.0433 | — |
| \|\|mean vector\|\| / mean \|\|v\|\| | **0.8325** | ~0.00 |
| PC1 variance share | **4.5%** | ~0.13% |
| PC1-8 variance share | 20.8% | ~1.04% |
| participation ratio (effective dims) | **100** | 768 |

## 2. What does the dominant direction encode?

- PC1 projection vs log(content length): r = **-0.113**

| node type | n | mean PC1 projection |
|---|---|---|
| decision | 1197 | +0.001 |
| finding | 1105 | -0.031 |
| principle | 630 | +0.090 |
| community | 624 | -0.001 |
| fact | 526 | -0.058 |
| lesson | 522 | -0.004 |
| architecture | 397 | -0.014 |
| insight | 348 | +0.093 |

## 3. Dynamic range (label-free) + 4. churn vs raw

| arm | random-pair cos | query→node cos mean | sigma | head spread (cos@1−cos@25) | top-25 overlap w/ raw |
|---|---|---|---|---|---|
| raw | 0.6930 | 0.5833 | **0.0619** | **0.0483** | 100% |
| centre | 0.0000 | 0.0000 | **0.0804** | **0.1274** | 67% |
| centre+PC1 | -0.0002 | 0.0000 | **0.0775** | **0.1273** | 62% |
| centre+PC2 | -0.0003 | 0.0000 | **0.0758** | **0.1290** | 59% |
| centre+PC4 | -0.0004 | 0.0000 | **0.0734** | **0.1286** | 55% |
| centre+PC8 | -0.0006 | 0.0000 | **0.0701** | **0.1271** | 48% |

## 5. LABEL-DEPENDENT (secondary — depends on corpus-v2 gold)

n=707 turns with a gold in this view · nodes created after each turn are masked out (as-of honest)

| arm | median gold rank | @5 | @25 |
|---|---|---|---|
| raw | 11 | 35.6% | 71.9% |
| centre | 13 | 36.9% | 67.3% |
| centre+PC1 | 14 | 34.9% | 64.1% |
| centre+PC2 | 16 | 33.7% | 62.0% |
| centre+PC4 | 15 | 31.7% | 60.0% |
| centre+PC8 | 18 | 30.6% | 55.3% |
