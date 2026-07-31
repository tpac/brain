# Cross-lane census — self-audit (base-rate / degree confounds)

## S1. Lane-group SUPPORT (nodes lit at z≥1.0, per turn)

| lane group | median | p25 | p75 | mean |
|---|---|---|---|---|
| current | 1414 | 1165 | 1601 | 1392 |
| episodic | 21 | 15 | 32 | 26 |
| history | 862 | 702 | 979 | 843 |

Degree: gold median 9 vs non-gold median 9 (band 6–25)

## S2+S3. Band 6–25: risk ratio vs ODDS ratio vs DEGREE-MATCHED

| verb class | lane | gold% | non% | risk ratio | ODDS ratio | matched gold% | matched non% | matched diff |
|---|---|---|---|---|---|---|---|---|
| complementary | current | 74.6% | 67.9% | 1.10× | **1.39×** | 83.1% | 73.9% | +9.2pp |
| complementary | episodic | 21.7% | 12.6% | 1.72× | **1.92×** | 24.6% | 16.9% | +7.7pp |
| complementary | history | 64.6% | 57.7% | 1.12× | **1.33×** | 72.5% | 64.2% | +8.3pp |
| hebbian | current | 37.6% | 35.5% | 1.06× | **1.10×** | 40.8% | 41.5% | -0.7pp |
| hebbian | episodic | 16.4% | 11.3% | 1.45× | **1.54×** | 16.2% | 15.3% | +0.9pp |
| hebbian | history | 34.9% | 32.4% | 1.08× | **1.12×** | 37.3% | 38.4% | -1.1pp |
| similarity | current | 27.0% | 30.4% | 0.89× | **0.85×** | 28.2% | 30.6% | -2.5pp |
| similarity | episodic | 5.8% | 4.0% | 1.44× | **1.46×** | 6.3% | 5.1% | +1.2pp |
| similarity | history | 22.8% | 24.3% | 0.94× | **0.92×** | 25.4% | 26.2% | -0.8pp |
| corrective_strict | current | 10.1% | 9.8% | 1.02× | **1.02×** | 12.7% | 10.9% | +1.8pp |
| corrective_strict | episodic | 0.0% | 1.3% | 0.00× | **0.21×** | 0.0% | 1.8% | -1.8pp |
| corrective_strict | history | 9.0% | 7.5% | 1.19× | **1.21×** | 11.3% | 9.0% | +2.3pp |

### Divergence check (episodic vs current, same verb class)

| verb class | risk-ratio divergence | ODDS divergence | matched-diff episodic | matched-diff current |
|---|---|---|---|---|
| complementary | 1.57× | **1.39×** | +7.7pp | +9.2pp |
| hebbian | 1.37× | **1.40×** | +0.9pp | -0.7pp |
| similarity | 1.62× | **1.73×** | +1.2pp | -2.5pp |
| corrective_strict | 0.00× | **0.20×** | -1.8pp | +1.8pp |

matched pairs: 142 turns
