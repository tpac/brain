# LAF layering test (§18.21)

```
LAYERING — one summed field (z-scored ops), 24 cues | episodic empty: 0 | window=('window', 1)

  config                 hit@5   hit@25  | brought   lost    reinforced (vs maxsim @25)
  maxsim (base)          14%     21%     | 
  +graph                 7%      27%     | +8  −3  ↑2/14
  +episodic              15%     26%     | +9  −3  ↑6/14
  +both (full)           12%     28%     | +9  −2  ↑5/15
  +both (lighter aux)    13%     29%     | +10  −2  ↑5/15

  brought = needs the stack reaches@25 that maxsim missed · lost = maxsim reached, stack dropped
  reinforced = needs both reach but the stack ranks higher (raised) — the overlap-still-has-value test
```
