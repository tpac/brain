# shuffle_control — §20.5 mandatory gate

- sample: 400 labeled turns (seed 20260715); donors: other-session non-machine turns; j=0 real, j≥1 donor-replaced, lanes fresh through production code
- K0 AUC on this subset: **0.7747**

| config | real AUC | shuffled AUC | shuffled − K0 | verdict |
|---|---|---|---|---|
| K1-exp0.5-turnsum-zsum-opanchor-me0 | 0.8425 | 0.7300 | -0.0447 | holds |
| K1-exp0.5-turnsum-zsum-opanchor-me0.1 | 0.8424 | 0.7299 | -0.0448 | holds |
| K1-exp0.5-turnsum-zsum-opanchor-me0.3 | 0.8423 | 0.7297 | -0.0451 | holds |
| K1-pow2.0-turnsum-zsum-op-me0 | 0.8007 | 0.7688 | -0.0059 | holds |
| K2-exp0.9-turnsum-lane-op-me0.3 | 0.8098 | 0.7110 | -0.0637 | holds |
| K1-exp0.9-turnsum-lane-opanchor-me0.1 | 0.8333 | 0.7076 | -0.0671 | holds |
| K8-pow1.0-turnmax-zsum-op-me0 | 0.7841 | 0.7268 | -0.0479 | holds |
| K2-exp0.5-turnmax-zsum-op-me0 | 0.7905 | 0.7361 | -0.0386 | holds |

**Overall: CONTROL HOLDS — history gain is not a norm/length artifact**
