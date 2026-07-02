# laf_v1 composition probe — production channels as lanes + uncapped episodic

23 cues · master 6874 · situation coverage 6874 nodes · trace matrix 8124×768

| config | need@5 | need@25 | brought | lost |
|---|---|---|---|---|
| maxsim (base) | 14% | 21% | +0 | −0 |
| stack capped (ref) | 16% | 28% | +11 | −4 |
| stack UNCAPPED | 11% | 27% | +9 | −2 |
| fts standalone | 7% | 12% | +4 | −11 |
| idf standalone | 7% | 13% | +2 | −9 |
| sit standalone | 7% | 17% | +4 | −7 |
| + fts | 8% | 17% | +3 | −7 |
| + idf | 14% | 22% | +2 | −2 |
| + sit | 14% | 20% | +2 | −2 |
| stack_u + fts | 9% | 23% | +6 | −3 |
| stack_u + idf | 15% | 25% | +7 | −2 |
| stack_u + sit | 12% | 30% | +10 | −1 |
| stack_u + fts+idf | 11% | 26% | +7 | −2 |
| stack_u + all three | 12% | 28% | +7 | −0 |
| stack_c + fts | 11% | 24% | +9 | −6 |
| stack_c + idf | 16% | 26% | +9 | −4 |
| stack_c + sit | 17% | 28% | +10 | −3 |
| stack_c + idf+sit | 18% | 27% | +10 | −3 |
| stack_c + fts@.25 | 12% | 26% | +9 | −4 |
| stack_c + sit@.25 | 16% | 27% | +9 | −3 |
| stack_c + idf@.25 | 16% | 27% | +10 | −4 |
| stack_c + u-lanes@.25 | 13% | 29% | +11 | −3 |

Uncapped scan latency: p50 66ms / p95 338ms per query (23 calls, 8124 traces)
Capped↔uncapped moment overlap (of 15): p50 4.0
