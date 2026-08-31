# laf_v1 composition probe — production channels as lanes + uncapped episodic

23 cues · master 10231 · situation coverage 10231 nodes · trace matrix 17931×768

| config | need@5 | need@25 | brought | lost |
|---|---|---|---|---|
| maxsim (base) | 14% | 21% | +0 | −0 |
| stack capped (ref) | 13% | 28% | +11 | −4 |
| stack UNCAPPED | 11% | 26% | +8 | −2 |
| fts standalone | 8% | 12% | +4 | −11 |
| idf standalone | 7% | 15% | +3 | −8 |
| sit standalone | 7% | 17% | +4 | −7 |
| + fts | 8% | 15% | +3 | −8 |
| + idf | 14% | 22% | +2 | −2 |
| + sit | 14% | 19% | +1 | −2 |
| stack_u + fts | 9% | 21% | +7 | −6 |
| stack_u + idf | 15% | 25% | +7 | −2 |
| stack_u + sit | 13% | 28% | +10 | −2 |
| stack_u + fts+idf | 10% | 26% | +8 | −3 |
| stack_u + all three | 13% | 27% | +9 | −3 |
| stack_c + fts | 12% | 23% | +9 | −7 |
| stack_c + idf | 16% | 26% | +9 | −4 |
| stack_c + sit | 15% | 28% | +10 | −3 |
| stack_c + idf+sit | 17% | 27% | +10 | −3 |
| stack_c + fts@.25 | 11% | 26% | +10 | −4 |
| stack_c + sit@.25 | 13% | 27% | +9 | −3 |
| stack_c + idf@.25 | 14% | 26% | +9 | −3 |
| stack_c + u-lanes@.25 | 12% | 28% | +11 | −3 |

Uncapped scan latency: p50 81ms / p95 507ms per query (23 calls, 17931 traces)
Capped↔uncapped moment overlap (of 15): p50 3.0
