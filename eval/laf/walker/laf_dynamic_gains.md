# Dynamic gains — conditional-gain regression (held-out)

n=707 clean valids · φ = 12 per-message features (maxsim_peak, maxsim_gap, maxsim_std, sit_peak, idf_peak, log_idf_sup, log_pick_sup, log_enc_sup, log_graph_sup, graph_peak, graph_conv2, graph_maxconv) · model score=Σ_lane[w0+Σ w_j φ_j]·z_lane · L2 λ=1.0 · 5-fold session-CV

VERDICT BAR: COND must beat best-FIXED **held-out** AND beat SHUFFLE; echo-ablation (drop pick/enc) must not collapse the win.

## Held-out pooled AUC (session-CV)

| model | lanes | picked-AUC | soft-AUC |
|---|---|---|---|
| best-FIXED | 6 | 0.8904 | 0.8116 |
| CONDITIONAL | 6 | 0.9110 | 0.7350 |
| COND · SHUFFLE φ | 6 | 0.8678 | 0.7721 |
| COND · echo-ablate | 4 | 0.6842 | 0.7126 |
| FIXED · echo-ablate | 4 | 0.7094 | 0.7892 |

## reach@5 — conditional gains applied on reach substrate
(in-sample gains; a ceiling read, not held-out reach)

| composition | reach@5 |
|---|---|
| shipped fixed gains | 51.2% |
| conditional (soft-fit) | 50.6% |
| conditional (picked-fit) | 38.2% |

## Verdict

- COND vs FIXED (held-out): picked +0.0207 · soft -0.0766
- COND vs SHUFFLE (held-out): picked +0.0432 · soft -0.0371
- echo-ablate COND: picked 0.6842 soft 0.7126 (vs full COND 0.9110/0.7350)

**NULL — conditional ≈ fixed/shuffle held-out; rich features do NOT capture generalizable dynamic gains (extends the router null 701d86f8 to a large feature set)**

