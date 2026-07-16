# q1_sweep — rank leg, registered grid (§20.5)

K0 baseline: {"auc_all": 0.7713, "auc_val": 0.8226, "auc_normal": 0.7829, "auc_flagged": 0.7666, "auc_pre_era": 0.7216, "auc_post_era": 0.8338, "soft_r": 0.173, "d_val": 0.0}

| config | ΔAUC val (June+) | AUC all | normal | flagged | pre-era | post-era | soft_r |
|---|---|---|---|---|---|---|---|
| K1-exp0.5-turnsum-zsum-opanchor-me0 | +0.0450 | 0.8446 | 0.8535 | 0.8411 | 0.8186 | 0.8770 | 0.273 |
| K1-pow1.0-turnsum-zsum-opanchor-me0 | +0.0450 | 0.8446 | 0.8535 | 0.8411 | 0.8186 | 0.8770 | 0.273 |
| K1-exp0.5-turnsum-zsum-opanchor-me0.1 | +0.0450 | 0.8445 | 0.8535 | 0.8410 | 0.8186 | 0.8770 | 0.273 |
| K1-pow1.0-turnsum-zsum-opanchor-me0.1 | +0.0450 | 0.8445 | 0.8535 | 0.8410 | 0.8186 | 0.8770 | 0.273 |
| K1-exp0.5-turnsum-zsum-opanchor-me0.3 | +0.0450 | 0.8444 | 0.8534 | 0.8409 | 0.8184 | 0.8770 | 0.273 |
| K1-pow1.0-turnsum-zsum-opanchor-me0.3 | +0.0450 | 0.8444 | 0.8534 | 0.8409 | 0.8184 | 0.8770 | 0.273 |
| K1-exp0.7-turnsum-zsum-opanchor-me0 | +0.0444 | 0.8493 | 0.8572 | 0.8462 | 0.8283 | 0.8756 | 0.286 |
| K1-exp0.7-turnsum-zsum-opanchor-me0.1 | +0.0444 | 0.8492 | 0.8572 | 0.8461 | 0.8282 | 0.8756 | 0.286 |
| K1-exp0.7-turnsum-zsum-opanchor-me0.3 | +0.0444 | 0.8492 | 0.8572 | 0.8461 | 0.8281 | 0.8756 | 0.286 |
| K2-exp0.5-turnsum-zsum-opanchor-me0.1 | +0.0422 | 0.8453 | 0.8507 | 0.8433 | 0.8230 | 0.8732 | 0.297 |
| K2-exp0.5-turnsum-zsum-opanchor-me0 | +0.0422 | 0.8453 | 0.8507 | 0.8433 | 0.8230 | 0.8732 | 0.297 |
| K2-exp0.5-turnsum-zsum-opanchor-me0.3 | +0.0422 | 0.8452 | 0.8506 | 0.8433 | 0.8229 | 0.8732 | 0.297 |
| K1-exp0.9-turnsum-zsum-opanchor-me0 | +0.0414 | 0.8500 | 0.8572 | 0.8472 | 0.8325 | 0.8720 | 0.292 |
| K1-exp0.9-turnsum-zsum-opanchor-me0.1 | +0.0414 | 0.8499 | 0.8571 | 0.8472 | 0.8325 | 0.8720 | 0.292 |
| K1-exp0.9-turnsum-zsum-opanchor-me0.3 | +0.0414 | 0.8499 | 0.8571 | 0.8471 | 0.8324 | 0.8720 | 0.292 |
| K2-exp0.3-turnsum-zsum-opanchor-me0 | +0.0411 | 0.8350 | 0.8435 | 0.8317 | 0.8040 | 0.8737 | 0.264 |
| K2-exp0.3-turnsum-zsum-opanchor-me0.1 | +0.0411 | 0.8350 | 0.8434 | 0.8316 | 0.8039 | 0.8737 | 0.264 |
| K2-exp0.3-turnsum-zsum-opanchor-me0.3 | +0.0411 | 0.8349 | 0.8433 | 0.8315 | 0.8037 | 0.8736 | 0.264 |
| K4-exp0.3-turnsum-zsum-opanchor-me0 | +0.0408 | 0.8354 | 0.8437 | 0.8321 | 0.8049 | 0.8733 | 0.269 |
| K4-exp0.3-turnsum-zsum-opanchor-me0.1 | +0.0408 | 0.8353 | 0.8436 | 0.8321 | 0.8048 | 0.8733 | 0.269 |
| K4-exp0.3-turnsum-zsum-opanchor-me0.3 | +0.0408 | 0.8352 | 0.8435 | 0.8319 | 0.8047 | 0.8733 | 0.269 |
| K8-exp0.3-turnsum-zsum-opanchor-me0 | +0.0408 | 0.8354 | 0.8437 | 0.8321 | 0.8050 | 0.8733 | 0.269 |
| K8-exp0.3-turnsum-zsum-opanchor-me0.1 | +0.0408 | 0.8353 | 0.8436 | 0.8321 | 0.8049 | 0.8733 | 0.269 |
| K8-exp0.3-turnsum-zsum-opanchor-me0.3 | +0.0408 | 0.8352 | 0.8435 | 0.8320 | 0.8047 | 0.8733 | 0.269 |
| K1-uniform-turnsum-zsum-opanchor-me0 | +0.0396 | 0.8496 | 0.8565 | 0.8470 | 0.8335 | 0.8699 | 0.294 |
| K1-uniform-turnsum-zsum-opanchor-me0.1 | +0.0396 | 0.8496 | 0.8564 | 0.8469 | 0.8335 | 0.8699 | 0.294 |
| K1-uniform-turnsum-zsum-opanchor-me0.3 | +0.0396 | 0.8495 | 0.8564 | 0.8469 | 0.8334 | 0.8698 | 0.294 |
| K1-exp0.3-turnsum-zsum-opanchor-me0 | +0.0395 | 0.8310 | 0.8411 | 0.8270 | 0.7978 | 0.8725 | 0.250 |
| K1-exp0.3-turnsum-zsum-opanchor-me0.1 | +0.0395 | 0.8310 | 0.8411 | 0.8269 | 0.7977 | 0.8725 | 0.249 |
| K1-exp0.3-turnsum-zsum-opanchor-me0.3 | +0.0395 | 0.8308 | 0.8410 | 0.8268 | 0.7975 | 0.8724 | 0.249 |
| K2-pow1.0-turnsum-zsum-opanchor-me0 | +0.0394 | 0.8435 | 0.8481 | 0.8419 | 0.8222 | 0.8701 | 0.301 |
| K2-pow1.0-turnsum-zsum-opanchor-me0.1 | +0.0394 | 0.8435 | 0.8480 | 0.8419 | 0.8222 | 0.8701 | 0.301 |
| K2-pow1.0-turnsum-zsum-opanchor-me0.3 | +0.0394 | 0.8434 | 0.8480 | 0.8418 | 0.8221 | 0.8701 | 0.301 |
| K2-pow2.0-turnsum-zsum-opanchor-me0 | +0.0387 | 0.8309 | 0.8393 | 0.8276 | 0.7984 | 0.8714 | 0.260 |
| K2-pow2.0-turnsum-zsum-opanchor-me0.1 | +0.0387 | 0.8308 | 0.8392 | 0.8275 | 0.7983 | 0.8714 | 0.260 |
| K2-pow2.0-turnsum-zsum-opanchor-me0.3 | +0.0387 | 0.8307 | 0.8391 | 0.8274 | 0.7980 | 0.8714 | 0.260 |
| K4-pow2.0-turnsum-zsum-opanchor-me0 | +0.0374 | 0.8315 | 0.8396 | 0.8283 | 0.8006 | 0.8700 | 0.274 |
| K4-pow2.0-turnsum-zsum-opanchor-me0.1 | +0.0374 | 0.8314 | 0.8396 | 0.8282 | 0.8005 | 0.8700 | 0.274 |
| K4-pow2.0-turnsum-zsum-opanchor-me0.3 | +0.0374 | 0.8313 | 0.8395 | 0.8281 | 0.8003 | 0.8699 | 0.274 |
| K4-exp0.5-turnsum-zsum-opanchor-me0 | +0.0369 | 0.8414 | 0.8465 | 0.8396 | 0.8204 | 0.8676 | 0.309 |

- configs evaluated: 673; full table: q1_sweep_full.json
- SHUFFLE CONTROL PENDING on top-3 — no verdict before it runs (registered order).

## VERDICT (pre-declared aggregate, §20.5 — recorded 2026-07-15)

**Q1: does any moment shape beat K=0 on BOTH reach and rank?**

- **Rank: YES, decisively.** Winner K1-exp0.5-turnsum-zsum-opanchor:
  ΔAUC +0.045 on the June+ holdout (0.823 → 0.868), gains in EVERY slice
  (normal +0.07, flagged +0.06, pre-era +0.10, post-era +0.04, soft_r
  0.173 → 0.273). Shuffle control HOLDS (shuffled history lands 0.04–0.07
  BELOW K0 — the gain is conversation-specific signal, not a norm artifact).
- **Reach (gold-24): NO.** Winner Δ@25 = +1 need (18/96 → 19/96), Δ@5 = −2
  (8 → 6). Inside the ±4pp noise band; the pre-committed +≥2 needs @25 is
  NOT MET. LongMemEval leg trails (adapter unbuilt) but cannot rescue the
  criterion — it required gold-24 AND LongMemEval.

**Both-or-no-ship → NO SHIP of the moment stack as a live default, as
registered.** The rank win stands as measured knowledge: the moment helps
ORDER a candidate pool; it does not (at these shapes) pull new gold into
reach on cold cues. That divergence is the surprising result, and per the
drift guards it spawns a NEW pre-registered question rather than bending
this one.
