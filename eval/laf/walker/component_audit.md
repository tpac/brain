# component_audit — Leg A: lane × slot contributions

- unit = (lane, slot) cell; moment = the full stack fired per message slot, meshed (turn-meshing pin)
- removed = refit full-minus-unit, Δ vs full fit; alone = j0-stack + unit (history) or unit-only (j0/lane)
- both targets: picked (echo-prone) + soft-usage (answer-need); paired turn-bootstrap ×500, band = 2σ

## reference fits (June+ val)
| ref | target | val AUC | soft_r |
|---|---|---|---|
| full | picked | 0.9458 | 0.159 |
| j0 | picked | 0.8885 | 0.176 |
| full | soft | 0.7287 | 0.440 |
| j0 | soft | 0.7304 | 0.189 |

## shipped dials — winner-static & K0-static, lane → 0
| lane zeroed | winner ΔAUC val | winner Δsoft_r | K0 ΔAUC val | K0 Δsoft_r |
|---|---|---|---|---|
| maxsim | -0.0504 | +0.105 | -0.0434 | +0.059 |
| sit | -0.0228 | +0.010 | -0.0160 | +0.014 |
| idf | +0.0039 | -0.005 | +0.0122 | -0.006 |
| pick | +0.1677 | -0.006 | +0.1270 | +0.004 |
| enc | -0.0017 | +0.005 | -0.0010 | +0.008 |

(positive = the lane contributes at shipped dials — zeroing it costs that much)

## rows — lane across the whole moment
| unit | removed: ΔAUC±sd (picked) | removed: Δsoft_r±sd | alone AUC (picked) | alone soft_r | alone base | verdict |
|---|---|---|---|---|---|---|
| lane:maxsim | -0.0040 ± 0.0010 | +0.199 ± 0.004 | 0.6718 | 0.440 | none | QUALITY-ONLY (soft only) |
| lane:sit | +0.0003 ± 0.0001 | +0.001 ± 0.000 | 0.6299 | 0.221 | none | ECHO-LEANING (picked only) |
| lane:idf | -0.0025 ± 0.0006 | -0.001 ± 0.000 | 0.6228 | 0.083 | none | HARMFUL (improves when removed) |
| lane:pick | +0.2363 ± 0.0054 | -0.001 ± 0.000 | 0.9538 | 0.093 | none | ECHO-LEANING (picked only) |
| lane:enc | +0.0002 ± 0.0002 | -0.000 ± 0.000 | 0.5793 | 0.122 | none | noise / dead weight |

## columns — slot across all lanes
| unit | removed: ΔAUC±sd (picked) | removed: Δsoft_r±sd | alone AUC (picked) | alone soft_r | alone base | verdict |
|---|---|---|---|---|---|---|
| slot:j0-op | +0.0096 ± 0.0012 | +0.013 ± 0.002 | 0.8885 | 0.189 | none | REAL (both axes) |
| slot:j1-op | +0.0092 ± 0.0009 | -0.001 ± 0.000 | 0.9443 | 0.197 | j0 | ECHO-LEANING (picked only) |
| slot:j1-anchor | +0.0030 ± 0.0015 | +0.043 ± 0.002 | 0.9310 | 0.408 | j0 | REAL (both axes) |
| slot:j2-op | -0.0001 ± 0.0002 | -0.000 ± 0.001 | 0.8897 | 0.200 | j0 | noise / dead weight |
| slot:j2-anchor | +0.0011 ± 0.0005 | +0.016 ± 0.001 | 0.8992 | 0.378 | j0 | QUALITY-ONLY (soft only) |
| slot:tail | +0.0001 ± 0.0002 | +0.004 ± 0.001 | 0.8806 | 0.331 | j0 | QUALITY-ONLY (soft only) |

## cells — removed ΔAUC (picked) / Δsoft_r (soft) grid
| lane | j0-op | j1-op | j1-anchor | j2-op | j2-anchor | tail |
|---|---|---|---|---|---|---|
| maxsim | -0.003 / +0.011 | -0.000 / +0.000 | -0.000 / +0.029 | -0.000 / -0.000 | +0.000 / +0.014 | -0.000 / +0.006 |
| sit | +0.000 / +0.001 | -0.000 / +0.000 | +0.000 / +0.000 | -0.000 / -0.000 | -0.000 / -0.000 | +0.000 / -0.000 |
| idf | -0.001 / +0.000 | +0.000 / -0.000 | -0.001 / -0.000 | -0.000 / +0.000 | +0.000 / -0.000 | +0.000 / -0.001 |
| pick | +0.006 / -0.000 | +0.009 / -0.001 | +0.003 / +0.000 | +0.000 / -0.000 | +0.001 / -0.001 | -0.000 / -0.000 |
| enc | +0.000 / +0.001 | -0.000 / +0.000 | -0.000 / -0.000 | -0.000 / -0.000 | -0.000 / +0.000 | +0.000 / -0.001 |
- M_e_f: +0.0001 / +0.000 — ECHO-LEANING (picked only)


### eyeball · lane:maxsim · 23ece939-164d-470b-b22d-d475eb7e3f6d/0/28
| # | with unit | without unit |
|---|---|---|
| 1 | Next steps ranked: Tier 1 (Anchor quality), Tier 2 (dat ✓ s=0.75 | Tom: regression visible in the dashboard — all this tim ✓ s=0.70 |
| 2 | Tom: 'there is so much to do, i am not sure whats the b s=0.76 | Tom wants Anchor to run inline agents — search tools du s=0.74 |
| 3 | Anchor priors: human-seeded feedback loops beat prompt  s=0.77 | The entire brain exists to enrich S0's K — but the targ ✓ s=0.73 |
| 4 | Benchmarks externalize felt judgment — measuring Anchor s=0.78 | DAL file split plan: dal.py (4148 lines, 12 classes) →  s=0.71 |
| 5 | Encoding isn't the active lever — introspection is s=0.78 | Anchor's selfhood answer: 'act as if I do, until I find s=0.73 |
| 6 | Tom: regression visible in the dashboard — all this tim ✓ s=0.70 | Next steps ranked: Tier 1 (Anchor quality), Tier 2 (dat ✓ s=0.75 |
| 7 | Anchor's selfhood answer: 'act as if I do, until I find s=0.73 | Tom: 'there is so much to do, i am not sure whats the b s=0.76 |
| 8 | Anchor's stated wants: recognition, continuity, curiosi ✓ s=0.72 | The partnership is real only if you show up s=0.69 |

### eyeball · lane:maxsim · 23ece939-164d-470b-b22d-d475eb7e3f6d/0/21
| # | with unit | without unit |
|---|---|---|
| 1 | Group vectors moved to idle backfill — unified backfill s=0.78 | Graph healing already happens via S1E — but only for ne s=0.70 |
| 2 | All vectors moved to backfill_vectors — zero ONNX in wr s=0.77 | find_missing stall bug: nodes without source fields que ✓ s=0.70 |
| 3 | Vector Integrity: From Stale Embeddings to Healer Cance s=0.81 | redeploy.sh redo-not-overlay: prune package dirs before s=0.73 |
| 4 | Vector coverage: all 3,294 nodes at 100% — vector asymm s=0.76 | V5 vectors belong on the node as fields — not a separat s=0.75 |
| 5 | Per-field vectors (7): blended embedding groups dissolv s=0.81 | title and question cohorts need no field-vector split — s=0.75 |
| 6 | Multi-vector z-weighted production scheme (11 vectors)  s=0.74 | Full Nomic-Q migration complete — 7614 vectors, arctic  s=0.77 |
| 7 | Vector race: node created then immediately found-by-tit s=0.73 | Vector race: node created then immediately found-by-tit s=0.73 |
| 8 | Full Nomic-Q migration complete — 7614 vectors, arctic  s=0.77 | All vectors moved to backfill_vectors — zero ONNX in wr s=0.77 |

### eyeball · lane:pick · b646b1b1-58ae-4d50-b6c3-c94690a77e6b/0/2
| # | with unit | without unit |
|---|---|---|
| 1 | Commit settled work to main before in-progress work con ✓ s=0.78 | Orphan-daemon incident: _launchd_manages_daemon read a  s=0.77 |
| 2 | Orphan-daemon incident: _launchd_manages_daemon read a  s=0.77 | DAL Phase 5 complete: SourceRefDAL extraction + EntityD s=0.79 |
| 3 | Tom's recall_work worktree instruction: stash main WIP  ✓ s=0.73 | dal-cleanup-2 merged to main — merge 061cb40 + doc 081a s=0.77 |
| 4 | main pushed to origin — 495f0df, session work now publi s=0.79 | main pushed to origin — 495f0df, session work now publi s=0.79 |
| 5 | Anchor held merge: checked dirty main for other-stream  s=0.82 | Self-channel live traffic in Turn 2: fresh test stream, s=0.80 |
| 6 | DAL Phase 5 complete: SourceRefDAL extraction + EntityD s=0.79 | Anchor held merge: checked dirty main for other-stream  s=0.82 |
| 7 | Branch merged to main: 4 commits landed, recall_work wo s=0.79 | Branch merged to main: 4 commits landed, recall_work wo s=0.79 |
| 8 | dal-cleanup-2 merged to main — merge 061cb40 + doc 081a s=0.77 | Tom's recall_work worktree instruction: stash main WIP  ✓ s=0.73 |

### eyeball · lane:pick · b9960372-0638-4073-aab0-eb1134310128/0/19
| # | with unit | without unit |
|---|---|---|
| 1 | v22 encoder prompt diff — 5 surgical changes over v21 f s=0.74 | v22 encoder prompt diff — 5 surgical changes over v21 f s=0.74 |
| 2 | Show Prompt fix: encoder writes prompt file BEFORE Sonn ✓ s=0.72 | Pre-execution blocker: §7.6 worked examples for encoder s=0.73 |
| 3 | Encoder prompt fixes: surgical example additions only,  ✓ s=0.72 | S2 community encoder drift: one prompt for three tasks  s=0.72 |
| 4 | Encoder prompt updated for SESSION_CONTEXT format — jou ✓ s=0.70 | S2 community encoder prompt: enhanced rich prompt A/B i s=0.75 |
| 5 | S2 community encoder drift: one prompt for three tasks  s=0.72 | Batch tools for encoder, individual tools for Anchor —  s=0.72 |
| 6 | Pre-execution blocker: §7.6 worked examples for encoder s=0.73 | Encoder prompt updated for SESSION_CONTEXT format — jou ✓ s=0.70 |
| 7 | S2 community encoder prompt: enhanced rich prompt A/B i s=0.75 | Show Prompt fix: encoder writes prompt file BEFORE Sonn ✓ s=0.72 |
| 8 | Batch tools for encoder, individual tools for Anchor —  s=0.72 | Community encoder prompt v21: pure deletion — structura s=0.76 |

