# component_audit — Leg A: lane × slot contributions

- unit = (lane, slot) cell; moment = the full stack fired per message slot, meshed (turn-meshing pin)
- removed = refit full-minus-unit, Δ vs full fit; alone = j0-stack + unit (history) or unit-only (j0/lane)
- both targets: picked (echo-prone) + soft-usage (answer-need); paired turn-bootstrap ×500, band = 2σ

## reference fits (June+ val)
| ref | target | val AUC | soft_r |
|---|---|---|---|
| full | picked | 0.9698 | 0.130 |
| j0 | picked | 0.9279 | 0.088 |
| full | soft | 0.7695 | 0.427 |
| j0 | soft | 0.7887 | 0.177 |

## shipped dials — winner-static & K0-static, lane → 0
| lane zeroed | winner ΔAUC val | winner Δsoft_r | K0 ΔAUC val | K0 Δsoft_r |
|---|---|---|---|---|
| maxsim | -0.0504 | +0.107 | -0.0434 | +0.062 |
| sit | -0.0228 | +0.005 | -0.0160 | +0.009 |
| idf | +0.0039 | -0.005 | +0.0122 | -0.004 |
| pick | +0.1677 | -0.006 | +0.1270 | +0.000 |
| enc | -0.0017 | +0.004 | -0.0010 | +0.006 |

(positive = the lane contributes at shipped dials — zeroing it costs that much)

## rows — lane across the whole moment
| unit | removed: ΔAUC±sd (picked) | removed: Δsoft_r±sd | alone AUC (picked) | alone soft_r | alone base | verdict |
|---|---|---|---|---|---|---|
| lane:maxsim | +0.0010 ± 0.0002 | +0.202 ± 0.003 | 0.6730 | 0.432 | none | REAL (both axes) |
| lane:sit | +0.0004 ± 0.0002 | +0.001 ± 0.000 | 0.6300 | 0.192 | none | REAL (both axes) |
| lane:idf | +0.0005 ± 0.0002 | -0.001 ± 0.000 | 0.6239 | 0.100 | none | ECHO-LEANING (picked only) |
| lane:pick | +0.2536 ± 0.0049 | -0.002 ± 0.001 | 0.9669 | 0.098 | none | ECHO-LEANING (picked only) |
| lane:enc | +0.0001 ± 0.0001 | -0.001 ± 0.000 | 0.5799 | 0.104 | none | noise / dead weight |

## columns — slot across all lanes
| unit | removed: ΔAUC±sd (picked) | removed: Δsoft_r±sd | alone AUC (picked) | alone soft_r | alone base | verdict |
|---|---|---|---|---|---|---|
| slot:j0-op | +0.0236 ± 0.0023 | +0.017 ± 0.001 | 0.9279 | 0.176 | none | REAL (both axes) |
| slot:j1-op | +0.0065 ± 0.0005 | -0.000 ± 0.000 | 0.9629 | 0.193 | j0 | ECHO-LEANING (picked only) |
| slot:j1-anchor | +0.0047 ± 0.0005 | +0.040 ± 0.002 | 0.9597 | 0.394 | j0 | REAL (both axes) |
| slot:j2-op | +0.0001 ± 0.0001 | +0.001 ± 0.000 | 0.9320 | 0.189 | j0 | noise / dead weight |
| slot:j2-anchor | +0.0006 ± 0.0002 | +0.016 ± 0.001 | 0.9403 | 0.366 | j0 | REAL (both axes) |
| slot:tail | +0.0003 ± 0.0001 | +0.006 ± 0.001 | 0.9336 | 0.316 | j0 | REAL (both axes) |

## cells — removed ΔAUC (picked) / Δsoft_r (soft) grid
| lane | j0-op | j1-op | j1-anchor | j2-op | j2-anchor | tail |
|---|---|---|---|---|---|---|
| maxsim | +0.001 / +0.010 | +0.000 / -0.000 | -0.000 / +0.030 | +0.000 / +0.001 | -0.000 / +0.013 | -0.000 / +0.005 |
| sit | +0.000 / +0.001 | +0.000 / -0.000 | -0.000 / -0.000 | -0.000 / -0.000 | +0.000 / -0.000 | +0.000 / -0.000 |
| idf | +0.000 / +0.000 | +0.000 / -0.000 | +0.000 / -0.000 | +0.000 / -0.001 | +0.000 / +0.000 | +0.000 / -0.000 |
| pick | +0.021 / +0.000 | +0.006 / +0.000 | +0.005 / -0.001 | +0.000 / -0.000 | +0.001 / -0.001 | +0.000 / +0.000 |
| enc | +0.000 / +0.000 | -0.000 / -0.000 | +0.000 / -0.000 | +0.000 / +0.000 | -0.000 / +0.000 | +0.000 / -0.001 |
- M_e_f: +0.0000 / -0.000 — noise / dead weight


### eyeball · lane:maxsim · 83869523-fc3a-40a4-ab06-e2b1e08cc5d1/0/3
| # | with unit | without unit |
|---|---|---|
| 1 | 5 local commits this session: Phase 1 through caching r s=0.75 | Session commit ledger: 4 fixes shipped — absorb savepoi s=0.84 |
| 2 | Anchor session discipline: when core works, scope out f s=0.76 | Session end: update existing docs, not create new ones  s=0.74 |
| 3 | Next steps ranked: Tier 1 (Anchor quality), Tier 2 (dat s=0.72 | Session close — 3 commits on main, not pushed, 4 commit s=0.78 |
| 4 | Session close: 10 commits shipped, 88 tests green, docs s=0.76 | Clean as you go: remove dead code same session as the m s=0.73 |
| 5 | Session handoff protocol: commit + doc + brain node (do s=0.75 | Anchor rule: check your own health before doing anythin ✓ s=0.70 |
| 6 | Session wrap open items — 2026-04-22 s=0.75 | Session close pattern: check loose threads before endin ✓ s=0.72 |
| 7 | Next-session priorities: render expansion → encoder sou s=0.75 | Session handoff protocol: commit + doc + brain node (do s=0.75 |
| 8 | Session close — 3 commits on main, not pushed, 4 commit s=0.78 | Settled work commits before in-progress work continues  s=0.76 |

### eyeball · lane:maxsim · c8b4c3fa-c0af-490e-8be3-9722b09276e9/0/18
| # | with unit | without unit |
|---|---|---|
| 1 | Confirm before execute: Tom's 1-by-1 protocol replaces  s=0.76 | S1E reconciliation committed — 6 forks fixed, 89 tests  s=0.80 |
| 2 | Anchor priors: human-seeded feedback loops beat prompt  s=0.79 | Tom confirmed Phase D dal.py split: 'lets do the 3 way' ✓ s=0.72 |
| 3 | 1-by-1 test triage discipline: what it tests, does it r s=0.78 | _do_restart fix plan: kickstart-when-managed, Popen onl s=0.77 |
| 4 | Tom: 'Other sessions are all waiting. It's all you, do  s=0.80 | S1E prompt-vs-code forks: 5 gaps between v-next draft a s=0.78 |
| 5 | Judge over-classification: action messages ('lets test  s=0.71 | Anchor persistence means sessions don't need handoffs — s=0.79 |
| 6 | S1E reconciliation committed — 6 forks fixed, 89 tests  s=0.80 | Full judge Workflow gated: pilot 5 cues first to check  s=0.76 |
| 7 | Anchor persistence means sessions don't need handoffs — s=0.79 | Anchor priors: human-seeded feedback loops beat prompt  s=0.79 |
| 8 | LAF next steps decided: temporal shuffled-gold control  s=0.77 | Confirm before execute: Tom's 1-by-1 protocol replaces  s=0.76 |

### eyeball · lane:pick · 83869523-fc3a-40a4-ab06-e2b1e08cc5d1/0/0
| # | with unit | without unit |
|---|---|---|
| 1 | recall_topical overshadowing: k=25 default, volume + po ✓ s=0.77 | 25 surface candidates are cues, not finalists: Haiku pi ✓ s=0.76 |
| 2 | 25 surface candidates are cues, not finalists: Haiku pi ✓ s=0.76 | recall_topical overshadowing: k=25 default, volume + po ✓ s=0.77 |
| 3 | S1Surface timeout root cause: surface_haiku = 55–75% of s=0.77 | v5_agentic surface loop: root cause of ~12s recall base s=0.78 |
| 4 | Recall bottleneck: candidate generation, not Haiku sele s=0.72 | TRIZ observation: select_edges pays full cost for 25 ca s=0.72 |
| 5 | Recall-quality baseline (teacher-on-production, n=90):  s=0.75 | Haiku-unpicked nodes as dynamic real-time inhibition fi s=0.73 |
| 6 | Haiku's recently-surfaced window: 5 turns, 20-node cap, s=0.80 | S1Surface timeout root cause: surface_haiku = 55–75% of s=0.77 |
| 7 | v5_agentic surface loop: root cause of ~12s recall base s=0.78 | Recall bottleneck: candidate generation, not Haiku sele s=0.72 |
| 8 | Fatigue layer coverage: three mechanisms at two distinc s=0.76 | Recall-quality baseline (teacher-on-production, n=90):  s=0.75 |

### eyeball · lane:pick · c8b4c3fa-c0af-490e-8be3-9722b09276e9/0/1
| # | with unit | without unit |
|---|---|---|
| 1 | Tom's activation criterion: if reach doesn't resolve th ✓ s=0.78 | xhigh-effort code review caught real bugs in just-commi s=0.77 |
| 2 | If I'm already doing something, do it clean — architect ✓ s=0.75 | See journal in action before building mining layer — pr s=0.79 |
| 3 | xhigh-effort code review caught real bugs in just-commi s=0.77 | S1E effort experiment: high vs medium — approved to run s=0.77 |
| 4 | Anchor rule: check your own health before doing anythin s=0.77 | Implementation effort misfiled to candidate generation  s=0.76 |
| 5 | Implementation effort misfiled to candidate generation  s=0.76 | Anchor built a risky fix before checking consumers — sh s=0.82 |
| 6 | See journal in action before building mining layer — pr s=0.79 | Anchor rule: check your own health before doing anythin s=0.77 |
| 7 | S1E effort experiment: high vs medium — approved to run s=0.77 | Tom's activation criterion: if reach doesn't resolve th ✓ s=0.78 |
| 8 | Anchor built a risky fix before checking consumers — sh s=0.82 | S1E effort A/B verdict: medium holds quality, 2.7× clea s=0.75 |

