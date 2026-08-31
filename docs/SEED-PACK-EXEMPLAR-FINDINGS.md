# Seed-Pack Exemplar Findings — Nursery gate-4 A/B (2026-08-31)

**Status:** complete. The measurement the Nursery redesign (D-5/5.6) reserved as
its last gate. Re-runnable: `build_corpus.py --seed-pack` + `pack_quality.py`
(recipe in [EVAL-PLATFORM.md](EVAL-PLATFORM.md)).

**Ran** 2026-08-31 · new 26-node pack vs old 19-node pack ·
**corpora** `1ef8aa` (old) / `41c667` (new) ·
**sweeps** `gate4_oldpack_sweep` / `gate4_newpack_sweep` ·
local artifacts under `eval/longmem/reports/` (gitignored)

**Claim under test (P5, "the exemplar effect"):** the seed pack is the encoder's
only early catalog, so the pack's register shapes encoding quality. Predicted:
the new pack produces measurably better encoder output than the old.

**Verdict: INCONCLUSIVE on the claim — and the reason is the finding.** The
channel P5 depends on is ~90% closed in this corpus: the encoder's prompt
contains a seed in **1 of 10 items per arm**, and the three nodes explicitly
written as the encoder's curriculum reach it **zero times**. No regression was
detected on task encoding (the guardrail read held), so the new pack ships
safely — but this run does not establish that it encodes better, and the
measured deltas below are not attributable to the pack.

---

## 1. Method and arm integrity

Two arms identical except the seed pack: same 10 stratified LongMemEval items
(2 per axis), same S1E/surface config, same dates, same `s2_every_n=2`. Old pack
recovered via `git show 1a7f244^:servers/seed_pack.py` and loaded with
`build_corpus.py --seed-pack` (new capability, committed with this gate).

Arm integrity verified, not assumed:

| check | old arm | new arm |
|---|---|---|
| seed nodes per brain | 19 | 26 |
| generation marker | `eval_ext_97dbaa46` | `nursery_v1` |
| sweep-time re-seed | refused ("not born from this pack") | no-op ("already seeded") |
| build errors / RED-class | 0 / 0 | 0 / 0 |
| gold-scan matched a seed (id:9e3afc4d) | none | none |
| sweep reps flagged suspect | 0/10 | 0/10 |

The old-arm refusal is the round-2 guard fix (69bd92b) working: without it the
current pack would have gap-filled 26 nodes into the old-pack brains at sweep
open (the id:5f935ada contamination).

## 2. The finding: the exemplar channel is nearly closed

Measured directly off the build's own captured round payloads — what the encoder
literally received, not a reconstruction.

| | old pack | new pack |
|---|---|---|
| seeds appearing in any S1E prompt | 9/19 | **7/26** |
| items where the encoder saw any seed | **1/10** (`cc5ded98`) | **1/10** (`gpt4_b0863698`) |
| exposure events | 90 | 26 |
| marked exemplars reaching the encoder | — | **0** (all three) |
| `good_memory_shape` reaching the encoder | — | **0** |
| seeds reaching S1R (recall/surface) | 19/19, all items | 26/26, all items |
| seeds reaching S2 | — | 24/26 |

Three consequences:

1. **The encoder's craft teacher is its prompt, not the pack.** The S1E default
   is 110K chars and already teaches `ASSUMED/REALITY/PATTERN`, situation,
   question, reasoning, emotion, and When-triggers. `ASSUMED:` appears in ~45 of
   68 S1E payloads in *both* arms — identical, because it comes from the prompt.
   That is why field coverage is 100%/100% and correction shape is unchanged.
2. **The catalog is relevance-selected, so craft nodes can never win it.** A
   node like "What a good memory looks like" is never semantically near a
   conversation about reading habits or Tokyo airport fares. The seeds that *did*
   reach the encoder were exclusively the operator-relationship ones
   (`operator_persists`, `learn_your_operator`, `dev_curious_about_person`,
   `partnership_purpose`, …) — surfaced because the transcripts are about a
   person's habits.
3. **The new pack reaches the encoder *less* than the old one** (26 exposure
   events vs 90). Directionally a regression on P5's own terms.

**Fairness caveat — this corpus is near-worst-case for the claim.**
LongMemEval is ten third-party topic conversations with no identity, meta, or
operator-learning talk. In a real first session the seeds are topically live,
and rehearsal #1 (id:2a9aa2c7) observed the register transferring visibly. The
honest statement is that **the exemplar effect is topic-gated**, and this corpus
gates it off. P5 is not disproven; it is unmeasured here, and cannot be measured
on this corpus.

## 3. Functional read (the guardrail): no regression

| | old pack | new pack |
|---|---|---|
| raw pass | 70% (7/10) | 80% (8/10) |
| recall-conditional pass | 70% | 80% |
| ENCODE_MISS | 0 | 0 |
| answerable at build | 4/10 | 4/10 |

Per-item movement: 7 both-pass, 2 both-fail, **1 flip to pass**
(`edced276_abs`, abstention), 0 flips to fail. One item at n=10 is noise, and it
landed on an abstention item — whose test is refusing to answer, the axis least
plausibly connected to seed register. `59524333` failed in both arms with
different buckets (RECALL_MISS → ANSWER_MISS), which is run variance in the same
failure.

**Pre-registered "worse" thresholds — none tripped:** recall-conditional did not
drop, ENCODE_MISS did not rise, no seed-gold contamination, no arm-integrity
failure.

## 4. Register read: flat where the pack can't reach, shifted where it might

Aggregated over 87 (old) / 78 (new) encoder-authored nodes.

**Flat — and now explained (prompt-driven, not pack-driven):**

| metric | old | new |
|---|---|---|
| situation / question / reasoning coverage | 100% / 100% / 100% | 100% / 100% / 100% |
| When-trigger form in situations | 96.6% | 97.4% |
| edges per node | 1.54 | 1.53 |
| edge description present | 100% | 100% |
| correction shape (ASSUMED/REALITY/PATTERN) | 0 of 5 | 0 of 1 |
| emotion non-neutral | **0%** | **0%** |

Emotion is the sharpest null: both packs model honest signed emotion on every
node, the S1E prompt mentions it, and the encoder wrote `neutral`/0.0 on
**every** node in both arms. Whatever teaches emotion, neither pack does.

**Shifted toward the new pack (directional, unattributable):**

| metric | old | new |
|---|---|---|
| abstract-type share (principle/mechanism/insight/lesson/…) | 10% | **24%** |
| concrete-type share (fact/event/reference/…) | 77% | 65% |
| situation length (mean) | 157 | 178 |
| reasoning length (mean) | 218 | 238 |
| distinct types used | 14 | 16 |
| `their_raw_quote` coverage | 55.2% | 59.0% |

**Costs on the new-pack side:**

| metric | old | new |
|---|---|---|
| nodes written | 87 | 78 (−10%) |
| `correction`-type nodes | 5 | **1** |
| `my_raw_quote` coverage | 8.0% | 3.8% |
| seed share of recall candidates | 64.5% | 68.8% |
| seeds *selected* by the surfacer | **0** | **0** |

Seed crowding is real at the candidate level (+4.3pp, mechanically expected from
26 vs 19 seeds in a tiny graph) but costs nothing downstream — the surfacer
selected zero seeds in either arm.

## 5. Individual case: the corrections drop is a strategy shift, not a loss of content

Item `2311e44b` (multi-session, both arms pass, 9 encoder nodes each). The
transcript contains two arithmetic slips the assistant makes and fixes.

- **Old pack** kept the *trajectory*: two `correction` nodes ("My books/month
  calculation used September as placeholder — correct figure is ~5/month", "My
  154-day remaining-year figure was wrong — correct is ~222 days").
- **New pack** kept the *conclusion*: a `fact` with the corrected arithmetic
  (~99 pages/day needed) plus an `insight` ("50-book goal is mathematically out
  of reach at Sapiens reading pace") and a `profile` node on reading
  preferences — and zero corrections.

Both readings are defensible, and they pull from different pack instructions.
"A correction is treasure… 'we thought X, now we know Y because Z' is often
worth more than Y alone" says the old arm did better. "Climb the abstraction
ladder" says the new arm did. The content was not lost; its altitude changed —
the same shift the global type distribution shows. At n=1, and with the encoder
seeing zero seeds on this item, this is an observation, not evidence.

The one item per arm where the encoder *did* see seeds moved in **opposite**
directions (old `cc5ded98`: 10 nodes / 28 edges vs arm averages 8.6 / 11.8; new
`gpt4_b0863698`: 4 nodes / 7 edges vs 8.2 / 12.4) — no consistent signal.

## 6. What this costs and what it bought

Wall clock: 2685s + 2740s for the two builds (parallel), 109s + 113s for the two
sweeps. Roughly $60–70 of API spend including smoke runs. The encode stage is
~99% of it, which is exactly why the frozen-corpus split exists.

Bought: a verified no-regression result for a pack already shipped, two closed
contamination channels, a reusable `--seed-pack` arm capability, and the
mechanism finding above — which is worth more than the delta the gate was
designed to measure.

## 7. Recommendations

1. **Correct the P5 claim in `servers/seed_pack.py`'s docstring.** It asserts
   "the pack is the encoder's only early catalog," which is false as
   implemented — the pack is the early catalog for *recall*, and the encoder
   sees it in ~10% of items. Needs a ruling, not a silent edit: P5 was
   ratified (id:394b90f2), and the honest revision is "topic-gated," not
   "wrong."
2. **If the encoder curriculum should actually reach the encoder, that's a
   mechanism change, not a prose change** — pin the three marked exemplars
   into the young-brain catalog, gated on the same `nursery_gate_stats` the
   Zero-Memory boot block already uses. Cheap, targeted, and it would make P5
   true rather than aspirational.
3. **To actually measure the exemplar effect, build a first-session-shaped
   corpus** — identity/meta/operator-learning conversation where the seeds are
   topically live. LongMemEval cannot test this claim at any sample size.
4. Emotion's total non-transfer deserves its own look: every node in both arms
   is `neutral`/0.0 despite prompt and pack both modeling it.
