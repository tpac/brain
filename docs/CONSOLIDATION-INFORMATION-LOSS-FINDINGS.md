# Consolidation Information Loss — Measured Instances

Evidence record, 2026-09-13. Found incidentally while auditing the survivor-pointer
backfill (the `laf_role_ids_unresolvable` investigation), not by a dedicated hunt.

The audit sent ten independent Sonnet judges over 207 April–May 2026 consolidation
absorbs, each blind to the encoder's stated reason, asking only: *does the survivor
actually carry the archived node's content?* The judges were looking for wrong
pairings. They found none — but they surfaced a different class: **pairings that are
correct while the content did not survive the merge.**

## What this is not

This is not a pointer-direction problem. Every pair below is a verified same-run
absorb: the survivor's revise trace and the archived node's `_sys_archived_at` stamp
land in the same transaction, 35ms–189ms apart. The forwarding pointers were written
on 2026-09-13 and resolve correctly. The question here is only what the survivor kept.

## Mechanism — already known

The mechanism is documented; this record adds measured instances, not a new theory.

| what | where |
|---|---|
| absorb keeps the survivor's content verbatim; the absorbed node's content is orphaned on the husk unless a `content=` override folds it in | node `c5f654e5` |
| `absorb` MCP description corrected from "lossless" to "content-destructive without content override" | node `ea54fe60` |
| preservation audit on a real pair (`96d2fdf8`→`426ae3cd`) — lost table schema, roadmap, structured quote field, access_count | node `988de522` |
| `source_refs` are never migrated on absorb — zero handling in `servers/scales/s2/` | node `0fc0b2eb` |
| voice/quote texture is dropped while facts (numbers/dates) are protected | node `c44e1efa` |
| consolidation journal overclaims preservation that did not happen | node `b06ca11b` |
| survivor-ladder age bias — every ranking signal is age-correlated, so the stale node wins | node `2a5b5c12` |
| merge rate collapsed ~100× between the May cold-start and May 31 — the audited window is not behaviourally uniform | node `cf89bbf1` |

## Measured instances

Ten of 207 pairs (≈5%) showed content that demonstrably did not transfer. All dates 2026.

| archived → survivor | what was lost |
|---|---|
| `f35329ac` → `9bfd8a2d` | The 13-row L2–L6 failure-layer attribution table. Survivor cites `(id:f35329ac)` and repeats only the one-line headline ("encoder causes 15% of failures"). |
| `b2a2d73c` → `66e3b75d` | MemoryAgentBench (arxiv 2507.05257), Mem2ActBench (arxiv 2601.19935), and the LongMemEval v2 release note. **See age-bias case below.** |
| `a0e2b141` → `e414e467` | Per-consumer render-format figures (`HAIKU_FORMAT` ~700 tokens / `ENCODER_FORMAT` ~2271 tokens), the `{id,title,direction,content,type}` return shape, and Tom's verbatim instruction "Stop between phases, tell me your plan before we continue". |
| `c4a33f1b` → `e35602ca` | Commit `c057c3f` entirely — 16 files, 1518 insertions / 254 deletions, the `db_maintenance.py` skeleton, silent-exception fixes, a 40s→21s timeout revert. Survivor lists 6 other commit hashes; this one is not among them. |
| `0f8f5146` → `83fcf1a3` | The 32-dimension decision-rules list (D1, D5, D7–D9, D14, D22, D29–D32 with D25–D27/D33–D36 exemptions) governing the v22 eval gate. |
| `d9a91459` → `96635da9` | Checkpoint detail: item `982b5123` FAIL and item `71017276` "still mid-encoding". Survivor's final table treats `71017276` as a completed PASS and never mentions `982b5123`. |
| `bae295c8` → `01bb19e5` | The second of two contract violations — the "invisible mode mapping" (undocumented `selected_mode` runtime contract). The inline-cfg-dict violation did carry. |
| `e0bd25fa` → `49346996` | The `ANTHROPIC_CLIENT_TIMEOUT = 180.0` decision in `s2/base.py`. Survivor's own lineage note names a different prior value (600.0 in `runner.py`, commit `02f5c32`) — this rung of the history is gone. |
| `7ffe4303` → `86f096b4` | Deferral reasoning for the Haiku `id_outside_candidates` bug: Phase B+1 timing, the S1R-surfacer vs vector-backfill contract distinction, and the "it may be a feature" hypothesis (Haiku recalling IDs from training). |
| `a66fe41b` → `5fea6323` | Not lost so much as silently contradicted: archived says 6 steps / "Step 1 next", survivor says 7 in title and 8 in body. No reconciliation recorded. |

## The age-bias case, in the wild

`b2a2d73c` is the sharpest instance and deserves its own look.

- created `2026-05-19T19:18:48`
- archived `2026-05-19T19:25:14` — **6 minutes 26 seconds after creation**
- absorbed into `66e3b75d`, created `2026-04-04` — six weeks older

The newer node carried the newer benchmarks. The older node won the survivor ladder
because every ranking signal it uses (`judge_preference`, `recall_count`, edge
richness) accumulates with age. The result is a benchmark-landscape node that is
missing the 2026 benchmarks, and a six-minute-old node archived for being young.

This is node `2a5b5c12` reproducing itself on real data.

## Two patterns worth separating

The judges flagged twelve pairs; only ten are loss. The other two are an artifact:

**Citation artifact.** When several nodes fold into one survivor, the survivor's
inline `(id:…)` names only the most recently absorbed sibling. `56ab6fe8` and
`297aff39` both went into `0c63792a` fifteen minutes apart; the survivor cites the
later one, so the earlier looks unaccounted for while its content did transfer.
14 survivors in the audited set take multiple absorbs, covering 31 of 207 pairs.

This also means any "does the survivor cite the absorbed id?" check systematically
undercounts — it can only ever confirm the last absorb into a given survivor.

## Rate

10 of 207 audited pairs (≈5%) show demonstrable loss. That is a floor, not an
estimate: the judges read a content excerpt, not full node bodies, and the citation
artifact above hides an unknown number of multi-absorb cases from this style of check.

The audited set is April–May 2026 only — the era before survivor pointers were
written at all. Whether the rate improved with the later consolidation prompt
revisions is **not measured here** and is the obvious next question.

## Open questions

1. Does the loss rate hold for June–September consolidations, or did the prompt
   revisions fix it? Same method, different date window.
2. Is the content recoverable? The absorbed node's content is intact on the archived
   husk — nothing was deleted, only dropped from the survivor. A repair pass could
   append husk content to survivors where the judge found a clean gap.
3. Does the age-bias fix (node `203d4f3a`, Fix A/B/C) actually close the
   `b2a2d73c` shape, and can that be verified against this cohort?
4. `source_refs` migration (`0fc0b2eb`) — still zero handling at time of writing;
   verify against current `servers/scales/s2/` before acting on it.

## Reproducing

All 207 per-pair verdicts with their evidence lines are in
`docs/data/consolidation-absorb-audit-2026-09-13.tsv`
(`archived_id, survivor_id, verdict, confidence, evidence`). They were produced by
content-blind judges against an isolated copy of `brain.db` — the pre-write backup,
decompressed — never against the live database.

The audited pair set is derivable from the trace substrate:

```sql
SELECT created_at, ref_id, metadata FROM trace_events
WHERE ref_type='node_revised'
  AND metadata LIKE '%s2:consolidation%'
  AND metadata LIKE '%"reason": "absorb%';
```

`ref_id` is the survivor; the 8 characters following `absorb ` in `reason` are the
absorbed node. Pair each against that node's `_sys_archived_at` in `node_metadata_kv`
to confirm the same-transaction write.
