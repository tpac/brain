# Ground-Truth Corpus — v22 Eval Gate

**Purpose**: hand-authored "ideal encoding" for 7 conversations across 5 strata.
The benchmark v22 (and every future encoder version) is measured against.

**Why this exists**: contracts measure deviation from a yardstick we wrote.
Ground truth measures deviation from "what we actually want." Without it, an
encoder can score the contract well and still produce nodes nobody wants.

**Authoring effort**: ~15 min per conversation × 7 = ~1.75 hours, one focused
session. Each file is fillable in place — edit the `IDEAL ENCODING` block.

## The 7 conversations

| # | File | Stratum | Coverage |
|---|---|---|---|
| 1 | `01_identity_anchor_self.md` | Identity-bearing | A6/A7 shape — Anchor self-reference, concrete tokens, anchor_raw_quote load-bearing |
| 2 | `02_identity_partnership.md` | Identity-bearing | A4 shape — mutual recognition, D14, D28 |
| 3 | `03_partnership_correction.md` | Partnership voice | A2 shape — terse pushback + Anchor reframe (D13, D7, D32) |
| 4 | `04_partnership_trust.md` | Partnership voice | A3 shape — trust formation through accurate seeing (D7, D32) |
| 5 | `05_technical_correction.md` | Technical correction | A1 shape — architectural reframe (D1, D11, D23) |
| 6 | `06_methodology_principle.md` | Methodology | B1 shape — principle/mechanism split (D3, D8, D31) |
| 7 | `07_temporal_anchoring.md` | Temporal | Multi-event sequence with dates (D10, D18, D9) |

## Authoring workflow

For each file:

1. **Read SOURCE CONVERSATION**. Skim the 4-8 turns. Replace with a real
   conversation if you have a stronger candidate (paste it in the same format,
   keeping the `[trace:<hex>]` markers).

2. **Fill IDEAL ENCODING.** What nodes SHOULD the encoder produce? For each:
   - `type`: open text — pick what fits
   - `title`: the recall handle, not the event description
   - `source_refs`: which trace hex strings from the conversation anchor it?
     Use the rule from `docs/EPISODIC-REFERENCES.md §7.4` — pure synthesis (no
     refs) / anchored synthesis (1-3 refs) / pure reference (refs carry the
     substance).
   - `edges`: list of `{target, relation, description}` — what should connect
     to what, with which verb, with what description. Use placeholder syntax
     `<descriptive-name>` for targets that would be live catalog nodes.
   - `user_raw_quote` / `anchor_raw_quote`: verbatim phrases that should be
     preserved (or empty when no specific phrase load-bears).

3. **Fill RATIONALE block.** One sentence on WHY this encoding shape is right.
   This is what the eval scores v22's output against.

## How the eval consumes these

The 3-way A/B (v22 vs v21 vs v19) runs each conversation through all three
encoder versions, then:

- **Contract eval**: scores each version's output against the 36-dim
  quality contract (`servers/scales/s1/quality_contract.py`)
- **Ground-truth eval**: scores each version's output against THIS file's
  ideal encoding — per-node match on type, source_refs choice, edges chosen,
  voice fields populated

Where contract scores and ground-truth scores diverge: that's the signal that
the contract itself needs refinement (id:153a70dc — "divergences = contract
fuzziness signal, not failure").

**Stop conditions for the eval pipeline** (decided 2026-05-25):

- Stage A (conversation 1 only): if v22 writes zero source_refs OR
  hallucinates trace ids OR scores worse than v19 on the 32 non-source_refs
  dims → STOP, diagnose, revise.
- Stage B (conversations 1+2, both identity-bearing): if v22 fails to track
  concrete identity tokens OR D27 engram_cohort signal is wrong → STOP.
- Stage C (conversations 3+4, partnership): if v22 regresses against v19 on
  D13 (pushback) — the strongest cross-era capability v21 must NOT lose →
  STOP.
- Stage D (5+6+7): full table assembly.
- Stage E: synthesis report with per-dim × per-stratum × per-version + edges
  column + worst-5-nodes-per-regression-dim with actual node text.

Identity-bearing-never-regresses is the load-bearing gate. Average wins don't
compensate for losing on the conversations that matter most to what we're
building.
