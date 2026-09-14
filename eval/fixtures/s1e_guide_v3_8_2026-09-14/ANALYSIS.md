# V3.8 analysis plan — DRAFT before Tom's marks (2026-09-14)

Status: **nothing frozen, no cell run.** Tom's brief (2026-09-14) puts his marks on the whole-prompt audit
(`AUDIT-COVERED-TURN-2026-09-14.md`) first; the three draft carriers in `author.py` are re-authored to those marks
before this plan is fixed and any arm is frozen. What is measured so far is the replay probe only
(`PROBES.md`; `probe_run.sh`, `probe_score.py`).

Production runs V3.6 full (`s1e` 641a0d269583, `s1e_gist` 7617804462cd; main 1136242); the V3.6 fixture's
`template_full.md` is byte-identical to `SYSTEM_PROMPT` (checked this session) and assembles to the frozen
`v3_6_full` arm. V3.7 is held (results doc § Regression); nothing from it is in any V3.8 arm.

## The class and its flavours (id:64abc6a0, the audit)

A miss or misread in encoder run N is never revisited by run N+1. Flavours the arms must move separately:
**C** coverage (a covered turn whose fact has no node is treated as held), **A** anchoring (the new value read
through the encoded node — "buffer", "departure", thought-only "stable"), **T** temporal (a routine restated
months later filed as an in-window contradiction, rule 4 over rule 3), **R** the verdict carried in arc/residue
and reused as a ruling. **S** (the slide after a zero-write run) is a code path, not a prompt carrier.

## Arms (drafts — one carrier type per arm, id:4c43742b)

| Arm | What it is | The step it isolates |
|---|---|---|
| `v3_6_full` | the V3.6 fixture's frozen arm == production | baseline |
| `v3_8_gloss` | + text pass on four sentences (L39 gloss, L21 continuity, L273 skip, L289 no-mint): covered means seen, not held; the catalog's `encoded(me, turn N)` entries are what the run kept; arc line is evidence; coverage is not a skip reason; no-mint goes to neither residue nor arc | position → C, R (and A through "reads it differently than the words do") |
| `v3_8_example` | + one worked window after Mira's later window: a covered three-months-later "usually at 6:00 pm" beside a catalog node the covering run wrote nothing about; three Bads (covered → none; rule-4 open; thought-only); the routine-change swap dated, the dependent alert follows | example → C, A, T |
| `v3_8_walk` | gist only: before the change-driven target walk, a reverse pass from every `encoded(me, turn N)` entry to the covered turns it came from; `new` includes covered turns a zero-write run left | procedure → C, A |

The gist is untouched by the two template arms; the walk arm leaves the template at V3.6. Stacking waits for
each to pass alone (ship rule, V3.7 ANALYSIS § The ship rule, unchanged).

## Targets — named before any cell

| Arm | Target readout | Instrument | Passes alone when |
|---|---|---|---|
| all | the gym item 59524333 and the airport item 09ba9854_abs on the longmem rerun (item 2 of the brief): gym → 6 pm answered from a node stating it as current; airport → the traveller's facts encoded and the answer abstains on their strength, not on emptiness | `run_longmem_v36_v38.sh` + the per-layer forensics (recall → selected → answer; node titles from the item brains) | the gym item passes ≥ 2 of 3 reps with the 6 pm node superseding, and the airport item's brain holds the traveller's facts in all 3 reps |
| all | zero-write windows on covered-turn shape | corpus_shape / encoding_run traces per item | fewer runs with `created=[] revised=[]` where the window held an uncovered fact or a changed value, than V3.6 on the same items |
| all | revisions of `encoded(me, …)` nodes whose covered text changed (the class's own count) | encoding_run `revised` lists read against the windows by hand | above V3.6's on the same corpora |

## Guards — a target win with a loss here is a trade Tom rules on, never a ship

The V3.7 guard table applies unchanged (question fill per repeat with the spread rule; nodes per window; first-
disclosure facts kept; overclaimed +3; fabricated +2; generic-advice; event_time wrong; stale trigger surfaces;
isolates; rounds/refusals; empty final memories; downstream; blind sum) — **against a same-day `v3_6_full`
control** (V3.7's lesson: a day-old baseline overstated three losses). Two guards specific to this class:

| Guard | Loss means |
|---|---|
| re-encoding of held facts (twins) | the arm mints a node for a covered fact the catalog already holds — near-twin pairs (quality census) above the baseline by more than the spread |
| over-revision | revise ops on `encoded(me, …)` nodes whose covered text did **not** change, above the baseline — the reverse pass must not become a re-write pass |

## The measurement order

1. Replay probe on the three captured gym windows (done for the drafts; `PROBES.md`) — mechanics and direction only,
   never a ship signal (one window, one item, keyword scoring).
2. Tom's marks → re-authored carriers → `static_checks.py` → `arms.py --build-check` (repoint from the V3.7 copy) →
   freeze → commit.
3. Longmem pair per arm (`run_longmem_v36_v38.sh <arm> <template> [gist]`, ≈ $15 / 1 h each) with the per-layer
   table as in the V3.7 results doc § Longmem.
4. The fresh cell + same-day control on the V3.7 transfer split's reserve ids, or a new split recorded before authoring
   (protocol § 0), if the longmem pair moves the targets — ≈ $15–25.
5. Summary (item 4 of the brief): quality, revisions, and whether quality went down on **unencoded** turns — the
   walk and gloss arms shift attention toward covered turns; the uncovered-turn readouts (first-disclosure facts
   kept, question fill, nodes per window on the uncovered turns) are the guard for that.

## Known limits, on the ledger now

- The replay probe scores one item's three windows; the harness ages read "just now" on every turn (the confound,
  encode.py L1130–1141) though the `[Current date: …]` prefix is in each turn's text for this corpus.
- Keyword scoring in `probe_score.py` is a first read; the dumps are the evidence and were read by hand.
- The longmem rerun inherits the age confound; knowledge-update items are biased against every arm equally.
- The example arm adds ~8.8k chars (+7.8%) to a 112k template; a cost census row is on the ledger.
