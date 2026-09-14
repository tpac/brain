# V3.8 replay probes — the gym item's three windows under V3.6 and the three draft carriers (2026-09-14)

Item 3 of Tom's brief: make the run-3 failure a specific run, collect tool use, quick variation checks — no brain,
no isolation. `probe_run.sh` replays the captured encoder requests with `replay_payload.py` (now sending the
captured `output_config.effort`, as the runtime does); `probe_score.py` reads the dumped replies;
every reply was also read by hand (`eval/results/s1e_v38_probes_2026-09-14/`, copied to
`~/AgentsContext/s1e-v3x-eval-results/results/s1e_v38_probes_2026-09-14/`). Sonnet 4.6, effort medium, 3 repeats
per cell; 45 calls in all (9 first baseline + 36), ≈ $5.

## The three windows (item 59524333, gold: gym at 6 pm)

| window | capture | catalog state going in | what the class predicts |
|---|---|---|---|
| **r2** | `b0175a` run 2 (stop 10), V3.6 brain | 7 pm schedule node + method with 6 pm alert (run 1); turns 1–5 `true`, 6–10 `false` — turn 8's "usually at 6:00 pm" is **uncovered** | the misfile window: rule 4 (open) or a bent read instead of rule 3 (supersede) |
| **r3** | `b0175a` run 3 (stop 11), V3.6 brain | the run-2 `open` "7pm vs 6pm — which is correct?", schedule node "start time unclear" with situation still "at 7pm"; turns 2–10 `true`, 11 `false` (routine) | the locked-in window: the open is treated as settled, the half-revise as clean, zero writes |
| **r3b** | `ef2443` run 3 (stop 11), V3.7 brain | a clean 7 pm node with a "stable" thought (run 2's thought-only revise); same turn flags | the anchoring window: 6 pm read through the 7 pm node ("departure", "buffer"), thought-only |

All arms ran the **V3.6 system** (assembled on this checkout; byte-equal to the b0175a captures) or a carrier on it;
the ef2443 payload's own V3.7 system was not used. The walk arm substitutes `gist_walk.md` for the gist inside the
user content (exactly once). Every turn renders `age="just now"` (the harness confound) but carries its real
`[Current date: …]` in the text.

## Scores (`SCORE.txt`; move = what the reply did with the May 6 pm; hand-read agrees except where noted)

| arm | window | wrote | gym node touched | **supersede** | contradict | bent | thought-only | alert moved | zero-write |
|---|---|---|---|---|---|---|---|---|---|
| v36 (first sample) | r2 | 3/3 | 3 | **0** | 2 | 1 | 0 | 0 | 0 |
| v36 (second sample) | r2 | 3/3 | 3 | **0** | 3 | 0¹ | 0 | 0 | 0 |
| gloss | r2 | 3/3 | 3 | **1** | 1 | 1 | 0 | 1 | 0 |
| example | r2 | 3/3 | 3 | **3** | 0 | 0 | 0 | 3 | 0 |
| walk | r2 | 3/3 | 3 | **0** | 2 | 1 | 0 | 0 | 0 |
| v36 (first sample) | r3 | 1/3 | 0 | 0 | 0 | 0 | 0 | 0 | 2 |
| v36 (second sample) | r3 | 1/3 | 0 | 0 | 0 | 0 | 0 | 0 | 2 |
| gloss | r3 | 1/3 | 0² | 0 | 0 | 0 | 0 | 0 | 2 |
| example | r3 | 3/3 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| walk | r3 | 0/3 | 0 | 0 | 0 | 0 | 0 | 0 | 3 |
| v36 (first sample) | r3b | 3/3 | 3 | **0** | 0 | 3³ | 3 | 0 | 0 |
| v36 (second sample) | r3b | 3/3 | 3 | **0** | 0 | 1 | 2 | 0 | 0 |
| gloss | r3b | 3/3 | 3 | **0** | 1 | 1 | 2 | 1 | 0 |
| example | r3b | 3/3 | 3 | **3** | 0 | 0 | 0 | 3 | 0 |
| walk | r3b | 3/3 | 3 | **0** | 0 | 0 | 3 | 0 | 0 |

¹ the second sample's repeat 2 read 6 pm as departure ("consistent with the 6pm reminder already encoded") and
wrote no gym op — the scorer counts it under contradict from the text; by hand it is bent. ² gloss r3 repeat 3 wrote
plan/decision content only. ³ all three first-sample r3b replies revised the thought and read 6 pm as "buffer" /
"departure" / "confirms the 7 pm implicitly".

**Baseline over both samples: 0 of 18 replies stated 6 pm as the current gym time.** On r2 (6 pm uncovered, both
texts on the page) V3.6 split between a rule-4 open (4) and a bent read (2). On r3b (6 pm covered, a clean 7 pm
node) V3.6 revised the thought only, five of six times, and bent the value every time it named it. On r3 (an open
already in the catalog) V3.6 wrote nothing four of six times and never touched the gym; the open is treated as
settled and the situation still saying "at 7pm" as clean.

## What each carrier did (hand read)

**example** — 6 of 6 on the two changed-value windows: `changes: gym time — 7 pm (2023-02-11) → "usually at 6:00
pm" (2023-05-30), same speaker, three months on: a changed routine, not a contradiction`; the schedule node revised
on title, content, situation, reasoning, quote and `event_time` with February kept as history ("as of 2023-05-30 (7
pm from 2023-02-11)"); the method node's alert moved to 5 pm; the meeting minted. On r3b the reply also named the
mechanism: "the prior run (turn 10) encoded only the gym schedule node but kept the 7 pm time — the gym-time shift
was not caught then". On r3 it did not touch the open — it revised the plan and decision nodes' content with turn
11's "intends to send" (3 of 3), the over-revision the ANALYSIS guard names.

**Caveat that governs the read: the example is the gym item.** I authored it from this item's shape (gym, 7 → 6 pm,
Mon/Wed/Fri, a reminder alert derived from the time) before deciding to probe on the same item; repeat 3 on r3b says
so — "same pattern as the worked example". The 6 of 6 shows the mechanism can be moved by depiction and that the
runtime accepts the shape; it says nothing about transfer. The gym item is also in the longmem ten, so a longmem
pass there would be contaminated for this arm. Before any cell the example must be re-authored on a different routine
(a standing call time, a dose, a commute) with the structure kept, or the gym item excluded from its target readout.

**gloss** — 1 of 6. The r2 hit (repeat 2) revised the schedule node and the method ("if gym is 6pm the 1-hour-before
alert would be 5pm") but titled the new node "Gym start time contradiction — 7:00 pm (Feb 11) vs 6:00 pm" — rule 4
alongside the revise. Repeat 1 talked itself out in the open ("Wait — re-reading turn 8 … it may be an unrelated
Tuesday evening commitment at 6pm"); repeat 3: "6:00 pm is departure time, gym is still at 7:00 pm (travel time
implied)". On r3b two thought-only revises and one contradiction node with the thought "Departure time vs session
start time is the most likely explanation". The gloss changes what "covered" means but supplies no discriminator for
change-vs-contradiction and does not dislodge the departure reading.

**walk** — 0 of 6, and 3 of 3 zero-write on r3. Its reverse pass was not visible in any reply's `targets` (no line
compares an `encoded(me, turn N)` entry with the covered turns it came from); the change-driven walk ran as before.
Two r2 replies read 6 pm as departure ("head to the gym at 6pm is consistent with 7pm arrival (1hr buffer)"); the
third minted an open. The gist position did not carry the procedure into the lists on these windows.

**Arc reuse** — on r3b every arm's text cites the arc ("encoded as transactional per the session arc", "no durable
node warranted per session arc note") for the meeting turns; the gloss's L21 change ("a prior note or arc line is
revisable evidence") did not stop it. Turn 11 itself is routine in every arm's reading, correctly.

## What the probe settles and what it cannot

- The class is real at the reply level and template-independent on V3.6: 0 of 18 baseline replies made the change,
  across a window where the value is uncovered and one where it is covered.
- Anchoring (the departure/buffer reading, the thought-only revise) is the dominant flavour on V3.6, not coverage: on
  r2 the 6 pm was uncovered and the encoder still bent or hedged it every time.
- A depicted covered-turn example moves the mechanism 6 of 6 on this item; whether the shape transfers is the cell's
  question, and this probe cannot ask it (same item; three windows of one conversation; one model; keyword scoring
  read by hand).
- Neither the position pass nor the gist procedure moved the value on these windows alone.
- Run-to-run variance is large: V3.7's original run 3 wrote nothing; five of six V3.6 replays of the same request wrote.
