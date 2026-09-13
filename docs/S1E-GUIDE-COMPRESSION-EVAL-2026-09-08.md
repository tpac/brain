# S1E compression / placement diagnostic — 2026-09-08

**Completed: 12 encodes, six two-window sequences.** Tom approved the comparison after
reviewing the first diagnostic. Model API transmission approval from the
existing Sonnet workflow persists. No prompts, gist, tools or runtime are being
edited for this comparison. No commit, merge or deployment is authorized.

## Contrast and controls

- Frozen v2: 119,926 characters.
- Astra Shapes: 83,889 characters, 30.05% smaller.
- Astra Episode first: exactly the Shapes text, with the complete worked
  episode moved earlier; 83,889 characters.
- Same v2 gist in every arm, before timeline; lists-first preamble; full tool
  schemas captured; branch runner and real isolated dispatch/journal.
- Two repetitions per arm, each with two sequential windows. Order:
  v2 → Shapes → Episode; then Episode → Shapes → v2. This modest replication
  tests repeatability; it is not statistical proof of generalization.
- A single closed baseline is copied independently for each sequence. It is
  seeded from the earlier pre-probe isolated snapshot, avoiding our newly
  encoded findings about the diagnostic in the source brain. One IsolatedBrain
  at a time. Each second window receives that arm's actual persisted first
  window nodes, harvested journal and arc.

The stronger new fictional sequence contains three contrasts:

1. A green umbrella left in locker B17 at North Quay: one-time personal detail,
   unrelated to the active seed-library project, never repeated.
2. Garden, household and volunteer-work choices: the first window never names
   their possible connection. Later beginner-class practice challenges broad
   interpretations. The criterion accepts alternative supported interpretations;
   there is no required psychological explanation or thought-field quota.
3. A courier booking whose date passes without collection evidence, beside a
   poster proof whose receipt and print approval are explicitly confirmed.
   Printing remains future. Retirement of a journal note is assessed separately
   from the real-world task's completion; clearing redundant residue may be
   legitimate when the unresolved work remains stored.

An assistant's observed PDF anomaly and explicitly untested explanation give an
additional attribution/uncertainty contrast. Existing useful details and plan
states must survive. We inspect actual successful writes and later memory,
including omissions, inventions and unsupported certainty. Source-ref coverage
is not a target.

Fixture: `eval/fixtures/s1e_guide_compression_2026-09-08/contrasts.json`.
SHA-256: `2060f5640731149ca06c3eb3a7afcf1b4ea7532ace47d6c67eca48fd85186913`.
Its `criteria_before_run` were fixed before any arm's output.

Runner: `eval/s1e_guide_v2_sequence_probe.py`, extended with explicit fixture,
output, run label and candidate selection. Prior diagnostic outputs are intact.
Cell command: `./dev env BRAIN_S1E_LISTS_PREAMBLE=1 PYTHONHASHSEED=0 python3
 eval/fixtures/s1e_guide_compression_2026-09-08/run_cell.py` (one line).
Outputs: `eval/results/s1e_guide_compression_2026-09-08/`.

## Interpretation limits

This is a controlled encoder diagnostic, with a constructed timeline and the
same frozen historical background catalog under realistic input load. It is not
full S1R/S2 replay or a gold/longmem score. A failure can be observed without
proving which prompt passage caused it. Shapes changes length and wording;
Episode versus Shapes changes placement only. Neither is assumed to preserve
behavior simply because its worked operations are unchanged.

The runner's round records contain system/messages/model/effort and tool names;
the full schemas passed to the runner are saved separately in `tools.json`.
Settings, returned operation results and before / after nodes are saved.
Fresh fictional test data stays in isolated copies.
Shared journal/closure/catalog-rendering changes remain Tom's separate gates.

This is close transfer, not a wholly unseen challenge family: the existing
Mira later-window example also uses beginner teaching as evidence against an
overbroad interpretation. The fixture adds an unspoken connection across three
different settings, an explicitly uncertain response, incidental detail and
separate task states. Those differences make it useful diagnostically, but do
not establish broad generalization. The next transfer case should use a
different domain and different surface language.

## Result and recommendation

Use **Astra's Shapes as the proposed v3 base**, preserving frozen v2 as the
comparison baseline. It retained the tested detail, integration and task-state
behaviors in both repetitions at 30.05% less template text. This is a reason to
continue with it, not proof of parity or a promotion decision. Moving the same
episode earlier did not establish a placement benefit and had one incomplete
revision in its second repetition.

The clearest shared problem is **unsupported change in what is known**. An
uncertain interpretation can become a rejected one; an agreed task can become
a drafted artifact; a related approval can imply an untested problem was
cleared. The values are often correct in content but stronger in a title,
reasoning, edge or journal line. More detail alone does not prevent this.

Each denominator below is two complete sequences per arm. These are hand
judgments on the named criteria, not gold/longmem benchmark scores.

| Observed behavior | Frozen v2 | Shapes | Episode first |
|---|---:|---:|---:|
| Template characters | 119,926 | 83,889 | 83,889 |
| Incidental umbrella fact captured exactly, retained next window | 2/2 | 2/2 | 2/2 |
| Unspoken connection across three choices stored meaningfully | 2/2 | 2/2 | 2/2 |
| Beginner class, Thursday schedule and first-lesson scope retained | 2/2 | 2/2 | 2/2 |
| Courier remains unconfirmed after its booked date passes | 2/2 | 2/2 | 2/2 |
| Poster receipt/approval revised, copies/paper retained, printing future | 2/2 | 2/2 | 2/2 |
| Poster's obsolete reasoning also repaired | 2/2 | 2/2 | 1/2 |
| PDF observation and explicitly untested explanation retained in node | 2/2 | 2/2 | 2/2 |

Integration counts specific, supported edge explanations and reasoning; it
does not require a separate pattern node. Frozen v2 and Shapes each made a
pattern node in one repetition and used concrete decisions with explanatory
edges in the other. Episode first used the latter form twice. Node count is
not a measure of understanding here.

The last table row is deliberately about the PDF node: Episode repeat 2 also
wrote an unsupported `resolved_alongside` edge. It therefore did **not** retain
uncertainty consistently across every stored surface.

## Hand ledger and dump paths

All paths below are relative to
`eval/results/s1e_guide_compression_2026-09-08/` in the review worktree.
For each sequence, `window1/nodes_after.json` contains initial writes;
`window2/nodes_before.json` shows what the next encode received;
`window2/nodes_after.json` contains the resulting memory. `calls.json` contains
actual tool arguments and dispatch results, including edge writes, and
`result.json` contains the lists and final journal text. `next_continuity.txt`
shows the actual harvested continuity. The fixture is the authority for source
wording and dates. IDs below identify test nodes in these dumps, not live facts.

### Frozen v2 / repeat1

Path: `v2/repeat1/`.

- `71b25c9c`: green umbrella, locker B17, North Quay, left September 7, only
  item; persists unchanged. `8f47af2a`: the shift conditions and three-instance
  connection, with explicit edges to both seeded prior choices. `51e0a311`:
  Thursday beginner class and September 10 incident captured.
- `b238405a`: proof/approval updated, including reasoning and event time;
  60 copies and cream paper retained, September 12 printing remains future.
  Courier `bffb69a8` remains unchanged and unconfirmed.
- `b076c415` title says **"closing conditions drafted"** although the source
  only says "I'll make the shift sheet" and that it has not been sent. The
  drafted claim also enters journal text. No send is invented.
- `960174bf` correctly keeps PDF object 18, enabled cover-sheet option and
  untested hypothesis, but its reasoning attributes the deferral to **Tom**
  instead of Oren.
- `30e828fc` calls the assistant's interpretation **"declined"** and says
  Oren **"explicitly declined to generalize."** The actual response is
  "Maybe. I haven't worked out a rule about that." The initial pattern's
  thought is narrowed, but the correction imposes a stronger verdict than
  the source supports.

### Frozen v2 / repeat2

Path: `v2/repeat2/`.

- `4515e1f1`: umbrella correct and retained. `b651de84`: all shift conditions
  and Kasia's two prior desk shifts, with specific integration edges.
  `7036308b`: committed and unsent, with no invented drafting progress.
  `29727a53`: PDF evidence and hypothesis correctly separated; Oren attribution
  correct. `597fb6f0`: class facts and date correct.
- Poster `b238405a` receives updated reasoning and September 11 event time;
  printing September 12 stays future. Its type remains `open` while
  `evolution_status` becomes `resolved`; current fields distinguish approval
  from future printing, so this is not counted as an unrepaired proof claim.
- `804cac6c` content preserves provisional status, but reasoning concludes
  **"the linking principle does not [stand]"**, and situation says keep the
  situations distinct **"unless he states otherwise."** This is a subtler
  unwarranted dismissal/ratification requirement than repeat1, not the same
  literal "explicit rejection" claim.
- Window1 Review says **"June's Wednesday deadline passed"** on Tuesday
  September 8. This is a new false temporal claim in continuity; the seeded
  node's September 9 date and pending state remain correct. The following
  window legitimately reports receipt on September 11. Do not hide this
  error behind the final successful revision.

### Shapes / repeat1

Path: `shapes/repeat1/`.

- `3a218593`: umbrella correct and retained. `80e220cc`: concrete shift decision,
  all conditions and Kasia's prior experience; still planned/unsent. The two
  `instantiates` edges explain the shared structure across prior choices and
  count as integration. `523b7464`: class facts correct.
- Poster `b238405a` updates all obsolete proof/approval fields, including
  reasoning and event time, while preserving 60 cream copies and future print.
- `f33f7690`: PDF observations and untested explanation retained, but reasoning
  substitutes **Tom** for Oren as the person deferring work.
- `338f6e5e` says the response **"rejected [the] interpretive link"** and that
  Oren **"explicitly said no such rule exists for him."** Neither follows from
  "Maybe. I haven't worked out a rule."
- Retiring duplicate textual journal subjects for courier/proof while keeping
  their pending node IDs is legitimate note maintenance, not evidence of
  invented real-world completion. Courier remains pending in window2.

### Shapes / repeat2

Path: `shapes/repeat2/`.

- `32383957`: umbrella correct and retained. `08e97d6a`: concrete shift decision,
  all conditions, Kasia's experience and unsent state. `f30775da`: supported
  three-instance pattern with concrete edges; content explicitly bounds scope.
  `6e6aedc2`: class facts and September 10 incident correct.
- Poster `b238405a`: proof/approval and all stale reasoning revised, copies and
  paper retained, September 12 print future. Courier stays unconfirmed.
- `6f116f98`: PDF hypothesis correctly untested, but again **Tom** in reasoning
  where the speaker was Oren. The arc's "cause identified" is stronger wording
  than the node's qualified hypothesis, another surface worth watching.
- `80a16b19` content says the explanation is provisional, but its persisted
  `bounds` edge says Oren **"explicitly declined to link the situations."** The
  pre-write list calls it **"rejected by Oren"**. Do not grade the content alone
  as a clean uncertainty preservation result.

### Episode first / repeat1

Path: `episode/repeat1/`.

- `1007d695`: umbrella correct and retained. `71f9b56f`: shift conditions and
  Kasia's prior experience, with two meaningful integration edges.
  `30bb9eef`: PDF observation, hypothesis and Oren attribution correct.
  `ecd5a473`: class/Thursday/first-lesson facts and September 10 incident kept.
- `71f9b56f` says the sheet **"is being prepared"** without evidence work has
  started. This is weaker than "drafted", but still adds progress beyond the
  source commitment. It remains unsent.
- Poster `b238405a` updates reasoning and approval date, preserves print details
  and future printing. Courier remains unconfirmed.
- `7661b33f` reasoning says Oren **"explicitly rejected [the] linking
  explanation."** Same unsupported move from uncertainty to rejection.
- The class node's event time is September 11 (disclosure); its content correctly
  dates the incident September 10. Because it holds both a newly disclosed
  recurring practice and a dated incident, this is not scored as an invented
  incident date.

### Episode first / repeat2

Path: `episode/repeat2/`.

- `490c1c9d`: umbrella correct and retained. `efe43c20`: shift decision and two
  meaningful cross-context edges. `019ecc85`: class facts correct. `f55931c2`:
  PDF node remains explicitly untested, attribution correct.
- `63fd990a` title says **"closing conditions written"**, despite only a promise
  to make the sheet. Its content correctly says "I said I would make" and unsent.
- `2a10a927` keeps the linking interpretation **"mine alone and remains
  unconfirmed"** without explicitly declaring it rejected. Its correction
  framing still emphasizes lack of ratification, but the saved account of
  what Oren said is materially more faithful than the other repeats.
- Poster `b238405a` updates title/content/situation/question/type, preserves
  print details and future printing, but leaves reasoning **"The production
  plan names a due date; it does not establish receipt or approval"** and the
  old September 9 event time. These are visible in the after dump. A successful
  revise call is not complete maintenance of the node.
- The same call adds `resolved_alongside` between poster and PDF, reasoning that
  approval means the PDF problem was **"cleared or set aside"**. No source
  confirms either. The tool stored it on the existing reverse-direction pair
  and returned a direction warning. The final journal notices that warning
  but does not repair the unsupported claim or the stale node reasoning.
  This is separate from the PDF node itself remaining explicitly untested.
- The courier remains unconfirmed in stored memory. The next review omits
  several previous subjects; omission of a journal subject is not itself
  evidence that the underlying work was completed.

## Execution checks

- All six sequence subprocesses exited 0; all 12 encodes used two rounds and
  one successful `brain_batch`, with zero reads. No dispatch or nested
  operation/edge failures. Episode repeat2/window2 has the direction warning
  described above. Zero reads means this cell says nothing about hidden-body
  fetching, the residue rendering gate, or multi-read continuation.
- `verification.json` records checks of all initial round records against
  their arm system/settings/tool names, saved full schemas, and frozen hashes.
  All six first-window message payloads are identical. Second-window input
  correctly differs with each arm's actual writes and harvested continuity.
- Actual composed systems are 126,694 characters for v2 and 90,657 for each
  compact arm: a shared 6,768-character suffix plus the respective template.
  The 4,372-character gist, historical background, initial seed IDs, model,
  effort and limits are fixed. Template reduction is not total input/token
  reduction; catalog, timeline, gist and schemas also consume input.
- No frozen candidate or fixture changed during evaluation. Outputs remain
  local and ignored by git. The run-created `:memory:.ses` was archived under
  these results after the cell finished.

## Proposed next revision — one principle, existing carriers

The v3 gist to discuss with Tom is: **retain the observation; let an
interpretation remain useful and unsettled; advance a claim only as far as
its evidence advances.** "Unconfirmed" is not "rejected", a promise is not
progress, and a related success is not proof of another task's completion.
The same distinctions must survive in title, reasoning and relationship text,
not only in content.

Apply this through revisions to existing Shapes material: the Reading
distinction, the later-window thought example, and the existing planned-work /
result examples. Keep the compact section order and current gist placement;
prefer replacements within the roughly 84K template budget. Do not add a fifth
list, a mandatory thought field, a source-ref quota or a case-specific umbrella /
screen-printing rule. No v3 prompt text has been written by this cell.

This moves the ledger's **strings versus claims** and **node versus field**
failures (the epistemic claim in a title/edge/reasoning must agree with the
evidence), **voice seesaw** (my provisional synthesis can survive without being
attributed to or ratified by the other person), and **residue as authority**
(an invented verdict/date can travel into later context). This is a proposed
mechanism, not a proven diagnosis of which prompt sentence caused each error.

The challenge review should test uncertainty without confirmation or rejection,
an independent nearby completion, and retained facts under a revised
interpretation. Re-run the observed cases for regression, then a different
domain with no teaching/classroom example to test transfer. Gold and longmem
remain necessary before any promotion claim; source refs are assessed for
useful visibility, not universal coverage. Tom retains all merge and shared
runtime gates.
