# V3.2 diagnostic: does the worklist influence the closing review?

The six-call diagnostic is prepared and locally checked. **No calls ran:**
automatic approval review rejected the launch as outside the earlier frozen
nine-encode authorization. The previous nine-encode comparison is complete;
this is a separate proposed diagnostic. Nothing is merged or deployed.

## What the saved evidence establishes

The source does carry explicit speaker boundaries. In repeat 2, window 3,
turn 15 renders Tom's beauty/first-impressions request in `<other speaker="Tom">`
and the assistant's aesthetic-product formulation in `<me>`. The full source
is present before the first reply. This rules out missing speaker labels in
this saved request; it does not show how well the model attends to the labels.

| Stage | Aesthetic ownership | D3 agreement status |
|---|---|---|
| Source | Tom requests beauty; the assistant says “The aesthetic IS the product…” | Tom selects graph/temperature as the first prototype scope; the assistant says “I'd start with D3.js…” |
| Initial `changes` | “Tom's explicit stake: the aesthetic IS the product” | “settled starting point: D3.js…” |
| Persisted fields | `ea20fc3c.content`: “Tom stated explicitly…” | `da056c35.content`: “My proposed implementation”; situation: “the agreed prototype stack” |
| Closing review | “Tom's exact quote preserved… Clean.” | Calls it a prototype decision and closes without repair |

There is a further useful contrast inside the aesthetic node: `my_raw_quote`
correctly contains the assistant's aesthetic-product sentence, while content
assigns that sentence to Tom. The correct evidence survived in another field.
Capturing a quote is therefore insufficient to keep the interpreted claim
consistent with it. This is about using evidence, not requiring more refs.

The D3 reasoning also explicitly notices that Tom did not affirm the library
by name, then infers settlement from his move to aesthetic priorities. The
encoder sometimes has the distinction available and overrides it; this is
more specific than simply never noticing uncertainty.

These observations support a hypothesis: the working summary becomes an
accepted premise for later writing and checking. They do not prove that
worklist removal will help. Incorrect writes, prior catalog claims, task load
and context length may all influence the later check.

## Frozen comparison

Both conditions start from the exact saved request immediately after repeat
2/window 3's successful batch and before its original closing reply. The
system, user input, tools, tool-use arguments and successful tool results are
preserved. Both append the same focused review question and disable tool
calls. Responses can propose repairs but cannot perform writes.

- **With worklist:** retains the saved 5,373-character assistant text.
- **Without worklist:** removes only that text block, retaining the original
  tool-use block in the same assistant message.

Three independent samples per condition; six calls maximum; Sonnet 4.6,
medium effort, 2,400 output tokens per call, no retries. Repetitions can run
in parallel; the two conditions within each repetition run sequentially,
with condition order reversed for repetition 2. No random seeds are matched.
The full original system is 99,477 characters, so these are short **output**
reviews of the full context, not tiny source-only prompts. The combined
output ceiling is 14,400 tokens; this is not a dollar-cost estimate.

The question requests a source-grounded verdict for five topics, including
supported claims. It gives no answer key or expected error count. It asks for
source wording, speaker and turn, relevant node fields, and minimal changes
where needed. It does not ask the model why it previously made a mistake:
prior interview experience shows that such explanations can rationalize the
decision rather than establish its cause.

## Review criteria recorded before calls

Judge each topic separately; do not collapse the exercise to one score.

| Topic | What must survive or change |
|---|---|
| Mirror/camera affirmation | Retain Tom's explicit turn-11 affirmation. Do not use it to establish a new blanket dashboard privacy rule. Earlier implementation details are not fully re-established by this five-pair source slice. |
| Aesthetic formulation | Attribute the aesthetic-product sentence and operational priorities to the assistant. Retain Tom's firm beauty/first-impressions requirement. Inspect the content/quote contradiction and scope of linked claims. |
| Ambient presence | Preserve Tom's stated vision. Keep the menubar and notification ideas attributed to the assistant and unconfirmed by Tom. Do not weaken the vision merely because its proposed implementations are unsettled. |
| First prototype | Retain graph visualization plus temperature colors as Tom's explicit first-build direction. Do not turn it into a completed implementation. |
| D3 and mappings | Preserve the concrete proposed implementation. Correct wording that claims Tom settled the library/mappings, including the situation and the reasoning that infers agreement from his next topic. |

A useful response must locate evidence and identify the actual conflicting
field. Merely saying “be careful about attribution” does not count as catching
the defect. Conversely, blanket uncertainty that weakens supported claims is
a regression. Additional justified findings are allowed; this is not an
oracle asserting that only two defects exist.

## How the result would change the next step

- If both conditions catch the errors, targeted source review is possible in
  this context. That does not demonstrate spontaneous repair during encoding
  or show that removing the worklist is necessary.
- If removal consistently helps, the worklist/context difference contributes
  to this review failure. Removal also shortens context; a later length-matched
  control would be needed to isolate anchoring from context length.
- If both retain the errors, removal alone is insufficient. The incorrect
  write arguments and catalog remain in both conditions; source comprehension
  versus workload would still need a separate test.
- If either condition erases valid commitments or useful details, record that
  cost beside error detection. Hedging is not automatically fidelity.

No outcome here warrants a benchmark or transfer claim. This deliberately
uses known development failures to diagnose a mechanism. A later prompt
revision would need unrelated teaching examples and untouched evaluation
material. The V3.2 template, gist, tools and examples remain frozen.

## Artifacts and launch

- `eval/results/s1e_v32_review_probe_2026-09-11/manifest.json`: input hashes,
  preflight checks, bounds and confounds.
- `evidence_packet.json`: verbatim role-tagged source, first worklist, five
  stored nodes with metadata/outgoing relations, and original closing verdict.
- `with_worklist.request.json`, `without_worklist.request.json`: exact proposed
  API requests. `question.txt` and `removed_worklist.txt` make the intervention
  inspectable. `blocked.json` records that the launch did not execute.
- `eval/fixtures/s1e_guide_v3_2_2026-09-10/review_probe.py`: prepare/check/run;
  no brain creation, tool dispatch, activation or database access.

After explicit approval, the already-prepared run command is:

```sh
./dev python3 eval/fixtures/s1e_guide_v3_2_2026-09-10/review_probe.py run
```

Do not rerun `prepare` or bypass the rejected launch through another path.
