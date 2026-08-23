# Probe scenarios for s1e candidate prompts

Production-faithful encoder inputs for stateless-Sonnet probes of a candidate
prompt. Each file is what the encoder actually receives — `<continuity>`,
`<node_catalog>`, `<scout_legend>`, `<timeline>` — not a prose description of a
situation. A probe validates only what its input exercises, so a scenario that
isn't production-shaped tests mechanics rather than quality.

## The pair

| file | what it exercises |
|---|---|
| `closure_settled.md` | three open catalog nodes the window **settles**: one fully (retry policy, 0.3% vs 11%), one **partially** (pool fix kills the 30s stalls, the 90s shape survives), one that settles *and spawns* a new question (incremental dedupe needs a watermark nobody designed) |
| `closure_unsettled.md` | the negative control — the same catalog, a window that **advances and settles nothing**: a distrusted partial test, an uninstrumented hunch, a stated lean explicitly withheld from being a decision |

**They only mean something as a pair.** The positive arm alone cannot separate a
working teaching from one that fires on everything; the control is what proves
the bar discriminates. Neither scenario names the move it is testing — if the
input pre-names the behaviour, the probe measures reading comprehension.

Pass criteria for the open-node closure teaching (corrections flavor 3):
`closure_settled` → one retype off `open` per fully-answered node, the partial
stays `open` and narrows, the spawned question becomes its own node.
`closure_unsettled` → **zero** type changes and zero `resolves` /
`partially_resolves`; content revisions are expected and correct.

## Building the arm

The candidate file is not the prompt the encoder sees. Production S1E appends
the contract field summary and binds `arc=True`, so an arm missing the `## Arc`
block is not production-faithful — probes A–O in the walk were built without it.
Order matters: field summary → arc → review → closure.

```bash
BRAIN_ALLOW_ANY_PYTHON=1 ./dev python3 -c "
import sys; sys.path.insert(0,'.')
from servers.contract import generate_field_summary
from servers.trace_contract import (render_journal_arc_block,
    render_journal_review_block, render_prompt_closure)
wip = open('eval/candidate_prompts/s1e_vnext5_wip.md').read()
arm = wip + '\n\n## Available Fields (from contract)\n\n' + generate_field_summary()
for block in (render_journal_arc_block(), render_journal_review_block(),
              render_prompt_closure()):
    arm = arm.rstrip() + '\n\n' + block
open('<out>.md','w').write(arm); print(len(arm))"
```

Then run a stateless Sonnet with the arm as its system prompt and one scenario
as input. The tool-ban clause is load-bearing — without it a probe will make a
real call:

> Adopt the system prompt as YOUR instructions. Do not call any tool other than
> reading these two files. ROUND 1 as verbatim JSON in fenced blocks, assume
> success, then ROUND 2 + the final reply. Your final message must be ONLY the
> run output.

## Reading a result

Single runs are evidence, not proof — run-to-run node-quality variance exceeds
most render deltas (brain id:079d9736), and green author-written probes do not
clear production (id:607930d0). Audit every emission against ALL standing
invariants, not just the behaviour under test (E9): temporal self-containment,
voice derivation, placeholder discipline, id copying.
