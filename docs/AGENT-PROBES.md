# Agent Probes — Prompt Quality Diagnostic System

**Renamed from "Agent Introspection" (2026-05-15) — "probes" is the name that fits what these things actually do.** Family of stateless-Sonnet probes that audit prompt + behavior pairs. Lets prompt iterations cycle in **minutes instead of hours** — predict whether a proposed change lands BEFORE paying eval costs.

Lives under [eval/agent_introspect/](../eval/agent_introspect/) (directory name preserved to avoid breaking imports; doc name updated to match the actual concept).

## Why it exists

The encoder prompt evolution arc kept producing aggregate scores inside the noise band. We couldn't distinguish "the prompt change had no effect" from "Sonnet randomly didn't comply this run." Re-running 50-item evals at ~50 min wall to get a single noisy signal made iteration painful.

The probe family solves this by replacing "ship and re-run the eval" with "ask Sonnet directly":
- *"Read this prompt — what does it actually say to you?"* (aspect)
- *"Here's the prompt + your actions — did you comply with these rules? Why or why not?"* (compliance)
- *"Where does this prompt contradict itself?"* (coherence)
- *"If we changed this part of the prompt, would your output have been different?"* (counterfactual)

Direct evidence from Sonnet's own reasoning, in 20-60 seconds per probe.

## The six modes

Two axes: **timing** (pre-run / post-run) × **target** (prompt / behavior).

| Mode | Built? | Question it answers | Timing | Target |
|---|:-:|---|---|---|
| **Aspect** | ✓ | "How does fresh agent READ this prompt across N lenses?" | pre-run | prompt |
| **Compliance** | ✓ | "Why did the agent comply or skip on each named rule?" | post-run | actions vs rules |
| **Coherence** | ✓ | "Where does the prompt contradict itself? Which rule wins when they overlap?" | pre-run | prompt internal |
| **Counterfactual** | ✓ | "You skipped Y. If the prompt said Z, would your output have been different?" | post-run | revision path |
| **Coverage / Replay** | ✓ | "With THIS prompt + the actual call inputs, what does the agent emit?" | pre-or-post-run | live agent call |
| **Edge-case** | – | "How would you handle scenario X (a corner case)?" | pre-run | extrapolation |

## File layout

```
eval/agent_introspect/
├── __init__.py            — package docstring + family overview
├── _common.py             — shared helpers (Sonnet call, artifact loading,
│                            action formatting, report rendering)
├── compliance_probe.py    — mode 2
├── coherence_probe.py     — mode 3
├── counterfactual_probe.py — mode 4
├── rules/                 — JSON rule lists for compliance audits
│   └── temporal_v15_7.json
└── changes/               — JSON change specs for counterfactual probes
    └── v15_8_canonical_example.json

eval/encoder_prompt_probe.py  — mode 1 (older, pre-naming)
```

## Usage — each probe

### Compliance probe

```bash
./dev python3 -m eval.agent_introspect.compliance_probe \
  --run-dir eval/longmem/reports/<run_name> \
  --qids gpt4_b0863698,gpt4_85da3956,e831120c \
  --rules-file eval/agent_introspect/rules/temporal_v15_7.json \
  --out eval/longmem/reports/compliance_v15_7.md
```

Inputs: eval run with `--keep_dbs` (provides traces, action_details, interactions), plus a rules JSON listing each rule's `id` + `text`.

Output per (item × rule): `status` (comply/partial/skip/not_applicable), `evidence` (verbatim from actions), `reasoning`, `prompt_contradiction` (if Sonnet sees an internal conflict driving non-compliance).

**Use when:** an eval surfaces failures and you want concrete per-rule reasons for skip/partial behavior — instead of inferring from aggregate metrics.

### Coherence probe

```bash
./dev python3 -m eval.agent_introspect.coherence_probe \
  --prompt eval/prompts/s1e_v15_7.txt \
  --out eval/longmem/reports/coherence_v15_7.md
```

Inputs: just a prompt file.

Output: list of findings, each with `severity` (high/medium/low), `kind` (contradiction/stale_example/priority_gap/ambiguity), `rule_quote` + `conflicting_quote` (both verbatim), `location_hint`, `explanation`.

**Use when:** writing or reviewing a prompt. Surfaces stale examples that violate stated rules (the single highest-leverage failure mode — Sonnet imitates examples more than rules). Surfaces priority gaps where two rules apply with no statement of which wins.

### Counterfactual probe

```bash
./dev python3 -m eval.agent_introspect.counterfactual_probe \
  --run-dir eval/longmem/reports/<run_name> \
  --qids gpt4_b0863698,e831120c \
  --changes-file eval/agent_introspect/changes/v15_8_canonical_example.json \
  --out eval/longmem/reports/counterfactual_v15_8.md
```

Inputs: eval run with `--keep_dbs` + a JSON list of proposed changes (each: `name`, `target_behavior`, `location_hint`, `before_text`, `after_text`).

Output per (item × change): `prediction` (yes/partial/no), `confidence`, `specific_action_change` (which node/field/edge would change), `reasoning`, `risk` (any unintended over-emission).

**Use when:** you've drafted a prompt revision and want to predict its effect on existing failure cases without paying for a full eval re-run. Lets you iterate the wording until probe predicts the right shift.

### Aspect probe

```bash
./dev python3 eval/encoder_prompt_probe.py eval/prompts/s1e_v15_8.txt
```

Output: 5-aspect interview report (goal, edge cases, emphasis, voice, bias) — how fresh Sonnet READS the prompt from each lens.

**Use when:** mature prompt — checking whether the message lands as intended across multiple reading angles.

## Findings the probe family has produced (2026-05-11)

### Compliance probe — encoder asymmetry
On 5 items × 5 temporal rules (v15.7 prompt):
- **Restraint rules** ("don't create X unless..."): 100% compliance (10/10)
- **Generative rules** ("MUST write event_time"): 20% compliance (1/5)

Sonnet complies with NEGATIVE rules and skips POSITIVE structural-write rules. This pattern argues for code-level enforcement (dispatcher) for required metadata, not more prompt language. Precedent: brain memory `c39b8cc8` — the `related/related_to` ban is enforced at dispatch level, 100% effective vs ~75% in prompt-only.

### Coherence probe — examples are the actual training pattern
v15.7 prompt audit returned 19 findings. 3 of 5 high-severity were **stale examples** in the canonical "five nodes" demonstration that didn't match current rules (no user_raw_quote on fact-derived node despite "ANY operator-derived node" rule; undefined `question` field shown; batch-level `connect_to` despite "per-node" rule wording). **Sonnet imitates examples more than abstract rules** — example staleness silently overrides newer rules. Lesson encoded as principle `928f5694`.

### Counterfactual probe — directional predictor
For v15.8's canonical-example rebuild change, probe predicted **5/5 high-confidence yes** that the change would shift Sonnet's behavior toward event_time emission on dated nodes. Eval confirmed: event_time kv compliance went from 0% (v15.7) to ~5-8% (v15.8). Probe direction was right; absolute level was over-predicted. **Probe is useful as a directional filter, not as a guarantor of magnitude.**

## When NOT to use probes

- For final shipping decisions (sample-size-of-1 Sonnet call has its own variance — confirm with eval)
- When the conversation context is empty/trivial (probe needs real material)
- As a substitute for noise-floor-aware multi-seed evals when the prompt is mature
- For ranker/recall layer issues — probes are about prompt-driven agent behavior, not retrieval mechanics

## Future modes (not yet built)

- **Coverage / Replay probe** — **BUILT 2026-05-15** as two domain-specific tools:
  - `eval/agent_introspect/encoder_replay.py` — replays the encoder agent against a candidate s1e prompt, using captured conversation + scout output. Reports tool_use rounds + emitted actions per item. ~$0.001–0.003 per item, ~30s wall.
  - `eval/agent_introspect/surface_replay.py` — replays the surface agent against a candidate surface prompt, using a saved per-item brain.db (from `--keep_dbs` evals). Swaps only the active 'surface' interaction; reconstructs candidates_data from `recall.json` + `nodes.jsonl`; calls `_call_surface` (handles v4 vs v5_agentic variant). Reports selection + tool_trace + timing. ~$0.001 per item, ~2–13s wall.

  Both tools support iteration loops where the dynamic content (operator turns, Frame, candidate pool) stays fixed and only the static instruction (system prompt) changes. The full eval pipeline pays seconds per iteration round instead of minutes.

- **Edge-case probe** — give Sonnet a corner-case scenario and ask how it would handle it. Useful for stress-testing rules. Not built.
- **Priority probe** — when two rules apply, which wins? Sonnet's stated hierarchy. Not built.

Build any of these when a specific iteration arc needs them. Three probes is enough for most current workflow.

## Naming history

Tom: *"Agent introspection - seems like you mapped 2 aspects but try and think of 1-N more than are distinct that will be interesting to add."*

Renamed from the assistant's working name "encoder compliance probe" to **agent introspection** because the technique is general — applies to any agent + prompt + action-trace triple, not just the encoder.
