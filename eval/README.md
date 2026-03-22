# Brain Evaluation Framework

Automated A/B testing for brain-to-Claude communication. Measures whether formatting changes improve Claude's encoding quality and recall attention.

## Evals

### `skill_eval.py` — Encoding Quality Baseline
Core infrastructure. Tests how different SKILL.md prompt variants affect encoding richness.

- **Shared exports**: `FAKE_BRAIN_TOOLS`, `SCENARIOS`, `score_run()`, `run_single()`
- **Scoring**: Composite 0-100 "richness" score (quantity, connections, depth, variety, keywords)
- **Scenarios**: bug_fix, architecture_discovery, operator_correction, decision_with_tradeoffs, multi_file_change

```bash
source .env && python3 eval/skill_eval.py --runs 5
```

### `adaptive_eval.py` — Checkpoint Strategy Testing
Tests whether encoding checkpoints (injected prompts at turn boundaries) improve encoding consistency across multi-turn sessions.

- **Strategies**: NoCheckpoint, StaticRotation, AdaptiveRotation, HookSpecific
- **Sessions**: 3 multi-turn scenarios (refactor, debugging, architecture)
- **Metrics**: organic vs checkpoint-triggered encoding

```bash
source .env && python3 eval/adaptive_eval.py --runs 3
```

### `brain_identity_eval.py` — SKILL.md Wrapper Testing
Tests whether wrapping SKILL.md content in `[BRAIN]...[/BRAIN]` or XML improves encoding. **Result: no significant difference** — SKILL is passive context, not active communication.

```bash
source .env && python3 eval/brain_identity_eval.py --runs 3
```

### `brain_recall_identity_eval.py` — Recall Output Formatting (key result)
Tests whether `[BRAIN]...[/BRAIN]` wrapping on **hook output** (recall injections) changes Claude's behavior. This tests the actual brain-to-Claude communication channel.

**3 Modes:**
1. Plain text — `BRAIN RECALL:` header, indented content
2. `[BRAIN]...[/BRAIN]` — same content with open/close tags
3. XML inside `[BRAIN]` — structured `<recall><node>` elements

**5 Scenarios** with varying recall relevance (1 node, 2 nodes, irrelevant noise).

**Key Finding (2026-03-21, 75 runs on Sonnet):**

| Mode | Richness | Recall Use% | Noise Encoding |
|---|---|---|---|
| Plain text | 39.8% | 51.4% | 10% (encoded noise) |
| [BRAIN]...[/BRAIN] | 36.0% | 51.4% | 2% |
| XML inside [BRAIN] | 38.8% | 49.8% | 0% |

- Encoding quality and recall usage nearly identical across modes
- **Noise rejection is the differentiator**: plain text still encoded irrelevant recall; [BRAIN] tags reduced this to near-zero
- XML adds marginal noise rejection but no other benefit
- **Decision**: Ship `[BRAIN]...[/BRAIN]` without XML internals

```bash
source .env && python3 eval/brain_recall_identity_eval.py --runs 5 --workers 8
```

## Shared Infrastructure

All evals import from `skill_eval.py`:
- `FAKE_BRAIN_TOOLS` — 9 brain tool schemas for API simulation
- `score_run()` — composite scoring (richness, connections, uncertainties, etc.)
- `run_single()` — single API call with tool-use loop handling
- ThreadPoolExecutor for parallel runs

## Running

```bash
# Set API key
source ../.env  # or export ANTHROPIC_API_KEY=...

# Quick smoke test (1 run)
python3 eval/brain_recall_identity_eval.py --runs 1 --scenarios add_column

# Full eval
python3 eval/brain_recall_identity_eval.py --runs 5 --workers 8
```

Results are saved to `eval/results/` as timestamped JSON files.
