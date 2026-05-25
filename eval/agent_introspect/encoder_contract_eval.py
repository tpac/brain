"""Encoder Contract Eval — applies quality_contract.py to encoder output.

Two use modes:

1. **Example validation** (authoring loop): score §7.6 worked examples
   against the contract. Catches authoring blind spots — dims claimed
   satisfied that aren't, dims missing from coverage.

2. **Live encoder evaluation** (post-ship): score live-encoded nodes
   produced by the active s1e prompt. Measures whether teaching against
   the example library produces higher-scoring output than v17 baseline.

Both modes use the same contract (`servers/scales/s1/quality_contract.py`)
and the same evaluator prompt (below). The difference is the input:
authored example vs. live encoded node + its source conversation.

Joins the existing eval/agent_introspect/ family alongside quality_probe,
encoder_replay, coherence_probe, etc. Reuses agent-introspection patterns;
adds contract-grounded structured scoring.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from servers.scales.s1.quality_contract import (
    DIMENSIONS,
    CROSS_DIM_RULES,
    RECALL_GATING,
    STRUCTURAL_FOLLOWUPS,
)


EVALUATOR_SYSTEM_PROMPT = """You are evaluating an encoder example against the 36-dimension quality contract.

You receive:
- `conversation`: source turns the encoder saw (S0 trace events with speaker labels and trace ids)
- `encoder_output`: the remember_batch / brain_batch / revise_batch calls the example demonstrates
- `example_intent`: which axis/dim this example was authored to teach
- `contract`: the 36 DIMENSIONS dict + CROSS_DIM_RULES from quality_contract.py
- `self_claimed_eval`: the example's own contract_eval claim (DO NOT read until AFTER independent scoring)

# Your task

For each dimension D1..D36, classify the encoder output INDEPENDENTLY (without looking at self_claimed_eval):
- `satisfied`: positive signals present, no violations
- `degraded`: partial — some satisfies present but degrades-list features visible
- `violated`: clear violation of the dim's stated criteria
- `n/a`: dimension structurally doesn't apply to this example shape

D33-D35 (sentinel range, ref internal consistency, voice annotation coverage) are mechanically checked by validate_example_authoring() — your job for these is to confirm the mechanical check and report the violations if any.

D36 (turn↔node language divergence) is YOUR primary semantic call. Read the source_conversation turns AND the node content/situation/reasoning fields. Score:
- satisfied: turns and nodes use different language registers; node names a structural axis the turn implies; specificity preserved; verbatim phrases bridged via raw_quote fields only
- violated: node content paraphrases the turn; ranges flattened; numbers smoothed; exact phrases re-stated in content prose
Per CR12: phrases shared between a turn and user_raw_quote/anchor_raw_quote are the legitimate verbatim bridge — that's NOT a D36 violation.

For each cross-dim rule CR1..CR12, identify:
- Did the tension this rule names actually surface in this example?
- If yes, was the rule's resolution applied correctly?

THEN compare your scoring to self_claimed_eval and flag divergences.

# Output JSON

{
  "example_id": "<id>",
  "per_dim": [
    {
      "dim": "D1_title_as_handle",
      "your_status": "satisfied" | "degraded" | "violated" | "n/a",
      "evidence": "<specific field values that drive the status>",
      "degradation_note": "<if degraded, where the partial-fail lives>",
      "matches_self_claim": true | false,
      "divergence_reason": "<if mismatch, why>"
    },
    ...36 entries
  ],
  "cross_dim": [
    {
      "rule": "CR1_title_compress_vs_verbatim",
      "fired": true | false,
      "resolution_correct": true | false | "n/a",
      "note": "<observation about how the tension was navigated>"
    },
    ...12 entries
  ],
  "verdict": {
    "is_canonical": true | false,
    "missing_demonstrations": ["<dims this example could have shown but did not>"],
    "contradictions_found": [["dim_a", "dim_b", "<reason>"]],
    "divergences_from_self_claim": ["<dim_n>: claimed X, scored Y because Z"],
    "summary": "<one paragraph: does this example teach its intent without contradicting other dims? Where does the author's self-eval drift from independent reading?>"
  }
}

# Discipline

- Cite specific field values when scoring (e.g., "title='Single-writer invariant' — noun phrase, <=80c").
- Don't hedge — if a dim is satisfied, say so; if violated, name the violation.
- Score independently FIRST, then compare. Anchoring bias is the failure mode.
- Multi-aspect verbs (D24): score based on whether edge_description disambiguates.
- Empty fields are not automatically n/a — check if the dim REQUIRES the field.
- Recall-gate dimensions (D25-D27): score the encoder's WRITING discipline, not what recall would display.
- Per CR4: novel types are degrade-not-violate when content is coherent with an existing aspect's meaning.
- Output strict JSON. No prose outside the JSON envelope.
"""


def load_contract_summary() -> str:
    """Render the contract as text for evaluator agent input."""
    lines = [f"# {len(DIMENSIONS)}-DIMENSION QUALITY CONTRACT", ""]
    for dim_name, dim in DIMENSIONS.items():
        lines.append(f"## {dim_name}")
        lines.append(f"**Group**: {dim['group']}")
        lines.append(f"**Intent**: {dim['intent']}")
        lines.append("**Satisfies** (positive signals):")
        for s in dim.get('satisfies', []):
            lines.append(f"  - {s}")
        lines.append("**Violates** (clear failures):")
        for v in dim.get('violates', []):
            lines.append(f"  - {v}")
        if dim.get('degrades'):
            lines.append("**Degrades** (partial fails):")
            for d in dim['degrades']:
                lines.append(f"  - {d}")
        if dim.get('interacts_with'):
            lines.append(f"**Interacts with**: {dim['interacts_with']}")
        lines.append("")

    lines.append("# CROSS-DIMENSION RULES")
    for cr in CROSS_DIM_RULES:
        lines.append(f"## {cr['name']}")
        lines.append(f"**Rule**: {cr['rule']}")
        if cr.get('applies'):
            lines.append(f"**Applies to**: {cr['applies']}")
        lines.append("")

    return "\n".join(lines)


def load_example_for_eval(example_module) -> dict:
    """Strip self_claimed_eval into a separate field so evaluator agent
    can score blind before comparing."""
    ex = dict(example_module.EXAMPLE)
    self_claim = ex.pop('contract_eval', None)
    return {
        'example_to_score': ex,
        'self_claimed_eval': self_claim,
    }


if __name__ == '__main__':
    # Smoke test — confirm contract loads and renders.
    summary = load_contract_summary()
    print(f"Contract summary: {len(summary)} chars")
    print(f"Dimensions: {len(DIMENSIONS)}")
    print(f"Cross-dim rules: {len(CROSS_DIM_RULES)}")
    print(f"Structural followups: {len(STRUCTURAL_FOLLOWUPS)}")
