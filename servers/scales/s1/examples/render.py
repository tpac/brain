"""Render structured EXAMPLE dicts as s1e prompt prose.

The structured form in examples/*.py is the authored source-of-truth —
rich with encoder_cognition, counterfactual_bad, voice_annotations, and
contract_eval for evaluator + future S3 consumption.

The rendered form is what Sonnet sees at encoding time: terse, formatted
to match the existing canonical training pattern in s1e v19. The render
extracts encoder_output node fields and formats them like the v19
canonical example block (see s1e v19 line ~883-908 for reference shape).

Meta-structure (encoder_cognition, counterfactual_bad, voice_annotations,
contract_eval) is INTENTIONALLY dropped at render time — it's for the
authoring/eval loop, not the encoder's pattern-match.

What gets included in rendered prose:
- Header: example id + intent (one line)
- The remember_batch / brain_batch call with all fields populated
- One-line "Teaches" note pulling the primary teaching

What stays in the structured file:
- Source conversation with register tags
- Encoder cognition (temptations, choice points)
- Counterfactual bad encoding
- Voice annotations
- Contract eval
- What this teaches (full)
"""

from typing import Any, Dict


def _format_value(v: Any, indent: int = 5) -> str:
    """Render a field value in canonical-pattern style. Multi-line strings
    get wrapped continuation; lists get inlined for short ones, expanded
    for long ones."""
    pad = " " * indent
    if isinstance(v, str):
        return f'"{v}"'
    elif isinstance(v, bool):
        return "true" if v else "false"
    elif isinstance(v, (int, float)):
        return str(v)
    elif isinstance(v, list):
        if all(isinstance(x, (int, str, bool, float)) for x in v):
            return repr(v)
        # List of dicts (e.g. connect_to edges) — expand with comma separators
        lines = ["["]
        for i, item in enumerate(v):
            sep = "," if i < len(v) - 1 else ""
            if isinstance(item, dict):
                # Rename edge_description -> why (canonical dispatcher field
                # per brain_remember._resolve_connect_to_entry — reads
                # entry.get('why', entry.get('description', '')))
                item_renamed = {('why' if k == 'edge_description' else k): val for k, val in item.items()}
                parts = ", ".join(f'{k}: {_format_value(val, indent)}' for k, val in item_renamed.items())
                lines.append(f"{pad}  {{{parts}}}{sep}")
            else:
                lines.append(f"{pad}  {_format_value(item, indent)}{sep}")
        lines.append(f"{pad}]")
        return "\n".join(lines)
    return repr(v)


def render_node(node: Dict[str, Any], indent: int = 5) -> str:
    """Render a single encoder_output node in canonical pattern style."""
    pad = " " * indent
    fields = []
    # Canonical field order (matches the v19 prompt's training-pattern style)
    canonical_order = [
        'type', 'title', 'content', 'situation', 'reasoning',
        'user_raw_quote', 'anchor_raw_quote',
        'event_time', 'emotional_context', 'correction_pattern',
        'trigger', 'locked', 'source_refs', 'connect_to',
    ]
    seen = set()
    for k in canonical_order:
        if k in node:
            fields.append(f'{pad}{k}: {_format_value(node[k], indent)}')
            seen.add(k)
    # Catch any non-canonical fields
    for k, v in node.items():
        if k not in seen:
            fields.append(f'{pad}{k}: {_format_value(v, indent)}')
    return "{\n" + ",\n".join(fields) + "}"


def render_for_encoder_prompt(example: Dict[str, Any]) -> str:
    """Render an EXAMPLE dict as canonical-training-pattern prose.

    Output is intended for embedding in the s1e prompt's §7.6 block.
    Drops meta-structure (encoder_cognition, counterfactual_bad,
    voice_annotations, contract_eval) — those are for evaluator + S3,
    not Sonnet's pattern-match input.

    Returns: ~600-1500 chars per example depending on node complexity.
    """
    out = []

    # Header
    out.append(f"### {example['id']} — {example['intent']}")
    out.append("")

    # The encoder_output formatted in canonical pattern style
    op = example['encoder_output']
    op_name = op['operation']
    nodes = op.get('nodes', [])

    out.append("```")
    out.append(f"{op_name}(")
    out.append("  nodes: [")
    for i, node in enumerate(nodes):
        rendered = render_node(node, indent=5)
        # First line gets "    " prefix; subsequent lines already indented
        rendered_lines = rendered.split("\n")
        rendered_lines[0] = "    " + rendered_lines[0]
        sep = "," if i < len(nodes) - 1 else ""
        out.append("\n".join(rendered_lines) + sep)
    out.append("  ]")
    out.append(")")
    out.append("```")
    out.append("")

    # One-line teaching note
    teaches = example.get('what_this_teaches', {})
    primary = teaches.get('primary', '')
    if primary:
        # Compress to one line if multi-sentence
        first_sentence = primary.split('.')[0].strip() + '.'
        out.append(f"**Teaches**: {first_sentence}")
        out.append("")

    return "\n".join(out)


def render_wave(examples: list) -> str:
    """Render all examples in a wave as a single §7.6 prose block."""
    return "\n".join(render_for_encoder_prompt(ex) for ex in examples)


if __name__ == '__main__':
    from . import WAVE_1
    rendered = render_wave(WAVE_1)
    print(rendered)
    print(f"\n---\nTotal chars: {len(rendered)}")
    print(f"Examples: {len(WAVE_1)}")
    print(f"Avg per example: {len(rendered) // len(WAVE_1)}")
