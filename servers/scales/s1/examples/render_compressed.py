"""Compressed render — smart compression hypothesis test.

Hypothesis (Tom's gut): compressing example prose smartly may teach
Sonnet better than verbose examples — less prompt budget consumed, pattern-
match signal less diluted, Sonnet reads short examples more attentively.

Compression rules:
- Keep verbatim contract fields INTACT — user_raw_quote, anchor_raw_quote
  cannot be compressed. The verbatim contract is the load-bearing teaching.
- Compress content to first 2 sentences (or ~200c) — keep the claim and
  the test, drop the elaboration.
- Compress situation to one sentence (~120c) — keep the re-fire condition,
  drop the explanation.
- Compress reasoning to one sentence (~100c) — keep the grounding citation,
  drop the elaboration.
- Compress edge_description to ~80c — keep the bridge naming, drop the
  argument.
- Drop the prose header and the "Teaches:" tail — those are for human
  readers of the file, not Sonnet pattern-match input.
- Keep structural fields intact: type, title, locked, source_refs,
  event_time, emotional_context, correction_pattern, trigger.

Target: 800-1200 chars per example (vs 2500-3800 for verbose). ~3x
reduction. Total wave-1 prose drops from ~19K to ~6-7K — a 12K savings
on prompt budget.

Compression is mechanical here; in a v20+ pipeline an LLM could compress
smarter (preserving load-bearing phrases over filler), but rule-based
truncation tests the hypothesis cleanly first.
"""

from typing import Any, Dict


# Verbatim-only fields — must NOT be compressed (the contract is sacred)
VERBATIM_FIELDS = {'user_raw_quote', 'anchor_raw_quote'}

# Structural fields — pass through unchanged
STRUCTURAL_FIELDS = {
    'type', 'title', 'locked', 'source_refs', 'event_time',
    'emotional_context', 'correction_pattern', 'trigger',
}

# Compressible prose fields with target lengths
COMPRESSIBLE = {
    'content': 280,
    'situation': 140,
    'reasoning': 120,
}


def _first_sentences(text: str, max_chars: int) -> str:
    """Keep the first N sentences that fit within max_chars."""
    if not text or len(text) <= max_chars:
        return text
    sentences = []
    used = 0
    for sentence in text.replace('\n', ' ').split('. '):
        s = sentence.strip()
        if not s:
            continue
        addition = len(s) + 2  # ". "
        if used + addition > max_chars and sentences:
            break
        sentences.append(s)
        used += addition
    return '. '.join(sentences).rstrip('.') + '.'


def _compress_edge(edge: Dict[str, Any]) -> Dict[str, Any]:
    """Compress edge_description but keep it substantive — preserve the
    'edges carry signal' teaching. Test surfaced that ~100c stripping made
    Sonnet write fewer edges; ~180c keeps the bridge-naming signal density.
    """
    out = dict(edge)
    if 'edge_description' in out:
        out['edge_description'] = _first_sentences(out['edge_description'], 180)
    return out


def _format_value(v: Any, indent: int = 5) -> str:
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
        lines = ["["]
        for i, item in enumerate(v):
            sep = "," if i < len(v) - 1 else ""
            if isinstance(item, dict):
                # Rename edge_description -> why (canonical dispatcher field
                # per brain_remember._resolve_connect_to_entry which reads
                # entry.get('why', entry.get('description', '')))
                item_renamed = {('why' if k == 'edge_description' else k): val for k, val in item.items()}
                parts = ", ".join(f'{k}: {_format_value(val, indent)}' for k, val in item_renamed.items())
                lines.append(f"{pad}  {{{parts}}}{sep}")
            else:
                lines.append(f"{pad}  {_format_value(item, indent)}{sep}")
        lines.append(f"{pad}]")
        return "\n".join(lines)
    return repr(v)


def compress_node(node: Dict[str, Any]) -> Dict[str, Any]:
    """Return a compressed copy of an encoder_output node."""
    compressed = {}
    for k, v in node.items():
        if k in VERBATIM_FIELDS:
            compressed[k] = v  # NEVER compress these
        elif k in COMPRESSIBLE and isinstance(v, str):
            compressed[k] = _first_sentences(v, COMPRESSIBLE[k])
        elif k == 'connect_to' and isinstance(v, list):
            compressed[k] = [_compress_edge(e) for e in v]
        else:
            compressed[k] = v  # structural / pass-through
    return compressed


def render_compressed_node(node: Dict[str, Any], indent: int = 5) -> str:
    """Render a compressed node in canonical pattern style."""
    compressed = compress_node(node)
    pad = " " * indent
    canonical_order = [
        'type', 'title', 'content', 'situation', 'reasoning',
        'user_raw_quote', 'anchor_raw_quote',
        'event_time', 'emotional_context', 'correction_pattern',
        'trigger', 'locked', 'source_refs', 'connect_to',
    ]
    fields = []
    seen = set()
    for k in canonical_order:
        if k in compressed:
            fields.append(f'{pad}{k}: {_format_value(compressed[k], indent)}')
            seen.add(k)
    for k, v in compressed.items():
        if k not in seen:
            fields.append(f'{pad}{k}: {_format_value(v, indent)}')
    return "{\n" + ",\n".join(fields) + "}"


def render_compressed(example: Dict[str, Any]) -> str:
    """Render an EXAMPLE dict in compressed form for s1e prompt embedding."""
    out = []
    out.append(f"### {example['id']}")  # bare id, no intent prose

    op = example['encoder_output']
    out.append("```")
    out.append(f"{op['operation']}(")
    out.append("  nodes: [")
    for i, node in enumerate(op.get('nodes', [])):
        rendered = render_compressed_node(node, indent=5)
        rendered_lines = rendered.split("\n")
        rendered_lines[0] = "    " + rendered_lines[0]
        sep = "," if i < len(op.get('nodes', [])) - 1 else ""
        out.append("\n".join(rendered_lines) + sep)
    out.append("  ]")
    out.append(")")
    out.append("```")

    return "\n".join(out)


def render_compressed_wave(examples: list) -> str:
    return "\n\n".join(render_compressed(ex) for ex in examples)


if __name__ == '__main__':
    from . import WAVE_1
    from .render import render_wave

    verbose = render_wave(WAVE_1)
    compressed = render_compressed_wave(WAVE_1)

    print("=== COMPRESSED WAVE 1 ===")
    print(compressed)
    print()
    print(f"--- SIZE COMPARISON ---")
    print(f"Verbose:    {len(verbose):>6} chars")
    print(f"Compressed: {len(compressed):>6} chars")
    print(f"Reduction:  {100 * (1 - len(compressed) / len(verbose)):.0f}%")
    print(f"Per example (compressed avg): {len(compressed) // len(WAVE_1)}")
