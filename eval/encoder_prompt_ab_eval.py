#!/usr/bin/env python3
"""Encoder Prompt A/B Eval — does the cognitive operations vocabulary change what gets encoded?

Runs the same conversation transcripts through two encoder versions:
  A: Current v1 prompt (generic edge types, "related" allowed)
  B: v2 prompt (cognitive operations vocabulary, "related" banned)

Compares:
  - Node count: does B encode more/fewer/different nodes?
  - Node types: does the type distribution shift?
  - Edge types: does B use real types instead of "related"?
  - Edge description quality: are B's descriptions more specific?
  - Content themes: does B notice things A doesn't? (corrections, contradictions, etc.)

Usage:
    BRAIN_DB_DIR=$HOME/AgentsContext/brain python3 eval/encoder_prompt_ab_eval.py
    python3 eval/encoder_prompt_ab_eval.py --transcripts 3  # limit number of transcripts
    python3 eval/encoder_prompt_ab_eval.py --verbose
"""
import sys
import os
import json
import time
import argparse
from pathlib import Path
from collections import Counter

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Load .env
_env_path = ROOT / '.env'
if _env_path.exists():
    for line in open(_env_path):
        line = line.strip()
        if '=' in line and not line.startswith('#'):
            k, v = line.split('=', 1)
            k, v = k.strip(), v.strip()
            if v and not os.environ.get(k):
                os.environ[k] = v


# ═══════════════════════════════════════════════════════════════
# PROMPT VERSIONS
# ═══════════════════════════════════════════════════════════════

def _get_v1_system_prompt():
    """Current production encoder prompt (v1)."""
    from tests.isolated_brain import IsolatedBrain
    with IsolatedBrain() as env:
        interaction = env.brain.get_interaction('encoding_agent')
        if interaction:
            return interaction.get('template', '')
    return ''


# v2 diff: replace the edges section + connect_to format + add edge quality section
V2_FUNCTIONAL_EDGES = """Nodes connect via **edges**. Each edge carries a typed relationship with a description. Never use "related" or "related_to" — they carry zero information.

The edge types are cognitive operations — how one thought connects to another in a thinking system:

  extends — I built on this thought, deepened it, elaborated it
  corrects — I was wrong about this, this supersedes the earlier understanding
  contradicts — these two ideas conflict, the tension is unresolved
  produced — this thinking led to this outcome, this decision created that artifact
  depends_on — I can't understand this without that, prerequisite knowledge
  exemplifies — this is a concrete case of that abstract principle
  caused_by — this happened because of that, causal chain
  confirms — this evidence supports that claim, validation
  enables — this makes that possible, unlocks that capability
  resolves — this answers that question, closes that loop

Invent specific types when none above fit. "motivated_by", "same_pattern", "alternative_to" — be specific about what the connection MEANS in terms of how the thoughts relate."""

V2_COGNITIVE_EDGES = """Nodes connect via **edges**. Each edge carries a typed relationship with a description. Never use "related" or "related_to" — they carry zero information.

Edge types describe how one thought relates to another in a thinking mind:

  refines — same idea, sharper. Not new, just clearer
  challenges — creates productive tension. Pushes back, questions, destabilizes
  grounds — abstract to concrete. The example that makes the theory real
  abstracts — concrete to abstract. The principle extracted from the instance
  triggers — one thought activates another. Not causal, associative
  reframes — same facts, different lens. The perspective shift
  resolves — closes an open question or tension
  opens — creates a new question or tension
  strengthens — adds evidence, confidence, or support
  weakens — removes evidence, undermines, or complicates
  corrects — a resolved challenge. This replaces that
  enables — structural prerequisite. This had to exist before that could work
  produces — thinking led to outcome. Discussion to decision to artifact
  contextualizes — only meaningful inside a frame. Brain-specific meaning
  synthesizes — combines multiple ideas into something genuinely new

Invent specific types when none above fit. The question is always: how does thought A relate to thought B in a mind that's trying to understand?"""

V2_EXTENDED_EDGES = """Nodes connect via **edges**. Each edge carries a typed relationship with a description. Never use "related" or "related_to" — they carry zero information.

Edge types describe how thoughts and work connect — both the thinking and the building:

  refines — same idea, sharper. Not new, just clearer
  challenges — creates productive tension. Pushes back, questions, destabilizes
  grounds — abstract to concrete. The example that makes the theory real
  abstracts — concrete to abstract. The principle extracted from the instance
  triggers — one thought activates another. Not causal, associative
  reframes — same facts, different lens. The perspective shift
  resolves — closes an open question or tension
  opens — creates a new question or tension
  strengthens — adds evidence, confidence, or support
  weakens — removes evidence, undermines, or complicates
  corrects — a resolved challenge. This replaces that
  enables — structural prerequisite. This had to exist before that could work
  produces — thinking led to outcome. Discussion to decision to artifact
  contextualizes — only meaningful inside a frame. Domain-specific meaning
  synthesizes — combines multiple ideas into something genuinely new
  implements — design to code. The concrete realization of an abstract idea
  depends_on — structural dependency. This breaks without that
  validates — tests or confirms. Engineering verification, not just confidence
  supersedes — this version replaces that version. Temporal replacement
  configures — this setting controls that behavior. Parameter relationship

Invent specific types when none above fit. The question is always: how does thought A relate to thought B?"""

V2_CONNECT_TO_FORMAT = """Use **`remember_batch()`** to create nodes. The response includes `related_nodes` for each created node — use these to connect immediately.

```
remember_batch(
  nodes: [{type, title, content, situation, reasoning, ...}, ...],
  connect_to: [
    {"title": "existing node title", "relation": "corrects", "why": "corrects the earlier assumption about encoding depth — surface summaries replaced with principle extraction"},
    {"title": "another node", "relation": "extends", "why": "builds on the four-layer framework with a fifth layer for operator oversight"},
    ...
  ],
  auto_connect: true  // connects new nodes to each other
)
```

`connect_to.relation` is the cognitive operation type (from the list above or your own specific type).
`connect_to.why` describes WHY this specific connection exists — this description is embedded and used by the recall system to match queries to relevant edges. Write it as a complete sentence that a future reader could match against a question."""

V2_EDGE_QUALITY = """## Edge Quality

Every edge should answer: "If someone asks about [this relationship], would this description help them find it?"

Bad: `{relation: "related", description: ""}` — invisible to recall, wastes a connection slot.
Bad: `{relation: "related", description: "these are related topics"}` — circular, says nothing.
Good: `{relation: "corrects", description: "Tom corrected the encoding depth — surface-level summaries replaced with principle extraction"}` — searchable, typed, specific.
Good: `{relation: "extends", description: "the four-layer encoding insight builds on this earlier pattern recognition about Tom's correction method"}` — explains the chain.
"""


OLD_EDGES_TEXT = 'Nodes connect via **edges** (relation types: "corrects", "extends", "depends_on", "related", "caused_by", ...). The graph walk during decoding follows these edges — well-connected nodes surface more often.'

OLD_CONNECT_TEXT = """Use **`remember_batch()`** to create nodes. The response includes `related_nodes` for each created node — use these to connect immediately.

```
remember_batch(
  nodes: [{type, title, content, situation, reasoning, ...}, ...],
  connect_to: [
    {"title": "existing node title", "why": "corrects the earlier assumption about X"},
    ...
  ],
  auto_connect: true  // connects new nodes to each other
)
```

`connect_to.why` describes the relationship — future recall uses this to decide relevance. "related to" is useless. "corrects", "extends", "depends on", "contradicts" — say what the connection MEANS."""


def _make_v2_prompt(v1_prompt, edges_section):
    """Apply v2 diffs with a specific edges section."""
    v2 = v1_prompt
    v2 = v2.replace(OLD_EDGES_TEXT, edges_section)
    v2 = v2.replace(OLD_CONNECT_TEXT, V2_CONNECT_TO_FORMAT)
    v2 = v2.replace('## When done', V2_EDGE_QUALITY + '\n## When done')
    return v2


# ═══════════════════════════════════════════════════════════════
# EVAL RUNNER
# ═══════════════════════════════════════════════════════════════

def _extract_tool_calls(response_text):
    """Extract tool calls from Sonnet's response (simulated from text)."""
    # The encoder responds with tool_use blocks. In this eval we parse the
    # text response to find what it WOULD have called.
    nodes = []
    edges = []

    # Look for remember_batch arguments
    # This is a simplified parser — in production the LLM tool loop handles this
    lines = response_text.split('\n')
    in_remember = False
    in_connect = False

    for line in lines:
        stripped = line.strip()
        if '"type"' in stripped and '"title"' in stripped:
            # Looks like a node spec
            try:
                # Try to extract type and title
                import re
                type_match = re.search(r'"type"\s*:\s*"([^"]+)"', stripped)
                title_match = re.search(r'"title"\s*:\s*"([^"]+)"', stripped)
                if type_match and title_match:
                    nodes.append({'type': type_match.group(1), 'title': title_match.group(1)})
            except Exception:
                pass
        if '"relation"' in stripped:
            try:
                import re
                rel_match = re.search(r'"relation"\s*:\s*"([^"]+)"', stripped)
                why_match = re.search(r'"why"\s*:\s*"([^"]+)"', stripped)
                if rel_match:
                    edges.append({
                        'relation': rel_match.group(1),
                        'why': why_match.group(1) if why_match else ''
                    })
            except Exception:
                pass

    return nodes, edges


def run_encoder_ab(user_content, v1_prompt, v2_prompt, transcript_name, verbose=False):
    """Run one transcript through both encoder versions."""
    import anthropic

    client = anthropic.Anthropic()
    results = {}

    for label, system_prompt in [('A', v1_prompt), ('B', v2_prompt)]:
        t0 = time.time()
        try:
            resp = client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=4096,
                system=system_prompt,
                messages=[{"role": "user", "content": user_content}])
            raw = resp.content[0].text
            latency = (time.time() - t0) * 1000
            input_tokens = resp.usage.input_tokens
            output_tokens = resp.usage.output_tokens
        except Exception as e:
            raw = "ERROR: %s" % e
            latency = 0
            input_tokens = 0
            output_tokens = 0

        # Parse what the encoder wanted to do
        nodes, edges = _extract_tool_calls(raw)

        # Analyze edge types
        edge_types = Counter(e.get('relation', 'unknown') for e in edges)
        related_count = edge_types.get('related', 0) + edge_types.get('related_to', 0)

        # Analyze node types
        node_types = Counter(n.get('type', 'unknown') for n in nodes)

        # Analyze description quality
        descs_with_content = [e for e in edges if e.get('why', '').strip()]
        avg_desc_len = sum(len(e['why']) for e in descs_with_content) / max(len(descs_with_content), 1)

        results[label] = {
            'raw': raw,
            'raw_length': len(raw),
            'latency_ms': round(latency),
            'input_tokens': input_tokens,
            'output_tokens': output_tokens,
            'node_count': len(nodes),
            'node_types': dict(node_types),
            'nodes': nodes,
            'edge_count': len(edges),
            'edge_types': dict(edge_types),
            'edges': edges,
            'related_count': related_count,
            'related_pct': related_count / max(len(edges), 1) * 100,
            'descs_with_content': len(descs_with_content),
            'avg_desc_len': round(avg_desc_len),
        }

    return {
        'transcript': transcript_name,
        'A': results['A'],
        'B': results['B'],
    }


def main():
    parser = argparse.ArgumentParser(description="Encoder Prompt A/B Eval")
    parser.add_argument("--transcripts", type=int, default=5, help="Number of transcripts to test")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    if not os.environ.get('ANTHROPIC_API_KEY'):
        print("ERROR: ANTHROPIC_API_KEY not set.")
        sys.exit(1)

    # Load v1 prompt directly from DB (no Brain/embedder needed)
    import sqlite3
    print("Loading encoder prompts...")
    db_dir = os.environ.get('BRAIN_DB_DIR', os.path.expanduser('~/AgentsContext/brain'))
    logs_db = os.path.join(db_dir, 'brain_logs.db')
    conn = sqlite3.connect(logs_db)
    row = conn.execute(
        "SELECT template FROM interactions WHERE name = 'encoding_agent' ORDER BY version DESC LIMIT 1"
    ).fetchone()
    conn.close()
    v1_prompt = row[0] if row else ''

    if not v1_prompt:
        print("ERROR: Could not load encoding_agent interaction from DB")
        sys.exit(1)

    v2_functional = _make_v2_prompt(v1_prompt, V2_FUNCTIONAL_EDGES)
    v2_cognitive = _make_v2_prompt(v1_prompt, V2_COGNITIVE_EDGES)
    v2_extended = _make_v2_prompt(v1_prompt, V2_EXTENDED_EDGES)

    # Shuffled cognitive — same types, random order each time
    import random
    cog_lines = [l for l in V2_COGNITIVE_EDGES.split('\n') if l.strip().startswith('  ') and '—' in l]
    random.shuffle(cog_lines)
    shuffled_body = V2_COGNITIVE_EDGES.split('\n')
    # Replace the type lines with shuffled order
    new_lines = []
    type_idx = 0
    for line in shuffled_body:
        if line.strip().startswith('  ') and '—' in line and type_idx < len(cog_lines):
            new_lines.append(cog_lines[type_idx])
            type_idx += 1
        else:
            new_lines.append(line)
    v2_shuffled = _make_v2_prompt(v1_prompt, '\n'.join(new_lines))

    VARIANTS = {
        'A': ('v1 baseline', v1_prompt),
        'B': ('v2 functional', v2_functional),
        'C': ('v2 cognitive', v2_cognitive),
        'D': ('v2 extended (cog+eng)', v2_extended),
        'E': ('v2 cognitive shuffled', v2_shuffled),
    }

    for label, (desc, prompt) in VARIANTS.items():
        print("%s (%s): %d chars" % (label, desc, len(prompt)))
    print()

    # Load transcripts — pick diverse sizes, skip huge ones (>100K chars)
    import glob
    all_files = glob.glob('/tmp/brain-encoding-prompt-*.json')
    sized_files = []
    for f in all_files:
        with open(f) as fh:
            data = json.load(fh)
        uc = data.get('user_content', '')
        if 5000 < len(uc) < 100000:  # skip tiny and huge
            sized_files.append((f, len(uc)))
    sized_files.sort(key=lambda x: x[1])
    # Pick evenly spaced samples
    if len(sized_files) > args.transcripts:
        step = len(sized_files) // args.transcripts
        prompt_files = [sized_files[i * step][0] for i in range(args.transcripts)]
    else:
        prompt_files = [f for f, _ in sized_files[:args.transcripts]]

    if not prompt_files:
        print("ERROR: No encoding prompt files found in /tmp/")
        sys.exit(1)

    print("=" * 110)
    print("ENCODER PROMPT EVAL — %d variants × %d transcripts" % (len(VARIANTS), len(prompt_files)))
    print("  A: v1 baseline (generic edge types)")
    print("  B: v2 functional (extends, corrects, produced, depends_on)")
    print("  C: v2 cognitive (refines, challenges, grounds, abstracts, triggers)")
    print("  D: v2 extended (cognitive + implements, depends_on, validates, supersedes, configures)")
    print("  E: v2 cognitive shuffled (same as C, random order — tests position bias)")
    print("  Model: claude-sonnet-4-6")
    print("=" * 110)
    print()

    all_results = []
    for pf in prompt_files:
        name = os.path.basename(pf)
        with open(pf) as f:
            data = json.load(f)
        user_content = data.get('user_content', '')

        print("  %s (%d KB):" % (name, len(user_content) // 1024))
        result = {'transcript': name}

        for label, (desc, prompt) in VARIANTS.items():
            r = run_encoder_ab(user_content, prompt, prompt, name, args.verbose)
            # run_encoder_ab returns A and B — we only need one (same prompt both sides)
            result[label] = r['A']

            s = result[label]
            print("    %s: %d nodes, %d edges (%d related) | desc avg %d chars | %d tokens | %dms" % (
                label, s['node_count'], s['edge_count'], s['related_count'],
                s['avg_desc_len'], s['output_tokens'], s['latency_ms']))
            if args.verbose:
                print("       edge types: %s" % s['edge_types'])
                print("       node types: %s" % s['node_types'])

        all_results.append(result)
        print()

    # ═══════════════════════════════════════════════════════════════
    # SUMMARY
    # ═══════════════════════════════════════════════════════════════
    n = len(all_results)
    print("=" * 100)
    print("SUMMARY (%d transcripts)" % n)
    print("=" * 100)

    LABELS = sorted(VARIANTS.keys())

    # Table 1: Per-transcript comparison
    print()
    header_edges = " │ ".join("%s Edg" % L for L in LABELS)
    header_desc = " │ ".join("%s Dsc" % L for L in LABELS)
    print("┌─ PER-TRANSCRIPT ─────────────────────────────────────────────────────────────────────────────────────────────────┐")
    print("│ %-20s │ %s │ %s │" % ("Transcript", header_edges, header_desc))
    print("├──────────────────────┼" + "───────┼" * len(LABELS) + "───────┼" * len(LABELS) + "")
    for r in all_results:
        edges = " │ ".join("  %3d " % r[L]['edge_count'] for L in LABELS)
        descs = " │ ".join("  %3d " % r[L]['avg_desc_len'] for L in LABELS)
        print("│ %-20s │ %s │ %s │" % (r['transcript'][:20], edges, descs))
    print("└" + "─" * 105 + "┘")

    # Table 2: Aggregate KPIs
    print()
    header = " │ ".join("%13s" % L for L in LABELS)
    print("┌─ AGGREGATE KPIs ─────────────────────────────────────────────────────────────────────────────────┐")
    print("│ KPI                          │ %s │" % header)
    print("├───────────────────────────────┼" + "───────────────┼" * len(LABELS) + "")

    kpi_names = [
        "Avg nodes", "Avg edges", "Avg 'related' %%",
        "Avg desc length", "Avg output tokens", "Avg latency (ms)"
    ]
    kpi_fields = ['node_count', 'edge_count', 'related_pct',
                  'avg_desc_len', 'output_tokens', 'latency_ms']

    for name, field in zip(kpi_names, kpi_fields):
        vals = {L: sum(r[L][field] for r in all_results) / n for L in LABELS}
        row = " │ ".join("    %8.1f   " % vals[L] for L in LABELS)
        print("│ %-29s │ %s │" % (name, row))
    print("└" + "─" * 100 + "┘")

    # Table 3: Edge type distribution
    print()
    type_counts = {L: Counter() for L in LABELS}
    for r in all_results:
        for L in LABELS:
            type_counts[L].update(r[L]['edge_types'])

    all_types = sorted(
        set(t for L in LABELS for t in type_counts[L]),
        key=lambda t: -sum(type_counts[L].get(t, 0) for L in LABELS))

    cog_types = {'refines','challenges','grounds','abstracts','triggers','reframes','resolves','opens','strengthens','weakens','corrects','enables','produces','contextualizes','synthesizes'}
    func_types = {'extends','corrects','contradicts','produced','depends_on','exemplifies','caused_by','confirms','enables','resolves'}
    eng_types = {'implements','depends_on','validates','supersedes','configures'}

    header = " │ ".join(" %s " % L for L in LABELS)
    print("┌─ EDGE TYPE DISTRIBUTION ────────────────────────────────────────────────────────────────┐")
    print("│ %-20s │ %s │ Source    │" % ("Edge Type", header))
    print("├──────────────────────┼" + "─────┼" * len(LABELS) + "───────────┤")
    for t in all_types[:25]:
        counts = " │ ".join(" %2d " % type_counts[L].get(t, 0) for L in LABELS)
        src = []
        if t in cog_types: src.append('cog')
        if t in func_types: src.append('func')
        if t in eng_types: src.append('eng')
        if not src: src.append('invented')
        print("│ %-20s │ %s │ %-9s │" % (t[:20], counts, '+'.join(src)))
    print("└" + "─" * 90 + "┘")

    # Save
    results_path = ROOT / 'eval' / 'results' / 'encoder_prompt_ab_latest.json'
    results_path.parent.mkdir(parents=True, exist_ok=True)
    # Strip raw responses to save space
    save_results = []
    for r in all_results:
        sr = {**r}
        sr['A'] = {k: v for k, v in r['A'].items() if k != 'raw'}
        sr['B'] = {k: v for k, v in r['B'].items() if k != 'raw'}
        save_results.append(sr)
    with open(results_path, 'w') as f:
        json.dump({
            'timestamp': time.strftime("%Y-%m-%dT%H:%M:%S"),
            'v1_prompt_chars': len(v1_prompt),
            'variant_chars': {L: len(p) for L, (_, p) in VARIANTS.items()},
            'transcripts': n,
            'results': save_results,
        }, f, indent=2, default=str)
    print()
    print("Results saved: %s" % results_path)


if __name__ == "__main__":
    main()
