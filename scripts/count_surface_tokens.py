"""Count tokens in the surface system block.

The question: does Haiku's prompt cache on the brain's surface call do anything?

Haiku 4.5's minimum cacheable prefix is 4096 tokens. If our surface system
block is shorter, the `cache_control: ephemeral` markers in
`servers/scales/s1/surface.py` produce zero cache writes and zero cache reads
— the API silently no-ops. The behavior looks like caching is enabled but
no tokens are ever cached.

This script:
  1. Resolves the EFFECTIVE 'surface' template — the deployed override if one
     exists, else the code default — the same rule the runtime resolves by
  2. Calls Anthropic's count_tokens endpoint with that template as the system
     block (matching how surface.py builds the request)
  3. Reports the count and whether it crosses the 4096 cacheability threshold

Read-only, via the daemon. Safe to run while the daemon is up.
"""
from __future__ import annotations
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# Load the API key the same way the daemon does.
from servers.scales.dispatch import load_env  # noqa: E402
load_env()

import anthropic  # noqa: E402

HAIKU_MIN_CACHEABLE = 4096  # tokens; per Anthropic prompt-caching docs


def effective_surface_template() -> tuple[str, str]:
    """Return (template_text, provenance) for the surface prompt the runtime
    actually sends.

    Resolution is the override model's, not "latest registered": the active
    pointer's template when a pointer exists and its template is non-empty,
    else the code default. Reading MAX(version) — what this did before — could
    report an un-eval'd dormant candidate's token count while production sends
    something else entirely, which defeats the only question the script asks.

    Routed through the daemon rather than a connection of its own; there is no
    daemon door for the RESOLVED value yet, so the override half comes from
    `get_interaction` and the default half from the code registry. One
    `get_interaction_effective` command would collapse this to a single call.
    """
    from servers.daemon_client import send_command
    from servers.interaction_defaults import INTERACTION_DEFAULTS

    r = send_command('get_interaction', {'name': 'surface'})
    row = (r.get('result') or {}) if isinstance(r, dict) else {}
    if row.get('template'):
        return row['template'], "override v%s" % row.get('version')

    template = INTERACTION_DEFAULTS['surface'][0]
    if not template:
        raise RuntimeError(
            "no surface override deployed and no code default in "
            "servers/interaction_defaults.py — nothing to measure")
    return template, 'code default (no override deployed)'


def main():
    template, provenance = effective_surface_template()
    print(f"Loaded 'surface' interaction — {provenance}")
    print(f"Template byte length: {len(template):,} bytes "
          f"({len(template) / 1024:.1f} KB)")

    client = anthropic.Anthropic()

    # Mirror the surface.py request shape exactly:
    # - model = haiku 4.5
    # - system block carries cache_control: ephemeral
    # - user content is the per-turn delta (we use a placeholder here)
    resp = client.messages.count_tokens(
        model='claude-haiku-4-5',
        system=[{
            'type': 'text',
            'text': template,
            'cache_control': {'type': 'ephemeral'},
        }],
        messages=[{'role': 'user', 'content': 'placeholder for token count'}],
    )

    n = resp.input_tokens
    print(f"\ncount_tokens(system + 1-message) = {n:,} tokens")
    print(f"Haiku 4.5 minimum cacheable prefix = {HAIKU_MIN_CACHEABLE:,} tokens")

    if n < HAIKU_MIN_CACHEABLE:
        deficit = HAIKU_MIN_CACHEABLE - n
        print(f"\n❌ BELOW THRESHOLD by {deficit:,} tokens.")
        print(
            "   The cache_control: ephemeral marker in surface.py is currently "
            "a no-op on Haiku. cache_creation_input_tokens=0, "
            "cache_read_input_tokens=0 on every surface call. "
            "Cache warmup at boot would also no-op."
        )
        print(
            "   Action items:\n"
            "     1. Either accept this and remove the cache_control marker\n"
            "        (it's misleading scaffolding right now), OR\n"
            "     2. Pad the system block past 4096 (e.g. add Frame, vocab,\n"
            "        or other stable prior into the cached prefix) to make\n"
            "        the marker do real work."
        )
    else:
        margin = n - HAIKU_MIN_CACHEABLE
        print(f"\n✅ ABOVE THRESHOLD by {margin:,} tokens.")
        print(
            "   The cache_control marker is doing real work. Warmup at boot\n"
            "   that hits the same prefix will write the cache (1.25× input\n"
            "   price) so the first user prompt reads at 0.1× and saves\n"
            "   latency."
        )


if __name__ == '__main__':
    main()
