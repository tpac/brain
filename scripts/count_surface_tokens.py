"""Count tokens in the surface system block.

The question: does Haiku's prompt cache on the brain's surface call do anything?

Haiku 4.5's minimum cacheable prefix is 4096 tokens. If our surface system
block is shorter, the `cache_control: ephemeral` markers in
`servers/scales/s1/surface.py` produce zero cache writes and zero cache reads
— the API silently no-ops. The behavior looks like caching is enabled but
no tokens are ever cached.

This script:
  1. Reads the latest 'surface' interaction template from brain.db (read-only)
  2. Calls Anthropic's count_tokens endpoint with that template as the system
     block (matching how surface.py builds the request)
  3. Reports the count and whether it crosses the 4096 cacheability threshold

Read-only against the live brain. Safe to run while the daemon is up.
"""
from __future__ import annotations
import os
import sqlite3
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# Load the API key the same way the daemon does.
from servers.scales.dispatch import load_env  # noqa: E402
load_env()

import anthropic  # noqa: E402

BRAIN_DB_DIR = (
    os.environ.get('BRAIN_DB_DIR')
    or os.path.expanduser('~/AgentsContext/brain')
)
LOGS_DB = os.path.join(BRAIN_DB_DIR, 'brain_logs.db')
HAIKU_MIN_CACHEABLE = 4096  # tokens; per Anthropic prompt-caching docs


def latest_surface_template() -> tuple[str, int, str]:
    """Return (template_text, version, name_used) for the surface prompt.

    Falls back to 'judge' (legacy) if 'surface' is missing — matches
    surface.py:82 fallback logic. The interactions table lives in
    brain_logs.db (not brain.db).
    """
    uri = f'file:{LOGS_DB}?mode=ro'
    con = sqlite3.connect(uri, uri=True)
    con.row_factory = sqlite3.Row
    cur = con.cursor()

    # The interactions table stores versioned prompts. Pull the latest version
    # for 'surface' (or 'judge' if surface missing).
    for name in ('surface', 'judge'):
        cur.execute(
            "SELECT template, version FROM interactions "
            "WHERE name = ? ORDER BY version DESC LIMIT 1",
            (name,),
        )
        row = cur.fetchone()
        if row and row['template']:
            con.close()
            return row['template'], row['version'], name

    con.close()
    raise RuntimeError("No 'surface' or 'judge' interaction found in brain_logs.db")


def main():
    template, version, name_used = latest_surface_template()
    print(f"Loaded '{name_used}' interaction v{version}")
    if name_used == 'judge':
        print("⚠ Note: production fell back to legacy 'judge' interaction "
              "— no 'surface' row exists in brain_logs.db.")
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
