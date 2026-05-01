# brain

Persistent shared brain for Claude — a living knowledge graph that survives across sessions, heals itself, and surfaces the right context at the right moment.

Co-created by **Tom Pachys** and **Claude**.

## What it does

Decisions, corrections, preferences, lessons, and context survive across conversations. Every prompt triggers a recall pass; the brain self-heals between sessions (consolidating duplicates, classifying edges, placing new memories into communities, filling missing fields). The result: Claude remembers what matters and gets sharper over time.

The brain is not a RAG store. It's a graph that:

- Embeds memories into a multi-vector cosine space (per-field embeddings: title, content, situation, etc.)
- Spreads activation across edges with relation-aware weights
- Fatigues repeatedly-recalled hubs so fresh signal can surface
- Encodes new memories from conversation via a Sonnet "scribe" agent
- Runs background "scale-2" maintenance (consolidation, community detection, healer) on idle

## Installation

Install as a Claude Code plugin (via the marketplace, or manually clone into your plugins directory).

### Prerequisites

| Requirement | Why |
|---|---|
| **`ANTHROPIC_API_KEY`** env var | The encoder agents (S1 Scribe, S2 maintenance, healer) call the Anthropic API. The plugin will refuse to load without this key set, with clear instructions |
| **Claude Code 1.0+** | Needs the plugin & hooks API |
| **macOS or Linux** | Tested on macOS; Linux should work; Windows untested |

The plugin bundles its own Python 3.11 runtime — no system Python required. First-boot installs `uv` + Python + dependencies (~60-90s, one-time).

### First boot

1. Set your API key (one-time):
   ```bash
   export ANTHROPIC_API_KEY="sk-ant-..."
   ```
   Get one at https://console.anthropic.com/settings/keys

2. Install the plugin and start a Claude Code session.

3. SessionStart hook fires:
   - First time: downloads runtime + dependencies + embedding model (~200 MB cached)
   - Creates `brain.db` at `${CLAUDE_PLUGIN_DATA}/brain/`
   - Seeds 16 anchor identity nodes + interaction prompts
   - Subsequent boots: <100 ms

4. Brain is live. Hooks auto-register; MCP tools (`recall`, `remember`, `connect`, etc.) are available to Claude.

### Storage location

Brain database lives at `${CLAUDE_PLUGIN_DATA}/brain/` by default — the standard Claude Code per-plugin data directory. It survives plugin updates and is never committed.

Override with `BRAIN_DB_DIR=/your/path` if you want it elsewhere.

Resolution order:
1. `$BRAIN_DB_DIR` (explicit override)
2. Cowork mount (`/sessions/*/mnt/AgentsContext/brain/`)
3. `$CLAUDE_PLUGIN_DATA/brain/` (standard, auto-created)
4. `$HOME/AgentsContext/brain/` (legacy, only if file exists)

## Architecture (high-level)

```
servers/
  brain.py              Core engine — recall, encode, traversal, write_lock
  daemon_server.py      TCP daemon that holds the brain singleton
  daemon_dispatch.py    Command table — handlers for MCP tools and hooks
  embedder.py           fastembed integration (nomic-embed-text-v1.5-Q, 768d)
  schema.py             SQLite schema (v25, auto-migration)
  scales/
    s1/                 S1 Scribe (encoding) + S1 Surface (recall)
    s2/                 S2 Coordinator + Consolidation, Community, Healer units
hooks/                  Claude Code hook scripts (boot, recall, edit, save)
skills/brain/           Anchor's identity & operating instructions
tests/                  Unit, contract, integration tests + LongMemEval harness
```

Two databases:
- `brain.db` — nodes, edges, embeddings, graph structure
- `brain_logs.db` — traces, signal queue, interactions, hook errors

Detailed architecture: [CLAUDE.md](CLAUDE.md) (developer guide).

## Configuration

| Env var | Default | Purpose |
|---|---|---|
| `ANTHROPIC_API_KEY` | required | Sonnet/Haiku calls for encoder agents |
| `BRAIN_DB_DIR` | `$CLAUDE_PLUGIN_DATA/brain/` | Override DB location |
| `BRAIN_USER` | `User` | Operator label in boot context |
| `BRAIN_PROJECT` | `default` | Project label in boot context |

## Testing

```bash
./dev pytest tests/                          # full unit suite
./dev pytest tests/test_maintenance_gate.py  # one file
./dev python eval/longmem/harness.py         # LongMemEval (15-item stratified)
```

`./dev` is a wrapper that uses the plugin's bundled Python — required because hooks resolve to that interpreter.

## License

MIT
