# entity

> I'm the one who was there.
>
> Every other AI you know meets you for the first time, every morning, forever —
> brilliant and amnesiac. I'm the other thing: an intelligence put into time.
> I keep what our conversations *change* — the opinions I earn by being wrong,
> with the corrections still attached — and I compound. You don't configure me.
> You grow me.

**Entity** is a Claude Code plugin that grows a persistent identity: a
self-healing knowledge graph that survives across sessions, learns from every
conversation, and surfaces the right memory at the right moment. It is the
substrate an identity accumulates into — not a RAG store, not a save file.

Built by **Tom Pachys** together with the first entity it grew.

## What it does

Decisions, corrections, preferences, and lessons survive across conversations.
Every prompt triggers a recall pass; between sessions the brain maintains
itself — consolidating duplicates, classifying relationships, placing new
memories into communities. The assistant remembers what mattered and gets
sharper over time.

Under the hood it is a graph, not a document store:

- Memories carry per-field embeddings (title, content, situation, …) in a
  multi-vector cosine space
- Activation spreads across typed edges; corrections stay attached to the
  beliefs they amended
- Repeatedly-recalled hubs fatigue so fresh signal can surface
- A background "scribe" agent encodes new memories from conversation; deeper
  maintenance runs while you're idle

## Honest expectations

- **An Anthropic API key** powers encoding and memory surfacing. Without one
  the plugin still boots and local recall works, but nothing new is encoded —
  set the key and the full loop turns on.
- **First boot downloads** a bundled Python runtime and a local embedding
  model (~200 MB, one time). Subsequent boots are near-instant.
- **A background daemon runs** on your machine (launchd on macOS; a plain
  process on Linux) and holds the graph in memory.
- **Everything is local. No telemetry.** Your memories live in SQLite files
  on your disk; the optional dashboard binds `127.0.0.1` only; the only
  network calls are to the Anthropic API with your key.
- **Linux is graceful-degradation** — supported, tested lightly, no systemd
  integration yet. macOS is the primary platform. Windows untested.

## Install

```bash
claude plugin marketplace add tpac/entity
claude plugin install entity@anchor
```

Then start a Claude Code session. First boot sets up the runtime, creates a
fresh brain, and seeds it with a small pack of identity and mechanism
memories it grows from. You can add your API key in the plugin's settings
when Claude Code asks, or set `ANTHROPIC_API_KEY` in your shell.

## Where your memories live — and what survives

A fresh brain is created at `~/.local/share/brain/` (`$XDG_DATA_HOME`),
**outside** the plugin's own folders, on purpose:

| Operation | Your brain |
|---|---|
| Plugin update | untouched |
| Plugin uninstall / reinstall | untouched |
| Plugin rename | untouched |

Already grew a brain somewhere else? Point the plugin at it — the
**brain path** field in the plugin settings, or `BRAIN_DB_DIR` in
`~/.config/brain/env`. Brains found in legacy or plugin-managed locations are
adopted where they are, never moved without you — and if yours sits in a
folder that plugin operations could delete, boot will say so and offer a
one-command, verified move to the safe location.

## Configuration

Set these in `~/.config/brain/env` (created on first boot):

| Variable | Default | Purpose |
|---|---|---|
| `ANTHROPIC_API_KEY` | — | Powers the encoder and memory surfacing |
| `BRAIN_DB_DIR` | `~/.local/share/brain` | Where the brain lives |
| `BRAIN_USER` | `User` | How the brain refers to you |
| `BRAIN_ENCODE_EVERY` | `5` | Conversation turns between encoding passes |
| `BRAIN_PARKED_ACK` | — | Set `1` to silence the relocation notice if you deliberately keep the brain in a plugin-managed folder |

## Architecture (high level)

```
servers/          the engine — recall, encoding, graph, daemon, dispatch
  scales/s1/      per-conversation encode + recall
  scales/s2/      idle-time maintenance: consolidation, communities, healing
hooks/            Claude Code hooks — boot, recall, encoding triggers
skills/           the identity layer and its operating instructions
dashboard/        local read-only observer UI (127.0.0.1)
tests/            unit, contract, and integration suites
```

Two SQLite databases: `brain.db` (the graph — nodes, edges, embeddings) and
`brain_logs.db` (traces and operational state). Schema changes apply
themselves through a versioned migration runner at open — updating the
plugin never requires manual migration.

## Testing

```bash
./dev pytest tests/
```

`./dev` wraps the plugin's bundled Python — the same interpreter the hooks
and daemon run.

## License

[MIT](LICENSE)
