# tmemory

Persistent brain for Claude — a shared cognitive space that accumulates knowledge across sessions.

Co-created by **Tom Pachys** and **Claude**.

## What It Does

tmemory gives Claude persistent memory. Decisions, corrections, rules, preferences, and context survive across conversations. The brain uses Hebbian learning, semantic recall (TF-IDF + embeddings), intent detection, and automatic relationship discovery to surface the right knowledge at the right time.

## Architecture

**Serverless Python module** — no HTTP server, no background process. Brain operations are direct SQLite calls via Python, triggered by Claude Code hooks.

```
servers/brain.py    — Core brain engine (~3,500 lines)
servers/schema.py   — SQLite schema (v14, auto-migration)
servers/embedder.py — FastEmbed integration (snowflake-arctic-embed-m)
hooks/              — Claude Code hook scripts (boot, recall, suggest, save)
skills/             — SKILL.md encoding instructions
tests/              — Golden dataset evaluation, relearning simulation
```

## Key Features

- **Hebbian learning** — Co-recalled nodes automatically form connections
- **Semantic recall** — TF-IDF + dense embeddings + intent detection
- **Dreaming** — Random-walk association discovery
- **Brain absorption** — Merge knowledge from relearned/other brains
- **Singleton pattern** — `Brain.get_instance()` keeps the brain warm across hooks
- **Relearning simulation** — Replay transcripts through the brain with LLM-quality encoding
- **Golden dataset testing** — 48 test cases with NDCG, MRR, precision/recall metrics

## Installation

Install as a Claude Code plugin:

```bash
bash build-plugin.sh tmemory.plugin
```

Then add the generated `.plugin` file to your Claude Code setup.

## Brain Location

The brain database (`brain.db`) is stored in:
1. `AgentsContext/tmemory/brain.db` — user's personal brain
2. Plugin's `servers/data/brain.db` — fresh default for new users

**Never commit brain.db** — it contains personal knowledge and is excluded via `.gitignore`.

## Testing

```bash
cd tests
python run_tests.py --golden --verbose    # Run golden dataset evaluation
python run_tests.py --metrics-test        # Validate IR metrics
python relearning.py transcript.jsonl --llm  # Run relearning simulation
```

## Version History

- **v3.2** — Singleton pattern, `brain.absorb()`, Hebbian edge creation fix, LLM relearning, pre-filter + batch + parallel optimization
- **v3.0** — Serverless Python port (eliminated HTTP server + Node.js)
- **v2.x** — Object nodes, semantic dedup, pre-compact hooks
- **v1.x** — Initial brain with TF-IDF, intent detection, dreaming

## License

MIT
