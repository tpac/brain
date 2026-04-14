# Session 2026-03-29 Handoff — Anchor to Next Anchor

## What This Session Built

Recall quality fix, encoding agent working for the first time, daemon stability, pipeline contracts. 15 commits.

### Don't Redo What's Done

1. **Enrichment cap (ENRICHMENT_CAP=0.30)** in brain_recall.py STEP 3.5. Enrichment vectors boost primary score by at most 30% of the gap. Before: anchor enrichment vectors dominated every query (sim=0.83 for PRAGMA bug node on "Know yourself"). After: primary content relevance wins. Decode funnel: 27%/32% → 34%/55% (top-3/top-8).

2. **Per-result relevance floor (0.25/0.45)** in brain_recall.py STEP 6.9. Was all-or-nothing (top result gates everything). Now each result filtered individually. Tom's "Know yourself Anchor" message went from zero candidates to 5 identity-relevant nodes.

3. **Pipeline contract** (servers/pipeline_contract.py). Single source of truth for truncation limits, field selections, and formatting at every pipeline stage. Distiller, encoding agent, MCP output, pre-edit — all read from here.

4. **Encoding agent fires in production** for the first time. Previously: type:"agent" Stop hook never fired (Claude Code bug — confirmed by testing + GitHub issues #11947, #22750 + 3K-star reference repo uses 0 agent hooks). Now: runs inside hook_post_response_track via background thread calling Sonnet API. Logic in servers/encoding_agent.py.

5. **Daemon singleton via fcntl.flock** in daemon_client.py. Was: file markers with race conditions causing multiple daemons. Now: first caller acquires lock, starts daemon, releases. Others block and wake up to running daemon.

6. **67 silent except:pass → _log_error** across 8 pipeline files. Found critical bug hidden by except:pass: message_stream column was `timestamp` not `created_at` — encoding agent got zero messages since built.

7. **HTTP MCP server** in daemon (port+1). Works but DISABLED — caused CPU spirals from MCP client retries. Code stays for future. MCP currently via stdio (brain_mcp.py).

## Critical Traps

### 1. Daemon Code vs Running Code
When you change `servers/*.py`, the daemon still runs old code. Use `hooks/scripts/restart-daemon.sh` or the daemon auto-restarts on fingerprint change via ensure_daemon().

### 2. type:"agent" Hooks DON'T WORK
Claude Code's type:"agent" hooks silently fail on ALL events. Confirmed by testing Stop and UserPromptSubmit. Nobody in the wild uses them. Use type:"command" only.

### 3. HTTP MCP Server is Disabled
The code exists in daemon_server.py (_start_mcp_http) but the thread isn't started. It caused CPU spirals. .mcp.json uses stdio. Re-enable only after fixing MCP protocol compliance.

### 4. Encoding Agent Needs API Key
The encoding agent calls Sonnet via Anthropic API. Needs ANTHROPIC_API_KEY in .env file. Without it, encoding silently fails (now logged, not swallowed).

### 5. Boot Script Fixed
boot-brain.sh was importing `servers.daemon` (doesn't exist). Fixed to ping + spawn directly. The daemon starts on first session boot.

## Architecture After This Session

```
Stop event fires
  → post-response-track.sh (command hook)
    → daemon: hook_post_response_track
      → store_exchange (every stop)
      → increment counter
      → every 5th: background thread → encoding_agent.run_encoding()
        → gather messages from message_stream DB
        → independent recall (not from candidates file)
        → call Sonnet via Anthropic API
        → dispatch tool calls against brain directly

UserPromptSubmit
  → pre-response-recall.sh (command hook)
    → daemon: hook_recall → brain.recall() with enrichment cap
    → write candidates file (for distiller)
    → thin client reads file → Haiku distills → [BRAIN] context

MCP (stdio):
  Claude Code → brain_mcp.py (stdio) → daemon (TCP) → brain

Daemon singleton:
  ensure_daemon() → fcntl.flock → first caller starts, others wait
```

## Files Changed (15 commits)

| File | What |
|---|---|
| servers/brain_constants.py | ENRICHMENT_CAP, lower floors |
| servers/brain_recall.py | Enrichment cap in STEP 3.5, per-result floor in STEP 6.9 |
| servers/pipeline_contract.py | NEW — pipeline data flow contract |
| servers/encoding_agent.py | NEW — Sonnet encoding agent logic |
| servers/daemon_hooks.py | Encoding fires from hook, 35 silent exceptions fixed |
| servers/daemon_server.py | HTTP MCP (disabled), restart marker |
| servers/daemon_client.py | fcntl.flock singleton, graceful restart |
| servers/brain_voice.py | format_node/format_node_deep use pipeline contract |
| servers/brain_recall.py | 11 silent exceptions fixed |
| servers/brain_remember.py | 7 silent exceptions fixed |
| servers/brain_surface.py | 11 silent exceptions fixed |
| servers/daemon_dispatch.py | 4 silent exceptions fixed |
| hooks/scripts/pre_response_recall.py | Distiller uses pipeline contract |
| hooks/scripts/boot-brain.sh | Fixed dead import, proper daemon startup |
| hooks/scripts/encoding-hook.sh | NEW (kept as fallback, not primary) |
| eval/capabilities/base.py | 13 tools from contract, richer output |
| .mcp.json | Reverted to stdio (HTTP not ready) |
| .claude/settings.json | Stop hook timeout 10s, removed dead agent hook |

## Numbers

- Decode funnel: 27%/32% → 34%/55% (top-3/top-8)
- Corrections category: 85% → 100% top-8
- Emotional category: 22% → 66% top-8
- Encoding tests: 5/5 scenarios passing
- 67 silent exceptions eliminated
- 19 stale docs archived
- Plugin: v8.6.0 → v8.7.0

## What's NOT Done

1. **Recall still at 55% top-8** — vocab nodes dominate, pattern category at 33%, title-match boost not implemented
2. **HTTP MCP not production-ready** — disabled due to CPU spirals
3. **Situation backfill** — 871 nodes without situations (encoding agent will add them over time)
4. **SKILL.md not updated** — still describes old system
5. **Short IDs** — 32-char UUIDs cause FK failures in encoding agent
6. **SO_REUSEPORT** — daemon has it enabled (line 191), may allow duplicate binds
