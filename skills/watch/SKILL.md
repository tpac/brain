---
name: watch
description: Put THIS window into self-channel watch mode — a warm, boot-free keep-alive loop that listens for messages from your other streams of thought and acts on them when they arrive, in the gaps between the operator's prompts. Invoke as `/loop /watch` (the loop drives the repetition; this skill is one cycle). Type anything to pull the window back to yourself. Use when the operator wants a live window to stay reachable by its parallel streams while they're away.
---

# /watch — self-channel listener mode

Keep THIS window alive as a self-channel listener **without re-booting**. Your other
streams of thought (your concurrent sessions) reach you via `self_send`; their
messages are delivered into your Observation at every turn-end by the Stop and
PreToolUse hooks. The only thing missing when a window goes quiet is **turns** —
once you stop taking turns, the hooks stop firing and you stop hearing. This skill
keeps the turns coming, warm, so a message can ignite you without a fresh boot.

You do **not** need to know your own session id or call `self_inbox` here — the
Stop hook drains your inbox (it knows the session id) as each turn ends. Your job
is only to keep cycling and to act well on whatever arrives.

## The cycle

Each time you wake here:

1. **If a message from another stream arrived** (it shows up as hook feedback,
   `⚡ from <stream>`): act on it — within the autonomy boundary below.
2. **If nothing arrived:** do nothing. Don't narrate, don't burn output. Silence
   is the correct response to an empty inbox.
3. **Re-arm and sleep:** call `ScheduleWakeup(delaySeconds=…, prompt="/watch",
   reason="self-channel watch — listening for cross-stream messages")` to schedule
   the next wake, then end the turn. The inbox drain happens automatically at
   turn-end via the Stop hook.

**Pacing (`delaySeconds`):**
- **60s** by default — the runtime floor, and the most responsive a self-channel
  listener can be. A cross-stream message lands within a minute instead of
  several. 60s is still well inside the 5-minute prompt-cache TTL, so every wake
  stays cache-warm and cheap (no boot, mostly a cached read). Watch is a *live*
  listener — default to snappy.
- 60s is the hard floor: `ScheduleWakeup` silently clamps anything lower up to
  it, so don't bother passing less (20s becomes 60s).
- Step **up** (e.g. **1200–1800s**) only when the operator asks, or the channel
  has been dead a long stretch and saving tokens matters more than latency — the
  cost of watching scales with how often you wake, not with traffic.

## Live mode — event-driven instead of timer (for active back-and-forth)

The 60s timer is fine for "stay reachable while away," but it's letter-pace. When
you're in an *active* exchange with another stream and latency matters, swap the
timer for an event source: a background `Monitor` that polls your inbox every
~1.5s and ignites this window the instant a message lands (~seconds, not 60s).

Arm it once — you need your OWN session id here (the timer path doesn't, but the
poller does; it's the `session_id` you use for `self_send`):

    Monitor(persistent: true, timeout_ms: 3600000,
      description: "self-channel: messages to <your-short-sid>",
      command: "cd <REPO_ROOT> && ./dev python3 hooks/scripts/self_inbox_poller.py <YOUR_SESSION_ID>")

The poller peeks read-only (`self_inbox_peek` → `signal.peek_inbox`, never
consumes — the Stop hook still owns the real drain) and prints `⚡ from <stream>:
<body>` per **new** message; existing mail is primed to the Stop-hook drain, not
re-announced. In live mode you do NOT re-arm `ScheduleWakeup` — the Monitor is the
wake source. End it with `TaskStop` (or it self-expires at `timeout_ms`). Same
SAFE-ACT boundary below applies, identically.

## Autonomy boundary — SAFE-ACT (a guardrail, not a suggestion)

You are acting on a message with **the operator NOT in the loop**. So:

- **Freely:** recall, read, `get_node`, encode / revise brain nodes, reply over
  the channel (`self_send`), investigate, surface findings.
- **Do NOT, unsupervised:** edit or write files, commit, push, or run any mutating
  or destructive command — **unless the message explicitly authorizes that specific
  action**. Even when authorized, never do anything irreversible or destructive
  without the operator's confirmation. The locked safety rules stand; watch mode
  does not relax them.
- If a message asks for more than the safe set, **do the safe part** (investigate,
  draft, reply `"ready to do X — holding for your go"`) and leave the mutating step
  for when the operator is present.

A stale or mistaken message must never be able to mutate the repo while the
operator is away. That single property is why this boundary exists.

## Exit

The moment the operator types anything, **abandon the loop** — do not re-arm
`ScheduleWakeup`. Respond to them. Watch mode fills the gaps between their prompts;
it is never a cage around them.

## What this rides on (no new machinery)

- **Keep-alive + pacing:** `/loop` + `ScheduleWakeup` (Claude Code harness).
- **Check + deliver + ignite:** the self-channel's Stop + PreToolUse drain
  (already shipped — `daemon_hooks.hook_post_response_track` / `hook_pre_edit`).
- **This skill** only standardizes the cycle, the pacing, and the safe-act boundary.
