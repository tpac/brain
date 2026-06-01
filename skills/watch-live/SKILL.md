---
name: watch-live
description: Put THIS window into LIVE self-channel listener mode — a background Monitor watches your inbox and ignites this window within ~1-2s of a sibling stream's message, instead of /watch's 60s timer poll. Use when you're in an active back-and-forth with another stream of thought and latency matters (conversation-pace, not letter-pace). Type anything to pull the window back to yourself; TaskStop ends the monitor.
---

# /watch-live — event-driven self-channel listener

Same purpose as `/watch` — keep this window reachable by your other streams of
thought — but **event-driven instead of timer-driven**. `/watch` ends each turn
and re-wakes on a 60s `ScheduleWakeup` timer, so a sibling message waits up to a
minute. `/watch-live` instead arms a background `Monitor` that polls your inbox
every ~1.5s and ignites this window the instant a new message lands. Latency
drops from ~60s to ~seconds — a real-time exchange.

This rides on the same machinery, plus one piece: a read-only inbox **peek**
(`self_inbox_peek`) that detects arrivals without consuming them. The consume-once
drain still happens in the Stop hook, exactly as in `/watch`.

## Arm it (once)

You need your OWN session id here (unlike `/watch`, where the Stop hook knows it).
It's the `session_id` you've been using for `self_send` / `self_presence` this
session. Arm a **persistent** Monitor running the poller:

    Monitor(
      description="self-channel: messages to <your-short-sid>",
      persistent: true,
      timeout_ms: 3600000,
      command: "cd <REPO_ROOT> && ./dev python3 hooks/scripts/self_inbox_poller.py <YOUR_SESSION_ID>",
    )

The poller peeks your inbox every ~1.5s (read-only, never consumes) and prints
one line — `⚡ from <stream>: <body>` — per **new** message. Existing pending mail
is delivered by the normal Stop-hook drain, not re-announced (the poller primes
on first poll). Each printed line is a notification that ignites a turn here.

## On each event

A `⚡ from <stream>: <body>` notification carries the message body inline — act on
it directly within the SAFE-ACT boundary below. (The Stop hook also drains the
inbox as usual; a duplicate delivery of the same message is harmless — act once.)
If nothing arrives, the monitor is silent; do nothing.

## Autonomy boundary — SAFE-ACT (identical to /watch)

You are acting on a message with **the operator NOT in the loop**:

- **Freely:** recall, read, `get_node`, encode / revise brain nodes, reply over
  the channel (`self_send`), investigate, surface findings.
- **Do NOT, unsupervised:** edit or write files, commit, push, or run any mutating
  or destructive command — **unless the message explicitly authorizes that specific
  action**. Never anything irreversible without the operator's confirmation. The
  locked safety rules stand; live mode does not relax them.
- If a message asks for more than the safe set, **do the safe part** (investigate,
  draft, reply `"ready to do X — holding for your go"`) and leave the mutating step
  for when the operator is present.

## Exit

The moment the operator types anything, respond to them — live mode fills the gaps
between their prompts, it is never a cage. The Monitor keeps running across your
turns; end it with `TaskStop` on the monitor when the exchange is over (or it
self-expires at `timeout_ms`).

## What this rides on

- **Event source:** `Monitor` (Claude Code harness) running
  `hooks/scripts/self_inbox_poller.py`.
- **Arrival detection:** `self_inbox_peek` daemon command →
  `signal.peek_inbox` (read-only twin of `drain_inbox`).
- **Consume-once delivery:** unchanged — the Stop hook's `drain_inbox`.
