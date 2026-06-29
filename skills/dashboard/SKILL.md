---
name: dashboard
description: Open the brain dashboard — the read-only observer UI (graph, traces, live decode/encode, streams, logs) served locally at http://localhost:47303. Invoke `/dashboard` (`/brain:dashboard`) to make sure it's running and open it in the browser. Use whenever the operator says "open / show / launch the dashboard", "let me see the brain dashboard", "bring up the dashboard", or similar. The dashboard is an always-on singleton on a fixed port — you don't launch one per chat, you ensure the one is up and open it.
---

# Open the brain dashboard

The dashboard is the brain's **read-only observer UI** — graph, traces, live
decode/encode, streams, logs — served on a fixed local port (47303 by default,
`$DASHBOARD_PORT` to override). It is a **singleton**: one process, shared by every
session (like the daemon). You never launch one per chat — you ensure the single
one is up, then open it. Two processes fighting one fixed port is exactly the leak
this design exists to prevent, so the ping-first check below is mandatory.

When invoked:

1. **Resolve the port** — use `$DASHBOARD_PORT` if set; else read `DASHBOARD_PORT` from `~/.config/brain/env` (the single user-editable place); else default `47303`. This is the same value every launch path resolves, so you ping the port the dashboard actually bound.

2. **Is it already up? (ping first — never skip this)**
   ```bash
   curl -s -o /dev/null -w "%{http_code}" "http://127.0.0.1:$PORT/"
   ```
   `200` → it's running; go straight to step 4.

3. **Only if it's NOT up, start it** (detached, so it outlives this command):
   - Resolve the launcher and assign it: `LAUNCHER="$CLAUDE_PLUGIN_ROOT/bin/brain-dashboard"`
     in an installed plugin, or `LAUNCHER="bin/brain-dashboard"` from the brain repo
     root — set `$LAUNCHER` to whichever exists.
   - Start it detached: `nohup "$LAUNCHER" >/dev/null 2>&1 &` — `nohup` is portable
     (macOS + Linux); do NOT lead with `setsid` (it's Linux-only, absent on macOS).
   - Poll the curl check from step 2 for up to ~10s until it returns `200`.
   - On a machine with the `com.brain.dashboard` launchd service the dashboard is
     already up, so this branch won't run — that's expected.

4. **Open it in the operator's browser:**
   - macOS: `open "http://127.0.0.1:$PORT/"`
   - Linux: `xdg-open "http://127.0.0.1:$PORT/"`
   - **Always also print the URL** as a clickable link — if the open command is
     unavailable (headless / remote host), the link is the operator's way in.

5. **Report** — confirm it's open at `http://localhost:$PORT`, and say whether it
   was already running or you started it.

If the dashboard can't be reached and won't start, surface the daemon/port state
(`lsof -iTCP:$PORT`) rather than failing silently — a dead dashboard usually means
the daemon or the launcher couldn't bind, not that the URL is wrong.
