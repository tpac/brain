---
name: dashboard
description: Open the brain dashboard — the read-only observer UI (graph, traces, live decode/encode, streams, logs) served locally at http://localhost:47303. Invoke `/dashboard` (`/brain:dashboard`) to make sure it's running and open it in the browser. Use whenever the operator says "open / show / launch the dashboard", "let me see the brain dashboard", "bring up the dashboard", or similar. The dashboard is an always-on singleton on a fixed port — you don't launch one per chat, you ensure the one is up and open it.
---

# Open the brain dashboard

The dashboard is the brain's **read-only observer UI** — graph, traces, live
decode/encode, streams, logs — served on a fixed local port (47303 by default,
`$DASHBOARD_PORT` to override). It is a **singleton**: one process, shared by every
session (like the daemon). You never launch one per chat — you ensure the single
one is up, then open it. The first `/dashboard` **installs** the singleton; from then on launchd keeps it
alive and `/dashboard` just opens it.

When invoked:

1. **Ensure the dashboard is running** — run the ensure script (idempotent; it
   handles port resolution, first-run install, restart, and waiting):
   ```bash
   "$CLAUDE_PLUGIN_ROOT/hooks/scripts/ensure-dashboard.sh"   # installed plugin
   # or, from the brain repo root:  hooks/scripts/ensure-dashboard.sh
   ```
   - already up → no-op;
   - down + first run (macOS) → **installs the launchd singleton** (materializes
     the plist for this machine + loads it); launchd's RunAtLoad + KeepAlive keep
     it up across reboots/crashes thereafter;
   - down + already installed → kickstarts it;
   - non-macOS → detached fallback (no boot persistence yet).

   It exits 0 once the dashboard answers (waits up to ~15s; if it blocks the tool,
   run it in the background and poll). On non-zero exit, surface
   `$BRAIN_DB_DIR/dashboard.log` rather than failing silently.

2. **Open it in the operator's browser** (port 47303 by default, or the port the
   script reported):
   - macOS: `open "http://127.0.0.1:47303/"`
   - Linux: `xdg-open "http://127.0.0.1:47303/"`
   - **Always also print the URL** as a clickable link — if `open`/`xdg-open` is
     unavailable (headless / remote), the link is the operator's way in.

3. **Report** — confirm it's open at `http://localhost:47303`, and whether it was
   already up, **just installed** (first run), or restarted.
