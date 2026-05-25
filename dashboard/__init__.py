"""Standalone Brain Dashboard.

Read-only observer over brain.db / brain_logs.db plus the daemon TCP socket.
Never writes to brain databases — the dashboard's job is to surface what the
brain already knows.

Package layout:
  brain_dashboard_standalone.py  — thin entry point (preserves the launch path)
  server.py                       — ThreadedHTTPServer + DashboardHandler routes
  daemon_client.py                — TCP client to the brain daemon
  clock.py                        — ISO timestamp helpers
  db.py                           — DB path resolution + read-only connect
  queries/                        — one module per data source (no raw SQL elsewhere)
  static/                         — index.html, style.css, app.js
"""
