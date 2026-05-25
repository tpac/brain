"""HTTP response envelope — Prometheus-style {status, data, error?, warnings?}.

This file defines the SHAPE every /api/* route should return. The frontend
fetch wrapper (lib/api.js) unwraps `data` and exposes `warnings` separately,
so panels can surface "served stale, daemon was locked" without escalating
to a hard error and breaking the panel.

Source: Prometheus HTTP API conventions
(https://prometheus.io/docs/prometheus/latest/querying/api/). Chosen for two
reasons: it's a four-key envelope (no overengineering), and the `warnings`
channel is the underrated win — most dashboard failures aren't hard errors,
they're degraded reads ("DB locked, served the cached value") that the
frontend currently has nowhere to put.

Envelope shape:
    success: {"status": "success", "data": <payload>, "warnings": [...]?}
    error:   {"status": "error",   "error": "<msg>", "errorType": "<type>"?}

HTTP code carries the gross status (2xx success, 4xx client, 5xx server);
envelope carries the detail. A panel can return 200 with a warnings array
when it served partial data.

Migration policy (2026-05-25): handlers DO NOT have to adopt the envelope
yet — the frontend currently expects bare shapes. Adoption happens
incrementally as part of each Phase 2 feature. New routes added in Phase 1+
should use the helpers from the start. Don't half-migrate one route at a
time without updating its caller in the same commit.
"""

from typing import Any, List, Optional


def envelope_ok(data: Any, warnings: Optional[List[str]] = None) -> dict:
    """Wrap a successful response. `warnings` is for non-fatal degradations
    (stale data, missing optional source, daemon was slow) — the panel
    should still render `data`, possibly with a small advisory."""
    out = {"status": "success", "data": data}
    if warnings:
        out["warnings"] = list(warnings)
    return out


def envelope_error(message: str, error_type: Optional[str] = None) -> dict:
    """Wrap a failure. `error_type` is a short machine-readable tag
    (e.g. 'daemon_unavailable', 'bad_node_id', 'sql_error'). The frontend
    can branch on it for retry / UI logic; humans read `message`."""
    out = {"status": "error", "error": message}
    if error_type:
        out["errorType"] = error_type
    return out
