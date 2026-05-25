"""Single stderr logger for the dashboard.

The dashboard intentionally degrades gracefully when something goes wrong —
half the panels would crash the page if any one query raised, so each query
catches its own exceptions and returns an empty result. The historic pattern
was `except Exception: pass / return []` which made every failure invisible.

This helper preserves the degradation but makes the failure VISIBLE in the
dashboard's stdout/stderr (which preview_console_logs surfaces, and which a
shell-launched dashboard prints to the terminal). Per the brain's
loud-by-default principle: silent failures are the most dangerous bug class.

Convention:
  warn('component', 'human-readable thing that failed', exc=e)
prints:
  [dashboard.component] thing that failed: <ExceptionType>: <message>

`component` is the module name (e.g. 'queries.recalls'), not a class.
"""

import sys


def warn(component: str, message: str, exc: BaseException = None) -> None:
    """Emit a single line to stderr. Never raises."""
    try:
        if exc is None:
            line = '[dashboard.%s] %s' % (component, message)
        else:
            line = '[dashboard.%s] %s: %s: %s' % (
                component, message, type(exc).__name__, exc,
            )
        print(line, file=sys.stderr, flush=True)
    except Exception:
        # Logging itself failing is the corner of corners. Swallow — we'd
        # rather lose the log line than crash the request handler.
        pass
