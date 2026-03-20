"""
brain -- Migration runner.

Discovers numbered migration files in this directory, compares against the
schema_migrations table, and applies any that haven't run yet — each inside
its own transaction so a failure rolls back only that migration.

Public API:
    from servers.migrations.runner import run_migrations
    run_migrations(conn)                  # brain.db
    run_migrations(logs_conn, prefix='logs')  # brain_logs.db
"""

import importlib
import os
import pkgutil
import sqlite3
import traceback
from datetime import datetime, timezone
from typing import List, Optional, Tuple


# ── Schema for the tracking table ──

_CREATE_TRACKING = """
CREATE TABLE IF NOT EXISTS schema_migrations (
    version     INTEGER PRIMARY KEY,
    name        TEXT NOT NULL,
    applied_at  TEXT NOT NULL,
    checksum    TEXT,
    execution_ms INTEGER
)
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _ensure_tracking_table(conn: sqlite3.Connection) -> None:
    """Create the schema_migrations table if it doesn't exist."""
    conn.execute(_CREATE_TRACKING)
    conn.commit()


def _applied_versions(conn: sqlite3.Connection) -> set:
    """Return the set of migration version numbers already applied."""
    try:
        rows = conn.execute(
            "SELECT version FROM schema_migrations"
        ).fetchall()
        return {r[0] for r in rows}
    except sqlite3.OperationalError:
        # Table doesn't exist yet
        return set()


def _discover_migrations(prefix: str = "") -> List[Tuple[int, str, str]]:
    """Find all migration modules in this package.

    Each module must be named NNN_description.py (e.g. 001_initial_schema.py).
    Returns a sorted list of (version, name, module_path) tuples.

    Args:
        prefix: Optional prefix filter. If set, only discover migrations whose
                name starts with "{prefix}_" after the number (e.g. prefix='logs'
                matches 002_logs_initial.py but not 002_add_vocabulary.py).
    """
    migrations_dir = os.path.dirname(os.path.abspath(__file__))
    results = []

    for item in sorted(os.listdir(migrations_dir)):
        if not item.endswith('.py'):
            continue
        if item.startswith('_') or item == 'runner.py':
            continue

        # Parse NNN_description.py
        parts = item[:-3].split('_', 1)
        if len(parts) < 2:
            continue
        try:
            version = int(parts[0])
        except ValueError:
            continue

        name = parts[1]

        # Apply prefix filter if given
        if prefix and not name.startswith(prefix + '_'):
            continue

        module_name = item[:-3]  # e.g. "001_initial_schema"
        results.append((version, module_name, item))

    results.sort(key=lambda x: x[0])
    return results


def _load_migration(module_name: str):
    """Import a migration module from this package."""
    return importlib.import_module(
        ".%s" % module_name, package="servers.migrations"
    )


def run_migrations(
    conn: sqlite3.Connection,
    prefix: str = "",
    db_path: Optional[str] = None,
) -> List[str]:
    """Apply all pending migrations to the given connection.

    Each migration module must define:
        def up(conn: sqlite3.Connection) -> None:
            '''Apply this migration.'''

    Optionally:
        def down(conn: sqlite3.Connection) -> None:
            '''Reverse this migration (for future rollback support).'''

        description: str  # Human-readable summary

    Each migration runs inside a SAVEPOINT so that a failure rolls back
    only that migration without affecting earlier ones.

    Args:
        conn:    SQLite connection to migrate.
        prefix:  Only run migrations whose name starts with this prefix
                 (useful for separating brain.db vs logs migrations).
        db_path: Optional path to DB file, passed to migration up() if it
                 accepts a db_path keyword argument.

    Returns:
        List of migration names that were applied.
    """
    _ensure_tracking_table(conn)
    applied = _applied_versions(conn)
    available = _discover_migrations(prefix=prefix)

    pending = [(v, name, path) for v, name, path in available if v not in applied]
    if not pending:
        return []

    applied_names = []

    for version, module_name, filename in pending:
        mod = _load_migration(module_name)
        up_fn = getattr(mod, 'up', None)
        if up_fn is None:
            print("[migrate] SKIP %s — no up() function" % module_name)
            continue

        desc = getattr(mod, 'description', module_name)
        print("[migrate] Applying %03d: %s ..." % (version, desc))

        t0 = _timer()

        # Use a savepoint so failure rolls back only this migration
        savepoint = "migration_%03d" % version
        try:
            conn.execute("SAVEPOINT %s" % savepoint)

            # Call up(); pass db_path if the function accepts it
            import inspect
            sig = inspect.signature(up_fn)
            if 'db_path' in sig.parameters:
                up_fn(conn, db_path=db_path)
            else:
                up_fn(conn)

            # Record success
            conn.execute(
                "INSERT INTO schema_migrations (version, name, applied_at, execution_ms) "
                "VALUES (?, ?, ?, ?)",
                (version, module_name, _now(), _elapsed_ms(t0))
            )
            conn.execute("RELEASE %s" % savepoint)
            conn.commit()

            elapsed = _elapsed_ms(t0)
            print("[migrate] Applied  %03d: %s (%d ms)" % (version, desc, elapsed))
            applied_names.append(module_name)

        except Exception as e:
            # Roll back this migration only
            try:
                conn.execute("ROLLBACK TO %s" % savepoint)
                conn.execute("RELEASE %s" % savepoint)
            except Exception:
                pass
            print("[migrate] FAILED  %03d: %s" % (version, desc))
            print("[migrate]   Error: %s" % e)
            traceback.print_exc()
            # Stop applying further migrations after a failure
            break

    return applied_names


def get_status(conn: sqlite3.Connection, prefix: str = "") -> dict:
    """Return migration status for diagnostics.

    Returns:
        {
            'applied': [(version, name, applied_at), ...],
            'pending': [(version, name), ...],
            'current_version': int or None,
        }
    """
    _ensure_tracking_table(conn)
    applied_set = _applied_versions(conn)
    available = _discover_migrations(prefix=prefix)

    applied_rows = conn.execute(
        "SELECT version, name, applied_at FROM schema_migrations ORDER BY version"
    ).fetchall()

    pending = [(v, name) for v, name, _ in available if v not in applied_set]

    current = max(applied_set) if applied_set else None

    return {
        'applied': [(r[0], r[1], r[2]) for r in applied_rows],
        'pending': pending,
        'current_version': current,
    }


# ── Timing helpers (stdlib only) ──

def _timer():
    import time
    return time.monotonic()


def _elapsed_ms(t0) -> int:
    import time
    return int((time.monotonic() - t0) * 1000)
