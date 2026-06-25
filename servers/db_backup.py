"""Automatic rolling database backups + grandfather-father-son retention.

Backend-agnostic. The consistent-snapshot mechanism (online backup API +
gzip) lives in the active db_backend (`backup_snapshot`); this module owns
the backend-independent policy: destination naming, listing/parsing the
backup set, and which snapshots to keep (GFS retention).

These are the AUTOMATIC rolling backups, distinct from the named
pre-destructive-operation `.bak` files a human makes by hand (the
"backup brain.db before destructive ops" rule). This module only ever
touches files it created — the `{basename}.{timestamp}.gz` it writes into
the backup dir. Hand-made `.bak*` files are never listed, kept, or pruned
by this code.

Retention is GFS:
- `keep_daily`   — newest snapshot of each of the most recent N days
- `keep_weekly`  — newest snapshot of each of the most recent N ISO weeks
- `keep_monthly` — newest snapshot of each of the most recent N months
A snapshot survives if any tier selects it (union). At daily cadence this
holds ~keep_daily + keep_weekly + keep_monthly files at steady state.
"""

from __future__ import annotations

import os
import re
from datetime import datetime, timezone
from typing import Dict, List, Optional, Set, Tuple

# Filename scheme: `{db_basename}.{YYYYMMDDTHHMMSSZ}.gz`
# e.g. `brain.db.20260625T151500Z.gz`. The timestamp is UTC, sortable,
# and round-trips back to a datetime for GFS bucketing.
_TS_FMT = '%Y%m%dT%H%M%SZ'
_TS_RE = re.compile(r'\.(\d{8}T\d{6}Z)\.gz$')

_DEFAULT_KEEP = {'daily': 7, 'weekly': 4, 'monthly': 3}


def _dest_path(db_path: str, backup_dir: str, ts: datetime) -> str:
    base = os.path.basename(db_path)
    return os.path.join(backup_dir, '%s.%s.gz' % (base, ts.strftime(_TS_FMT)))


def list_backups(db_path: str, backup_dir: str) -> List[Tuple[datetime, str]]:
    """All auto-backups for this DB, newest first: [(timestamp, path)].

    Matches only `{basename}.{ts}.gz` — hand-made `.bak*` files don't have
    a parseable timestamp segment and are skipped, so they're invisible to
    retention."""
    base = os.path.basename(db_path)
    prefix = base + '.'
    out: List[Tuple[datetime, str]] = []
    try:
        names = os.listdir(backup_dir)
    except OSError:
        return out
    for name in names:
        if not name.startswith(prefix):
            continue
        m = _TS_RE.search(name)
        if not m:
            continue
        try:
            ts = datetime.strptime(m.group(1), _TS_FMT).replace(tzinfo=timezone.utc)
        except ValueError:
            continue
        out.append((ts, os.path.join(backup_dir, name)))
    out.sort(key=lambda t: t[0], reverse=True)
    return out


def select_retained(timestamps: List[datetime],
                    keep_daily: int, keep_weekly: int,
                    keep_monthly: int) -> Set[datetime]:
    """GFS retained set. For each tier, keep the newest snapshot of each
    of the most recent N periods. A snapshot is retained if any tier
    selects it."""
    retained: Set[datetime] = set()
    ordered = sorted(timestamps, reverse=True)   # sort once, reuse per tier

    def pick(key_fn, keep_n: int) -> None:
        if keep_n <= 0:
            return
        newest_per_period: Dict = {}
        for ts in ordered:
            k = key_fn(ts)
            if k not in newest_per_period:   # newest first → first wins
                newest_per_period[k] = ts
        for k in sorted(newest_per_period, reverse=True)[:keep_n]:
            retained.add(newest_per_period[k])

    pick(lambda ts: ts.date(), keep_daily)
    pick(lambda ts: ts.isocalendar()[:2], keep_weekly)   # (iso_year, iso_week)
    pick(lambda ts: (ts.year, ts.month), keep_monthly)
    return retained


def prune(db_path: str, backup_dir: str,
          keep_daily: int, keep_weekly: int, keep_monthly: int) -> Dict:
    """Delete auto-backups not selected by the GFS retained set."""
    backups = list_backups(db_path, backup_dir)
    if not backups:
        return {'kept': 0, 'deleted': 0}
    retained = select_retained([ts for ts, _ in backups],
                               keep_daily, keep_weekly, keep_monthly)
    # Never delete the newest snapshot, whatever the policy — guards against
    # an all-zero keep config silently wiping the just-created backup.
    retained.add(backups[0][0])   # backups is newest-first
    deleted = 0
    for ts, path in backups:
        if ts not in retained:
            try:
                os.remove(path)
                deleted += 1
            except OSError:
                pass
    return {'kept': len(retained), 'deleted': deleted}


def seconds_since_last_backup(db_path: str, backup_dir: str,
                              now: Optional[datetime] = None) -> float:
    """Age of the newest auto-backup, in seconds. `inf` if none exist.

    Lets the scheduler seed its clock from the filesystem so a daemon
    restart doesn't re-snapshot a DB it backed up minutes ago — no extra
    persisted state needed."""
    backups = list_backups(db_path, backup_dir)
    if not backups:
        return float('inf')
    if now is None:
        now = datetime.now(timezone.utc)
    return max(0.0, (now - backups[0][0]).total_seconds())


def backup_database(db_path: str, backup_dir: str, *,
                    keep_daily: int = _DEFAULT_KEEP['daily'],
                    keep_weekly: int = _DEFAULT_KEEP['weekly'],
                    keep_monthly: int = _DEFAULT_KEEP['monthly'],
                    now: Optional[datetime] = None) -> Dict:
    """Snapshot `db_path` into `backup_dir` (compressed), then prune to the
    GFS retained set. Returns snapshot diagnostics + kept/deleted counts."""
    if now is None:
        now = datetime.now(timezone.utc)
    os.makedirs(backup_dir, exist_ok=True)
    dest = _dest_path(db_path, backup_dir, now)

    from . import db_backends
    result = dict(db_backends.current.backup_snapshot(db_path, dest))

    result['dest'] = os.path.basename(dest)
    result.update(prune(db_path, backup_dir, keep_daily, keep_weekly, keep_monthly))
    return result
