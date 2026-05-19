"""Database backend plug-points.

The brain currently uses SQLite. The db_maintenance scheduler talks to
whatever backend implements the BackendOps protocol (declared in
servers/db_maintenance.py). To swap SQLite for another store later,
implement a new module in this package and re-export it as `current`.
"""

from . import sqlite as current  # noqa: F401 — re-exported

__all__ = ['current']
