"""Dashboard query modules — one module per data source.

The dashboard reads-only. Each module here owns the SQL for one slice of
the brain (recalls, encoding activity, S2 runs, traces, errors, system
status, sessions). Server-side route handlers in `dashboard.server` import
from this package and never touch SQL directly.

Eventually these should call into the DAL (`servers.dal.*`), but the
dashboard is a read-only observer and the DAL doesn't yet expose every
shape the dashboard needs — so for now this package owns those queries.
"""
