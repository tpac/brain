"""Backward-compat shim — S1 encoding agent moved to scales/s1/encode.py.

All logic now lives in servers.scales.s1.encode. This file re-exports
run_encoding for callers that import from servers.encoding_agent.
"""

from .scales.s1.encode import run_encoding  # noqa: F401
