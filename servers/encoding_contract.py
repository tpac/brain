"""Backward-compat shim — encoding contract moved to scales/s1/encode_contract.py."""

from .scales.s1.encode_contract import (  # noqa: F401
    ENCODING_AGENT,
    format_node_for_encoder,
    build_encoder_node_catalog,
    correction_enrich,
)
