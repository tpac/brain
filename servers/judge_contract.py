"""Backward-compat shim — judge contract moved to scales/s1/recall_contract.py."""

from .scales.s1.recall_contract import (  # noqa: F401
    _relative_time,
    CANDIDATES_FILE,
    JUDGE,
    NEIGHBOR_D1_FIELDS,
    NEIGHBOR_D2_FIELDS,
    NEIGHBOR_D3_FIELDS,
    NEIGHBOR_TRUNCATION,
    PRECISION,
    enrich_candidate_metadata,
    correction_enrich,
    format_candidate_for_judge,
    _dedup_candidates,
    build_judge_prompt,
    format_judge_output,
)
