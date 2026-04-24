"""Shared scout contract — single source of truth for scout I/O.

All scouts (Quote, Temporal, Facts, Synthesis) inherit the same:
- Orientation preamble (what the operator/assistant/catalog/surfaced/conversation means)
- Output envelope: {scout, category_statement, candidates[], scanned}

## Cache layout

The runner splits the call into two cacheable blocks:

    system (1h TTL, per-scout):
        SCOUT_SYSTEM_PROMPT + interaction.template
        → byte-identical across every run of the same scout; read on
          every subsequent cycle

    user content (5m TTL, shared across scouts in one cycle):
        orientation + session context + current date + node catalog +
        surfaced nodes + conversation window
        → byte-identical across the 4 scouts in a single encoding cycle;
          first scout writes, the other 3 read

The shared-prefix builder (build_shared_prefix) returns the user-content
blocks with cache_control on the last one. The runner (scouts.base)
composes the system prompt from the interaction template.

## Output validation

- ENVELOPE_REQUIRED: what every scout must return at the top level
- CANDIDATE_REQUIRED: fields every candidate must carry
- SCOUT_FIELD_SPECS: per-scout required + optional extras
- FIELD_LIMITS: char caps on string fields (soft-truncated with warning)

Validation philosophy (per loud-by-default):
- Structural failures (missing required, wrong types): return {ok: False, errors}
  so the base runner logs to brain_errors and emits an empty-findings output.
- Soft violations (char-limit overflow): truncate + append to warnings list;
  do not fail the scout.
- Invalid JSON at runner boundary: handled by base.py, not here.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple


# ─── Scout registry ────────────────────────────────────────────────────────

SCOUT_NAMES: Tuple[str, ...] = ("quote", "temporal", "facts", "synthesis")

SCOUT_INTERACTION_PREFIX = "s1_scout_"  # e.g. s1_scout_quote


def interaction_name(scout: str) -> str:
    if scout not in SCOUT_NAMES:
        raise ValueError(f"Unknown scout {scout!r}. Valid: {SCOUT_NAMES}")
    return f"{SCOUT_INTERACTION_PREFIX}{scout}"


# ─── Envelope + candidate schemas ──────────────────────────────────────────

ENVELOPE_REQUIRED = ("scout", "category_statement", "candidates", "scanned")

CANDIDATE_REQUIRED = ("handle", "evidence_quote", "evidence_turns", "why_candidate")

SCANNED_REQUIRED = ("turns",)  # other keys (considered/passed_threshold/etc) optional

# Per-scout additional required fields (beyond CANDIDATE_REQUIRED).
# If a scout needs a field for downstream use (e.g. S1S composition), it
# belongs here. Optional/diagnostic fields are NOT listed.
SCOUT_FIELD_SPECS: Dict[str, Dict[str, Tuple[str, ...]]] = {
    "quote": {
        "required": ("speaker",),
        "optional": ("grounds_candidates", "echo_count", "catalog_match"),
    },
    "temporal": {
        "required": ("source_phrase",),
        "optional": ("resolution", "event_description", "existing_anchor_id",
                     "catalog_tension", "precision", "relational_marker"),
    },
    "facts": {
        "required": ("entity", "feature", "value"),
        "optional": ("unit", "catalog_match"),
    },
    "synthesis": {
        "required": ("turn_evidence",),
        "optional": ("abstraction_candidate", "catalog_tension"),
    },
}


# ─── Field char limits (soft) ──────────────────────────────────────────────

FIELD_LIMITS = {
    "handle":                120,
    "evidence_quote":        300,
    "why_candidate":         150,
    "source_phrase":          50,
    "resolution":            100,
    "event_description":     100,
    "entity":                 80,
    "feature":                50,
    "value":                 100,
    "unit":                   30,
    "abstraction_candidate": 200,
    # category_statement is envelope-level
    "category_statement":    300,
}


# ─── Shared orientation preamble ───────────────────────────────────────────
# Identical bytes across every scout in a single encoding cycle → shared cache.

SCOUT_SYSTEM_PROMPT = (
    "You are a scout in a persistent AI brain's S1 Scribe encoding pipeline. "
    "Return findings as a single JSON object. Do not write nodes directly — "
    "your output informs the writer's composition. Quality over quantity; "
    "an empty candidates list is valid when nothing qualifies."
)

# The scout orientation preamble is composed from the shared-orientation
# module so S1S and scouts read identical section labels. Only the framing
# wrapper (scout voice, parallel-run disclaimer) is scout-specific.
from ..orientation import SCOUT_ORIENTATION_PREAMBLE  # noqa: E402,F401  (re-export)


# ─── Shared prefix assembly ────────────────────────────────────────────────

def build_shared_prefix(
    session_context: str,
    current_date: str,
    catalog_rendered: str,
    surfaced_by_turn_rendered: str,
    conversation_rendered: str,
) -> List[Dict[str, Any]]:
    """Build the shared content blocks for a scout call.

    Returns a list of content blocks suitable for passing as the full
    user-role message.content. The LAST block carries cache_control:
    ephemeral — everything up to and including it is cached at 5m TTL.

    All four scouts pass byte-identical user content in a single encoding
    cycle → single shared cache entry across the parallel fan-out. The
    first scout to hit it writes; the others read.

    The per-scout task is NOT appended to this user content — it lives in
    the scout's system prompt (1h TTL, per-scout cache) composed by the
    runner. See scouts/base.py for the assembly.

    Args are pre-rendered strings because rendering decisions (which
    HAIKU_FORMAT, how much to truncate, turn numbering) live at the muster
    layer, not inside the contract.
    """
    sections = [
        ("About this input", SCOUT_ORIENTATION_PREAMBLE.strip()),
        ("Session context",  (session_context or "").strip() or "(empty)"),
        ("Current date",     current_date.strip()),
        ("Node catalog",     catalog_rendered.strip() or "(empty)"),
        ("Surfaced nodes per turn", surfaced_by_turn_rendered.strip() or "(none)"),
        ("Conversation window", conversation_rendered.strip() or "(empty)"),
    ]

    blocks: List[Dict[str, Any]] = []
    for label, body in sections:
        # Preamble block carries only its own body (already contains ##
        # headings and explanation).  Other sections get "## {label}\n{body}".
        if label == "About this input":
            text = body
        else:
            text = f"## {label}\n{body}"
        blocks.append({"type": "text", "text": text})

    # Cache breakpoint on the final shared block.
    blocks[-1]["cache_control"] = {"type": "ephemeral"}
    return blocks


# ─── Output validation ─────────────────────────────────────────────────────

class ScoutOutputError(Exception):
    """Raised by validate_scout_output on structural (unrecoverable) violations."""


def _truncate(text: Any, limit: int) -> Tuple[str, bool]:
    """Coerce to str, truncate to limit. Returns (trimmed_text, was_truncated)."""
    s = "" if text is None else str(text)
    if len(s) <= limit:
        return s, False
    return s[:limit], True


def validate_scout_output(
    output: Any,
    scout: str,
) -> Tuple[bool, Dict[str, Any], List[str], List[str]]:
    """Validate + soft-truncate a scout's JSON output.

    Returns:
        ok: True if the envelope is structurally sound (truncations are OK).
            False if envelope is unusable (e.g. not a dict, missing required
            top-level fields, wrong scout name).
        normalized: a cleaned copy of the output with truncations applied
            and candidates list coerced to the expected shape. When ok=False,
            this is a stub empty-findings object that can still be handed to
            S1S (fail loud but don't break the cycle).
        errors: list of structural error messages (non-empty when ok=False).
        warnings: list of soft-violations (truncations, unknown fields,
            candidates dropped due to missing required fields).

    All four values are always returned — callers decide how to log them.
    Per loud-by-default: both errors and warnings should be logged to
    brain_errors by the runner. This function never raises on input data;
    it raises only on programmer errors (wrong scout name).
    """
    if scout not in SCOUT_NAMES:
        raise ScoutOutputError(f"Unknown scout {scout!r}")

    spec = SCOUT_FIELD_SPECS[scout]
    errors: List[str] = []
    warnings: List[str] = []

    stub = {
        "scout": scout,
        "category_statement": "",
        "candidates": [],
        "scanned": {"turns": 0, "considered": 0, "passed_threshold": 0},
    }

    if not isinstance(output, dict):
        errors.append(f"output must be a dict, got {type(output).__name__}")
        return False, stub, errors, warnings

    # Envelope required fields
    for key in ENVELOPE_REQUIRED:
        if key not in output:
            errors.append(f"missing envelope field: {key}")

    # scout identity check (cheap guardrail — wrong scout means wrong caller)
    out_scout = output.get("scout")
    if out_scout != scout:
        errors.append(f"envelope.scout is {out_scout!r}, expected {scout!r}")

    if errors:
        return False, stub, errors, warnings

    normalized: Dict[str, Any] = {"scout": scout}

    # category_statement
    cat = output.get("category_statement")
    cat_trimmed, trunc = _truncate(cat, FIELD_LIMITS["category_statement"])
    if trunc:
        warnings.append(
            f"category_statement truncated ({len(str(cat))} -> {FIELD_LIMITS['category_statement']})")
    if not cat_trimmed:
        errors.append("category_statement is empty")
        return False, stub, errors, warnings
    normalized["category_statement"] = cat_trimmed

    # candidates list
    cands_in = output.get("candidates")
    if not isinstance(cands_in, list):
        errors.append(f"candidates must be a list, got {type(cands_in).__name__}")
        return False, stub, errors, warnings

    normalized_candidates: List[Dict[str, Any]] = []
    for idx, c in enumerate(cands_in):
        if not isinstance(c, dict):
            warnings.append(f"candidates[{idx}] dropped: not a dict")
            continue
        cand_errors: List[str] = []
        # Required fields
        for field in CANDIDATE_REQUIRED:
            if field not in c or c[field] in (None, ""):
                cand_errors.append(f"missing required field '{field}'")
        # Scout-specific required
        for field in spec["required"]:
            if field not in c or c[field] in (None, ""):
                cand_errors.append(f"missing scout-required field '{field}'")

        if cand_errors:
            warnings.append(
                f"candidates[{idx}] dropped: " + "; ".join(cand_errors))
            continue

        # Soft char-limit truncation on string fields we know about
        cleaned: Dict[str, Any] = {}
        for k, v in c.items():
            if k in FIELD_LIMITS and isinstance(v, (str, int, float)):
                trimmed, trunc = _truncate(v, FIELD_LIMITS[k])
                if trunc:
                    warnings.append(
                        f"candidates[{idx}].{k} truncated to {FIELD_LIMITS[k]} chars")
                cleaned[k] = trimmed
            else:
                cleaned[k] = v

        # evidence_turns: must be a list (coerce strings/None)
        et = cleaned.get("evidence_turns")
        if isinstance(et, str):
            cleaned["evidence_turns"] = [et]
            warnings.append(f"candidates[{idx}].evidence_turns coerced from str to list")
        elif not isinstance(et, list):
            cleaned["evidence_turns"] = []
            warnings.append(f"candidates[{idx}].evidence_turns reset (was {type(et).__name__})")

        normalized_candidates.append(cleaned)

    normalized["candidates"] = normalized_candidates

    # scanned envelope — coerce to dict with at least 'turns'
    scanned = output.get("scanned")
    if not isinstance(scanned, dict):
        warnings.append(f"scanned coerced to dict (was {type(scanned).__name__})")
        scanned = {"turns": 0, "considered": 0, "passed_threshold": 0}
    else:
        if "turns" not in scanned:
            scanned["turns"] = 0
            warnings.append("scanned.turns missing, defaulted to 0")
    normalized["scanned"] = scanned

    return True, normalized, errors, warnings


# ─── Formatting for S1S prompt ─────────────────────────────────────────────

def format_scout_report_for_s1s(
    scout_outputs: Dict[str, Dict[str, Any]],
) -> str:
    """Render combined scout findings into a single block for S1S's prompt.

    Shape (per scout):
        ### {scout}
        {category_statement}

        Candidates ({N}):
        - {handle}  [turns: t3,t7]
          evidence: "{evidence_quote}"
          why: {why_candidate}
          {scout-specific fields on extra lines}

        (scanned: turns=N, considered=K, passed_threshold=M)

    Empty-findings scouts still render with "(no candidates)" — so S1S can
    tell a scout looked at X and found nothing, vs didn't look at all.

    Scouts that errored (runner returned stub with empty candidates +
    errors metadata) also render visibly so S1S doesn't assume coverage.
    """
    sections: List[str] = []
    for scout in SCOUT_NAMES:
        out = scout_outputs.get(scout)
        if out is None:
            sections.append(f"### {scout}\n(scout did not run)\n")
            continue
        sections.append(_format_one_scout(scout, out))
    return "\n".join(sections)


def _format_one_scout(scout: str, out: Dict[str, Any]) -> str:
    lines: List[str] = [f"### {scout}"]
    cat = out.get("category_statement") or "(no category statement)"
    lines.append(cat)
    cands = out.get("candidates") or []
    lines.append("")
    if not cands:
        lines.append("Candidates (0): (nothing qualified)")
    else:
        lines.append(f"Candidates ({len(cands)}):")
        spec = SCOUT_FIELD_SPECS.get(scout, {"required": (), "optional": ()})
        extra_fields = [f for f in list(spec["required"]) + list(spec["optional"])
                        if f not in CANDIDATE_REQUIRED]
        for c in cands:
            handle = c.get("handle") or "(no handle)"
            turns = c.get("evidence_turns") or []
            turns_str = ",".join(str(t) for t in turns) or "-"
            lines.append(f"- {handle}  [turns: {turns_str}]")
            quote = c.get("evidence_quote") or ""
            lines.append(f"  evidence: \"{quote}\"")
            why = c.get("why_candidate") or ""
            lines.append(f"  why: {why}")
            for f in extra_fields:
                if f in c and c[f] not in (None, "", [], {}):
                    lines.append(f"  {f}: {c[f]}")
    scanned = out.get("scanned") or {}
    if scanned:
        parts = [f"{k}={v}" for k, v in scanned.items()]
        lines.append(f"(scanned: {', '.join(parts)})")
    lines.append("")
    return "\n".join(lines)


__all__ = [
    "SCOUT_NAMES",
    "SCOUT_INTERACTION_PREFIX",
    "interaction_name",
    "ENVELOPE_REQUIRED",
    "CANDIDATE_REQUIRED",
    "SCANNED_REQUIRED",
    "SCOUT_FIELD_SPECS",
    "FIELD_LIMITS",
    "SCOUT_SYSTEM_PROMPT",
    "SCOUT_ORIENTATION_PREAMBLE",
    "build_shared_prefix",
    "validate_scout_output",
    "format_scout_report_for_s1s",
    "ScoutOutputError",
]
