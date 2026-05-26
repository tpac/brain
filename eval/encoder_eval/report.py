"""report — render per-axis × per-version aggregate from per_cell.jsonl.

Standalone: can be run after a halt-then-resume cycle, or against a
completed run.
"""
import json
from pathlib import Path
from typing import Any, Dict, List, Optional


def _load_cells(per_cell_path: Path) -> List[Dict[str, Any]]:
    cells = []
    with open(per_cell_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                cells.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return cells


def _axes(cells: List[Dict[str, Any]]) -> List[str]:
    return sorted({c.get('axis') or 'unknown' for c in cells})


def _versions(cells: List[Dict[str, Any]]) -> List[int]:
    return sorted({c['version'] for c in cells})


def _correct(cell: Dict[str, Any]) -> bool:
    return bool((cell.get('longmem_result') or {}).get('correct', False))


def _probe_score(cell: Dict[str, Any], probe_name: str) -> Optional[float]:
    p = (cell.get('probes') or {}).get(probe_name, {})
    if p.get('skipped') or 'error' in p:
        return None
    s = p.get('score')
    return float(s) if isinstance(s, (int, float)) else None


def _mean(values: List[Optional[float]]) -> Optional[float]:
    vals = [v for v in values if v is not None]
    if not vals:
        return None
    return sum(vals) / len(vals)


def _fmt_pct(v: Optional[float], fmt: str = '{:.1%}') -> str:
    return fmt.format(v) if v is not None else '—'


def _fmt_delta(a: Optional[float], b: Optional[float],
                fmt: str = '{:+.1f}pp') -> str:
    if a is None or b is None:
        return '—'
    return fmt.format((b - a) * 100)


def render_report(per_cell_path: Path, out_path: Path,
                   baseline_version: int = 19) -> str:
    """Render markdown report. Returns the markdown content."""
    cells = _load_cells(per_cell_path)
    if not cells:
        out_path.write_text("# Encoder Eval Report\n\n(no cells)\n")
        return "(no cells)"

    versions = _versions(cells)
    axes = _axes(cells)
    new_version = max(versions)  # convention: highest is the candidate

    # Pass rate matrix
    by_va = {}  # (version, axis) → list of correct booleans
    for c in cells:
        by_va.setdefault((c['version'], c.get('axis') or 'unknown'), []) \
            .append(_correct(c))

    def passrate(v, a):
        vals = by_va.get((v, a), [])
        return (sum(vals) / len(vals)) if vals else None

    lines = []
    lines.append(f"# Encoder Eval Report\n")
    lines.append(f"**Versions**: {', '.join('v' + str(v) for v in versions)}")
    lines.append(f"**Total cells**: {len(cells)}")
    lines.append(f"**Axes**: {', '.join(axes)}")
    lines.append("")

    # ─── Pass rate ────────────────────────────────────────────────
    lines.append("## Answer correctness (pass rate)")
    lines.append("")
    header = "| Axis | " + " | ".join(f"v{v}" for v in versions) + " | "
    header += " | ".join(f"Δ vs v{baseline_version}" for v in versions
                          if v != baseline_version)
    header += " |"
    lines.append(header)
    lines.append("|---|" + "---|" * (len(versions) +
                                      len([v for v in versions if v != baseline_version])))
    for a in axes:
        row = [f" {a} "]
        for v in versions:
            row.append(f" {_fmt_pct(passrate(v, a))} ")
        for v in versions:
            if v != baseline_version:
                row.append(f" {_fmt_delta(passrate(baseline_version, a), passrate(v, a))} ")
        lines.append("|" + "|".join(row) + "|")
    # Overall row
    row = [" **OVERALL** "]
    for v in versions:
        all_v = [_correct(c) for c in cells if c['version'] == v]
        rate = sum(all_v) / len(all_v) if all_v else None
        row.append(f" **{_fmt_pct(rate)}** ")
    for v in versions:
        if v != baseline_version:
            base_all = [_correct(c) for c in cells if c['version'] == baseline_version]
            new_all = [_correct(c) for c in cells if c['version'] == v]
            br = sum(base_all) / len(base_all) if base_all else None
            nr = sum(new_all) / len(new_all) if new_all else None
            row.append(f" {_fmt_delta(br, nr)} ")
    lines.append("|" + "|".join(row) + "|")
    lines.append("")

    # ─── Per-probe table ─────────────────────────────────────────
    PROBE_NAMES = [
        'brain_presence', 'specificity_preservation', 'source_refs_coverage',
        'atomization_shape', 'edge_structure', 'voice_balance',
    ]
    for probe in PROBE_NAMES:
        lines.append(f"## Probe: {probe}")
        lines.append("")
        lines.append(f"| Axis | " + " | ".join(f"v{v}" for v in versions) +
                      " |")
        lines.append("|---|" + "---|" * len(versions))
        for a in axes:
            row = [f" {a} "]
            for v in versions:
                axis_cells = [c for c in cells if c['version'] == v and (
                    c.get('axis') or 'unknown') == a]
                m = _mean([_probe_score(c, probe) for c in axis_cells])
                row.append(f" {m:.2f} " if m is not None else " — ")
            lines.append("|" + "|".join(row) + "|")
        lines.append("")

    # ─── Source_refs coverage detail ─────────────────────────────
    lines.append("## Source_refs coverage detail (v22 vs v21 vs v19)")
    lines.append("")
    lines.append("| Version | Nodes encoded (total) | With refs | Coverage | "
                  "Hex-format failures | Sparsity violations (>5) |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for v in versions:
        v_cells = [c for c in cells if c['version'] == v]
        nodes = sum((c.get('probes') or {}).get('source_refs_coverage', {})
                     .get('nodes_encoded', 0) for c in v_cells)
        with_refs = sum((c.get('probes') or {}).get('source_refs_coverage', {})
                         .get('nodes_with_refs', 0) for c in v_cells)
        hex_fail = sum((c.get('probes') or {}).get('source_refs_coverage', {})
                        .get('hex_format_failures', 0) for c in v_cells)
        sp_viol = sum((c.get('probes') or {}).get('source_refs_coverage', {})
                       .get('sparsity_violations_gt5', 0) for c in v_cells)
        cov_pct = (with_refs / nodes * 100) if nodes else 0
        lines.append(f"| v{v} | {nodes} | {with_refs} | {cov_pct:.1f}% | "
                      f"{hex_fail} | {sp_viol} |")
    lines.append("")

    # ─── Edge structure detail ────────────────────────────────────
    lines.append("## Edge structure detail")
    lines.append("")
    lines.append("| Version | Total edges | Typed connect_to | co_anchored "
                  "| related_to | related_to % |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for v in versions:
        v_cells = [c for c in cells if c['version'] == v]
        edges_total = sum((c.get('probes') or {}).get('edge_structure', {})
                           .get('total_edges', 0) for c in v_cells)
        typed = sum((c.get('probes') or {}).get('edge_structure', {})
                     .get('typed_connect_pairs', 0) for c in v_cells)
        coanchored = sum((c.get('probes') or {}).get('edge_structure', {})
                          .get('co_anchored_pairs', 0) for c in v_cells)
        rel_to = sum((c.get('probes') or {}).get('edge_structure', {})
                      .get('related_to_count', 0) for c in v_cells)
        rel_pct = (rel_to / edges_total * 100) if edges_total else 0
        lines.append(f"| v{v} | {edges_total} | {typed} | {coanchored} | "
                      f"{rel_to} | {rel_pct:.1f}% |")
    lines.append("")

    # ─── Failure attachments — worst cells per metric ─────────────
    lines.append("## Failure attachments")
    lines.append("")
    lines.append("Worst v22 cells per probe (lowest score with non-trivial measurement):")
    lines.append("")
    for probe in PROBE_NAMES:
        v22_with_score = [
            (c, _probe_score(c, probe))
            for c in cells if c['version'] == new_version
        ]
        v22_with_score = [(c, s) for c, s in v22_with_score if s is not None]
        if not v22_with_score:
            continue
        v22_with_score.sort(key=lambda cs: cs[1])
        lines.append(f"### {probe}")
        for c, s in v22_with_score[:3]:
            lines.append(f"- `{c['item_id']}` (axis={c.get('axis')}): score={s:.2f}")
            evidence = (c.get('probes') or {}).get(probe, {})
            for k in ('best_match', 'dropped_examples', 'relation_distribution',
                       'type_distribution', 'identity_bearing_symmetry'):
                if k in evidence and evidence[k] not in (None, [], {}):
                    val = evidence[k]
                    if isinstance(val, dict):
                        val = ', '.join(f'{k2}={v2}' for k2, v2 in
                                         list(val.items())[:5])
                    elif isinstance(val, list):
                        val = ', '.join(str(x) for x in val[:5])
                    lines.append(f"  - {k}: {val}")
        lines.append("")

    text = '\n'.join(lines)
    out_path.write_text(text)
    return text
