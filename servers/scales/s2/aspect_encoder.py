"""S2 Aspect Encoder — classifies candidates into the 14 aspects via Sonnet.

Reads aspects_v1.json (the menu + current member lists), builds a prompt
with the menu + candidate strings + example records, calls Sonnet once,
parses the JSON response, validates each classification, and merges
into aspects_v1.json. Auto-merge — no operator review gate in v1.

Writes both the merged result (aspects_v1.json) and an audit record
(aspects_proposed.json) so the per-cycle delta is inspectable.

The unit does NOT mutate brain state — pure JSON in / JSON out.
"""

import json
import os
import tempfile

from servers.trace_contract import build_delta_metadata

from .base import IntegrationUnit
from .aspect_contract import ASPECT, aspects_json_path, aspects_proposed_path


# Which categories each aspect accepts. Derived from the design — not
# discoverable from current member lists alone (an aspect with empty
# node_types might be edge-only by design or just not yet populated).
# 15 of the 16 required aspects are LLM-routable here. Only survivor_lineage is
# excluded (system-generated absorbed_into edges, never classified). `wisdom` IS
# routable so it GROWS: it's a multi-membership view (its node types also live in
# lesson_insight/identity_bearing), and the encoder multi-homes generative types
# into it, guided by the aspect's `meaning`. The decoder proposes only
# unclassified strings, so existing types stay as seeded while NEW generative
# types auto-join wisdom alongside their primary aspect.
ASPECT_ACCEPTS = {
    'identity_bearing':       {'node_types'},
    'episodic_anchor':        {'node_types'},
    'active_thread':          {'node_types'},
    'lesson_insight':         {'node_types'},
    'wisdom':                 {'node_types'},
    'correction_improvement': {'node_types', 'edge_relations'},
    'extension_refinement':   {'edge_relations'},
    'explanation_causation':  {'edge_relations'},
    'dependency_flow':        {'edge_relations'},
    'contradiction_conflict': {'edge_relations'},
    'validation_evidence':    {'edge_relations'},
    'hierarchical_structure': {'edge_relations'},
    'temporal_sequence':      {'edge_relations'},
    'generic_relation':       {'edge_relations'},
    'noise':                  {'node_types', 'edge_relations'},
}


class AspectEncoder(IntegrationUnit):
    NAME = 'aspect_integration'
    SCALE = 's2'
    ENCODING_SOURCE = 's2:aspect_integration'

    O_SOURCES = ['aspect_proposals']
    K_SOURCES = ['llm_aspect_classifier', 'aspects_v1.json']

    def __init__(self, brain, dispatch_fn=None, config=None):
        super().__init__(brain, dispatch_fn)
        self.config = config or ASPECT

    def run(self, proposals):
        """Classify proposals into existing aspects.

        Args:
            proposals: list from decoder. Each: {category, value, count, examples}

        Returns: {classified, rejected, errors, journal}
        """
        if not proposals:
            self.trace('delta', 'aspect_classified', 'No proposals to process')
            return {'classified': 0, 'rejected': 0, 'errors': [], 'journal': ''}

        aspects = self._load_aspects()
        user_content = self._format_prompt(aspects, proposals)
        result, telemetry = self._call_llm('s2_aspects', user_content)

        if result is None:
            err = 'LLM call failed'
            self.brain._log_error(self.NAME, Exception(err), 'classifying %d proposals' % len(proposals))
            return {'classified': 0, 'rejected': 0, 'errors': [err], 'journal': ''}

        # Sonnet's JSON output isn't fully predictable in shape — sometimes
        # returns the wrapped object {classifications: [...]}, sometimes a
        # bare array [...]. Accept both. base._extract_json picks the first
        # JSON structure it sees, so a bare array comes through as a list.
        if isinstance(result, list):
            classifications = result
        elif isinstance(result, dict):
            classifications = result.get('classifications', [])
        else:
            classifications = []
        if not classifications:
            err = 'response had no classifications'
            self.brain._log_error(self.NAME, Exception(err),
                                  'response type: %s, keys: %s' % (
                                      type(result).__name__,
                                      list(result.keys()) if isinstance(result, dict) else 'n/a'))
            return {'classified': 0, 'rejected': 0, 'errors': [err], 'journal': ''}

        # Validate + merge
        accepted, rejected = self._validate_classifications(classifications, proposals, aspects)
        self._merge_into_aspects(aspects, accepted)
        self._write_aspects(aspects)
        self._write_audit_trail(accepted, rejected)

        # Per-aspect counts for trace. Multi-membership: each classification
        # touches potentially multiple aspects — primary first, then secondaries.
        per_aspect_primary = {}
        per_aspect_any = {}
        multi_count = 0
        for c in accepted:
            asp_list = c['aspects']
            primary = asp_list[0]
            per_aspect_primary[primary] = per_aspect_primary.get(primary, 0) + 1
            for a in asp_list:
                per_aspect_any[a] = per_aspect_any.get(a, 0) + 1
            if len(asp_list) > 1:
                multi_count += 1
        per_aspect = per_aspect_primary  # for backwards-compat downstream

        journal = '%d classified (%d multi-aspect), %d rejected. By primary aspect: %s' % (
            len(accepted), multi_count, len(rejected),
            ', '.join('%s=%d' % kv for kv in sorted(per_aspect_primary.items())))

        # Structured Δ for Aspect. It mutates aspects_v1.json, not the graph,
        # so created/revised/archived don't apply — the real change record is
        # WHICH string routed to WHICH aspect(s). Carried in `classifications`
        # (extras) so the per-classification decision survives, not just counts.
        classifications_made = [
            {'category': c['category'], 'value': c['value'], 'aspects': c['aspects']}
            for c in accepted
        ]

        self.trace('delta', 'aspect_classified',
                   '%d classified, %d rejected, %dms, %d→%d tok' % (
                       len(accepted), len(rejected),
                       telemetry.get('elapsed_ms', 0),
                       telemetry.get('input_tokens', 0),
                       telemetry.get('output_tokens', 0)),
                   metadata=build_delta_metadata(
                       actions=len(classifications),
                       write_actions=len(accepted),
                       rounds=1,
                       inputs_processed=len(proposals),
                       outcomes={
                           'classified': len(accepted),
                           'rejected': len(rejected),
                           **{'aspect_' + k: v for k, v in per_aspect.items()},
                       },
                       journal_entry=journal,
                       errors=[r['reason'] for r in rejected[:5]],
                       classifications=classifications_made,
                       elapsed_ms=telemetry.get('elapsed_ms', 0),
                       input_tokens=telemetry.get('input_tokens', 0),
                       output_tokens=telemetry.get('output_tokens', 0),
                       cache_read_tokens=telemetry.get('cache_read_tokens', 0),
                       cache_creation_tokens=telemetry.get('cache_creation_tokens', 0),
                   ))

        return {
            'classified': len(accepted),
            'rejected': len(rejected),
            'rejected_details': rejected,
            'errors': [],
            'journal': journal,
            'per_aspect': per_aspect,
        }

    # ─── prompt construction ─────────────────────────────────────────

    def _format_prompt(self, aspects, proposals):
        """Build the user message: ASPECT MENU + CANDIDATES."""
        lines = []

        lines.append('═' * 70)
        lines.append('ASPECT MENU — %d aspects, closed list. Route every candidate to one of these.'
                     % len(ASPECT_ACCEPTS))
        lines.append('═' * 70)
        lines.append('')

        # Render aspects in a stable order: required-node-only first (Frame),
        # then both, then required-edge-only, then catch-alls. Helps the
        # encoder build a mental map.
        order = [
            'identity_bearing', 'episodic_anchor', 'active_thread', 'lesson_insight',
            'wisdom',
            'correction_improvement',
            'extension_refinement', 'explanation_causation', 'dependency_flow',
            'contradiction_conflict', 'validation_evidence',
            'hierarchical_structure', 'temporal_sequence',
            'generic_relation', 'noise',
        ]
        for name in order:
            aspect = aspects.get(name, {})
            accepts = sorted(ASPECT_ACCEPTS.get(name, set()))
            members_n = aspect.get('node_types', [])
            members_e = aspect.get('edge_relations', [])
            lines.append('── %s ──' % name)
            lines.append('  meaning: %s' % aspect.get('meaning', ''))
            lines.append('  accepts: %s' % ', '.join(accepts))
            if members_n:
                lines.append('  current node_types: %s' % ', '.join(sorted(members_n)))
            if members_e:
                lines.append('  current edge_relations: %s' % ', '.join(sorted(members_e)))
            lines.append('')

        lines.append('═' * 70)
        lines.append('CANDIDATES — %d unclassified strings to route' % len(proposals))
        lines.append('═' * 70)
        lines.append('')

        for i, p in enumerate(proposals, 1):
            lines.append('── #%d ──' % i)
            lines.append('  category: %s' % p['category'])
            lines.append('  value: "%s"' % p['value'])
            lines.append('  count: %d' % p['count'])
            lines.append('  examples:')
            for ex in p.get('examples', []):
                tier = ex.get('tier', '')
                tier_tag = '  [%s]' % tier if tier else ''
                if p['category'] == 'node_types':
                    lines.append('    - [type: %s] "%s"%s' % (
                        ex.get('type', p['value']), ex.get('title', ''), tier_tag))
                    if ex.get('content_snippet'):
                        lines.append('      content:   %s' % ex['content_snippet'])
                    if ex.get('situation'):
                        lines.append('      situation: %s' % ex['situation'])
                else:
                    lines.append('    - [%s] "%s" --%s--> [%s] "%s"%s' % (
                        ex.get('src_type', ''), ex.get('src_title', ''),
                        p['value'],
                        ex.get('tgt_type', ''), ex.get('tgt_title', ''),
                        tier_tag))
                    if ex.get('description'):
                        lines.append('      description: %s' % ex['description'])
                    if ex.get('src_content_snippet'):
                        lines.append('      src content: %s' % ex['src_content_snippet'])
                    if ex.get('tgt_content_snippet'):
                        lines.append('      tgt content: %s' % ex['tgt_content_snippet'])
            lines.append('')

        lines.append('═' * 70)
        lines.append('Return JSON: {"classifications": [...]} with one entry per candidate, in order.')

        return '\n'.join(lines)

    # ─── validation + merge ──────────────────────────────────────────

    def _validate_classifications(self, classifications, proposals, aspects):
        """Drop classifications that target invalid aspects or wrong categories.

        Multi-membership: each classification carries `aspects` (a list, primary
        first). Validation drops the whole entry if any listed aspect is invalid
        or doesn't accept the category. Returns (accepted, rejected).

        Backward-compatible with the older single-aspect shape (`aspect: name`)
        — coerces it to a single-element list before validation.
        """
        candidates_by_value = {}
        for p in proposals:
            candidates_by_value[(p['category'], p['value'])] = p

        accepted = []
        rejected = []
        seen = set()

        for c in classifications:
            if not isinstance(c, dict):
                rejected.append({'classification': c, 'reason': 'not a dict'})
                continue

            value = c.get('value', '')
            category = c.get('category', '')
            # Accept both shapes: single `aspect: name` (legacy) or `aspects: [list]`.
            aspects_list = c.get('aspects')
            if aspects_list is None and c.get('aspect'):
                aspects_list = [c['aspect']]

            if not (value and category and aspects_list):
                rejected.append({'classification': c, 'reason': 'missing field (value/category/aspects)'})
                continue
            if not isinstance(aspects_list, list) or not aspects_list:
                rejected.append({'classification': c, 'reason': 'aspects must be a non-empty list'})
                continue

            if (category, value) not in candidates_by_value:
                rejected.append({'classification': c, 'reason': 'value+category not in candidate list'})
                continue

            # Validate every listed aspect against the closed list. Reject on
            # unknown aspect names (encoder hallucination).
            invalid = [a for a in aspects_list if a not in ASPECT_ACCEPTS]
            if invalid:
                rejected.append({'classification': c,
                                 'reason': 'aspects not in closed list: %s' % invalid})
                continue
            # Filter to aspects that accept this category. Edge-only aspects
            # for a node_type candidate get filtered out; if any survive,
            # accept the entry with the filtered list (first survivor = new
            # primary). If nothing survives, reject. Logged so we can see
            # how often the encoder picks the wrong category.
            valid_for_cat = [a for a in aspects_list if category in ASPECT_ACCEPTS[a]]
            dropped = [a for a in aspects_list if a not in valid_for_cat]
            if not valid_for_cat:
                rejected.append({'classification': c,
                                 'reason': 'no listed aspect accepts category "%s": %s' % (category, aspects_list)})
                continue
            if dropped:
                self.brain._log_error(
                    self.NAME,
                    Exception('aspect/category mismatch — filtered'),
                    'value="%s" category=%s dropped=%s kept=%s' % (
                        value, category, dropped, valid_for_cat))
                aspects_list = valid_for_cat

            # Noise-exclusivity boundary. `noise` means "no semantic claim"
            # (graph mechanics + bookkeeping artifacts). If the encoder also
            # routed this string to a real aspect, that semantic claim refutes
            # noise — keep the meaning, strip noise. Preserves the invariant
            # noise ∩ {any other aspect} = ∅, which is what lets exclusion
            # filters trust "not in noise" == "is real knowledge". Mirrors the
            # category-mismatch filter above: drop the bad member, keep the
            # survivors, log loud.
            if 'noise' in aspects_list and len(aspects_list) > 1:
                kept = [a for a in aspects_list if a != 'noise']
                self.brain._log_error(
                    self.NAME,
                    Exception('noise + semantic aspect — stripped noise'),
                    'value="%s" category=%s aspects=%s kept=%s' % (
                        value, category, aspects_list, kept))
                aspects_list = kept

            if (category, value) in seen:
                rejected.append({'classification': c, 'reason': 'duplicate classification of (%s, %s)' % (category, value)})
                continue
            seen.add((category, value))

            # Normalize to canonical shape on the way out — `aspects` list,
            # primary first, no separate `aspect` key.
            normalized = {
                'category': category,
                'value': value,
                'aspects': aspects_list,
                'rationale': c.get('rationale', ''),
            }
            accepted.append(normalized)

        return accepted, rejected

    def _merge_into_aspects(self, aspects, accepted):
        """Add each value to EACH of its classified aspects' member lists.

        Multi-membership: a value can appear in multiple aspects' lists.
        Idempotent — duplicates filtered.
        """
        for c in accepted:
            field = c['category']  # 'node_types' or 'edge_relations'
            for aspect_name in c['aspects']:
                aspect_def = aspects.setdefault(aspect_name, {})
                members = aspect_def.setdefault(field, [])
                if c['value'] not in members:
                    members.append(c['value'])

    # ─── file I/O ────────────────────────────────────────────────────

    def _load_aspects(self):
        json_path = aspects_json_path()
        try:
            with open(json_path, 'r') as f:
                return json.load(f)
        except (OSError, json.JSONDecodeError) as e:
            self.brain._log_error(
                self.NAME, e,
                'failed to read %s — encoder cannot proceed' % json_path)
            raise

    def _write_aspects(self, aspects):
        """Atomic write: temp file + rename."""
        json_path = aspects_json_path()
        d = os.path.dirname(json_path)
        fd, tmp = tempfile.mkstemp(prefix='aspects_v1_', suffix='.json.tmp', dir=d)
        try:
            with os.fdopen(fd, 'w') as f:
                json.dump(aspects, f, indent=2, sort_keys=False)
                f.write('\n')
            os.replace(tmp, json_path)
        except Exception:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise

    def _write_audit_trail(self, accepted, rejected):
        """Write per-cycle audit JSON (overwrites prior cycle's audit).

        Useful for debugging which cycle classified what and for the
        operator review path (when auto-merge gets disabled).
        """
        from datetime import datetime, timezone
        record = {
            'cycle_at': datetime.now(timezone.utc).isoformat(),  # clock-ok — S2 cycle telemetry
            'classifications_accepted': accepted,
            'classifications_rejected': rejected,
        }
        try:
            proposed_path = aspects_proposed_path()
            d = os.path.dirname(proposed_path)
            fd, tmp = tempfile.mkstemp(prefix='aspects_proposed_', suffix='.json.tmp', dir=d)
            with os.fdopen(fd, 'w') as f:
                json.dump(record, f, indent=2)
                f.write('\n')
            os.replace(tmp, proposed_path)
        except Exception as e:
            self.brain._log_error(
                self.NAME, e,
                'failed to write audit trail — classifications already merged into aspects_v1.json')
