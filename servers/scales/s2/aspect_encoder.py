"""S2 Aspect Encoder — classifies candidates into the required aspects via Sonnet.

Reads the taxonomy via brain.aspects (the menu + current member lists),
builds a prompt with the menu + candidate strings + example records, calls
Sonnet once, parses the JSON response, validates each classification, and
writes through the registry's door (brain.aspects.add_members). Auto-merge —
no operator review gate in v1.

Also writes an audit record (aspects_proposed.json) so the per-cycle delta
is inspectable.

The unit does NOT mutate brain state — taxonomy in / taxonomy out.
"""

from servers.aspect_store import atomic_json_write
from servers.trace_contract import build_delta_metadata

from .base import IntegrationUnit
from .aspect_contract import ASPECT, aspects_proposed_path


# Which aspects the classifier may route to, and which categories each takes,
# are per-aspect FACTS in aspects_v1.json (`routable`, `accepts`) — read via
# the registry, never hardcoded here. `accepts` is design intent, not
# discoverable from current member lists alone (an aspect with empty
# node_types might be edge-only by design or just not yet populated).
# survivor_lineage declares routable=false (system-generated absorbed_into
# edges, never classified). `wisdom` IS routable so it GROWS: it's a
# multi-membership view (its node types also live in lesson_insight /
# identity_bearing), and the encoder multi-homes generative types into it,
# guided by the aspect's `meaning`. The decoder proposes only unclassified
# strings, so existing types stay as seeded while NEW generative types
# auto-join wisdom alongside their primary aspect.


def _routing_map(aspects):
    """{aspect_name: accepts-set} for the routable aspects — the classifier's
    closed list, derived from the per-aspect facts. `aspects` is the
    name→Aspect dict from brain.aspects.all()."""
    return {name: set(a.accepts) for name, a in aspects.items() if a.routable}


class AspectEncoder(IntegrationUnit):
    NAME = 'aspect_integration'
    SCALE = 's2'
    ENCODING_SOURCE = 's2:aspect_integration'

    O_SOURCES = ['aspect_proposals']
    K_SOURCES = ['llm_aspect_classifier', 'aspects_v1.json', 'journal_notes']

    # Residue flows to journal_note trace rows via the journal binding on
    # _call_llm (decorate → harvest), read back via continuity(). For this
    # unit the notes are the ONLY durable record of a classification's
    # reasoning — aspects_proposed.json is overwritten every cycle, and a
    # classification is permanent (never re-examined).

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

        aspects = self.brain.aspects.all()   # {name: Aspect} — registry view
        user_content = self.journal.continuity() + self._format_prompt(
            aspects, proposals)
        result, telemetry = self._call_llm('s2_aspects', user_content,
                                           journal=True)

        if result is None:
            err = 'LLM call failed'
            self.brain._log_error(self.NAME, Exception(err), 'classifying %d proposals' % len(proposals))
            self._trace_failed_delta(err, telemetry, len(proposals))
            return {'classified': 0, 'rejected': 0, 'errors': [err], 'journal': ''}

        # Sonnet's JSON output isn't fully predictable in shape — sometimes
        # returns the wrapped object {classifications: [...]}, sometimes a
        # bare array [...]. Accept both. runner.extract_json picks the first
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
            self._trace_failed_delta(err, telemetry, len(proposals))
            return {'classified': 0, 'rejected': 0, 'errors': [err], 'journal': ''}

        # Validate, then write through the registry's single door — it merges
        # (idempotent append), writes atomically, and re-derives its own maps.
        accepted, rejected = self._validate_classifications(classifications, proposals)
        self.brain.aspects.add_members(accepted, source=self.ENCODING_SOURCE)
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

    def _trace_failed_delta(self, err, telemetry, n_proposals):
        """Delta trace for a failed run — every exit path stamps the unit's
        delta ref_type (the _last_run_timestamp contract), and with the
        journal binding a failed call may already have written residue rows
        on this chain: without a delta they'd be orphaned (unreachable from
        every run view) and the burned tokens unrecorded."""
        self.trace('delta', 'aspect_classified',
                   'run failed: %s (%d proposals, %dms, %d→%d tok)' % (
                       err, n_proposals,
                       telemetry.get('elapsed_ms', 0),
                       telemetry.get('input_tokens', 0),
                       telemetry.get('output_tokens', 0)),
                   metadata=build_delta_metadata(
                       actions=0, write_actions=0, rounds=1,
                       inputs_processed=n_proposals,
                       outcomes={'classified': 0, 'rejected': 0},
                       journal_entry='',
                       errors=[err],
                       elapsed_ms=telemetry.get('elapsed_ms', 0),
                       input_tokens=telemetry.get('input_tokens', 0),
                       output_tokens=telemetry.get('output_tokens', 0),
                       cache_read_tokens=telemetry.get('cache_read_tokens', 0),
                       cache_creation_tokens=telemetry.get('cache_creation_tokens', 0),
                   ))

    # ─── prompt construction ─────────────────────────────────────────

    def _format_prompt(self, aspects, proposals):
        """Build the user message: ASPECT MENU + CANDIDATES.

        `aspects` is the {name: Aspect} view from brain.aspects.all().
        """
        lines = []
        routing = _routing_map(aspects)

        lines.append('═' * 70)
        lines.append('ASPECT MENU — %d aspects, closed list. Route every candidate to one of these.'
                     % len(routing))
        lines.append('═' * 70)
        lines.append('')

        # Render routable aspects in a stable order: node-only first (Frame),
        # then the semantic rest, then catch-alls (the prompt-invisible
        # structural aspects) last. Helps the encoder build a mental map.
        # Groups derive from the per-aspect facts; within a group, taxonomy
        # (JSON) order — reproduces the hand-curated order this replaced.
        def _menu_group(name):
            if routing[name] == {'node_types'}:
                return 0
            return 1 if aspects[name].prompt_visible else 2

        order = sorted(routing, key=_menu_group)
        for name in order:
            aspect = aspects[name]
            members_n = list(aspect.node_types)
            members_e = list(aspect.edge_relations)
            lines.append('── %s ──' % name)
            lines.append('  meaning: %s' % aspect.meaning)
            lines.append('  accepts: %s' % ', '.join(sorted(routing[name])))
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

    def _validate_classifications(self, classifications, proposals):
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

        routing = _routing_map(self.brain.aspects.all())
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

            # Validate every listed aspect against the closed list (the
            # routable aspects from the registry). Reject on unknown or
            # non-routable aspect names (encoder hallucination).
            invalid = [a for a in aspects_list if a not in routing]
            if invalid:
                rejected.append({'classification': c,
                                 'reason': 'aspects not in closed list: %s' % invalid})
                continue
            # Filter to aspects that accept this category. Edge-only aspects
            # for a node_type candidate get filtered out; if any survive,
            # accept the entry with the filtered list (first survivor = new
            # primary). If nothing survives, reject. Logged so we can see
            # how often the encoder picks the wrong category.
            valid_for_cat = [a for a in aspects_list if category in routing[a]]
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

            # Noise + semantic is a LEGAL answer, kept as returned.
            # Noise vetoes at read time, so the noise member is
            # the load-bearing half: stripping it — as this boundary used to,
            # on the reasoning that a semantic claim refutes noise — deleted
            # exactly the half that does the filtering and silently un-noised
            # machinery the classifier had correctly flagged. Keeping both
            # preserves what the string MEANS while the veto governs what it
            # DOES. Recorded, not refused: dual routing is legal now, so it
            # belongs in the debug log rather than the errors table, and it is
            # the only signal telling us whether the classifier uses this at all.
            if 'noise' in aspects_list and len(aspects_list) > 1:
                self.brain._log_warning(
                    self.NAME,
                    'noise + semantic aspect — kept both, noise vetoes on read',
                    'value="%s" category=%s aspects=%s' % (
                        value, category, aspects_list))

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

    # ─── audit trail ─────────────────────────────────────────────────

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
            atomic_json_write(aspects_proposed_path(), record)
        except Exception as e:
            self.brain._log_error(
                self.NAME, e,
                'failed to write audit trail — classifications already merged into aspects_v1.json')
