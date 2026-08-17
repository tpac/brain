"""Re-assemble an S1E prompt from traces and compare it to what the daemon
actually sent — the view-policy A/B's cheap first gate.

Sibling to encoder_prompt_composition.py (which counts a captured payload's
bytes); this one REBUILDS the payload through the real builders and scores the
result. Two modes, one run each:

  control (default)  — flag off. Its output must reproduce the captured
      000-prompt.md: same turn count, same catalog id set, size within a few
      percent. This is the harness's OWN integrity check — the stored capture
      is the oracle (id:53d46908: assert the internal stages, or the harness
      silently lies). If the control arm can't reproduce what the daemon
      produced, nothing downstream is trustworthy.
  --view-policy      — flag on. Same inputs, policy render. The report is the
      per-section byte delta; --out writes the prompt for eyeballing.

No brain writes anywhere: reads run against an IsolatedBrain COPY of
production (never a second writer on the live DB), muster is off (scouts are
LLM calls; their blocks are stripped from the capture before comparing), and
the builders are the production functions — no forked assembly.

THE AS-OF BOUND — the capture is the run's INPUT, written before the run's
own writes, so the trace gather is time-bounded (gather's `older_than`) to
the run's encoding_prompt O trace: the run's encode delta, its anchor_touched
flushes and its catalog widening are excluded, the way the run itself never
saw them. Two reads CANNOT be bounded from outside and drift accordingly —
<continuity> (the run's journal/arc notes land right after the capture; the
gate skips this section) and recall_episodes (the message window; quiescent
sessions only). Compare against a session's NEWEST capture, freshly: node
bodies drift as they're revised, and a node the run itself absorbed renders
missing (the report marks those explained — the body is gone NOW). Idle-tail
runs (the 1h-quiet final encode) are the cleanest oracles.

Usage:
    ./dev python3 eval/encoder_prompt_reassembly.py <payloads/.../000-prompt.md>
        [--view-policy] [--out FILE]
"""
import argparse
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'tests'))


SECTIONS = ('continuity', 'failed_encodes', 'node_catalog', 'scout_legend',
            'timeline')


def parse_chain(capture_path):
    """s1e-<sess8>-<stop> from .../payloads/<date>/<chain>/000-prompt.md."""
    chain = os.path.basename(os.path.dirname(os.path.abspath(capture_path)))
    m = re.fullmatch(r's1e-([0-9a-f]{8})-(\d+)(?:\.\d+)?', chain)
    if not m:
        raise SystemExit('not an s1e capture path: %s' % capture_path)
    return chain, m.group(1), int(m.group(2))


def resolve_run(brain, chain):
    """(session_id, created_at) from the run's encoding_prompt O trace — the
    full session key and the as-of instant for the time-bounded gather."""
    events = (brain.query_traces(ref_type='encoding_prompt', hours=None,
                                 limit=1000) or {}).get('events') or []
    for e in events:
        if e.get('chain_id') == chain:
            return e.get('session_id'), e.get('created_at')
    raise SystemExit('no encoding_prompt trace for chain %s' % chain)


def assemble(brain, session_id, view_policy, older_than=None):
    """The production assembly path, muster off. Returns the full prompt
    string exactly as record_payload composes it (preamble + body).

    Time-bounded twin of encode._build_catalog: the same production pieces
    (gather → session_node_ids → build_node_catalog), plus the `older_than`
    as-of bound production never needs — the oracle capture reflects the
    streams as they stood when the run gathered them, before its own writes.
    """
    from servers.scales.s1.encode import (_gather_messages, _build_user_content,
                                          _conversation_now_safe)
    from servers.scales.s1.trace_links import gather, session_node_ids
    from servers.scales.s1.encode_contract import build_node_catalog
    messages = _gather_messages(brain, session_id)
    if not messages:
        raise SystemExit('no messages for session %s' % session_id[:8])
    judge_outputs = [m.get('judge_output') for m in messages
                     if m.get('role') == 'user']
    streams = gather(brain, session_id, older_than=older_than)
    extra_ids = session_node_ids(streams['encode'], streams['touched'])
    now = _conversation_now_safe(brain, session_id, messages) if view_policy else None
    catalog_text, catalog_ids = build_node_catalog(
        judge_outputs, brain, extra_ids=extra_ids,
        scope=brain.session_scope(session_id), view_policy=view_policy, now=now)
    preamble, body, _t, _i = _build_user_content(
        brain, messages, 0, session_id, lived_sequence=True,
        precomputed=(catalog_text, catalog_ids, streams),
        scout_outputs=None, view_policy=view_policy, view_now=now)
    return preamble + "\n\n" + body


def strip_scout_blocks(text):
    """Muster is off in the harness — remove the capture's scout renders so
    size comparison is apples-to-apples."""
    text = re.sub(r'<scout_legend>.*?</scout_legend>\n?', '', text,
                  flags=re.DOTALL)
    return re.sub(r'\s*<scout_notes>.*?</scout_notes>\n?', '\n', text,
                  flags=re.DOTALL)


def section(text, name):
    # attrs allowed: the view policy stamps <timeline now="…">
    m = re.search(r'<%s(?:\s[^>]*)?>(.*?)</%s>' % (name, name), text, re.DOTALL)
    return m.group(1) if m else ''


def catalog_ids_of(text):
    """Entry ids: an entry HEADER is `[tags…] [type] "title" (id:…` at column 0
    (render_rich_node's first line, optionally provenance/[aged]-tagged).
    Edge/correction refs are indented; multi-line node CONTENT can put an
    id-shaped substring at column 0 but not in this exact header shape."""
    return set(re.findall(
        r'^(?:\[[^\]]+\]\s+)*\[\w+\] ".*?" \(id:([0-9a-f]{6,8})',
        section(text, 'node_catalog'), re.MULTILINE))


def turn_count(text):
    return len(re.findall(r'<turn n="\d+"', section(text, 'timeline')))


def report(captured, rebuilt, brain=None):
    """Compare the two prompts. The gate covers what the as-of bound makes
    reproducible: turn count, the catalog id set (missing ids whose node is
    gone NOW are explained — revised/absorbed since the capture), and the
    <node_catalog>/<timeline> sizes (±5%). <continuity> is reported ungated
    (the run's own journal/arc writes land after the capture, unboundable)."""
    lines = []
    ok = True

    ct, rt = turn_count(captured), turn_count(rebuilt)
    lines.append('turns:        captured %d | rebuilt %d %s'
                 % (ct, rt, 'OK' if ct == rt else 'MISMATCH'))
    ok &= ct == rt

    cid, rid = catalog_ids_of(captured), catalog_ids_of(rebuilt)
    missing, extra = cid - rid, rid - cid
    if missing and brain is not None:
        alive = set(brain.get_node(list(missing)) or {})
        explained = missing - alive
        if explained:
            lines.append('catalog ids:  %d missing but node gone now '
                         '(archived/absorbed since capture): %s'
                         % (len(explained), ','.join(sorted(explained)[:8])))
            missing -= explained
    if not missing and not extra:
        lines.append('catalog ids:  %d, identical OK' % len(cid))
    else:
        ok = False
        lines.append('catalog ids:  captured %d | rebuilt %d MISMATCH '
                     '(missing: %s | extra: %s)'
                     % (len(cid), len(rid),
                        ','.join(sorted(missing)[:8]) or '-',
                        ','.join(sorted(extra)[:8]) or '-'))

    for name in SECTIONS:
        c, r = len(section(captured, name)), len(section(rebuilt, name))
        if not c and not r:
            continue
        delta = (r - c) * 100.0 / c if c else float('inf')
        gated = name in ('node_catalog', 'timeline')
        verdict = ''
        if gated:
            verdict = ' OK' if (c and abs(delta) <= 5.0) else ' OUTSIDE ±5%'
            ok &= verdict == ' OK'
        lines.append('  <%s>: %d -> %d chars (%+.1f%%)%s'
                     % (name, c, r, delta, verdict))

    c, r = len(captured), len(rebuilt)
    lines.append('total:        %d -> %d chars (%+.1f%%)'
                 % (c, r, (r - c) * 100.0 / c))
    return ok, '\n'.join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('capture', help='payloads/<date>/<chain>/000-prompt.md')
    ap.add_argument('--view-policy', action='store_true',
                    help='assemble with BRAIN_S1E_VIEW_POLICY semantics ON')
    ap.add_argument('--out', help='write the reassembled prompt here')
    args = ap.parse_args()

    chain, _short, _stop = parse_chain(args.capture)
    with open(args.capture) as f:
        captured = strip_scout_blocks(f.read())

    from isolated_brain import IsolatedBrain
    with IsolatedBrain(cleanup=True, load_env=False) as env:
        brain = env.brain
        session_id, run_ts = resolve_run(brain, chain)
        print('chain %s -> session %s, as-of %s'
              % (chain, session_id[:8], run_ts))
        rebuilt = assemble(brain, session_id, view_policy=args.view_policy,
                           older_than=run_ts)
        arm = 'view-policy' if args.view_policy else 'control'
        ok, text = report(captured, rebuilt, brain=brain)

    if args.out:
        with open(args.out, 'w') as f:
            f.write(rebuilt)
        print('wrote %s (%d chars)' % (args.out, len(rebuilt)))
    print('\n[%s arm] vs %s\n%s' % (arm, args.capture, text))
    if not args.view_policy:
        print('\nintegrity: %s' % ('PASS' if ok else 'FAIL — control arm did '
                                   'not reproduce the capture; see limits in '
                                   'the module docstring'))
        sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
