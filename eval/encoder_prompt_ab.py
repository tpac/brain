"""Encoder view-policy A/B — every arm assembled from ONE frozen brain copy.

Sibling to encoder_prompt_reassembly.py, which rebuilds a SINGLE arm and scores
it against the stored capture (the control arm's integrity gate). This one
assembles every arm together, because a pair built by two invocations of that
tool is not a pair: each invocation copies a MOVING database, and the timeline
read (recall_episodes) is the one input no `older_than` can bound. Measured on
one unchanged capture five minutes apart: <timeline> -2.7% then +5.5%, integrity
PASS then FAIL. Arms compared across copies compare different worlds.

The arms:

  A  control   flag off — the pre-policy render.
  B  rounds    flag on, aging cutoff = the CATALOG_FULL_ROUNDS-th newest encode
               run's stop. What is merged on main today.
  C  window    flag on, aging cutoff = the first turn the timeline renders, so
               the full-depth catalog and the chat window share ONE knob. B lets
               them drift apart — a 10-turn window against a 42-round catalog
               ages by a yardstick the encoder cannot see; under C, widening the
               chat window widens the full catalog with it.

Sizes are the default pass. `--behavior` runs the questions bytes cannot answer:
does the encoder re-encode turns whose <actions> were trimmed, does it expand an
aged entry before writing or duplicate it, do source_refs stay sane. Each arm's
prompt goes to the encoder model with WRITES INTERCEPTED (recorded, never
executed — no brain mutation) and READS SERVED FOR REAL against the same frozen
copy. Reads are not stubbed on purpose: an aged entry's whole contract is
"get_nodes returns the rest", and a stubbed expansion would make
revise-vs-duplicate unmeasurable. Scoring stops at the first round that writes (connect_ab's rule) —
that round carries the answer, and nothing downstream of a fabricated write id
is trustworthy anyway.

Usage:
    ./dev python3 eval/encoder_prompt_ab.py <payloads/.../000-prompt.md> [more…]
        [--arms A,B,C] [--out-dir DIR] [--behavior]
"""
import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'tests'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from encoder_prompt_reassembly import (  # noqa: E402  (path set above)
    catalog_ids_of, parse_chain, report, resolve_run, section,
    strip_scout_blocks, turn_count)

# (label, view_policy, window_aligned, aged_content_chars)
#   -1 = the policy's own config — body whole since arm D shipped as the
#   default; B/C pin the retired 400-char cap explicitly so they keep
#   reproducing what was measured.
ARMS = {
    'A': ('control', False, False, -1),
    'B': ('rounds', True, False, 400),
    'C': ('window', True, True, 400),
    'D': ('keep-body', True, True, -1),
}


def gather_inputs(brain, session_id, older_than=None):
    """The one read every arm shares: messages, trace streams, window head.

    Hoisted out of the arms deliberately — "same-state" means the arms differ
    by RENDER and nothing else, so they must not each re-read a database that
    moves under them.
    """
    from servers.scales.s1.encode import _gather_messages, window_first_turn
    from servers.scales.s1.trace_links import gather
    messages = _gather_messages(brain, session_id)
    if not messages:
        raise SystemExit('no messages for session %s' % session_id[:8])
    return {
        'messages': messages,
        'streams': gather(brain, session_id, older_than=older_than),
        'head': window_first_turn(brain, session_id, messages),
    }


def assemble_arm(brain, session_id, inputs, view_policy, window_aligned,
                 aged_content_chars=-1):
    """One arm through the PRODUCTION builders — no forked assembly.

    Mirrors encoder_prompt_reassembly.assemble, plus the window-aligned cutoff:
    `window_first_turn` comes from the same _lived_turns window the timeline
    renders, so arm C's catalog is dated against the conversation the encoder
    is actually shown.
    """
    from servers.scales.s1.encode import (_build_user_content,
                                          _conversation_now_safe)
    from servers.scales.s1.trace_links import session_node_ids
    from servers.scales.s1.encode_contract import build_node_catalog
    messages, streams = inputs['messages'], inputs['streams']
    judge_outputs = [m.get('judge_output') for m in messages
                     if m.get('role') == 'user']
    extra_ids = session_node_ids(streams['encode'], streams['touched'])
    now = _conversation_now_safe(brain, session_id, messages) if view_policy else None
    head = inputs['head'] if window_aligned else None
    catalog_text, catalog_ids = build_node_catalog(
        judge_outputs, brain, extra_ids=extra_ids,
        scope=brain.session_scope(session_id), view_policy=view_policy,
        now=now, window_first_turn=head,
        aged_content_chars=aged_content_chars)
    preamble, body, _t, _i = _build_user_content(
        brain, messages, 0, session_id, lived_sequence=True,
        precomputed=(catalog_text, catalog_ids, streams),
        scout_outputs=None, view_policy=view_policy, view_now=now)
    return preamble + "\n\n" + body, head


def aged_ids_of(text):
    """Ids of aged catalog entries. No tag marks them anymore (retired
    2026-08-18: the render is self-describing) — an aged entry is one whose
    withheld edges announce themselves in place, so detection keys on that
    announce line inside the entry, never on a whole-prompt scan (our own
    nodes quote the render format in their content). An aged entry with zero
    edges is undetectable — it also renders identically to a full one."""
    import re
    ids = set()
    for entry in section(text, 'node_catalog').split('\n\n'):
        if 'not shown — get_nodes for them' in entry:
            m = re.search(r'\(id:([0-9a-f]{6,8})', entry)
            if m:
                ids.add(m.group(1))
    return ids


def compare(arms, base='A'):
    """Per-section deltas against the base arm. Same copy, same instant — the
    only difference between two arms here is the render."""
    lines = []
    # --arms C alone is a normal case (re-running one arm on a new prompt);
    # fall back to whatever arm IS present rather than KeyError on 'A'.
    if base not in arms:
        base = sorted(arms)[0]
    b = arms[base][1]
    for key, (name, text, head) in sorted(arms.items()):
        if key == base:
            lines.append('  %s %-8s %8d chars   catalog %d entries, %d aged'
                         % (key, name, len(text), len(catalog_ids_of(text)),
                            len(aged_ids_of(text))))
            continue
        tag = '' if head is None else '  (cutoff turn %s)' % head
        lines.append('  %s %-8s %8d chars  %+6.1f%%  catalog %d aged%s'
                     % (key, name, len(text),
                        (len(text) - len(b)) * 100.0 / len(b),
                        len(aged_ids_of(text)), tag))
    for sec in ('node_catalog', 'timeline'):
        row = ['  <%s>' % sec]
        for key, (name, text, _h) in sorted(arms.items()):
            n = len(section(text, sec))
            d = '' if key == base else ' (%+.1f%%)' % (
                (n - len(section(b, sec))) * 100.0 / max(1, len(section(b, sec))))
            row.append('%s=%d%s' % (key, n, d))
        lines.append('  '.join(row))
    return '\n'.join(lines)


# ── Behavior pass ────────────────────────────────────────────────────────

def turn_index(brain, session_id, messages, streams):
    """{trace_id: {'turn': 1-based number, 'encoded': bool}} for the rendered
    window — built from the SAME builders that produced the render
    (_lived_turns + _turn_links), never by scraping the prompt back. Our own
    nodes quote turn tags and `<actions>` markup verbatim, so any regex over
    the assembled text reads its own documentation as data.

    EVERY episode in a turn is indexed, not just the user message: source_refs
    cite whatever `trace=` the timeline showed, and the encoder sees one on the
    assistant side and one per action too. Indexing only user traces reports
    those as off-window and hides refs landing on already-encoded turns — which
    is the headline signal.
    """
    from servers.scales.s1.encode import _lived_turns, _turn_links
    from servers.scales.s1.trace_links import display_turn
    n_turns = sum(1 for m in (messages or []) if m.get('role') == 'user') or 20
    turns = _lived_turns(brain, session_id, n_turns)
    links, _frontier = _turn_links(brain, session_id, turns, streams=streams)
    idx = {}
    for i, t in enumerate(turns):
        u = t.get('user') or {}
        n = display_turn(u.get('chain_id')) if u else None
        encoded = bool((links.get(u.get('id')) or {}).get('encoded_by'))
        for ep in [u, t.get('assistant')] + list(t.get('actions') or []):
            if ep and ep.get('id'):
                idx[ep['id']] = {'turn': n if n is not None else i,
                                 'encoded': encoded}
    return idx


class _Metered:
    """Thin proxy over the Anthropic client that tallies tokens.

    run_llm_loop returns its usage dict only on a clean exit, and this harness
    always leaves through the stop sentinel — so the count is taken at the
    call, not read off a return value that never arrives. Wrapping beats
    teaching RunLoopError to carry usage: production has no need for it.
    """

    def __init__(self, inner, usage):
        self._inner, self._usage = inner, usage
        self.messages = self._Messages(inner.messages, usage)

    def __getattr__(self, name):
        return getattr(self._inner, name)

    class _Messages:
        def __init__(self, inner, usage):
            self._inner, self._usage = inner, usage

        def __getattr__(self, name):
            return getattr(self._inner, name)

        def _tally(self, u):
            self._usage['input'] += getattr(u, 'input_tokens', 0) or 0
            self._usage['output'] += getattr(u, 'output_tokens', 0) or 0
            self._usage['cache_read'] += getattr(
                u, 'cache_read_input_tokens', 0) or 0
            self._usage['cache_write'] += getattr(
                u, 'cache_creation_input_tokens', 0) or 0

        def create(self, **kw):
            resp = self._inner.create(**kw)
            self._tally(getattr(resp, 'usage', None))
            return resp

        def stream(self, **kw):
            # run_llm_loop STREAMS; wrapping only create() silently metered
            # nothing. Usage rides the final message, so the tally happens
            # when the caller asks for it.
            outer = self

            class _Ctx:
                def __init__(self, inner_ctx):
                    self._ctx = inner_ctx

                def __enter__(self):
                    self._s = self._ctx.__enter__()
                    return self

                def __exit__(self, *a):
                    return self._ctx.__exit__(*a)

                def __iter__(self):
                    return iter(self._s)

                def __getattr__(self, name):
                    return getattr(self._s, name)

                def get_final_message(self):
                    msg = self._s.get_final_message()
                    outer._tally(getattr(msg, 'usage', None))
                    return msg

            return _Ctx(self._inner.stream(**kw))


def _synth_write_result(args):
    """A write result shaped like the real one — ids and all.

    An intercepted write must LOOK like it landed. Returning a bare note
    instead made the encoder disbelieve it and re-issue the whole batch every
    round until max_rounds, inflating counts ~5x and burning the budget before
    it ever wrote its journal. Ids are content-derived so a `connect_to`
    naming a sibling created earlier in the same run still resolves.
    """
    import hashlib
    ops = _ops_of({'args': args})
    results = []
    for i, op in enumerate(ops):
        nid = str(op.get('node_id') or op.get('survivor_id') or
                  hashlib.sha1((op.get('title') or str(i)).encode()
                               ).hexdigest()[:8])[:8]
        results.append({'op': op.get('op', 'remember'), 'index': i,
                        'ok': True,
                        'result': {'id': nid, 'title': op.get('title')}})
    return {'ok': True,
            'result': {'total': len(ops), 'succeeded': len(ops), 'failed': 0,
                       'results': results}}


def stored_contents(brain, ids):
    """{id8: stored content} for revise-target comparison."""
    if not ids:
        return {}
    got = brain.get_node(list(ids)) or {}
    return {k[:8]: (v.get('content') or '') for k, v in got.items()}


def run_arm_behavior(brain, prompt_text, aged_ids, index):
    """One arm through the encoder model — production engine, scored output.

    run_llm_loop (not a hand-rolled request loop) so caching, tool-result
    formatting and round structure match what the Scribe really does; the only
    substitution is the dispatch function.
    """
    from servers.scales.s1.encode import _build_system_prompt, _get_tool_schemas
    from servers.scales.s1.encode_contract import ENCODING_AGENT
    from servers.scales.runner import run_llm_loop
    from eval.longmem.replay import _make_local_dispatch
    from eval.longmem.connect_ab import WRITE_TOOLS

    real = _make_local_dispatch(brain)
    log = {'reads': [], 'writes': [], 'final_text': '', 'rounds': 0,
           'usage': {'input': 0, 'output': 0, 'cache_read': 0,
                     'cache_write': 0}}

    def dispatch(cmd, args=None):
        args = args or {}
        name = str(cmd).split('__')[-1]
        if name in WRITE_TOOLS:
            log['writes'].append({'tool': name, 'args': args})
            return _synth_write_result(args)
        log['reads'].append({'tool': name, 'args': args})
        return real(cmd, args)

    enc = brain.get_interaction('s1e') or {}
    cfg = brain.get_interaction_config('s1e') or {}
    # Runs to completion rather than cutting at the write round: the journal /
    # session-context the encoder emits is FINAL text, so stopping early threw
    # away one of the qualitative angles. Costs the wrap-up round (production
    # runs 2), and the return value carries final_text + rounds for free.
    res = run_llm_loop(
        client=_Metered(brain._ensure_anthropic_client(), log['usage']),
        model=cfg.get('model') or 'claude-sonnet-4-6',
        effort=cfg.get('effort') or None,
        max_tokens=ENCODING_AGENT['max_tokens'],
        max_rounds=ENCODING_AGENT.get('max_rounds', 5),
        system_prompt=_build_system_prompt(
            prompt_instructions=enc.get('template') or None, lived=True),
        user_content=prompt_text,
        tools=_get_tool_schemas(),
        dispatch_fn=dispatch) or {}
    log['final_text'] = res.get('final_text') or ''
    log['rounds'] = res.get('rounds', 0)
    return score_arm(log, aged_ids, index, brain), log


RICH_FIELDS = ('situation', 'reasoning', 'source_refs', 'connect_to',
               'user_raw_quote', 'anchor_raw_quote')


def score_shape(ops):
    """Node quality, not node count: how furnished is each created node."""
    if not ops:
        return {}
    n = len(ops)
    out = {f: round(sum(1 for o in ops if o.get(f)) / n, 2) for f in RICH_FIELDS}
    out['avg_content'] = int(sum(len(o.get('content') or '') for o in ops) / n)
    out['avg_edges'] = round(
        sum(len(o.get('connect_to') or ()) for o in ops) / n, 1)
    return out


def score_partial_view(revises, stored, aged_ids):
    """The check that caught the silent rewrite — automatic now.

    A revise REPLACES content. When the encoder only saw a stub, the tail it
    never read is destroyed and re-derived. Compares the proposed body against
    the stored one and reports how much of the original survives; `aged` marks
    the ones where the encoder was reasoning from ~400 chars.
    """
    rows = []
    for op in revises:
        nid = str(op.get('node_id') or op.get('survivor_id') or '')[:8]
        old, new = stored.get(nid), op.get('content')
        if not old or new is None:
            continue
        old_lines = {ln.strip() for ln in old.split('\n') if len(ln.strip()) > 25}
        kept = sum(1 for ln in old_lines if ln in new)
        rows.append({
            'id': nid, 'aged': nid in aged_ids,
            'old_chars': len(old), 'new_chars': len(new),
            'lines_kept': '%d/%d' % (kept, len(old_lines)),
            'dropped': round(1 - kept / max(1, len(old_lines)), 2),
        })
    return rows


def score_arm(log, aged_ids, index, brain=None):
    """The three behavioral questions, as counts over the recorded calls."""
    expanded = set()
    for r in log['reads']:
        for v in (r['args'] or {}).values():
            for i in (v if isinstance(v, list) else [v]):
                if isinstance(i, str) and i[:8] in aged_ids:
                    expanded.add(i[:8])

    creates, revises, refs, refs_unknown, on_encoded = 0, 0, 0, 0, 0
    revise_on_aged = 0
    created_ops, revise_ops = [], []
    for w in log['writes']:
        for op in _ops_of(w):
            kind = op.get('op') or ('revise' if 'node_id' in op else 'remember')
            if kind in ('revise', 'absorb', 'archive'):
                revises += 1
                revise_ops.append(op)
                if str(op.get('node_id') or op.get('survivor_id'))[:8] in aged_ids:
                    revise_on_aged += 1
                continue
            creates += 1
            created_ops.append(op)
            for ref in (op.get('source_refs') or []):
                refs += 1
                hit = index.get(ref)
                if hit is None:
                    refs_unknown += 1
                elif hit['encoded']:
                    on_encoded += 1
    return {
        'rounds': log['rounds'],
        'reads': len(log['reads']),
        'aged_expanded': len(expanded),
        'creates': creates, 'revises': revises,
        'revise_on_aged': revise_on_aged,
        'source_refs': refs,
        'refs_not_in_window': refs_unknown,
        'refs_on_encoded_turns': on_encoded,
        'usage': log['usage'],
        'shape': score_shape(created_ops),
        'partial_view': score_partial_view(
            revise_ops,
            stored_contents(brain, [str(o.get('node_id') or
                                        o.get('survivor_id') or '')
                                    for o in revise_ops]) if brain else {},
            aged_ids),
        'journal_chars': len(log['final_text']),
    }


def _ops_of(write):
    """Flatten a write call to its individual ops (batch tools carry a list)."""
    args = write.get('args') or {}
    for key in ('operations', 'nodes', 'revisions', 'connections'):
        v = args.get(key)
        if isinstance(v, list):
            return v
    return [args]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('captures', nargs='+',
                    help='payloads/<date>/<chain>/000-prompt.md')
    ap.add_argument('--arms', default='A,B,C')
    ap.add_argument('--out-dir', help='write each arm here as <chain>-<arm>.md')
    ap.add_argument('--behavior', action='store_true',
                    help='score each arm\'s emitted ops (spends API tokens)')
    ap.add_argument('--repeat', type=int, default=1,
                    help='behavior runs per arm — the model is stochastic, so '
                         'a single run per arm cannot separate policy effect '
                         'from spread (id:49a08613: a one-run encoder verdict '
                         'had to be retracted as variance)')
    ap.add_argument('--dump-ops', help='write each run\'s emitted ops here as '
                                       'JSON, for qualitative inspection')
    ap.add_argument('--s1e-patch',
                    help='markdown file inserted into the s1e template before '
                         'the run. Lands in the ISOLATED copy only — the live '
                         'daemon\'s prompt is never touched, so a candidate '
                         'can be measured before it is registered at all.')
    ap.add_argument('--patch-before', default='### Required fields (not optional)',
                    help='the template line the patch is inserted above')
    ap.add_argument('--s1e-template',
                    help='complete s1e template to run instead of the active '
                         'one — for candidates that EDIT existing text rather '
                         'than append. Isolated copy only, same as --s1e-patch.')
    args = ap.parse_args()
    want = [a.strip().upper() for a in args.arms.split(',') if a.strip()]

    from isolated_brain import IsolatedBrain
    # The behavior pass needs a key for the encoder model; the size pass never
    # calls out, so it stays keyless.
    with IsolatedBrain(cleanup=True, load_env=args.behavior) as env:
        brain = env.brain
        if args.s1e_template:
            from eval.longmem.ab_encode import inject_prompt
            active = brain.get_interaction('s1e') or {}
            with open(args.s1e_template) as f:
                tmpl = f.read()
            ver = inject_prompt(brain, tmpl, active.get('parameters') or '')
            print('[template] s1e %d -> %d chars from %s → eval-brain v%d '
                  '(live daemon untouched)'
                  % (len(active.get('template') or ''), len(tmpl),
                     args.s1e_template, ver))
        elif args.s1e_patch:
            from eval.longmem.ab_encode import inject_prompt
            active = brain.get_interaction('s1e') or {}
            tmpl = active.get('template') or ''
            if args.patch_before not in tmpl:
                raise SystemExit('anchor %r not in the s1e template — the '
                                 'patch would land somewhere arbitrary'
                                 % args.patch_before)
            with open(args.s1e_patch) as f:
                patch = f.read().rstrip() + '\n\n'
            ver = inject_prompt(brain, tmpl.replace(args.patch_before,
                                                    patch + args.patch_before, 1),
                                active.get('parameters') or '')
            print('[patch] s1e +%d chars from %s → eval-brain v%d (live '
                  'daemon untouched)' % (len(patch), args.s1e_patch, ver))
        for cap_path in args.captures:
            chain, _short, _stop = parse_chain(cap_path)
            with open(cap_path) as f:
                captured = strip_scout_blocks(f.read())
            session_id, run_ts = resolve_run(brain, chain)
            inputs = gather_inputs(brain, session_id, older_than=run_ts)
            arms = {}
            for key in want:
                name, vp, win, acc = ARMS[key]
                text, head = assemble_arm(brain, session_id, inputs, vp, win,
                                          aged_content_chars=acc)
                arms[key] = (name, text, head)
                if args.out_dir:
                    out = os.path.join(args.out_dir,
                                       '%s-%s-%s.md' % (chain, key, name))
                    with open(out, 'w') as f:
                        f.write(text)

            print('\n%s\n%s  (session %s, as-of %s)\n%s'
                  % ('=' * 72, chain, session_id[:8], run_ts, '=' * 72))
            print('captured: %d chars, %d turns, %d catalog ids'
                  % (len(captured), turn_count(captured),
                     len(catalog_ids_of(captured))))
            if 'A' in arms:
                ok, text = report(captured, arms['A'][1], brain=brain)
                print('control integrity vs the stored capture: %s'
                      % ('PASS' if ok else 'FAIL (drift — the A/B pair below '
                         'is still same-state and honest)'))
                print('\n'.join('  ' + ln for ln in text.splitlines()))
            print('\narms (one copy, one instant):')
            print(compare(arms))
            if args.out_dir:
                print('\nwrote %d arm file(s) to %s' % (len(arms), args.out_dir))

            if args.behavior:
                index = turn_index(brain, session_id, inputs['messages'],
                                   inputs['streams'])
                print('\nbehavior — %d turns in window, %d already encoded'
                      % (len(index),
                         sum(1 for v in index.values() if v['encoded'])))
                spend = {'input': 0, 'output': 0, 'cache_read': 0,
                         'cache_write': 0}
                for key in want:
                    name, text, _h = arms[key]
                    aged = aged_ids_of(text)
                    for run in range(1, args.repeat + 1):
                        s, log = run_arm_behavior(brain, text, aged, index)
                        for k in spend:
                            spend[k] += s['usage'][k]
                        print('  %s %-8s run%d rounds=%d reads=%d '
                              'aged-expanded=%d | creates=%d revises=%d '
                              '(on-aged=%d) | refs=%d on-encoded-turns=%d '
                              'off-window=%d'
                              % (key, name, run, s['rounds'], s['reads'],
                                 s['aged_expanded'], s['creates'],
                                 s['revises'], s['revise_on_aged'],
                                 s['source_refs'], s['refs_on_encoded_turns'],
                                 s['refs_not_in_window']))
                        if s['shape']:
                            sh = s['shape']
                            print('       shape: content~%d edges~%.1f | %s'
                                  % (sh['avg_content'], sh['avg_edges'],
                                     ' '.join('%s=%.0f%%' % (f[:4], 100 * sh[f])
                                              for f in RICH_FIELDS)))
                        for pv in s['partial_view']:
                            print('       PARTIAL-VIEW revise id:%s aged=%s '
                                  '%d→%d chars, kept %s lines (%.0f%% dropped)'
                                  % (pv['id'], pv['aged'], pv['old_chars'],
                                     pv['new_chars'], pv['lines_kept'],
                                     100 * pv['dropped']))
                        if args.dump_ops:
                            out = os.path.join(
                                args.dump_ops,
                                '%s-%s-run%d.json' % (chain, key, run))
                            with open(out, 'w') as f:
                                json.dump({'arm': key, 'arm_name': name,
                                           'chain': chain, 'run': run,
                                           'aged_ids': sorted(aged),
                                           'score': s,
                                           'journal': log['final_text'],
                                           'reads': log['reads'],
                                           'writes': log['writes']}, f,
                                          indent=2, default=str)
                # cache_write is billed input too — omitting it under-reports
                # the run, which is the number the scaling decision rests on.
                print('  spend: %d in = %d fresh + %d cache-write + %d '
                      'cache-read | %d out'
                      % (spend['input'] + spend['cache_write']
                         + spend['cache_read'], spend['input'],
                         spend['cache_write'], spend['cache_read'],
                         spend['output']))


if __name__ == '__main__':
    main()
