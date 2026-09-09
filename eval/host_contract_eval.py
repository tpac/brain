"""Paired code evaluation on one frozen IsolatedBrain database snapshot.

Run this script with each code revision, the SAME --source-dir and --session,
and --now instant, and separate --out directories. By default inspect the production
input; --encode also runs s1_encode_eval's dry-write LLM evaluation. Never point
--source-dir at live data: freeze once with IsolatedBrain(cleanup=False) first.
The snapshot includes the catalog and interaction overrides, not just actions.
"""
import argparse
from collections import Counter
from datetime import datetime
import hashlib
from html import unescape
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tests.isolated_brain import IsolatedBrain, _default_production_dir


def digest(path):
    with open(path, 'rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def code_revision(root):
    try:
        return subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'],
                                       text=True, stderr=subprocess.DEVNULL).strip()
    except subprocess.CalledProcessError:
        return 'exported tree; identified by consumer_hashes'


def _check_rendered_calls(calls, prompt):
    """Match measured output to actual action blocks; return unmeasured stubs.

    Encoded turns also have action blocks but do not call the condenser. Match
    measured blocks in order and compare the remaining blocks between arms.
    """
    blocks = [tuple(unescape(line.strip()) for line in body.splitlines() if line.strip())
              for body in re.findall(r'<actions>(.*?)</actions>', prompt, flags=re.S)]
    remaining, start = [], 0
    for call in calls:
        if not call['lines']:
            continue
        try:
            index = blocks.index(tuple(call['lines']), start)
        except ValueError:
            raise ValueError('rendered prompt omits a measured action block') from None
        remaining.extend(blocks[start:index])
        start = index + 1
    return remaining + blocks[start:]


def compare_inputs(before, after):
    """Refuse a code A/B with any input drift outside the changed actions."""
    a = json.loads((before / 'input.json').read_text())
    b = json.loads((after / 'input.json').read_text())
    if any(report.get('action_allocation', {}).get('notice') for report in (a, b)):
        raise ValueError('strict edit-retention comparison exceeded the timeline action limit; '
                         'use a smaller window or bounded balanced control settings')
    # Both absent preserves comparisons between historical reports. A new/old
    # pair lacks an effective-policy snapshot and must be prepared again.
    if a.get('action_allocation', {}).get('policy') != b.get('action_allocation', {}).get('policy'):
        raise ValueError('paired input differs: action policy (regenerate legacy reports)')
    for key in ('snapshot_hashes', 'session', 'render_now', 'flags', 'messages',
                'catalog_ids', 'interaction_stamp', 'interaction_config'):
        if a[key] != b[key]:
            raise ValueError('paired input differs: ' + key)
    for name in ('system.txt', 'tools.json'):
        if (before / name).read_bytes() != (after / name).read_bytes():
            raise ValueError('paired input differs: ' + name)
    # Older input-only reports predate this field; both absent is acceptable.
    if a.get('eval_execution') != b.get('eval_execution'):
        raise ValueError('paired input differs: eval_execution')
    prompts = [(p / 'prompt.txt').read_text() for p in (before, after)]
    outside = [re.sub(r'<actions>.*?</actions>', '<actions/>', p, flags=re.S)
               for p in prompts]
    if outside[0] != outside[1]:
        raise ValueError('paired prompt differs outside actions (check rendering clock)')
    if [(c['episodes'], c['is_tail']) for c in a['calls']] != [
            (c['episodes'], c['is_tail']) for c in b['calls']]:
        raise ValueError('different action windows reached the production condenser')
    protected_edits = sum(bool(action and action['protected'])
                          for c in b['calls'] for ep, action in zip(c['episodes'], c['parsed'])
                          if isinstance(ep.get('metadata'), dict)
                          and ep['metadata'].get('kind') == 'edit'
                          and ep['metadata'].get('kind_status') == 'ok')
    if not b['stamped_edits'] or protected_edits != b['stamped_edits']:
        raise ValueError('normalized edits are not all protected')
    for call in b['calls']:
        # Group by recorded identity, accepting every actual label variant.
        # A multiline trim (or its collision with the label cap) can change
        # the visible caption without changing the edit's grouping identity.
        groups = {}
        for ep, action in zip(call['episodes'], call['parsed']):
            md = ep.get('metadata')
            if not isinstance(md, dict) or md.get('kind') != 'edit' or md.get('kind_status') != 'ok':
                continue
            key = (str(ep.get('summary', action['label'])).split('\n', 1)[0], md.get('tool'))
            groups.setdefault(key, Counter())[action['label']] += 1
        seen_labels = set()
        for labels in groups.values():
            if seen_labels.intersection(labels):
                raise ValueError('ambiguous rendered edit captions across recorded identities; '
                                 'cannot prove per-target/tool retention from text')
            seen_labels.update(labels)
            cue = re.compile(r'(?:Closing: )?(?:' + '|'.join(re.escape(label) for label in labels)
                             + r')(?: ×(\d+)| \((\d+) edit calls\))?$')
            shown = 0
            for line in call['lines']:
                if match := cue.fullmatch(line):
                    shown += int(match[1] or match[2] or 1)
            if shown < sum(labels.values()):
                raise ValueError('normalized edits are not all retained in rendered actions')
    if _check_rendered_calls(a['calls'], prompts[0]) != _check_rendered_calls(b['calls'], prompts[1]):
        raise ValueError('unmeasured action blocks changed')
    return {'paired': True, 'measured_actions': sum(len(c['episodes']) for c in b['calls']),
            'stamped_edits': b['stamped_edits'], 'protected_edits': protected_edits,
            'retained_edits': protected_edits,
            'added_prompt_chars': len(prompts[1]) - len(prompts[0])}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check-pair', nargs=2, type=Path, metavar=('BEFORE', 'AFTER'))
    parser.add_argument('--source-dir', type=Path)
    parser.add_argument('--session')
    parser.add_argument('--now', type=datetime.fromisoformat,
                        help='fixed timezone-aware rendering instant for BOTH revisions')
    parser.add_argument('--out', type=Path)
    parser.add_argument('--encode', action='store_true')
    args = parser.parse_args()
    if args.check_pair:
        print(json.dumps(compare_inputs(*args.check_pair)))
        return
    if not all((args.source_dir, args.session, args.now, args.out)):
        parser.error('preparing inputs requires --source-dir, --session, --now and --out')
    if args.now.tzinfo is None:
        parser.error('--now must include a timezone')
    source = args.source_dir.resolve()
    production = _default_production_dir()
    if production and source == Path(production).resolve():
        parser.error('--source-dir must be a frozen copy, not production')
    args.out.mkdir(parents=True, exist_ok=False)
    hashes = {name: digest(source / name)
              for name in ('brain.db', 'brain_logs.db', 'aspects_v1.json')
              if (source / name).exists()}

    from servers.scales.s1 import encode, encoder_actions, encoder_view
    from eval.s1_encode_eval import (run_encoding, EVAL_MODEL,
                                    EVAL_MAX_TOKENS, EVAL_TOOL_ROUNDS)
    if not (encode._lived_sequence_enabled() and encoder_view.view_policy_enabled()):
        parser.error('production lived-sequence and view-policy flags must be enabled')

    with IsolatedBrain(production_dir=str(source)) as env:
        brain = env.brain
        messages = encode._gather_messages(brain, args.session)
        if not messages:
            raise RuntimeError('empty session window')
        calls = []
        original = encoder_actions.prepare_action_block
        original_render = encoder_actions.render_action_blocks
        allocation = {}

        def measured(episodes, *, is_tail, encoded, view_policy, policy):
            block = original(episodes, is_tail=is_tail, encoded=encoded,
                             view_policy=view_policy, policy=policy)
            if not encoded:
                parsed = [encoder_actions.parse_action(ep) for ep in episodes]
                calls.append({'is_tail': is_tail, 'episodes': episodes,
                              'parsed': [{key: getattr(a, key) for key in a.__slots__}
                                         if a else None for a in parsed],
                              'lines': [line.text for line in block.lines]})
            return block

        def measured_render(blocks, policy):
            rendered, notice = original_render(blocks, policy)
            allocation.update(policy=vars(policy), blocks=rendered, notice=notice,
                              serialized_bytes=len((''.join(rendered) + notice).encode('utf-8')))
            return rendered, notice

        journal = encode._journal(brain, args.session)
        system = encode._build_system_prompt(
            prompt_instructions=brain.get_interaction_prompt('s1e') or None,
            lived=True, journal=journal)
        tools = encode._get_tool_schemas()
        catalog = encode._build_catalog(brain, messages, args.session, True,
                                        view_policy=True, now=args.now)
        with patch.object(encoder_actions, 'prepare_action_block', measured), \
                patch.object(encoder_actions, 'render_action_blocks', measured_render):
            preamble, body, _, catalog_ids = encode._build_user_content(
                brain, messages, 5, args.session, journal=journal,
                precomputed=catalog, view_now=args.now)
        if not calls or not allocation:
            raise RuntimeError('production prompt did not prepare and allocate unencoded actions')
        stamped_edits = [ep for call in calls for ep in call['episodes']
                         if isinstance(ep.get('metadata'), dict)
                         and ep['metadata'].get('kind_status') == 'ok'
                         and ep['metadata'].get('kind') == 'edit']
        if not stamped_edits:
            raise RuntimeError('measured action window has no stamped edits')
        content = preamble + '\n' + body
        (args.out / 'system.txt').write_text(system)
        (args.out / 'prompt.txt').write_text(content)
        (args.out / 'tools.json').write_text(json.dumps(tools, indent=2))
        report = {
            'revision': code_revision(Path(encode.__file__).resolve().parents[3]),
            'consumer_hashes': {name: digest(Path(encoder_actions.__file__).parent / name)
                                for name in ('encoder_actions.py', 'encoder_view.py')},
            'action_policy_hash': digest(Path(encode.__file__).resolve().parents[2] / 'action_policy.py'),
            'snapshot_hashes': hashes, 'session': args.session,
            'render_now': args.now.isoformat(),
            'flags': {k: v for k, v in os.environ.items() if k.startswith('BRAIN_S1E_')},
            'interaction_stamp': brain.get_interaction_stamp('s1e'),
            'interaction_config': brain.get_interaction_config('s1e'),
            # The existing dry-write helper deliberately runs a bounded common
            # loop. Its execution settings differ from the production runner;
            # do not mistake the recorded interaction config for applied knobs.
            'eval_execution': {'helper': 'eval.s1_encode_eval.run_encoding',
                               'helper_sha256': digest(Path(__file__).with_name('s1_encode_eval.py')),
                               'model': EVAL_MODEL, 'max_tokens': EVAL_MAX_TOKENS,
                               'effort': 'API default', 'max_calls': 1 + EVAL_TOOL_ROUNDS,
                               'writes': 'dry-run'},
            'messages': messages, 'catalog_ids': sorted(catalog_ids),
            'calls': calls, 'stamped_edits': len(stamped_edits),
            'action_allocation': allocation,
            'prompt_chars': len(content),
        }
        (args.out / 'input.json').write_text(json.dumps(report, indent=2, default=str))
        print(json.dumps({'out': str(args.out), 'calls': len(calls),
                          'stamped_edits': len(stamped_edits), 'prompt_chars': len(content)}), flush=True)
        if args.encode:
            import anthropic
            result = run_encoding(anthropic.Anthropic(), system, content, tools, brain)
            (args.out / 'result.json').write_text(json.dumps(result, indent=2, default=str))
            print(json.dumps(result, default=str), flush=True)
    assert hashes == {name: digest(source / name) for name in hashes}, 'frozen source changed'


if __name__ == '__main__':
    main()
