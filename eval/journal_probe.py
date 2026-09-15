#!/usr/bin/env python3
"""Inspect current journals, probe model edits, or run S2 on an isolated copy.

    ./dev python3 eval/journal_probe.py --output /tmp/journal-probe
    ./dev python3 eval/journal_probe.py --output /tmp/journal-probe --llm
    ./dev python3 eval/journal_probe.py --output /tmp/journal-s2 --s2

The LLM mode uses a fresh synthetic fixture and code-default model configs;
it never opens production data. It checks journal protocol, not encoding quality.
The optional S2 mode uses the real coordinator and its normal gates. All writes
are isolated. Saved prompts, configs, responses and fingerprints support review;
no automatic quality verdict substitutes for reading model behavior.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tests.isolated_brain import IsolatedBrain
from servers.scales.journal import JournalBinding
from servers.scales.runner import make_client, run_llm_once
from servers.trace_contract import journal_tool_schema, parse_journal_operations

ENCODERS = {
    's1e': 's1e',
    'community_detection': 's2_community_enrichment',
    'consolidation': 's2_consolidation_enrichment',
    'healer': 's2_healer',
    'aspect_integration': 's2_aspects',
}


def digest(text):
    return hashlib.sha256(text.encode()).hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + '\n')


def probe_model(brain, name, interaction, output, client):
    scope = dict(scale='s1', session_id='journal-probe-session') if name == 's1e' else dict(
        scale='s2', unit='journal_probe_' + name)
    binding = JournalBinding(brain, **scope)
    prefix = 's1e-probe-' if name == 's1e' else 's2-probe-'
    suffix = '' if name == 's1e' else '-' + scope['unit']
    chain = prefix + '0' + suffix
    binding.continuity(chain_id=chain)
    binding.apply({'operations': [
        dict(op='note', subject='deployment', text='Verify the loaded source fingerprint.', persist=True),
        dict(op='note', subject='deployment', text='Investigate intermittent recall timeouts.', persist=True),
    ]}, chain)
    config = brain.get_interaction_config(interaction)
    system = binding.decorate_system(
        'You have completed a run of %s. Review the supplied observations.' % name,
        multi_round=False)
    steps = [
        'Tool results confirmed the loaded production fingerprint matches the intended main revision. '
        'Nothing was learned about the intermittent recall timeouts. No other findings.',
        'No new observations or changes since the previous batch.',
    ]
    results = []
    for batch, observations in enumerate(steps, 1):
        chain = prefix + '1' + suffix  # two requests, one invocation
        continuity = binding.continuity(chain_id=chain)
        user = continuity + observations
        before = binding._view['notes']
        calls, usage = run_llm_once(client, config['model'], 2048, system, user, tools=[
            journal_tool_schema(), dict(name='submit_probe_result',
                description='Finish this synthetic protocol exercise. Call once, alongside journal if there are changes.',
                input_schema={'type': 'object', 'properties': {}, 'additionalProperties': False})])
        operations, errors = [], []
        for call in calls:
            if call['name'] == 'journal':
                accepted, rejected = parse_journal_operations(call['input'].get('operations'))
                operations.extend(accepted)
                errors.extend(rejected)
                outcome = binding.apply(call['input'], chain)
                if not outcome['ok']:
                    errors.append(outcome['error'])
            elif call['name'] != 'submit_probe_result':
                errors.append('Unknown tool: ' + call['name'])
        if not any(call['name'] == 'submit_probe_result' for call in calls):
            errors.append('Missing task result')
        after = brain.journal_view(**scope)['notes']
        record = dict(batch=batch, model=config['model'], max_tokens=2048, encoder_config=config,
                      system=system, user=user, system_sha256=digest(system),
                      user_sha256=digest(user), tool_calls=calls, usage=usage,
                      operations=operations, parse_errors=errors, before=before, after=after)
        save(output / ('%s-batch-%d.json' % (name, batch)), record)
        results.append(dict(batch=batch, operations=len(operations), parse_errors=errors, usage=usage))
        print('%s batch %d: %d operations, %d parse errors' % (
            name, batch, len(operations), len(errors)), flush=True)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--production-dir')
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--llm', action='store_true')
    mode.add_argument('--s2', action='store_true')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    diff = subprocess.check_output(['git', 'diff', 'HEAD', '--', 'servers'], cwd=ROOT, text=True)
    report = dict(revision=revision, runtime_diff_sha256=digest(diff), views={}, probes={})
    (args.output / 'runtime.diff').write_text(diff)
    if args.llm:
        from tests.brain_test_base import BrainTestBase
        from servers.scales.dispatch import load_env
        import unittest
        report['source_dataset'] = 'fresh synthetic fixture; no production database opened'

        class SyntheticProbe(BrainTestBase):
            needs_embedder = False

            def runTest(self):
                load_env()
                client = make_client()
                for name, interaction in ENCODERS.items():
                    report['probes'][name] = probe_model(self.brain, name, interaction, args.output, client)
                    save(args.output / 'report.json', report)

        result = SyntheticProbe().run(unittest.TestResult())
        if not result.wasSuccessful():
            raise RuntimeError('Synthetic probe failed:\n' + '\n'.join(
                detail for _, detail in result.errors + result.failures))
        return
    with IsolatedBrain(production_dir=args.production_dir, load_env=args.s2) as env:
        report['source_dataset'] = env.production_dir
        for name in ENCODERS:
            if name == 's1e':
                continue  # S1 continuity must name a particular session.
            start = time.perf_counter()
            binding = JournalBinding(env.brain, scale='s2', unit=name)
            view = binding.continuity()  # read-only; no invocation clock advance
            report['views'][name] = dict(milliseconds=1000 * (time.perf_counter() - start),
                                         **binding.stats)
            (args.output / (name + '-view.txt')).write_text(view)
        if args.s2:
            report['s2'] = env.brain.run_s2()
        save(args.output / 'report.json', report)
    print('Saved %s' % args.output, flush=True)


if __name__ == '__main__':
    main()
