"""Version the context corrections without altering the completed sanity cell."""
import difflib
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
OLD_FIXTURE = ROOT / 'eval/fixtures/s1e_guide_sanity_2026-09-08/contrasts_three_windows.json'
OLD_RUNNER = ROOT / 'eval/s1e_guide_v2_sequence_probe.py'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    expected = {
        OLD_FIXTURE: '9a13976581370395b8a25109e24b64d09ea6c544e1226c230e86eb4586fe7a6c',
        OLD_RUNNER: 'c7c33d9b880b54551c89bb4607a1a8b2227e4edb5658a3ea554f5dcc1c465e1c',
    }
    for path, digest in expected.items():
        assert sha(path) == digest, 'Old cell input changed: ' + str(path)
    old = json.loads(OLD_FIXTURE.read_text())
    new = dict(old)
    new['description'] = (
        'Context revision of the completed 12-turn Oren sanity sequence: speaker explicitly '
        'identified in timeline markup, and displayed journal first-seen dates use the '
        'conversation window that produced the note. Dialogue, actions, seed facts and '
        'all predeclared criteria are unchanged. Requires this directory\'s versioned runner. '
        'This remains one synthetic regression sequence, not an independent benchmark corpus.')
    new['context_version'] = 'oren_explicit_conversation_journal_dates_v2'
    new['other_speaker'] = 'Oren'
    new['journal_date_policy'] = 'map first_seen at rendering only; real journal lifecycle and transaction timestamps retained'
    source = OLD_RUNNER.read_text()
    replacements = [
        ('WT = Path(__file__).resolve().parents[1]',
         'from context_support import ReplayJournal\n\nWT = Path(__file__).resolve().parents[3]'),
        ("OUT = WT / 'eval/results/s1e_guide_v2_2026-09-08/sequence'",
         "OUT = WT / 'eval/results/s1e_guide_context_v2_2026-09-08'"),
        ("FIXTURE = WT / 'eval/fixtures/s1e_guide_v2_2026-09-08/discovery_sequence.json'",
         "FIXTURE = Path(__file__).resolve().parent / 'contrasts_three_windows.json'"),
        ("SID = 'd15c0a72-v2-fresh-sequence'", "SID = 'e42ad630-context-v2-sequence'"),
        ('journal = _journal(brain, SID)', 'journal = ReplayJournal(_journal(brain, SID))'),
        ("            d = folder / f'window{wn}'", "            journal.set_window(window['now'])\n            d = folder / f'window{wn}'"),
        ('''parts.append(f'  <other trace="{turn["other_trace"]}">{escape(turn["other"])}</other>')''',
         '''parts.append(f'  <other speaker="{escape(pinned["fixture"]["other_speaker"])}" trace="{turn["other_trace"]}">{escape(turn["other"])}</other>')'''),
        ("'continuity':'real JournalBinding harvest/render after window 1'",
         "'continuity':'real JournalBinding lifecycle; displayed first_seen dates mapped to the producing fixture window'"),
    ]
    revised = source
    for before, after in replacements:
        assert revised.count(before) == 1, 'Runner anchor not unique: ' + before
        revised = revised.replace(before, after)
    files = {
        'contrasts_three_windows.json': json.dumps(new, indent=2, ensure_ascii=False)+'\n',
        'run_sequence.py': revised,
        'runner.diff': ''.join(difflib.unified_diff(source.splitlines(True), revised.splitlines(True),
                                                  fromfile=str(OLD_RUNNER.relative_to(ROOT)), tofile='run_sequence.py')),
        'fixture.diff': ''.join(difflib.unified_diff(OLD_FIXTURE.read_text().splitlines(True),
            (json.dumps(new,indent=2,ensure_ascii=False)+'\n').splitlines(True),
            fromfile=str(OLD_FIXTURE.relative_to(ROOT)), tofile='contrasts_three_windows.json')),
    }
    if any((HERE / name).exists() for name in [*files, 'manifest.json']):
        raise FileExistsError('Context v2 already exists; author another version')
    for name, value in files.items():
        (HERE / name).write_text(value)
    paths = [HERE/name for name in files] + [Path(__file__), HERE/'context_support.py']
    manifest = {
        'status': 'authored_for_offline_review', 'model_calls': 0,
        'prior_cell_inputs': {str(p.relative_to(ROOT)): h for p,h in expected.items()},
        'files': {str(p.relative_to(ROOT)): sha(p) for p in paths},
        'turn_counts': [len(w['turns']) for w in new['windows']],
        'guide_and_tools': 'unchanged; the versioned runner still selects frozen arms and frozen tools',
        'comparison_rule': 'All future arms must share this new context; do not pool with old-context samples.',
        'runner_scope': 'Versioned snapshot with eight exact substitutions; no shared runtime edit.',
    }
    (HERE/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print('AUTHORED context v2: same 12 turns, explicit Oren, conversation-date journal render')


if __name__ == '__main__':
    main()
