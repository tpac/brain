"""Verify captured requests, sequential memory carry and usage; no semantic scorer."""
import json
from pathlib import Path
import sys
from xml.sax.saxutils import escape

from run_cell import ARMS, OUT, WT, check_pin

sys.path.insert(0, str(WT / 'eval/fixtures/s1e_guide_freeze_2026-09-08'))
from frozen_arms import digest, load_arm


def core(node):
    result = {key:node.get(key) for key in ('type','title','content','_metadata','situation','confidence','evolution_status')}
    result['connections'] = sorted(
        (edge['id'], edge['direction'], relation['relation'], relation.get('description',''))
        for edge in node.get('connections',[]) for relation in edge.get('relations',[])
    )
    return result


def main():
    pin = check_pin()
    report = {'status':'passed', 'sequences':[], 'limits':'Instrument verification, not semantic scoring.'}
    first_hashes = set()
    for arm in ARMS:
        frozen = load_arm(arm)
        for rep in range(1,4):
            folder = OUT / arm / f'repeat{rep}'
            manifest = json.loads((folder / 'manifest.json').read_text())
            assert manifest['arm_sha256'] == frozen['arm_sha256']
            assert (folder / 'system.txt').read_text() == frozen['system_prompt']
            assert json.loads((folder / 'tools.json').read_text()) == frozen['tools']
            record = {'arm':arm,'repeat':rep,'windows':[], 'usage':dict(input=0,output=0,cache_read=0,cache_write=0)}
            previous_nodes = None; previous_continuity = None
            for wn in range(1,4):
                d = folder / f'window{wn}'
                result = json.loads((d/'result.json').read_text())
                before = json.loads((d/'nodes_before.json').read_text())
                after = json.loads((d/'nodes_after.json').read_text())
                if previous_nodes is not None:
                    assert set(previous_nodes).issubset(before)
                    assert all(core(before[nid]) == core(node) for nid,node in previous_nodes.items())
                user = (d/'user.txt').read_text()
                if previous_continuity is not None:
                    assert '<continuity>\n'+escape(previous_continuity)+'\n</continuity>' in user
                if wn == 1:
                    first_hashes.add(digest(user))
                captures = sorted(d.glob('round*.json'))
                assert len(captures) == result['result']['rounds']
                for path in captures:
                    request = json.loads(path.read_text())
                    assert request['system'] == frozen['system_prompt']
                    assert request['model'] == frozen['settings']['model']
                    assert request['effort'] == frozen['settings']['effort']
                    assert request['tools'] == [t['name'] for t in frozen['tools']]
                first = json.loads(captures[0].read_text())
                assert '\n\n'.join(b['text'] for b in first['messages'][0]['content']) == user
                failed = []; warnings = []; totals = []; operations = []
                for call in json.loads((d/'calls.json').read_text()):
                    outer = call['result']; value = outer.get('result',outer)
                    if outer.get('ok') is False:
                        failed.append(outer)
                    if 'total' in value:
                        totals.append({k:value.get(k) for k in ('total','succeeded','failed','connect_to_failures')})
                    for op in value.get('results',[]):
                        operations.append({k:op.get(k) for k in ('op','index','node_id','ok','status','error')})
                        if op.get('ok') is False:
                            failed.append(op)
                        warnings.extend(op.get('warnings',[]))
                        details = op.get('result',{})
                        warnings.extend(details.get('warnings',[]))
                        failed.extend(details.get('verification_failures',[]))
                    failed.extend(value.get('connect_to_failed',[]))
                for key,n in result['usage'].items():
                    record['usage'][key] += n
                record['windows'].append({'window':wn,'rounds':result['result']['rounds'],
                    'writes':len(result['writes']),'reads':len(result['reads']),
                    'elapsed_ms':result['result']['elapsed_ms'],'operation_totals':totals,
                    'operation_results':operations,
                    'failures':failed,'warnings':warnings,'truncations':result['result'].get('truncations')})
                previous_nodes = after
                previous_continuity = (d/'next_continuity.txt').read_text()
            record['usage']['all_input'] = sum(record['usage'][k] for k in ('input','cache_read','cache_write'))
            report['sequences'].append(record)
    assert len(first_hashes) == 1
    assert first_hashes == {(OUT/'first_user_sha256.txt').read_text()}
    report['first_user_sha256'] = first_hashes.pop()
    report['encodes'] = sum(len(s['windows']) for s in report['sequences'])
    assert report['encodes'] == pin['total_encodes'] == 18
    report['api_rounds'] = sum(w['rounds'] for s in report['sequences'] for w in s['windows'])
    (OUT/'verification.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__ == '__main__':
    main()
