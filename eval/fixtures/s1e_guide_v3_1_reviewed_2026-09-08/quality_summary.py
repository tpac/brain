"""Derived census plus explicitly hand-selected semantic classifications.

No API calls and no edits to frozen inputs. Advice membership was assigned by
reading each final node against source, not by type/name regex. Membership
means standalone advice/reference, not a judgment that every such node is bad.
"""
import collections
import json
from pathlib import Path
import re
from quality_inventory import OUT, FIELDS, TEXT_FIELDS, summary, words

ADVICE={
 ('v3_titles','repeat1'):'58594a2d 5bafc753 b3385d07 ca3d150f d4f809d7 f8ecaced',
 ('v3_titles','repeat2'):'',
 ('v3_titles','repeat3'):'14f7c834 2b212a0d 5c669659 dd5b323b',
 ('v3_titles_new_tools','repeat1'):'0cca46fd 2ef16a81 5f2d6291 612b4031 69af872f 984dd3d0 af732747',
 ('v3_titles_new_tools','repeat2'):'2f060fd3 6cc65710 7bb706b0 9c748ef6 9d2f7f91 a0839d93 b91da88e c49f2301 d5c04091 ee1330a7',
 ('v3_titles_new_tools','repeat3'):'',
 ('v3_1_titles','repeat1'):'',
 ('v3_1_titles','repeat2'):'06bc19e8 21050595 2f621e4e 2fca1f0e 6a580d3b 8e7185ae c2453095',
 ('v3_1_titles','repeat3'):'580faf5f 594aa907 5b626338 6b63ea6f aade1632 b495c696 e8171841 f0cd9e6b fdd507b1',
}


def main():
    inventory=json.loads((OUT/'quality_inventory.json').read_text())
    nodes=inventory['nodes']; sequences=inventory['sequences']; groups=[]; advice=[]
    for arm in ('v3_titles','v3_titles_new_tools','v3_1_titles'):
        for corpus in ('creative_design','longmem_unseen'):
            ns=[n for n in nodes if n['arm']==arm and n['corpus']==corpus]
            ss=[s for s in sequences if s['arm']==arm and s['corpus']==corpus]
            groups.append({'arm':arm,'corpus':corpus,'nodes':len(ns),
                'per_repeat':[{'repeat':s['repeat'],'nodes':s['nodes'],'node_words':s['node_text_words'],
                    'relations':s['edges'],'relation_words':s['edge_words']['total'],
                    'actual_field_revised_nodes':s['actual_revised_nodes']} for s in ss],
                'fields':{f:{**summary([words(n['fields'][f]) for n in ns if n['fields'][f] not in (None,'')]),
                            'explicitly_authored_nodes':sum(f in n['explicitly_authored_fields'] for n in ns)} for f in FIELDS},
                'words':summary([n['text_words'] for n in ns]),
                'turn_coordinate_nodes':sum(any(re.search(r'\bturns?\s+\d',str(n['fields'][f]),re.I) for f in TEXT_FIELDS) for n in ns),
                'usage':dict(sum((collections.Counter(s['usage']) for s in ss),collections.Counter())),
                'operations':dict(sum((collections.Counter(s['operations']) for s in ss),collections.Counter()))})
    for (arm,repeat),ids in ADVICE.items():
        ns=[n for n in nodes if n['arm']==arm and n['repeat']==repeat and n['corpus']=='longmem_unseen']
        chosen=[n for n in ns if n['node_id'] in ids.split()]
        advice.append({'arm':arm,'repeat':repeat,'nodes':len(chosen),'node_ids':ids.split(),
            'words':sum(n['text_words'] for n in chosen),'all_node_words':sum(n['text_words'] for n in ns),
            'note':'Standalone advice/reference; local utility varies. Advice embedded in personal/arc nodes is not counted here.'})
    value_rows=[]
    for n in nodes:
        is_advice=n['corpus']=='longmem_unseen' and n['node_id'] in ADVICE[(n['arm'],n['repeat'])].split()
        category='advice/reference' if is_advice else 'specific context or evolving plan'
        if n['corpus']=='creative_design':
            category='useful synthesis' if n['type'] in ('vision','concept','insight','principle') else 'specific design knowledge'
            if n['node_id'] in ('16f7194c','4eff3406','a17ae07f','c66a7311'):
                category='redundant quote handle (formulation retained elsewhere)'
        value_rows.append({'arm':n['arm'],'repeat':n['repeat'],'corpus':n['corpus'],'node_id':n['node_id'],
            'primary_information_role':category,'future_use':n['fields']['question'] or n['fields']['situation'],
            'word_count':n['text_words'],'evidence_dump':n['dump'],
            'qualification':'Information role is not an accuracy score. Cross-field/edge defects and marginal value are adjudicated in REVIEW-NOTES.md.'})
    payload={'groups':groups,'standalone_advice':advice,'node_information_roles':value_rows,
             'method':'Descriptive counts; semantic claims require source comparison. No composite score.'}
    (OUT/'quality_summary.json').write_text(json.dumps(payload,indent=2,ensure_ascii=False)+'\n')
    for g in groups:
        print(g['arm'],g['corpus'],'nodes/words',[(r['nodes'],r['node_words']['total']) for r in g['per_repeat']],
              'text words',g['words']['total'],'turn coordinates',g['turn_coordinate_nodes'])
    print('ADVICE',[(a['arm'],a['repeat'],a['nodes'],round(100*a['words']/a['all_node_words'],1)) for a in advice])


if __name__=='__main__': main()
