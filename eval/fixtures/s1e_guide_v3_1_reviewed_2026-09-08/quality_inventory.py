"""Descriptive census and readable packets from saved, persisted node snapshots.

No model, database or semantic auto-grader. Field counts do not imply quality.
"""
from collections import Counter,defaultdict
import json
from pathlib import Path
import statistics

ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'eval/results/s1e_v31_cross_corpus_2026-09-08'
FIELDS=('title','content','situation','question','reasoning','thought','their_raw_quote','my_raw_quote','event_time','emotion','emotion_label','confidence','type','evolution_status')
TEXT_FIELDS=FIELDS[:8]


def field(node,key):
    return node.get(key,(node.get('_metadata') or {}).get(key))


def words(value):
    return len(value.split()) if isinstance(value,str) else 0


def quantile(values,p):
    values=sorted(values)
    return values[min(len(values)-1,round((len(values)-1)*p))] if values else 0


def summary(values):
    return {'n':len(values),'total':sum(values),'median':statistics.median(values) if values else 0,
            'min':min(values,default=0),'max':max(values,default=0),'p90':quantile(values,.9)}


def get_edges(node):
    out=[]
    for edge in node.get('connections') or []:
        if edge.get('direction')!='outgoing': continue
        for relation in edge.get('relations') or [edge]:
            if relation.get('relation') not in ('co_anchored','community_member'):
                out.append({'target_id':edge['id'],'target_title':edge.get('title'),
                            'relation':relation.get('relation'),'description':relation.get('description','')})
    return out


def authored_fields(folder):
    out=defaultdict(set)
    for path in sorted(folder.glob('window*/calls.json')):
        for record in json.loads(path.read_text()):
            name=record['call']['tool']; args=record['call']['args']; response=record['result'].get('result',{})
            if not isinstance(response,dict): continue
            operations=(args.get('operations',[]) if name=='brain_batch' else args.get('nodes',[]) if name=='remember_batch'
                        else args.get('revisions',[]) if name=='revise_batch' else [])
            for i,result in enumerate(response.get('results',[])):
                if result.get('ok') is False or result.get('error'): continue
                index=result.get('index',i)
                if index>=len(operations): continue
                op=operations[index]; returned=result.get('result',result)
                nid=returned.get('id') or op.get('node_id')
                if nid: out[nid].update(k for k in op if k in FIELDS)
    return out


def main():
    reports=[]; rows=[]
    for path in sorted(OUT.glob('*/repeat*/*/window3/nodes_after.json')):
        folder=path.parent.parent; corpus=folder.name; repeat=folder.parent.name; arm=folder.parent.parent.name
        nodes=json.loads(path.read_text())
        if not isinstance(nodes,dict): raise TypeError('Unexpected saved node shape')
        nodes={k:v for k,v in nodes.items() if isinstance(v,dict) and v.get('title')}
        fixture=json.loads((OUT/(corpus+'.json')).read_text())
        sources={role:'\n'.join(t[role] for w in fixture['windows'] for t in w['turns']) for role in ('other','me')}
        authored=authored_fields(folder)
        text=['# '+arm+' / '+repeat+' / '+corpus,'','Source and field evidence must be reviewed; word counts below are descriptive.','']
        all_words=[]; field_words=defaultdict(list); by_type=Counter(); edges_seen=set(); edges=[]; quote_checks=[]
        for nid,node in nodes.items():
            values={k:field(node,k) for k in FIELDS}; by_type[values['type']]+=1
            nwords=sum(words(values[k]) for k in TEXT_FIELDS); all_words.append(nwords)
            text.extend(['## '+nid+' ['+str(values['type'])+'] '+str(values['title']),f'Authored text including quotes: {nwords} words',''])
            for key in FIELDS:
                v=values[key]
                if v not in (None,'',[],{}):
                    field_words[key].append(words(v)); text.extend([key+': '+(v if isinstance(v,str) else json.dumps(v,ensure_ascii=False)),''])
            for key,role in [('their_raw_quote','other'),('my_raw_quote','me')]:
                quote=values[key]
                if isinstance(quote,str) and quote:
                    quote_checks.append({'node_id':nid,'field':key,'exact_source_substring':quote in sources[role],
                        'whitespace_normalized_substring':' '.join(quote.split()) in ' '.join(sources[role].split()),'quote':quote})
            node_edges=get_edges(node)
            for edge in node_edges:
                target=edge.get('target_id') or edge.get('id') or edge.get('node_id')
                relation=edge.get('relation'); why=edge.get('description') or edge.get('why') or ''
                ek=(nid,str(target),str(relation),str(why))
                if ek not in edges_seen: edges_seen.add(ek); edges.append({'source':nid,'target':target,'relation':relation,'why':why})
                text.append('EDGE '+json.dumps(edge,ensure_ascii=False))
            rows.append({'arm':arm,'repeat':repeat,'corpus':corpus,'node_id':nid,'type':values['type'],
                         'fields':values,'explicitly_authored_fields':sorted(authored[nid]),'text_words':nwords,'edges':node_edges,'dump':str(path.relative_to(ROOT))})
            text.append('')
        op_counts=Counter(); total_usage=Counter(); changes=[]; failures=[]
        for window in sorted(folder.glob('window*')):
            before=json.loads((window/'nodes_before.json').read_text()); after=json.loads((window/'nodes_after.json').read_text())
            result=json.loads((window/'result.json').read_text()); total_usage.update(result['usage'])
            text.extend(['# '+window.name+' operations and closing text',''])
            for call in json.loads((window/'calls.json').read_text()):
                name=call['call']['tool']; args=call['call']['args']
                op_counts['tool:'+name]+=1
                if name=='brain_batch':
                    for op in args.get('operations',[]): op_counts['op:'+op.get('op','?')]+=1
                elif name=='remember_batch': op_counts['op:remember']+=len(args.get('nodes',[]))
                elif name=='revise_batch': op_counts['op:revise']+=len(args.get('revisions',[]))
                elif name=='connect_batch': op_counts['op:connect']+=len(args.get('connections',[]))
                response=call['result']; inner=response.get('result',{}) if isinstance(response,dict) else {}
                if isinstance(inner,dict) and (inner.get('failed') or inner.get('connect_to_failures')):
                    failures.append({'window':window.name,'call':call})
                text.append('CALL '+json.dumps(call,ensure_ascii=False))
            for nid,new in after.items():
                previous=before.get(nid)
                if isinstance(new,dict) and isinstance(previous,dict):
                    deltas={k:{'old':field(previous,k),'new':field(new,k)} for k in FIELDS if field(previous,k)!=field(new,k)}
                    if previous.get('connections')!=new.get('connections'):
                        deltas['connections']={'old':previous.get('connections'),'new':new.get('connections')}
                    if deltas: changes.append({'window':window.name,'node_id':nid,'changes':deltas})
            text.append('FINAL '+str(result['result'].get('final_text','')))
        text.extend(['# Actual before/after changes',json.dumps(changes,ensure_ascii=False,indent=2)])
        (folder/'quality_packet.md').write_text('\n'.join(text)+'\n')
        report={'arm':arm,'repeat':repeat,'corpus':corpus,'nodes':len(nodes),'types':dict(by_type),
            'node_text_words':summary(all_words),'fields':{k:summary(v) for k,v in field_words.items()},
            'explicitly_authored_field_counts':dict(Counter(k for nid in nodes for k in authored[nid])),
            'edges':len(edges),'edge_words':summary([words(e['why']) for e in edges]),
            'operations':dict(op_counts),'usage':dict(total_usage),
            'actual_revised_nodes':len({c['node_id'] for c in changes if any(k!='connections' for k in c['changes'])}),
            'nodes_with_changed_edge_neighborhood':len({c['node_id'] for c in changes if 'connections' in c['changes']}),
            'changes':changes,'partial_failure_records':failures,'quote_source_checks':quote_checks}
        report['metadata_key_counts']=dict(Counter(k for node in nodes.values() for k in (node.get('_metadata') or {})))
        report['custom_fields']={nid:{k:v for k,v in (node.get('_metadata') or {}).items() if k not in FIELDS}
                                 for nid,node in nodes.items() if any(k not in FIELDS for k in (node.get('_metadata') or {}))}
        report['encoding_sources']=dict(Counter(str(node.get('encoding_source')) for node in nodes.values()))
        report['locked_nodes']=[nid for nid,node in nodes.items() if node.get('locked')]
        report['personal_flag_nodes']=[nid for nid,node in nodes.items() if node.get('personal')]
        reports.append(report)
    (OUT/'quality_inventory.json').write_text(json.dumps({'sequences':reports,'nodes':rows},indent=2,ensure_ascii=False)+'\n')
    for r in reports:
        print(r['arm'],r['repeat'],r['corpus'],'nodes',r['nodes'],'words',r['node_text_words']['total'],
              'median',r['node_text_words']['median'],'revised',r['actual_revised_nodes'])


if __name__=='__main__': main()
