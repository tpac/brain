"""Frozen V3.1 comparison: old, tools-only, and revised guide with new tools.

All replacements are eval artifacts. Shared runtime and previous freezes stay
unchanged. Author before selecting the new evaluation material.
"""
import argparse
import difflib
import hashlib
import json
from pathlib import Path
import re
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path[:0] = [str(ROOT/'eval/fixtures/s1e_guide_freeze_2026-09-08'),
                str(ROOT/'eval/fixtures/s1e_tool_descriptions_2026-09-08')]
from frozen_arms import load_arm as load_original
from candidate import load_candidate


def digest(value):
    if not isinstance(value, bytes):
        value = value.encode() if isinstance(value, str) else json.dumps(value,sort_keys=True).encode()
    return hashlib.sha256(value).hexdigest()


def replace(text, old, new):
    if text.count(old) != 1:
        raise ValueError('Non-unique authoring anchor: '+old[:100])
    return text.replace(old,new)


def build():
    old = load_original('v3_titles')
    tools = load_candidate('v3_titles')
    base = (ROOT/'eval/candidate_prompts/s1e_guide_v3_enhanced_titles_2026-09-08.md').read_text()
    template = replace(base,
        'Every field takes a whole new value or `{old, new}` (a list for multiple swaps).',
        'Writable text fields take a whole new value or `{old, new}` (a list for multiple swaps); other fields take bare values.')
    template = replace(template,
        '**Read what I lack → encode → inspect results → close.** Usually two rounds, three with a needed read; the count is not a writing budget.',
        '**Read what I lack → encode → inspect → repair if needed → close.** Reads supply missing evidence; successful writes supply changes, not proof that the resulting memory is complete.')
    template = replace(template,
        'Then compare the plan against successful operations and returned field changes. A missed write or field remains work; a dense window or needed repair can require another write before closing.',
        'Then read the resulting claims against the conversation: combine the prior fields with successful changes, including fields initially called clean. Check what was learned, what changed and what still holds. A missed or unsupported claim calls for repair before closing; a successful batch is not that comparison.')
    template = replace(template,
        'My first reply makes the evidence and intended destinations visible:',
        'I compare claims before assigning verdicts. “The room booking is not confirmed” is still a claim even inside reasoning; the new confirmation changes it. The event date and the question about where and when remain appropriate. I group fields that share a claim so the comparison stays compact:')
    template = replace(template,
        'targets: a6b0139d · title stale · content stale · situation stale · question clean · reasoning stale · event_time clean · why→82c41f0b clean · why→61de80a2 clean',
        'targets: a6b0139d · provisional venue / unconfirmed booking → Annex confirmation, same slot → room booked, access unknown: title stale · content stale · situation stale · reasoning stale; October 17 and where/when question still fit: event_time clean · question clean; access dependency and earlier correction still apply: why→82c41f0b clean · why→61de80a2 clean')
    template = replace(template,
        'targets: 82c41f0b · title stale · content stale · situation stale · question stale · reasoning stale',
        'targets: 82c41f0b · room and ramp unanswered → room confirmed, ramp unanswered → retain only the access question: title stale · content stale · situation stale · question stale · reasoning stale')
    template = replace(template,
        'The read confirms the earlier instance; its node is clean. I link the new\ninterpretation to it.',
        'The read describes a dated correction about welcome versus setup. The new\nexchange repeats that distinction without falsifying the earlier event or\nits scope: those fields stay clean. I link the new interpretation to it.')
    template = replace(template,
        "I check the other returned changes too: booking claims changed,\n17:00–19:00 and browsing survived, and access stays open. The plan's\ntitle, content and edges all describe agreed work. Successful memory\nwrites do not mean its card or checks were completed. A listed board\nwithout a successful remember would still be a miss.",
        "I read the resulting memory against the exchange, not just the list:\nconfirmation now reaches the venue's title, body, trigger and reasoning;\nthe unchanged date and where/when question still fit. The time and browsing\noption survived. The access node and its edge still leave the ramp unanswered.\nThe earlier correction remains a dated event, while the new interpretation\nis my supported read. The board's location is independently findable.\nThe plan's title, body and edges describe agreed work, not a completed card\nor walkthrough. If any of these claims failed that comparison—even one\nI initially called clean—I would repair it in another tool call before\nwriting Arc and Review.")
    gist = old['gist']
    gist = replace(gist,
        'A read reply is followed by the write reply; the reply after the write is the close.',
        'After a read I write from the evidence returned. After a write I inspect the resulting claims and repair what remains before closing.')
    gist = replace(gist,
        '`targets` — for each change, I walk EVERY catalog entry, every Edges line, and every id my continuity names, and write ONE LINE PER NODE the change touches,',
        '`targets` — for each change, I walk EVERY catalog entry, every Edges line, and every id my continuity names. For each touched node I first compare its old assertion with the new evidence and state what now holds; fields sharing an assertion can share one compact comparison. Then I give the field verdicts on that same node line,')
    gist = replace(gist,
        'Then the close: compare every `new` and stale field with successful writes and returned changes.',
        'After tool results, reconstruct the resulting claims from prior fields and successful changes and compare them with the evidence, including fields initially marked clean. Did the new knowledge survive, did changed claims move, and did still-true detail remain? Repair supported omissions or contradictions before the close.')
    strategy = '''## Working strategy

I read prior memory and the conversation for details and developing meaning.
In the four lists, the evidence comparison produces each verdict; a verdict
does not substitute for the comparison. I fetch missing material, then preserve
new knowledge and revise changed claims at their supported scope.

After tool results, I reconstruct what the memory now says—including fields
left unchanged—and compare it with the evidence. I repair remaining supported
changes before closing. Arc and Review carry unresolved questions and movement
left by that work; they do not defer a repair I can make or bind a future run
to a no-mint verdict.
'''
    closure = '''## Finishing

The run ends on the first reply with no tool call. Tool results may require
another call: use a read's evidence for the write, inspect what the write
changed, and repair remaining supported changes before closing. The final
reply is the only place for Arc and Review, whether changes were needed or not.

End the final reply with the `## Review`, then write "DONE".'''
    suffix = old['system_prompt'][len(base):]
    field_old = 'On revise a field takes its NEW VALUE (the whole field replaced) or a swap `{old, new}`'
    suffix = replace(suffix,field_old,
        'On revise a writable text field takes its NEW VALUE (the whole field replaced) or a swap `{old, new}`')
    strategy_at = suffix.index('## Working strategy')
    suffix = suffix[:strategy_at]+strategy.rstrip()+'\n\n'+closure
    # The second existing example and its gist excerpt must demonstrate the
    # same comparison, rather than teach a competing bare-verdict procedure.
    comparisons = [
        ('targets: e91a6d05 · title clean · content stale',
         'targets: e91a6d05 · auth first → abandonment agreed → gateway next: content stale; Q3 queue still names this work: title clean'),
        ('targets: 7d21c4aa · title stale · content stale',
         'targets: 7d21c4aa · awaiting review/merge → branch deleted, never merged → abandoned implementation: title stale · content stale'),
        ('targets: b8e05f92 · title stale · content stale · situation stale',
         'targets: b8e05f92 · live merge verdict/rebuild → implementation abandoned → preserve verdict as history: title stale · content stale · situation stale'),
        ('targets: c37d10be · title stale · content stale · edges unread',
         'targets: c37d10be · six active branches including auth → auth branch deleted → active inventory changes: title stale · content stale · edges unread'),
        ('targets: a45c88f1 · title stale · content unread · why→e91a6d05 stale',
         'targets: a45c88f1 · auth before gateway → replacement order agreed → prior order superseded: title stale · why→e91a6d05 stale; body unavailable: content unread'),
    ]
    for before, after in comparisons:
        template = replace(template, before, after)
        gist = replace(gist, before, after)
    revised = {**tools,'arm_id':'v3_1_titles','system_prompt':template+suffix,'gist':gist}
    # Prose/procedure changes only: all JSON tool demonstrations remain exact.
    assert re.findall(r'```json\n(.*?)\n```',base,re.S) == re.findall(r'```json\n(.*?)\n```',template,re.S)
    assert len(template) < len(base)*1.025
    assert revised['system_prompt'].endswith(closure)
    assert 'the reply after the write is the close' not in gist
    assert "the write's results by the final reply" not in revised['system_prompt']
    assert revised['system_prompt'].count('## Working strategy') == 1
    return old,tools,revised,{'template.md':template,'gist.md':gist,'strategy.md':strategy,'closure.md':closure}


def author():
    old,tools,new,parts=build()
    if (HERE/'manifest.json').exists(): raise FileExistsError('V3.1 already frozen')
    arms={'v3_titles':old,'v3_titles_new_tools':tools,'v3_1_titles':new}
    records={}
    for name,value in arms.items():
        data={k:value[k] for k in ('system_prompt','gist','tools','settings')}
        data.update(arm_id=name,arm_sha256=digest(data))
        filename=name+'.json'
        (HERE/filename).write_text(json.dumps(data,indent=2,ensure_ascii=False)+'\n')
        records[name]={'file':filename,'arm_sha256':data['arm_sha256'],
            'system_chars':len(data['system_prompt']),'gist_chars':len(data['gist']),
            'tools_chars':len(json.dumps(data['tools'],ensure_ascii=False))}
    for filename,value in parts.items(): (HERE/filename).write_text(value)
    for name,source,dest in [('system',old['system_prompt'],new['system_prompt']),('gist',old['gist'],new['gist'])]:
        (HERE/(name+'.diff')).write_text(''.join(difflib.unified_diff(source.splitlines(True),dest.splitlines(True),fromfile='v3_titles',tofile='v3_1_titles')))
    paths=[Path(__file__)]+[HERE/n for n in parts]+[HERE/(n+'.json') for n in arms]+[HERE/'system.diff',HERE/'gist.diff']
    pin={'status':'frozen_before_new_corpus_selection','arms':records,
         'files':{p.name:digest(p.read_bytes()) for p in paths},
         'scope':'eval-only assembled system/gist/tool snapshots; no shared runtime edits',
         'target_mechanisms':['mistaken-clean verification','known repair deferred to later run'],
         'no_changes':['JSON examples','schema mechanics','tool membership','field quotas','journal header/nudge','limits','source-reference policy']}
    (HERE/'manifest.json').write_text(json.dumps(pin,indent=2)+'\n')
    print(json.dumps(records,indent=2))


def load_arm(name):
    pin=json.loads((HERE/'manifest.json').read_text())
    for filename,expected in pin['files'].items():
        if digest((HERE/filename).read_bytes()) != expected: raise ValueError('Arm artifact changed: '+filename)
    load_original('v3_titles'); load_candidate('v3_titles')
    data=json.loads((HERE/pin['arms'][name]['file']).read_text())
    identity={k:data[k] for k in ('system_prompt','gist','tools','settings')}
    if digest(identity)!=data['arm_sha256']: raise ValueError('Arm identity mismatch')
    return data


if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--author',action='store_true'); args=parser.parse_args()
    if args.author: author()
    else:
        for name in ('v3_titles','v3_titles_new_tools','v3_1_titles'):
            print(name,load_arm(name)['arm_sha256'])
