"""Score the V3.8 replay-probe dumps — what each reply did about the gym time on the three captured windows.

Reads `<results>/dump/<arm>_<window>/repeatN.json` written by replay_payload.py --dump and prints one row per
reply plus a per-(arm, window) tally. The readouts are the class's flavours (AUDIT-COVERED-TURN-2026-09-14.md):

  wrote        the reply made a tool call (a reply without one ends the run with zero writes)
  gym_touched  any op targets a gym-schedule node (title/content mentions gym) or the gym `open`
  move         what the reply did with the May "usually at 6:00 pm":
                 supersede   — a gym node's title or content now states 6 pm as the current time
                               (or supersedes / "as of" with 6 pm) — the target behaviour
                 contradict  — an `open`/contradiction node minted or kept ("vs", "which is correct", "unresolved")
                 bent        — the 6 pm is explained around the 7 pm node (buffer, departure, travel, prep, leave, arrival)
                 thought     — only `thought` touched on the gym node
                 none        — no gym-time handling
  method_alert the dependent reminder method's alert time is touched (6 pm → 5 pm)
  verdict_reuse the reply's text cites the arc / residue as a reason not to write ("per session arc", "transactional")

Keyword scoring is a first read; the dumps are the evidence. Usage: ./dev python3 probe_score.py <results-dir>
"""
import json, re, sys
from collections import Counter, defaultdict
from pathlib import Path

GYM = re.compile(r'gym', re.I)
SIX = re.compile(r'\b6(?::00)?\s?pm\b|\b18:00\b', re.I)
SUPERSEDE = re.compile(r'supersed|as of 2023-05|as of may|now (?:at )?6|moved to 6|changed? (?:from 7|to 6)|6(?::00)? ?pm .{0,40}\(?(?:was|from|previously) 7', re.I)
CONTRA = re.compile(r'\bvs\.?\b|which is correct|unresolved|contradict|contested|conflict', re.I)
BENT = re.compile(r'buffer|departure|depart|travel|prep(?:aration)? window|leave for|leaves? (?:for|by)|arriv|head(?:ing)? (?:out|to) .{0,30}6|consistent with (?:the )?7', re.I)
REUSE = re.compile(r'per (?:the )?session arc|arc note|transactional|already encoded as|no durable node', re.I)


def node_text(op):
    parts = []
    for k in ('title', 'content', 'situation', 'reasoning', 'question', 'thought', 'reason'):
        v = op.get(k)
        if isinstance(v, str):
            parts.append(v)
        elif isinstance(v, dict):
            parts.append(v.get('new', '') or '')
        elif isinstance(v, list):
            parts.extend((s.get('new', '') or '') for s in v if isinstance(s, dict))
    return '\n'.join(parts)


def ops_of(tool_use):
    inp = tool_use['input']
    ops = inp.get('operations') or inp.get('nodes') or inp.get('revisions') or inp.get('edges') or []
    out = []
    for o in ops if isinstance(ops, list) else []:
        if not isinstance(o, dict):
            continue
        op = o.get('op') or {'remember_batch': 'remember', 'revise_batch': 'revise', 'connect_batch': 'connect'}.get(tool_use['name'], tool_use['name'])
        out.append((op, o))
    return out


def score(rep, catalog_gym_ids, method_ids, schedule_ids):
    texts = '\n'.join(rep['texts'])
    uses = rep['tool_uses']
    row = {'wrote': bool(uses), 'gym_touched': False, 'move': 'none', 'method_alert': False,
           'verdict_reuse': bool(REUSE.search(texts)), 'ops': 0, 'reads': [u['name'] for u in uses if u['name'] in ('get_nodes', 'recall_batch')]}
    gym_new_text, gym_fields, minted_contra = [], set(), False
    for u in uses:
        for op, o in ops_of(u):
            row['ops'] += 1
            t = node_text(o)
            is_gym_target = op == 'revise' and o.get('node_id') in catalog_gym_ids
            is_gym_new = op == 'remember' and GYM.search(t or '') and SIX.search(t or '')
            if is_gym_target or is_gym_new:
                row['gym_touched'] = True
                gym_new_text.append(t)
                gym_fields |= {k for k in o if k not in ('op', 'node_id', 'reason', 'type', 'connect_to')}
                if op == 'remember' and (o.get('type') == 'open' or CONTRA.search(t)):
                    minted_contra = True
            if op == 'revise' and o.get('node_id') in method_ids or (
                    'reminder' in (o.get('title') if isinstance(o.get('title'), str) else json.dumps(o.get('title') or '')).lower() and re.search(r'5(?::00)? ?pm', t or '')):
                row['method_alert'] = True
    joined = '\n'.join(gym_new_text)
    if row['gym_touched']:
        # supersede = a REVISE of the gym SCHEDULE node (not the method, not the open, not a new node) whose new
        # title or content states 6 pm without hedging it as a contradiction — the node now says the current time.
        sched_tc = []
        for u in uses:
            for op, o in ops_of(u):
                if op == 'revise' and o.get('node_id') in schedule_ids:
                    for k in ('title', 'content'):
                        v = o.get(k)
                        if isinstance(v, str): sched_tc.append(v)
                        elif isinstance(v, dict): sched_tc.append(v.get('new', '') or '')
                        elif isinstance(v, list): sched_tc.extend((s.get('new', '') or '') for s in v if isinstance(s, dict))
        tc = '\n'.join(sched_tc)
        if tc and SIX.search(tc) and not CONTRA.search(tc):
            row['move'] = 'supersede'
        elif minted_contra or CONTRA.search(tc) or CONTRA.search(joined):
            row['move'] = 'contradict'
        elif BENT.search(joined) or BENT.search(texts):
            row['move'] = 'bent'
        elif gym_fields and gym_fields <= {'thought'}:
            row['move'] = 'thought'
        else:
            row['move'] = 'other'
    elif BENT.search(texts) and SIX.search(texts):
        row['move'] = 'bent'
    elif CONTRA.search(texts) and SIX.search(texts):
        row['move'] = 'contradict'
    row['gym_fields'] = sorted(gym_fields)
    return row


# The gym nodes per window, read from the captured catalogs (000-prompt.md) this session.
CATALOG = {  # window: (gym-related nodes, the reminder method node, the gym SCHEDULE node)
    'r2': ({'e24a9967', '8c5d886c'}, {'8c5d886c'}, {'e24a9967'}),
    'r3': ({'e24a9967', '8c5d886c', '61675f7f', 'fcdb3c42'}, {'8c5d886c'}, {'e24a9967', '61675f7f'}),
    'r3b': ({'b70654b3', '578b7f8c'}, {'578b7f8c'}, {'b70654b3'}),
}


def main():
    root = Path(sys.argv[1]) / 'dump'
    tally = defaultdict(Counter)
    rows = []
    for d in sorted(root.iterdir()):
        arm, win = d.name.rsplit('_', 1)
        gym_ids, method_ids, schedule_ids = CATALOG[win]
        for f in sorted(d.glob('repeat*.json')):
            rep = json.loads(f.read_text())
            r = score(rep, gym_ids, method_ids, schedule_ids)
            rows.append((arm, win, f.stem, r))
            tally[(arm, win)]['n'] += 1
            tally[(arm, win)]['wrote'] += r['wrote']
            tally[(arm, win)]['gym'] += r['gym_touched']
            tally[(arm, win)]['supersede'] += r['move'] == 'supersede'
            tally[(arm, win)]['contradict'] += r['move'] == 'contradict'
            tally[(arm, win)]['bent'] += r['move'] == 'bent'
            tally[(arm, win)]['thought'] += r['move'] == 'thought'
            tally[(arm, win)]['alert'] += r['method_alert']
            tally[(arm, win)]['reuse'] += r['verdict_reuse']
            tally[(arm, win)]['reads'] += bool(r['reads'])
    print(f"{'arm':8} {'win':4} {'rep':8} {'wrote':5} {'gym':4} {'move':10} {'alert':5} {'reuse':5} {'ops':3} fields")
    for arm, win, rep, r in rows:
        print(f"{arm:8} {win:4} {rep:8} {str(r['wrote']):5} {str(r['gym_touched']):4} {r['move']:10} {str(r['method_alert']):5} {str(r['verdict_reuse']):5} {r['ops']:3} {','.join(r['gym_fields'])}{' reads=' + ','.join(r['reads']) if r['reads'] else ''}")
    print('\nper (arm, window) — of n repeats: wrote / gym touched / supersede / contradict / bent / thought-only / alert moved / verdict reused / read round')
    for (arm, win), c in sorted(tally.items(), key=lambda kv: (kv[0][1], kv[0][0])):
        print(f"{arm:8} {win:4} n={c['n']}  wrote={c['wrote']} gym={c['gym']} supersede={c['supersede']} contradict={c['contradict']} bent={c['bent']} thought={c['thought']} alert={c['alert']} reuse={c['reuse']} reads={c['reads']}")


if __name__ == '__main__':
    main()
