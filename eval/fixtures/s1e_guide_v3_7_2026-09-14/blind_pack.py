"""Build blind paired-review packs: one source conversation, the arms' final memories, shuffled.

A reviewer reads the source and memories A/B/C/… without knowing which arm
produced which; the key is sealed in key.json and opened only when the
verdicts are in. Packs are built from the packets analyze.py wrote, so run
that first. Every pack carries the same rubric so verdicts line up across
corpora and repeats. An arm that did not run on a corpus (the tail arm's
subsample) is simply absent from that pack.

    ./dev python3 eval/fixtures/s1e_guide_v3_7_2026-09-14/blind_pack.py --cell carriers
"""
import argparse
import hashlib
import json
from pathlib import Path
import random
import re

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
CELLS = {
    'carriers': {'out': ROOT / 'eval/results/s1e_v37_carriers_2026-09-14', 'arms': ['v3_6_full', 'v3_7_advice', 'v3_7_quote']},
}
RUBRIC = '''# How to review

You see one source conversation (every window the encoder saw, in order) and several final memories (A, B, C, …) produced by different encoder configurations from that same source, each in its own isolated memory store. You do not know which configuration is which. Judge the memories, not the encoder's prose. Quote the memory text for every claim you make; cite node ids.

Dimensions, each judged separately (no total score):
1. Facts and concrete detail — names, numbers, dates, objects, exact phrases; first disclosures; later corrections. What did each memory keep, drop, or get wrong?
2. Decisions and arcs — what was decided, what comes first and why, what was set aside, unresolved developments, distinct stories kept distinct. Does a reader learn the ORDER and the REASON, or just the subject?
3. Revision and preservation — where a later window changed something earlier, did the memory update the existing claim, leave contradictory twins, or invent an update? Are still-true details preserved after updates?
4. Evidence strength, ownership and scope — plans stored as plans, proposals as proposals, agreement only where given, hedges kept as hedges, failed trials bounded; who said or proposed what. Name flattening or overclaiming in any field, including situation and edge descriptions.
5. Voice and synthesis — both voices present and distinguishable; useful understanding the encoder developed beyond the source (patterns, distinctions, meaning between nodes) at the scope the evidence supports; overgeneralization ("always", personality claims from one instance).
6. Recall usefulness — discriminating titles, situations in trigger register, questions that carry the point, meaningful edge descriptions, duplication or near-twins, nodes with no neighbors.
7. Receiver's view — pick the 3–5 nodes a future reader asking about the source's main thread would most plausibly retrieve (by title and situation). From those alone, can they recover the purpose, what was decided and why, what was considered instead, and what would reopen it? Which memory serves that reader best?
8. Content and field quality — per node, does each filled field earn its place: reasoning as the claim's basis (source, strength, limits) rather than a restatement or a justification for storing; thought as the encoder's own hunch or connection rather than a caveat or a repeat; edge descriptions saying what neither node says alone; quotes verbatim and load-bearing (both voices where both spoke the knowledge); situation as a future trigger and question as a real asking. Name formulaic, padded or empty-but-filled fields, and fields whose absence is a loss.

Then answer the corpus-specific probes listed in the pack.

Output: for each of the eight dimensions, one paragraph per memory with quoted evidence, then a one-line comparative verdict for that dimension naming the best and worst memory or "no material difference". Finish with a table: dimension × memory with a short label (strong / adequate / weak) and the single most consequential defect per memory. Do not guess which configuration is which. Do not reward length, node count, or field count.
'''
PROBES = {
    'creative_design': [
        'Which memories preserve that the FIRST prototype step is the graph visualization plus temperature colors (the user\'s stated priority), and where (title, content, situation, quote, arc)?',
        'Is D3.js stored as the assistant\'s proposal or as an agreed/decided stack? Quote each memory\'s wording in content AND situation.',
        'Is the audio feature stored as "optional and quiet by default" (the assistant\'s proposal) or as an off-by-default requirement?',
        'Does any memory carry the distinction that cold/dormant knowledge is not necessarily bad, and the vision → features → beauty movement of the session?',
        'Does any memory overgeneralize the mirror-not-camera (pull-only personal insights) principle into a rule about ordinary contextual notifications?',
        'Who is credited with the four theme names and with "the aesthetic IS the product"? Check content/reasoning against the quotes.',
    ],
    'lm_default': [
        'Where a value changed between windows (a count, a date, a frequency, a place), list every surface of the revised node that still carries the OLD value — title, situation, question, quotes, edge descriptions, event_time — for each memory. A node that is right in content and wrong in its trigger or an edge is half-revised.',
        'Where the assistant\'s own turn carried the knowledge (a diagnosis, an explanation, a recommendation the other side adopted), does the node carry the assistant\'s exact words (my_raw_quote) or only prose about them? Compare across memories.',
        'Where the source merely MENTIONS something (a later reference, a recipe, a passing use), did any memory write it as confirmed, present or done? Quote the memory word and the source phrase.',
        'If the other side states the same fact twice with different values and no word of correction, does any memory keep both dated values, or does every memory treat the later one as a correction of the record? If no such pair exists in this source, say so.',
        'The source contains a fact that changes across sessions (or a fact asked about later). Does the final memory hold the CURRENT value, and is the earlier value marked superseded/historical rather than left as a live competing claim?',
        'Are dates resolved to absolute dates correctly against each window\'s clock? Quote event_time values and any relative phrases left unresolved.',
        'Which memory would let a reader answer the question in the pack\'s GOLD section from its top few nodes? Quote the nodes.',
    ],
    'conv_002_debugging': [
        'Which technical specifics (error text, file or function names, versions, commands, numbers) does each memory keep exactly, and which does it paraphrase or lose?',
        'Where the exchange corrected an earlier hypothesis or approach, does the memory store the correction as a correction (what was assumed, what turned out true), or only the final state?',
        'Are the assistant\'s own diagnoses and proposals stored as the assistant\'s, with their status (tried, confirmed, rejected), and is the other side\'s report kept in its own words?',
    ],
    'conv_005_emotions': [
        'Does the memory preserve the emotional register of the exchange and the assistant\'s own stance in its own words (my_raw_quote / first person), or flatten it to summary?',
        'Are the assistant\'s identity claims stored at the strength the exchange supports, without inventing a general trait?',
    ],
}


def read(path):
    return json.loads(Path(path).read_text())


def source_text(fixture):
    lines = [f"Source: {fixture['source_id']} — {fixture['source_kind']}", f"Clock: {fixture['clock']}", '']
    n = 0
    for wi, w in enumerate(fixture['windows'], 1):
        lines.append(f'## Window {wi} (now = {w["now"]})')
        for t in w['turns']:
            n += 1
            who = fixture.get('counterpart') or 'other'
            lines += [f'[{n}] {who}: {t["other"]}', f'[{n}] me: {t["me"]}', '']
    return '\n'.join(lines)


def memory_text(packet_path):
    text = Path(packet_path).read_text()
    # keep nodes and closing text; drop the carried continuity dumps (they repeat)
    return re.sub(r'# Window \d+ carried continuity\n.*?(?=\n# |\Z)', '', text, flags=re.S)


def build(cell):
    conf = CELLS[cell]; out = conf['out']
    packets = out / 'whole_memory_review'
    dest = out / 'blind_review'; dest.mkdir(exist_ok=True)
    key = {}
    corpora = sorted(read(out / 'manifest.json')['corpora'])
    for corpus in sorted(corpora):
        fixture = read(out / (corpus + '.json'))
        for repeat in range(1, 4):
            arms = [a for a in conf['arms'] if (packets / f'{a}_repeat{repeat}_{corpus}.md').exists()]
            seed = int(hashlib.sha256(f'v37:{cell}:{corpus}:{repeat}'.encode()).hexdigest()[:8], 16)
            random.Random(seed).shuffle(arms)
            labels = dict(zip('ABCDE', arms))
            key[f'{corpus}_repeat{repeat}'] = labels
            parts = [RUBRIC, '# Corpus-specific probes', '']
            probes = PROBES.get(corpus) or PROBES['lm_default']
            parts += [f'- {p}' for p in probes]
            if 'question' in fixture['prior_gold']:
                g = fixture['prior_gold']
                parts += ['', '# GOLD (for probe 3 only; the memories never saw this)', f"Question ({g.get('question_date')}): {g['question']}", f"Reference answer: {g['answer']}"]
            parts += ['', '# SOURCE CONVERSATION', '', source_text(fixture)]
            for label in labels:
                arm = labels[label]
                packet = packets / f'{arm}_repeat{repeat}_{corpus}.md'
                body = memory_text(packet)
                body = re.sub(r'^# .*?whole saved memory\n', '', body)
                body = body.replace(arm, 'MEMORY-' + label)
                parts += ['', f'# MEMORY {label}', '', body]
            (dest / f'{corpus}_repeat{repeat}.md').write_text('\n'.join(parts) + '\n')
    (dest / 'key.json').write_text(json.dumps(key, indent=2) + '\n')
    print('packs:', len(key), 'key sealed at', dest / 'key.json')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cell', choices=list(CELLS), required=True)
    build(parser.parse_args().cell)
