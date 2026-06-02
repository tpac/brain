#!/usr/bin/env python3
"""Ask the consolidation encoder WHY it under-merged specific corpus clusters.

We don't guess the cause of under-merge — we ask the model. Feed Sonnet its own
candidate prompt (as system) + the exact clusters it KEPT instead of merging
(from the ground-truth corpus, with their signals: members, types, cosines,
pre_class) + the ground-truth rationale for why each IS a duplicate. Then have it
introspect: what about the prompt, the signals, or its own reasoning made it not
merge?

Usage:
    ./dev python3 eval/why_underconsolidate_probe.py
    ./dev python3 eval/why_underconsolidate_probe.py --clusters 5,6,10,11,12
"""
import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

CANDIDATE = os.path.join(ROOT, 'eval', 'candidate_prompts', 's2_consolidation_absorb.md')
CLUSTERS = os.path.join(ROOT, 'eval', 'corpus', 'clusters.json')
LABELS = os.path.join(ROOT, 'eval', 'corpus', 'labels.json')

# Clusters the candidate UNDER-merged in corpus_score_k3_v2 (kept genuine duplicates).
DEFAULT_UNDER = [5, 6, 10, 11, 12]


def build_payload(cluster_ids):
    clusters = {c['cluster_id']: c for c in json.load(open(CLUSTERS))}
    labels = json.load(open(LABELS))['labels']
    out = []
    for cid in cluster_ids:
        c = clusters[cid]
        lab = labels[str(cid)]
        out.append({
            'cluster_id': cid,
            'decoder_pre_class': c['pre_class'],
            'content_cosine': c['content_cosine'],
            'title_cosine': c['title_cosine'],
            'members': [{'id': m['id'], 'type': m.get('type'),
                         'title': m.get('title'), 'content': (m.get('content') or '')[:450]}
                        for m in c['members']],
            'GROUND_TRUTH_action': lab['action'],
            'GROUND_TRUTH_why_duplicate': lab['rationale'],
        })
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--clusters', default=','.join(map(str, DEFAULT_UNDER)),
                    help='comma-separated cluster ids the encoder under-merged')
    args = ap.parse_args()
    cluster_ids = [int(x) for x in args.clusters.split(',')]

    from eval.agent_introspect._common import call_sonnet, load_env
    from servers.scales.s2.consolidation_contract import CONSOLIDATION
    load_env()

    system = open(CANDIDATE).read().strip()
    payload = build_payload(cluster_ids)

    question = (
        "You ARE the S2 consolidation encoder — the prompt above is YOUR system prompt.\n\n"
        "On each cluster below you chose KEEP (a `similar_to` edge) instead of merging. "
        "But each is a GENUINE DUPLICATE — the `GROUND_TRUTH_action` is absorb/split and "
        "`GROUND_TRUTH_why_duplicate` says why. So you UNDER-merged: you kept things you "
        "should have consolidated.\n\n"
        "Introspect honestly — we are trying to fix under-merging, so don't defend the keep:\n"
        "1. For each cluster, what specifically made you NOT merge? Name the cause: was it "
        "the `decoder_pre_class: likely_keep` signal? a rule/line in the prompt (quote it)? "
        "the type difference between members? the claim test? caution/'when unsure keep'? "
        "your own default bias?\n"
        "2. Across all of them, what is the SINGLE strongest pull toward keeping that the "
        "prompt should counter?\n"
        "3. What concrete change — to the prompt, or to a signal you're given (e.g. the "
        "decoder's pre_class) — would have made you correctly merge these, WITHOUT making "
        "you over-merge genuine keeps?\n\n"
        "Be specific and self-critical. Quote the prompt where relevant.\n\n"
        "=== CLUSTERS YOU UNDER-MERGED ===\n" + json.dumps(payload, indent=2))

    out = call_sonnet(system, question, max_tokens=2600, model=CONSOLIDATION['model'])
    print(out['text'])


if __name__ == '__main__':
    main()
