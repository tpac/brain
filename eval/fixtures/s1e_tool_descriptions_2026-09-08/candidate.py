"""Generic tool-description candidate; no runtime mutation or model calls.

The frozen toolset owns argument shapes. This candidate changes only descriptive
strings and exports API/MCP forms plus a description-only diff for review.
load_candidate() gives an eval caller a distinct identity with its guide fixed.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / 'eval/fixtures/s1e_guide_freeze_2026-09-08'))
from frozen_arms import load_arm

VERSION = 'brain_tools_generic_v1_2026-09-08'

REVISION = (
    'Update an existing node by ID. Omitted fields keep their current values. '
    'A writable text field accepts its complete new value, a swap {old, new}, '
    'or a list of swaps applied in order. Each old span must match exactly once; '
    'a missing or ambiguous match rejects the revision. Full replacement discards '
    'text omitted from the new value. Other fields take bare values. '
    'reason is the required audit note; reasoning is a separate stored field. '
    'Immutable id, created_at and locked are skipped with warnings. '
    'connect_to can update existing edges or add edges to existing nodes. '
    'source_refs replaces the reference list when supplied; omission preserves it, '
    'and [] clears it. Results report changes, warnings and failures.'
)
RELATION = (
    'Specific relationship verb, such as supports, corrects, depends_on or implements. '
    'Use a specific verb rather than related/related_to or an empty value.'
)
WHY = (
    'Meaning of this particular relationship, used in retrieval. At least 30 characters '
    'explaining how these nodes connect, beyond restating their titles. '
    'Example: "The revised estimate accounts for the delay missing from the original schedule."'
)
TARGET = (
    'Existing node: its exact 8-character hex ID from a returned record or supplied context. '
    'For remember only, another node created in this batch can be named by its exact title, '
    'in any declaration order. New siblings take precedence over title matches; creating '
    'a duplicate title does not update the existing node. Hex-shaped values are treated '
    'as IDs. Unresolved edge targets are reported and skipped without failing node creation. '
    'On revise, only existing node IDs are accepted.'
)
CREATE_REFS = (
    'Optional links to the trace events that support this memory or provide useful scene '
    'visibility. Use existing 8-character hex trace IDs from source records or trace markers. '
    'Keep references selective; an abstraction without a particular source event may omit them.'
)
REVISE_REFS = (
    'Complete replacement of this node\'s trace-reference list. Omit to preserve existing '
    'references; [] deliberately clears them. Values are existing 8-character hex trace IDs.'
)

FIELDS = {
    'type': 'Node category, such as fact, decision, lesson, mechanism, correction or open; open vocabulary.',
    'title': 'Specific title used for identification and semantic retrieval.',
    'content': 'Memory content. In a revision, a complete value replaces the stored text; swaps change selected spans.',
    'confidence': 'Degree of support from 0.0 to 1.0. Hedged, contested or inferred claims use a value below 1.0.',
    'locked': 'Protection flag. Lock requests from automated provenance are demoted; anchor provenance can create locked nodes.',
    'emotion': 'Signed emotional intensity associated with the memory; pair with emotion_label.',
    'emotion_label': 'Name of the emotional register, such as satisfaction or frustration.',
    'evolution_status': 'Claim lifecycle: active, resolved, validated, confirmed, disproven or dismissed.',
    'source_turn_id': 'Originating message_stream ID for episode linkage.',
    'situation': 'Recall cue describing when this memory is relevant; used to match future situations.',
    'question': 'Question this memory answers, phrased as a retrieval query.',
    'event_time': 'When the remembered event happened, in ISO 8601; distinct from record creation time. Resolve relative dates against the source date and omit unsupported precision.',
    'reasoning': 'Stored basis for the claim: evidence, inference, uncertainty and what would change it. Separate from reason, the revision audit note.',
    'thought': 'Optional interpretation, hypothesis or connection beyond the stored account and its evidence. Can be revised independently and is returned beside the memory.',
    'their_raw_quote': 'Verbatim words from the person or source whose account is being recorded.',
    'my_raw_quote': 'Verbatim words from the agent\'s own contribution.',
    'correction_pattern': 'Underlying error or behavioral pattern identified by the correction.',
    'source_context': 'Context in which this memory originated.',
}

TOP = {
    'remember_batch': (
        'Create multiple memory nodes and return their IDs and per-node outcomes. '
        'Each node accepts the remember fields and optional connect_to edges. '
        'New-node edges are resolved after all siblings are created, so sibling '
        'declaration order does not matter. Node creation and edge creation have '
        'separate outcomes; a created node can have an unresolved edge.'
    ),
    'connect_batch': (
        'Create or update relationships between existing nodes. Each '
        '(source_id, target_id, relation) identifies one relationship: supplied fields '
        'update it, omitted fields preserve it, and an archived relationship is revived. '
        'Repeated calls do not increase weight unless a new weight is supplied. '
        'Each description must explain the specific relationship in at least 30 characters. '
        'Returns per-connection outcomes.'
    ),
    'brain_batch': (
        'Apply remember, revise, connect, disconnect, archive and absorb operations in '
        'one batch. Operations run in list order within a transaction; edges from new '
        'nodes are resolved after sibling creation. Returns per-operation results and '
        'separate connect_to failures. A successful batch response can contain rejected '
        'operations; use the returned outcomes to determine which changes took effect. '
        'Operation names select actions; relationship verbs belong in relation fields.\n\n'
        'For edges involving new nodes, use connect_to on remember with existing node IDs '
        'or exact sibling titles. New siblings take precedence on title collisions; a '
        'duplicate-title remember creates another node rather than revising the existing '
        'one. Do not also emit the same edge as a connect operation. connect_to on revise '
        'updates or adds this node\'s edges to existing IDs. Separate connect operations '
        'require two existing IDs and upsert the named relationship. Multiple relationships '
        'for a pair can use relations: [{relation, why}, ...].\n\n'
        'absorb folds absorbed_id into survivor_id and archives the absorbed node. '
        'Edges, source_refs, access counts and metadata transfer, but the survivor keeps '
        'its own content unless a content override is supplied. Include any needed '
        'absorbed content in that full replacement; content_edits is not supported on '
        'absorb. The absorbed node must be archivable; the survivor may be locked. '
        'A failed absorb does not commit a partial merge.'
    ),
    'revise_batch': (
        'Revise multiple nodes, with a node_id and audit reason for each item. '
        + REVISION + ' Each revision records its own history event.'
    ),
    'get_nodes': (
        'Fetch existing nodes by ID, including IDs obtained outside the current result set. '
        'The default view for up to 10 returned nodes includes full content, up to 8 edges '
        'with the total edge count, correction summaries and community references. '
        'Larger results use a scan view with 800 content characters and 5 edges per node. '
        'rich=true requests full content, all edges and full correction fields regardless '
        'of batch size. Missing IDs are reported. Recall results use the same default views.'
    ),
    'recall_batch': (
        'Search memory by meaning for several queries. Returns a ranked result group '
        'for each query, with a shared optional field filter and per-query result limit.'
    ),
}


def digest(value):
    if not isinstance(value, bytes):
        value = value.encode() if isinstance(value, str) else json.dumps(value, sort_keys=True).encode()
    return hashlib.sha256(value).hexdigest()


def mechanics(value):
    """Remove prose annotations, preserving fields literally named description."""
    if isinstance(value, dict):
        return {k: mechanics(v) for k, v in value.items()
                if not (k == 'description' and isinstance(v, str))}
    if isinstance(value, list):
        return [mechanics(v) for v in value]
    return value


def descriptions(value, path=''):
    if isinstance(value, dict):
        for k, v in value.items():
            child = path + '/' + k.replace('~', '~0').replace('/', '~1')
            if k == 'description' and isinstance(v, str):
                yield child, v
            else:
                yield from descriptions(v, child)
    elif isinstance(value, list):
        for i, v in enumerate(value):
            yield from descriptions(v, path + '/' + str(i))


def build(base):
    tools = copy.deepcopy(base)
    by_name = {t['name']: t for t in tools}
    for name, description in TOP.items():
        by_name[name]['description'] = description

    def props(name):
        return by_name[name]['input_schema']['properties']

    for name, array_key in [('remember_batch', 'nodes'), ('revise_batch', 'revisions')]:
        fields = props(name)[array_key]['items']['properties']
        for field, description in FIELDS.items():
            fields[field]['description'] = description
        fields['source_refs']['description'] = CREATE_REFS if name == 'remember_batch' else REVISE_REFS
    props('revise_batch')['revisions']['items']['properties']['locked']['description'] = (
        'Immutable on revision; changes to locked are skipped with a warning.')
    props('remember_batch')['nodes']['description'] = 'Nodes to create, each with type, title, content and optional fields and edges.'
    props('remember_batch')['nodes']['items']['properties']['connect_to']['description'] = (
        'Relationships from this new node to existing IDs or siblings created in this batch. '
        'Sibling titles resolve after all nodes are created. Use this field for new-node '
        'edges rather than a separate connect operation, and emit each relationship once.')
    props('remember_batch')['connect_to']['description'] = (
        'Apply the same relationship from every created node to one existing target; '
        'sibling targets are excluded. Node-level connect_to specifies individual edges.')
    revisions = props('revise_batch')['revisions']
    revisions['description'] = 'Revisions, each with an existing node_id, audit reason and fields to change.'
    revisions['items']['properties']['reason']['description'] = (
        'Required audit note explaining this revision; recorded in history, not stored on '
        'the node. Supply reasoning separately to change the node\'s evidence statement.')

    for tool in tools:
        for def_name in ['connect_to_item', 'revise_connect_to_item']:
            definition = tool['input_schema'].get('$defs', {}).get(def_name)
            if not definition:
                continue
            fields = definition['properties']
            fields['target']['description'] = TARGET if def_name == 'connect_to_item' else (
                'Exact 8-character hex ID of an existing node. Sibling titles are not accepted on revise.')
            fields['relation']['description'] = RELATION
            fields['why']['description'] = WHY
            if def_name == 'revise_connect_to_item':
                fields['relation']['description'] += (
                    ' A bare value identifies the relationship; required when the pair has '
                    'multiple relationships. A swap {old, new} renames it while preserving '
                    'weight and history. If no edge exists, the bare value names the new relation.')
                fields['why']['description'] += (
                    ' A bare value replaces the description; a swap {old, new} patches it. '
                    'A new edge requires a complete description.')

    operations = props('brain_batch')['operations']['items']['oneOf']
    op = {x['properties']['op']['const']: x for x in operations}
    op['remember']['description'] = (
        'Create a node with type, title and content. Also accepts the remember fields, '
        'including situation, reasoning, question, thought, event_time, their_raw_quote, '
        'my_raw_quote and source_refs. '
        'connect_to can address existing IDs or new sibling titles.')
    op['remember']['properties']['content']['description'] = FIELDS['content']
    op['revise']['description'] = REVISION
    op['revise']['properties']['reason']['description'] = revisions['items']['properties']['reason']['description']
    op['absorb']['description'] = (
        'Merge absorbed_id into survivor_id, then archive the absorbed node. '
        'Accepts complete field overrides; see the tool description for content preservation.')
    op['absorb']['properties']['content']['description'] = (
        'Complete survivor content after merging. Required to preserve any absorbed claim '
        'not already stated by the survivor; omitted content keeps only the survivor text.')
    connections = props('connect_batch')['connections']
    connections['description'] = 'Relationships to create or update between existing nodes.'
    connections['items']['properties']['relation']['description'] = RELATION
    connections['items']['properties']['description']['description'] = WHY
    op['connect']['properties']['description']['description'] = WHY
    props('get_nodes')['rich']['description'] = (
        'False uses the batch-size-dependent bounded view. True requests full content, '
        'all edges and full correction fields for every returned node.')
    props('recall_batch')['filter']['description'] = (
        'Field filter applied to every query: {field: {operator: value}}. '
        'Operators: exists, equals, in, contains, gte, lte.')
    return tools


def review(base, candidate):
    from jsonschema import Draft202012Validator
    assert mechanics(base) == mechanics(candidate), 'Non-description tool contract changed'
    before = dict(descriptions(base)); after = dict(descriptions(candidate))
    assert before.keys() == after.keys(), 'Description locations changed'
    validators = []
    for toolset in [base, candidate]:
        row = {}
        for t in toolset:
            Draft202012Validator.check_schema(t['input_schema'])
            row[t['name']] = Draft202012Validator(t['input_schema'])
        validators.append(row)
    # Real saved calls, including the rejected reason-less revision, must retain
    # the same schema verdict. No existing error is reclassified as a success.
    calls = 0; invalid = []
    for path in sorted((ROOT / 'eval/results/s1e_guide_sanity_2026-09-08').glob('*/repeat*/window*/calls.json')):
        for record in json.loads(path.read_text()):
            call = record['call']; name = call['tool']
            errors = [[(list(e.absolute_path), e.validator) for e in v[name].iter_errors(call['args'])]
                      for v in validators]
            assert errors[0] == errors[1], f'Schema verdict changed at {path}'
            calls += 1
            if errors[0]:
                invalid.append({'path':str(path.relative_to(ROOT)), 'tool':name, 'errors':errors[0]})
    assert calls == 19, f'Expected all 19 saved tool calls, found {calls}'
    return {'status':'passed', 'scope':'Offline structural and recorded-call compatibility; no behavioral eval.',
            'tools':len(candidate), 'description_entries':len(after),
            'changed_entries':sum(before[k] != after[k] for k in before),
            'description_chars_before':sum(map(len,before.values())),
            'description_chars_after':sum(map(len,after.values())),
            'serialized_chars_before':len(json.dumps(base,ensure_ascii=False)),
            'serialized_chars_after':len(json.dumps(candidate,ensure_ascii=False)),
            'saved_calls_checked':calls, 'existing_schema_invalid_calls':invalid,
            'non_description_contract_sha256':digest(mechanics(candidate))}


def load_candidate(base_arm='v3_titles'):
    """Use unchanged frozen guide/gist/settings with independently identified tools."""
    base = load_arm(base_arm)
    manifest = json.loads((HERE / 'manifest.json').read_text())
    for filename, expected in manifest['files'].items():
        if digest((HERE / filename).read_bytes()) != expected:
            raise ValueError(f'Tool candidate file changed: {filename}')
    candidate = json.loads((HERE / 'tools.api.json').read_text())
    if digest(base['tools']) != manifest['baseline_tools_sha256']:
        raise ValueError('Frozen baseline tools changed')
    if mechanics(base['tools']) != mechanics(candidate):
        raise ValueError('Candidate changed the non-description contract')
    identity = {'base_arm_sha256':base['arm_sha256'], 'tools_sha256':digest(candidate)}
    return {**base, 'arm_id':base_arm+'__'+VERSION,
            'base_arm_sha256':base['arm_sha256'], 'arm_sha256':digest(identity),
            'tools':candidate}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--write', action='store_true')
    args = parser.parse_args()
    base = load_arm('v3_titles')['tools']
    candidate = build(base)
    result = review(base, candidate)
    before = dict(descriptions(base)); after = dict(descriptions(candidate))
    edits = [{'path':p,'old':before[p],'new':after[p]} for p in before if before[p] != after[p]]
    mcp = [{'name':t['name'],'description':t['description'],'inputSchema':t['input_schema']} for t in candidate]
    documents = {'tools.api.json':candidate, 'tools.mcp.json':mcp,
                 'brain_batch.mcp.json':next(t for t in mcp if t['name']=='brain_batch'),
                 'description_edits.json':edits, 'offline_review.json':result}
    readable = ['# Generic brain tool descriptions — candidate v1', '',
                'Authoring/review artifact. No guide, journal or runtime change; no behavioral result.', '']
    for tool in candidate:
        readable.extend(['## '+tool['name'], '', tool['description'], ''])
        for path, value in descriptions(tool['input_schema']):
            readable.extend(['- `'+path+'`: '+value])
        readable.append('')
    files = {name:(json.dumps(value,indent=2,ensure_ascii=False)+'\n').encode() for name,value in documents.items()}
    files['descriptions.md'] = ('\n'.join(readable)+'\n').encode()
    if args.write:
        if any((HERE / name).exists() for name in [*files,'manifest.json']):
            raise FileExistsError('Candidate artifacts exist; use a new version for revisions')
        for name,data in files.items():
            (HERE / name).write_bytes(data)
        manifest = {'candidate':VERSION, 'status':'authored_for_review',
                    'baseline_tools_sha256':digest(base),
                    'files':{**{name:digest(data) for name,data in files.items()},
                             'candidate.py':digest(Path(__file__).read_bytes())}}
        (HERE / 'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    else:
        loaded = load_candidate()
        assert loaded['tools'] == candidate
        for name,data in files.items():
            assert (HERE / name).read_bytes() == data, f'Artifact drift: {name}'
    print(json.dumps(result,indent=2))


if __name__ == '__main__':
    main()
