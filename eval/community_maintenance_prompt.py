"""Exact-anchor candidate transform for the S2 community maintenance repair.

Use the same transform for the frozen control/candidate probe and landing.
"""


def make_candidate(prompt):
    edits = [
        ('Every proposal gets a decision: accept (tool call) or reject (just '
         "don't act — it's recorded for you).",
         'Every proposal gets a decision: accept (tool call) or reject (just '
         "don't act — it's recorded for you). On acceptance, maintain the "
         "community's current account: a member edge and the meaning-changing "
         'revisions it calls for are one action.'),
        ('- `community_members` — ALL member IDs as "id: title" pairs\n', ''),
        ('## ADD TO EXISTING\n\nConnect the new member to the community.',
         'The creation-time `community_members` value is an orphan-recovery seed. '
         'Current membership comes from live edges. Leave the seed unchanged '
         'after creation: missing or stale entries in it are expected, not '
         'membership defects. Do not rewrite it or ask the healer to synchronize '
         'it, even if old review notes request that. Judge actual missing '
         'membership from the live edges shown in DECISION EVIDENCE.\n\n'
         '## ADD TO EXISTING\n\nConnect the new member to the community. '
         'Compare its evidence with the community\'s current account. When it '
         'changes the latest development, revise `community_latest_development` '
         'in the same batch; an older historical addition leaves the current '
         'frontier intact. Repair claims in `content`, `situation`, or `question` '
         'that this evidence makes stale. Put a useful new interpretation in '
         '`thought`. These judgment fields belong to this community encoder.'),
        ('  {op: "connect", source_id: "comm1234", target_id: "node5678", '
         'relation: "community_member", weight: 0.3}\n]})\n```',
         '  {op: "connect", source_id: "comm1234", target_id: "node5678", '
         'relation: "community_member", weight: 0.3},\n'
         '  {op: "revise", node_id: "comm1234",\n'
         '   reason: "The new evidence advances this community from a pending '
         'fix to a validated result",\n'
         '   community_latest_development: "The fix passed its validation '
         '(id:node5678); the remaining work is rollout",\n'
         '   content_edits: [{old: "The fix is awaiting validation.", '
         'new: "The fix passed validation (id:node5678); rollout remains."}]}\n'
         ']})\n```\n\nThis example advances the current account. For an older '
         'historical addition that leaves the account current, connect it '
         'without this revise.'),
        ('Partial action beats complete analysis. A community created with decent '
         'content and missing polish will be improved by the healer. A community '
         'that was never created because you wanted more inspection is permanently lost.',
         'Act on the evidence available. The community encoder maintains its '
         'communities\' meaning; the healer fills missing question, situation, and '
         'reasoning fields. Use the review for unresolved observations; a clean '
         'community needs no open maintenance note.'),
    ]
    for old, new in edits:
        assert prompt.count(old) == 1, 'community maintenance anchor drift: %r' % old
        prompt = prompt.replace(old, new)
    return prompt
