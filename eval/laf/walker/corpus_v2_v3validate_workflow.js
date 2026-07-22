// Corpus-v2 rubric-v3 validation — re-judge the 194-turn echo audit sample
// with SONNET under the sharpened rubric. Three-way compare downstream:
//   Sonnet-v1 (all echo) → Opus-audit → Sonnet-v3.
// Confirms the rubric was the bug IFF v3 restores Opus's flips AND holds on
// the echoes Opus agreed with. args = [{bi, n}, ...] over audit_batch_<bi>.md.
export const meta = {
  name: 'corpus-v2-v3-validate',
  description: 'Sonnet re-judges the echo audit sample under sharpened rubric v3',
  phases: [{ title: 'Validate' }],
}
const DIR = '/Users/tpac/brain/eval/laf/walker'
const SCHEMA = {
  type: 'object',
  properties: { batch: { type: 'integer' }, n: { type: 'integer' }, wrote: { type: 'string' } },
  required: ['batch', 'n', 'wrote'],
}
const batches = typeof args === 'string' ? JSON.parse(args) : args
if (!Array.isArray(batches) || !batches.length) throw new Error('pass args = [{bi, n}, ...]')
phase('Validate')
const results = await parallel(batches.map(({ bi, n }) => () => {
  const id = String(bi).padStart(3, '0')
  return agent(
    `You are a corpus gold judge. Read ${DIR}/corpus_v2_judge_prompt.md (your full rubric — the ` +
    `Calibration rulings, especially the BOUNDARY clarifications on pattern-class and same-session, and ` +
    `the surface-distance-is-a-MISS rule, are binding) and ${DIR}/corpus_v2_batches/audit_batch_${bi}.md ` +
    `(the turns to judge). These turns have NOT been judged — judge each fresh per the rubric; there are ` +
    `${n} turns, no skips. Reason briefly per turn before its verdict; keep reasoning out of the final JSON. ` +
    `Then: (1) Write the complete verdict array — one object {key, verdict, stratum, gap, bridge, style_note} ` +
    `per turn in presentation order, keys copied EXACTLY from the turn headers, as pure JSON (bare array, no ` +
    `wrapper) to ${DIR}/corpus_v2_batches/verdicts_v3_${id}.json; (2) return {batch: ${bi}, n: <count>, ` +
    `wrote: "<path>"} via the structured output.`,
    { label: `v3:batch_${id}`, schema: SCHEMA, model: 'sonnet' }
  )
}))
const done = results.filter(Boolean)
log(`${done.length}/${batches.length} v3 batches judged`)
return { judged: done.map(r => r.batch), failed: batches.length - done.length }
