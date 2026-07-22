// Corpus-v2 chunk-B — re-judge the 753 v2-half echoes (batches 0-51) under
// rubric v3 on SONNET. Writes verdicts_rejudge_<id>.json (distinct from the
// v1/v2 verdicts_batch_*.json so nothing is clobbered until the collector
// merges). args = [{bi, n}, ...] over rejudge_batch_<bi>.md.
export const meta = {
  name: 'corpus-v2-rejudge',
  description: 'Sonnet re-judges v2-half echoes under sharpened rubric v3',
  phases: [{ title: 'Rejudge' }],
}
const DIR = '/Users/tpac/brain/eval/laf/walker'
const SCHEMA = {
  type: 'object',
  properties: { batch: { type: 'integer' }, n: { type: 'integer' }, wrote: { type: 'string' } },
  required: ['batch', 'n', 'wrote'],
}
const batches = typeof args === 'string' ? JSON.parse(args) : args
if (!Array.isArray(batches) || !batches.length) throw new Error('pass args = [{bi, n}, ...]')
phase('Rejudge')
const results = await parallel(batches.map(({ bi, n }) => () => {
  const id = String(bi).padStart(3, '0')
  return agent(
    `You are a corpus gold judge. Read ${DIR}/corpus_v2_judge_prompt.md (your full rubric — the ` +
    `Calibration rulings, the BOUNDARY clarifications on pattern-class and same-session, and the ` +
    `surface-distance-is-a-MISS rule are binding) and ${DIR}/corpus_v2_batches/rejudge_batch_${bi}.md ` +
    `(the turns to judge). Judge each fresh per the rubric; there are ${n} turns, no skips. Reason briefly ` +
    `per turn before its verdict; keep reasoning out of the final JSON. Then: (1) Write the complete verdict ` +
    `array — one object {key, verdict, stratum, gap, bridge, style_note} per turn in presentation order, ` +
    `keys copied EXACTLY from the turn headers, as pure JSON (bare array, no wrapper) to ` +
    `${DIR}/corpus_v2_batches/verdicts_rejudge_${id}.json; (2) return {batch: ${bi}, n: <count>, ` +
    `wrote: "<path>"} via the structured output.`,
    { label: `rejudge:batch_${id}`, schema: SCHEMA, model: 'sonnet' }
  )
}))
const done = results.filter(Boolean)
log(`${done.length}/${batches.length} rejudge batches done`)
return { judged: done.map(r => r.batch), failed: batches.length - done.length }
