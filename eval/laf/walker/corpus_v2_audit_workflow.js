// Corpus-v2 echo-audit workflow — Opus re-judges a blind sample of Sonnet's
// echo_mislabel verdicts (no prior verdict shown; identical rubric). The
// overturn rate (Opus → valid/ambiguous) is Sonnet's false-echo rate.
// args = [{bi, n}, ...] over audit_batch_<bi>.md files.
export const meta = {
  name: 'corpus-v2-echo-audit',
  description: 'Opus blind re-judge of sampled echo_mislabel verdicts to measure false-echo rate',
  phases: [{ title: 'Audit' }],
}
const DIR = '/Users/tpac/brain/eval/laf/walker'
const SCHEMA = {
  type: 'object',
  properties: { batch: { type: 'integer' }, n: { type: 'integer' }, wrote: { type: 'string' } },
  required: ['batch', 'n', 'wrote'],
}
const batches = typeof args === 'string' ? JSON.parse(args) : args
if (!Array.isArray(batches) || !batches.length) throw new Error('pass args = [{bi, n}, ...]')
phase('Audit')
const results = await parallel(batches.map(({ bi, n }) => () => {
  const id = String(bi).padStart(3, '0')
  return agent(
    `You are a corpus gold judge. Read ${DIR}/corpus_v2_judge_prompt.md (your full rubric — the ` +
    `Calibration rulings and Anaphora rule are binding) and ${DIR}/corpus_v2_batches/audit_batch_${bi}.md ` +
    `(the turns to judge). These turns have NOT been judged — judge each fresh per the rubric; there are ` +
    `${n} turns, no skips. Reason briefly per turn before its verdict; keep reasoning out of the final JSON. ` +
    `Then: (1) Write the complete verdict array — one object {key, verdict, stratum, gap, bridge, style_note} ` +
    `per turn in presentation order, keys copied EXACTLY from the turn headers, as pure JSON (a bare array, ` +
    `no wrapper) to ${DIR}/corpus_v2_batches/verdicts_audit_${id}.json; (2) return {batch: ${bi}, n: ` +
    `<count written>, wrote: "<path>"} via the structured output.`,
    { label: `audit:batch_${id}`, schema: SCHEMA, model: 'opus' }
  )
}))
const done = results.filter(Boolean)
log(`${done.length}/${batches.length} audit batches judged`)
return { judged: done.map(r => r.batch), failed: batches.length - done.length }
