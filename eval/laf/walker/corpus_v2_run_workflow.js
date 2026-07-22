// Corpus-v2 full-run judge workflow (protocol node 25cea181).
// Invoke with args = [{bi: <batch index>, n: <turn count>}, ...] — the main
// loop passes a chunk of not-yet-judged batches (corpus_v2_collect.py tells
// it which are missing), runs this, then collects + checks spend before the
// next chunk. Each judge agent WRITES verdicts_batch_NNN.json itself — the
// per-batch checkpoint survives a dead workflow.
export const meta = {
  name: 'corpus-v2-judge-run',
  description: 'Judge walker gold turns per corpus-v2 rubric, one agent per 25-turn batch',
  phases: [{ title: 'Judge', detail: 'one agent per batch file' }],
}
const DIR = '/Users/tpac/brain/eval/laf/walker'
const SCHEMA = {
  type: 'object',
  properties: {
    batch: { type: 'integer' },
    n: { type: 'integer' },
    wrote: { type: 'string' },
  },
  required: ['batch', 'n', 'wrote'],
}
if (!Array.isArray(args) || !args.length) {
  throw new Error('pass args = [{bi, n}, ...]')
}
phase('Judge')
const results = await parallel(args.map(({ bi, n }) => () => {
  const id = String(bi).padStart(3, '0')
  return agent(
    `You are a corpus gold judge. Read ${DIR}/corpus_v2_judge_prompt.md (your full rubric — the ` +
    `Anaphora rule under STRATUM is binding) and ${DIR}/corpus_v2_batches/batch_${id}.md (the turns ` +
    `to judge). Judge EVERY turn per the rubric — there are ${n} turns; no skips. Reason briefly per ` +
    `turn before writing its verdict; keep reasoning out of the final JSON. Then: (1) Write the ` +
    `complete verdict array — one object {key, verdict, stratum, gap, bridge, style_note} per turn, ` +
    `in presentation order, keys copied EXACTLY from the turn headers — as pure JSON (a bare array, ` +
    `no wrapper) to ${DIR}/corpus_v2_batches/verdicts_batch_${id}.json; (2) return {batch: ${bi}, ` +
    `n: <number of verdicts written>, wrote: "<the file path>"} via the structured output.`,
    { label: `judge:batch_${id}`, schema: SCHEMA }
  )
}))
const done = results.filter(Boolean)
log(`${done.length}/${args.length} batches judged`)
return { judged: done.map(r => r.batch), failed: args.length - done.length }
