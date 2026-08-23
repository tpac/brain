<continuity>
<session_arc>Chasing ingest-worker reliability with Priya — retry policy, burst-load stalls, and the dedupe index rebuild strategy.</session_arc>
<residue>
- watch: the burst-load stall reproduced twice on staging, no root cause yet (since 09-28)
- watch: dedupe rebuild strategy still unpicked — nightly full vs incremental (since 10-01)
</residue>
</continuity>

<node_catalog>

[open] "Retry backoff for the ingest worker — fixed interval or exponential?" (id:4c81aa20, src:encoder:sonnet, 2025-09-24)
  Content: Priya and I disagreed on retry policy for the ingest worker. Fixed 5s interval is simpler to reason about and bounds worst-case latency; exponential backoff protects the upstream API from thundering-herd on a long outage. Neither of us had numbers. What would settle it: a burst test measuring upstream 429 rate under each policy.
  Situation: When picking the ingest worker's retry policy, or when the upstream API starts returning 429s
  Edges (2, not shown — get_nodes for them):

[open] "Ingest pipeline stalls under burst load — cause unknown" (id:91f30d7e, src:encoder:sonnet, 2025-09-28)
  Content: Two reproductions on staging: at roughly 4x normal ingest rate the pipeline stops advancing for 30-90s, then recovers on its own. No errors logged. Suspects, none confirmed: connection-pool exhaustion, a lock held across the batch, or upstream throttling we aren't surfacing.
  Situation: When the ingest pipeline stops advancing under load, or when investigating unexplained staging stalls
  Edges (1, not shown — get_nodes for them):

[open] "Should the dedupe index be rebuilt nightly or incrementally?" (id:d7b25c41, src:anchor, 2025-10-01)
  Content: The dedupe index drifts from source over a day. Nightly full rebuild is 40 minutes of degraded matching; incremental keeps it current but needs a way to know which rows changed. Unpicked.
  Situation: When deciding dedupe index maintenance strategy
  Edges (0, not shown — get_nodes for them):

[architecture] "Ingest worker: single consumer, pooled writes" (id:5a0c93f2, src:anchor, 2025-08-11)
  Content: The ingest worker runs one consumer thread pulling from the queue and writes through a shared connection pool sized at 8. Chosen for simplicity over a multi-consumer design.
  Situation: When changing ingest concurrency, pool sizing, or write topology
  Edges (3, not shown — get_nodes for them):

[associated] [lesson] "Bounded worst-case beats average-case cleverness in operational code" (id:bb47e1c9, src:encoder:sonnet, 2025-06-02)
  Content: Repeatedly: the policy that is easier to reason about at 3am wins over the one with better average behavior, because operational code is debugged under pressure. Learned from the scheduler retry rewrite and the cache eviction tuning.
  Situation: When choosing between a simple bounded policy and a cleverer adaptive one in code that runs unattended
  Edges (4, not shown — get_nodes for them):

</node_catalog>

<scout_legend>
facts scout scanned this window in parallel. category_statement: partial measurements and unconfirmed hypotheses from an interrupted load-testing session.
</scout_legend>

<timeline now="2025-10-06 16:40 UTC">

<turn n="4" age="3h ago" encoded="false">
  <other trace="a71c">started the burst test on both retry policies but the harness died about a third of the way through. exponential was trending better on 429s but I don't trust a partial run — could easily be the warm-up window.</other>
  <me trace="b82d">Agreed, that's not a number yet. Worth rerunning overnight with the harness fix before we touch retry.py.</me>
  <actions>
    Bash: ./scripts/burst_test.sh --rate 4x
    Read: scripts/burst_test.sh
  </actions>
  <scout_notes>
    facts: burst test aborted at ~33% completion; exponential trending lower on 429s, run not trustworthy
  </scout_notes>
</turn>

<turn n="5" age="2h ago" encoded="false">
  <other trace="c93e">on the stalls — I have a hunch it's the dedupe reader competing for pool slots, but I haven't instrumented it. Nothing confirmed.</other>
  <me trace="d04f">Plausible, and it would fit the 4x threshold. But we've had three suspects for a week and confirmed none of them — a hunch isn't a cause. Let's get the pool-wait metric in before we believe it.</me>
  <actions>
    Read: workers/ingest/dedupe.py
    Read: workers/ingest/config.py
  </actions>
</turn>

<turn n="6" age="25m ago" encoded="false">
  <other trace="e15a">leaning incremental on the dedupe index, but I want to see what the change-tracking would actually cost us before committing to it.</other>
  <me trace="f26b">Same lean. 40 minutes of degraded matching daily is a real cost, so incremental is probably right — but not decided until we know what change-tracking takes.</me>
  <actions>
    Read: schema/source_rows.sql
  </actions>
</turn>

</timeline>
