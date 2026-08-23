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
facts scout scanned this window in parallel. category_statement: measurements and configuration values from a load-testing session.
</scout_legend>

<timeline now="2025-10-06 16:40 UTC">

<turn n="4" age="3h ago" encoded="false">
  <other trace="a71c">ran the burst test overnight on both retry policies. exponential gave us 0.3% upstream 429s, fixed 5s gave 11%. so exponential wins and it's not close — let's ship it.</other>
  <me trace="b82d">Agreed, that settles it — 11% vs 0.3% is a different regime, not a tuning difference. Shipping exponential with a 30s cap.</me>
  <actions>
    Edit: workers/ingest/retry.py
    Bash: pytest tests/test_retry_policy.py
  </actions>
  <scout_notes>
    facts: exponential backoff → 0.3% upstream 429 rate; fixed 5s interval → 11% under the same burst test
  </scout_notes>
</turn>

<turn n="5" age="2h ago" encoded="false">
  <other trace="c93e">also I think I found part of the stall. pool was sized 8 and we had 8 in-flight writes plus the dedupe reader, so the reader waited. bumped to 16 and one of the two stall shapes went away completely.</other>
  <me trace="d04f">That accounts for the 30s stalls. But the 90s ones reproduced again this morning at 16 — so there's a second thing. The pool was one cause, not the cause.</me>
  <actions>
    Edit: workers/ingest/config.py
    Bash: ./scripts/burst_test.sh --rate 4x
    Read: workers/ingest/dedupe.py
  </actions>
  <scout_notes>
    facts: connection pool 8 → 16; the 30s stall shape resolved, the 90s shape still reproduces
  </scout_notes>
</turn>

<turn n="6" age="25m ago" encoded="false">
  <other trace="e15a">on the dedupe index — let's go incremental. nightly is 40 minutes of bad matching every single day and we can't keep paying that.</other>
  <me trace="f26b">Incremental it is. Though that needs something we don't have — a way to know which source rows changed since the last pass. There's no watermark or change-feed on that table today, so someone has to design that before incremental can actually run.</me>
  <actions>
    Read: schema/source_rows.sql
  </actions>
</turn>

</timeline>
