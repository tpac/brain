# Per-miss reverse diagnosis — which lane reaches within 1 hop

misses vs today's LAF (gold rank>5, shipped λ=0.65): **345**

## A. Per-lane reach

| lane | organic | +1 hop | total | EXCLUSIVE | hop verbs (top) | hop desc-len med | seed rank med | gold age med | cue len med |
|---|---|---|---|---|---|---|---|---|---|
| maxsim | 184 | 26 | 210 | **63** | extends(6), grounds(3), depends_on(3) | 181 | 6 | 16d | 138 |
| sit | 76 | 58 | 134 | **12** | extends(9), implements(8), grounds(7) | 166 | 12 | 13d | 119 |
| idf | 41 | 39 | 80 | **13** | extends(6), instantiates(5), depends_on(4) | 166 | 6 | 12d | 136 |
| pick | 33 | 47 | 80 | **9** | extends(12), implements(6), instantiates(5) | 167 | 5 | 11d | 128 |
| enc | 28 | 12 | 40 | **2** | grounds(4), extends(2), validates(1) | 185 | 11 | 11d | 119 |
| mh | 58 | 33 | 91 | **9** | extends(8), instantiates(4), depends_on(3) | 173 | 10 | 7d | 153 |

## B. The UNTOUCHED class — no lane, no hop

- reached by >=1 lane (organic or hop): n=283 · gold age med 16d · cue len med 125 · deg med 10 · size med 1072
- **UNTOUCHED**: n=62 · gold age med 24d · cue len med 108 · deg med 7 · size med 1006

| class | share | types (top 5) | strata |
|---|---|---|---|
| reached | 82% | lesson(37), finding(37), decision(32), architecture(28), principle(23) | cue(116), window(107), session(60) |
| UNTOUCHED | 18% | lesson(9), community(7), finding(7), principle(7), architecture(6) | cue(34), window(19), session(9) |

### UNTOUCHED sample (for qualitative read)

**cue** (window, rank 257): `anything else worth doing in this session?`
- gold `aa720d3c` [finding, 24d old, deg 1, 1099 chars]: Parallel session decision model: observation-only from Anchor, fire-and-forget for independent work

**cue** (session, rank 65): `I want recent moves to become inline in S1Scribe and leveraged traces for that.we shouldnt remove it from frame until we implement S1 though, and i want to test that. I have another Session cleaning a lot of the frame.  After its done, I'll tell you and you ca`
- gold `0d5c11cc` [decision, 43d old, deg 0, 602 chars]: Surface prompt §5 dropped: pivot detection folds into §3

**cue** (cue, rank 59): `before we do that see what the other session said about our commits today: The main branch was advanced by other session.  Current issue is that we have lots of errors Ran a command, used a tool Investigating. First — let me check if my polling loop is still r`
- gold `abea7887` [architecture, 16d old, deg 4, 1047 chars]: Parallel session safety — five unlocked writers identified pre-fix

**cue** (cue, rank 204): `Stage 3 design, we can also queue the encoding if there is a multisession situation right?  Anyway, would really prefer not only not to use SQL, if possible not to use DAL but actually use higher functions that get access to sessions . yes SCRIBE_TAIL_IDLE_SEC`
- gold `e703a7c7` [architecture, 25d old, deg 0, 1078 chars]: DAL Phase 2: repository aggregate — hold 5 DALs on Brain, replace 68 construction sites

**cue** (cue, rank 111): `I would say 2, but i think there is a huge difference between dropping surfaced nodes, which logically makes sense just because there is no need in 2026 session lengths to restate a surface node and nodes that werent selected before. I'm not sure they all get `
- gold `5ee7d67f` [design, 103d old, deg 24, 1461 chars]: Fatigue todo: document where fatigue applies and fine-tune through traversal

**cue** (cue, rank 22): `I'll restart on another session, can you just update the documents that we have? We wrote something somewhere about that work.`
- gold `f60e4ccf` [lesson, 59d old, deg 8, 660 chars]: NEXT-WORK doc can be stale — Anchor flagged already-done work as outstanding

**cue** (cue, rank 23): `I absolutely want this but i want to make sure the architecture is right.  So we merge all data to a single scheme? Then is there a more solid traces mechanism?  I'm beginning to think a better design than adding more and more things to brain_batch its perhaps`
- gold `2890e908` [rule, 31d old, deg 25, 3439 chars]: Anchor rule: when designing brain mechanics, recall the brain first

**cue** (session, rank 222): `can you update documents? mark what we did what's still left. Perhaps share more context or audit more for next`
- gold `4d5da77e` [rule, 73d old, deg 10, 481 chars]: Rule: After architecture changes, audit ALL shipped files for stale references before releasing

**cue** (session, rank 26): `continue in an agent mode in eval and util you have an MVP to test our new approach`
- gold `2e6986a2` [community, 9d old, deg 173, 1866 chars]: Spread Activation and Recall Sampling: From Reach Quantification to Agentic Redesign

**cue** (cue, rank 1331): `no need. ill do it later. Thanks`
- gold `77b23cb9` [open, 14d old, deg 6, 419 chars]: MCP server + dashboard server need restart to pick up ec86aca DAEMON_DOWN fix

**cue** (cue, rank 177): `lol no. Look at prompt examples through out and see every example that has the wrong signature and correct it. show me before and after`
- gold `3a8e302e` [rule, 60d old, deg 4, 1869 chars]: After register_interaction on encoder prompts, run ./dev sync-prompts before committing

**cue** (window, rank 46): `I want a. i want the plugin to work and not find a work around`
- gold `cb6b6c08` [community, -1d old, deg 5, 883 chars]: Plugin Skills Commitment Discipline: From Silent Manifest Drift to Shipped-Means-Built

**cue** (window, rank 137): `you can restart`
- gold `b5f9fc40` [lesson, 57d old, deg 7, 707 chars]: Dashboard temporal display: daemon restart required to pick up new code

**cue** (cue, rank 1871): `yeah check that. Also let's have the decisions of what is filtered and what not in a clear contract?`
- gold `78ae43a1` [mechanism, 62d old, deg 17, 584 chars]: Contract system: contract.py is the single source of truth for node fields

