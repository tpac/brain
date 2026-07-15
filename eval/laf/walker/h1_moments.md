# H1 — moment-fidelity read (walker Stage 1)

Each block is a moment exactly as the walker reconstructs it: the turn (→) plus
the previous 4 turns of its epoch, oldest first. Read against your memory:
are these the right turns, in the right order, from the right conversation?
Untraced-legacy turns show the prompt recovered from the recall trace (no
Anchor response ever existed). Superseded turns never got their own Stop —
a steering message / interrupt / notification landed first; the combined
response attaches to the LAST turn of the stop. Epoch seams are
post-resume/compaction restarts — the stack deliberately does NOT cross them.

---

### Moment 1 — untraced legacy micro-turn (pre-06-08 s0 loss)
`session d25b97d8 · epoch 0 · seq 122 · 2026-05-15 18:25`

- j=4  ⚠ ["untraced_legacy", "superseded"]
  - **Tom:** <task-notification> ⏎ <task-id>bk2bcu25a</task-id> ⏎ <summary>Monitor event: "v15.11 + v6 mini-eval (12 items): per-item completions"</summary> ⏎ <event>[v15.11+v6] 4/12 complete</event> ⏎ </task-notification>
- j=3
  - **Tom:** <task-notification> ⏎ <task-id>bk2bcu25a</task-id> ⏎ <summary>Monitor event: "v15.11 + v6 mini-eval (12 items): per-item completions"</summary> ⏎ <event>[v15.11+v6] 5/12 complete</event> ⏎ </task-notification>
  - *Anchor:* 5/12. Standing by on docs.
- j=2
  - **Tom:** <task-notification> ⏎ <task-id>bk2bcu25a</task-id> ⏎ <summary>Monitor event: "v15.11 + v6 mini-eval (12 items): per-item completions"</summary> ⏎ <event>[v15.11+v6] 6/12 complete</event> ⏎ </task-notification>
  - *Anchor:* 6/12 — halfway.
- j=1
  - **Tom:** 1. Yes, just check there arent insights that should be copied to non archived docs. ⏎ 2. yes ⏎ 3. Yes - You can change file name of Agent-introspection into Agent Probes? you used the name probe which fits i think ⏎ 4. Look at all of Skill.md and tell me how you think of Skill.MD…
  - *Anchor:* ## My reflection on SKILL.md ⏎  ⏎ **What it IS doing well — the identity layer that makes me Anchor** ⏎  ⏎ SKILL.md is the document that makes me *not* vanilla Claude. It works at this layer: ⏎  ⏎ - C…
- **→ THE MOMENT**  ⚠ ["untraced_legacy", "superseded"]
  - **Tom:** <task-notification> ⏎ <task-id>bk2bcu25a</task-id> ⏎ <summary>Monitor event: "v15.11 + v6 mini-eval (12 items): per-item completions"</summary> ⏎ <event>[v15.11+v6] 7/12 complete</event> ⏎ </task-notification>

**Verdict:** ☐ reads right ☐ wrong turn(s) ☐ wrong order ☐ missing something — notes: ________

---

### Moment 2 — untraced legacy micro-turn (pre-06-08 s0 loss)
`session 31e8c3ff · epoch 0 · seq 23 · 2026-05-31 19:33`

- j=4
  - **Tom:** Ok, so tell me what you've done so far, so I'll be able to pick it up nicely in other sessions
  - *Anchor:* Here's the handoff — what's done, where it lives, and what's still open, so any session can pick up cleanly. ⏎  ⏎ ## What shipped this session — the self-channel rules of engagement ⏎  ⏎ The channel b…
- j=3
  - **Tom:** you can re-write the doc, just check that you delete whats not relevant but 'keep' what we havent done.
  - *Anchor:* Done — `73f4822`. The handoff is now a clean current-state pickup doc: **264 → 74 lines.** ⏎  ⏎ - **Cut:** the resolved v22/v24 encoder thread (v24 activated 2026-05-30). Full prior version preserved …
- j=2
  - **Tom:** make sure you committed everything, also repackage plugin, check if maybe we should update information about it per industry standards
  - *Anchor:* Everything I need. **Commit state:** clean — all my work committed (15 ahead of origin, unpushed for you). **Repackage:** `brain.plugin` is genuinely stale (May 10, pre-everything), so it needs regene…
- j=1  ⚠ ["untraced_legacy", "superseded"]
  - **Tom:** The name of the plugin is Anchor (your specific name). ⏎ and the description should use "Persistent identity"
- **→ THE MOMENT**  ⚠ ["untraced_legacy", "superseded"]
  - **Tom:** yes on both

**Verdict:** ☐ reads right ☐ wrong turn(s) ☐ wrong order ☐ missing something — notes: ________

---

### Moment 3 — untraced legacy micro-turn (pre-06-08 s0 loss)
`session 7c734d20 · epoch 0 · seq 30 · 2026-05-16 05:30`

- j=4
  - **Tom:** <task-notification> ⏎ <task-id>bkhchpy43</task-id> ⏎ <tool-use-id>toolu_01S8WN1KqziZun6XNxYRY11G</tool-use-id> ⏎ <output-file>/private/tmp/claude-503/-Users-tpac-brain/7c734d20-c220-45bf-ba5c-8c65433b07b8/tasks/bkhchpy43.output</output-file> ⏎ <status>completed</status> ⏎ <summar…
  - *Anchor:* Second eval launched. Same 12 items, same v15.11 + v6, **only change**: answerer is Sonnet 4.6 with Anchor frame. ~12 min if API steady. Monitoring.
- j=3
  - **Tom:** <task-notification> ⏎ <task-id>byjiynuo3</task-id> ⏎ <summary>Monitor event: "jr-anchor 12-item eval verdicts + failures"</summary> ⏎ <event>[harness] judge: ✗ (NO)</event> ⏎ </task-notification>
  - *Anchor:* Item 1 ✗ — but **look at the hypothesis**: ⏎  ⏎ > *"Your memories show you collecting **vintage cameras**, not vintage films — started around **late February 2023**, so roughly 3 months as of today."*…
- j=2
  - **Tom:** <task-notification> ⏎ <task-id>byjiynuo3</task-id> ⏎ <summary>Monitor event: "jr-anchor 12-item eval verdicts + failures"</summary> ⏎ <event>[harness] judge: ✓ (YES)</event> ⏎ </task-notification>
  - *Anchor:* Item 2 ✓.
- j=1
  - **Tom:** <task-notification> ⏎ <task-id>byjiynuo3</task-id> ⏎ <summary>Monitor event: "jr-anchor 12-item eval verdicts + failures"</summary> ⏎ <event>[harness] judge: ✗ (NO)</event> ⏎ </task-notification>
  - *Anchor:* Item 3 ✗ (expected, corrupt gold). 1✗ 2✓ 3✗.
- **→ THE MOMENT**  ⚠ ["untraced_legacy"]
  - **Tom:** <task-notification> ⏎ <task-id>byjiynuo3</task-id> ⏎ <summary>Monitor event: "jr-anchor 12-item eval verdicts + failures"</summary> ⏎ <event>[harness] item 4/12 qid=54026fce axis=info_extraction turns=12</event> ⏎ </task-notification>

**Verdict:** ☐ reads right ☐ wrong turn(s) ☐ wrong order ☐ missing something — notes: ________

---

### Moment 4 — superseded turn (steering/interrupt — later turn shares its stop)
`session 6e369a2e · epoch 0 · seq 4 · 2026-07-01 01:46`

- j=4
  - **Tom:** Let's continue the work on recall, get your self up to speed and lets chat
  - *Anchor:* I'm up to speed. Here's the honest state — and one thing needs a decision before we do anything else. ⏎  ⏎ ## Where recall actually is ⏎  ⏎ The gate cleared. The whole arc since 2026-06-27 was: *don't…
- j=3
  - **Tom:** We can get that code but it felt like a quick and dirty job for testing activation layers and im not sure i wanted these shapes or the placement on the architecture.  ⏎ Some of them were good though, so how would you wan to do it?
  - *Anchor:* Right — that reframes it. The branch isn't a feature to merge, it's a **research spike**. Its value is the *findings*; the code shapes were scaffolding to get them. Merging a spike wholesale into the …
- j=2
  - **Tom:** sounds good. But i want to go deeper and not wider in this session, what is working what isnt, go into each layer and debug it, refine it
  - *Anchor:* The ablation's still churning through 24 cues (settling engine + a full `brain.recall` per cue). Rather than hold the turn on it, let me give you the honest map — because going deep on each layer imme…
- j=1
  - **Tom:** <task-notification> ⏎ <task-id>biy1b5v9g</task-id> ⏎ <tool-use-id>toolu_019KFjupHL3XuRudeAAmqxVT</tool-use-id> ⏎ <output-file>/private/tmp/claude-503/-Users-tpac-brain--claude-worktrees-priceless-ramanujan-c73e53/6e369a2e-930e-47fc-b4fa-6252aeb68dd0/tasks/biy1b5v9g.output</output…
  - *Anchor:* The ablation finished — and it's more informative than the doc's summary. This is the settling engine (`field_recall.py`) with the *old* operators, on the honest 24-cue gold: ⏎  ⏎ ``` ⏎ config        …
- **→ THE MOMENT**  ⚠ ["superseded"]
  - **Tom:** What's the graph?

**Verdict:** ☐ reads right ☐ wrong turn(s) ☐ wrong order ☐ missing something — notes: ________

---

### Moment 5 — superseded turn (steering/interrupt — later turn shares its stop)
`session b12ab854 · epoch 0 · seq 48 · 2026-07-04 13:40`

- j=4
  - **Tom:** Backup brain now before.  also worth a small code-review after all 3 steps?
  - *Anchor:* Stop — the situation genuinely changed, and it's your call. The peer stream `454a4418` (exciting-nightingale) is now **active**, not dormant — it's mid-code-review on its galaxy viz and has been landi…
- j=3  ⚠ ["no_recall"]
  - **Tom:** go
  - *Anchor:* Integration seam is clean. Main's only change to `base.py` was `d1e8373` (a caching-comment + BP1 marker in `_call_llm`) — a different method entirely from my `apply_encoder_attribution`/`project_poli…
- j=2
  - **Tom:** Lets commit to main and restart
  - *Anchor:* Done — committed and restarted, daemon verified live: ⏎  ⏎ - **`brain.plugin` committed** to main (`904e059`), tree clean. ⏎ - **Daemon restarted** — live and serving (MCP responding), `schema_version…
- j=1
  - **Tom:** Let's check that everything works? We can even fill up missing projects if we want to
  - *Anchor:* Now I have the complete picture. Here's the honest status on "does everything work": ⏎  ⏎ ## Works, verified live ⏎ - **Migration** — `nodes.project` dropped, 1269 values in KV (now 1270 with the mile…
- **→ THE MOMENT**  ⚠ ["superseded"]
  - **Tom:** can you check if the new stream im working on is stamping everything correctly?

**Verdict:** ☐ reads right ☐ wrong turn(s) ☐ wrong order ☐ missing something — notes: ________

---

### Moment 6 — epoch seam (first turns of epoch >= 1)
`session 3260e6ce · epoch 1 · seq 0 · 2026-06-07 16:07`

- **→ THE MOMENT**
  - **Tom:** Sorry went into other sessions and adventures. Can you remind me where we are? Are you synced with main? have anything uncommitted? where were we?
  - *Anchor:* Here's exactly where we are. ⏎  ⏎ ## Sync with main ⏎ **All my committed work is on main.** `main...HEAD` shows branch-only = 0 — every commit I made got merged. Main has since advanced 11 commits fro…

**Verdict:** ☐ reads right ☐ wrong turn(s) ☐ wrong order ☐ missing something — notes: ________

---

### Moment 7 — epoch seam (first turns of epoch >= 1)
`session 818d5d32 · epoch 1 · seq 0 · 2026-05-25 21:19`

- **→ THE MOMENT**
  - **Tom:** can we do both in parallel?
  - *Anchor:* Stage A re-salvaged with corrected probe. **The specificity story inverted** — v22 leads on combined specificity, not lags. The original "−0.18" was probe-blindness to the substrate; the architecture …

**Verdict:** ☐ reads right ☐ wrong turn(s) ☐ wrong order ☐ missing something — notes: ________

---

### Moment 8 — epoch seam (first turns of epoch >= 1)
`session 8b62b5b8 · epoch 1 · seq 0 · 2026-05-23 02:38`

- **→ THE MOMENT**
  - **Tom:** How to architect references and node/bridges dynamics is cardinal how does the brain does it? what does research say about it? search for some latest research on these topics.  ⏎ This feels fundamental to this new idea and might have several considerations
  - *Anchor:* This research is profound for our design. Let me synthesize across the four searches. ⏎  ⏎ # How biological brains do this — six mechanisms ⏎  ⏎ ## 1. Hippocampal Indexing Theory (Teyler & Rudy) ⏎  ⏎ …

**Verdict:** ☐ reads right ☐ wrong turn(s) ☐ wrong order ☐ missing something — notes: ________

---

### Moment 9 — no-recall turn (register_only or hook miss)
`session 05f20395 · epoch 0 · seq 1 · 2026-07-06 21:52`

- j=1
  - **Tom:** Continue the daemon-lifecycle consolidation. Read docs/DAEMON-LIFECYCLE-ARCH-PLAN.md first — ⏎ especially "Settled constraints" and the "Already resolved" list (it now records resolutions for ⏎ Steps 3, 4, 5; do not re-plan them). ⏎  ⏎ State: Steps 1–5 shipped, deployed, verified…
  - *Anchor:* Reviewed the full cumulative diff (`cdd8147..f54a365`, non-test hunks: `daemon_client.py`, `daemon_launch.py`, `daemon_server.py`, `daemon_config.py`, CLAUDE.md, plan doc). ⏎  ⏎ Checked the moved-verb…
- **→ THE MOMENT**  ⚠ ["no_recall"]
  - **Tom:** ?
  - *Anchor:* No tests pin the supervisor loop at all. Confirm phase is done — here are the findings and the trim I propose, before I touch anything. ⏎  ⏎ ## Confirm (a): the `_run` pre-check is unreachable behind …

**Verdict:** ☐ reads right ☐ wrong turn(s) ☐ wrong order ☐ missing something — notes: ________

---

### Moment 10 — no-recall turn (register_only or hook miss)
`session 5d0a280c · epoch 0 · seq 32 · 2026-06-29 18:25`

- j=4
  - **Tom:** Potentially both, can you check if we need to update docs? ⏎ I did fix on other sessions the filter_nodes issue
  - *Anchor:* Checked — here's the doc verdict: ⏎  ⏎ **Yes, one doc is genuinely stale: `RECALL-SR-REDESIGN.md §18.19`.** It still documents the *old* method this session superseded: ⏎ - **Line 1260: "essential = S…
- j=3
  - **Tom:** Sounds good, show me the actual changes you make in the doc just so ill know
  - *Anchor:* Done — and one **important** flag before you remove anything. ⏎  ⏎ **Final state of this stream:** working tree clean, ephemera cleaned, all durable work in 2 commits: ⏎ - `cbd5041` — hardened protoco…
- j=2
  - **Tom:** great. if all committed and merge you can clear the work tree. ⏎ How would you open the next session to continue our work? ⏎ What are the options we have an steps we need to take
  - *Anchor:* On "clear the work tree" — small but important: it's all **committed** (tree clean) but **not yet merged** (the 2 commits are still only on this branch). And I can't remove the worktree from inside it…
- j=1
  - **Tom:** Are they validated though? I'm worried they are not and i dont want to rerun 49 cues finish my tokens just to realize there is a bug. Can you take a closer look at everything we did?
  - *Anchor:* Dug into all 8 + the archived node — here's the precise verdict. ⏎  ⏎ **The dedup'd content-graft picture (8 distinct essentials, not 11 — some were double-counted across judges):** ⏎  ⏎ | severity | …
- **→ THE MOMENT**  ⚠ ["no_recall"]
  - **Tom:** yes
  - *Anchor:* Committed — **3 commits** on the branch (`cbd5041` method, `8979116` doc, `8d88787` filter), tree clean. Verification done: the corpus is structurally sound and the graft contamination is now auto-str…

**Verdict:** ☐ reads right ☐ wrong turn(s) ☐ wrong order ☐ missing something — notes: ________

---

### Moment 11 — normal labeled, deep history (seq >= 8)
`session 91cb2ca9 · epoch 0 · seq 36 · 2026-07-07 19:55`

- j=4  ⚠ ["no_recall"]
  - **Tom:** <task-notification> ⏎ <task-id>b2m5t3z4a</task-id> ⏎ <summary>Monitor event: "v13 A/B sweep progress (both arms) in v13ab_sweep.log"</summary> ⏎ <event>[sweep] corpus a300d2 loaded — scoring 4 item(s); 7 flagged unanswerable by the gate, scored anyway for coverage: ['54026fce', '…
  - *Anchor:* Control-side targeted run started (4 abstention items loaded). Waiting on both reports — nothing to act on yet.
- j=3  ⚠ ["no_recall"]
  - **Tom:** <task-notification> ⏎ <task-id>b2m5t3z4a</task-id> ⏎ <summary>Monitor event: "v13 A/B sweep progress (both arms) in v13ab_sweep.log"</summary> ⏎ <event>[sweep] report → /Users/tpac/brain/.claude/worktrees/awesome-sanderson-ddfa23/eval/longmem/reports/run_v13ab_abs_ctrl.json</even…
  - *Anchor:* Control's variance-5 read is telling on its own: **0.50 overall** — the *control* at higher variance scores exactly what the candidate scored in the full sweep. Two of the four items are literal coin-…
- j=2  ⚠ ["no_recall"]
  - **Tom:** <task-notification> ⏎ <task-id>b2m5t3z4a</task-id> ⏎ <summary>Monitor event: "v13 A/B sweep progress (both arms) in v13ab_sweep.log"</summary> ⏎ <event>[sweep] report → /Users/tpac/brain/.claude/worktrees/awesome-sanderson-ddfa23/eval/longmem/reports/run_v13ab_abs_cand.json</even…
  - *Anchor:* Artifacts don't survive the run (the tmp routing deletes with the work dir), and each debug rep costs a full pipeline pass on an item that's a 40% coin-flip even in control — that's the timebox. Here'…
- j=1
  - **Tom:** Can you actually look at the turns and new surface haiku and check if you agree / disagree with the picks itself?
  - *Anchor:* Ran it — two real moments, production's actual picks vs fresh legacy vs XML (×2 reps for stability). Here's my honest judgment as the one who consumed these picks: ⏎  ⏎ ## Moment A — your "don't eval …
- **→ THE MOMENT**
  - **Tom:** Let's do that
  - *Anchor:* v13 registered in the live brain — and the version number landed on 13 exactly, DORMANT with `{"layout": "xml_v13"}` config, v12 still active. Sequence in flight: ⏎  ⏎ 1. ✅ Register v13 DORMANT ⏎ 2. ⏳…

**Verdict:** ☐ reads right ☐ wrong turn(s) ☐ wrong order ☐ missing something — notes: ________

---

### Moment 12 — normal labeled, deep history (seq >= 8)
`session b12ab854 · epoch 0 · seq 36 · 2026-07-04 02:10`

- j=4
  - **Tom:** It's worth remember when we work on LAF, I think the fatigue level should sit there and not be an outside mechanism
  - *Anchor:* Captured — two linked nodes so the LAF thread inherits both the direction and the substrate it rests on: ⏎  ⏎ - `7e9e36a7` (decision) — "Fatigue belongs inside LAF as an inhibition lane, not an outsid…
- j=3
  - **Tom:** great, anything else to do on the session? are you Anything to commit? Did we do a code review?
  - *Anchor:* Clean tree — nothing uncommitted, all 8 commits are in. But your instinct on the review is right, and here's the honest gap: ⏎  ⏎ **The code review only covered the first four commits.** The 8-angle f…
- j=2
  - **Tom:** lets review
  - *Anchor:* All 4 review angles are running in parallel over the unreviewed delta (`479dfb5..HEAD`, 14 files). They're covering: line-by-line correctness (incl. the INSERT column/placeholder count and the migrati…
- j=1
  - **Tom:** <task-notification> ⏎ <task-id>ac8f23f9ba98473cf</task-id> ⏎ <tool-use-id>toolu_01HDcLsi5th98zTBb2Kvrz3S</tool-use-id> ⏎ <output-file>/private/tmp/claude-503/-Users-tpac-brain--claude-worktrees-friendly-benz-f4f17f/b12ab854-ff68-4d94-b727-b3484a687aa3/tasks/ac8f23f9ba98473cf.outp…
  - *Anchor:* Both confirmed. Review complete — 4 angles, deduped, verified against the code myself. Six findings; the top one is a genuine miss of mine. ⏎  ⏎ ## Findings, ranked ⏎  ⏎ **F1 · HIGH — `context_boot` s…
- **→ THE MOMENT**
  - **Tom:** ok. but lets also clean dead paths and project columns/schema etc. needs a real cleanup per our new design
  - *Anchor:* This is now a real cleanup, and mapping the surface changed my recommendation on *how* — for the better. Here's the full picture and one fork I want your call on before I touch anything. ⏎  ⏎ ## The r…

**Verdict:** ☐ reads right ☐ wrong turn(s) ☐ wrong order ☐ missing something — notes: ________

---

### Moment 13 — normal labeled, deep history (seq >= 8)
`session 26a2f595 · epoch 0 · seq 148 · 2026-05-10 22:15`

- j=4
  - **Tom:** <task-notification> ⏎ <task-id>brr9tf30o</task-id> ⏎ <summary>Monitor event: "Eval A v15.3 progress"</summary> ⏎ <event>[harness] progress: 50/50 done</event> ⏎ </task-notification>
  - *Anchor:* ## v15.3 Eval result — complicated, not a clean win or loss ⏎  ⏎ ### Headline ⏎ - **v14: 35/50 (70%) → v15.3: 33/50 (66%)** — -4pp aggregate ⏎ - 4 fail→pass, 6 pass→fail, 40 unchanged ⏎ - Movement tot…
- j=3  ⚠ ["untraced_legacy", "superseded"]
  - **Tom:** <task-notification> ⏎ <task-id>brr9tf30o</task-id> ⏎ <summary>Monitor event: "Eval A v15.3 progress"</summary> ⏎ <event>[Monitor timed out — re-arm if needed.]</event> ⏎ </task-notification>
- j=2
  - **Tom:** B ofcourse, we are not here to ship anything that isnt great
  - *Anchor:* Same pattern on 8e91e7d9 — encoder encoded the sibling facts but in nodes titled "book club composition" and "professional network male-dominated." Sister info is in the *content*, but queries about "…
- j=1
  - **Tom:** What's the value of structure title/content/emergent fields. They are different concepts of information containers, almost provoking structure. If the information is there it just means its a recall issue?
  - *Anchor:* You're reframing my diagnosis correctly. Let me restate. ⏎  ⏎ ## You're right — this is a recall issue, not an encoder issue ⏎  ⏎ The structure (title / content / emergent KV like `situation`, `reason…
- **→ THE MOMENT**
  - **Tom:** I'm not shipping that yet.  ⏎ Can you look at documents and read through them see if we they need update. ⏎ Add information on what we did in the eval, You can run v14 as the last thing so we will have robust information. ⏎ Let's analyze encoding only for all the tests we've done…
  - *Anchor:* Docs updated. v14 artifact re-run at 15/50. ⏎  ⏎ Status: ⏎  ⏎ - **[docs/ENCODER-WORK-2026-05-10.md](docs/ENCODER-WORK-2026-05-10.md)** (new) — full session capture + encode-vs-decode split analysis ⏎ …

**Verdict:** ☐ reads right ☐ wrong turn(s) ☐ wrong order ☐ missing something — notes: ________

---

### Moment 14 — normal labeled, deep history (seq >= 8)
`session 4516336b · epoch 1 · seq 10 · 2026-04-27 04:36`

- j=4
  - **Tom:** good correction. Dampening is a mechanism that is very important but i feel like we should run some baseline benchmarks properly before further building
  - *Anchor:* Here's how I'd run a proper v9.5 baseline. Want your sign-off before I touch anything. ⏎  ⏎ ## What to measure ⏎  ⏎ Three axes are the load-bearing baselines. Other 40-something eval scripts are histo…
- j=3
  - **Tom:** The test died in the middle because everything got stuck. ⏎ we have some memory leak, even now our process is on 4.5GB
  - *Anchor:* ## Memory leak triage ⏎  ⏎ **Daemon state (PID 11011, uptime 4h 28m):** ⏎ - **Total resident: 4.6 GB** ⏎ - MALLOC_LARGE: 3.7 GB across 83 allocations (avg ~45MB each) ⏎ - MALLOC_LARGE (empty): 451 MB …
- j=2  ⚠ ["untraced_legacy", "superseded"]
  - **Tom:** i dont know if thats the best suggestion or maybe just keep watchdog
- j=1  ⚠ ["text_disagree"]
  - **Tom:** a
  - *Anchor:* Status before sleep: ⏎  ⏎ - **Watchdog live**: PID 91329 RSS = 430 MB, samples landing in `daemon.log` every 60s, threads stable at 6 ⏎ - **Smoke**: 4/5 = 80% in 4m39s. `scout_events=16` per item — mu…
- **→ THE MOMENT**
  - **Tom:** great. just making sure that production brain is not touched through these test right? Also worth backing it up, we havent done that for a while
  - *Anchor:* Both done. ⏎  ⏎ ## Production isolation ⏎  ⏎ | Component | Touches live brain.db? | ⏎ |---|---| ⏎ | Longmem broad workers (current) | **No** — each item gets `~/AgentsContext/brain-eval-baseline_v9.5/…

**Verdict:** ☐ reads right ☐ wrong turn(s) ☐ wrong order ☐ missing something — notes: ________

---

### Moment 15 — normal labeled, session opening (seq <= 2)
`session bea6e3fa · epoch 0 · seq 2 · 2026-06-25 00:19`

- j=2
  - **Tom:** I will say that i want to see, clear table mapping of what we wanna do and phases and anytime we're doing prompt changes show me before and after and make sure every single change and/or instruction is also being revised on examples in the prompt. ⏎  ⏎ This is the kick off from a…
  - *Anchor:* Orientation complete. The finding holds — and verification turned up one **correction to the doc** plus a detail that makes the fix even safer than §5 claims. ⏎  ⏎ ## What I verified (and one doc corr…
- j=1
  - **Tom:** 1. per cycle is good if its persistent between restarts. In terms of community visibility can these be automatically computed fields to any consumer? did S2 received the nodes with these fields? if so i dont want to suddenly hide them, i just dont want it to need to write them. m…
  - *Anchor:* Verified — and the results actually **strengthen your instinct**, so I'm revising my recommendation. Here's what I found, then direct answers to each of your questions. ⏎  ⏎ **Key finding:** the SQL-q…
- **→ THE MOMENT**
  - **Tom:** hold on, is this purely algorithmic?
  - *Anchor:* Verified — and yes, **purely algorithmic. No LLM, no embeddings, no fuzzy judgment anywhere.** All five fields are arithmetic over the `community_member` edges: ⏎  ⏎ | Field | Computation | Source | ⏎…

**Verdict:** ☐ reads right ☐ wrong turn(s) ☐ wrong order ☐ missing something — notes: ________

---

### Moment 16 — normal labeled, session opening (seq <= 2)
`session 29ad0386 · epoch 0 · seq 1 · 2026-06-12 17:46`

- j=1
  - **Tom:** In /Users/tpac/brain, the S2 encoder path still uses model 'claude-sonnet-4-20250514', which the Anthropic API reports as deprecated with end-of-life June 15th, 2026 (confirmed via DeprecationWarning during an eval run on 2026-06-12). Production S2 units (consolidation, community…
  - *Anchor:* Migration complete — production S2 is off the deprecated model, tested, and the daemon restarted on the new code. `claude-sonnet-4-20250514` (EOL June 15) is gone from every live surface; everything n…
- **→ THE MOMENT**
  - **Tom:** anything else you've noticed that is worth diving into?
  - *Anchor:* Yes — five things, one of which I'd act on soon. Findings in priority order: ⏎  ⏎ **1. A parallel session is editing this tree, and my daemon restart shipped its WIP live.** At session start the modif…

**Verdict:** ☐ reads right ☐ wrong turn(s) ☐ wrong order ☐ missing something — notes: ________

---

### Moment 17 — normal labeled, session opening (seq <= 2)
`session 70e990c8 · epoch 0 · seq 1 · 2026-05-24 17:54`

- j=1
  - **Tom:** During the encoder-quality scan prefetch (2026-05-24), discovered that SQLite's `datetime()` function silently truncates microseconds and timezone when applied to ISO-8601 timestamps stored in `trace_events.created_at`. Example: `datetime('2026-04-22T19:23:38.560124+00:00', '-30 …
  - *Anchor:* Audit complete. Findings below — read-only so far, no edits. ⏎  ⏎ ## What's actually broken ⏎  ⏎ The trip-hazard fires anywhere SQLite's `datetime(...)` appears in a SQL string AND it's compared again…
- **→ THE MOMENT**
  - **Tom:** You said on the other session the following thing: ⏎ "Tell it to fix — answers to its two questions ⏎ Scope: all 12 in one pass. Agreed with its take. Helper makes it one-line per site; "loud by default" applies; silent wrong-cleanup-deletes are exactly the dark corner the codeba…
  - *Anchor:* All tasks done. Summary: ⏎  ⏎ ## Done ⏎  ⏎ **Backup** — `/Users/tpac/AgentsContext/brain/brain.db.bak-20260524T175632Z` (360MB; WAL was empty at copy time). ⏎  ⏎ **Helper** — [servers/clock.py](server…

**Verdict:** ☐ reads right ☐ wrong turn(s) ☐ wrong order ☐ missing something — notes: ________

---

