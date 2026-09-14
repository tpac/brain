# V3.8 audit — the V3.6 prompt read whole against one question (2026-09-14)

**The question:** where does the guide fail at revisiting recently encoded nodes inside the window it is exposed to —
turns marked `encoded="true"` whose facts have no node, a misread locked in by the next run, a new value bent to fit
the encoded node (brain id:64abc6a0, Tom's brief item 1). One lens, the whole prompt; Tom marks first, in the last
column. Line numbers are `servers/scales/s1/encoding_prompt.py` on main 1136242 (`template_full.md` of the V3.6
fixture is byte-identical to `SYSTEM_PROMPT`, checked this session). The gist is `eval/fixtures/s1e_guide_v3_6_2026-09-13/gist_full.md`
(rendered once in the user content, between catalog and timeline). Runtime blocks (field summary, Arc, Review,
Finishing) are appended by `encode._build_system_prompt` from `servers/trace_contract.py` — shared by every encoder,
not part of the s1e template.

## The evidence the table is read against (gym item 59524333, both longmem corpora, read this session)

| run | window | V3.6 (`b0175a`) did | V3.7 (`ef2443`) did |
|---|---|---|---|
| 1 (stop 5) | turns 1–5 `false`, Feb 2023 | 5 nodes incl. gym 7 pm MWF, method with 6 pm alert | 4 nodes incl. gym 7 pm MWF |
| 2 (stop 10) | 1–5 `true`, 6–10 `false`; turn 8 (May 30) "the gym, which is usually at 6:00 pm" | filed rule 4: one `open` "7pm (turn 3) vs 6pm (turn 8): which is correct?", schedule node content "start time unclear", **situation left "at 7pm"**, `event_time` left 2023-02-11 | `revise_batch` on the 7 pm node: **thought only** — "mentioned unprompted 3.5 months later … stable"; Arc: "one-off meeting-planning episode … encoded as transactional — no durable nodes minted" |
| 3 (stop 11) | 2–10 `true`, 11 `false` | one reply, no tool call; 14 target lines all `clean`; "Neither voice introduced new factual disclosures beyond what's already encoded" | one reply, no tool call; `changes: gym schedule — mentioned again unprompted in turn 7 … reinforces stability`; `new: Tuesday 2:00 pm … transactional, no durable node warranted per session arc note` |

Both windows show every turn's real `[Current date: …]` in its text (Feb 11 / May 30), so the February-to-May gap was
legible even with the eval's `age="just now"` on every turn. Neither run 3 rendered an `encoded(me, …)` provenance
line for turns 6–9: the covering run wrote nothing for them, and nothing on the page says so.

## The table

Flavour key: **C** coverage (the flag read as "held"); **A** anchoring (the new value read through the node);
**T** temporal (a changed routine filed as an in-window contradiction); **R** residue/arc carrying a no-mint verdict;
**S** the slide (turns gone after a zero-write run). Carrier: position (a sentence where a rule binds), example
(worked material), procedure (the gist's lists), code (not a prompt change).

| § / lines | what it says that bears on the question | what the worked material models | gap | flavour | carrier a fix would take | Tom |
|---|---|---|---|---|---|---|
| Opening L7–13 | "revise what changed … the skip I regret is the one I can't undo" | — | no mention that a prior run's coverage can itself be the skip | C | position (text) | |
| What I Receive L19 | "Read the catalog first (the prior), then the timeline (the delta)" | — | the timeline is not all delta: covered turns are re-rendered prior text; the framing primes "catalog = held, timeline = new", so a covered turn's un-minted fact is neither | C | position | |
| L21 `<continuity>` | "a prior note is revisable evidence, not a ruling for this run" | — | says *note* (Review); the **Arc** line arrives in the same block and carried the V3.7 verdict ("no durable nodes minted") that run 3 quoted as a ruling; nothing names the arc | R | position | |
| L22 catalog tags | `[encoded(me, turn 12)]` identifies origin; "If it holds the same claim, I revise by id instead of minting a twin" | — | the tag is never used for anything; "the same claim" is the anchoring licence — a 6 pm gym read as "the same claim" as the 7 pm node gets folded (V3.7 thought, the probe's "prep window") | A, C | position | |
| L24–37 timeline schema | `<turn n age encoded>`; `now=`; "The free-text guide immediately before the timeline activates these rules" | the schema block shows one turn, `encoded="false"` | no covered turn is ever depicted anywhere in the prompt (checked: zero `encoded="true"` in ~12 worked windows); the only teaching is the gloss at L39 | C | example | |
| **L39 the gloss** | "`encoded="true"` means a prior run covered the turn … I reread covered text for cross-turn patterns and contradictions, not fresh atoms; later evidence can revise its encoded substance. Previously encoded never means untouchable." | — | "covered" ≠ "held": the flag is stamped by run range (trace_links.py L300–325), a zero-write run stamps too; "not fresh atoms" rules out exactly the un-minted fact; "patterns" is what run 3 found (turn 7 confirms MWF) while the contradiction one turn later went unread — rereading for patterns finds the confirming turn | C, A | position (the primary carrier of the class) | |
| L43 provenance | "`encoded(me, turn N)` identifies prior writes at the covering run's last turn" | — | a zero-write run renders **no** line — empty coverage is invisible, and nothing says "compare what turn N's run kept with what turns ≤ N say" | C | position (+ code: render "covered by run X, 0 nodes") | |
| L45 | "Turn coordinates orient this reading, never stored memory" | Bad/Good pair | V3.6 stored "turn 3 vs turn 8" in the open's title and content anyway — a symptom of reading a multi-session window as one moment | T | (already a rule; example) | |
| Reading L53 | "A stated fact about a life, schedule or possession earns its atom on first disclosure" | — | "first disclosure" reads as "new in this window"; a covered turn's fact with no node is a first disclosure to memory, and nothing says so | C | position | |
| **L72–77 four forms** | rule 3 "Changed value or state: update the stale claim … history with independent value earns a dated successor + supersedes"; rule 4 "Unresolved in-window contradiction: preserve both values … Do not silently choose" | — | no discriminator between 3 and 4 when the same speaker restates a routine months later; "in-window" is a window-coordinate notion, and a multi-session window makes two dated sessions look like one; V3.6 run 2 chose 4 with both texts in view | T | example (protocol id:4c43742b forbids new rule clauses) | |
| L79 developing understanding | "A theme that builds across turns … 3+ distinct turns … when later evidence fits, the read firms up" | — | V3.7 run 2 used "unprompted reconfirmation" as firming evidence and the contradiction one turn later did not register; the pattern reading absorbed the change | A | example | |
| Nodes L111 `thought` | "When new evidence moves my read, updating only the thought is normal maintenance … without … losing the fresh fact" | — | licenses the thought-only revise; the guard is a subordinate clause; V3.7 run 2 did exactly this and lost 6 pm | A | position | |
| Temporal L155 | "Resolve against the **conversation's date**" | Nadia | a window spanning sessions has several dates; the encoder kept `event_time` 2023-02-11 on a node whose content now speaks of May | T | example | |
| Temporal L163 Validity | "routine parameter change → in-place swap, old value retained in prose … History with independent weight → new dated node plus `supersedes`" | Priya (L817–827): frequency 2 → 3/week, "as of 2023-11-30 (was twice a week from 2023-08-11)" | this IS the gym shape and the right move — but Priya's old value sits only in the catalog; the gym window re-rendered the old statement's **text** beside the new one, and that is what turned a change into a "contradiction" (rule 4). No example has the old text in view as a covered turn | T via C | example | |
| Actions L257–261 reads | `get_nodes` / `recall_batch` "Ask once for the missing material" | — | neutral; run 3 had everything visible and read nothing — the class is not a fetch problem | — | — | |
| L263 | "revise an existing claim that changed or developed" | — | neutral | — | — | |
| L265 revision preserves | "Walk EVERY contradicted surface" | — | V3.6 run 2 left the situation at "7pm" (class R, known); relevant because a half-revise is what run 3 then read as clean | A | (class R, measured elsewhere) | |
| L271 capture gate | "an omitted detail falls out of the sliding window" | — | true and unhelpful: the detail is still *in* the window as a covered turn, and L39 rules it out | C | position | |
| **L273 Skip** | zero writes only when "the substance is already held or the exchange is routine — greetings, acknowledgements, **covered restatements**, abandoned questions" | — | "covered restatements" makes coverage itself a reason to skip; run 3 called the whole May session "transactional" | C | position | |
| L275 traps | brevity, packing, smoothing voice, skipping uncertainty, my voice as mere response, hedging, hardening a leaning | — | no trap named for trusting the flag or reading a covered turn through its node | C, A | position | |
| Cadence L281 | "The window slides; graph and continuity carry forward. Handle visible material now rather than counting on another run." | — | the right stance, abstract; nothing ties it to covered turns | C | position | |
| L285 first reply | four lists, "otherwise the write" | — | neutral | — | — | |
| L287 inspect | "read the resulting claims against the conversation … Check what was learned, what changed and what still holds" | Mira repair | inspection is of the *written result*; nothing inspects the catalog against the covered text | C | procedure | |
| **L289 / L469** | "A no-mint verdict never goes to residue" | Mira: "A no-mint verdict never goes to residue" | names residue only; V3.7's verdict travelled in the **Arc**; the runtime Review block (last in the prompt) says "capture residue freely … a pattern forming" and the airport item's "threshold held cleanly — pattern is confirmed" fits that invitation exactly | R | position (template) — the runtime block is trace_contract's, out of s1e scope; flag | |
| Mira episode 1 L297–473 | catalog of 2 nodes, five fresh turns, lists → read → write → inspect → repair | the change-driven walk | no covered turn | — | — | |
| Mira episode 2 L475–599 | "A later window" two days on: the first batch's nodes revised from fresh turns | revising recently encoded nodes — the closest thing to the class | the earlier turns are gone and the nodes were *right*; nothing shows a later window where the prior node misread or missed turns still on the page | C, A | example (the slot where the class's example belongs) | |
| embed_queue / Aisha / Sam L601–683 | measured finding, answered open, formative phrase | — | fresh material | — | — | |
| Thin window L685–745 | cold catalog, three turns, four nodes | "write the facts down before deciding whether the window was worth anything" | the right instinct for an uncovered window; nothing says it applies to covered turns whose facts have no node | C | example (a covered thin window) | |
| Priya revise L779–832 | swap with "as of … (was …)" | the routine-change shape | see L163 row | T | example | |
| Sweep L834–975 | one event, many stale claims; deixis through actions; residue closed in Review | the target walk at scale | fresh turns; the residue line closed is a *work* note ("awaiting review"), not a verdict about a window | R | — | |
| Identity examples L977–1245 | — | — | fresh | — | — | |
| Closure L1247–1257 | "a doubt about what it might have missed is a Review note" | — | invites the miss into residue as a doubt; fine, but nothing distinguishes a doubt from a verdict | R | position | |
| **Gist `targets`** | "for each **change**, I walk EVERY catalog entry …" | the auth-rewrite sweep | the walk is change-driven: with no change from the uncovered turn, it degenerates into "no change · all clean" ×14 (run 3, both arms) — a ritual that certifies the catalog against nothing; no pass runs the other way (catalog entry → the covered turns it came from) | C, A | procedure | |
| Gist `fetch` | "'Revise next time it surfaces' means fetch it now, and the note closes as `resolved`" | — | the anti-deferral rule for residue; no equivalent for "the arc says no durable nodes minted" or "the flag says covered" | R, C | procedure | |
| Gist `new` | "One line per first-disclosure fact **the window carries** … I write these lines before I judge the window: a window that has them is never routine" | — | "the window carries" includes covered turns, but L39 (read first, in the system prompt) says not fresh atoms; the two texts conflict and the gloss wins; run 3 (V3.7) wrote `new:` lines for Tuesday 2 pm and Google Drive and then talked itself out of them "per session arc note" | C, R | procedure | |
| Runtime Review block (trace_contract.py L997–1019) | "capture residue freely … a doubt, a friction, a surprise, a pattern forming … A clean run is an empty fence" | — | recency position, shared owner; it does not forbid a no-mint verdict and "a pattern forming" invites it | R | code/contract (not this round's carrier; flag) | |
| Runtime Arc block (L1073–1081) | "what progressed in this stretch of work … only the new movement" | — | "no durable nodes minted" is read as progress; the next run reads the arc as a ruling (L21 covers notes only) | R | code/contract; the template side is L21 | |

## Outside the prompt — the code paths the class rides on (flag for ruling, not this round's carrier)

| path | what it does | direction (id:64abc6a0) |
|---|---|---|
| `servers/scales/s1/trace_links.py` L300–325 | a turn's `encoded_by` = the first encoding_run whose stop is after it; the run's created/revised list may be empty | (a) coverage by write — a zero-write run leaves its turns `false` |
| `servers/scales/s1/encode.py` L1097–1129 | covered turns keep full text; `encoded="true"` from `encoded_by`; provenance lists the run's writes at its last turn — nothing when it wrote none | (b) render "covered by run X, 0 nodes" so empty coverage is visible |
| `servers/scales/s1/encode.py` L1130–1141 | per-turn `age` from trace `created_at`; replayed corpora stamp turns at replay wall-clock → every turn "just now" (Tom: the injectable brain-time "was leaking") | (d) fix the harness ages before reading knowledge-update items |
| the window after a zero-write run | the earlier turns are no longer rendered; catalog empty of them | (a) removes it; nothing in the prompt can |

## What the read adds to id:64abc6a0

1. **The covered turn's text is what caused the temporal misfile.** Priya (L779–832) teaches the routine-change swap with
   the old value only in the catalog. The gym window re-rendered the February statement as a covered turn beside the
   May one, and V3.6 read two texts in one window as rule 4. The class's example must have the old text on the page.
2. **The verdict leaked through the Arc, not the Review.** L289/L469 forbid residue only; L21 excuses notes only. The
   runtime blocks (shared, recency position) invite "a pattern forming".
3. **Rereading for patterns finds the confirming turn.** L39's "cross-turn patterns and contradictions" was followed
   literally: run 3 (V3.7) cited turn 7's MWF as reinforcement and did not read turn 8's 6 pm.
4. **No worked window depicts a covered turn.** The gloss is the only teaching; examples outrank rules (A3).
5. **The change-driven target walk certifies nothing when the delta is routine.** Fourteen `clean` lines against a
   catalog the encoder never compared with the covered text.

## Candidate carriers (drafts for Tom's marks — nothing frozen, nothing measured beyond the replay probe)

One carrier type per arm (id:4c43742b). Each is an exact-once replacement on `template_full.md` / `gist_full.md`;
the text is in `author.py`; the replay probe (`PROBES.md`) shows what each does on the three captured gym windows.

| arm | type | sites | what it teaches |
|---|---|---|---|
| `v3_8_gloss` | position (text pass) | L39, L21, L273, L289 | coverage is a claim about a run, not about memory: the catalog shows what the run kept; a covered turn whose fact has no node, or whose node reads it differently, is mine now; a prior *arc line* is evidence, not a ruling; "covered restatements" → "restatements of what a node already holds"; a no-mint verdict goes to neither residue nor arc |
| `v3_8_example` | example | one worked window after Mira's later window (L599) | a catalog node `[encoded(me, turn 5)]` "gym 7 pm MWF"; a window whose covered turns include a three-months-later "usually at 6:00 pm" and whose only uncovered turn is routine; three Bads (turn 3 vs turn 8 open; "covered, nothing new"; thought-only "reconfirmed stable"); the lists mark the node stale on every surface incl. `event_time`; the swap "as of … (was …)" and the dependent method node's alert time |
| `v3_8_walk` | procedure (gist) | `targets`, `new` | before the change-driven walk, one pass the other way: every `encoded(me, turn N)` entry against the covered turns it came from — a fact no node carries goes on `new`, a node that reads a covered turn differently than the text is `stale`; "the window carries — covered turns included" |
