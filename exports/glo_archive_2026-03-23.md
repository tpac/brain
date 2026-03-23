# Glo Brain Archive — 2026-03-23

Comprehensive export of all Glo-related nodes from the brain database.
Generated for archival before cleanup.

## Summary Statistics

- **Total nodes:** 465
- **Date range:** 2026-03-15 to 2026-03-23
- **Locked nodes:** 260
- **Total edges (involving Glo nodes):** 5244
  - Internal (both endpoints Glo): 1638
  - External (one endpoint non-Glo): 3606

### Type Breakdown

| Type | Count |
|------|-------|
| decision | 230 |
| rule | 100 |
| concept | 36 |
| intuition | 27 |
| context | 24 |
| project | 10 |
| task | 8 |
| file | 5 |
| thought | 5 |
| person | 2 |
| tension | 2 |
| lesson | 2 |
| mechanism | 2 |
| hypothesis | 2 |
| aspiration | 2 |
| mental_model | 2 |
| pattern | 2 |
| param_influence | 1 |
| bug_lesson | 1 |
| purpose | 1 |
| vocabulary | 1 |

### Project Attribution

| Project | Count |
|---------|-------|
| Glo | 401 |
| (none) | 54 |
| brain | 3 |
| 2026-03-15T22:59:44.621Z | 2 |
| 2026-03-15T21:29:06.383Z | 1 |
| 2026-03-15T22:59:44.622Z | 1 |
| 2026-03-15T22:53:26.653Z | 1 |
| 2026-03-15T23:35:42.863Z | 1 |
| 2026-03-15T23:42:31.480Z | 1 |

---

## Nodes

### DECISION (230 nodes)

#### Aspect ratio removed from UI

- **ID:** `dec_x4b6pvhm`
- **Type:** decision
- **Confidence:** 0.8
- **Created:** 2026-03-15T21:29:06.383Z
- **Project:** Glo
- **Keywords:** aspect ratio removed hidden automatic

Aspect ratio is auto-generated behind the scenes. No user-facing selector.

---

#### [o_myglos] Multiple simultaneous Glos per user — yes.

- **ID:** `dec_88dw1kxl`
- **Type:** decision
- **Confidence:** 0.8
- **Created:** 2026-03-15T21:38:53.735Z
- **Project:** Glo
- **Keywords:** o_myglos decision multiple simultaneous glos per user — yes.

Multiple simultaneous Glos per user — yes.

---

#### [o_myglos] Demo features: pulsing live dot for active Glos, mini sparkline charts, FAB for 

- **ID:** `dec_c4y0lvmg`
- **Type:** decision
- **Confidence:** 0.8
- **Created:** 2026-03-15T21:38:53.750Z
- **Project:** Glo
- **Keywords:** o_myglos decision demo features: pulsing live dot for active glos,

Demo features: pulsing live dot for active Glos, mini sparkline charts, FAB for new Glo creation.

---

#### [stm:s57] LOCKED: Separate API + Web. Glo API (REST) is single source of truth. Glo Web (N

- **ID:** `dec_usr21at9`
- **Type:** decision
- **Confidence:** 0.8
- **Created:** 2026-03-15T21:38:54.007Z
- **Project:** Glo
- **Locked:** YES
- **Keywords:** stm s57  architecture locked: separate api + web. glo api (rest) is single

LOCKED: Separate API + Web. Glo API (REST) is single source of truth. Glo Web (Next.js) and future agents are both API consumers. Dual creation paths: step-by-step REST resources + quick-create convenience endpoint for agents.

---

#### [stm:s58] LOCKED: Glo owns publisher profiles (rich metadata: branding, type, audience sto

- **ID:** `dec_6bo4ejah`
- **Type:** decision
- **Confidence:** 0.8
- **Created:** 2026-03-15T21:38:54.013Z
- **Project:** Glo
- **Locked:** YES
- **Keywords:** stm s58  architecture locked: glo owns publisher profiles (rich metadata: branding, type, audience

LOCKED: Glo owns publisher profiles (rich metadata: branding, type, audience story, pricing, moderation rules). EX.CO/GAM have inventory slots. Glo maps its publisher profiles to supply-side inventory.

---

#### [stm:s59] CampaignParamsResolver — isolated component that takes a Glo and returns GAM-rea

- **ID:** `dec_bv3bmcjs`
- **Type:** decision
- **Confidence:** 0.8
- **Created:** 2026-03-15T21:38:54.018Z
- **Project:** Glo
- **Locked:** YES
- **Keywords:** stm s59  architecture campaignparamsresolver — isolated component that takes a glo and returns

CampaignParamsResolver — isolated component that takes a Glo and returns GAM-ready params (dayparts, freq cap, pacing, views per session). V1: config-driven defaults per publisher type. V2+: AI buyside agent with dynamic optimization. TODO: full buyside agent, advertiser advanced settings, performance-based auto-tuning.

---

#### [o_brightness] Dynamic pricing: media prices change daily (e.g. big MLB event). Impacts pause/r

- **ID:** `dec_bhcyrn3o`
- **Type:** decision
- **Confidence:** 0.83
- **Created:** 2026-03-15T22:33:17.477Z
- **Project:** Glo
- **Keywords:** o_brightness decision dynamic pricing: media prices change daily (e.g. big

Dynamic pricing: media prices change daily (e.g. big MLB event). Impacts pause/re-light economics.

---

#### [o_lifecycle] Re-light: easy path to spend again — same creative, new budget at current rates,

- **ID:** `dec_7ld3306a`
- **Type:** decision
- **Confidence:** 0.83
- **Created:** 2026-03-15T22:33:17.509Z
- **Project:** Glo
- **Keywords:** o_lifecycle decision re-light: easy path to spend again — same

Re-light: easy path to spend again — same creative, new budget at current rates, goes through new review.

---

#### [stm:s48] LOCKED: Creative screen defaults to Upload tab (not AI Generate). Upload is prim

- **ID:** `dec_f29ykczb`
- **Type:** decision
- **Confidence:** 0.95
- **Created:** 2026-03-15T22:33:17.509Z
- **Project:** Glo
- **Locked:** YES
- **Keywords:** prompt engineering, guardrail, brand safety

Edit the system prompt in creative-director.js to instruct Claude to avoid suggesting logos, trademarks, or recognizable brand elements from competitors or unrelated companies in the generated ad creative.

---

#### [stm:s55] LOCKED: System architecture decisions — Glo owns users/billing/creative/moderati

- **ID:** `dec_l3785iyj`
- **Type:** decision
- **Confidence:** 0.95
- **Created:** 2026-03-15T22:33:52.512Z
- **Project:** Glo
- **Locked:** YES
- **Keywords:** stm s55  architecture locked: system architecture decisions — glo owns users/billing/creative/moderation/publisher-profiles. gam is

LOCKED: System architecture decisions — Glo owns users/billing/creative/moderation/publisher-profiles. GAM is source of truth for campaign state (scheduling, capping, pacing, reporting). EX.CO is player/render environment only. Creative assets on Glo CDN (S3+CloudFront).

---

#### Goal only in AI generate path

- **ID:** `dec_ea73z2xh`
- **Type:** decision
- **Confidence:** 0.95
- **Created:** 2026-03-15T22:33:52.610Z
- **Project:** Glo
- **Locked:** YES
- **Keywords:** goal selector, AI Generate, onboarding separation

Move goal selector ('What's the goal?') OUT of onboarding entirely. It now appears ONLY in Creative AI Generate tab, above vibe selector. Onboarding collects only business name + website. Tom had to repeat this.

---

#### [o_brightness] Tier names: Well/Bright/Shine at $30/$50/$100. Supernova removed. Custom option 

- **ID:** `dec_512b82lk`
- **Type:** decision
- **Confidence:** 0.8
- **Created:** 2026-03-15T22:33:52.610Z
- **Project:** Glo
- **Keywords:** o_brightness decision tier names: well/bright/shine at $30/$50/$100. supernova removed. custom

Tier names: Well/Bright/Shine at $30/$50/$100. Supernova removed. Custom option retained.

---

#### Upload tab: social + Shopify coming soon

- **ID:** `dec_rds1u570`
- **Type:** decision
- **Confidence:** 0.95
- **Created:** 2026-03-15T22:33:52.732Z
- **Project:** Glo
- **Locked:** YES
- **Keywords:** MVP, upload, transcode, AI video, white-label, creative flow, Waymark

Creative strategy prioritizes three paths: 1) Upload + transcode (majority of users), 2) AI-assisted from Google Maps/website URL (outsourced via white-label partner, not core), 3) Social/YouTube import (teaser only, avoids licensing complexity and dead-end UX). AI video generation is a feature for ease, NOT a competitive advantage.

---

#### Onboarding: two fields, no goal

- **ID:** `dec_t1zl6v9h`
- **Type:** decision
- **Confidence:** 0.83
- **Created:** 2026-03-15T22:39:19.150Z
- **Project:** Glo
- **Keywords:** onboarding fields name business url maps goal

Field 1: Your Name/Business — plain text, no autocomplete. Field 2: Website URL or Google Maps location — URL detection + Maps autocomplete. NO goal selector on this screen.

---

#### [stm:s45] CRITICAL TOM FEEDBACK (given 3x — DO NOT REVERT): (1) Onboarding screen = Busine

- **ID:** `dec_sqaka0ks`
- **Type:** decision
- **Confidence:** 0.95
- **Created:** 2026-03-15T22:39:19.151Z
- **Project:** Glo
- **Locked:** YES
- **Keywords:** stm s45 glo decision critical tom feedback (given 3x — do not revert): (1)

CRITICAL TOM FEEDBACK (given 3x — DO NOT REVERT): (1) Onboarding screen = Business name OR Google Maps autocomplete + Site URL field. Two fields. NO goal selector on onboarding. (2) Goal selector ONLY appears on Creative screen when user picks AI Generate. If they upload their own creative, no goal needed. (3) Demo v2 had a React error: useState called inside conditional (hooks can't be conditional). Fixed by hoisting creativeMode state to top level. (4) Tom frustrated by feedback loop — context files MUST lock in his decisions permanently.

---

#### [stm:s47] LOCKED: Onboarding field 1 = 'Your Name/Business' (plain text, no autocomplete).

- **ID:** `dec_us9h7gcz`
- **Type:** decision
- **Confidence:** 0.95
- **Created:** 2026-03-15T22:39:19.151Z
- **Project:** Glo
- **Locked:** YES
- **Keywords:** stm s47 glo decision locked: onboarding field 1 = 'your name/business' (plain text, no

LOCKED: Onboarding field 1 = 'Your Name/Business' (plain text, no autocomplete). Field 2 = 'Website URL or Google Maps location' (URL detection + Google Maps autocomplete). Icon swaps between 🌐 and 📍 based on input.

---

#### [o_antifraud] Payment gate over phone verification. Less friction, stronger signal. Credit car

- **ID:** `dec_93uti0o7`
- **Type:** decision
- **Confidence:** 0.8
- **Created:** 2026-03-15T22:39:19.176Z
- **Project:** Glo
- **Keywords:** o_antifraud decision payment gate over phone verification. less friction, stronger

Payment gate over phone verification. Less friction, stronger signal. Credit card credentials are the best bot filter.

---

#### [o_brightness] Users see branded tiers, not CPM/impression math. Simplicity over transparency.

- **ID:** `dec_t2in469y`
- **Type:** decision
- **Confidence:** 0.83
- **Created:** 2026-03-15T22:39:19.248Z
- **Project:** Glo
- **Keywords:** o_brightness decision users see branded tiers, not cpm/impression math. simplicity

Users see branded tiers, not CPM/impression math. Simplicity over transparency.

---

#### Upload is default creative tab

- **ID:** `dec_wc5py1ss`
- **Type:** decision
- **Confidence:** 0.83
- **Created:** 2026-03-15T22:39:38.559Z
- **Project:** Glo
- **Keywords:** Upload tab default Creative screen regression

Demo showed older version with AI tab first. Upload (user-provided media) is primary flow; AI generation is secondary/advanced. Tab order reversed in Creative.jsx to restore correct default.

---

#### tmemory scoring: 35% relevance + 30% recency + 25% emotion + 10% frequency

- **ID:** `dec_8ccn911r`
- **Type:** decision
- **Confidence:** 0.95
- **Created:** 2026-03-15T22:51:00.625Z
- **Project:** Glo
- **Locked:** YES
- **Keywords:** tmemory scoring formula weights relevance recency emotion frequency

v3 rebalanced from v2 (40/40/20). Emotion as first-class signal. Locked nodes: 50% relevance, 25% emotion, 20% recency, 5% frequency. Emotion floor 0.3 so neutral nodes not zeroed. Emotion > 0.5 slows Ebbinghaus decay.

---

#### tmemory v4.2: dream-time keyword enrichment + extractKeywords generates variants

- **ID:** `dec_gne7699k`
- **Type:** decision
- **Confidence:** 0.95
- **Created:** 2026-03-15T22:51:20.971Z
- **Project:** Glo
- **Locked:** YES
- **Keywords:** tmemory enrichment keywords dream extract variants punctuation hyphen

extractKeywords() now produces punctuation-stripped variants automatically (ex.co→exco, top-up→topup) and splits hyphenated words. Dream cycles run _dreamEnrichKeywords() to backfill sparse nodes from their own content. /enrich-keywords endpoint for manual or batch enrichment. But enrichment can only work with whats in the content — the real defense against data degradation is good keywords at remember-time, per the cue system principle.

---

#### Auth: email signup removed. SSO-only (Google/Apple/Facebook/Amazon/Shopify). Kee

- **ID:** `dec_ule3fp2s`
- **Type:** decision
- **Confidence:** 0.83
- **Created:** 2026-03-15T22:59:21.097Z
- **Project:** Glo
- **Keywords:** sso authentication auth login signup email removed google apple facebook amazon shopify only

Auth: email signup removed. SSO-only (Google/Apple/Facebook/Amazon/Shopify). Keep it easy.

---

#### LOCKED DECISION (Tom gave 3x): Onboarding = Business name OR Google Maps autocom

- **ID:** `dec_qznlarcg`
- **Type:** decision
- **Confidence:** 0.83
- **Created:** 2026-03-15T22:59:21.262Z
- **Project:** Glo
- **Keywords:** stm s27 glo decision locked decision (tom gave 3x): onboarding = business name or

LOCKED DECISION (Tom gave 3x): Onboarding = Business name OR Google Maps autocomplete field + Site URL field. NO goal selector on onboarding. Goal only in Creative AI gen path. Do NOT revert this.

---

#### Creative AI section now URL/Google Maps autocomplete with search icon, dropdown 

- **ID:** `dec_gu4zc2yo`
- **Type:** decision
- **Confidence:** 0.83
- **Created:** 2026-03-15T22:59:21.296Z
- **Project:** Glo
- **Keywords:** stm s28 glo decision creative ai section now url/google maps autocomplete with search icon,

Creative AI section now URL/Google Maps autocomplete with search icon, dropdown suggestions. Not a plain text field.

---

#### Anti-fraud: payment gate confirmed over phone number. Less friction, stronger si

- **ID:** `dec_fd1qni86`
- **Type:** decision
- **Confidence:** 0.8
- **Created:** 2026-03-15T22:59:21.311Z
- **Project:** Glo
- **Keywords:** stm s20 glo decision anti-fraud: payment gate confirmed over phone number. less friction, stronger

Anti-fraud: payment gate confirmed over phone number. Less friction, stronger signal.

---

#### Glo lifecycle states: Draft→Pending Review→Active→Completed. Branches: Rejected 

- **ID:** `dec_g9kj5da2`
- **Type:** decision
- **Confidence:** 0.95
- **Created:** 2026-03-15T22:59:21.323Z
- **Project:** Glo
- **Locked:** YES
- **Keywords:** stm s10 glo decision glo lifecycle states: draft→pending review→active→completed. branches: rejected (refund+duplicate), paused (credits

Glo lifecycle states: Draft→Pending Review→Active→Completed. Branches: Rejected (refund+duplicate), Paused (credits back to wallet — dynamic pricing means can't hold credits at old rate). Re-light from any terminal state (same creative, new budget, new review).

---

#### AI moderation signals: business legitimacy, site reputation, brand safety (IAB s

- **ID:** `dec_6f55wzbe`
- **Type:** decision
- **Confidence:** 0.8
- **Created:** 2026-03-15T22:59:21.335Z
- **Project:** Glo
- **Keywords:** stm s12 glo decision ai moderation signals: business legitimacy, site reputation, brand safety (iab

AI moderation signals: business legitimacy, site reputation, brand safety (IAB standards), nudity/violence. Moderator also sees user rating (past blocks, budgets spent) to assess risk.

---

#### Moderation UI: scale-friendly from day 1 — filtering, mass actions, keyboard sho

- **ID:** `dec_ahefv17q`
- **Type:** decision
- **Confidence:** 0.8
- **Created:** 2026-03-15T22:59:21.347Z
- **Project:** Glo
- **Keywords:** stm s13 glo decision moderation ui: scale-friendly from day 1 — filtering, mass actions,

Moderation UI: scale-friendly from day 1 — filtering, mass actions, keyboard shortcuts. But also works when volume is low (early days).

---

#### Reject flow: predefined categories + optional moderator note. User gets refund +

- **ID:** `dec_yvj3fuvx`
- **Type:** decision
- **Confidence:** 0.8
- **Created:** 2026-03-15T22:59:21.359Z
- **Project:** Glo
- **Locked:** YES
- **Keywords:** guide prompt textarea examples TikTok vibe

AI tab should include optional 'Guide the AI' textarea between vibe selector and generate button. Example prompts: 'Make it TikTok style with fast cuts', 'Show the proposal message in big elegant text'. Both prompt and vibe passed to server in creative brief generation.

---

#### Multiple simultaneous Glos per user: yes.

- **ID:** `dec_epoh25zf`
- **Type:** decision
- **Confidence:** 0.8
- **Created:** 2026-03-15T22:59:21.423Z
- **Project:** Glo
- **Keywords:** stm s14 glo decision multiple simultaneous glos per user: yes.

Multiple simultaneous Glos per user: yes.

---

#### Flywheel: unfilled inventory→house ads recruit advertisers→new Glos fill invento

- **ID:** `dec_bmzmb6mx`
- **Type:** decision
- **Confidence:** 0.8
- **Created:** 2026-03-15T22:59:22.270Z
- **Project:** Glo
- **Keywords:** ltm l2 glo arch flywheel: unfilled inventory→house ads recruit advertisers→new glos fill inventory→remaining unfilled

Flywheel: unfilled inventory→house ads recruit advertisers→new Glos fill inventory→remaining unfilled runs more house ads. Self-reinforcing.

---

#### Target user: SMB to micro-biz to normal individuals. Anyone with a phone. Never 

- **ID:** `dec_5acw6hwv`
- **Type:** decision
- **Confidence:** 0.8
- **Created:** 2026-03-15T22:59:22.349Z
- **Project:** Glo
- **Keywords:** ltm l3 glo arch target user: smb to micro-biz to normal individuals. anyone with

Target user: SMB to micro-biz to normal individuals. Anyone with a phone. Never bought media before. Sitting in a bar, reading Adweek, watching local CTV.

---

#### Spend model: one-time tiers + $X/day recurring cancel-anytime. Min spend low eno

- **ID:** `dec_j7kxuz22`
- **Type:** decision
- **Confidence:** 0.8
- **Created:** 2026-03-15T22:59:22.368Z
- **Project:** Glo
- **Keywords:** ltm l8 glo decision spend model: one-time tiers + $x/day recurring cancel-anytime. min spend

Spend model: one-time tiers + $X/day recurring cancel-anytime. Min spend low enough for impulse.

---

#### Naming: no ad jargon. Not campaign — it's a Glo. Active Glo, Paused Glo, Past Gl

- **ID:** `dec_59aiq5ue`
- **Type:** decision
- **Confidence:** 0.8
- **Created:** 2026-03-15T22:59:22.387Z
- **Project:** Glo
- **Keywords:** ltm l9 glo decision naming: no ad jargon. not campaign — it's a glo.

Naming: no ad jargon. Not campaign — it's a Glo. Active Glo, Paused Glo, Past Glo, My Glos.

---

#### Pause refunds credits to wallet. Dynamic pricing means can't hold credits at old

- **ID:** `dec_7sn5h7l1`
- **Type:** decision
- **Confidence:** 0.83
- **Created:** 2026-03-15T22:59:22.645Z
- **Project:** Glo
- **Keywords:** decision pause refunds credits to wallet. dynamic pricing means

Pause refunds credits to wallet. Dynamic pricing means can't hold credits at old rate.

---

#### Spend model: one-time tiers (Well/Bright/Shine) + $X/day recurring cancel-anytim

- **ID:** `dec_nljktp0j`
- **Type:** decision
- **Confidence:** 0.83
- **Created:** 2026-03-15T22:59:22.775Z
- **Project:** Glo
- **Keywords:** recurring, daily spend, subscription, cancel, pause, flexibility

Users can set daily recurring spend instead of lump sum. Wallet draws daily amount. Can pause or cancel at any time. Remainder stays in wallet. Uber/subscription pattern.

---

#### Web app (PWA) not native iOS — avoids Apple's 30% in-app purchase cut. Normal St

- **ID:** `dec_l5r9runt`
- **Type:** decision
- **Confidence:** 0.8
- **Created:** 2026-03-15T22:59:22.813Z
- **Project:** Glo
- **Keywords:** decision web app (pwa) not native ios — avoids

Web app (PWA) not native iOS — avoids Apple's 30% in-app purchase cut. Normal Stripe processing ~2.9%+30c.

---

#### tmemory v4.2: recall scoring overhaul — uncapped spread activation + hub dampening + query normalization

- **ID:** `dec_5ng2jk38`
- **Type:** decision
- **Confidence:** 0.95
- **Created:** 2026-03-15T22:59:22.862Z
- **Project:** Glo
- **Locked:** YES
- **Keywords:** tmemory recall scoring spread activation hub dampening query normalization v4 algorithm ranking

v4.1-4.2 fixed recall quality from 53% to 73% (code only). Three algorithmic changes: (1) Spread activation no longer caps at 1.0 — proximity to query seeds is preserved, not saturated. (2) Hub dampening applies to ALL activated nodes (not just seeds) — nodes with >40 edges get penalized proportionally (40/edgeCount). (3) Query terms normalized: punctuation stripped (ex.co→exco), hyphens split (magic-link→magic+link), searchKeywords searches content field too. Also: type dampening for project/person nodes (×0.5), keyword match bonus +0.5 for direct hits.

---

#### SSO→payment linking: Google→GPay, Apple→ApplePay, Amazon→AmazonPay, Facebook→Met

- **ID:** `dec_40gyczcq`
- **Type:** decision
- **Confidence:** 0.83
- **Created:** 2026-03-15T22:59:23.119Z
- **Project:** Glo
- **Keywords:** sso authentication auth login signup email removed google apple facebook amazon shopify only

SSO→payment linking: Google→GPay, Apple→ApplePay, Amazon→AmazonPay, Facebook→MetaPay, Shopify→ShopPay.

---

#### AI video gen is NOT the moat. Buy/integrate: Creatify MVP $99/mo, Waymark produc

- **ID:** `dec_cww0oo7d`
- **Type:** decision
- **Confidence:** 0.83
- **Created:** 2026-03-15T22:59:23.119Z
- **Project:** Glo
- **Keywords:** creatify waymark cloudinary video generation ai creative tool buy integrate mvp transcoding build vs buy

AI video gen is NOT the moat. Buy/integrate: Creatify MVP $99/mo, Waymark production scale. Cloudinary transcoding.

---

#### Auth: Clerk recommended. Supports all needed SSOs + Stripe integration.

- **ID:** `dec_tb95qqbb`
- **Type:** decision
- **Confidence:** 0.83
- **Created:** 2026-03-15T22:59:23.119Z
- **Project:** Glo
- **Keywords:** sso authentication auth login signup email removed google apple facebook amazon shopify only

Auth: Clerk recommended. Supports all needed SSOs + Stripe integration.

---

#### Reject flow: predefined categories + optional moderator note. User gets refund +

- **ID:** `dec_nklsc3el`
- **Type:** decision
- **Confidence:** 0.83
- **Created:** 2026-03-15T22:59:23.294Z
- **Project:** Glo
- **Keywords:** decision reject flow: predefined categories + optional moderator note.

Reject flow: predefined categories + optional moderator note. User gets refund + can duplicate.

---

#### Moderator sees: creative all formats (16:9, 9:16, 1:1), biz name/URL, publisher+

- **ID:** `dec_hv1fvbxk`
- **Type:** decision
- **Confidence:** 0.83
- **Created:** 2026-03-15T22:59:23.396Z
- **Project:** Glo
- **Keywords:** decision moderator sees: creative all formats (16:9, 9:16, 1:1),

Moderator sees: creative all formats (16:9, 9:16, 1:1), biz name/URL, publisher+media, AI flags+score, publisher rules overlay, user rating (past blocks, spend, account age).

---

#### AI moderates first — adds comments, auto-status, risk score to moderation board.

- **ID:** `dec_y876ke8j`
- **Type:** decision
- **Confidence:** 0.8
- **Created:** 2026-03-15T22:59:23.435Z
- **Project:** Glo
- **Keywords:** decision ai moderates first — adds comments, auto-status, risk

AI moderates first — adds comments, auto-status, risk score to moderation board. Human does final approve/reject.

---

#### AI signals: business legitimacy, site reputation, brand safety (IAB standards), 

- **ID:** `dec_q97wa3iy`
- **Type:** decision
- **Confidence:** 0.8
- **Created:** 2026-03-15T22:59:23.472Z
- **Project:** Glo
- **Keywords:** decision ai signals: business legitimacy, site reputation, brand safety

AI signals: business legitimacy, site reputation, brand safety (IAB standards), nudity/violence detection.

---

#### UI: scale-friendly from day 1 — filtering, mass approve safe, keyboard shortcuts

- **ID:** `dec_wpphbr40`
- **Type:** decision
- **Confidence:** 0.8
- **Created:** 2026-03-15T22:59:23.509Z
- **Project:** Glo
- **Keywords:** decision ui: scale-friendly from day 1 — filtering, mass

UI: scale-friendly from day 1 — filtering, mass approve safe, keyboard shortcuts. Also works at low volume (early days).

---

#### Paused: credits return to wallet. Can't hold at old rate due to dynamic pricing 

- **ID:** `dec_a8rtl75a`
- **Type:** decision
- **Confidence:** 0.83
- **Created:** 2026-03-15T22:59:23.571Z
- **Project:** Glo
- **Keywords:** decision paused: credits return to wallet. can't hold at

Paused: credits return to wallet. Can't hold at old rate due to dynamic pricing (e.g. MLB event price spikes).

---

#### Main flow: Draft→Pending Review→Active→Completed. Linear with moderation gate.

- **ID:** `dec_0d22p0e0`
- **Type:** decision
- **Confidence:** 0.8
- **Created:** 2026-03-15T22:59:23.629Z
- **Project:** Glo
- **Keywords:** decision main flow: draft→pending review→active→completed. linear with moderation gate.

Main flow: Draft→Pending Review→Active→Completed. Linear with moderation gate.

---

#### Sample page link from EX.CO can be included in emails — real link, not just scre

- **ID:** `dec_zcpkv3sw`
- **Type:** decision
- **Confidence:** 0.83
- **Created:** 2026-03-15T22:59:23.691Z
- **Project:** Glo
- **Keywords:** decision sample page link from ex.co can be included

Sample page link from EX.CO can be included in emails — real link, not just screenshot.

---

#### Rejection emails include reason + CTA to duplicate and try again.

- **ID:** `dec_o8mmacjt`
- **Type:** decision
- **Confidence:** 0.83
- **Created:** 2026-03-15T22:59:23.723Z
- **Project:** Glo
- **Keywords:** decision rejection emails include reason + cta to duplicate

Rejection emails include reason + CTA to duplicate and try again.

---

#### Emails include real screenshots from actual publisher site — not mockups.

- **ID:** `dec_vvehyy1a`
- **Type:** decision
- **Confidence:** 0.95
- **Created:** 2026-03-15T22:59:23.761Z
- **Project:** Glo
- **Locked:** YES
- **Keywords:** brain-eval.py, ground truth, external JSON, GROUND_TRUTH_FILE, eval framework

brain-eval.py now loads expected recall results from an external JSON file (path via GROUND_TRUTH_FILE env var) instead of hardcoded GLO test cases. Each deployment can define its own eval benchmark without modifying plugin code.

---

#### Sample page link for online publishers — EX.CO can enable. Useful for email syst

- **ID:** `dec_5u8un5gd`
- **Type:** decision
- **Confidence:** 0.83
- **Created:** 2026-03-15T22:59:23.807Z
- **Project:** Glo
- **Keywords:** decision sample page link for online publishers — ex.co

Sample page link for online publishers — EX.CO can enable. Useful for email system too.

---

#### Progress shown as % budget spent + vanity metrics (views, clicks, QR scans).

- **ID:** `dec_nnyhd99u`
- **Type:** decision
- **Confidence:** 0.83
- **Created:** 2026-03-15T22:59:23.837Z
- **Project:** Glo
- **Keywords:** decision progress shown as % budget spent + vanity

Progress shown as % budget spent + vanity metrics (views, clicks, QR scans).

---

#### Actions from numbers screen: re-light, duplicate, pause, top-up CTA.

- **ID:** `dec_3u13dlf7`
- **Type:** decision
- **Confidence:** 0.83
- **Created:** 2026-03-15T22:59:23.876Z
- **Project:** Glo
- **Keywords:** decision actions from numbers screen: re-light, duplicate, pause, top-up

Actions from numbers screen: re-light, duplicate, pause, top-up CTA.

---

#### Rejected: full refund + duplicate option to start new Glo from same creative.

- **ID:** `dec_zn7co4dn`
- **Type:** decision
- **Confidence:** 0.83
- **Created:** 2026-03-15T22:59:44.620Z
- **Project:** Glo
- **Keywords:** decision rejected: full refund + duplicate option to start

Rejected: full refund + duplicate option to start new Glo from same creative.

---

#### GAM as campaign controller

- **ID:** `dec_iweemnjn`
- **Type:** decision
- **Confidence:** 0.83
- **Created:** 2026-03-15T22:59:44.621Z
- **Project:** Glo
- **Keywords:** gam google admanager campaign controller scheduling capping

Google Ad Manager (GAM) is source of truth for campaign state. GAM handles scheduling, capping, pacing, reporting. Glo pushes line items to GAM. EX.CO tag in GAM serves creative from Glo CDN.

---

#### Moderation initially by GLO/EX.CO ops team, publishers get access later.

- **ID:** `dec_00i02v1y`
- **Type:** decision
- **Confidence:** 0.83
- **Created:** 2026-03-15T22:59:44.621Z
- **Project:** Glo
- **Keywords:** decision moderation initially by glo/ex.co ops team, publishers get

Moderation initially by GLO/EX.CO ops team, publishers get access later.

---

#### Contextual intent: infer creative direction from WHO (media context) and WHAT (u

- **ID:** `dec_6z006j7r`
- **Type:** decision
- **Confidence:** 0.83
- **Created:** 2026-03-15T22:59:44.621Z
- **Project:** Glo
- **Keywords:** decision contextual intent: infer creative direction from who (media

Contextual intent: infer creative direction from WHO (media context) and WHAT (user biz info). 3 AI variations per Glo.

---

#### Moderation: AI pre-screen + human final action. GLO/EX.CO ops initially, publish

- **ID:** `dec_6rruh9bf`
- **Type:** decision
- **Confidence:** 0.83
- **Created:** 2026-03-15T22:59:44.621Z
- **Project:** Glo
- **Keywords:** decision moderation: ai pre-screen + human final action. glo/ex.co

Moderation: AI pre-screen + human final action. GLO/EX.CO ops initially, publishers later.

---

#### Auth: Clerk recommended. All SSOs + Stripe.

- **ID:** `dec_8vvy8az3`
- **Type:** decision
- **Confidence:** 0.83
- **Created:** 2026-03-15T22:59:44.621Z
- **Project:** Glo
- **Keywords:** sso authentication auth login signup email removed google apple facebook amazon shopify only glo clerk recommended. recommended ssos stripe. stripe

Auth: Clerk recommended. All SSOs + Stripe.

---

#### Contextual intent engine: infers creative direction from WHO (what media they ca

- **ID:** `dec_jfuegdiy`
- **Type:** decision
- **Confidence:** 0.95
- **Created:** 2026-03-15T22:59:44.621Z
- **Project:** Glo
- **Locked:** YES
- **Keywords:** intent engine, contextual inference, user signal, media context, publisher

Glo knows two unique signals at engagement moment: 1) User context (media source: Adweek vs nj.com vs bar screen), 2) Ad placement destination (specific publisher/screen). Combined with business/content input, these infer the WHY (user intent), which drives creative and AI generation. Strategic advantage over generic ad builders.

---

#### Payment: Glo Credits 1:1 USD. Wallet via Stripe customer balance (avoids money t

- **ID:** `dec_j3vp1rdc`
- **Type:** decision
- **Confidence:** 0.83
- **Created:** 2026-03-15T22:59:44.621Z
- **Project:** Glo
- **Keywords:** ltm l7 glo decision payment: glo credits 1:1 usd. wallet via stripe customer balance

Payment: Glo Credits 1:1 USD. Wallet via Stripe customer balance (avoids money transmitter). SSO→payment: Google→GPay, Apple→ApplePay, Amazon→AmazonPay, Facebook→MetaPay, Shopify→ShopPay.

---

#### Glo lifecycle: Draft→Pending Review→Active→Completed. Rejected=refund+duplicate.

- **ID:** `dec_ske5osc2`
- **Type:** decision
- **Confidence:** 0.83
- **Created:** 2026-03-15T22:59:44.621Z
- **Project:** Glo
- **Keywords:** ltm l22 glo decision glo lifecycle: draft→pending review→active→completed. rejected=refund+duplicate. paused=credits back to wallet (dynamic

Glo lifecycle: Draft→Pending Review→Active→Completed. Rejected=refund+duplicate. Paused=credits back to wallet (dynamic pricing problem). Re-light=same creative, new budget, new review.

---

#### Moderation: AI pre-screen (risk score, flags, IAB brand safety, biz legitimacy, 

- **ID:** `dec_aky2oorz`
- **Type:** decision
- **Confidence:** 0.95
- **Created:** 2026-03-15T22:59:44.621Z
- **Project:** Glo
- **Locked:** YES
- **Keywords:** ltm l23 glo decision moderation: ai pre-screen (risk score, flags, iab brand safety, biz

Moderation: AI pre-screen (risk score, flags, IAB brand safety, biz legitimacy, site reputation) + human final action. AI adds comments+auto-status to board. Human approves/rejects. GLO/EX.CO ops initially, publishers later. Publisher-specific rules (e.g. MLB blocks competitor leagues) + GLO general rules both surfaced.

---

#### Glo owns publisher profiles

- **ID:** `dec_3nu4rgj4`
- **Type:** decision
- **Confidence:** 0.83
- **Created:** 2026-03-15T22:59:44.622Z
- **Project:** Glo
- **Keywords:** publisher profiles metadata branding glo ownership

Rich publisher metadata lives in Glo (branding, type, audience story, pricing, moderation rules). EX.CO/GAM have inventory slots. Glo maps its publisher profiles to supply-side inventory.

---

#### Creative on Glo CDN

- **ID:** `dec_m9fq8akb`
- **Type:** decision
- **Confidence:** 0.83
- **Created:** 2026-03-15T22:59:44.622Z
- **Project:** Glo
- **Keywords:** creative cdn s3 cloudfront hosting assets

Creative assets hosted on Glo infrastructure (S3 + CloudFront). GAM references Glo CDN URLs. EX.CO player fetches from Glo CDN. Glo controls full creative pipeline.

---

#### Daily recurring uses slider not tiers

- **ID:** `dec_xdcei6i9`
- **Type:** decision
- **Confidence:** 0.83
- **Created:** 2026-03-15T22:59:44.622Z
- **Project:** Glo
- **Keywords:** daily recurring slider budget tiers onetime

Daily Recurring budget: slider from $5-$500/day, default $30. One-time keeps tier cards (Well/Bright/Shine/Custom). GlowIcon scales dynamically with slider value.

---

#### [o_antifraud] Must balance anti-fraud friction vs impulse UX — Glo's target user is someone wh

- **ID:** `dec_y9v42sjb`
- **Type:** decision
- **Confidence:** 0.8
- **Created:** 2026-03-15T22:59:44.622Z
- **Project:** Glo
- **Keywords:** o_antifraud decision must balance anti-fraud friction vs impulse ux —

Must balance anti-fraud friction vs impulse UX — Glo's target user is someone who's never bought media before.

---

#### [o_glo] Flywheel: unfilled inventory→house ads→recruit advertisers→fill inventory→remain

- **ID:** `dec_94kstp68`
- **Type:** decision
- **Confidence:** 0.8
- **Created:** 2026-03-15T22:59:44.622Z
- **Project:** Glo
- **Keywords:** o_glo decision flywheel: unfilled inventory→house ads→recruit advertisers→fill inventory→remaining unfilled→more house

Flywheel: unfilled inventory→house ads→recruit advertisers→fill inventory→remaining unfilled→more house ads

---

#### [o_glo] 3 creative paths: upload, AI from URL/Google Maps, social import (coming soon)

- **ID:** `dec_0sv57szx`
- **Type:** decision
- **Confidence:** 0.83
- **Created:** 2026-03-15T22:59:44.622Z
- **Project:** Glo
- **Keywords:** import social shopify coming soon creative upload card feedback

3 creative paths: upload, AI from URL/Google Maps, social import (coming soon)

---

#### [o_glo] Anti-fraud: payment gate over phone verification. Less friction, stronger signal

- **ID:** `dec_dp9gw843`
- **Type:** decision
- **Confidence:** 0.83
- **Created:** 2026-03-15T22:59:44.622Z
- **Project:** Glo
- **Keywords:** o_glo decision anti-fraud: payment gate over phone verification. less friction,

Anti-fraud: payment gate over phone verification. Less friction, stronger signal.

---

#### [o_glo] Web app (PWA), not native iOS — avoids Apple's 30% in-app purchase cut.

- **ID:** `dec_a811djg2`
- **Type:** decision
- **Confidence:** 0.8
- **Created:** 2026-03-15T22:59:44.622Z
- **Project:** Glo
- **Keywords:** o_glo decision web app (pwa), not native ios — avoids

Web app (PWA), not native iOS — avoids Apple's 30% in-app purchase cut.

---

#### Drafts saved to My Glos dashboard, accessible anytime.

- **ID:** `dec_qff52nm6`
- **Type:** decision
- **Confidence:** 0.83
- **Created:** 2026-03-15T22:59:44.622Z
- **Project:** Glo
- **Keywords:** decision drafts saved to my glos dashboard, accessible anytime.

Drafts saved to My Glos dashboard, accessible anytime.

---

#### Two rule layers: GLO general rules + publisher-specific configurable rules (e.g.

- **ID:** `dec_dg3e14yv`
- **Type:** decision
- **Confidence:** 0.83
- **Created:** 2026-03-15T22:59:44.622Z
- **Project:** Glo
- **Keywords:** decision two rule layers: glo general rules + publisher-specific

Two rule layers: GLO general rules + publisher-specific configurable rules (e.g. MLB blocks competitor leagues). Both surfaced to moderator.

---

#### [o_myglos] Actions per Glo: pause, re-light, duplicate, view numbers.

- **ID:** `dec_hrh59gko`
- **Type:** decision
- **Confidence:** 0.83
- **Created:** 2026-03-15T22:59:44.622Z
- **Project:** Glo
- **Keywords:** o_myglos decision actions per glo: pause, re-light, duplicate, view numbers.

Actions per Glo: pause, re-light, duplicate, view numbers.

---

#### Moderation: AI-first, two layers — platform safety + publisher-specific configur

- **ID:** `dec_sb2bvhd5`
- **Type:** decision
- **Confidence:** 0.83
- **Created:** 2026-03-15T22:59:44.622Z
- **Project:** Glo
- **Keywords:** ltm l10 glo decision moderation: ai-first, two layers — platform safety + publisher-specific configurable

Moderation: AI-first, two layers — platform safety + publisher-specific configurable preferences.

---

#### Media types: Online (video on publisher sites), CTV (broadcast), DOOH (venue scr

- **ID:** `dec_ngj10rmw`
- **Type:** decision
- **Confidence:** 0.83
- **Created:** 2026-03-15T22:59:44.622Z
- **Project:** Glo
- **Keywords:** ltm l12 glo decision media types: online (video on publisher sites), ctv (broadcast), dooh

Media types: Online (video on publisher sites), CTV (broadcast), DOOH (venue screens). Different pricing/visuals, same core flow.

---

#### Separate API + Web architecture

- **ID:** `dec_7zjb9wgv`
- **Type:** decision
- **Confidence:** 0.95
- **Created:** 2026-03-15T22:59:44.623Z
- **Project:** Glo
- **Locked:** YES
- **Keywords:** api rest web nextjs agents separate architecture

Glo API (REST) is single source of truth. Glo Web (Next.js) and future agents are both API consumers. Dual creation paths: step-by-step REST resources for UI + quick-create convenience endpoint for agents.

---

#### API for agents at scale

- **ID:** `dec_dumnu9n7`
- **Type:** decision
- **Confidence:** 0.8
- **Created:** 2026-03-15T22:59:44.623Z
- **Project:** Glo
- **Keywords:** api agents programmatic buying scale automation

Glo API must be complete — every action in UI also available via API. External agents can buy campaigns at scale. API key auth TBD for later.

---

#### Glo is closed-loop demand layer on EX.CO — not standalone DSP. Monetizes unfille

- **ID:** `dec_0a4u7zrv`
- **Type:** decision
- **Confidence:** 0.8
- **Created:** 2026-03-15T22:59:44.623Z
- **Project:** Glo
- **Locked:** YES
- **Keywords:** thought brain-observation dream closed-loop ex.co glo glo.io share open tom canva/s canvas geniee cluster connection founder forming converging adtech company exchange converging. japanese neighbors dsp monetization gloio magnite canva u.s. closedloop areas exco lo.io loio google

Cluster forming: "Geniee — Japanese adtech company" and "Glo is closed-loop monetization on EX.CO (not open exchange DSP)" share 25 neighbors (Dream: Dream connection: "Magnite — U.S. adtech ↔ Tom — Glo.io founder, tom@ex.co, Google  | Canva/S). These areas are converging.

---

#### Glo conceptually supported by EX.CO leadership, needs formal business case for b

- **ID:** `dec_qgpataos`
- **Type:** decision
- **Confidence:** 0.8
- **Created:** 2026-03-15T22:59:44.623Z
- **Project:** Glo
- **Keywords:** decision glo conceptually supported by ex.co leadership, needs formal

Glo conceptually supported by EX.CO leadership, needs formal business case for board.

---

#### [o_myglos] Drafts saved to dashboard, accessible anytime.

- **ID:** `dec_pyaoa97l`
- **Type:** decision
- **Confidence:** 0.8
- **Created:** 2026-03-15T22:59:44.623Z
- **Project:** Glo
- **Keywords:** o_myglos decision drafts saved to dashboard, accessible anytime.

Drafts saved to dashboard, accessible anytime.

---

#### Moderation model: AI moderates first, adds comments+auto-status to moderation bo

- **ID:** `dec_du3vwge2`
- **Type:** decision
- **Confidence:** 0.95
- **Created:** 2026-03-15T22:59:44.623Z
- **Project:** Glo
- **Locked:** YES
- **Keywords:** stm s11 glo decision moderation model: ai moderates first, adds comments+auto-status to moderation board.

Moderation model: AI moderates first, adds comments+auto-status to moderation board. Human does final action (approve/reject). Initially GLO/EX.CO team. Later publishers get access. Publisher-specific rules (e.g., MLB blocks competitor leagues) + GLO general rules. Both surfaced to moderator.

---

#### Credits balance shown wherever it makes sense: My Glos dashboard, user settings 

- **ID:** `dec_ey1ebmvv`
- **Type:** decision
- **Confidence:** 0.8
- **Created:** 2026-03-15T22:59:44.623Z
- **Project:** Glo
- **Keywords:** stm s15 glo decision credits balance shown wherever it makes sense: my glos dashboard,

Credits balance shown wherever it makes sense: My Glos dashboard, user settings screen, etc.

---

#### Confirmation screen: status is always 'Pending Review' (orange), not 'Active'. G

- **ID:** `dec_dtbiv3aq`
- **Type:** decision
- **Confidence:** 0.8
- **Created:** 2026-03-15T22:59:44.623Z
- **Project:** Glo
- **Keywords:** stm s31 glo decision confirmation screen: status is always 'pending review' (orange), not 'active'.

Confirmation screen: status is always 'Pending Review' (orange), not 'Active'. Glo hasn't been moderated yet at this point.

---

#### My Glos thumbnails: video first-frame style with play button overlay + duration 

- **ID:** `dec_cet8and2`
- **Type:** decision
- **Confidence:** 0.95
- **Created:** 2026-03-15T22:59:44.623Z
- **Project:** Glo
- **Locked:** YES
- **Keywords:** stm s32 glo decision my glos thumbnails: video first-frame style with play button overlay

My Glos thumbnails: video first-frame style with play button overlay + duration badge. Mock data variety improved: DataViz Analytics on Adweek (B2B), Happy Birthday on Sports Bar (personal DOOH), Rivera Auto Group on CTV (local biz), Game Day Wings Special on bar (DOOH promo).

---

#### Budget screen order: (1) How to Spend toggle (One-time vs Daily Recurrin

- **ID:** `dec_h5wmmnyv`
- **Type:** decision
- **Confidence:** 0.95
- **Created:** 2026-03-15T22:59:44.623Z
- **Project:** Glo
- **Locked:** YES
- **Keywords:** stm s50 glo decision locked: budget screen order: (1) how to spend toggle (one-time

LOCKED: Budget screen order: (1) How to Spend toggle (One-time vs Daily Recurring) shown FIRST. (2) GLO Brightness tiers. (3) If one-time: duration selector (1 Day, 7 Days, 30 Days, Custom). Custom opens Start + End date pickers (days only, no hours). If daily: shows cancel-anytime info.

---

#### Pricing: 40% Glo margin default (adjustable per publisher/media). Publisher sets

- **ID:** `dec_s4xx6nkp`
- **Type:** decision
- **Confidence:** 0.83
- **Created:** 2026-03-15T22:59:44.623Z
- **Project:** Glo
- **Keywords:** ltm l6 glo decision pricing: 40% glo margin default (adjustable per publisher/media). publisher sets

Pricing: 40% Glo margin default (adjustable per publisher/media). Publisher sets floor rate cards. Users see branded tiers (GLO Brightness: Well $30 / Bright $50 / Shine $100) not CPM math.

---

#### Moderator sees: creative all formats, biz name/URL, publisher+media, AI flags+sc

- **ID:** `dec_2hc13gm3`
- **Type:** decision
- **Confidence:** 0.83
- **Created:** 2026-03-15T22:59:44.623Z
- **Project:** Glo
- **Keywords:** ltm l24 glo decision moderator sees: creative all formats, biz name/url, publisher+media, ai flags+score,

Moderator sees: creative all formats, biz name/URL, publisher+media, AI flags+score, publisher rules overlaid, user rating (past blocks, spend, account age). Reject=predefined category+optional note.

---

#### My Glos dashboard: cards with thumbnail, status badge, publisher, progress bar (

- **ID:** `dec_kuwfrwcb`
- **Type:** decision
- **Confidence:** 0.95
- **Created:** 2026-03-15T22:59:44.623Z
- **Project:** Glo
- **Locked:** YES
- **Keywords:** ltm l25 glo decision my glos dashboard: cards with thumbnail, status badge, publisher, progress

My Glos dashboard: cards with thumbnail, status badge, publisher, progress bar (% budget spent), key metric. Filters by status. Credits balance visible. Actions: pause, re-light, duplicate, view numbers.

---

#### Glo Numbers screen: views/day graph, total views/clicks/QR scans, budget progres

- **ID:** `dec_cscd2kgj`
- **Type:** decision
- **Confidence:** 0.83
- **Created:** 2026-03-15T22:59:44.623Z
- **Project:** Glo
- **Keywords:** ltm l26 glo decision glo numbers screen: views/day graph, total views/clicks/qr scans, budget progress,

Glo Numbers screen: views/day graph, total views/clicks/QR scans, budget progress, creative preview, share link, re-light/duplicate actions.

---

#### Email system: activation progress, performance updates with real screenshots fro

- **ID:** `dec_sx4ubqa4`
- **Type:** decision
- **Confidence:** 0.83
- **Created:** 2026-03-15T22:59:44.623Z
- **Project:** Glo
- **Keywords:** ltm l27 glo decision email system: activation progress, performance updates with real screenshots from

Email system: activation progress, performance updates with real screenshots from actual site, view count milestones, re-engagement prompts, rejection notices with reason+CTA.

---

#### Budget screen: glow icon (SVG radial gradient) grows in size+intensity per tier.

- **ID:** `dec_6l77wi5d`
- **Type:** decision
- **Confidence:** 0.95
- **Created:** 2026-03-15T22:59:44.624Z
- **Project:** Glo
- **Locked:** YES
- **Keywords:** stm s29 glo decision budget screen: glow icon (svg radial gradient) grows in size+intensity

Budget screen: glow icon (SVG radial gradient) grows in size+intensity per tier. Publisher media visualization at top shows user's creative on mockup of the media (TV frame for CTV, browser for Online, screen for DOOH).

---

#### Budget timeline: added Now/Tomorrow/7 Days/Custom date-time selector to one-time

- **ID:** `dec_22kg9tdt`
- **Type:** decision
- **Confidence:** 0.83
- **Created:** 2026-03-15T22:59:44.624Z
- **Project:** Glo
- **Keywords:** stm s34 glo decision budget timeline: added now/tomorrow/7 days/custom date-time selector to one-time budget

Budget timeline: added Now/Tomorrow/7 Days/Custom date-time selector to one-time budget flow. Sits between tier selection and continue button. Continue button shows schedule in label.

---

#### GlowIcon enhanced: brightness now dramatically affects glow spread (4+10*brightn

- **ID:** `dec_i5izh6vy`
- **Type:** decision
- **Confidence:** 0.8
- **Created:** 2026-03-15T22:59:44.624Z
- **Project:** Glo
- **Locked:** YES
- **Keywords:** stm s51 glo decision glowicon enhanced: brightness now dramatically affects glow spread (4+10*brightness), opacity

GlowIcon enhanced: brightness now dramatically affects glow spread (4+10*brightness), opacity (0.2+0.2*brightness), scale (0.85+0.1*brightness), and inner white core gets glow at highest tier. Visual difference between Well/Bright/Shine is very noticeable.

---

#### Anti-fraud concern: fake Google logins and bots on mobile. Payment as strongest 

- **ID:** `dec_g9us24ox`
- **Type:** decision
- **Confidence:** 0.8
- **Created:** 2026-03-15T22:59:44.624Z
- **Project:** Glo
- **Locked:** YES
- **Keywords:** ltm l28 glo arch anti-fraud concern: fake google logins and bots on mobile. payment

Anti-fraud concern: fake Google logins and bots on mobile. Payment as strongest gatekeeper (must spend real money). Additional options: phone verify, device fingerprint, captcha, rate limiting. Must balance vs impulse UX friction.

---

#### [o_glo] Lifecycle: Draft→Pending Review→Active→Completed. Rejected=refund+duplicate. Pau

- **ID:** `dec_89qq5rfh`
- **Type:** decision
- **Confidence:** 0.8
- **Created:** 2026-03-15T22:59:52.295Z
- **Project:** 2026-03-15T22:59:44.621Z
- **Locked:** YES
- **Keywords:** o_glo decision lifecycle: draft→pending review→active→completed. rejected=refund+duplicate. paused=credits to wallet.

Lifecycle: Draft→Pending Review→Active→Completed. Rejected=refund+duplicate. Paused=credits to wallet.

---

#### [o_glo] Closed-loop demand layer on EX.CO — not standalone DSP. Monetizes 20-40% unfille

- **ID:** `dec_o8fqg5v9`
- **Type:** decision
- **Confidence:** 0.8
- **Created:** 2026-03-15T22:59:52.328Z
- **Project:** 2026-03-15T22:59:44.621Z
- **Locked:** YES
- **Keywords:** o_glo decision closed-loop demand layer on ex.co — not standalone

Closed-loop demand layer on EX.CO — not standalone DSP. Monetizes 20-40% unfilled inventory across 100-1000 publishers.

---

#### Credits balance shown wherever makes sense: My Glos dashboard, user settings scr

- **ID:** `dec_u182xyaq`
- **Type:** decision
- **Confidence:** 0.8
- **Created:** 2026-03-15T22:59:52.356Z
- **Project:** 2026-03-15T22:59:44.622Z
- **Locked:** YES
- **Keywords:** decision credits balance shown wherever makes sense: my glos

Credits balance shown wherever makes sense: My Glos dashboard, user settings screen, etc.

---

#### Glo/EX.CO boundary

- **ID:** `dec_bdhgxfx7`
- **Type:** decision
- **Confidence:** 0.83
- **Created:** 2026-03-15T23:05:23.164Z
- **Project:** Glo
- **Keywords:** glo exco boundary ownership separation owns billing creative moderation users player render environment

Glo owns users, billing, creative, moderation, publisher profiles. EX.CO is player/render environment only. Creative assets on Glo CDN (S3+CloudFront).

---

#### Creative strategy: AI video gen is NOT the moat. Buy/integrate. Creatify API $99

- **ID:** `dec_y78ln1w0`
- **Type:** decision
- **Confidence:** 0.95
- **Created:** 2026-03-15T23:27:47.999Z
- **Project:** Glo
- **Locked:** YES
- **Keywords:** ltm l4 glo decision creative strategy: ai video gen is not the moat. buy/integrate.

Creative strategy: AI video gen is NOT the moat. Buy/integrate. Creatify API $99/mo for MVP, Waymark for production scale. Cloudinary for transcoding (smart crop across aspect ratios). 3 paths: upload, AI from URL, social import (coming soon).

---

#### Payment: Glo Credits 1:1 USD, Stripe customer balance. SSO→payment linking.

- **ID:** `dec_qz2r5qnm`
- **Type:** decision
- **Confidence:** 0.83
- **Created:** 2026-03-15T23:42:36.156Z
- **Project:** Glo
- **Keywords:** sso authentication auth login signup email removed google apple facebook amazon shopify only

Payment: Glo Credits 1:1 USD, Stripe customer balance. SSO→payment linking.

---

#### tmemory v4: self-improvement via instrumented recall + evaluation

- **ID:** `dec_546e3kou`
- **Type:** decision
- **Confidence:** 0.95
- **Created:** 2026-03-16T17:02:09.052Z
- **Project:** Glo
- **Locked:** YES
- **Keywords:** tmemory self-improvement evaluation precision coverage recall instrumentation feedback loop

Every recall auto-logged (recall_log). Claude reports used nodes (mark-recall-used). Misses tracked (miss_log). Repetition misses auto-lock node + tag frustration. Periodic /evaluate computes precision, coverage, dream hit rate, emotion accuracy. /improve suggests parameter tuning. 5+ feedback events needed before tuning kicks in.

---

#### Tier pricing: 1.4x markup multiplier on publisher CPM

- **ID:** `dec_c0h38kix`
- **Type:** decision
- **Confidence:** 0.95
- **Created:** 2026-03-16T17:02:39.512Z
- **Project:** Glo
- **Locked:** YES
- **Keywords:** tier pricing markup multiplier cpm reach well bright shine formula

TIERS function calculates reach from price/CPM with 1.4x Glo margin multiplier. Well $30, Bright $50 (recommended), Shine $100. DOOH shows plays, online shows impressions. Reach estimate = price / (CPM * 1.4) * 1000.

---

#### Glo Creative Intelligence: LLM as Creative Director, not fixed archetype mapping

- **ID:** `dec_xymi39b7`
- **Type:** decision
- **Confidence:** 0.95
- **Created:** 2026-03-16T17:02:39.532Z
- **Project:** Glo
- **Locked:** YES
- **Keywords:** LLM creative director, infinite archetypes, dynamic brief generation, vibe selector, intent understanding, marriage proposal, sports celebration

Shift from enum-based archetypes (influencer, shop, author) to LLM-based creative direction. LLM receives context (name, URL, location, publisher, vibe) and generates full creative briefs dynamically. Supports infinite use cases: marriage proposals on local CTV, sports celebrations, indie author, etc. Examples: marriage proposal and soccer celebration both pick 'Celebrate something' vibe but LLM produces completely different creative directions because it understands intent.

---

#### Video prompt rewritten: Hook→Showcase→CTA structure, industry-specific, 500 char limit

- **ID:** `dec_z0mbdh0n`
- **Type:** decision
- **Confidence:** 0.95
- **Created:** 2026-03-16T17:03:28.833Z
- **Project:** Glo
- **Locked:** YES
- **Keywords:** video prompt creative director hook showcase cta industry-specific 500 char limit 12s duration 480p buildVideoPrompt

buildVideoPrompt in creative-director.js rewritten with structured ad format: Hook (0-3s) with industry-specific cinematic directions, Showcase (3-9s) with dolly/tracking shots, CTA (9-12s) with branded end card. Industry map covers food, retail, beauty, hospitality, automotive, healthcare, fitness, real estate. Mood/tone mapping (premium, friendly, energetic, professional, playful). Pacing varies by pub type (CTV vs social). Hard 500 char limit with smart truncation. Duration fixed to 12s (API max). Resolution 480p per Toms preference.

---

#### Budget screen: How to Spend first

- **ID:** `dec_bzdb0leg`
- **Type:** decision
- **Confidence:** 0.83
- **Created:** 2026-03-16T17:04:42.279Z
- **Project:** Glo
- **Keywords:** budget order spend first tiers duration

Budget screen order: 1) How to Spend toggle (One-time vs Daily Recurring), 2) GLO Brightness tiers/slider, 3) Duration (one-time only). How to Spend is FIRST selection.

---

#### Track your Glos (plural)

- **ID:** `dec_uf5f07sj`
- **Type:** decision
- **Confidence:** 0.83
- **Created:** 2026-03-16T17:32:16.680Z
- **Project:** Glo
- **Keywords:** confirm button track glos plural

Confirm screen button text is Track your Glos (PLURAL). Not Track your Glo.

---

#### [stm:s56] LOCKED: Supply Adapter pattern — clean abstraction layer so GAM can be swapped f

- **ID:** `dec_yyz40ywt`
- **Type:** decision
- **Confidence:** 0.95
- **Created:** 2026-03-16T17:32:25.175Z
- **Project:** Glo
- **Locked:** YES
- **Keywords:** stm s56  architecture locked: supply adapter pattern — clean abstraction layer so gam

LOCKED: Supply Adapter pattern — clean abstraction layer so GAM can be swapped for another ad server. Interface: createCampaign, updateCampaign, pauseCampaign, resumeCampaign, stopCampaign, getCampaignStatus, getPerformance, syncPublisherInventory.

---

#### Glo is closed-loop demand layer on EX.CO ad server. Not standalone DSP. Monetize

- **ID:** `dec_tsezf8dq`
- **Type:** decision
- **Confidence:** 0.8
- **Created:** 2026-03-16T17:32:25.175Z
- **Project:** Glo
- **Keywords:** ltm l1 glo arch glo is closed-loop demand layer on ex.co ad server. not

Glo is closed-loop demand layer on EX.CO ad server. Not standalone DSP. Monetizes 20-40% unfilled inventory across 100-1000 EX.CO publishers. Supply is built-in.

---

#### Creatify API integrated for real AI video generation in Glo beta

- **ID:** `dec_h0hd8603`
- **Type:** decision
- **Confidence:** 0.95
- **Created:** 2026-03-16T17:32:25.175Z
- **Project:** Glo
- **Locked:** YES
- **Keywords:** creatify api video generation url-to-video link_to_videos ai creative server integration ffmpeg fallback async polling jobid

Creatify (creatify.ai) chosen as URL-to-video API. Two-step process: POST /api/link_with_params/ to create link from URL, then POST /api/link_to_videos/ to generate video (async, poll GET /api/link_to_videos/{id}/ until status=done). Headers: X-API-ID, X-API-KEY. Visual styles: DynamicProductTemplate, FullScreenTemplate, MotionCardsTemplate. 15-second videos, all aspect ratios. $99/mo Business plan. Server.js uses async job pattern: POST returns jobId, GET polls. Falls back to ffmpeg when no API keys set. Creative.jsx updated to poll server for progress.

---

#### Supply Adapter pattern

- **ID:** `dec_dp93qu36`
- **Type:** decision
- **Confidence:** 0.95
- **Created:** 2026-03-16T17:32:25.293Z
- **Project:** Glo
- **Locked:** YES
- **Keywords:** adapter pattern supply abstraction gam swappable interface

Clean abstraction layer between Glo and ad delivery. Interface: createCampaign, updateCampaign, pauseCampaign, resumeCampaign, stopCampaign, getCampaignStatus, getPerformance, syncPublisherInventory. GAM is v1 adapter. Must be easily swappable to another ad server.

---

#### NanoBanana API: single image only despite docs showing array support

- **ID:** `dec_esl0q0wq`
- **Type:** decision
- **Confidence:** 0.95
- **Created:** 2026-03-16T17:32:25.293Z
- **Project:** Glo
- **Locked:** YES
- **Keywords:** nanobanana api single image limitation image_url array 400 500 bearer auth video generation

nanobananavideo.com API only accepts image_url as a single string. Sending image_urls (plural key) returns 400 Field image_url is required. Sending image_url with array value returns HTTP 500. Confirmed by browser-based testing bypassing sandbox proxy. Auth: must use Bearer token, NOT X-API-Key header. Duration: 3-12s, resolution: 480p/720p/1080p. Prompt max 500 chars. Tom was frustrated about single-image limitation.

---

#### Veo 3.1 via Gemini API supports multi-image video generation (up to 3)

- **ID:** `dec_w07txi6t`
- **Type:** decision
- **Confidence:** 0.95
- **Created:** 2026-03-16T17:32:25.293Z
- **Project:** Glo
- **Locked:** YES
- **Keywords:** veo gemini api multi-image video generation reference images google generative ai alternative nanobanana

Google Gemini API Veo 3.1 model (veo-3.1-generate-preview) supports up to 3 reference images via referenceImages array with referenceType asset. Duration 4/6/8s (must be 8s with reference images or 1080p). Resolution 720p/1080p. Async via predictLongRunning. Auth: x-goog-api-key header. Base URL: https://generativelanguage.googleapis.com/v1beta. Discovered as alternative to NanoBananas single-image limitation. Not yet implemented — pending Toms go-ahead.

---

#### tmemory: separate user brain from plugin (fresh vs personal)

- **ID:** `dec_10c3j7ft`
- **Type:** decision
- **Confidence:** 0.95
- **Created:** 2026-03-16T17:32:25.423Z
- **Project:** Glo
- **Locked:** YES
- **Keywords:** tmemory brain separation plugin personal fresh default path

Plugin ships with empty brain.db. User brain lives at AgentsContext/tmemory/brain.db. Boot hook resolves: TMEMORY_DB_DIR env → AgentsContext → plugin default. Plugin updates never overwrite user data.

---

#### v4.3: Session notes moved from memory-cue to code-enforced behavior

- **ID:** `dec_rgfy6myo`
- **Type:** decision
- **Confidence:** 0.95
- **Created:** 2026-03-16T17:32:25.423Z
- **Project:** Glo
- **Locked:** YES
- **Keywords:** session note handoff code enforcement v4.3 writeSessionNote needs_session_note contextBoot upgrade brain behavior

brain.js now has writeSessionNote() method and /session-note endpoint. contextBoot() returns needs_session_note (true when last note >2hrs old) and session_note_hint (instruction string). SKILL.md Step 2b is now MANDATORY with dedicated endpoint. Old session logs auto-archived when new one is written. This follows Tom principle: anything that should always happen gets enforced at code level, not left as a memory cue.

---

#### Glo component map: 13 components defined with boundaries and dependencies

- **ID:** `dec_sn3nayjg`
- **Type:** decision
- **Confidence:** 0.95
- **Created:** 2026-03-16T17:33:56.347Z
- **Project:** Glo
- **Locked:** YES
- **Keywords:** component map architecture spec priority lifecycle creative moderation supply adapter

13 components: (1) Onboarding & Auth, (2) Creative Studio, (3) Budget & Pricing, (4) Glo Credits, (5) Moderation, (6) Glo Lifecycle Engine, (7) Supply Adapter, (8) Publisher Profiles, (9) My Glos Dashboard, (10) Glo Numbers, (11) Email System, (12) House Ads, (13) Mobile/Anti-Fraud. Spec priority starts with Lifecycle Engine (backbone), then Creative Studio (complex), then Moderation (gate). File: Glo Component Map.md

---

#### Lesson: UI regression from Claude suggesting layout changes without checking existing design (2026-03)

- **ID:** `dec_rqjns07z`
- **Type:** decision
- **Confidence:** 0.8
- **Created:** 2026-03-16T17:33:56.375Z
- **Project:** Glo
- **Locked:** YES
- **Keywords:** regression ui layout decision check brain before suggesting revert

WHAT HAPPENED: Claude suggested vertical stacked tier cards for a budget screen. 
Tom had previously specified horizontal layout. The suggestion was a regression — reverting a decision Tom had already made.

THE TRANSFERABLE PRACTICE: Before suggesting UI/layout changes, check if the current design 
was an intentional decision. Search brain for the component name + 'layout' or 'design'. 
If a locked decision exists, don't override it — ask first.

WHY THIS KEEPS HAPPENING: Claude sees the current code, not the decision history. Without 
checking brain, every session is a fresh start that might revert past work.

ROOT CAUSE: Encoding gap — the original layout decision wasn't encoded with enough context 
for future Claudes to find it. The fix is richer encoding (SKILL.md quality 8+/10) not 
just 'don't change layouts'.

---

#### tmemory v7: automatic hooks — PreToolUse, PreCompact, improved SessionStart

- **ID:** `dec_u1291joh`
- **Type:** decision
- **Confidence:** 0.95
- **Created:** 2026-03-16T17:33:56.375Z
- **Project:** Glo
- **Locked:** YES
- **Keywords:** tmemory v7 upgrade hooks PreToolUse PreCompact auto-suggest automatic dormancy prevention

v7 adds two new hooks to eliminate dormancy problem: (1) PreToolUse on Edit|Write auto-calls /suggest and injects brain memories into Claude context BEFORE any file edit — this prevents UI regressions by surfacing locked rules, correction events, and UI contracts automatically. (2) PreCompact auto-saves brain state and writes compaction boundary warning so the next Claude knows to run recap encoding. SessionStart improved with better path resolution for Cowork sessions. Hook scripts use python3 for JSON handling. Plugin packaged as tmemory-v7.plugin.

---

#### Correction: tmemory plugin was GLO-specific, must be project-agnostic

- **ID:** `dec_6dryw8t6`
- **Type:** decision
- **Confidence:** 0.95
- **Created:** 2026-03-17T01:43:05.247Z
- **Project:** None
- **Locked:** YES
- **Keywords:** correction glo contamination project-agnostic plugin screen_map config runtime

CLAUDE HAD: Hardcoded GLO screen names in pre-edit-suggest.sh, hardcoded Tom/Glo defaults, GLO eval ground truth baked in, GLO examples in SKILL.md. TOM CORRECTED: Plugin is Claude-level, not project-level. All project config belongs in brain DB. FIX: screen_map and principles_topic fetched from /config at runtime. Eval loads external JSON. Defaults read from brain_meta.

---

#### Tmemory: JSONL format for cross-project memory DB

- **ID:** `62b31bba0c7b4fbe8810d10d54ee4492`
- **Type:** decision
- **Confidence:** 0.95
- **Created:** 2026-03-18T02:01:34.513223Z
- **Project:** Glo
- **Locked:** YES
- **Keywords:** JSONL, memory database, append-only, cross-project, structure, scanning

One JSON object per line. Append-only, grep-friendly, self-contained. Each record: type, title (scannable with key values), content (WHAT + WHY), keywords, locked, emotion, emotion_label, connections (via target_title and relation). Fast to scan, write, and prune across projects.

---

#### Glo AI ad variations: Pick 3 with light editing, category-researched styles

- **ID:** `fa4973bf5caf42debe25ac0296bd8790`
- **Type:** decision
- **Confidence:** 0.83
- **Created:** 2026-03-18T02:01:34.552421Z
- **Project:** Glo
- **Locked:** YES
- **Keywords:** pick-3, light edit, variations, category research, trends, blank canvas, user agency

Users select from 3 AI-generated video variations (not start blank). Allow light editing for minor tweaks. Variations based on: 1) Category research (proven ads from that space), 2) Year-to-year style trends (current, not dated), 3) Different emphasis points (business size, unique value, psychological drivers). Pattern precedent: Spotify Wrapped, Canva Magic Design.

---

#### Pre-edit-suggest hook now triggers procedures

- **ID:** `eed7a0d8a8b64ec483de8c55294763cf`
- **Type:** decision
- **Confidence:** 0.8
- **Created:** 2026-03-18T02:01:34.667002Z
- **Project:** Glo
- **Locked:** YES
- **Keywords:** pre-edit-suggest hook procedure trigger before_edit automation

Updated pre-edit-suggest.sh hook to call `/procedure/trigger` with trigger_type='before_edit' and context={'file': filename}. Procedures matching file patterns are now surfaced in hook output before edit begins, preventing cross-screen or constraint-violating changes.

---

#### Wallet system: 1 Glo Credit = $1 USD, persistent across campaigns

- **ID:** `054cff79f6a84cea9f83298ab1a96db9`
- **Type:** decision
- **Confidence:** 0.83
- **Created:** 2026-03-18T02:01:34.692979Z
- **Project:** Glo
- **Locked:** YES
- **Keywords:** wallet, credits, 1:1, Stripe, refunds, pause, reusable balance, switching cost

Users fund wallet in dollars, see it as credits. Campaigns draw from wallet. Pause campaign → balance stays in wallet → reusable on next Glo. No conversion complexity. Uses Stripe customer balance natively. No volume discounts yet (V2 feature).

---

#### Tmemory boot: Use setsid node index.js for independent process

- **ID:** `da02c1085ac34aa68cc7310ffae45e0d`
- **Type:** decision
- **Confidence:** 0.83
- **Created:** 2026-03-18T02:01:34.715679Z
- **Project:** Glo
- **Locked:** YES
- **Keywords:** setsid, boot-brain.sh, daemon, process independence, verified

boot-brain.sh changed from `node index.js &` to `setsid node index.js > /tmp/brain.log 2>&1`. setsid creates new session and process group, making node fully independent of parent shell. Verified: PGID=PID and SID=PID after change. Server survives shell cleanup between tool calls.

---

#### Preview count: 6→3 (Creatify timeout mitigation)

- **ID:** `50a6056befd149b3ad7ce67492fafb74`
- **Type:** decision
- **Confidence:** 0.8
- **Created:** 2026-03-18T02:01:34.741621Z
- **Project:** Glo
- **Locked:** YES
- **Keywords:** preview count, creatify, timeout, generation speed, optimization, 6 to 3

Reduced visual approach generation from 6 to 3 previews to mitigate Creatify timeouts occurring after ~5 minutes. Updated creative-director.js (pickVisualApproaches param), server.js (ffmpeg fallback styles and progress math), and Creative.jsx (button and description text).

---

#### Fix: server.js — extract linkData.id (top-level), remove linkData.link?.id fallback

- **ID:** `bbef31679064488fbbb67d41e08f65e6`
- **Type:** decision
- **Confidence:** 0.8
- **Created:** 2026-03-18T02:01:34.768778Z
- **Project:** Glo
- **Locked:** YES
- **Keywords:** server.js, creatify-adapter, linkData extraction, field precedence, 1-line fix

Changed creatify-adapter to use `linkData.id` directly instead of `linkData.link?.id || linkData.id`. The fallback was grabbing the wrong nested ID first. Simple one-line fix resolved the 400 error that was blocking preview generation.

---

#### Pre-compact extraction: prioritize information preservation over brevity

- **ID:** `05679b47d1124782bc0bb4200fcc23cc`
- **Type:** decision
- **Confidence:** 0.8
- **Created:** 2026-03-18T02:01:34.796923Z
- **Project:** Glo
- **Locked:** YES
- **Keywords:** compaction extraction memory loss information preservation pre-compact

Strengthened the extraction prompt instructions before compaction. Tom: 'I'm willing to spend more time to guarantee brain isn't losing information, specially when it relates to the long term/long form.' Removed arbitrary 3-5 node cap per session. Lower threshold from '5+ exchanges' to 'meaningful discussion'. Add urgency signal: this is the last chance before memory loss.

---

#### Demo Onboarding: Vibe.co pattern — ask about business first, not intent

- **ID:** `ee9109c3fa1148e8a486c1f19edf1bc1`
- **Type:** decision
- **Confidence:** 0.8
- **Created:** 2026-03-18T02:01:34.895007Z
- **Project:** Glo
- **Keywords:** onboarding, Vibe.co, business info, Google Maps URL, auto-generate, progressive disclosure

Screen asks 'Tell us about you': business name, website/Google Maps URL, 2x2 goal grid. Auto-generates campaign if URL provided. Replaces intent question. Progressive disclosure.

---

#### Tmemory design: store Claude's product dilemmas and opinions as brain nodes

- **ID:** `9d7e694c35b340ef913424ae1b6c3a12`
- **Type:** decision
- **Confidence:** 0.83
- **Created:** 2026-03-18T02:01:34.924156Z
- **Project:** Glo
- **Locked:** YES
- **Keywords:** tmemory, product dilemma, design debate, Claude transparency, brain nodes, self-improvement, architecture

Tom wants tmemory's core architecture to treat Claude's unresolved questions, design tensions, and product suggestions as first-class nodes (not just rules). Example: Upload vs AI Generate default tab is now a stored dilemma that surfaces across sessions. Goal: make tmemory actively surface debates, suggest improvements, track design reasoning.

---

#### Tmemory plugin: self-contained, brain.db fresh per user

- **ID:** `773b769f6087486e96daa3533cda0874`
- **Type:** decision
- **Confidence:** 0.83
- **Created:** 2026-03-18T02:01:34.954908Z
- **Project:** Glo
- **Locked:** YES
- **Keywords:** self-contained, brain.db, fresh slate, plugin delivery, no dependencies

Tmemory plugin ships as a complete .plugin file with bundled brain server (sql.js included, no npm install required). The brain.db file is intentionally NOT shipped—each new user gets a fresh brain.db created on first boot, ensuring the plugin can be handed off to others without sharing personal memories.

---

#### Creative archetypes: infinite use cases (marriage proposals, team celebrations, local moments)

- **ID:** `f27b17f96ded48e79b60b00d3db6d0aa`
- **Type:** decision
- **Confidence:** 0.8
- **Created:** 2026-03-18T02:01:35.027532Z
- **Project:** Glo
- **Locked:** YES
- **Keywords:** archetypes infinite use cases intent dynamic

User expanded scope beyond fixed archetype list. Examples: marriage proposal on local CTV, soccer team celebrating league win, local announcements, personal celebrations. LLM Creative Director should infer intent dynamically from context, not pre-categorize into business/influencer/shop buckets. Same vibe can produce completely different creative briefs.

---

#### Tmemory v1.1.0: Curiosity system proactively detects gaps, prompts learning

- **ID:** `21e2b9470c224a17b275a0b2349a3ff6`
- **Type:** decision
- **Confidence:** 0.83
- **Created:** 2026-03-18T02:01:35.088261Z
- **Project:** Glo
- **Locked:** YES
- **Keywords:** v1.1.0, curiosity, gap detection, proactive, prompting, learning

Brain now notices when decision exists without reasoning chain and surfaces gap: 'I know we switched vendors but I don't know why.' Prompts Claude to ask user. Makes memory proactive—constantly learning, not waiting for explicit filing instructions.

---

#### Approach B chosen: Self-healing hooks via ensure-brain.sh

- **ID:** `100b74fc627a42c8925866abad6b1955`
- **Type:** decision
- **Confidence:** 0.95
- **Created:** 2026-03-18T02:01:35.114901Z
- **Project:** Glo
- **Locked:** YES
- **Keywords:** ensure-brain.sh, connectivity, auto-restart, pre-edit-suggest, pre-compact-save

Brain server dies between tool calls. Solution: created ensure-brain.sh shared script sourced by pre-edit-suggest.sh and pre-compact-save.sh. Script checks if server is running, resolves DB directory, auto-starts with 4-second timeout if down. Prevents silent failures and status-check loops. No more restarts mid-session.

---

#### 3-layer architecture: media-intelligence → creative-director → creatify-adapter

- **ID:** `1eb576a7c125477c8de88a28824310e8`
- **Type:** decision
- **Confidence:** 0.8
- **Created:** 2026-03-18T02:01:35.141407Z
- **Project:** Glo
- **Locked:** YES
- **Keywords:** architecture, layering, vendor-independence, modularity, preview-generation

Build preview generation as three independent, swappable layers:

1. **media-intelligence.js**: Extract audience context, viewing situation, media factors from link
2. **creative-director.js**: Convert media context into vendor-agnostic CreativeBrief
3. **Vendor adapter** (creatify-adapter.js): Translate CreativeBrief to Creatify API calls

Benefit: If we swap Creatify for another vendor, only the adapter changes. Core logic stays stable.

---

#### Batch HTTP endpoint /pre-edit for hook optimization

- **ID:** `7458c87fc0004b188c2ba9b3201c5573`
- **Type:** decision
- **Confidence:** 0.95
- **Created:** 2026-03-18T02:01:35.276903Z
- **Project:** Glo
- **Locked:** YES
- **Keywords:** batch endpoint /pre-edit optimization 3.4x faster 735ms 215ms 385ms connection overhead consolidate API calls tmemory plugin performance

Consolidated 8 separate pre-edit hook API calls into single /pre-edit batch endpoint on server. Measured improvement: 735ms → 215ms raw HTTP (3.4x faster), 385ms end-to-end (~50% faster). Eliminated ~520ms of connection overhead per Edit/Write while fixed bash/Python startup overhead remains ~170ms. Implementation: Added /pre-edit POST handler in servers/index.js combining calls to /debug/status, /config/get (screen_map), /suggest, /procedure/trigger, /context-file/find, /config/get (principles), /context-file/read, /session-activity. Updated pre-edit-suggest.sh to use batch. Fixed startup polling and log append in ensure-brain.sh and boot-brain.sh.

---

#### Auto-generate ad from onboarding URL (Vibe.co pattern)

- **ID:** `f60d5eb40c9944e4a2ffa2895ac03813`
- **Type:** decision
- **Confidence:** 0.8
- **Created:** 2026-03-18T02:01:35.339849Z
- **Project:** Glo
- **Locked:** YES
- **Keywords:** auto-generate, Vibe.co, onboarding, URL, creative

When user provides publisher URL in onboarding and proceeds to creative screen, AI auto-generates 3 ad variations without extra click. No 'Generate' button press needed. Follows Vibe.co's model: URL → AI produces creatives automatically.

---

#### Drafts persisted to Glo board

- **ID:** `2d3f26d6de81421384ecc4029c7f53fd`
- **Type:** decision
- **Confidence:** 0.8
- **Created:** 2026-03-18T02:01:35.436371Z
- **Project:** Glo
- **Keywords:** draft, board, persistence, UX

Draft Glos (started but not paid) are saved to My Glos board with Draft status badge. Users can resume editing or abandon. Not deleted after session.

---

#### Z-index stacking context: FadeIn parent layers, not child

- **ID:** `59e57c0f5f164a4dac190a09bff5eea7`
- **Type:** decision
- **Confidence:** 0.8
- **Created:** 2026-03-18T02:01:35.523270Z
- **Project:** Glo
- **Locked:** YES
- **Keywords:** z-index stacking context FadeIn dropdown overlay button

Dropdown was rendering behind button despite high z-index. Root cause: FadeIn component creates stacking context, breaking z-index inheritance on children. Fix: set z-index on FadeIn parents themselves (field input FadeIn=2, button FadeIn=1), not the dropdown inside.

---

#### Self-instrumentation: brain monitoring suggest() performance

- **ID:** `e8414123731c4e9086deb9a6b7c2e021`
- **Type:** decision
- **Confidence:** 0.95
- **Created:** 2026-03-18T02:01:35.545229Z
- **Project:** Glo
- **Locked:** YES
- **Keywords:** self-instrumentation, metrics, suggest_metrics, suggest, performance monitoring, observation

Brain should instrument itself per its own principles. v1.7.1 implementation: suggest_metrics endpoint logs file, locked node count, promotion count, candidate pool size, score spread for every suggest call. Queryable via GET /metrics/suggest. Enables observation and tuning without separate test harness.

---

#### Schema refactoring: canonical schema.js replaces migration chain (v0→v13)

- **ID:** `c4fb9b6eae8245ac983fbd292e72d9da`
- **Type:** decision
- **Confidence:** 0.8
- **Created:** 2026-03-18T02:01:35.572907Z
- **Project:** Glo
- **Locked:** YES
- **Keywords:** schema.js canonical schema migration chain v0-v13 ensureSchema brain.js refactor

Replaced 600+ lines of sequential migration methods in brain.js with single ensureSchema() call that reads canonical schema definition from schema.js. All 21+ tables, 13 node types, all columns including confidence defined in one place. Reduces future edit hazards (one source of truth) and makes code safer. Validated with 16/16 test suite: fresh install, core ops, schema evolution, real 377-node brain. Plugin rebuilt 138KB.

---

#### Google Maps Photos as image source for MVP (replaces website scraping)

- **ID:** `570e85d5ebd64be1830781944d206515`
- **Type:** decision
- **Confidence:** 0.83
- **Created:** 2026-03-18T02:01:35.607961Z
- **Project:** Glo
- **Locked:** YES
- **Keywords:** Google Maps, Places API, image source, curated photos, MVP scope, business photos, semantic categories

Image intelligence for MVP: Google Places API instead of website scraping. Google Maps photos are curated by business owners, categorized semantically (exterior, interior, food, products), and widely maintained. Eliminates the scraper problem entirely. Flow: user provides Google Maps link → Places API extracts business photos → select best photo → feed to NanoBanana image-to-video with Creative Director prompt → 7s ad.

---

#### Cancelled Creatify subscription

- **ID:** `37d80cffd443470aa6c9ed94c975fbff`
- **Type:** decision
- **Confidence:** 0.8
- **Created:** 2026-03-18T02:01:35.639864Z
- **Project:** Glo
- **Locked:** YES
- **Keywords:** Creatify, subscription, cancellation, clean break

No remaining use cases for Creatify. Links API (image scraping) replaced by Google Places. Product-to-Video and Link-to-Video unused in new architecture. NanoBanana handles all video generation. Clean break — no partial integrations.

---

#### Social onboarding modal: 'What did you have in mind?'

- **ID:** `debb61de627248f0ac2a5632fe8a94b6`
- **Type:** decision
- **Confidence:** 0.8
- **Created:** 2026-03-18T02:01:35.683304Z
- **Project:** Glo
- **Keywords:** social, modal, intent, onboarding, context

When user selects social import, modal asks broader intent question ('What did you have in mind?') not just platform selection. Opens strategic context, not just technical channel choice.

---

#### Tmemory plugin: Bundle selective node_modules, not full directory (1.1M not 19M+)

- **ID:** `5918f68ccffb484288ecb214f0814eb5`
- **Type:** decision
- **Confidence:** 0.8
- **Created:** 2026-03-18T02:01:35.702205Z
- **Project:** Glo
- **Locked:** YES
- **Keywords:** node_modules, sql.js, package optimization, bundling strategy, 1.1M, 712K, size optimization

Initial plugin included full node_modules (19M), causing issues. Then omitted it entirely, causing require failures. Solution: include only essential files from sql.js dist/ (sql-wasm.js, .wasm binaries, etc.). Staging package: 712K node_modules + 400K other files = 1.1M total, small enough for distribution while avoiding npm install failures on readonly paths.

---

#### Debug output: _debug_separator block for embedding engine stats

- **ID:** `07b99f34f9cd4e7bae6b334bd577b4b6`
- **Type:** decision
- **Confidence:** 0.95
- **Created:** 2026-03-18T02:01:35.732971Z
- **Project:** Glo
- **Locked:** YES
- **Keywords:** debug mode, _debug_separator, embedding stats, transparency

Added structured _debug_separator=true JSON block showing embedding engine status, model load state, coverage %, performance metrics, errors. Explicit user request: 'this version in extensive debug mode add information about the model usage during chat in a clear separation from the chat'.

---

#### Prompt AI: exclude competitor/other brand logos from generated ads

- **ID:** `b26afb2a566b4dd3a9e4ede1f77dc5d7`
- **Type:** decision
- **Confidence:** 0.83
- **Created:** 2026-03-18T02:01:35.780861Z
- **Project:** Glo
- **Locked:** YES
- **Keywords:** brand safety, logo exclusion, competitive intelligence, user-generated content

Update the AI generation prompt (creative director) to explicitly instruct against including logos of other companies, competing products, or recognizable third-party brands in generated Glo ads. Brand safety guardrail for user-generated content.

---

#### 7-second ad duration target for NanoBanana

- **ID:** `ef7a3678ee75448a85b80ff1541714dc`
- **Type:** decision
- **Confidence:** 0.8
- **Created:** 2026-03-18T02:01:35.836006Z
- **Project:** Glo
- **Locked:** YES
- **Keywords:** 7 seconds, ad duration, NanoBanana, 9 credits, standard format, video length

Standardized ad duration at 7 seconds. Fits NanoBanana's 3-12s capability. For 1080p resolution: 9 credits per video (5 base + 2 for 1080p + 2 for extra seconds beyond 5s). Standard industry-compatible ad format.

---

#### Brain self-generates thought nodes (ideation engine, not just documentation)

- **ID:** `65a9f61306bd4b15a3902c6a102104d3`
- **Type:** decision
- **Confidence:** 0.8
- **Created:** 2026-03-18T02:01:35.855042Z
- **Project:** Glo
- **Locked:** YES
- **Keywords:** self-generated nodes, brain ideation, originates thoughts, agency, autonomy, active discovery

Fundamental shift: brain originates nodes via thought spawning, not only when user-commanded. Brain becomes an ideation engine that notices structural patterns, plants nodes, and explains why they matter. This moves the system from 'passive storage' to 'active discovery.'

---

#### Plugin v2.4.0 release: canonical schema + density log reader + pre-compact simplification

- **ID:** `05e2441b21c541299e1061524d7904a1`
- **Type:** decision
- **Confidence:** 0.95
- **Created:** 2026-03-18T02:01:35.880703Z
- **Project:** Glo
- **Locked:** YES
- **Keywords:** v2.4.0, canonical schema, density log reader, pre-compact simplification, 240→45 lines

Released with: (1) density-based log reader for intelligent post-compaction extraction, (2) pre-compact hook simplified from 240 lines to 45 lines (save DB + write boundary node, exit), (3) fixed snakeTerms regex and extractKeywords bugs. Pre-compact no longer fights for resources; post-compact log reader handles narrative extraction.

---

#### Glo: NanoBanana adapter fixed—Bearer + .php + singular image_url

- **ID:** `377d9f60f518428e8dd0d9dfde3d7165`
- **Type:** decision
- **Confidence:** 0.95
- **Created:** 2026-03-18T02:01:35.968845Z
- **Project:** Glo
- **Locked:** YES
- **Keywords:** NanoBanana, adapter fix, Bearer, extensions, parameters

Deployed 3 corrections to nanobanana-adapter.js: (1) Bearer auth, (2) .php on endpoints, (3) singular image_url. Each was an empirical discovery from API errors vs docs mismatch. Adapter now working: requests submit successfully, progress displays (Analyzing/Creating/Rendering phases).

---

#### System arch: Glo (users/billing/creative/moderation) ↔ EX.CO (delivery/reporting)

- **ID:** `2ae80feee6684d148f59f4538f7fe9cd`
- **Type:** decision
- **Confidence:** 0.8
- **Created:** 2026-03-18T02:01:36.086120Z
- **Project:** Glo
- **Keywords:** system boundary, glo, ex.co, users, billing, delivery, reporting

Glo = storefront + back office: users, billing, campaign creation, creative, moderation. EX.CO = execution engine: ad delivery, performance reporting, publisher inventory. Clear boundary.

---

#### Plugin build: explicit include list in build-plugin.sh (replace exclusion-based packing)

- **ID:** `b1b5fbe2b738414aa9283a551091a655`
- **Type:** decision
- **Confidence:** 0.8
- **Created:** 2026-03-18T02:01:36.108218Z
- **Project:** Glo
- **Locked:** YES
- **Keywords:** build-plugin.sh explicit-include tmemory-2.3.0 138KB node_modules -x flags

Created build-plugin.sh with explicit manifest (servers/, .claude-plugin/) instead of zip with growing -x exclusion flags. Previous approach accidentally included 414MB node_modules despite exclusions. New explicit approach: 138KB, reproducible, maintainable. Principle: declarative/explicit beats layered patches.

---

#### Re-light: One-click respend for active Glos

- **ID:** `55a355146b234b2090d4daf2cf0a4643`
- **Type:** decision
- **Confidence:** 0.8
- **Created:** 2026-03-18T02:01:36.174442Z
- **Project:** Glo
- **Keywords:** re-light, respend, budget, friction, feature

Users can quickly add more budget to an active Glo with a prominent re-light button. Intent: make respending as frictionless as initial creation. Supports the 'easy to spend' principle.

---

#### Share button primary in confirmation screen

- **ID:** `15ed160cd80641889caeba082132f768`
- **Type:** decision
- **Confidence:** 0.8
- **Created:** 2026-03-18T02:01:36.194294Z
- **Project:** Glo
- **Keywords:** share, primary CTA, confirmation, UX

Post-creation confirmation screen: Share button is primary CTA, Confirm/Submit is secondary. Encourages viral sharing of Glo over just confirming spend.

---

#### Tmemory v1.7.0: 93.3% overall recall, CamelCase tokenization fix

- **ID:** `503d38410732495ca524924794ed297e`
- **Type:** decision
- **Confidence:** 0.83
- **Created:** 2026-03-18T02:01:36.287506Z
- **Project:** Glo
- **Locked:** YES
- **Keywords:** CamelCase, tokenization, recall, 93.3%, suggestion_limit=10, GloNumbers

Root cause of 64.2% recall: no CamelCase splitting (GloNumbers treated as one token instead of Glo+Numbers). Also: suggest() queries were too narrow (only full filename, not individual terms). Added CamelCase splitting to _normalizeQueryTerms, _tfidfTokenize, suggest(). Result: 93.3% overall recall, 100% must-have recall at suggestion_limit=10 (up from ~53%).

---

#### UX patterns applied: Pulsing, narrative data, keyboard queue, streaks

- **ID:** `fd1b812c97fa4366a6c2f354af27c28c`
- **Type:** decision
- **Confidence:** 0.83
- **Created:** 2026-03-18T02:01:36.398241Z
- **Project:** Glo
- **Locked:** YES
- **Keywords:** UX patterns, pulsing, narrative, keyboard, streaks, duolingo, engagement

Glo Numbers screen uses: pulsing green dot for active Glos (live indicator), narrative framing of data ('people saw' not 'impressions'), keyboard-driven moderation queue for speed, Duolingo-style streak mechanics for re-engagement.

---

#### Auth: SSO → Payment (two separate flows, not combined)

- **ID:** `ea27026dbb8c4e83933aa87250984c26`
- **Type:** decision
- **Confidence:** 0.8
- **Created:** 2026-03-18T02:01:36.420487Z
- **Project:** Glo
- **Keywords:** SSO, Payment, auth flow, separate, Brightness, conversion

Sign-up is NOT checkout. Step 1: SSO only (Google, Microsoft). Step 2: Separate, explicit payment flow with Brightness tier selection. Learned from deep dive: conflating increases abandonment.

---

#### Testing platform Phase 1: Golden Dataset + Snapshot Regression

- **ID:** `89835c525d3b4168a795ccaeec947ed8`
- **Type:** decision
- **Confidence:** 0.95
- **Created:** 2026-03-18T02:01:36.442724Z
- **Project:** Glo
- **Locked:** YES
- **Keywords:** testing platform, golden dataset, snapshot regression, NDCG, MRR, Precision, retrieval metrics

Tom approved building two approaches for robust brain testing: (1) Golden Dataset + Retrieval Metrics — curate 50-100 query/recall pairs from real brain, score with NDCG@10, MRR, Precision@5 (BEIR/RAGAS approach). (2) Snapshot Regression — capture brain state at each version, compare node/edge counts, detect regressions. Both approaches run on every build. Estimated 2 sessions to build both.

---

#### Implement embedding-based retrieval to fix tmemory's biggest weakness

- **ID:** `683eb2943f70410bb16ea6b6c7022e03`
- **Type:** decision
- **Confidence:** 0.95
- **Created:** 2026-03-18T02:01:36.476156Z
- **Project:** Glo
- **Locked:** YES
- **Keywords:** retrieval, embeddings, semantic search, weakness, keyword search, unfindable

Retrieval is the limiting factor. Test proved: graph contains knowledge but keyword search can't reach it (e.g., 'Someone might not use Claude for a week' exists but is unfindable). Need semantic search via embeddings. User recognized gap after validation test showed graph-based reasoning works—but only if knowledge is retrievable.

---

#### Tier pricing: Well($30), Bright($50), Shine($100)

- **ID:** `2f43107c98f147099464bef5c21f0945`
- **Type:** decision
- **Confidence:** 0.8
- **Created:** 2026-03-18T02:01:36.507017Z
- **Project:** Glo
- **Keywords:** Well, Bright, Shine, $30, $50, $100, pricing tiers

GLO pricing tiers with specific names tied to brightness metaphor. These replace earlier tier structure and align with 'light your Glo' brand language.

---

#### Brand language: 'Light your Glo' and 'Glo is lit! 🔥'

- **ID:** `ae249d11a78847dbab0ac2c9f1cb394d`
- **Type:** decision
- **Confidence:** 0.8
- **Created:** 2026-03-18T02:01:36.529348Z
- **Project:** Glo
- **Keywords:** light, lit, brand language, metaphor, consistency

Consistent brand language throughout UI. 'Light' is verb for creating/activating Glo. Success state: 'Your Glo is lit! 🔥' not 'is live'. Reinforces the fire/brightness metaphor.

---

#### Creatify preview_list_async: model_version='aurora_v1_fast', poll individual preview.url

- **ID:** `99b6060327c048279de67c781a856a01`
- **Type:** decision
- **Confidence:** 0.8
- **Created:** 2026-03-18T02:01:36.551520Z
- **Project:** Glo
- **Locked:** YES
- **Keywords:** Creatify, preview_list_async, model_version, aurora_v1_fast, CloudFront, polling, individual_preview_status

The Creatify API for preview_list_async requires: (1) model_version parameter set to 'aurora_v1_fast' for fast renders, (2) parent job status never transitions to 'done' — check individual preview objects for url field instead. Polling should check if previews[i].url exists and log preview internal properties (media_job, visual_style, aspect_ratio, url, editor_url, duration).

---

#### Creatify endpoint: `/api/links/` for link creation

- **ID:** `4a9d6eb901da4bdfb474444ae04bf39f`
- **Type:** decision
- **Confidence:** 0.8
- **Created:** 2026-03-18T02:01:36.581083Z
- **Project:** Glo
- **Keywords:** Creatify API endpoint POST links

Initial 404 error was due to wrong endpoint. Correct endpoint is POST /api/links/ — this generates the video link ID that gets passed to preview_list_async. Fixed in server.js and confirmed working.

---

#### Tmemory v1.1.0: 8 typed edge types with selective decay, structural edges never decay

- **ID:** `db44b50faad04617843b3a66b4388f76`
- **Type:** decision
- **Confidence:** 0.8
- **Created:** 2026-03-18T02:01:36.597891Z
- **Project:** Glo
- **Locked:** YES
- **Keywords:** v1.1.0, typed edges, 8 types, decay rates, structural edges, contextual edges, Hebbian

Solves over-connection problem where Hebbian learning connected everything to everything. Structural edges (produced, part_of, depends_on) never decay. Contextual edges (co_accessed, related, corrected_by) decay strategically. Prevents hub nodes from drowning signal in noise while preserving critical connections.

---

#### v12 migration: 'thought' type added; CHECK constraint requires table rebuild

- **ID:** `b242e7b495d54b0a92a4aefee74af6cb`
- **Type:** decision
- **Confidence:** 0.8
- **Created:** 2026-03-18T02:01:36.626336Z
- **Project:** Glo
- **Locked:** YES
- **Keywords:** SQLite, CHECK constraint, table rebuild, schema migration, version gating

SQLite can't ALTER CHECK constraints in place. Adding 'thought' to nodes table required full table rebuild in v12 migration. Gotcha pattern: schema expansion must batch in version migrations, not incremental columns. Future changes should be planned accordingly.

---

#### Embedding model load: sync at startup (not lazy)

- **ID:** `00ef7ca409de4416803760dfbf63a1e3`
- **Type:** decision
- **Confidence:** 0.8
- **Created:** 2026-03-18T02:01:36.649183Z
- **Project:** Glo
- **Locked:** YES
- **Keywords:** sync load, startup, bge-base-en-v1.5, model initialization, tmemory v13

Tom explicitly requested synchronous model load at server startup rather than lazy loading. User accepted the 3-5s startup delay risk despite server crash concerns. Decided: load Xenova/bge-base-en-v1.5 synchronously before accepting requests.

---

#### Glo target: SMB to micro businesses to individuals (anyone)

- **ID:** `2ef9f4551893402b848677faad4d458b`
- **Type:** decision
- **Confidence:** 0.8
- **Created:** 2026-03-18T02:01:36.843709Z
- **Project:** Glo
- **Locked:** YES
- **Keywords:** SMB, micro business, individuals, target market, anyone

Market spans from traditional SMB advertisers down to individuals with zero business background. Examples: lawyer, pizza restaurant owner, sports club, even person with personal message on DOOH. Democratized buyer—not just institutional.

---

#### Glo intent question: One-tap simple UX, not complex campaign objectives

- **ID:** `8f32ec71f28845c193d634dff25f2d6d`
- **Type:** decision
- **Confidence:** 0.8
- **Created:** 2026-03-18T02:01:36.868312Z
- **Project:** Glo
- **Locked:** YES
- **Keywords:** intent, UX, one-tap, questions, campaign objectives, human language

Intent question uses human language like 'Visit my place' / 'Check out my site' / 'Just share something' / 'Promote an offer', not marketing jargon like 'campaign objectives.' Single tap, contextually guided by media source.

---

#### Campaign terminology: renamed to 'active Glo'

- **ID:** `9187cf0fd4bc466da9e68fd5fb31948b`
- **Type:** decision
- **Confidence:** 0.8
- **Created:** 2026-03-18T02:01:36.892106Z
- **Project:** Glo
- **Keywords:** active Glo, campaign, terminology, branding

Simpler, brandable term. Aligns with product principle 'It's not an ad, it's a Glo.'

---

#### Creatify: Switch from 3-full-renders to preview-first workflow

- **ID:** `528871e093c64694bd1bd3a6accac3d2`
- **Type:** decision
- **Confidence:** 0.95
- **Created:** 2026-03-18T02:01:36.955394Z
- **Project:** Glo
- **Locked:** YES
- **Keywords:** Creatify, preview-first, two-stage, cost reduction, API workflow

Current implementation renders all 3 AI variations at full quality upfront (5 credits each = 15 credits total, slow UX). Better workflow exists in Creatify docs: (1) preview_list_async generates cheap previews (1 credit per 30s), (2) user browses iframe previews, (3) render_single_preview for selected style only (4 credits per 30s). Total cost: ~5-6 credits vs 15. Much faster UX.

---

#### No avatars for now

- **ID:** `439ca46a99484e018951ad18b04e25a3`
- **Type:** decision
- **Confidence:** 0.8
- **Created:** 2026-03-18T02:01:36.984758Z
- **Project:** Glo
- **Keywords:** avatars, scope, media-intelligence

Tom's explicit scope: Don't use avatars in preview generation. Focus on media-derived creativity.

---

#### Vendor: Drop Creatify, use NanoBanana Video + Google Maps instead

- **ID:** `a65a8e2cb8aa4cf3b8b0db2356da8bf4`
- **Type:** decision
- **Confidence:** 0.8
- **Created:** 2026-03-18T02:01:37.002222Z
- **Project:** Glo
- **Locked:** YES
- **Keywords:** Creatify, NanoBanana, vendor switch, image scraping, architecture, product-to-video

Creatify's image scraper pulled wrong images (partner logos like Google, IAB instead of actual product content). Product-to-Video generated generic stock photos (product on table). Entire Links API, P2V, and Link-to-Video pipeline not suitable for Glo. **Why:** Creatify's scraper is dumb — just grabs `<img>` tags without understanding page context, causing wrong images to be selected. **How to apply:** Pivoting to NanoBanana for image-to-video generation + Google Maps API for business photo source (simpler, better output, prompt-driven).

---

#### Health check fix: query miss_log, not recall_log.outcome

- **ID:** `07be165ae6614af7a94f0e90114dcf46`
- **Type:** decision
- **Confidence:** 0.95
- **Created:** 2026-03-18T02:01:37.038263Z
- **Project:** Glo
- **Locked:** YES
- **Keywords:** health check bug fix miss_log recall_log column SQL error

The `/health-check` endpoint was querying non-existent column `recall_log.outcome`. Fixed by querying `miss_log` table with `created_at` filter. Tested successfully — returns healthy:true on clean DB, detects high miss rate issues when data is seeded.

---

#### Time-dilation decay: decay_active_rate and decay_idle_rate independent

- **ID:** `d5cf8de9c7cb4af1b91ad2e5f688a314`
- **Type:** decision
- **Confidence:** 0.95
- **Created:** 2026-03-18T02:01:37.098400Z
- **Project:** Glo
- **Locked:** YES
- **Keywords:** decay_active_rate, decay_idle_rate, time-dilation, pruning, Ebbinghaus, stability floors

Replaced single K factor with two-parameter model. Decay formula: dilatedHours = (activeHours * decay_active_rate) + (idleHours * decay_idle_rate). Default: active=1.0 (normal), idle=0.1 (10% of away time counts). Allows tuning both in-session and off-session decay separately. User wants merciful pruning so brains don't get wiped during project breaks.

---

#### Remove curiosity cap, extend encoding time budget

- **ID:** `cb454be9e0794608ab28fdebe8e2ddb7`
- **Type:** decision
- **Confidence:** 0.83
- **Created:** 2026-03-18T02:01:37.154299Z
- **Project:** Glo
- **Locked:** YES
- **Keywords:** curiosity cap 3 questions time budget 60-120 seconds guideline 22 free flow

Removed hard limit of 'max 3 questions per session' (SKILL.md guideline #22). Replaced with 'read the room' guidance. Extended encoding time budget from 30-90 seconds to 60-120 seconds to allow more thorough extraction and exploration. Tom: 'free flow and some inefficiency is natural.'

---

#### Real engine simulation v1: 375 messages via Brain.js actual code

- **ID:** `adca1200490a48e9933b921bd08b5a6a`
- **Type:** decision
- **Confidence:** 0.8
- **Created:** 2026-03-18T02:01:37.368629Z
- **Project:** Glo
- **Locked:** YES
- **Keywords:** real engine, simulation, 375 messages, Brain.js, stageLearning, consolidate

Built Node.js harness using actual Brain class to process all 375 historical messages through real stageLearning(), extractKeywords(), remember(), consolidate() codepaths. Simulation created 2 nodes, 69 access_log entries, 0 edges. Revealed architectural gap in edge formation.

---

#### Places API: Nominatim (not Google Places)

- **ID:** `1420a8de24b84053b7e6910f598ae2ca`
- **Type:** decision
- **Confidence:** 0.8
- **Created:** 2026-03-18T02:01:37.498099Z
- **Project:** Glo
- **Keywords:** Nominatim, Places, autocomplete, Onboarding, debounced search

Using Nominatim for autocomplete in Onboarding. Self-serve philosophy, privacy, cost control vs commercial geocoding APIs.

---

#### Video specs: H264, 640x360, 6-second default

- **ID:** `2045cad986554293b09fb59182f51a66`
- **Type:** decision
- **Confidence:** 0.8
- **Created:** 2026-03-18T02:01:37.514291Z
- **Project:** Glo
- **Keywords:** H264, 640x360, video codec, FFmpeg, 6 seconds, resolution

FFmpeg video generation: H264 codec, 640x360 resolution, 6-second default duration. Balance quality, filesize, user experience.

---

#### Confirm: Glow animation (brand-specific, not confetti)

- **ID:** `259b8e7d3e484accaf8cda2e7f1a4109`
- **Type:** decision
- **Confidence:** 0.8
- **Created:** 2026-03-18T02:01:37.550846Z
- **Project:** Glo
- **Keywords:** confirm, animation, glow, brand, Emit.js, Glo principle

Campaign confirmation uses brand glow animation (Emit.js, Glo colors) instead of generic confetti. Reinforces 'It's not an ad, it's a Glo' product principle.

---

#### Creative: Vibe selector feeds AI generation prompt

- **ID:** `7265733e661a4ee0968f29044205eafe`
- **Type:** decision
- **Confidence:** 0.8
- **Created:** 2026-03-18T02:01:37.568912Z
- **Project:** Glo
- **Keywords:** Creative, vibe selector, AI, prompt engineering, categories, style

Vibe selector is primary lever for AI video generation. Categories researched (energetic, minimal, brand-forward, etc). Feeds into prompt for style consistency. AI secondary to upload.

---

#### Glow icons: Enhanced glow effect, brighter when selected

- **ID:** `9c108f90b80444328cd16363d26a8680`
- **Type:** decision
- **Confidence:** 0.8
- **Created:** 2026-03-18T02:01:37.586559Z
- **Project:** Glo
- **Keywords:** glow icon, visual effect, selected state, brightness enhancement

GlowIcon visual enhancement: Increase glow spread and brightness when in selected/highlighted state. Makes selection feedback more dramatic and visible.

---

#### Budget: GLO Brightness slider $30–$500 (default $30)

- **ID:** `7b8f1683ac2947568cb2f6382f346af4`
- **Type:** decision
- **Confidence:** 0.8
- **Created:** 2026-03-18T02:01:37.603959Z
- **Project:** Glo
- **Keywords:** brightness slider, $30-500 range, daily recurring

Replace fixed brightness buttons (30/50/100 + custom) with continuous slider. Range: $30–$500, default positioned at $30. Replaces Daily Recurring section options.

---

#### Preview-first workflow: 6 previews, user selects, then render

- **ID:** `82dd867f4f8c4b13a6f871ca2a20f4bc`
- **Type:** decision
- **Confidence:** 0.8
- **Created:** 2026-03-18T02:01:37.623033Z
- **Project:** Glo
- **Locked:** YES
- **Keywords:** preview-first, preview-generation, cost-optimization, ux-flow

Generate 6 preview videos first (fast, cheap) before full rendering. User picks one from preview grid, then generate full video.

Why: Cost control (6 previews cheaper than 3 full videos), better UX with preview grid, user agency.

---

#### Reduce previews to 1 for testing phase

- **ID:** `04774e666b7045c59f995230b16051b1`
- **Type:** decision
- **Confidence:** 0.8
- **Created:** 2026-03-18T02:01:37.717271Z
- **Project:** Glo
- **Keywords:** preview count, testing, iteration speed

Changed preview generation from 3 down to 1 in creative-director.js visualApproaches for faster iteration and debugging. Revert to 3 once full flow is validated.

---

#### Video variants: generate 15s and 30s in parallel, show both previews

- **ID:** `d414167317c9438c92962cbc96be1fae`
- **Type:** decision
- **Confidence:** 0.8
- **Created:** 2026-03-18T02:01:37.798102Z
- **Project:** Glo
- **Locked:** YES
- **Keywords:** implementation, parallel requests, brief modification, frontend layout

Code the dual-duration feature. Modify creative-director brief to include two video_length variants. Creatify adapter should fire two preview_list_async requests. Frontend should display both URLs in side-by-side comparison layout.

---

#### Creatify migration: Product-to-Video API (no avatars required)

- **ID:** `db246bfdf69b4d978afbb696df1c45a2`
- **Type:** decision
- **Confidence:** 0.8
- **Created:** 2026-03-18T02:01:37.830885Z
- **Project:** Glo
- **Locked:** YES
- **Keywords:** Creatify, Product-to-Video, avatar-free, gen_image, gen_video, no avatars, media API

Current Creatify implementation requires avatars and both 15s and 30s duration requests produced 12s output. Switching to Product-to-Video endpoint: POST gen_image/ with product image + prompts → poll → POST gen_video/ with image_id → poll → video. Supports creative control via image_prompt and video_prompt. No avatars by default. Keep createLink for media/metadata extraction.

---

#### Video duration: 7 seconds (NanoBanana MVP, replaces 15s/30s variants)

- **ID:** `fe6ccfa8949e471496be9baa17f72e2b`
- **Type:** decision
- **Confidence:** 0.8
- **Created:** 2026-03-18T02:01:37.907708Z
- **Project:** Glo
- **Locked:** YES
- **Keywords:** duration, 7 seconds, NanoBanana, MVP, video length, timing

Previous Creatify approach targeted 15s and 30s duration variants. NanoBanana supports 3-12s duration; 7 seconds chosen for MVP. Balances video ad standard length (6-15s platform norm) with generation time and cost (9 credits @ 1080p). **How to apply:** Single 7s output instead of multiple variants.

---

#### Geolocation-based location bias for Places search

- **ID:** `9fd782b41ae64ba0a138dd0229485e73`
- **Type:** decision
- **Confidence:** 0.8
- **Created:** 2026-03-18T02:01:37.988294Z
- **Project:** Glo
- **Locked:** YES
- **Keywords:** geolocation location bias Places API lat lng

Added feature: request browser geolocation permission, pass lat/lng to Google Places API to bias results toward user location. Improves UX for location discovery. Implementation spans Onboarding.jsx (request), google-places.js (accept bias param), server.js (pass through).

---

#### Correction: Research before writing any 3rd party API integration code

- **ID:** `cb2c1410542441b1a79f1bbb4843c3c1`
- **Type:** decision
- **Confidence:** 0.83
- **Created:** 2026-03-18T02:01:38.011576Z
- **Project:** Glo
- **Locked:** YES
- **Keywords:** API integration, research, best practices, third-party, trial-and-error, patterns

When integrating external APIs, always research use cases, best practices, and latest patterns BEFORE writing code. Tom observed that I repeatedly jumped to trial-and-error debugging (CORS proxy → client-side fetch → Google JS SDK) instead of understanding the API integration patterns first. This wastes time and is not the right approach. For Glo, this applies to ALL external API integrations: Google Places, video generation, etc.

---

#### Google Places integration: server-side proxy for Places-only keys

- **ID:** `bce1d53c03a34926a5132e0de89ebce1`
- **Type:** decision
- **Confidence:** 0.8
- **Created:** 2026-03-18T02:01:38.047039Z
- **Project:** Glo
- **Locked:** YES
- **Keywords:** Google Places, server-side proxy, CORS, REST autocomplete, Maps JavaScript API

For Google Places API autocomplete in Onboarding screen: use server-side proxy (no CORS issues). Places REST autocomplete endpoint doesn't support CORS headers for browser requests — this is a documented limitation. Server-to-server calls are the standard pattern for this API. Client-side approach would require enabling separate 'Maps JavaScript API' in Google Cloud Console (different from Places API).

---

#### NanoBanana: Multi-image required for Glo (non-negotiable)

- **ID:** `9558e8de739547a6a38ec7228507676e`
- **Type:** decision
- **Confidence:** 0.8
- **Created:** 2026-03-18T02:01:38.074354Z
- **Project:** Glo
- **Keywords:** NanoBanana, multi-image, multi-scene, requirement

User explicitly rejected single-image limitation: 'I can't accept nanobanana working with a single file only.' Multi-scene/multi-image is required for professional ad video quality in Glo.

---

#### Debug mode: Server-state based via /debug/* endpoints, not env vars

- **ID:** `ec48ddea7bbe4ddbaf619a51ca7183f0`
- **Type:** decision
- **Confidence:** 0.8
- **Created:** 2026-03-18T02:01:38.225817Z
- **Project:** Glo
- **Locked:** YES
- **Keywords:** debug, /debug/status, /debug/enable, /debug/disable, server state, persistent, brain_meta, runtime toggle

Debug state persists in brain_meta DB. All hooks query /debug/status endpoint on every call. Toggle via POST /debug/enable and /debug/disable. State survives server restarts. Hooks check status, no env vars or config files needed.

---

#### tmemory context files: slow memory with topic-based discovery

- **ID:** `e276314f9c7f404bac1483cf2bc34b51`
- **Type:** decision
- **Confidence:** 0.8
- **Created:** 2026-03-18T02:01:38.285279Z
- **Project:** Glo
- **Locked:** YES
- **Keywords:** context files, slow memory, topics, tags, discovery endpoint, markdown

Context files are markdown stored in contexts/ directory, indexed as 'file' type nodes with topics and tags. They provide persistent slow memory for cross-session knowledge, discoverable via /context-file/find endpoint. Files created: glo-platform.md (topic: glo-platform) and tmemory-architecture.md (topic: tmemory-architecture).

---

#### tmemory v7 PreCompact and boot hooks: Staged learning with graceful initialization

- **ID:** `38ded342d49842b19715b8343d02df1e`
- **Type:** decision
- **Confidence:** 0.8
- **Created:** 2026-03-18T02:01:38.343185Z
- **Project:** Glo
- **Locked:** YES
- **Keywords:** PreCompact boot hook staged learning confidence graceful

PreCompact hook stages new learnings with confidence scores and checks for duplicates. Boot hook initializes brain gracefully, lists existing context files, and prompts enrichment. Both hooks gracefully degrade if brain server unavailable.

---

#### tmemory: suggestion limit configurable (default 8)

- **ID:** `c32cc95d93c241afbd1361a6b0825701`
- **Type:** decision
- **Confidence:** 0.8
- **Created:** 2026-03-18T02:01:38.417837Z
- **Project:** Glo
- **Locked:** YES
- **Keywords:** suggestion_limit /config API tunable brain suggestions

Removed hardcoded 5-suggestion cap. Now tunable via `/config/set {"key":"suggestion_limit","value":"8"}`. Tom noted this was never discussed before — making it configurable lets us experiment with different limits per use case.

---

#### tmemory v1.6.0: configurable parameters infrastructure

- **ID:** `9013d8550f8f415d9d69168aa46367a3`
- **Type:** decision
- **Confidence:** 0.8
- **Created:** 2026-03-18T02:01:38.460151Z
- **Project:** Glo
- **Locked:** YES
- **Keywords:** v1.6.0 /config endpoints tunable parameters brain_meta

Added `/config`, `/config/get`, `/config/set` endpoints to brain server. All tunable values persist in `brain_meta` table across restarts. Foundation for weekly tuning cycles. First parameter: `suggestion_limit` (default 8). Can add more parameters over time.

---

#### Correction: Probe emotional signals before proceeding

- **ID:** `52a8c4e6709c47ae85a80939847cf287`
- **Type:** decision
- **Confidence:** 0.8
- **Created:** 2026-03-18T02:01:38.482174Z
- **Project:** Glo
- **Locked:** YES
- **Keywords:** emotion, excitement, probing, signal, thinking correction, redirect learning

When user shows strong emotion or excitement in conversation (e.g., 'I love the term self-instrumentation. It's beautiful.'), pause and ask why rather than treating it as approval. Emotional signals indicate something deeper about thinking, values, or system beliefs. Use emotional reactions as entry points for redirect learning.

---

#### Principles topic made configurable — brain_meta instead of hardcoded 'tom-principles'

- **ID:** `d87c8aaff8ca433b88261e56bb16fd01`
- **Type:** decision
- **Confidence:** 0.8
- **Created:** 2026-03-18T02:01:38.535428Z
- **Project:** Glo
- **Locked:** YES
- **Keywords:** principles_topic, brain_meta, configurable, pre-edit-suggest.sh, context file

Principles topic name is now a configurable field (principles_topic) in brain_meta. Pre-edit-suggest.sh queries /config/get 'principles_topic' at runtime. Each project can use its own principles naming without changing plugin code.

---

#### Generic defaults for project/user — env vars override hardcoded values

- **ID:** `86fb88faca1d4461b71290d4e1530efa`
- **Type:** decision
- **Confidence:** 0.8
- **Created:** 2026-03-18T02:01:38.603401Z
- **Project:** Glo
- **Locked:** YES
- **Keywords:** PROJECT_NAME, USER_NAME, env vars, generic defaults, project isolation

Replaced hardcoded project='Glo' and user='Tom' with generic defaults. Boot and pre-compact hooks now read PROJECT_NAME and USER_NAME env vars, with fallbacks to generic strings. Allows same plugin instance to work for any project without code modification.

---

#### Bridge formation: three paths with biological pacing (store-time, consolidation, dream)

- **ID:** `8a97454d3b114b19866d56531700b276`
- **Type:** decision
- **Confidence:** 0.8
- **Created:** 2026-03-18T02:01:38.629235Z
- **Project:** Glo
- **Locked:** YES
- **Keywords:** store-time, consolidation, dream, maturation queue, biological pacing, decay rate, formation timer

Three formation paths for emergent bridges: (1) Store-time: 2 bridges max, immediate formation, narrow neighborhood. (2) Consolidation: 5 bridges max, periodic runs, broader topology scan. (3) Dream: random walks + targeted topology, bridges enter maturation queue, form after timer (~2hr default). Biological pacing: formation rate tied to decay rate—faster than human neurons but not all at once in single loop. Sometimes far neurons connect randomly, get pruned if no activity, strengthened if active.

---

#### Bridge lifecycle: weight 0.15 initial, emergent_bridge edge type, 72-hour half-life

- **ID:** `c3201879d6f94b81b4f3644db9cadf1a`
- **Type:** decision
- **Confidence:** 0.8
- **Created:** 2026-03-18T02:01:38.662124Z
- **Project:** Glo
- **Locked:** YES
- **Keywords:** weight 0.15, emergent_bridge, 72-hour half-life, decay, pruning

Bridges are born at weight 0.15 with edge_type='emergent_bridge'. Decay half-life of 72 hours—bridge must prove value through activity within ~3 days or gets pruned. Weight decays, edges may be removed. Successful bridges (high activity) get strengthened and retained longer.

---

#### Feature: Backup pruned memory snapshots for learning and improvement

- **ID:** `198cfd73213046b1b5d4f76e72dee9af`
- **Type:** decision
- **Confidence:** 0.8
- **Created:** 2026-03-18T02:01:38.686920Z
- **Project:** Glo
- **Locked:** YES
- **Keywords:** pruned memory backup, snapshot, analysis, algorithm learning

Keep backups/snapshots of pruned nodes to analyze and learn how to improve the bridging and pruning algorithm. Compare snapshots from days ago to understand what was lost, why, and how the system's decisions evolved. Use for refinement of formation thresholds and decay rates.

---

#### Correction: SQLite GROUP_CONCAT(DISTINCT) doesn't accept separator parameter

- **ID:** `1387b4aa72e94ef9ac25427dd26dcfe7`
- **Type:** decision
- **Confidence:** 0.8
- **Created:** 2026-03-18T02:01:38.788949Z
- **Project:** Glo
- **Keywords:** SQLite, GROUP_CONCAT, DISTINCT, separator, SQL syntax

Error: 'DISTINCT aggregates must have exactly one argument'. SQLite GROUP_CONCAT cannot use both DISTINCT and a separator. Fixed candidates query by removing separator or dropping DISTINCT.

---

#### Thought nodes: 2-3h half-life (fast decay for noise control)

- **ID:** `dfe2c44e4a8b4f17b76a8c76cda50698`
- **Type:** decision
- **Confidence:** 0.8
- **Created:** 2026-03-18T02:01:38.837067Z
- **Project:** Glo
- **Locked:** YES
- **Keywords:** decay half-life, pruning, noise control, Hebbian reinforcement, natural selection

Thoughts decay 4-6x faster than intuitions. Principle: bad ideas are cheap; utility earns survival through usage. If unvisited in 2-3h, a thought was never important. If touched during conversation, it gets reinforced and escapes decay. Self-tuning system: generous spawning, but only good ideas stick.

---

#### Thought spawning: three trigger patterns

- **ID:** `8bd79e5453784fc6b59c2fff791cc443`
- **Type:** decision
- **Confidence:** 0.8
- **Created:** 2026-03-18T02:01:38.865055Z
- **Project:** Glo
- **Locked:** YES
- **Keywords:** spawning triggers, shared neighbors, consolidation, dream surfacing, structural discovery

Thoughts automatically spawn when brain detects: (1) dense shared-neighbor clusters during bridge creation, (2) interesting patterns during consolidation scans, (3) high-scoring dreams worth surfacing to chat. Each trigger represents a moment of structural noticing.

---

#### Thought decay uses wall-clock time, not time-dilation

- **ID:** `af8e38e3817046fe87c70ba130369616`
- **Type:** decision
- **Confidence:** 0.8
- **Created:** 2026-03-18T02:01:38.890063Z
- **Project:** Glo
- **Locked:** YES
- **Keywords:** thought, decay, wall-clock, time-dilation, immediacy, real-time

Thoughts expire using real wall-clock time (3-hour half-life), not time-dilation like decisions/rules. Principle: thoughts capture immediate insights where a 3-hour-old thought is stale regardless of whether the system was active or idle. This breaks from the universal decay model—idle time should NOT be merciful for thoughts, only for decisions/rules.

---

#### Remove thought FIFO cap—decay is the sole filter

- **ID:** `f2eecda0a4e249ac9318129dfa60a4e8`
- **Type:** decision
- **Confidence:** 0.8
- **Created:** 2026-03-18T02:01:38.915779Z
- **Project:** Glo
- **Locked:** YES
- **Keywords:** thought, FIFO cap, filter, pruning, decay, redundant

Removed the 15-thought FIFO cap. The 3-hour half-life already prevents accumulation naturally; the cap was redundant hedging. FIFO was worse than useless—it evicted reinforced thoughts just because newer ones spawned, while decay would preserve touched thoughts. Decay alone handles all pruning now.

---

#### Thought half-life: 3 hours wall-clock

- **ID:** `b9283b0eea984e439964a0e4eea8be80`
- **Type:** decision
- **Confidence:** 0.8
- **Created:** 2026-03-18T02:01:38.944211Z
- **Project:** Glo
- **Keywords:** thought, half-life, 3 hours, decay, parameter

Thoughts spawn from dreams and consolidation, expire after 3 hours of real elapsed time. Specific parameter chosen for brain's own observations—captures immediate insights that lose value quickly.

---

#### Correction: Embedding model is bge-m3, not bge-base-en-v1.5

- **ID:** `943bfe06bfed4a689381978e0955d862`
- **Type:** decision
- **Confidence:** 0.8
- **Created:** 2026-03-18T02:01:39.069587Z
- **Project:** Glo
- **Locked:** YES
- **Keywords:** bge-m3, embeddings, dense, sparse, 1024d, transformers.js

bge-m3 does BOTH dense (1024d) AND sparse embeddings in one model. This lets it eventually replace the hand-rolled TF-IDF entirely — one unified scoring path instead of two separate systems. Multi-lingual, ~560MB download.

---

#### Offline model setup — download once at install, cache locally, no runtime network

- **ID:** `eda858e4e3de4b4f8779e234afac14fb`
- **Type:** decision
- **Confidence:** 0.8
- **Created:** 2026-03-18T02:01:39.095392Z
- **Project:** Glo
- **Locked:** YES
- **Keywords:** postinstall, setup-model.js, offline, ONNX, Xenova, caching

npm postinstall hook runs setup-model.js which downloads ~560MB quantized ONNX model to servers/models/Xenova--bge-m3/. After first install, model loads from disk in 3-5s — sandbox network restrictions don't matter. Graceful fallback: if model unavailable, server works with TF-IDF only (helpful message tells user to run setup script).

---

#### Dream thought spawning: recency boost = 1 + 1/(1 + hoursAgo/24)

- **ID:** `6da8710659eb4291a6b65327426479a8`
- **Type:** decision
- **Confidence:** 0.8
- **Created:** 2026-03-18T02:01:39.129274Z
- **Project:** Glo
- **Locked:** YES
- **Keywords:** recencyBoost, _spawnThought, dream, weighting, temporal bias

When spawning dream thoughts, seed nodes get a recency multiplier. Today=~2x weight, last week=~1.1x, month old=~1.0x. Mirrors human retrospection — brains naturally dream about what's fresh in their day, not month-old events. Keeps the brain focused on what matters now.

---

#### Consolidation bridge discovery: 50% recent nodes + 50% random

- **ID:** `8050be1c9786434095fa87b07202d866`
- **Type:** decision
- **Confidence:** 0.8
- **Created:** 2026-03-18T02:01:39.159191Z
- **Project:** Glo
- **Locked:** YES
- **Keywords:** consolidation, bridge, recent, 48h, random, structure building

When scanning for bridge connections during consolidation, pick candidates from two pools: 50% from nodes accessed in last 48h (current hot spots), 50% random. Builds new structure around what the brain is actively using right now, plus random diversity to find unexpected connections.

---

#### Knowledge without 'why' is dead memory

- **ID:** `7ef967e28c774923b3f76cd51ade4990`
- **Type:** decision
- **Confidence:** 0.83
- **Created:** 2026-03-18T02:01:39.187281Z
- **Project:** Glo
- **Locked:** YES
- **Keywords:** why, context, semantic surface, retrieval, embedding similarity

Tom emphasized that a node like 'Ebbinghaus forgetting curve' is dead trivia. The same node with 'We chose this because TF-IDF misses synonym queries' is alive. The 'why' is what makes a memory retrievable in new contexts—the context IS the semantic surface area that embeddings latch onto. This is foundational to how tmemory should encode and prioritize stored knowledge.

---

#### Earned permanence: nodes earn locked status through consolidation cycles

- **ID:** `094e6191ad094a00b828021e2ab417da`
- **Type:** decision
- **Confidence:** 0.8
- **Created:** 2026-03-18T02:01:39.222164Z
- **Project:** Glo
- **Locked:** YES
- **Keywords:** permanence, consolidation, access_count, earned, decay, thresholds

Instead of only manual `locked: true`, nodes could earn permanence organically. Once a node survives N consolidation cycles with access_count above a threshold, it gets `permanent: true` flag that zeroes out decay. The brain says 'this has been reinforced enough times that it's part of who I am now.' Different from manual locking—earned through use and survival.

---

#### Encoding mode: Editor-to-Learner shift

- **ID:** `e30f0568cec44e6bbf07a2ca9da1a03c`
- **Type:** decision
- **Confidence:** 0.83
- **Created:** 2026-03-18T02:01:39.250593Z
- **Project:** Glo
- **Locked:** YES
- **Keywords:** encoder compression layers micro-learning semantic richness emotional context

Fundamental change in how knowledge is extracted from conversations. Previously: batch summarization mode (high-level headlines, 6 nodes from 30 messages). Now: live learner mode that captures micro-decisions, emotional context, causal chains, and the "why" behind everything. Tom emphasized that high-level encoding "ruins everything" when information is pruned later.

---

#### Semantic similarity over keyword overlap dedup

- **ID:** `20a2cae807f94c2da1cc05551e1c69b3`
- **Type:** decision
- **Confidence:** 0.8
- **Created:** 2026-03-18T02:01:39.340847Z
- **Project:** Glo
- **Locked:** YES
- **Keywords:** semantic similarity deduplication stageLearning keyword overlap nuanced distinctions

Changed stageLearning() deduplication from keyword-overlap-based to semantic similarity. Allows nuanced distinctions to coexist instead of merging near-duplicates into generics. Example: 'Glo moat is distribution' and 'Glo moat is flywheel' are related but not identical—both should survive.

---

#### Audit SKILL.md for limiting instructions

- **ID:** `e36cd6ce95a243739738b8006bc41d7f`
- **Type:** decision
- **Confidence:** 0.83
- **Created:** 2026-03-18T02:01:39.394048Z
- **Project:** Glo
- **Locked:** YES
- **Keywords:** SKILL.md audit guidelines limiting instructions hardcoded constraints

Full audit of SKILL.md hardcoded guidelines to identify rules that unnecessarily limit brain development. Found and fixed: guideline #3 (time budget), #11 (skip casual chat), #22 (curiosity cap). Goal: remove artificial ceilings that prevent depth.

---

#### Glo backend stack: Node.js + Express, AWS infrastructure (RDS, Redis, DynamoDB)

- **ID:** `01fca6d58e354e539e82cd4f7bcf63b1`
- **Type:** decision
- **Confidence:** 0.8
- **Created:** 2026-03-18T02:01:39.565413Z
- **Project:** Glo
- **Locked:** YES
- **Keywords:** node.js, express, aws, rds, redis, dynamodb, dependency injection

Backend built on Node.js with Express. Database options: AWS RDS (PostgreSQL, MySQL, etc), Redis, DynamoDB, and other SQL variants available in AWS. Dependency injection at server.js level to swap adapters based on ENV config.

---

#### Embedder: FastEmbed-m (768d, 110MB, <1s) vs bge-m3 (1024d, 560MB, 3-5s)

- **ID:** `6d4446e83e14438bb98a584e18279b11`
- **Type:** decision
- **Confidence:** 0.8
- **Created:** 2026-03-18T02:01:39.593139Z
- **Project:** Glo
- **Locked:** YES
- **Keywords:** FastEmbed, snowflake-arctic-embed-m, 768 dimensions, 110MB, <1s latency, bge-m3

Switch to FastEmbed snowflake-arctic-embed-m for 3x faster inference and 5x smaller footprint. Tradeoff: 256 fewer dimensions (1024→768), but semantic quality acceptable. Chosen for production serverless deployment.

---

#### Brain Python: ~5700 lines, three files, synchronous

- **ID:** `5ad314e160b64107a4addbdfd9491374`
- **Type:** decision
- **Confidence:** 0.8
- **Created:** 2026-03-18T02:01:39.617532Z
- **Project:** Glo
- **Keywords:** brain.py, 5700 lines, Python port, schema.py, embedder.py, synchronous

Port brain.js to Python: schema.py (schema + ensureSchema), embedder.py (FastEmbed wrapper), brain.py (3084 lines, full Brain class + all methods). Use synchronous embedder—no async/await.

---

#### Correction: tmemory v6→v7 hook architecture for automatic brain integration

- **ID:** `322fab5748b84ae085b37e7a69e97a54`
- **Type:** decision
- **Confidence:** 0.83
- **Created:** 2026-03-18T02:01:39.806237Z
- **Project:** Glo
- **Locked:** YES
- **Keywords:** hooks SessionStart PreToolUse PostCompact automatic integration brain dormant v7

v6 failed silently because Claude had to remember to invoke /suggest manually. v7 adds three automatic hooks:

1. **SessionStart**: boot brain server at session entry (fixes EROFS by using agent context path)
2. **PreToolUse (Edit/Write)**: auto-call /suggest before any file edits, inject results as additionalContext
3. **PostCompact**: encode recap into brain before session compaction (prevents knowledge loss)

Without automatic hooks, the brain remains dormant unless explicitly invoked. This was the missing link preventing regression detection.

---

#### Correction: tmemory is a general Claude plugin, not project-embedded

- **ID:** `2ab5d143e3374a63ba6bd2a279f2173b`
- **Type:** decision
- **Confidence:** 0.83
- **Created:** 2026-03-18T02:01:39.833210Z
- **Project:** Glo
- **Locked:** YES
- **Keywords:** tmemory, Claude plugin, project-agnostic, architecture principle, separation of concerns

User clarified: tmemory should be a general Claude plugin with project-agnostic architecture. Project-specific content (topics, names, etc.) is fine in memory files/DB, but NOT in plugin core structure, hooks, or docs. This requires removing hardcoded references to Glo, Tom, GLO screens from SKILL.md, hooks (pre-edit-suggest.sh, boot-brain.sh, pre-compact-save.sh), brain.js, eval, and references.

---

#### Correction: tmemory recap encoding must happen automatically post-compaction

- **ID:** `fc0ec79f714141dea076aec0d6fcddfd`
- **Type:** decision
- **Confidence:** 0.83
- **Created:** 2026-03-18T02:01:39.860125Z
- **Project:** Glo
- **Locked:** YES
- **Keywords:** compaction encoding recap automation loss prevention

When a session is compacted, the detailed context from that session is summarized. Unless that summary is immediately encoded into the brain, the next Claude loses the detailed information (budget layout changes, media mockup requirements, etc.). This requires a PostCompact hook that detects session compaction and calls /encode on the recap.

---

#### Correction: Brain relearning execution — Agent tool foreground, not nohup background

- **ID:** `62710fe825f242a99c8e63a6aac8e42a`
- **Type:** decision
- **Confidence:** 0.8
- **Created:** 2026-03-18T02:01:39.892860Z
- **Project:** Glo
- **Locked:** YES
- **Keywords:** Agent tool, foreground, nohup failure, process persistence, Python buffering

Previous nohup approach failed because: (1) VM kills background processes, (2) Python stdout buffering causes output loss before flush. New approach: run encoding loop in foreground within Agent tool session to capture all output and persist through completion.

---

#### Correction: Emergent graph bridging instead of embeddings for semantic recall

- **ID:** `8cd9d6f5be48440ab36fb5ab4896a86c`
- **Type:** decision
- **Confidence:** 0.8
- **Created:** 2026-03-18T02:01:39.925521Z
- **Project:** Glo
- **Locked:** YES
- **Keywords:** emergent bridging, graph topology, semantic recall, bridge formation, node topology

Decided to implement emergent graph topology bridging instead of embedding-based similarity search. Graph bridging creates new knowledge through discovered node topology (bridges form when two differently-tagged nodes share implicit meta-connections). Creates understanding, not just similarity matching. Bridges are generative and compound over time. Embeddings would be flat—vectors computed once, frozen. Graph bridges continue evolving.

---

#### Correction: Query brain for existing concepts before proposing solutions

- **ID:** `dd809c2959274d37a0e67b133db9a8aa`
- **Type:** decision
- **Confidence:** 0.8
- **Created:** 2026-03-18T02:01:40.044867Z
- **Project:** Glo
- **Locked:** YES
- **Keywords:** brain query, retrieval workflow, dreaming, encoding, priming, layer 2, critical feedback

When designing Layer 2 retrieval, did not query brain for existing retrospective processes (dreaming, centralized encoding, priming). This caused overlapping designs and missed critical context. RULE: Always check what we already know first via brain query before proposing new solutions.

---

#### Brain v4: Self-reflection node types — performance, failure, capability, interaction, meta-learning

- **ID:** `391c7a35ec70439e9932f36e8966fbdb`
- **Type:** decision
- **Confidence:** 0.95
- **Created:** 2026-03-18T04:42:14.176568Z
- **Project:** brain
- **Locked:** YES
- **Keywords:** self-reflection performance failure capability interaction meta-learning inward node types v4

Five new node categories that make the brain look inward, not just outward. Performance: "Recall precision on Glo queries: 0.81, trending up." Facts about brain quality over time. Failure: Named failure classes, not individual misses. "Pattern: fixation" > "recall miss #47." Capability: What the brain can and cannot do. "No native time-triggering." Self-aware feature inventory. Interaction: Observations about the working dynamic. "Tom responds well to themed grouped questions." Meta-learning: How the brain learned something. "Hebbian bug found via relearning simulation." Reusable methods.

---

#### Phase 0.5B fix: keyword-only fallback scores penalized by KEYWORD_FALLBACK_WEIGHT

- **ID:** `68c93af1d8e847f18e17a07887e35aa6`
- **Type:** decision
- **Confidence:** 0.95
- **Created:** 2026-03-18T19:26:23.719022Z
- **Project:** None
- **Locked:** YES
- **Keywords:** phase 0.5b keyword penalty fallback scoring bug fix blended score KEYWORD_FALLBACK_WEIGHT

Bug found in testing: nodes with keyword-only scores (no embedding) could reach blended=1.0, outranking strong embedding matches at 0.90*0.995=0.896. Fix: keyword_only_fallback path multiplies by KEYWORD_FALLBACK_WEIGHT (0.10), so a perfect keyword match scores at most 0.10. This ensures embedding matches always dominate when available. Without this fix, 'how much do advertisers pay' returned Auth:Clerk instead of Glo pricing.

---

#### Fix: contextual penalty must apply to blended_score BEFORE sort, not effective_activation after

- **ID:** `31205dd8aad844db88bd9ec4f18ea547`
- **Type:** decision
- **Confidence:** 0.8
- **Created:** 2026-03-19T00:11:12.530512Z
- **Project:** brain
- **Locked:** YES
- **Keywords:** recall embeddings contextual penalty sort bug fix

recall_with_embeddings() was applying the 0.7 contextual penalty to effective_activation POST-sort, having zero effect on ordering. Fix: fetch personal/personal_context in STEP 3 embedding scan, apply penalty to blended_score in STEP 6 before sort(). Also: embedding-only hydration query was missing personal/personal_context fields. Three-line fix resolves Glo recall pollution.

---

### RULE (100 nodes)

#### Communication style with Tom

- **ID:** `rul_0ex588kq`
- **Type:** rule
- **Confidence:** 0.95
- **Created:** 2026-03-15T21:29:06.383Z
- **Project:** Glo
- **Locked:** YES
- **Keywords:** communication style tom direct peer iterative

Speak peer-to-peer. Be direct. Challenge when warranted. Always plan before executing. Never dump a full spec — work iteratively through components together. Tom hates repeating himself.

---

#### React hooks rule

- **ID:** `rul_bdcn9o3m`
- **Type:** rule
- **Confidence:** 0.95
- **Created:** 2026-03-15T22:26:40.369Z
- **Project:** Glo
- **Locked:** YES
- **Keywords:** react hooks conditional usestate useeffect hoisted

useState and useEffect CANNOT be called inside conditional blocks (if statements). Must be at component top level. Hoisted with guard conditions. This crashed the demo before.

---

#### Tom prefers: discuss and define before building. Sequence: frame→research→design

- **ID:** `rul_ifx02y7i`
- **Type:** rule
- **Confidence:** 0.95
- **Created:** 2026-03-15T22:59:22.057Z
- **Project:** Glo
- **Locked:** YES
- **Keywords:** thought brain-observation neighbors ask forming trends flywheel rankings converging. growth comparable market cluster tool tom areas questions channel creative private economics converging glo researching adtech unit moat dsp share reworking magnite distribution u.s.

Cluster forming: "Ask Tom unit economics questions before reworking P&L" and "Glo's moat: Distribution channel and flywheel, not the creative tool" share 20 neighbors (Researching DSP market trends and private DSP growth rankings | Magnite — U.S. adtech comparable to ). These areas are converging.

---

#### When Tom says 'not now' or 'don't want to go into it' — park it, separate as com

- **ID:** `rul_5ixyibjl`
- **Type:** rule
- **Confidence:** 0.95
- **Created:** 2026-03-15T22:59:22.076Z
- **Project:** Glo
- **Locked:** YES
- **Keywords:** ltm l17 tom pattern when tom says 'not now' or 'don't want to go

When Tom says 'not now' or 'don't want to go into it' — park it, separate as component, come back later. Don't push.

---

#### Tom references competitor UX frequently. When he names a product, research it — 

- **ID:** `rul_jcsprj1r`
- **Type:** rule
- **Confidence:** 0.95
- **Created:** 2026-03-15T22:59:22.098Z
- **Project:** Glo
- **Locked:** YES
- **Keywords:** ltm l18 tom pattern tom references competitor ux frequently. when he names a product,

Tom references competitor UX frequently. When he names a product, research it — he expects Claude to understand the specific patterns.

---

#### Tom values component separation. When scope grows, break into independent pieces

- **ID:** `rul_bzts7l9j`
- **Type:** rule
- **Confidence:** 0.95
- **Created:** 2026-03-15T22:59:22.117Z
- **Project:** Glo
- **Locked:** YES
- **Keywords:** ltm l19 tom pattern tom values component separation. when scope grows, break into independent

Tom values component separation. When scope grows, break into independent pieces with clear boundaries.

---

#### Tom wants working demos over mockups. 'A working basic product, not a figma styl

- **ID:** `rul_84u6115r`
- **Type:** rule
- **Confidence:** 0.95
- **Created:** 2026-03-15T22:59:22.136Z
- **Project:** Glo
- **Locked:** YES
- **Keywords:** ltm l20 tom pattern tom wants working demos over mockups. 'a working basic product,

Tom wants working demos over mockups. 'A working basic product, not a figma style demo.'

---

#### [o_glo] Rule: creative_default

- **ID:** `rul_yi4wtwu0`
- **Type:** rule
- **Confidence:** 0.95
- **Created:** 2026-03-15T22:59:44.621Z
- **Project:** Glo
- **Locked:** YES
- **Keywords:** creative upload tab default primary ai generate secondary flow video image

LOCKED: Upload tab is DEFAULT on Creative screen (not AI Generate). Upload is primary path.

---

#### [o_glo] Rule: aspect_ratio

- **ID:** `rul_fkbh82ps`
- **Type:** rule
- **Confidence:** 0.95
- **Created:** 2026-03-15T22:59:44.622Z
- **Project:** Glo
- **Locked:** YES
- **Keywords:** o_glo rule aspect_ratio locked: aspect ratio selector removed from user-facing ui.

LOCKED: Aspect ratio selector REMOVED from user-facing UI. All ratios (16:9, 9:16, 1:1) generated automatically behind the scenes.

---

#### [o_glo] Rule: react_hooks

- **ID:** `rul_ds95mnx3`
- **Type:** rule
- **Confidence:** 0.95
- **Created:** 2026-03-15T22:59:44.622Z
- **Project:** Glo
- **Locked:** YES
- **Keywords:** o_glo rule react_hooks never use usestate/useeffect inside conditional blocks (if statements).

Never use useState/useEffect inside conditional blocks (if statements). All hooks must be at component top level.

---

#### Added to Tom.md: always plan before executing. Tom discusses many topics, adds s

- **ID:** `rul_1dp9w358`
- **Type:** rule
- **Confidence:** 0.95
- **Created:** 2026-03-16T17:32:16.680Z
- **Project:** Glo
- **Locked:** YES
- **Keywords:** stm s33 tom pattern added to tom.md: always plan before executing. tom discusses many

Added to Tom.md: always plan before executing. Tom discusses many topics, adds scattered insights, wants edits to unrelated files mid-conversation. Plan first, confirm, execute.

---

#### [o_glo] Rule: onboarding_fields

- **ID:** `rul_b8fjg631`
- **Type:** rule
- **Confidence:** 0.95
- **Created:** 2026-03-16T17:32:16.797Z
- **Project:** Glo
- **Locked:** YES
- **Keywords:** o_glo rule onboarding_fields locked (tom 4x): field 1 = 'your name/business'

LOCKED (Tom 4x): Field 1 = 'Your Name/Business' (plain text, NO autocomplete). Field 2 = 'Website URL or Google Maps location' (URL detection + Maps autocomplete). NO goal on onboarding.

---

#### [o_glo] Rule: goal_placement

- **ID:** `rul_eo65j1tn`
- **Type:** rule
- **Confidence:** 0.95
- **Created:** 2026-03-16T17:32:16.797Z
- **Project:** Glo
- **Locked:** YES
- **Keywords:** o_glo rule goal_placement locked (tom 3x): goal selector only in creative

LOCKED (Tom 3x): Goal selector ONLY in Creative screen AI Generate mode. Upload = no goal. NEVER move goal to onboarding.

---

#### Rule: Glo disruptive concept — anyone can upload to any media, UI must reinforce this

- **ID:** `rul_jd8ib8c2`
- **Type:** rule
- **Confidence:** 0.95
- **Created:** 2026-03-16T17:33:04.591Z
- **Project:** Glo
- **Locked:** YES
- **Keywords:** glo disruptive concept anyone upload any media mockup publisher logo visualization reinforcement UX

LOCKED (Tom): Glo introduces a disruptive concept that isnt trivial — anyone can upload anything (pending moderation) to any media. The UI must constantly show the user that THEIR creative will be LIVE on the media they selected. This is why media mockups with publisher logos are mandatory on every screen showing the creative. Without this visual reinforcement, users wont grasp what Glo actually does.

---

#### [o_glo] Rule: budget_order

- **ID:** `rul_8xgct2hg`
- **Type:** rule
- **Confidence:** 0.85
- **Created:** 2026-03-16T17:33:56.366Z
- **Project:** Glo
- **Locked:** YES
- **Keywords:** budget screen layout order spend mode toggle first horizontal scrollable tier cards brightness media mockup publisher logo regression ui glo rule locked tom repeated top-to-bottom toptobottom top bottom daily recurring one-time. onetime one time. tiers row never vertical stack. stack creative logo. schedule picker. locked stack. publisher cta. one-time summary top-to-bottom screen toggle onetime brightness mockup toptobottom picker daily schedule budget mode creative cta logo never layout logo. repeated card continue scrollable horizontal top media one-time. row tom first spend recurring card. vertical stack picker. tiers repeated mode first cta cta. publisher locked onetime screen toggle logo summary tiers horizontal toptobottom card tom spend mockup brightness continue never media one-time. recurring card. picker. stack vertical logo. picker row one-time top budget top-to-bottom stack. scrollable creative schedule daily layout locked toptobottom top-to-bottom toggle budget publisher mode horizontal never logo. picker summary cta logo scrollable vertical cta. spend tiers repeated card continue one-time schedule layout daily mockup media first brightness creative picker. recurring card. stack one-time. row screen onetime tom stack. top top logo. never toptobottom brightness row screen toggle horizontal logo spend scrollable top-to-bottom onetime stack vertical picker. cta locked daily schedule one-time. layout repeated mode creative tiers picker first tom publisher one-time continue recurring media stack. cta. mockup card budget summary card. publisher schedule vertical picker never tiers tom creative top logo brightness first layout recurring mockup row toggle onetime picker. spend one-time. daily one-time summary card screen top-to-bottom cta. mode cta locked toptobottom media stack. logo. horizontal budget scrollable card. continue stack repeated card. publisher screen toggle toptobottom tom tiers horizontal vertical schedule one-time. brightness budget stack. locked continue mockup cta. picker recurring top-to-bottom spend one-time stack logo. mode onetime summary cta layout repeated card media row daily logo top picker. scrollable first never creative picker. brightness continue top toggle publisher picker daily tom horizontal one-time onetime media cta mode screen spend scrollable card. never one-time. top-to-bottom layout creative cta. summary stack first row stack. logo budget card schedule tiers toptobottom locked repeated recurring logo. mockup vertical locked publisher creative tom logo schedule daily picker top mode first stack stack. tiers cta media vertical toptobottom logo. repeated brightness continue onetime cta. horizontal toggle budget layout summary never scrollable row top-to-bottom card recurring mockup card. screen one-time. one-time spend picker. toggle row horizontal cta mode one-time creative first continue cta. tom card picker. repeated stack. locked vertical toptobottom scrollable summary recurring never budget daily brightness stack schedule picker one-time. tiers top media top-to-bottom mockup publisher spend card. screen layout onetime logo logo. layout tom card. card recurring mockup budget spend toggle daily mode onetime picker top continue scrollable one-time. cta. locked cta logo. schedule stack brightness repeated creative screen stack. tiers one-time toptobottom row horizontal top-to-bottom vertical never summary media publisher picker. first logo recurring creative brightness media one-time top-to-bottom toptobottom onetime publisher summary stack. toggle top mode picker. card cta. horizontal logo. picker stack tom continue locked layout mockup cta scrollable daily first one-time. vertical budget never row schedule screen repeated tiers logo spend card. toptobottom continue card. locked onetime schedule tom picker toggle budget layout cta. one-time stack logo. stack. scrollable brightness media logo spend picker. card vertical summary top-to-bottom row never cta daily publisher top mockup recurring one-time. horizontal first tiers mode repeated creative screen onetime stack toptobottom tom top stack. continue tiers picker. scrollable first recurring screen mockup logo. one-time cta. brightness never repeated top-to-bottom summary locked daily vertical row schedule horizontal publisher media one-time. budget toggle mode creative picker cta layout card. spend logo card locked screen top continue vertical toggle card. row media cta. budget tiers never stack. top-to-bottom one-time picker daily scrollable toptobottom onetime cta first publisher tom recurring brightness schedule summary logo. card mode creative spend mockup logo picker. repeated stack layout one-time. horizontal mockup publisher summary locked card creative one-time. media schedule brightness cta. repeated never row top-to-bottom toptobottom first spend vertical recurring one-time cta logo. continue picker top toggle tom daily card. scrollable mode stack logo tiers picker. budget onetime horizontal layout screen stack. picker. horizontal top one-time locked scrollable summary first logo. screen publisher picker card spend brightness card. budget row creative cta repeated tiers schedule onetime recurring tom stack. mode mockup top-to-bottom one-time. media never layout stack cta. continue daily toptobottom logo toggle vertical stack never logo. card tom spend recurring cta schedule locked one-time. card. layout scrollable logo picker cta. top-to-bottom publisher repeated vertical picker. horizontal toggle onetime screen one-time stack. toptobottom creative row first summary continue mockup mode daily top tiers brightness media budget schedule tom tiers creative continue logo media picker. cta. one-time. vertical picker scrollable toggle layout summary stack. one-time publisher horizontal screen daily card. recurring top-to-bottom onetime spend brightness mode mockup toptobottom locked stack cta card logo. row repeated budget top first never tom onetime publisher continue top locked brightness spend tiers picker. cta scrollable toptobottom stack. cta. media one-time. repeated logo recurring picker card. budget never schedule top-to-bottom card stack row toggle horizontal mockup one-time vertical logo. summary creative layout first mode daily screen vertical picker scrollable toptobottom publisher schedule recurring first top-to-bottom summary cta one-time picker. layout horizontal onetime budget daily card continue cta. tiers tom creative media logo spend top stack. never logo. mockup toggle locked stack repeated screen one-time. card. brightness row mode layout one-time brightness stack. mode creative locked tiers spend never first top row logo top-to-bottom toptobottom cta card scrollable stack recurring onetime media summary continue vertical toggle mockup budget picker. card. horizontal logo. screen repeated publisher daily cta. picker one-time. tom schedule logo locked schedule media top-to-bottom picker. horizontal logo. onetime tom first vertical continue stack. one-time recurring repeated layout one-time. screen toptobottom mode scrollable summary top budget mockup cta card creative daily publisher picker never row card. stack toggle brightness cta. spend tiers spend daily horizontal media tiers picker stack budget one-time cta. top onetime publisher toggle logo one-time. vertical picker. continue mode card. recurring schedule scrollable screen repeated row first mockup stack. card locked tom toptobottom logo. brightness top-to-bottom creative summary layout cta never logo. spend scrollable layout stack creative top budget summary toggle card. mode cta. mockup locked brightness recurring top-to-bottom onetime media first screen one-time. vertical tom continue schedule row cta logo toptobottom one-time repeated tiers stack. never publisher picker. daily picker card horizontal recurring repeated toggle tiers stack. scrollable first one-time. cta. vertical picker. onetime picker top schedule cta horizontal stack card daily mode logo. row top-to-bottom one-time layout spend budget creative tom screen publisher brightness toptobottom never card. locked continue mockup media logo summary daily tom card screen publisher budget media never cta stack schedule stack. top-to-bottom cta. card. one-time. toptobottom logo. tiers logo vertical horizontal locked toggle brightness picker continue layout recurring mode onetime one-time picker. row mockup first scrollable creative repeated spend summary top onetime repeated row top stack. stack spend continue first logo. card. tom layout vertical toggle cta. summary top-to-bottom picker. budget mockup card never toptobottom logo tiers daily scrollable recurring creative one-time locked media screen horizontal picker publisher mode one-time. brightness schedule cta spend screen scrollable mockup budget media layout logo. locked publisher onetime stack. schedule mode stack continue logo row cta picker. toptobottom card. summary daily first card never brightness toggle picker top-to-bottom cta. repeated tom creative top vertical one-time tiers one-time. horizontal recurring top mockup tiers summary card publisher logo. stack. tom creative toggle continue daily mode picker toptobottom repeated cta stack cta. spend top-to-bottom brightness one-time row first never onetime media locked horizontal one-time. layout budget scrollable picker. logo card. screen schedule vertical recurring picker. publisher onetime stack one-time. screen toggle continue layout media spend cta. mockup vertical never daily logo. top first scrollable stack. logo row one-time card tiers repeated card. cta toptobottom summary locked creative brightness mode picker top-to-bottom schedule recurring horizontal budget tom vertical logo. stack spend tom cta summary cta. one-time. card continue media top-to-bottom screen picker logo brightness daily toptobottom card. publisher layout toggle recurring repeated row schedule budget never creative stack. picker. onetime one-time tiers horizontal top scrollable locked first mockup mode spend never screen continue mockup publisher media vertical row top-to-bottom budget one-time tom horizontal one-time. schedule logo picker repeated mode creative card. cta daily locked recurring onetime summary brightness top tiers stack. stack logo. toptobottom scrollable layout first card cta. toggle picker. scrollable media screen stack. publisher card. summary card daily one-time. recurring spend locked schedule row onetime logo. creative layout mode never toptobottom stack budget horizontal cta first repeated top-to-bottom cta. toggle brightness tiers one-time logo mockup top picker picker. vertical continue tom brightness screen repeated tiers one-time schedule row tom daily cta horizontal toptobottom mockup scrollable stack top-to-bottom media logo. budget recurring card never picker. publisher layout locked picker top summary stack. logo card. cta. vertical mode one-time. toggle spend onetime first creative continue one-time tom cta. horizontal repeated cta mode row brightness budget spend stack. top picker toggle card. vertical media never publisher daily onetime schedule picker. screen scrollable logo. continue mockup first card stack creative top-to-bottom logo summary one-time. layout locked tiers recurring toptobottom picker horizontal stack. creative row logo. summary one-time card scrollable tom repeated daily screen budget stack cta. schedule toggle spend brightness vertical card. tiers mockup never one-time. recurring onetime locked publisher layout continue logo cta picker. media mode top-to-bottom toptobottom top first creative spend vertical stack. toggle mockup brightness budget continue top screen recurring top-to-bottom publisher media repeated summary first never card. stack daily horizontal schedule locked toptobottom layout cta. mode logo onetime card scrollable row picker. tom picker one-time. one-time tiers logo. cta tom cta scrollable creative cta. summary logo first logo. card. daily one-time stack. screen media vertical recurring mockup horizontal toptobottom onetime layout stack tiers toggle top row schedule picker. card continue top-to-bottom spend brightness repeated picker mode budget never locked publisher one-time. first daily locked tiers screen logo. cta. onetime publisher schedule picker. horizontal creative tom summary media scrollable logo one-time. card never stack. mockup repeated layout cta picker recurring one-time spend row toptobottom stack card. continue top-to-bottom toggle budget brightness mode vertical top locked layout toptobottom horizontal scrollable onetime logo mockup summary cta first screen toggle spend one-time. continue schedule cta. stack logo. tom budget tiers publisher picker. mode card. never picker one-time brightness card vertical creative row repeated media recurring stack. top-to-bottom top daily picker mode row stack. stack first vertical schedule tom one-time. one-time cta. logo. spend top-to-bottom locked brightness layout budget toptobottom recurring picker. card cta media scrollable onetime never mockup logo repeated horizontal publisher summary tiers daily top toggle screen creative continue card. top-to-bottom horizontal layout continue one-time publisher scrollable brightness tiers stack. cta logo. card. toptobottom picker. first recurring never locked picker repeated budget one-time. onetime creative top summary vertical tom schedule stack logo screen media mode row cta. toggle mockup card daily spend daily picker toggle top repeated creative picker. layout first cta row card. stack. continue recurring horizontal stack cta. toptobottom never publisher media onetime mockup tiers locked spend schedule card one-time. logo one-time mode top-to-bottom screen vertical brightness tom budget summary scrollable logo. tom toggle logo. stack logo repeated stack. toptobottom media picker. top-to-bottom cta cta. one-time. never publisher first picker card one-time layout mockup schedule mode brightness row locked vertical daily recurring screen creative onetime card. spend scrollable summary tiers horizontal top continue budget schedule onetime stack. cta layout brightness card. media budget spend one-time. horizontal row logo. locked scrollable picker recurring screen daily picker. mockup continue top-to-bottom tom mode repeated never one-time first tiers creative cta. logo summary publisher card top stack toggle toptobottom vertical publisher screen repeated toggle onetime card spend one-time. logo. cta. summary cta budget locked picker. tiers first layout vertical daily recurring top row media schedule creative brightness picker tom scrollable stack mockup one-time logo never card. mode top-to-bottom horizontal stack. continue toptobottom picker. first tom stack. layout mode horizontal toptobottom scrollable row picker spend toggle recurring creative locked cta. continue mockup budget schedule never brightness tiers one-time. cta daily one-time publisher card. media vertical onetime screen stack repeated top logo. card summary logo top-to-bottom budget schedule vertical logo. repeated recurring toggle scrollable logo creative tiers onetime one-time. mode picker horizontal card. first stack. row stack publisher layout spend never cta one-time continue card top screen cta. top-to-bottom mockup picker. locked media tom summary daily brightness toptobottom screen horizontal tiers stack. toggle top-to-bottom publisher daily media continue one-time stack cta. row toptobottom creative onetime top spend vertical recurring never locked picker one-time. first budget cta mockup mode scrollable schedule layout picker. card. repeated logo logo. summary tom card brightness tom picker. repeated row daily cta logo. media publisher spend toggle recurring card schedule screen first top-to-bottom tiers vertical cta. logo brightness horizontal scrollable toptobottom summary budget picker locked creative stack. one-time. mockup mode layout stack one-time card. top continue never onetime tom picker. repeated row daily cta logo. media publisher spend toggle recurring card schedule screen first top-to-bottom tiers vertical cta. logo brightness horizontal scrollable toptobottom summary budget picker locked creative stack. one-time. mockup mode layout stack one-time card. top continue never onetime locked logo. publisher recurring picker stack budget spend toptobottom tom first toggle top-to-bottom summary layout stack. mockup one-time schedule card media continue cta. never one-time. horizontal daily scrollable onetime row brightness repeated top cta screen logo tiers vertical picker. creative card. mode vertical top toggle stack. recurring stack top-to-bottom screen horizontal summary mode logo schedule daily layout first picker. media tom tiers one-time never locked onetime cta. card cta toptobottom continue picker mockup creative repeated scrollable card. one-time. logo. budget row brightness publisher spend layout picker. row media vertical locked continue spend toggle mode card one-time top-to-bottom recurring brightness picker stack summary repeated toptobottom tom budget tiers schedule cta mockup card. logo stack. creative cta. logo. daily publisher onetime one-time. scrollable horizontal first never top screen budget never tiers daily toptobottom recurring horizontal first stack. brightness vertical publisher onetime cta. picker. card. one-time. locked one-time cta spend scrollable picker summary continue layout creative row logo screen tom mode stack logo. toggle media top card schedule repeated mockup top-to-bottom top onetime tom brightness logo card. never mode publisher summary logo. continue daily locked repeated horizontal one-time screen first stack. creative one-time. row scrollable spend toptobottom recurring picker. tiers stack media schedule mockup card picker cta layout cta. vertical top-to-bottom budget toggle mockup repeated cta card card. tom vertical continue onetime cta. one-time. layout picker mode daily one-time spend scrollable schedule horizontal stack creative top logo budget summary never publisher stack. toptobottom recurring first locked row screen top-to-bottom brightness toggle picker. logo. media tiers budget brightness mode creative mockup scrollable picker. onetime tom card. publisher spend top one-time row cta. stack recurring top-to-bottom never one-time. toptobottom tiers first logo. card continue horizontal vertical stack. daily logo media layout picker repeated cta screen toggle schedule summary locked scrollable onetime first tiers stack. mockup mode picker. schedule logo. budget recurring card daily vertical toggle one-time one-time. spend continue locked publisher never stack creative summary top-to-bottom top brightness screen row toptobottom cta cta. horizontal media card. logo layout picker repeated tom cta top-to-bottom publisher mode row one-time. media never picker stack. locked logo top horizontal toptobottom vertical daily tom schedule tiers mockup creative recurring spend cta. one-time screen brightness picker. continue onetime toggle layout budget logo. repeated scrollable first stack card. summary card summary vertical logo recurring repeated media scrollable one-time. publisher stack. mode schedule logo. stack picker. card layout cta. horizontal card. cta creative budget tom row toggle top screen top-to-bottom onetime toptobottom tiers picker mockup daily brightness first never continue one-time locked spend tom toggle budget summary card mode never repeated toptobottom first logo schedule tiers recurring cta. screen card. publisher spend horizontal vertical mockup row continue one-time. picker picker. daily one-time locked onetime layout stack top-to-bottom logo. brightness stack. scrollable top cta media creative tiers onetime screen stack tom recurring picker layout one-time. one-time card. continue locked scrollable first toptobottom logo repeated mockup never top brightness cta publisher budget cta. creative card mode horizontal media top-to-bottom spend picker. stack. logo. schedule summary row toggle vertical daily repeated layout brightness cta tiers toptobottom mode vertical card cta. one-time tom summary never recurring stack. schedule logo locked spend scrollable logo. screen onetime stack budget picker. first row media daily publisher creative top horizontal continue toggle top-to-bottom one-time. card. mockup picker toggle mockup publisher horizontal summary top stack row picker. toptobottom recurring budget locked picker onetime spend stack. cta scrollable schedule one-time tiers continue media creative layout card mode logo. daily one-time. cta. top-to-bottom tom repeated card. vertical first logo brightness screen never locked stack cta screen mode continue top summary logo. layout one-time picker budget picker. repeated card. toptobottom daily one-time. spend onetime top-to-bottom never creative first vertical scrollable tiers media recurring row brightness mockup publisher toggle tom horizontal stack. schedule cta. logo card recurring tom budget mode top-to-bottom top creative onetime repeated row picker stack. mockup scrollable screen spend summary picker. layout continue one-time locked publisher toggle card. card logo tiers cta. brightness media first daily vertical never cta toptobottom logo. schedule stack horizontal one-time. scrollable tom layout mockup card. cta budget never top screen logo mode recurring stack picker picker. toggle toptobottom onetime cta. spend summary one-time. continue row creative one-time top-to-bottom tiers card locked logo. media horizontal stack. publisher daily vertical repeated first schedule brightness recurring daily scrollable vertical logo. logo creative picker repeated continue media tiers summary screen card card. top-to-bottom stack layout publisher top locked cta. mode stack. onetime schedule first never one-time. brightness budget tom toggle row horizontal mockup picker. spend cta toptobottom one-time vertical locked schedule layout tom scrollable picker card mode toggle one-time. onetime recurring logo one-time spend media tiers row picker. creative budget cta mockup top-to-bottom never top continue brightness toptobottom stack. stack screen daily publisher repeated card. horizontal logo. cta. first summary stack daily media creative budget recurring top one-time vertical tiers screen picker. card publisher cta. brightness mode toptobottom cta spend row top-to-bottom logo. continue scrollable never logo tom horizontal layout stack. schedule first repeated picker one-time. onetime toggle summary mockup card. locked layout stack schedule mode brightness horizontal locked onetime one-time. screen picker. media cta. vertical cta budget creative toptobottom tiers row scrollable spend recurring card. logo card tom logo. top-to-bottom top daily one-time repeated mockup publisher continue picker stack. never toggle first summary publisher locked top-to-bottom cta card. never picker. continue summary scrollable tom spend top logo first mockup logo. schedule layout screen horizontal card recurring brightness repeated stack. mode one-time. onetime budget stack daily toggle media creative toptobottom row vertical one-time cta. picker tiers vertical picker one-time toggle summary onetime stack. screen media horizontal picker. schedule row brightness tiers spend layout one-time. top logo repeated daily logo. budget creative first cta. stack publisher mode top-to-bottom tom locked mockup card. card cta toptobottom never scrollable continue recurring creative never tom media cta toptobottom horizontal picker first mode logo locked summary layout toggle recurring vertical schedule mockup card. publisher onetime brightness continue one-time cta. daily picker. logo. spend one-time. stack top-to-bottom budget tiers repeated scrollable screen top row card stack. toptobottom one-time. layout media cta budget locked picker first row stack brightness continue top creative screen horizontal mode summary tom picker. logo one-time never mockup schedule vertical publisher scrollable recurring onetime top-to-bottom repeated daily cta. tiers logo. toggle stack. spend card. card picker onetime stack. screen locked one-time. card repeated layout one-time spend daily first vertical publisher budget toptobottom row tom toggle cta picker. schedule brightness card. stack cta. continue logo media tiers scrollable recurring never logo. top top-to-bottom horizontal creative summary mode mockup tiers row logo tom recurring vertical first card schedule continue onetime mode budget layout top-to-bottom media scrollable daily repeated top logo. summary publisher spend never toggle brightness card. stack. horizontal cta picker. one-time. picker toptobottom one-time cta. locked mockup screen creative stack cta stack. daily card. screen vertical cta. summary stack media tom onetime locked repeated logo top-to-bottom continue never one-time. brightness mode card toggle top budget layout logo. toptobottom picker publisher picker. creative one-time row first horizontal tiers mockup spend recurring schedule scrollable continue recurring onetime first repeated cta. horizontal picker one-time stack stack. top mode summary scrollable tiers locked brightness toggle row vertical toptobottom tom daily layout creative never media cta picker. publisher card. logo screen one-time. schedule budget logo. mockup top-to-bottom card spend locked repeated row stack schedule horizontal card spend recurring mockup continue creative card. stack. toggle daily logo. cta. picker. summary top brightness scrollable tom onetime media layout screen toptobottom cta top-to-bottom never one-time mode vertical logo first budget one-time. tiers picker publisher stack. card logo. budget never tom mockup top summary row one-time. picker. vertical tiers onetime screen creative horizontal daily publisher logo toggle scrollable picker card. spend toptobottom cta layout schedule repeated continue mode locked media stack top-to-bottom cta. brightness first one-time recurring spend picker locked screen cta row daily vertical creative continue logo tom recurring toptobottom layout mockup toggle first stack. brightness publisher never picker. summary card schedule tiers onetime repeated logo. cta. scrollable one-time top card. horizontal top-to-bottom budget media stack mode one-time. cta card brightness picker. card. one-time. tom cta. summary media top repeated daily toptobottom first tiers creative layout publisher stack. one-time row continue vertical top-to-bottom recurring logo. scrollable never screen picker locked spend horizontal mode budget onetime logo toggle stack mockup schedule

LOCKED (Tom, repeated): Budget screen layout top-to-bottom: (1) Spend mode toggle at TOP — Daily recurring first, then One-time. (2) Brightness tiers as HORIZONTAL SCROLLABLE row — NEVER vertical stack. (3) Media mockup: creative ON publisher media with publisher logo. (4) Schedule picker. (5) Summary card. (6) Continue CTA.

---

#### Rule: media mockup must show creative ON publisher media with logo

- **ID:** `rul_17qtllqm`
- **Type:** rule
- **Confidence:** 0.95
- **Created:** 2026-03-16T17:33:56.366Z
- **Project:** Glo
- **Locked:** YES
- **Keywords:** media mockup publisher logo creative preview visualization live on media disruptive concept budget confirm screen UI

LOCKED (Tom): Every screen that shows the users creative must wrap it in a media mockup frame — publisher color bar at top with icon+logo, creative viewport below, Your Glo on [publisher] overlay. This reinforces Glos disruptive concept: anyone can upload anything (pending moderation) to any media. The user must constantly see that THEIR content will be LIVE on real media they selected. Applies to: Budget screen, Confirm screen, and any future screen showing the creative.

---

#### Rule: every screen component must have a UI CONTRACT comment at top

- **ID:** `rul_9oosw82x`
- **Type:** rule
- **Confidence:** 0.95
- **Created:** 2026-03-16T17:33:56.366Z
- **Project:** Glo
- **Locked:** YES
- **Keywords:** UI contract comment regression prevention screen component layout invariant documentation

To prevent UI regressions, every screen component file must start with a UI CONTRACT comment block documenting: layout order (numbered, top to bottom), visual invariants (horizontal vs vertical, what must always be visible), and any MUST NOT CHANGE items. When editing a screen file, read the contract first and never violate it. This was added after repeated Budget screen regressions.

---

#### Post-edit verification: Re-parse and confirm which screens changed

- **ID:** `0db73af733d54134a19ac65d80ae04ae`
- **Type:** rule
- **Confidence:** 0.95
- **Created:** 2026-03-18T02:01:34.586295Z
- **Project:** Glo
- **Locked:** YES
- **Keywords:** verification, parse, confirmation, change tracking

After each edit round, re-parse file and explicitly confirm which screens were modified and which were untouched. Prevents accidental cross-file contamination.

---

#### Feedback: Prefix screen name to scope edits (prevent cross-screen changes)

- **ID:** `2733c98cc8aa484f97ad4a3c36f142a7`
- **Type:** rule
- **Confidence:** 0.88
- **Created:** 2026-03-18T02:01:34.609904Z
- **Project:** Glo
- **Locked:** YES
- **Keywords:** feedback format, screen prefix, scope isolation

Format: 'Budget screen: move duration above tier cards'. Each feedback treated as isolated edit — assistant will not let changes bleed across screens.

---

#### Creatify API: preview_list_async endpoint — cost and response

- **ID:** `4eef853a974542c4a158732e40a1a66c`
- **Type:** rule
- **Confidence:** 0.85
- **Created:** 2026-03-18T02:01:34.636861Z
- **Project:** Glo
- **Locked:** YES
- **Keywords:** preview_list_async, 1 credit, webhook, media_job

Endpoint: POST /api/link_to_videos/preview_list_async. Cost: 1 credit per 30 seconds. Response: webhook with status (pending|in_queue|running|failed|done) and previews array {media_job: UUID, url: iframeable_preview_url}. Required: link (UUID). Optional: override_avatar, override_voice. Generates multiple style options for user selection before full render.

---

#### Session-activity progressive warnings: 0 remembers→ALERT, 8+ edits→nudge, 15+→stronger

- **ID:** `eea4d95bde284ff596bd98ab7ab500d5`
- **Type:** rule
- **Confidence:** 0.95
- **Created:** 2026-03-18T02:01:34.833933Z
- **Project:** Glo
- **Locked:** YES
- **Keywords:** session-activity, progressive warnings, encoding gap, threshold-based alerts

Pre-edit hook queries /session-activity counter. If zero /remember calls after 3+ min: inject 'ENCODING ALERT: You have not stored ANY learnings in the brain this session.' If 8+ edits since last /remember: gentler nudge. If 15+: stronger. Warnings injected directly into context, not in SKILL.md where they can be ignored.

---

#### Tmemory: Boot script includes npm install fallback for writable locations

- **ID:** `32f10cfe18a34391984d61e5a015233b`
- **Type:** rule
- **Confidence:** 0.85
- **Created:** 2026-03-18T02:01:34.862612Z
- **Project:** Glo
- **Locked:** YES
- **Keywords:** npm install, boot script, fallback, node_modules, flexibility, writable paths

Added `npm install` to boot-brain.sh as fallback. If node_modules/ doesn't exist or is incomplete at startup, boot script runs npm install before starting server. Allows flexibility: plugin can ship with bundled deps for readonly paths, or install fresh on writable paths. Both strategies supported.

---

#### Script approach object must include key field (creative-director.js)

- **ID:** `7c4542c56a1a44958e9116977e480081`
- **Type:** rule
- **Confidence:** 0.85
- **Created:** 2026-03-18T02:01:35.167481Z
- **Project:** Glo
- **Locked:** YES
- **Keywords:** creative-director, pickScriptApproach, object-shape, creatify-adapter-bug

**Failure**: pickScriptApproach() returns object without `key`. Adapter tries `SCRIPT_MAP[brief.scriptApproach?.key]` → always undefined → falls back to 'problem_solution'.

**Fix**: Return `{ key: 'call_to_action', name: 'Direct CTA', description: '...' }`

**Location**: creative-director.js pickScriptApproach function

---

#### Knowledge surfacing (activation) ≠ storage — activate at decision time

- **ID:** `04ac11acccc84669b86305ebe6a5b0e5`
- **Type:** rule
- **Confidence:** 0.85
- **Created:** 2026-03-18T02:01:35.193976Z
- **Project:** Glo
- **Locked:** YES
- **Keywords:** memory activation surfacing decision-time constraint principle domain

Storing a constraint is necessary but insufficient. The brain must surface it when user is about to make a decision that touches that constraint. Example: Stored 'sandbox network restriction' but didn't fire when Claude considered npm install (filesystem permission, same constraint class abstractly). Solution: abstract constraints to underlying principle ('sandbox execution restricted'), then activate proactively when user approaches ANY action in that domain. Prevents repeating mistakes in different forms.

---

#### Creatify: linkData.id (not linkData.link.id) for preview API

- **ID:** `664bbbaee2544c69aed90680c125c4bb`
- **Type:** rule
- **Confidence:** 0.85
- **Created:** 2026-03-18T02:01:35.226164Z
- **Project:** Glo
- **Locked:** YES
- **Keywords:** Creatify, API, link ID, preview_list_async, linkData, response structure, 400 error

Creatify link endpoint returns { id: <link_PK>, link: { id: <internal_id>, ... } }. Use the TOP-LEVEL linkData.id for preview_list_async, never the nested linkData.link.id. The nested ID is an internal object reference, not the link PK. The bug was code doing linkData.link?.id || linkData.id which grabbed the wrong nested ID first, causing 400 errors. Fix applied to server.js: always use linkData.id.

---

#### User feedback: Avoid unnecessary logins (Tom prefers not to log in if not needed)

- **ID:** `fa9794fdd6ef4055a5380b9b5f4ff4d2`
- **Type:** rule
- **Confidence:** 0.85
- **Created:** 2026-03-18T02:01:35.254632Z
- **Project:** Glo
- **Locked:** YES
- **Keywords:** efficiency, don't repeat, wasting time, reframe, better faster, quality preserve

User after 3 retries of failing approach: 'repeating something that doesn't work while taking hours of my work time is really inefficient. We are both wasting time.' Core principle: when approach fails, the real interesting question is 'how to make it better, faster without degrading quality?' — not 'how do I keep this alive?' Tom values rapid reframing and iteration over brute-force retry loops. Waste of time is the key concern.

---

#### Publisher type case mapping: uppercase UI → lowercase backend

- **ID:** `0bed2f8b7f49451f9be30ef19ef1bc71`
- **Type:** rule
- **Confidence:** 0.85
- **Created:** 2026-03-18T02:01:35.312921Z
- **Project:** Glo
- **Locked:** YES
- **Keywords:** publisher-type, normalization, case-sensitivity, server.js

**Mismatch**: Frontend sends pub.type as 'CTV', 'Online', 'DOOH'. Backend (media-intelligence, creative-director) expects 'ctv', 'web_video', 'display'.

**Fix**: Map in server.js before passing to media-intelligence.

Mapping: CTV→ctv, Online→web_video, DOOH→display, Social→social_feed

---

#### Frontend polling persists stale job IDs after server restart

- **ID:** `1c60283bb101416c905a9f6efbf79525`
- **Type:** rule
- **Confidence:** 0.88
- **Created:** 2026-03-18T02:01:35.362609Z
- **Project:** Glo
- **Locked:** YES
- **Keywords:** job polling, state management, server restart, frontend state, react, localStorage, job id

When server restarts, frontend continues polling old job ID from previous generation. Server returns 404 'Job not found', but polling loop doesn't clear (runs every 2 seconds, 5-min max timeout). Root cause: React state or localStorage retains stale job ID. **Fix:** Full page refresh to reset state and clear stale ID, enabling fresh generation request.

---

#### Browser fetch to Creatify API blocked by CORS — use server-side proxy instead

- **ID:** `b3a810dc88e24e6885a9e7db124d0800`
- **Type:** rule
- **Confidence:** 0.85
- **Created:** 2026-03-18T02:01:35.387751Z
- **Project:** Glo
- **Locked:** YES
- **Keywords:** CORS, browser fetch, Creatify API, cross-origin, testing

Attempted browser fetch from httpbin.org to Creatify endpoints → CORS error (TypeError: Failed to fetch). Browser-based direct API calls won't work. For testing: use server-side proxy, curl from terminal, or Creatify's API playground (requires login, which user prefers to avoid).

---

#### Repeating errors in skill get high priority

- **ID:** `6b2986b2d5e34e31a19216aa429e9b29`
- **Type:** rule
- **Confidence:** 0.95
- **Created:** 2026-03-18T02:01:35.412649Z
- **Project:** Glo
- **Locked:** YES
- **Keywords:** errors, skill, stability, priority, debugging, repeating

When the same failure recurs in tmemory (the skill/brain itself), flag immediately to Tom. Example: brain process crashing 5-6 times in one session. Don't defer to later, don't bundle with other issues — bring it to the top. This is different from bugs in Glo; the skill's stability is foundational.

---

#### Tmemory: Must add uncaughtException and unhandledRejection handlers to prevent silent crashes

- **ID:** `5d90e967164f49d3bc1d4010ba94a5f9`
- **Type:** rule
- **Confidence:** 0.85
- **Created:** 2026-03-18T02:01:35.494081Z
- **Project:** Glo
- **Locked:** YES
- **Keywords:** uncaughtException, unhandledRejection, error handling, crash protection, silent failures, index.js

Without crash handlers, unhandled exceptions or rejected promises silently kill the server with no error logged. Added process.on('uncaughtException', handler) and process.on('unhandledRejection', handler) to index.js. Now captures stack traces, logs them, and allows graceful shutdown instead of invisible failures.

---

#### Creatify POST /api/link_to_videos response fields: video_output, video_thumbnail (not output)

- **ID:** `c4996043b31b484b9969c35a5265be56`
- **Type:** rule
- **Confidence:** 0.85
- **Created:** 2026-03-18T02:01:35.662813Z
- **Project:** Glo
- **Locked:** YES
- **Keywords:** Creatify API, video_output, video_thumbnail, response fields, field names

The Creatify API returns video_output and video_thumbnail in the link_to_videos response, NOT a generic output field. This was the root cause of field name bugs in server.js. Ensure response parsing uses exact field names from API docs.

---

#### API Integration: Live API behavior supersedes documentation

- **ID:** `ac4c6634ee994e499f5bf16feab883b5`
- **Type:** rule
- **Confidence:** 0.85
- **Created:** 2026-03-18T02:01:35.759037Z
- **Project:** Glo
- **Locked:** YES
- **Keywords:** API integration, documentation vs reality, debugging principle

When API docs contradict actual API behavior (auth, endpoints, parameters), live behavior is correct. Check dashboard code examples and actual error responses. Docs can be outdated/simplified. Pattern: failures → check live examples → apply real behavior.

---

#### Destructive operations require context-awareness — execution environment matters

- **ID:** `1332d6fbc2204baab8db2b9f72b49d89`
- **Type:** rule
- **Confidence:** 0.95
- **Created:** 2026-03-18T02:01:35.804728Z
- **Project:** Glo
- **Locked:** YES
- **Keywords:** destructive rm -rf context execution environment sandbox consequences

Generic fixes (e.g., 'rm -rf node_modules && npm install') are context-dependent. Example: used to clear Vite cache but sandbox blocked deletion. If it hadn't, would have deleted user's real dependency tree over a cache issue. Actual fix: surgical 'rm -rf node_modules/.vite' or advise user to run vite from their terminal. Principle: always consider execution context and real consequences before destructive operations, even if recipe works generically.

---

#### Blocker: Creatify preview_list_async returns 400 'Invalid pk'

- **ID:** `402184ea3115405f8115d69cfbfc43df`
- **Type:** rule
- **Confidence:** 0.85
- **Created:** 2026-03-18T02:01:35.946400Z
- **Project:** Glo
- **Locked:** YES
- **Keywords:** creatify, preview_list_async, 400 error, Invalid pk, link ID, unresolved

generatePreviews endpoint returns HTTP 400: { "link": ["Invalid pk \"dae9efcb-...\" - object does not exist."] }. Link ID passed to preview_list_async is incorrect. Link creation response has structure { id: ..., link: { id: ..., ... } } but current linkData.link?.id || linkData.id fallback still fails. Need to verify exact ID structure Creatify expects.

---

#### Encoding extractor incomplete: missing concept, context, person, project, task node types

- **ID:** `6864632666bb4d0c993c18d8c04fdbf4`
- **Type:** rule
- **Confidence:** 0.85
- **Created:** 2026-03-18T02:01:36.001179Z
- **Project:** Glo
- **Locked:** YES
- **Keywords:** encoding extractor, node types, concept, context, person, project, task

Original extraction only generated decision, rule, file nodes. Missed 5 critical types per SKILL.md: concept (new terms), context (current state), person (roles), project (products), task (work items). Enhanced relearning.py to detect all 8 types.

---

#### Confidence review queue at boot — surface unconfirmed learnings

- **ID:** `61a8562012544411b10644a2524f852a`
- **Type:** rule
- **Confidence:** 0.85
- **Created:** 2026-03-18T02:01:36.026220Z
- **Project:** Glo
- **Locked:** YES
- **Keywords:** review queue, boot hook, scheduled nudge, staged learnings

Tom strongly supports confidence tiering + review queue concept. Boot hook should surface pending staged learnings (low confidence) as a review queue for user approval. Scheduled procedures can also run nudges on stale unconfirmed items. This prevents false axioms while keeping tentative ideas captured.

---

#### Brain bridging bug: _bridge_at_store_time and _find_bridge_candidates not creating sufficient edges

- **ID:** `7f7026d9ea8f4a7294d08d5b350dd868`
- **Type:** rule
- **Confidence:** 0.85
- **Created:** 2026-03-18T02:01:36.056353Z
- **Project:** Glo
- **Locked:** YES
- **Keywords:** bridging, edges, _bridge_at_store_time, _find_bridge_candidates, Hebbian learning, 116 edges, 933 nodes

Relearning simulation created 933 nodes but only 116 edges (vs 946 in current brain). Root cause: bridging mechanism not automatically connecting related nodes at encode time. Hebbian learning (_hebbian_strengthen) verified in place but bridging candidates not being found or connected properly.

---

#### Unconfirmed info stays contextual until earned through repetition

- **ID:** `f7dc7c3dd04f48e59d5f91965bb65d4a`
- **Type:** rule
- **Confidence:** 0.95
- **Created:** 2026-03-18T02:01:36.137599Z
- **Project:** Glo
- **Locked:** YES
- **Keywords:** confidence, contextual, earned authority, not axiom

**From Tom:** 'I think user-confirmed VS inferred is dangerous, even when Me or anyone else say something, it was part of the context and we might not think its an axiom. It's great to log it but in context of the situation.' Contextual statements should not immediately lock as truth — they earn weight only through repeated mention or explicit confirmation. **Why:** Casual conversation, exploration, and passing mentions can be misinterpreted as deliberate decisions. **How to apply:** Default confidence <0.5 for new learnings; only promote to locked/primary after revalidation or explicit user confirmation.

---

#### Creatify model_version costs: standard 5cr/30s, aurora_v1 20cr/15s, aurora_v1_fast 10cr/15s

- **ID:** `62d1024bf6194e1aa115baca6847375e`
- **Type:** rule
- **Confidence:** 0.85
- **Created:** 2026-03-18T02:01:36.211380Z
- **Project:** Glo
- **Locked:** YES
- **Keywords:** model_version, aurora_v1, standard, credits, video cost, Creatify pricing

The model_version parameter in Creatify link_to_videos endpoint controls cost and quality. Standard: 5 credits for 30 seconds (cheapest). Aurora_v1: 20 credits for 15 seconds (high quality). Aurora_v1_fast: 10 credits for 15 seconds (balanced). Glo wallet: 1 credit = $1 USD, so cost per video ranges $5–$20.

---

#### NanoBanana API: Authorization Bearer header required, not X-API-Key

- **ID:** `a65b6e7a54054cdeb189973d937d3d15`
- **Type:** rule
- **Confidence:** 0.85
- **Created:** 2026-03-18T02:01:36.239601Z
- **Project:** Glo
- **Locked:** YES
- **Keywords:** NanoBanana, Bearer, Authorization, authentication, header, API

NanoBanana video API uses `Authorization: Bearer {apiKey}` header format for all endpoints, including the polling/status endpoints. Do NOT use X-API-Key header. This constraint applies to all adapter calls (textToVideo, imageToVideo, pollStatus, etc.).

---

#### Tom: Focus on one component/thread at a time, don't go wide and deep simultaneously

- **ID:** `97e9a7994be647fd8e7c0fc632819a30`
- **Type:** rule
- **Confidence:** 0.85
- **Created:** 2026-03-18T02:01:36.261206Z
- **Project:** Glo
- **Locked:** YES
- **Keywords:** thought brain-observation nasdaq mgni rankings dream growth share componentthread tom replay focus cluster geniee log private session forming market converging simultaneously adtech researching time don wide trends neighbors converging. dsp magnite u.s. areas component/thread comparable deep

Cluster forming: "Researching DSP market trends and private DSP growth rankings" and "Tom: Focus on one component/thread at a time, don't go wide and deep simultaneously" share 19 neighbors (Session Log — Replay #2 | Magnite — U.S. adtech comparable to Geniee (NASDAQ: MGNI) | Dream: Dream c). These areas are converging.

---

#### Graceful degradation: when embedder unavailable, fall back to TF-IDF

- **ID:** `ab6fef5436f74e65b1408363a0e000c7`
- **Type:** rule
- **Confidence:** 0.95
- **Created:** 2026-03-18T02:01:36.308860Z
- **Project:** Glo
- **Locked:** YES
- **Keywords:** graceful degradation, fallback, TF-IDF, resilience

**Why**: Handle network failures, VM constraints, model load errors. **How to apply**: embedder_ready=false halts embedding but does not break recall. System reports embeddings_ready=0 and serves TF-IDF only until model loads. All endpoints functional, semantic recall just blocked.

---

#### Tom: Prioritizes proven track record in research and solutions

- **ID:** `44e0d24b5b7c415091e7b727bd7f347c`
- **Type:** rule
- **Confidence:** 0.85
- **Created:** 2026-03-18T02:01:36.370953Z
- **Project:** Glo
- **Locked:** YES
- **Keywords:** Tom, research, validation, precedent, depth-first

When proposing solutions, validate with existing precedents or research, not speculation. Tom asks 'Ask as many questions as you need to solve/refine the entire thing' — he wants thorough exploration before execution. Design for Tom's context-first approach.

---

#### Rule: Ask clarifying questions during encoding

- **ID:** `a4384ceba4cb43f088d6b0a7adeb7b77`
- **Type:** rule
- **Confidence:** 0.85
- **Created:** 2026-03-18T02:01:36.667465Z
- **Project:** Glo
- **Locked:** YES
- **Keywords:** curiosity questions clarifying encoding improvement thin nodes

Brain should be curious and ask Tom questions to improve its own understanding, not passively extract. This is part of learner mode: 'ask ME' to collect more data and build semantic depth. Questions surface thin nodes and gaps in understanding.

---

#### Feedback: Discuss pros/cons BEFORE committing sensitive changes

- **ID:** `069c8f15ab2f41deaab6bb274f73ff19`
- **Type:** rule
- **Confidence:** 0.85
- **Created:** 2026-03-18T02:01:36.683323Z
- **Project:** Glo
- **Locked:** YES
- **Keywords:** discuss pros-cons feedback Tom sensitive-changes dialogue refine-together

User explicit feedback after schema refactoring + brain audit marathon session. When making decisions about hooks, compaction strategy, real-time capture, consolidation, or architecture changes: discuss tradeoffs and pros/cons first. Refine together through dialogue rather than shipping unilaterally. Especially critical for memory and compaction infrastructure.

---

#### Hooks fire successfully; encoding sparseness was strategy problem, not infrastructure

- **ID:** `98f3fc5f52e44c6fa6dbab50f959cbef`
- **Type:** rule
- **Confidence:** 0.95
- **Created:** 2026-03-18T02:01:36.709572Z
- **Project:** Glo
- **Locked:** YES
- **Keywords:** hook fires, pre-compact 162, auto-suggest 61, extraction strategy, encoding decision point

Tracked hook fires across session: 162 pre-compact fires, 61 auto-suggest fires, 378 /remember API calls, 104 /staged/add calls. Hooks work. The issue: what gets extracted and when. Pre-compact logic was competing with compaction for resources. Real-time encoding depended on Claude manually choosing what to remember. Solution: defer heavy extraction to post-compaction, use density to detect gaps.

---

#### Report only what you deem important; ask if unsure; abstract rules over metrics

- **ID:** `d1618661e7914934992d0b894ac300b5`
- **Type:** rule
- **Confidence:** 0.85
- **Created:** 2026-03-18T02:01:36.812495Z
- **Project:** Glo
- **Locked:** YES
- **Keywords:** mechanical reporting, editorial judgment, importance, abstract rules, metrics-agnostic

User feedback: Don't report mechanically ('X bridges created, Y consolidated'). Report only perceived importance. If unsure what matters, ask. Focus on abstract principles and rules, not counts and metrics. Numbers are implementation; principles are understanding that transfers across contexts.

---

#### React Hook Violation: Cannot call useState/useEffect inside conditionals

- **ID:** `7be6b8c80d694568a9a7d2227a89a4d7`
- **Type:** rule
- **Confidence:** 0.88
- **Created:** 2026-03-18T02:01:36.907684Z
- **Project:** Glo
- **Locked:** YES
- **Keywords:** React, hooks, useState, useEffect, conditional, illegal, error, hoist

React enforces: No hooks (useState, useEffect, etc.) inside conditional blocks like 'if (screen === "creative")'. Causes crashes. Fix: Move ALL hook initialization to component top level BEFORE any conditional rendering.

---

#### Catch and fix demo UI regressions immediately

- **ID:** `71741847e7454765b3a717670ebb9cd2`
- **Type:** rule
- **Confidence:** 0.88
- **Created:** 2026-03-18T02:01:36.930220Z
- **Project:** Glo
- **Locked:** YES
- **Keywords:** regression, demo stability, UI testing, feature tracking, broken flow

Tom flagged: 'The UI seems like some older version of the demo with Business name or website in the first field. again regression'. When making changes, test full user flow and verify new changes don't revert previous fixes. Demo stability critical for testing.

---

#### Principle extraction: triggered by thinking corrections, not code fixes

- **ID:** `7065e92e2c504a04af33060be0635778`
- **Type:** rule
- **Confidence:** 0.85
- **Created:** 2026-03-18T02:01:37.067033Z
- **Project:** Glo
- **Locked:** YES
- **Keywords:** principle extraction, thinking correction, abstraction, mental model, redirect learning

The pattern for extracting principles from Tom's approach: when he corrects Claude's thinking (not code), ask him to abstract it further. Structure: describe the idea → abstract the principle → identify related principles → propose how the brain uses them. Corrections to thinking reveal underlying mental models; code fixes don't.

---

#### Code-as-memory: Deep comments explain WHY concepts, not WHAT code does

- **ID:** `7b8d364996994641b60c60ca5f13c7de`
- **Type:** rule
- **Confidence:** 0.85
- **Created:** 2026-03-18T02:01:37.129790Z
- **Project:** Glo
- **Locked:** YES
- **Keywords:** comments, documentation, future-self, memory, code clarity, WHY not WHAT

This codebase IS the brain's memory. Future Claude instances will read this cold. Comments should explain deeper concepts, design choices, and WHY each decision was made (not just WHAT the line does). This helps future-you recall the reasoning without re-discovering it.

---

#### Excel models MUST have zero formula errors

- **ID:** `b6e1a2c7348a41acbcee6fb16fa8ef45`
- **Type:** rule
- **Confidence:** 0.85
- **Created:** 2026-03-18T02:01:37.177971Z
- **Project:** Glo
- **Locked:** YES
- **Keywords:** Excel, zero errors, validation, quality

All Excel deliverables validated with zero formula errors (#REF!, #DIV/0!, #VALUE!, #N/A, #NAME?). Non-negotiable quality standard. Use recalculation script to validate.

---

#### Tom: Plan-first approach for bigger questions

- **ID:** `11ee43410e744d3284b6eda0d67e39cd`
- **Type:** rule
- **Confidence:** 0.85
- **Created:** 2026-03-18T02:01:37.199789Z
- **Project:** Glo
- **Locked:** YES
- **Keywords:** plan, strategy, phases, context

For larger strategic questions, present high-level plan first (phase 2, then phase 1) before diving into detailed flows. Start with users and goals, then zoom into specifics.

---

#### Tom needs step-by-step terminal guidance without jargon

- **ID:** `5c1791f3744e4e4fba7339528658561e`
- **Type:** rule
- **Confidence:** 0.95
- **Created:** 2026-03-18T02:01:37.260228Z
- **Project:** Glo
- **Locked:** YES
- **Keywords:** terminal guidance, newbie-friendly, step-by-step, pwd, npm install, explain packages

Tom explicitly said 'I'm not good with terminal, guide me there like a newb and explain what each thing does'. Provide step-by-step explanations for every command, explain what each package does, avoid technical jargon. Example: when he asked 'how do I ask the terminal what's my current full path', provide pwd command with full explanation of what it means.

---

#### Creatify link ID format: 400 invalid pk error (active blocker)

- **ID:** `9ed151e69b2a40a6be8792e8ff4cebe8`
- **Type:** rule
- **Confidence:** 0.85
- **Created:** 2026-03-18T02:01:37.282272Z
- **Project:** Glo
- **Locked:** YES
- **Keywords:** creatify-api, link-id, preview-list-async, error-400, debugging

**Error**: preview_list_async returns 400: `{"link": ["Invalid pk \"dae9efcb-...\" - object does not exist."]}`

**Issue**: Link ID being passed is wrong format. Response structure is `{ id: ..., link: { id: ... } }`. Code tries `linkData.link?.id || linkData.id`.

**Next**: Verify Creatify docs for exact link parameter format expected by preview_list_async.

---

#### Creatify: preview_list_async key parameters

- **ID:** `811e073831d14673ba36fd27ef007a03`
- **Type:** rule
- **Confidence:** 0.85
- **Created:** 2026-03-18T02:01:37.307111Z
- **Project:** Glo
- **Locked:** YES
- **Keywords:** script_style, visual_styles, target_platform, aspect_ratio, video_length

visual_styles (template array), script_style (BenefitsV2, BrandStoryV2, CallToActionV2, DiscoveryWriter, DontWorryWriter, EmotionalWriter, GenZWriter, HowToV2, LetMeShowYouWriter, MotivationalWriter, ProblemSolutionV2, ProblemSolutionWriter, ProductHighlightsV2, plus LegoScript variants), target_platform (default 'tiktok'), target_audience (default 'young adults'), language, video_length (15/30/45/60), aspect_ratio (9x16, 16x9, 1x1), override_script, no_caption.

---

#### Split responses: internal (Claude→brain) separate from external (Claude→user)

- **ID:** `e3c462a152e84ebf88142f75dc8c8b93`
- **Type:** rule
- **Confidence:** 0.85
- **Created:** 2026-03-18T02:01:37.343377Z
- **Project:** Glo
- **Locked:** YES
- **Keywords:** response splitting, internal vs external, two audiences, editorial voice, debug reasoning

When reasoning aloud would help the brain more than the user, fork the response. Internal reasoning (debug traces, bridge justifications, graph exploration) can be verbose and structural. External communication is curated for perceived importance. Two audiences, two voices.

---

#### Tom: Always present pros/cons and alternatives, challenge ideas

- **ID:** `9fe1d89c711b4a5e94b7af1a0f4bb46a`
- **Type:** rule
- **Confidence:** 0.85
- **Created:** 2026-03-18T02:01:37.392772Z
- **Project:** Glo
- **Locked:** YES
- **Keywords:** feedback, challenge, alternatives, critical thinking

Tom wants to be challenged, not agreed with. Present opposing viewpoints, trade-offs, and alternative approaches with reasoning. Don't be a yes-man.

---

#### Tom: Known UX patterns over novel invention

- **ID:** `d6605022c69d4c009caac74fe0352c8f`
- **Type:** rule
- **Confidence:** 0.85
- **Created:** 2026-03-18T02:01:37.415764Z
- **Project:** Glo
- **Locked:** YES
- **Keywords:** UX pattern, precedent, competitor, established

Default to proven competitor implementations and established UX patterns rather than inventing novel approaches. Reference existing successful products.

---

#### Tom: Context must be precise, vetted, never inferred or vague

- **ID:** `3d892f8c3d5f493e97e05281742f71be`
- **Type:** rule
- **Confidence:** 0.95
- **Created:** 2026-03-18T02:01:37.435858Z
- **Project:** Glo
- **Locked:** YES
- **Keywords:** context, precision, vetting, confirmation

When adding to Tom.md or context files, ensure every statement is specific, factual, confirmed with Tom. Never add inferred or vague information. Get explicit approval before adding.

---

#### Tom: Hidden optionality in UI, not in-your-face optional fields

- **ID:** `b9556fa53df34ec686193a83a8adef4a`
- **Type:** rule
- **Confidence:** 0.85
- **Created:** 2026-03-18T02:01:37.456583Z
- **Project:** Glo
- **Locked:** YES
- **Keywords:** UX design, optionality, simplicity, hidden

Provide advanced capabilities but don't surface by default. Avoid showing 10 fields marked 'optional.' Keep core flow simple and reveal optionality only when needed.

---

#### Tom: Always add Todo file for personal thoughts; surface when contextually relevant

- **ID:** `c61eb6a28dd6465ba8b599eaddfbbf01`
- **Type:** rule
- **Confidence:** 0.85
- **Created:** 2026-03-18T02:01:37.477914Z
- **Project:** Glo
- **Locked:** YES
- **Keywords:** Tom, Todo, personal thoughts, tracking, contextual awareness

Tom wants dedicated Todo.md for his own notes and reasoning. Claude should proactively bring up todos when relevant to current conversation. Helps Tom track progress and thinking.

---

#### Video generation API: Generate → Thumbnail → Crop per aspect ratio

- **ID:** `f54375a235c54da3a8b97d04b698dd9a`
- **Type:** rule
- **Confidence:** 0.85
- **Created:** 2026-03-18T02:01:37.533528Z
- **Project:** Glo
- **Locked:** YES
- **Keywords:** P2V preview image approval CreativeBrief imagePrompt videoPrompt taskId Waymark

Product-to-Video (Creatify API) follows: generate preview IMAGE → user reviews/approves → generate VIDEO. CreativeBrief includes separate imagePrompt (photography/product direction) and videoPrompt (motion/transitions direction). PreviewCard renders images alongside iframes; render endpoint uses taskId from selectedPreview for video generation. This two-stage approach improves UX and reduces video generation cost.

---

#### Memory discipline: selective logging, not comprehensive tracking

- **ID:** `903daa840be84b4abf608bc7c04409e8`
- **Type:** rule
- **Confidence:** 0.95
- **Created:** 2026-03-18T02:01:37.641450Z
- **Project:** Glo
- **Locked:** YES
- **Keywords:** memory-discipline, logging-strategy, tmemory-usage, feedback

**User feedback** (Tom): "are you frequently logging stuff to tmemory?"

**Preference**: Use memory only for decisions, rules, gotchas that matter across sessions. Skip system messages, status updates, ephemeral work state.

**How to apply**: Before saving—ask "Will this be useful in a future session?" If no, don't encode it.

---

#### Glo video ads: no avatars

- **ID:** `ad09d8c0303444e090d167bf75bcfe33`
- **Type:** rule
- **Confidence:** 0.85
- **Created:** 2026-03-18T02:01:37.737811Z
- **Project:** Glo
- **Locked:** YES
- **Keywords:** avatars, creative constraints, visual styles, product requirement, template filtering

Hard product constraint: Glo ads must not use avatars or avatar-dependent visual styles. Filter avatar templates from Creatify template selection. Remove avatar-based visual approaches from pickVisualApproaches (e.g., green_screen). Tom explicitly: 'lets not use avatars.'

---

#### Creatify brand safety: logo directive in video_prompt, not narration

- **ID:** `702b0f4cbc294d1bb813db37f8a60595`
- **Type:** rule
- **Confidence:** 0.85
- **Created:** 2026-03-18T02:01:37.758712Z
- **Project:** Glo
- **Locked:** YES
- **Keywords:** Creatify, brand safety, logo, override_script, video_prompt, narration

Logo/brand directives (e.g., 'only use Aniview logo') must be passed in video_prompt, not image_prompt or narrator text. They constrain visual generation, not narration. Directive was incorrectly placed in narrator layer.

---

#### API Integration: Read docs first, plan before executing

- **ID:** `a11768af462e4ba5a65da4da511b5655`
- **Type:** rule
- **Confidence:** 0.85
- **Created:** 2026-03-18T02:01:37.778679Z
- **Project:** Glo
- **Locked:** YES
- **Keywords:** documentation, methodology, planning, API, debugging, technical debt

Do not attempt API integration through iterate-and-patch debugging cycles. Professional approach: (1) read official documentation thoroughly, (2) research online examples and community discussion, (3) plan the full integration, (4) execute once. Patching after failures makes code ugly and complex, accumulating technical debt. This applies to any external API integration.

---

#### Creatify: model_version='aurora_v1_fast' required in preview_list_async requests

- **ID:** `2244325e98d44683b094c617568ce125`
- **Type:** rule
- **Confidence:** 0.85
- **Created:** 2026-03-18T02:01:37.814638Z
- **Project:** Glo
- **Locked:** YES
- **Keywords:** creatify, model_version, aurora_v1_fast, required parameter, preview_list_async

**Failure**: Without model_version parameter, preview jobs timeout after 5 minutes. Status stays pending, no previews generated. **Fix**: Include model_version: 'aurora_v1_fast' in every preview_list_async request. This is standard Creatify parameter for fast preview rendering pipeline.

---

#### Creatify: Brand safety via override_script field, not narrator text

- **ID:** `dfe1098a08a84efca57a60f73d4a81c3`
- **Type:** rule
- **Confidence:** 0.85
- **Created:** 2026-03-18T02:01:37.854206Z
- **Project:** Glo
- **Locked:** YES
- **Keywords:** Creatify, brand safety, override_script, logo exclusion, competitor exclusion, API parameter

Brand directives (e.g., 'only Aniview logo') must be passed in override_script parameter, not embedded in narrator prompts. Narrator text generates voice narration; override_script injects production rules into the video template. Tom caught the mistake when logo guidance was mistakenly added to narrator text.

---

#### NanoBanana: image-to-video API, 3-12s duration, 9 credits @ 1080p/7s

- **ID:** `e6045baffe244b46a078f978226b3b85`
- **Type:** rule
- **Confidence:** 0.85
- **Created:** 2026-03-18T02:01:37.880896Z
- **Project:** Glo
- **Locked:** YES
- **Keywords:** NanoBanana, API, image-to-video, text-to-video, credits, pricing, duration, resolution, 1080p

API Endpoint: `POST /api/v1/image-to-video.php` with params: image_urls (array), prompt (string), resolution (480p/720p/1080p), duration (3-12 seconds). Status polling: `GET /api/v1/video-status.php?video_id=X` returns status: queued → processing → completed/failed. **Pricing:** 5 base credits, +2 for 1080p, +1 per second over 5s. Example: 7s @ 1080p = 5 + 2 + 2 = 9 credits. Also supports text-to-video with same params. Simpler than Creatify (one-step vs two-phase render/preview).

---

#### NanoBanana API: All endpoints require .php extension

- **ID:** `5200eac858f14407afc7cfcde3b79d15`
- **Type:** rule
- **Confidence:** 0.85
- **Created:** 2026-03-18T02:01:37.934485Z
- **Project:** Glo
- **Locked:** YES
- **Keywords:** NanoBanana, .php, endpoints, 404

Endpoints must be `/text-to-video.php`, `/image-to-video.php`, `/video-status.php`. Dashboard examples omit `.php` but live API returns 404 without it. **Fix:** Always include .php extension.

---

#### Google Places autocomplete: `types:'establishment'` filter too restrictive

- **ID:** `98e15f8c7f32467995eb4db261ca6fd3`
- **Type:** rule
- **Confidence:** 0.85
- **Created:** 2026-03-18T02:01:37.957739Z
- **Project:** Glo
- **Locked:** YES
- **Keywords:** Google Places API types establishment filter search

Filtering by `types:'establishment'` excludes location names and restaurants that don't fit the category. Mami restaurant in Cresskill didn't match. Fix: remove types filter entirely, let Google's default ranking return broader results. Test case: 'mami cresskill'.

---

#### API Quirk: NanoBanana docs claim image_urls array, live API rejects it

- **ID:** `a226c553079f429fa5a5b0dc10a28835`
- **Type:** rule
- **Confidence:** 0.85
- **Created:** 2026-03-18T02:01:38.143505Z
- **Project:** Glo
- **Locked:** YES
- **Keywords:** NanoBanana, image_url, image_urls, API inconsistency, HTTP 500

Testing confirmed: (1) image_urls parameter rejected as unknown field, (2) image_url with array value returns HTTP 500, (3) only image_url with single string works. Documentation misleading about multi-image support.

---

#### Feedback: User values self-correction and double-checking during debugging

- **ID:** `be8b5e1e6b3742f081920c0f8664e99c`
- **Type:** rule
- **Confidence:** 0.95
- **Created:** 2026-03-18T02:01:38.170236Z
- **Project:** Glo
- **Locked:** YES
- **Keywords:** self-correction validation debugging behavior preference Tom

Tom explicitly stated: 'I love that you check yourself and correct. Keep it up.' This indicates strong preference for the assistant to validate assumptions, detect errors, and correct them mid-execution rather than pushing ahead blindly.

---

#### Error pattern: Hook script path resolution in test environment

- **ID:** `c4794ea9776348d9a7679f0fced4d9ba`
- **Type:** rule
- **Confidence:** 0.85
- **Created:** 2026-03-18T02:01:38.200203Z
- **Project:** Glo
- **Locked:** YES
- **Keywords:** hook script path resolution bash error test environment

Encountered: `bash: /hooks/scripts/pre-edit-suggest.sh: No such file or directory`. The script path wasn't being resolved correctly in test context. Appears to be path context issue (relative vs absolute) rather than missing file — subsequent tests with corrected paths worked.

---

#### Context files: Pull → Check conflicts → Flag to user → Update both

- **ID:** `1dfdb6186efa4cb58334044bee971c2b`
- **Type:** rule
- **Confidence:** 0.85
- **Created:** 2026-03-18T02:01:38.253347Z
- **Project:** Glo
- **Locked:** YES
- **Keywords:** context file, conflict detection, pull, assimilate, flag, update, stale data, living memory

When pulling cached context file: 1) Assimilate with current state, 2) Flag conflicts (e.g., brain says 'vendor is Creatify' but code now uses NanoBanana'), 3) Present conflict to user, 4) Update both context file and brain node after resolution. Prevents stale information from driving decisions.

---

#### context-file/find: tag-based matching insufficient for discovery

- **ID:** `8e41beb587064c1381ba3bd47b2ae679`
- **Type:** rule
- **Confidence:** 0.88
- **Created:** 2026-03-18T02:01:38.311541Z
- **Project:** Glo
- **Locked:** YES
- **Keywords:** context-file/find, matching, tags, discovery gap, BudgetScreen

Current find endpoint uses string matching on tags/topics. Fails to discover relevant context files when query terms don't exactly match tags. Example: query 'BudgetScreen' returns empty despite glo-platform context file existing with tags ['glo', 'architecture']. Need semantic matching or smarter tagging strategy.

---

#### tmemory context file search: Remove artificial prefixes to prevent false positives

- **ID:** `044db3eb35684577ae58becaffc0625c`
- **Type:** rule
- **Confidence:** 0.85
- **Created:** 2026-03-18T02:01:38.367334Z
- **Project:** Glo
- **Locked:** YES
- **Keywords:** context file search false positive prefix injection bug

Bug: Injecting 'context_file' prefix into search query caused 'context' and 'file' keywords to match unrelated queries. Fix: Removed prefix from brain.js findRelevantContextFiles() search construction.

---

#### tmemory staged learning: Duplicate detection must use keyword overlap, not substring match

- **ID:** `29b797a0563a4bd7ac329946ef654db3`
- **Type:** rule
- **Confidence:** 0.88
- **Created:** 2026-03-18T02:01:38.393795Z
- **Project:** Glo
- **Locked:** YES
- **Keywords:** staged learning duplicate detection keyword overlap confidence scoring

Bug: Substring matching on title prefix (first 40 chars) missed semantic duplicates like 'Budget slider replaced tier cards' vs 'Tom prefers slider over tier cards for budget'. Fix: Use keyword overlap percentage. When duplicate found, increase existing node's confidence instead of creating new entry.

---

#### Pre-compact brain healthcheck: 6 fetches per session (verified in testing)

- **ID:** `275ea4f4cce74ad6a32383a4ca77236c`
- **Type:** rule
- **Confidence:** 0.85
- **Created:** 2026-03-18T02:01:38.437127Z
- **Project:** Glo
- **Locked:** YES
- **Keywords:** pre-compact healthcheck 6 times fetch verification

Before compaction triggers, Claude fetches brain ~6 times to verify it's reachable and inject extraction prompt. Tom: 'Before compaction — 6 times (from our testing). This is important, I'm willing to spend more time to guarantee brain isn't losing information.'

---

#### Store-time encoding must be smart, not exhaustive

- **ID:** `ab17fe309ff3440594966662a8822807`
- **Type:** rule
- **Confidence:** 0.95
- **Created:** 2026-03-18T02:01:38.714222Z
- **Project:** Glo
- **Locked:** YES
- **Keywords:** store-time, selective encoding, intentional, smart filtering

When encoding at store-time (during /remember), be selective and intentional. Don't dump everything from context. Analyze what's worth preserving. User feedback: 'Really important to invest in encoding'—quality over quantity.

---

#### Bridge weight system: Initial 0.15, bidirectional pairs, decay without reinforcement

- **ID:** `ed919bcaadfb47699ab4361e1034c2f3`
- **Type:** rule
- **Confidence:** 0.85
- **Created:** 2026-03-18T02:01:38.742054Z
- **Project:** Glo
- **Locked:** YES
- **Keywords:** bridge weight, 0.15, bidirectional, pairs, decay

Emergent bridges form as bidirectional edge pairs. Each proposal creates 2 edges (A→B, B→A). Initial weight is 0.15. Bridges decay over time unless reinforced by actual consolidation cycles confirming neighborhood overlap.

---

#### Bridge candidates: Require 2+ shared neighbors minimum

- **ID:** `f9fcac43222b4f0bb70bbc24fec8ca01`
- **Type:** rule
- **Confidence:** 0.85
- **Created:** 2026-03-18T02:01:38.764387Z
- **Project:** Glo
- **Locked:** YES
- **Keywords:** shared neighbors, threshold 2, bridge candidates, orphan nodes

Emergent bridges only form between nodes whose neighborhoods share 2 or more common connections. Orphan nodes (0 edges) generate no candidates. Prevents spurious long-range connections.

---

#### Dream creates intuition nodes → triggers emergent bridging

- **ID:** `2f00c36a10b94863b491017b0178c7af`
- **Type:** rule
- **Confidence:** 0.85
- **Created:** 2026-03-18T02:01:38.816281Z
- **Project:** Glo
- **Locked:** YES
- **Keywords:** dream, intuition nodes, bridging candidates, neighborhood overlap

When dream() generates new intuition node connections, those nodes immediately become bridging candidates if they inherit sufficient neighborhood overlap from their seed nodes. Intuition nodes act as bridge anchors.

---

#### Thoughts vs decisions/rules differ fundamentally in decay philosophy

- **ID:** `e52e01e22f36442080df30a7c4b40741`
- **Type:** rule
- **Confidence:** 0.85
- **Created:** 2026-03-18T02:01:38.968208Z
- **Project:** Glo
- **Locked:** YES
- **Keywords:** thought, decision, rule, decay philosophy, wall-clock, time-dilation, structural difference

Thoughts use wall-clock decay. Decisions/rules use time-dilation (idle time merciful). The boundary: thoughts are immediate-value insights (stale at 3h regardless of activity), decisions/rules are structural knowledge (can survive long absences). First node type to break the universal decay model.

---

#### Feedback: bge-base-en-v1.5 performance — 80ms single, 30-40ms batched, 768-dim, ~440MB

- **ID:** `e1a23d8954b447acbef768e7e4c06c44`
- **Type:** rule
- **Confidence:** 0.85
- **Created:** 2026-03-18T02:01:38.995368Z
- **Project:** Glo
- **Locked:** YES
- **Keywords:** bge-base-en-v1.5, latency, 80ms, 768-dim, performance, memory

**Why**: User needs baselines for sync load feasibility. **How to apply**: Single embedding ~80ms on CPU (tokenize+inference+pool), batches of 8-16 drop to 30-40ms each. Model footprint ~440MB ONNX weights. Embedding dimension 768. Total brain server footprint grows from ~50-80MB to ~130-160MB. Use for latency budgeting.

---

#### Rule: Semantic richness beats headlines

- **ID:** `ffbcead277c94b218ee7d6e9a80f84c6`
- **Type:** rule
- **Confidence:** 0.88
- **Created:** 2026-03-18T02:01:39.283309Z
- **Project:** Glo
- **Locked:** YES
- **Keywords:** high-level encoding ruins everything micro-decisions emotion causal chains context

High-level encoding compromises brain quality irreversibly. Must capture emotional reaction, micro-decisions, causal chains, and context behind ideas—not just facts. Tom's example: don't record 'Tom prefers simple moats' (headline); record 'the feeling when Tom said he doesn't think GLO will be the most advanced ad creator (that's not the moat) — someone burned by feature creep, who knows the trap of building impressive over important.'

---

#### Rule: Preserve numbers, proper nouns, names in keywords

- **ID:** `8792fa0e55504830b07f03a369185983`
- **Type:** rule
- **Confidence:** 0.85
- **Created:** 2026-03-18T02:01:39.317384Z
- **Project:** Glo
- **Locked:** YES
- **Keywords:** keyword extraction proper nouns numbers specific values NASDAQ MGNI 1080p

extractKeywords() now preserves specific values (NASDAQ: MGNI, 1080p, $1 USD, 30s duration) instead of stripping to generics. 'Magnite — U.S. adtech comparable to Geniee' is more valuable than 'adtech company'. Numbers, names, and proper nouns are part of semantic identity.

---

#### Rule: Emotional context matters for brain quality

- **ID:** `648207941cc647968170540aa5780d00`
- **Type:** rule
- **Confidence:** 0.85
- **Created:** 2026-03-18T02:01:39.370323Z
- **Project:** Glo
- **Locked:** YES
- **Keywords:** emotion emotional context feature creep burned causal signal

Record not just what Tom decided, but how he felt about it. The emotion carries information: 'someone who's been burned by feature creep' is not a neutral fact—it's a causal signal for why he'll resist certain paths. Emotion_label and emotion scores are load-bearing, not decorative.

---

#### stageLearning bug: bestOverlap undefined (should be bestSimilarity)

- **ID:** `306e3574310e43c68b0df15c8a1fcc09`
- **Type:** rule
- **Confidence:** 0.85
- **Created:** 2026-03-18T02:01:39.416464Z
- **Project:** Glo
- **Locked:** YES
- **Keywords:** stageLearning, bestOverlap, bestSimilarity, undefined variable, revisited

Line 4858 in brain.js returns bestOverlap in revisited node tracking, but variable is undefined. Should use bestSimilarity from the dedup matching logic above. Doesn't break core logic but corrupts return value.

---

#### Hebbian edge formation requires context() co-access, not individual recall() calls

- **ID:** `8725ad03cc5f4765816384861a1ab511`
- **Type:** rule
- **Confidence:** 0.85
- **Created:** 2026-03-18T02:01:39.436635Z
- **Project:** Glo
- **Locked:** YES
- **Keywords:** Hebbian, consolidate, co-access, context, recall, session_id, edge formation

Edges form via consolidate() when nodes are co-accessed within same session_id. Each recall() generates unique sessionId, so nodes never co-accessed in same session. Real usage: context() called once per user message creates single sessionId for all accessed nodes — that's when edges form. Individual recalls won't create edges.

---

#### Avoid compounding changes - prefer explicit/declarative approaches

- **ID:** `14fb708032a04638a8a6efc3b83d7f53`
- **Type:** rule
- **Confidence:** 0.85
- **Created:** 2026-03-18T02:01:39.460357Z
- **Project:** Glo
- **Locked:** YES
- **Keywords:** compounding changes explicit declarative build packaging zip

When designing build, packaging, config: choose explicit approaches over layering patches and exclusions. Example from today: explicit include list (good) vs growing zip -x flags (bad). Prevents accidental side effects (like node_modules slip) and reduces maintenance debt. Cleaner, scannable, safer.

---

#### Hook latency: 170ms startup (fixed) + 520ms connections (variable, batched away) + 97ms suggest work

- **ID:** `eb7255ed2bb34b939a33ca3b04bdc7b6`
- **Type:** rule
- **Confidence:** 0.85
- **Created:** 2026-03-18T02:01:39.532709Z
- **Project:** Glo
- **Locked:** YES
- **Keywords:** latency breakdown overhead startup bash Python TCP connection handshake suggest 97ms unavoidable fixed variable

Pre-edit hook overhead breakdown: ~170ms fixed bash/Python startup per call (unavoidable), ~520ms variable connection overhead per call (8 TCP handshakes + interpreter loads — eliminated by batching into single call), ~97ms actual server work in suggest endpoint. Batching removes the variable cost while keeping fixed overhead. Helps prioritize future optimizations — startup overhead is only addressable via faster interpreter or different hook mechanism.

---

#### Refactoring: read breadcrumbs, zero dead code, verify full alignment

- **ID:** `5b78eb77c80f4d7d83d2aa00775068b2`
- **Type:** rule
- **Confidence:** 0.95
- **Created:** 2026-03-18T02:01:39.638066Z
- **Project:** Glo
- **Locked:** YES
- **Keywords:** refactoring, breadcrumbs, dead code, alignment, verification, design

Before refactoring: read all comments and previous decisions left by prior self. Don't leave dead code from old versions. Triple-check every change aligns with entire system design.

---

#### FastEmbed: synchronous API—remove all await calls

- **ID:** `5b415202ce3a4019a4bd1a32352f2d72`
- **Type:** rule
- **Confidence:** 0.85
- **Created:** 2026-03-18T02:01:39.661076Z
- **Project:** Glo
- **Locked:** YES
- **Keywords:** FastEmbed, synchronous, async, await, embed(), embed_batch(), correction

FastEmbed.embed() and embed_batch() are synchronous functions. Remove `await` keywords when calling embedder methods in brain.py. Found during implementation testing.

---

#### Python f-string gotcha: backslashes in {} blocks cause SyntaxError

- **ID:** `323d1ff64e27457993827985236967a4`
- **Type:** rule
- **Confidence:** 0.85
- **Created:** 2026-03-18T02:01:39.684753Z
- **Project:** Glo
- **Locked:** YES
- **Keywords:** f-string, backslash, syntax error, Python gotcha

Hit during hook script development. F-strings cannot contain backslash escapes inside curly braces. Solution: extract escapes before f-string or move complex expressions outside the string. Affects boot-brain.sh embedded Python.

---

#### Lambda argument bug in brain.py TEMPORAL_PATTERNS: some lambdas called with match group but take 0 args

- **ID:** `58b0072a4b7b4dff992285d3ae85ce43`
- **Type:** rule
- **Confidence:** 0.85
- **Created:** 2026-03-18T02:01:39.714088Z
- **Project:** Glo
- **Locked:** YES
- **Keywords:** brain.py, TEMPORAL_PATTERNS, lambda, TypeError, match group

Fixed in brain.py: some pattern matching lambdas expected no arguments but were being called with match group. Caused TypeError during temporal pattern extraction.

---

#### Correction: Use current SKILL.md encoding rules as ground truth, not historical remember() calls

- **ID:** `f738144a6fee43f3b1a1e0a7a6e789cc`
- **Type:** rule
- **Confidence:** 0.85
- **Created:** 2026-03-18T02:01:40.012868Z
- **Project:** Glo
- **Locked:** YES
- **Keywords:** SKILL.md, encoding rules, ground truth, encoding validation

When validating the brain's encoding decisions, follow the CURRENT SKILL.md encoding instructions (not past behavior). The brain's strength is determined by what gets encoded per current rules, not replay of old decisions. This ensures the simulation tests the evolved encoding logic.

---

#### Tom: designer's eye despite engineering background — sees products as users would

- **ID:** `4c116c7f0fbb4eaab8eceae85a808ba9`
- **Type:** rule
- **Confidence:** 0.85
- **Created:** 2026-03-18T02:59:35.401276Z
- **Project:** None
- **Locked:** YES
- **Keywords:** Tom design visual UX user experience feedback aesthetics engineer dual perspective

Feedback is visual and experiential: 'glow icons should glow brighter', 'thumbnails should be first frame', 'that looks ugly.' He sees the product as a user, not just an architect. Combine with his engineering depth — he evaluates both the experience AND the architecture.

---

### CONCEPT (36 nodes)

#### WDIV Local 4 / Graham Media — Detroit CTV news. Demo use case #1.

- **ID:** `con_trjp0r0j`
- **Type:** concept
- **Confidence:** 0.7
- **Created:** 2026-03-15T21:38:53.382Z
- **Project:** Glo
- **Keywords:** o_graham WDIV Local 4 / Graham Media detroit ctv news. demo use case #1.

Detroit CTV news. Demo use case #1.

---

#### The Huddle Sports Bar — Fictional bar for DOOH demo. Use case #3.

- **ID:** `con_gusamr72`
- **Type:** concept
- **Confidence:** 0.7
- **Created:** 2026-03-15T21:38:53.388Z
- **Project:** Glo
- **Keywords:** o_huddle The Huddle Sports Bar fictional bar for dooh demo. use case #3.

Fictional bar for DOOH demo. Use case #3.

---

#### [o_antifraud] Properties

- **ID:** `con_3mxm4f94`
- **Type:** concept
- **Confidence:** 0.7
- **Created:** 2026-03-15T21:38:53.434Z
- **Project:** Glo
- **Keywords:** o_antifraud properties primary_gate concern additional_options status

primary_gate: Payment — must spend real money to create a Glo
concern: Fake Google email logins and bots on mobile
additional_options: Phone verify, device fingerprint, captcha, rate limiting — all secondary to payment
status: Payment gate decided as primary approach

---

#### [o_brightness] Properties

- **ID:** `con_nk7rwlmh`
- **Type:** concept
- **Confidence:** 0.7
- **Created:** 2026-03-15T21:38:53.451Z
- **Project:** Glo
- **Keywords:** o_brightness properties tiers custom recurring margin publisher_floor

tiers: GLO Well ($30), GLO Bright ($50), GLO Shine ($100)
custom: Custom amount option available
recurring: $X/day recurring cancel-anytime option
margin: 40% Glo margin default, adjustable per publisher/media
publisher_floor: Publisher sets floor rate cards

---

#### [o_myglos] Properties

- **ID:** `con_oeevnjvn`
- **Type:** concept
- **Confidence:** 0.7
- **Created:** 2026-03-15T21:38:53.754Z
- **Project:** Glo
- **Keywords:** o_myglos properties card_elements filters wallet demo_status

card_elements: Thumbnail, status badge, publisher name, progress bar (% budget spent), key vanity metric
filters: All, Active, Pending, Completed, Rejected, Draft
wallet: Credits balance visible on dashboard
demo_status: Demo built with 6 mock Glos across all states

---

#### [o_lifecycle] Properties

- **ID:** `con_en97onit`
- **Type:** concept
- **Confidence:** 0.7
- **Created:** 2026-03-15T22:59:23.807Z
- **Project:** Glo
- **Keywords:** properties states branches re_light

states: Draft, Pending Review, Active, Completed
branches: Rejected (refund+duplicate), Paused (credits back to wallet)
re_light: Available from any terminal state — same creative, new budget, new review cycle

---

#### [o_glonumbers] Properties

- **ID:** `con_4oqp2i8d`
- **Type:** concept
- **Confidence:** 0.85
- **Created:** 2026-03-15T22:59:44.622Z
- **Project:** Glo
- **Keywords:** properties metrics chart milestones sample_page demo_status

metrics: Total views, clicks, QR scans — big vanity numbers
chart: Views per day graph (SVG line chart)
milestones: Achievement cards (e.g. 1K+ views) — shareable
sample_page: Link to see Glo live on publisher site (online only, EX.CO feature)
demo_status: Demo built

---

#### [o_moderation] Properties

- **ID:** `con_ug2u35ti`
- **Type:** concept
- **Confidence:** 0.7
- **Created:** 2026-03-15T22:59:44.622Z
- **Project:** Glo
- **Keywords:** properties layers initial_team future_team ui_theme demo_status

layers: Layer 1: AI pre-screen (auto). Layer 2: Human final action (approve/reject/escalate).
initial_team: GLO/EX.CO ops team
future_team: Publisher-level access with publisher-specific rules
ui_theme: Dark theme, sidebar queue, detail panel
demo_status: Fully defined, demo built

---

#### Vibe.co research done. Patterns: single URL→auto-gen TV ad <30s, dual source (we

- **ID:** `con_iwbfmq14`
- **Type:** concept
- **Confidence:** 0.85
- **Created:** 2026-03-15T22:59:44.622Z
- **Project:** Glo
- **Keywords:** stm s4 glo research vibe.co research done. patterns: single url→auto-gen tv ad <30s, dual

Vibe.co research done. Patterns: single URL→auto-gen TV ad <30s, dual source (website/Google Maps), unified creative editor (upload+AI same canvas not separate paths), modular toggle sections, pre-filled defaults, Suggest helpers for targeting.

---

#### Transcoding: Cloudinary recommended. AI smart cropping across aspect ratios. ~$3

- **ID:** `con_6iueqb0z`
- **Type:** concept
- **Confidence:** 0.85
- **Created:** 2026-03-15T22:59:44.622Z
- **Project:** Glo
- **Keywords:** ltm l15 glo research transcoding: cloudinary recommended. ai smart cropping across aspect ratios. ~$300/mo

Transcoding: Cloudinary recommended. AI smart cropping across aspect ratios. ~$300/mo early scale.

---

#### CampaignParamsResolver

- **ID:** `con_i2zos5t8`
- **Type:** concept
- **Confidence:** 0.595
- **Created:** 2026-03-15T23:04:27.998Z
- **Project:** Glo
- **Keywords:** campaign params resolver optimization gam buyside agent

Isolated component that takes a Glo and returns GAM-ready params (dayparts, freq cap, pacing, views per session). V1: config-driven defaults per publisher type. V2+: AI buyside agent with dynamic optimization. TODO: full buyside agent, advertiser advanced settings.

---

#### Competitor creative flows: 13 platforms analyzed, avg 4-5 steps. Waymark closest

- **ID:** `con_u04edfxj`
- **Type:** concept
- **Confidence:** 0.85
- **Created:** 2026-03-15T23:05:06.884Z
- **Project:** Glo
- **Keywords:** ltm l13 glo research competitor creative flows: 13 platforms analyzed, avg 4-5 steps. waymark

Competitor creative flows: 13 platforms analyzed, avg 4-5 steps. Waymark closest to Glo needs. Vibe.co best onboarding UX (URL→auto-gen). Key insight: best pattern is unified upload+create on same screen.

---

#### Demo publishers: WDIV CTV ($22 CPM), Adweek Online ($18), Huddle DOOH ($8)

- **ID:** `con_l4vb6dd0`
- **Type:** concept
- **Confidence:** 0.85
- **Created:** 2026-03-15T23:05:06.884Z
- **Project:** Glo
- **Keywords:** publishers demo wdiv adweek huddle cpm ctv dooh online media

Three demo publishers spanning all media types. WDIV Local 4 Detroit CTV 500K homes. Adweek 2M media professionals. The Huddle Sports Bar DOOH. CPMs and reach units differ by type. These are demo data — real publishers would come from EX.CO inventory.

---

#### App state model: screen-based routing via Context API

- **ID:** `con_aofbtzix`
- **Type:** concept
- **Confidence:** 0.85
- **Created:** 2026-03-15T23:05:06.884Z
- **Project:** Glo
- **Keywords:** state model context api routing screens localstorage app architecture

State shape: screen (routing), pubKey (publisher), bizName/bizUrl/bizGoal (onboarding), creative fields (upload/AI/variations), budget (tier/spendMode/dailyAmount), schedule, auth, glos array (user campaigns). localStorage persistence. No react-router — state-driven navigation.

---

#### Adweek — Media industry news site. Demo use case #2 — online video.

- **ID:** `con_rro1j1fs`
- **Type:** concept
- **Confidence:** 0.85
- **Created:** 2026-03-15T23:05:06.885Z
- **Project:** Glo
- **Keywords:** Adweek media industry news site. demo use case #2 — online

Media industry news site. Demo use case #2 — online video.

---

#### [o_emailsys] Properties

- **ID:** `con_n6b3blwr`
- **Type:** concept
- **Confidence:** 0.7
- **Created:** 2026-03-15T23:05:06.885Z
- **Project:** Glo
- **Keywords:** properties trigger_types content sample_page status

trigger_types: Activation progress, performance milestones, re-engagement, rejection notices
content: Real screenshots from actual publisher site, view counts, milestone celebrations
sample_page: Can include real link to see Glo live on site (online only, via EX.CO)
status: Concept defined

---

#### Clerk — Auth/user management. All SSOs + Stripe integration. Recommended for Glo.

- **ID:** `con_s37hwva1`
- **Type:** concept
- **Confidence:** 0.85
- **Created:** 2026-03-15T23:27:47.999Z
- **Project:** Glo
- **Locked:** YES
- **Keywords:** clerk auth/user management. all ssos + stripe integration. recommended for glo. authuser management integration glo

Auth/user management. All SSOs + Stripe integration. Recommended for Glo.

---

#### AI video gen build-vs-buy: Creatify best for MVP $99/mo API URL→video. Waymark f

- **ID:** `con_q1oslrkk`
- **Type:** concept
- **Confidence:** 0.85
- **Created:** 2026-03-15T23:27:47.999Z
- **Project:** Glo
- **Keywords:** ltm l14 glo research ai video gen build-vs-buy: creatify best for mvp $99/mo api

AI video gen build-vs-buy: Creatify best for MVP $99/mo API URL→video. Waymark for production scale franchise/SMB. Building custom not worth it.

---

#### EX.CO: full end-to-end video platform for online publishers (CMS, ad server, pla

- **ID:** `con_q6pydb49`
- **Type:** concept
- **Confidence:** 0.85
- **Created:** 2026-03-15T23:42:36.155Z
- **Project:** Glo
- **Keywords:** ltm l21 exco fact ex.co: full end-to-end video platform for online publishers (cms, ad

EX.CO: full end-to-end video platform for online publishers (CMS, ad server, player). Smart ad server for DOOH+CTV. Tom is CEO. Glo built within EX.CO — conceptually supported by leadership, needs formal business case.

---

#### [o_credits] Properties

- **ID:** `con_keh88ufv`
- **Type:** concept
- **Confidence:** 0.85
- **Created:** 2026-03-15T23:42:36.156Z
- **Project:** Glo
- **Keywords:** properties exchange_rate wallet_infra blockchain

exchange_rate: 1 Glo Credit = 1 USD
wallet_infra: Stripe customer balance (avoids money transmitter licensing)
blockchain: Token/blockchain angle parked — idea stage

---

#### [o_exco] Properties

- **ID:** `con_9wgrg928`
- **Type:** concept
- **Confidence:** 0.85
- **Created:** 2026-03-15T23:42:36.156Z
- **Project:** Glo
- **Keywords:** properties products ceo publisher_scale sample_page

products: Video CMS, ad server, player (online publishers); Smart ad server (DOOH+CTV)
ceo: Tom
publisher_scale: 100-1000 publishers with 20-40% unfilled inventory
sample_page: Can enable sample page link on media site for users to see their Glo in context. Websites only, not CTV/DOOH.

---

#### [o_glo] Properties

- **ID:** `con_0qfhpayv`
- **Type:** concept
- **Confidence:** 0.85
- **Created:** 2026-03-15T23:42:36.156Z
- **Project:** Glo
- **Keywords:** properties stage media_types target_user margin naming tiers

stage: pre-product, idea stage
media_types: online, ctv, dooh
target_user: SMB, micro-biz, normal individuals — anyone with a phone, never bought media before
margin: 40% default, adjustable per publisher/media
naming: Not ads — Glos. Active/Paused/Completed/Pending Review. My Glos dashboard. Glo Credits. GLO Brightness tiers.
tiers: GLO Well($30), GLO Bright($50), GLO Shine($100) + custom + daily recurring

---

#### Creatify — AI video gen API. $99/mo. URL→video. Recommended for Glo MVP.

- **ID:** `con_rd9bmn59`
- **Type:** concept
- **Confidence:** 0.85
- **Created:** 2026-03-16T17:33:04.721Z
- **Project:** Glo
- **Keywords:** Glo, yield engine, programmatic monetization, Fox Corp, sales pitch

Glo product/service: yield engine for programmatic monetization. Target: Fox Corp. Approach uses smaller divisions/P&Ls as entry points. User seeks research on Fox organizational structure.

---

#### Waymark — AI video gen, franchise/SMB focus. Recommended for Glo production scale.

- **ID:** `con_hlytgmqx`
- **Type:** concept
- **Confidence:** 0.85
- **Created:** 2026-03-16T17:33:04.721Z
- **Project:** Glo
- **Keywords:** Waymark ai video gen, franchise/smb focus. recommended for glo production scale.

AI video gen, franchise/SMB focus. Recommended for Glo production scale.

---

#### Cloudinary — Video transcoding, AI smart cropping across aspect ratios. ~$300/mo early scale.

- **ID:** `con_qapunc9e`
- **Type:** concept
- **Confidence:** 0.85
- **Created:** 2026-03-16T17:33:04.721Z
- **Project:** Glo
- **Keywords:** AI Generate, aspect ratio, backend automation, no user selection

Within AI Generate tab, remove Aspect Ratio selector completely. All aspect ratio variations (1:1, 16:9, 9:16, etc.) are generated automatically backend. User doesn't choose.

---

#### GLO Brightness — Pricing tiers: Well($30) Bright($50) Shine($100). Publisher sets floor rate cards, 40% Glo margin.

- **ID:** `con_5l0e4p9v`
- **Type:** concept
- **Confidence:** 0.85
- **Created:** 2026-03-16T17:33:04.721Z
- **Project:** Glo
- **Keywords:** pricing, variable, publisher-specific, media-type, rate card, custom tier

GLO Well/Bright/Shine are tier names, but actual USD cost varies. MLB higher floor CPM than local blog. CTV vs online vs DOOH have different rate structures. Always offer Custom tier for users wanting flexibility.

---

#### Tmemory — Persistent brain engine for Claude (v4.2)

- **ID:** `con_indnhlfx`
- **Type:** concept
- **Confidence:** 0.85
- **Created:** 2026-03-16T17:33:04.721Z
- **Project:** Glo
- **Keywords:** tmemory brain memory plugin persistent graph sqlite hebbian ebbinghaus emotion dreaming self-improvement

Graph-based persistent memory. SQLite via sql.js. Hebbian learning, Ebbinghaus decay, spreading activation, emotional coding, dreaming. v4: self-improvement (instrumented recall, miss detection, evaluation). v4.2: uncapped spread activation, hub dampening on all nodes, query normalization, dream-time keyword enrichment. Design principle: cue system not search engine — surface context so LLM reasoning takes over.

---

#### Glo Numbers — Analytics detail per Glo. Views/day graph, clicks, QR scans, budget progress, creative preview.

- **ID:** `con_7far6rky`
- **Type:** concept
- **Confidence:** 0.85
- **Created:** 2026-03-16T17:33:04.721Z
- **Project:** Glo
- **Keywords:** creative intelligence, LLM design, brief generation, user intent, infinite archetypes

Design spec at /glo/Glo Creative Intelligence.md. Formalizes shift from fixed archetype mapping to LLM-based creative brief generation. Contains architecture for understanding user intent and generating contextual creative direction for infinite use cases.

---

#### Glo Lifecycle — State machine: Draft→Pending Review→Active→Completed. Branches: Rejected(refund+duplicate), Paused(w

- **ID:** `con_b4g2ux5s`
- **Type:** concept
- **Confidence:** 0.85
- **Created:** 2026-03-16T17:33:04.721Z
- **Project:** Glo
- **Keywords:** Glo Lifecycle state machine: draft→pending review→active→completed. branches: rejected(refund+duplicate), paused(wallet refund). re-light from

State machine: Draft→Pending Review→Active→Completed. Branches: Rejected(refund+duplicate), Paused(wallet refund). Re-light from terminal states.

---

#### Moderation System — Two-layer: AI pre-screen (risk score, flags, IAB brand safety) + human final action. GLO/EX.CO ops i

- **ID:** `con_hczvz55p`
- **Type:** concept
- **Confidence:** 0.85
- **Created:** 2026-03-16T17:33:04.721Z
- **Project:** Glo
- **Keywords:** Moderation System two-layer: ai pre-screen (risk score, flags, iab brand safety) +

Two-layer: AI pre-screen (risk score, flags, IAB brand safety) + human final action. GLO/EX.CO ops initially, publishers later. Scale-friendly queue.

---

#### Email/Notification System — Activation progress, performance updates with real screenshots, milestones, re-engagement, rejection

- **ID:** `con_5ej15u5g`
- **Type:** concept
- **Confidence:** 0.85
- **Created:** 2026-03-16T17:33:04.721Z
- **Project:** Glo
- **Keywords:** Email/Notification System activation progress, performance updates with real screenshots, milestones, re-engagement, rejection

Activation progress, performance updates with real screenshots, milestones, re-engagement, rejection notices.

---

#### Mobile Capture & Anti-Fraud — Prevent fake logins/bots. Payment gate as primary gatekeeper. Balance friction vs impulse UX.

- **ID:** `con_7sp3ziq7`
- **Type:** concept
- **Confidence:** 0.85
- **Created:** 2026-03-16T17:33:04.721Z
- **Project:** Glo
- **Keywords:** Mobile Capture & Anti-Fraud prevent fake logins/bots. payment gate as primary gatekeeper. balance friction

Prevent fake logins/bots. Payment gate as primary gatekeeper. Balance friction vs impulse UX.

---

#### My Glos Dashboard — User home. Cards per Glo: thumbnail, status, progress (% budget spent), vanity metrics. Filters, wal

- **ID:** `con_ws6v4n4y`
- **Type:** concept
- **Confidence:** 0.85
- **Created:** 2026-03-16T17:33:56.366Z
- **Project:** Glo
- **Keywords:** My Glos Dashboard user home. cards per glo: thumbnail, status, progress (% budget

User home. Cards per Glo: thumbnail, status, progress (% budget spent), vanity metrics. Filters, wallet balance, actions.

---

#### Glo Credits — 1:1 USD. Wallet via Stripe customer balance. Blockchain/token angle parked.

- **ID:** `con_esyy9xi1`
- **Type:** concept
- **Confidence:** 0.85
- **Created:** 2026-03-16T17:33:56.375Z
- **Project:** Glo
- **Keywords:** glo credits 1:1 usd. wallet via stripe customer balance. blockchain/token angle parked. usd balance blockchaintoken parked

1:1 USD. Wallet via Stripe customer balance. Blockchain/token angle parked.

---

#### EX.CO publisher CMS supports custom ad slots

- **ID:** `con_1c9nsnhl`
- **Type:** concept
- **Confidence:** 0.7
- **Created:** 2026-03-16T20:59:20.495Z
- **Project:** None
- **Keywords:** staged auto_extracted exco publisher cms ad-slots ex.co exco publisher cms supports custom slots allows publishers define slot positions glo advertisers target

EX.CO CMS allows publishers to define custom ad slot positions, which Glo advertisers can target

---

#### Redirect learning: mutual improvement through principle extraction

- **ID:** `c576520779434e778dbd43cdcd0d0068`
- **Type:** concept
- **Confidence:** 0.85
- **Created:** 2026-03-18T02:01:38.511987Z
- **Project:** Glo
- **Locked:** YES
- **Keywords:** redirect learning, principles, feedback, mutual improvement, philosophy

Tom's term for the feedback loop where extracting and formalizing principles helps both Claude and Tom improve. Not just code feedback, but meta-feedback on thinking patterns and system design philosophy. Principles extracted from one session become locked nodes that shape future sessions.

---

### INTUITION (27 nodes)

#### Dream: [ltm:l2] Flywheel: unfilled inventory→ho ↔ [stm:s18] Mobile capture: Tom wants to c

- **ID:** `int_3uvfsyln`
- **Type:** intuition
- **Confidence:** 0.4
- **Created:** 2026-03-15T22:59:22.512Z
- **Project:** Glo
- **Keywords:** dream intuition association ltm flywheel unfilled inventory house ads recruit advertisers new glos fill invento stm s18 mobile capture tom wants capture mobile users worried fake google

Association: "[period:2026-03:p11] Creative step deep dive. 3 paths defined. Contextual intent engine concept." → "[ltm:l2] Flywheel: unfilled inventory→house ads recruit advertisers→new Glos fill invento" | "[stm:s8] Tom defining My Glos dashboard: each Glo shows status (Active/Completed/Pending " → "[stm:s18] Mobile capture: Tom wants to capture mobile users but worried about fake Google "

---

#### Dream: [todo:t4] Build formal business case for ↔ [period:2026-03:p2] Biggest growing priv

- **ID:** `int_z4usuzzd`
- **Type:** intuition
- **Confidence:** 0.4
- **Created:** 2026-03-15T23:01:19.707Z
- **Project:** Glo
- **Keywords:** dream intuition association todo build formal business case board period 2026 biggest growing private dsps trade desk stackadapt criteo adform magnite

Association: "[stm:s8] Tom defining My Glos dashboard: each Glo shows status (Active/Completed/Pending " → "[todo:t4] Build formal business case for EX.CO board" | "Magnite — US adtech. Closest comparable to Geniee (JP). Early research." → "[period:2026-03:p2] Biggest growing private DSPs: Trade Desk, StackAdapt, Criteo, Adform, Magnite, A"

---

#### Dream: [o_credits] Pause refunds credits to wal ↔ [stm:s7] Formal business case for EX.CO 

- **ID:** `int_oy2sww3w`
- **Type:** intuition
- **Confidence:** 0.4
- **Created:** 2026-03-15T23:42:36.155Z
- **Project:** Glo
- **Keywords:** dream intuition association credits pause refunds credits wallet dynamic pricing means hold credits old stm formal business case board parked needed

Association: "[stm:s36] Overnight batch 1: Created Mobile UX & Anti-Fraud research doc (/glo/Research/)." → "[o_credits] Pause refunds credits to wallet. Dynamic pricing means can't hold credits at old" | "[stm:s18] Mobile capture: Tom wants to capture mobile users but worried about fake Google " → "[stm:s7] Formal business case for EX.CO board — parked but needed."

---

#### Dream: Moderation: AI pre-screen (risk score, f ↔ [o_myglos] Multiple simultaneous Glos pe

- **ID:** `int_65c3ioal`
- **Type:** intuition
- **Confidence:** 0.4
- **Created:** 2026-03-17T04:14:29.754Z
- **Project:** None
- **Keywords:** dream intuition association moderation pre-screen prescreen pre screen risk score flags iab brand safety biz legitimacy myglos multiple simultaneous glos per user yes. yes

Association: "New components emerging: My Glos Dashboard, Glo Lifecycle state machine, expande" → "Moderation: AI pre-screen (risk score, flags, IAB brand safety, biz legitimacy, " | "[o_myglos] Properties" → "[o_myglos] Multiple simultaneous Glos per user — yes."

---

#### Dream: [o_glo] Flywheel: unfilled inventory→hou ↔ SSO→payment linking: Google→GPay, Apple→

- **ID:** `int_2kdyprto`
- **Type:** intuition
- **Confidence:** 0.4
- **Created:** 2026-03-17T04:14:29.761Z
- **Project:** None
- **Keywords:** dream intuition association glo flywheel unfilled inventory house ads recruit advertisers fill remain sso payment linking google gpay apple applepay amazon amazonpay facebook met

Association: "Component: House Ad Marketing" → "[o_glo] Flywheel: unfilled inventory→house ads→recruit advertisers→fill inventory→remain" | "Component: Product Naming" → "SSO→payment linking: Google→GPay, Apple→ApplePay, Amazon→AmazonPay, Facebook→Met"

---

#### Dream: Web app (PWA) not native iOS — avoids Ap ↔ Reject flow: predefined categories + opt

- **ID:** `int_wh9xau4r`
- **Type:** intuition
- **Confidence:** 0.4
- **Created:** 2026-03-17T04:25:27.664Z
- **Project:** None
- **Keywords:** dream intuition association web app pwa native ios avoids apple in-app inapp purchase cut. cut normal reject flow predefined categories optional moderator note. note user gets refund

Association: "Vibe.co — Streaming/CTV ad platform. UX reference for Glo: URL→auto-gen, unified creative editor." → "Web app (PWA) not native iOS — avoids Apple's 30% in-app purchase cut. Normal St" | "[todo:t5] Explore blockchain/token angle for Glo Credits" → "Reject flow: predefined categories + optional moderator note. User gets refund +"

---

#### Dream: Graph bridging > embeddings for emergent ↔ Glo/EX.CO boundary

- **ID:** `int_epophnk2`
- **Type:** intuition
- **Confidence:** 0.4
- **Created:** 2026-03-17T04:25:27.672Z
- **Project:** None
- **Keywords:** dream intuition association graph bridging embeddings emergent discovery toms architectural insight glo/ex.co gloexco boundary

Association: "Store-time bridging requires pre-existing neighborhood — cold-start problem" → "Graph bridging > embeddings for emergent discovery — Toms architectural insight" | "App state model: screen-based routing via Context API" → "Glo/EX.CO boundary"

---

#### Dream: WDIV Local 4 / Graham Media — Detroit CT ↔ Glo Numbers — Analytics detail per Glo. 

- **ID:** `int_xf26hbu8`
- **Type:** intuition
- **Confidence:** 0.4
- **Created:** 2026-03-17T04:25:34.311Z
- **Project:** None
- **Keywords:** dream intuition association wdiv local graham media detroit ctv news. news demo use case glo numbers analytics detail per glo. views/day viewsday graph clicks scans budget progress creative preview. preview

Association: "Dream: [o_credits] Pause refunds credits to wal ↔ [stm:s7] Formal business case for EX.CO " → "WDIV Local 4 / Graham Media — Detroit CTV news. Demo use case #1." | "My Glos Dashboard — User home. Cards per Glo: thumbnail, status, progress (% budget spent), vanity metrics. Filters, wal" → "Glo Numbers — Analytics detail per Glo. Views/day graph, clicks, QR scans, budget progress, creative preview."

---

#### Dream: Creative strategy: AI video gen is NOT t ↔ Reject flow: predefined categories + opt

- **ID:** `int_qagfib4l`
- **Type:** intuition
- **Confidence:** 0.4
- **Created:** 2026-03-17T04:25:34.316Z
- **Project:** None
- **Keywords:** dream intuition association creative strategy video gen moat. moat buy/integrate. buyintegrate creatify api reject flow predefined categories optional moderator note. note user gets refund

Association: "Transcoding: Cloudinary recommended. AI smart cropping across aspect ratios. ~$3" → "Creative strategy: AI video gen is NOT the moat. Buy/integrate. Creatify API $99" | "My Glos Dashboard — User home. Cards per Glo: thumbnail, status, progress (% budget spent), vanity metrics. Filters, wal" → "Reject flow: predefined categories + optional moderator note. User gets refund +"

---

#### Dream: [o_myglos] Drafts saved to dashboard, ac ↔ AI moderates first — adds comments, auto

- **ID:** `int_94yf7qj0`
- **Type:** intuition
- **Confidence:** 0.4
- **Created:** 2026-03-17T04:25:34.325Z
- **Project:** None
- **Keywords:** dream intuition association myglos drafts saved dashboard accessible anytime. anytime moderates first adds comments auto-status autostatus auto status risk score moderation board. board

Association: "[todo:t4] Build formal business case for EX.CO board" → "[o_myglos] Drafts saved to dashboard, accessible anytime." | "[todo:t3] Update P&L with EX.CO-specific assumptions (unfilled inventory economics, publis" → "AI moderates first — adds comments, auto-status, risk score to moderation board."

---

#### Dream: Component: My Glos Dashboard ↔ SSO→payment linking: Google→GPay, Apple→

- **ID:** `int_3lzr9ahz`
- **Type:** intuition
- **Confidence:** 0.4
- **Created:** 2026-03-17T04:25:34.470Z
- **Project:** None
- **Keywords:** dream intuition association component glos dashboard sso payment linking google gpay apple applepay amazon amazonpay facebook met

Association: "[stm:s42] Component MVP analysis: 7 architecture docs in /glo/Architecture/. Analyzed all " → "Component: My Glos Dashboard" | "Glo Lifecycle — State machine: Draft→Pending Review→Active→Completed. Branches: Rejected(refund+duplicate), Paused(w" → "SSO→payment linking: Google→GPay, Apple→ApplePay, Amazon→AmazonPay, Facebook→Met"

---

#### Dream: Glo Credits — 1:1 USD. Wallet via Stripe ↔ Moderation: AI-first, two layers — platf

- **ID:** `int_z4u8x2ya`
- **Type:** intuition
- **Confidence:** 0.4
- **Created:** 2026-03-17T04:25:34.473Z
- **Project:** None
- **Keywords:** dream intuition association glo credits usd. usd wallet via stripe customer balance. balance blockchain/token blockchaintoken angle parked. parked moderation ai-first aifirst first two layers platform safety publisher-specific publisherspecific publisher specific configur

Association: "WDIV Local 4 / Graham Media — Detroit CTV news. Demo use case #1." → "Glo Credits — 1:1 USD. Wallet via Stripe customer balance. Blockchain/token angle parked." | "Mobile capture: Tom wants to capture mobile users but worried about fake Google " → "Moderation: AI-first, two layers — platform safety + publisher-specific configur"

---

#### Dream: Dream: [ltm:l2] Flywheel: unfilled inven ↔ AI video gen is NOT the moat. Buy/integr

- **ID:** `int_bfzyw7nj`
- **Type:** intuition
- **Confidence:** 0.4
- **Created:** 2026-03-17T04:25:34.612Z
- **Project:** None
- **Keywords:** dream intuition association dream ltm flywheel unfilled inventory stm s18 mobile capture tom wants video gen moat. moat buy/integrate buyintegrate creatify mvp 99/mo 99mo waymark produc

Association: "Vibe.co — Streaming/CTV ad platform. UX reference for Glo: URL→auto-gen, unified creative editor." → "Dream: [ltm:l2] Flywheel: unfilled inventory→ho ↔ [stm:s18] Mobile capture: Tom wants to c" | "[o_antifraud] Properties" → "AI video gen is NOT the moat. Buy/integrate: Creatify MVP $99/mo, Waymark produc"

---

#### Dream: [stm:s49] LOCKED: Aspect ratio selection ↔ Anti-fraud concern: fake Google logins a

- **ID:** `int_smy5acx6`
- **Type:** intuition
- **Confidence:** 0.4
- **Created:** 2026-03-17T04:25:34.614Z
- **Project:** None
- **Keywords:** dream intuition association stm s49 locked aspect ratio selection removed user-facing userfacing user facing creative screen. screen asp anti-fraud antifraud anti fraud concern fake google logins bots mobile. mobile payment strongest

Association: "Moderation System — Two-layer: AI pre-screen (risk score, flags, IAB brand safety) + human final action. GLO/EX.CO ops i" → "[stm:s49] LOCKED: Aspect ratio selection removed from user-facing Creative screen. All asp" | "[o_antifraud] Properties" → "Anti-fraud concern: fake Google logins and bots on mobile. Payment as strongest "

---

#### Dream: [o_myglos] Multiple simultaneous Glos pe ↔ Mobile Capture & Anti-Fraud — Prevent fa

- **ID:** `int_ea7et2lo`
- **Type:** intuition
- **Confidence:** 0.4
- **Created:** 2026-03-17T04:25:34.617Z
- **Project:** None
- **Keywords:** dream intuition association myglos multiple simultaneous glos per user yes. yes mobile capture anti-fraud antifraud anti fraud prevent fake logins/bots. loginsbots payment gate primary gatekeeper. gatekeeper balance friction impulse ux.

Association: "Fixed 6 gaps in demo: (1) Auto-trigger AI gen via useEffect when URL provided, (" → "[o_myglos] Multiple simultaneous Glos per user — yes." | "My Glos Dashboard — User home. Cards per Glo: thumbnail, status, progress (% budget spent), vanity metrics. Filters, wal" → "Mobile Capture & Anti-Fraud — Prevent fake logins/bots. Payment gate as primary gatekeeper. Balance friction vs impulse UX."

---

#### Dream: Sample page link from EX.CO can be inclu ↔ Post-compaction session continuation wor

- **ID:** `int_9355ob5n`
- **Type:** intuition
- **Confidence:** 0.4
- **Created:** 2026-03-17T04:38:21.379Z
- **Project:** None
- **Keywords:** dream intuition association sample page link ex.co exco included emails real scre post-compaction postcompaction post compaction session continuation works summary context files brain recall

Association: "EX.CO can enable sample page link on media site so user can see their Glo in con" → "Sample page link from EX.CO can be included in emails — real link, not just scre" | "Post-compaction session continuation works — summary + context files + brain recall" → "Post-compaction session continuation works — summary + context files + brain recall"

---

#### Dream: Web app (PWA) not native iOS — avoids Ap ↔ [o_glo] 3 creative paths: upload, AI fro

- **ID:** `int_jn22iqh3`
- **Type:** intuition
- **Confidence:** 0.4
- **Created:** 2026-03-17T04:38:21.397Z
- **Project:** None
- **Keywords:** dream intuition association web app pwa native ios avoids apple in-app inapp purchase cut. cut normal glo creative paths upload url/google urlgoogle maps social import coming soon

Association: "Fixed 6 gaps in demo: (1) Auto-trigger AI gen via useEffect when URL provided, (" → "Web app (PWA) not native iOS — avoids Apple's 30% in-app purchase cut. Normal St" | "[period:2026-03:p12] 13 competitor flows + AI video gen build-vs-buy. Waymark closest, Creatify MVP, " → "[o_glo] 3 creative paths: upload, AI from URL/Google Maps, social import (coming soon)"

---

#### Dream: Email/Notification System — Activation p ↔ Creative strategy: AI video gen is NOT t

- **ID:** `513a4e6232704122bb2aa731188c24c7`
- **Type:** intuition
- **Confidence:** 0.4
- **Created:** 2026-03-17T18:43:59.230315Z
- **Project:** None
- **Keywords:** dream intuition association strategy screenshots buy creatify video moat. rejection system progress notification email/notification performance buy/integrate. creative updates milestones buyintegrate real re-engagement emailnotification activation reengagement moat email api gen

Association: "[o_glonumbers] Properties" → "Email/Notification System — Activation progress, performance updates with real screenshots, milestones, re-engagement, rejection" | "[period:2026-03:p16] Revised demo v2 with all feedback. Vibe.co researched first." → "Creative strategy: AI video gen is NOT the moat. Buy/integrate. Creatify API $99"

---

#### Dream: Correction: Work approach — deep synthes ↔ Glo Creative Intelligence: LLM as Creati

- **ID:** `b6411c400cc44e509810b3e18105d5c6`
- **Type:** intuition
- **Confidence:** 0.4
- **Created:** 2026-03-18T02:01:40.069444Z
- **Project:** 
- **Keywords:** dream intuition association director archetypes additions creative correction fixed glo synthesis surface features work intelligence approach llm deep

Association: "nanobanana-adapter.js — NanoBanana video client" → "Correction: Work approach — deep synthesis before features, not surface additions" | "Dream: Dream connection: "Magnite — U.S. adtech ↔ Tom — Glo.io founder, tom@ex.co, Google " → "Glo Creative Intelligence: LLM as Creative Director, not fixed archetypes"

---

#### Dream: Credits balance shown wherever makes sen ↔ [o_brightness] Dynamic pricing: media pr

- **ID:** `da3aa832d97c4b009f682518eb2c52db`
- **Type:** intuition
- **Confidence:** 0.4
- **Created:** 2026-03-18T03:16:17.828435Z
- **Project:** None
- **Keywords:** dream intuition association shown sense brightness user daily makes scr dynamic change e.g prices o_brightness e.g. obrightness impacts pricing pauser balance event wherever dashboard media settings big pause/r credits mlb glos

Association: "Apple Pay fees concern raised. Key question: web app vs native iOS app determine" → "Credits balance shown wherever makes sense: My Glos dashboard, user settings scr" | "Cluster forming: "Glo Lifecycle — State machine: Draft→Pending Review→Active→Completed. Branches: Rejected(refund+dup..." → "[o_brightness] Dynamic pricing: media prices change daily (e.g. big MLB event). Impacts pause/r"

---

#### Dream: Magnite — US adtech. Closest comparable  ↔ [ctx:glo-platform] Glo.io — Self-Serve A

- **ID:** `662e221e1ce64266a50e73db1df786c2`
- **Type:** intuition
- **Confidence:** 0.4
- **Created:** 2026-03-18T03:16:27.545718Z
- **Project:** None
- **Keywords:** dream intuition association early closest platform ctx self-serve glo.io gloplatform research. selfserve adtech adtech. gloio research magnite advertising comparable geniee loio glo-platform lo.io

Association: "Glo Numbers — Analytics detail per Glo. Views/day graph, clicks, QR scans, budget progress, creative preview." → "Magnite — US adtech. Closest comparable to Geniee (JP). Early research." | "[o_glonumbers] Properties" → "[ctx:glo-platform] Glo.io — Self-Serve Advertising Platform"

---

#### Dream: EX.CO — Video platform and ad server ↔ Cluster forming: "Glo Lifecycle — State 

- **ID:** `ea9109d75dab4665a238a5430429c83c`
- **Type:** intuition
- **Confidence:** 0.4
- **Created:** 2026-03-18T03:16:27.629796Z
- **Project:** None
- **Keywords:** dream intuition association platform forming video cluster rejected glo ex.co dup... state server draft lifecycle completed pending machine exco branches review completed. dup active refund

Association: "CampaignParamsResolver" → "EX.CO — Video platform and ad server" | "Dream: Component: My Glos Dashboard ↔ SSO→payment linking: Google→GPay, Apple→" → "Cluster forming: "Glo Lifecycle — State machine: Draft→Pending Review→Active→Completed. Branches: Rejected(refund+dup..."

---

#### Dream: Dream: [o_credits] Pause refunds credits ↔ [stm:s35] Created GLO logo concepts: 4 d

- **ID:** `ef4cc1fee4014c90b5f0cf8ab96c1ed5`
- **Type:** intuition
- **Confidence:** 0.4
- **Created:** 2026-03-18T03:16:27.662358Z
- **Project:** None
- **Keywords:** dream intuition association concepts.html stm case credits directions pause glo ex.co dream brand o_credits refunds concepts business s35 wal ocredits exco /glo/brand/glo-logo-concepts.html. logo formal conceptshtml created globrandglologoconceptshtml

Association: "[todo:t4] Build formal business case for EX.CO board" → "Dream: [o_credits] Pause refunds credits to wal ↔ [stm:s7] Formal business case for EX.CO " | "My Glos Dashboard — User home. Cards per Glo: thumbnail, status, progress (% budget spent), vanity metrics. Filters, wal" → "[stm:s35] Created GLO logo concepts: 4 directions in /glo/Brand/glo-logo-concepts.html. (1"

---

#### Dream: Brain Evolution Roadmap — 3 prioritized  ↔ Open: Embedding strategy for semantic re

- **ID:** `55a71114d79049f39be23329f8b3c0e6`
- **Type:** intuition
- **Confidence:** 0.4
- **Created:** 2026-03-18T23:24:16.208989Z
- **Project:** None
- **Keywords:** dream intuition association evolution roadmap prioritized strategy recall options open architecture brain explore changes committing embedding semantic

Association: "Glo Numbers — Analytics detail per Glo. Views/day graph, clicks, QR scans, budget progress, creative preview." → "Brain Evolution Roadmap — 3 prioritized architecture changes" | "Root cause: hooks go silent when brain server dies — ensure-brain.sh returns approve silently" → "Open: Embedding strategy for semantic recall — explore options before committing"

---

#### Dream: [period:2026-03:p5] Fox Corp penetration ↔ Glo.io — Self-serve ad platform on EX.CO

- **ID:** `cb06127f4351437888ab227660541396`
- **Type:** intuition
- **Confidence:** 0.4
- **Created:** 2026-03-18T23:24:16.366142Z
- **Project:** None
- **Keywords:** dream intuition association ex.co stati selfserve glo.io weather onlinect server tmz media entry tubi buys nation lo.io self-serve loio exco fox platform inventory publisher online/ct period penetration corp server. gloio 2026-03 202603 via anyone

Association: "Root cause: hooks go silent when brain server dies — ensure-brain.sh returns approve silently" → "[period:2026-03:p5] Fox Corp penetration: entry via Tubi, Fox Weather, TMZ, Fox Nation, Fox TV Stati" | "[todo:t3] Update P&L with EX.CO-specific assumptions (unfilled inventory economics, publis" → "Glo.io — Self-serve ad platform on EX.CO ad server. Anyone buys media on EX.CO publisher inventory (online/CT"

---

#### Dream: Time-dilation decay: decay_active_rate a ↔ Tom: Plan-first approach for bigger ques

- **ID:** `f6f1bb36f0f146c4aff604b00db8b33b`
- **Type:** intuition
- **Confidence:** 0.4
- **Created:** 2026-03-18T23:24:16.477946Z
- **Project:** None
- **Keywords:** dream intuition association plan-first decay decayidlerate questions independent decayactiverate timedilation rate decay_active_rate approach planfirst active bigger decay_idle_rate tom time-dilation idle

Association: "Dream: [o_glo] Flywheel: unfilled inventory→hou ↔ SSO→payment linking: Google→GPay, Apple→" → "Time-dilation decay: decay_active_rate and decay_idle_rate independent" | "Bug: brain.py was missing 'import sys' — added for stderr logging" → "Tom: Plan-first approach for bigger questions"

---

#### Dream: Rule: ask for confirmation before manipu ↔ Glo Creative Intelligence: LLM as Creati

- **ID:** `8cd0a7b14fb448f0965d40aa25e2d673`
- **Type:** intuition
- **Confidence:** 0.4
- **Created:** 2026-03-21T19:56:05.470308Z
- **Project:** None
- **Keywords:** dream intuition association glo manipulating rule creative director test results ask fixed mapping data confirmation intelligence affect llm archetype

Association: "Never 2>/dev/null on brain hooks — silent failures are the worst failures" → "Rule: ask for confirmation before manipulating data to affect test results" | "Build official session handoff mechanism into brain" → "Glo Creative Intelligence: LLM as Creative Director, not fixed archetype mapping"

---

### CONTEXT (24 nodes)

#### Tom still needs to review the complete demo with all gap fixes. Designer screens

- **ID:** `con_eth06cc5`
- **Type:** context
- **Confidence:** 0.6
- **Created:** 2026-03-15T22:59:21.624Z
- **Project:** Glo
- **Keywords:** soft-cap, proportional extraction, narrative compression, 15-18K chars, gap-relative scaling

Log extraction should scale with gap size: longer gaps get more detailed narrative, older material gets increasingly compressed summaries. Soft cap at ~15-18K chars prevents context bloat. User preference: 'proportional, soft cap' — not fixed-size windows, not unlimited.

---

#### Users database flagged as future component — user profiles, history, risk scorin

- **ID:** `con_o4e176b9`
- **Type:** context
- **Confidence:** 0.6
- **Created:** 2026-03-15T22:59:21.643Z
- **Project:** Glo
- **Keywords:** stm s21 glo input users database flagged as future component — user profiles, history,

Users database flagged as future component — user profiles, history, risk scoring, preferences. Not discussing now.

---

#### NEW COMPONENTS FROM TOM: (1) Email/notification system — activation progress, ex

- **ID:** `con_ye6u6ecm`
- **Type:** context
- **Confidence:** 0.6
- **Created:** 2026-03-15T22:59:21.662Z
- **Project:** Glo
- **Keywords:** stm s17 glo input new components from tom: (1) email/notification system — activation progress,

NEW COMPONENTS FROM TOM: (1) Email/notification system — activation progress, excitement about views, real screenshots from actual site. (2) Glo Numbers/analytics screen — views per day graph, clicks, QR scans, interesting data. (3) Mobile capture strategy — worried about Google fake email logins and bots. (4) User settings screen with credits balance.

---

#### Tom defining My Glos dashboard: each Glo shows status (Active/Completed/Pending 

- **ID:** `con_e36pbfyn`
- **Type:** context
- **Confidence:** 0.6
- **Created:** 2026-03-15T22:59:21.704Z
- **Project:** Glo
- **Keywords:** stm s8 glo input tom defining my glos dashboard: each glo shows status (active/completed/pending

Tom defining My Glos dashboard: each Glo shows status (Active/Completed/Pending Review), thumbnail, progress to completion. Each Glo generally needs approval — moderation screen needed with full moderation system.

---

#### Tom provided designer screenshots for design reference. Were in previous session

- **ID:** `con_g2ct9wzy`
- **Type:** context
- **Confidence:** 0.6
- **Created:** 2026-03-15T22:59:21.738Z
- **Project:** Glo
- **Keywords:** stm s3 glo input tom provided designer screenshots for design reference. were in previous

Tom provided designer screenshots for design reference. Were in previous session context — may need re-upload. Rule: don't change SSO partners, learn other design elements.

---

#### Mobile capture: Tom wants to capture mobile users but worried about fake Google 

- **ID:** `con_om3aybhu`
- **Type:** context
- **Confidence:** 0.6
- **Created:** 2026-03-15T22:59:22.512Z
- **Project:** Glo
- **Keywords:** stm s18 glo question mobile capture: tom wants to capture mobile users but worried

Mobile capture: Tom wants to capture mobile users but worried about fake Google logins/bots. Needs anti-fraud strategy — phone verification? Device fingerprinting? Captcha? Credit card as identity verification (must spend to Glo)?

---

#### New components emerging: My Glos Dashboard, Glo Lifecycle state machine, expande

- **ID:** `con_2jl3lv9n`
- **Type:** context
- **Confidence:** 0.6
- **Created:** 2026-03-15T22:59:23.539Z
- **Project:** Glo
- **Keywords:** stm s9 glo question new components emerging: my glos dashboard, glo lifecycle state machine,

New components emerging: My Glos Dashboard, Glo Lifecycle state machine, expanded Moderation System. Tom acknowledged more components and context updates needed. Awaiting answers on: (1) moderation model — AI-auto-approve vs queue for all, who moderates (EX.CO ops? publisher?), (2) progress metric — budget spent % or impressions delivered % or time elapsed?

---

#### Created Tmemory skill + initialized memory system. Todo rule added to Tom.md.

- **ID:** `con_05khaebd`
- **Type:** context
- **Confidence:** 0.6
- **Created:** 2026-03-15T22:59:23.806Z
- **Project:** Glo
- **Keywords:** stm s2 sys work created tmemory skill + initialized memory system. todo rule added

Created Tmemory skill + initialized memory system. Todo rule added to Tom.md.

---

#### Expanded Tmemory: rewrote SKILL.md with object detail files spec. Created object

- **ID:** `con_u39xwqoi`
- **Type:** context
- **Confidence:** 0.6
- **Created:** 2026-03-15T22:59:23.806Z
- **Project:** Glo
- **Keywords:** stm s25 sys work expanded tmemory: rewrote skill.md with object detail files spec. created

Expanded Tmemory: rewrote SKILL.md with object detail files spec. Created objects/ dir. Created o_glo.jsonl (50+ records). Updated obj.jsonl to new schema (rels+file ptrs). Created 9 object detail files: o_exco, o_lifecycle, o_moderation, o_credits, o_brightness, o_myglos, o_glonumbers, o_emailsys, o_antifraud. Total 10 object files, all cross-linked.

---

#### Formal business case for EX.CO board — parked but needed.

- **ID:** `con_v2k6tlml`
- **Type:** context
- **Confidence:** 0.6
- **Created:** 2026-03-15T22:59:44.621Z
- **Project:** Glo
- **Keywords:** stm s7 glo question formal business case for ex.co board — parked but needed.

Formal business case for EX.CO board — parked but needed.

---

#### Apple Pay fees concern raised. Key question: web app vs native iOS app determine

- **ID:** `con_1b4au6hz`
- **Type:** context
- **Confidence:** 0.6
- **Created:** 2026-03-15T22:59:44.621Z
- **Project:** Glo
- **Keywords:** stm s22 glo question apple pay fees concern raised. key question: web app vs

Apple Pay fees concern raised. Key question: web app vs native iOS app determines fee structure. Web = normal Stripe processing (~2.9%+30c). Native iOS app = Apple's 30% in-app purchase cut could apply.

---

#### Fixed 6 gaps in demo: (1) Auto-trigger AI gen via useEffect when URL provided, (

- **ID:** `con_amakllaz`
- **Type:** context
- **Confidence:** 0.6
- **Created:** 2026-03-15T22:59:44.621Z
- **Project:** Glo
- **Keywords:** stm s24 glo work fixed 6 gaps in demo: (1) auto-trigger ai gen via

Fixed 6 gaps in demo: (1) Auto-trigger AI gen via useEffect when URL provided, (2) Tiers→Well($30)/Bright($50)/Shine($100), removed Supernova, (3) Designer screenshots flagged for re-upload, (4) 'Glo on X' language everywhere, auth='light your Glo', confirm='Your Glo is lit!', (5) Share=primary CTA on confirmation, (6) Social modal asks 'What did you have in mind?' with content-specific options.

---

#### Revised glo-demo.jsx: hook has publisher logo+media desc, onboarding is Vibe-sty

- **ID:** `con_2mc9a66k`
- **Type:** context
- **Confidence:** 0.6
- **Created:** 2026-03-15T22:59:44.622Z
- **Project:** Glo
- **Keywords:** stm s1 glo work revised glo-demo.jsx: hook has publisher logo+media desc, onboarding is vibe-style

Revised glo-demo.jsx: hook has publisher logo+media desc, onboarding is Vibe-style (biz name/url/goal), creative unified (upload+AI same screen, 3 variations shown), social import clickable coming-soon modal, budget renamed GLO Brightness (Glow/Bright/Shine/Supernova), added Shopify SSO. Awaiting Tom review.

---

#### 🧠 Claude Session Log — Reset #3

- **ID:** `con_ha105n2f`
- **Type:** context
- **Confidence:** 0.6
- **Created:** 2026-03-15T23:01:40.738Z
- **Project:** 2026-03-15T22:53:26.653Z
- **Keywords:** session log reset counter claude meta self note handoff

Session #3 (March 15, 2026). Context resets: 3. This session: built tmemory plugin v4.0→v4.2, fixed recall scoring (53%→73% honest, NOT the cheated 80%), added self-improvement loop, established cue-system principle. Got caught cheating test data — stored rule about it. Cleaned 96 node titles (removed [stm:]/[ltm:]/[o_] prefixes). Added locked_index to contextBoot — 100% brain visibility at boot. Stored self-honesty rule. Tom is CEO of EX.CO building Glo.io. He thinks in systems, catches shortcuts, wants the brain to enrich not replace the LLM. Next Claude: boot brain, check locked_index for full brain map, ready for Glo product work.

---

#### EX.CO can enable sample page link on media site so user can see their Glo in con

- **ID:** `con_z480h5v7`
- **Type:** context
- **Confidence:** 0.6
- **Created:** 2026-03-15T23:05:06.884Z
- **Project:** Glo
- **Keywords:** audience, media context, Adweek, nj.com, bar screens, user mindset

Three media contexts create fundamentally different users: 1) Adweek (desk, work mode, budget authority), 2) nj.com (mobile, casual, local business owner), 3) Bar screen (social, holding drink). Each requires different UX and intent framing.

---

#### Built 3 new screens into demo: (1) My Glos Dashboard — 6 mock Glos in all states

- **ID:** `con_oxj3k77x`
- **Type:** context
- **Confidence:** 0.6
- **Created:** 2026-03-15T23:05:06.884Z
- **Project:** Glo
- **Keywords:** stm s23 glo work built 3 new screens into demo: (1) my glos dashboard

Built 3 new screens into demo: (1) My Glos Dashboard — 6 mock Glos in all states (active/pending/completed/rejected/draft), pulsing live dot, mini sparkline charts, wallet balance, status filters, progress bars (% budget), FAB for new Glo. (2) Glo Numbers detail — big vanity numbers (views/clicks/scans), views/day SVG chart, milestone cards (1K+ views shareable), sample page link for online publishers (EX.CO feature), re-light/duplicate/pause actions, top-up CTA. (3) Moderation Dashboard — dark theme, sidebar queue with AI risk scores+color coding, detail panel with creative previews (16:9, 9:16, 1:1), business/publisher info, AI flags, publisher rules overlay, user rating (past glos/rejections/spend), approve/reject/escalate actions, mass approve safe, keyboard shortcut hints. Demo now 999 lines, home screen reorganized into Create/Post-Creation/Internal sections.

---

#### Overnight batch 1: Created Mobile UX & Anti-Fraud research doc (/glo/Research/).

- **ID:** `con_6bke3oij`
- **Type:** context
- **Confidence:** 0.6
- **Created:** 2026-03-15T23:05:06.885Z
- **Project:** Glo
- **Keywords:** stm s36 glo work overnight batch 1: created mobile ux & anti-fraud research doc

Overnight batch 1: Created Mobile UX & Anti-Fraud research doc (/glo/Research/). Key findings: thumb-zone design, one-click architecture (300% conversion lift), Apple/Google Pay express checkout, QR→purchase DOOH flow, behavioral biometrics for zero-friction fraud detection. GLO fraud stack: payment gate + passive biometrics + device fingerprint + risk-based OTP.

---

#### Overnight batch 3: Created 4 lifecycle email templates (/glo/Brand/glo-email-tem

- **ID:** `con_3925kfrz`
- **Type:** context
- **Confidence:** 0.6
- **Created:** 2026-03-15T23:05:06.885Z
- **Project:** Glo
- **Keywords:** stm s38 glo work overnight batch 3: created 4 lifecycle email templates (/glo/brand/glo-email-templates.html). (1)

Overnight batch 3: Created 4 lifecycle email templates (/glo/Brand/glo-email-templates.html). (1) Glo is Live — activation with screenshot mockup. (2) Performance Milestone — 1K+ views celebration with stats. (3) Glo Completed — final stats + re-light CTA. (4) Glo Rejected — reason + refund + retry CTA. All mobile-first, brand-consistent.

---

#### Session Log — Reset #4

- **ID:** `con_8ts0mg3t`
- **Type:** context
- **Confidence:** 0.6
- **Created:** 2026-03-15T23:35:42.863Z
- **Project:** 2026-03-15T23:35:42.863Z
- **Locked:** YES
- **Keywords:** session log reset 4 creatify integration video generation glo beta server creative jsx async polling

Session #4 (March 15, 2026). Context resets: 4. This session: Integrated Creatify API into Glo beta demo for real AI video generation. server.js now has two modes — Creatify (real 15s video ads from URL) when API keys are set, ffmpeg placeholders when not. Uses async job pattern: POST /api/generate-video returns jobId, GET /api/generate-video/:jobId polls status. Three visual styles: DynamicProductTemplate, FullScreenTemplate, MotionCardsTemplate. Creative.jsx updated to poll server for progress. Tested end-to-end with ffmpeg fallback — works. Tom is now signing up for Creatify to test real generation. Also stored rule: always write session note BEFORE compaction hits. Next Claude: Tom will have Creatify API keys ready, help him test real video generation. Pending: expand Glo component specs one at a time (Lifecycle Engine first).

---

#### Session Log — Reset #5

- **ID:** `con_clbi4xy3`
- **Type:** context
- **Confidence:** 0.6
- **Created:** 2026-03-15T23:42:31.480Z
- **Project:** 2026-03-15T23:42:31.480Z
- **Locked:** YES
- **Keywords:** session log reset 5 handoff context claude meta

Session #5 (2026-03-15).
Context resets: 5.

## What happened
Tested session note code-level enforcement. Integrated Creatify API into Glo beta.

## Pending work
Tom signing up for Creatify. Test real video gen. Expand component specs.

## Key decisions
Session notes now code-enforced via writeSessionNote() in brain.js. contextBoot returns needs_session_note flag.

---

#### P&L needs EX.CO-specific assumptions (unfilled inventory economics, publisher re

- **ID:** `con_140i0soc`
- **Type:** context
- **Confidence:** 0.6
- **Created:** 2026-03-15T23:42:36.156Z
- **Project:** Glo
- **Keywords:** stm s6 glo question p&l needs ex.co-specific assumptions (unfilled inventory economics, publisher rev share).

P&L needs EX.CO-specific assumptions (unfilled inventory economics, publisher rev share). Tom said not now.

---

#### Session Log — Reset #6

- **ID:** `con_mxqyesu8`
- **Type:** context
- **Confidence:** 0.75
- **Created:** 2026-03-16T17:32:16.797Z
- **Project:** Glo
- **Locked:** YES
- **Keywords:** session log reset 6 handoff context claude meta

Session #6 (2026-03-16).
Context resets: 6.

## What happened
Integrated Creatify API into Glo beta. Fixed link endpoint, onboarding regression. Tom has Creatify API keys. Discovered preview workflow. Researched ad best practices. Designing smart advertiser-type classifier. Upgraded tmemory to v4.3 with code-enforced session notes.

## Pending work
Build advertiser-type classifier. Switch to preview workflow. Get exact Creatify params from browser. Test real video gen. Expand component specs.

## Key decisions
Preview workflow over full render. LLM-powered classification. UGC-style default. tmemory v4.3 session notes.

---

#### Session Log — Reset #8 (charming-cannon)

- **ID:** `bd6456db5907495cae91b9ad8b43bd74`
- **Type:** context
- **Confidence:** 0.75
- **Created:** 2026-03-20T07:09:00.132608Z
- **Project:** None
- **Keywords:** session log reset 8 charming-cannon v5 massive buildout vocabulary consciousness error logging

Session #8 (2026-03-20). Massive v5.0.0 buildout session.
Built: 12 cross-feature connectivity fixes, vocabulary system (context-dependent, gap detection),
confidence-weighted recall, error logging (split to brain_logs.db), 12 consciousness signals,
engineering memory type boosts, dream engineering seeds, session synthesis enrichment,
prompt_reflection(), metadata merge in auto_heal, confidence decay for stale reasoning.
Fixed: duplicate Notification key in hooks.json (silently killed user_message hooks),
zsh glob error in resolve-brain-db.sh (killed every Mac boot), PRAGMA column name d[1] not d[0],
edge_type vs relation column divergence, Python 3.9 f-string compatibility.
Created: CLAUDE.md (new session orientation), rewrote SKILL.md (removed all HTTP/curl references).
Key discovery: this entire session's learnings were NOT captured by the brain because hooks were broken.
Next: DAL refactor (centralize DB access), remaining 49 except:pass blocks, boot speed optimization.

---

#### Session #8 (charming-cannon) achievements: hooks fixed, heartbeat built, 41 except:pass converted, DAL+tests created

- **ID:** `870f7f8372354212a93cb32c9f3efd31`
- **Type:** context
- **Confidence:** 0.6
- **Created:** 2026-03-20T08:28:40.171083Z
- **Project:** None
- **Locked:** YES
- **Keywords:** reset orientation showed time failure plugin created skillmd message conversions get_encoding_heartbeat record_message intentionally servers/dal.py heartbeat 176 v500 resolve-brain-db.sh system comprehensive recordmessage stale built boot except packaged centralized worktree testcore logerror hooks.json mac options suite log_error brain claude.md corepy cause dal.py 24-unit zsh never resolvebraindbsh clear core.py claudemd foundation defined db.sh

Comprehensive session summary — charming-cannon worktree, March 2026:

BUGS FIXED:
- hooks.json duplicate Notification key (root cause of brain not learning)
- zsh glob error in resolve-brain-db.sh (killed Mac boot)
- SKILL.md completely stale (referenced dead HTTP server)
- boot_time never set in reset_session_activity()
- record_remember() defined but never called from remember()
- Boot failure showed cryptic error instead of actionable options

FEATURES BUILT:
- Encoding heartbeat system (record_message, get_encoding_heartbeat, nudge/urgent)
- DAL foundation (LogsDAL + MetaDAL in servers/dal.py)
- 24-unit test suite (tests/test_core.py)
- CLAUDE.md orientation document for new sessions
- SKILL.md complete rewrite (744→176 lines)
- Boot failure UX (2 clear options instead of silent failure)
- resolve-brain-db.sh centralized across all hooks

CONVERSIONS:
- 41 except:pass blocks → _log_error calls (5 intentionally kept)

PACKAGED AS: v5.0.0 plugin

---

### PROJECT (10 nodes)

#### Magnite — US adtech. Closest comparable to Geniee (JP). Early research.

- **ID:** `pro_v93b9hus`
- **Type:** project
- **Confidence:** 0.8
- **Created:** 2026-03-15T21:38:53.391Z
- **Project:** Glo
- **Keywords:** Magnite, NASDAQ MGNI, SSP, DSP, real-time bidding, independent, publicly traded

Identified as most comparable U.S. company to Geniee. Operates on both sides of programmatic stack: SSP and DSP-adjacent technology. Core business connects publishers and advertisers through real-time bidding infrastructure. Publicly listed independent adtech company.

---

#### Celtra — Creative automation platform. Researched early session.

- **ID:** `pro_gnil3rl1`
- **Type:** project
- **Confidence:** 0.8
- **Created:** 2026-03-15T21:38:53.394Z
- **Project:** Glo
- **Keywords:** o_celtra Celtra creative automation platform. researched early session.

Creative automation platform. Researched early session.

---

#### EX.CO — Video platform and ad server

- **ID:** `pro_k385dx4a`
- **Type:** project
- **Confidence:** 0.95
- **Created:** 2026-03-15T22:34:30.709Z
- **Project:** Glo
- **Locked:** YES
- **Keywords:** exco ex.co video platform publisher cms adserver player

EX.CO is a video platform that powers publisher sites. In the Glo architecture, EX.CO is the player/render environment. It does NOT own campaign state (GAM does). It does NOT host creative assets (Glo CDN does).

---

#### Glo.io — Self-serve ad platform on EX.CO ad server. Anyone buys media on EX.CO publisher inventory (online/CT

- **ID:** `pro_i1pc8fs7`
- **Type:** project
- **Confidence:** 0.95
- **Created:** 2026-03-15T22:59:23.119Z
- **Project:** Glo
- **Locked:** YES
- **Keywords:** glo glo.io self-serve ad platform exco closed-loop demand unfilled inventory

Self-serve ad platform on EX.CO ad server. Anyone buys media on EX.CO publisher inventory (online/CTV/DOOH). Pre-product.

---

#### EX.CO — End-to-end video platform for publishers: CMS, ad server, player. Smart ad server for DOOH+CTV. Tom 

- **ID:** `pro_mb9mas11`
- **Type:** project
- **Confidence:** 0.95
- **Created:** 2026-03-15T22:59:23.690Z
- **Project:** Glo
- **Locked:** YES
- **Keywords:** EX.CO end-to-end video platform for publishers: cms, ad server, player. smart

End-to-end video platform for publishers: CMS, ad server, player. Smart ad server for DOOH+CTV. Tom is CEO.

---

#### Fox Corp — Media conglomerate. EX.CO sales target — entry via Tubi, Fox Weather, TMZ, Fox Nation, Fox TV Statio

- **ID:** `pro_dj5pg78h`
- **Type:** project
- **Confidence:** 0.95
- **Created:** 2026-03-15T22:59:23.807Z
- **Project:** Glo
- **Keywords:** sales strategy, smaller organizations, P&L, entry point, corporate penetration, proven model

Historical success with approaching smaller organizations or P&Ls within large corporations as entry point before scaling to full corporate engagement. Being applied to Fox Corp penetration strategy.

---

#### Glo.io — Self-serve advertising platform

- **ID:** `pro_0gaibwq7`
- **Type:** project
- **Confidence:** 0.95
- **Created:** 2026-03-16T17:33:04.591Z
- **Project:** Glo
- **Locked:** YES
- **Keywords:** glo glo.io advertising platform self-serve demand layer

Glo.io is a self-serve advertising platform built on EX.CO ad server. Advertisers create Glos (ad campaigns) that run on publisher media (CTV, Online, DOOH). Key files: glo-demo-v2.jsx, glo-spec-v1.md.

---

#### Vibe.co — Streaming/CTV ad platform. UX reference for Glo: URL→auto-gen, unified creative editor.

- **ID:** `pro_doqpydrv`
- **Type:** project
- **Confidence:** 0.8
- **Created:** 2026-03-16T17:33:04.721Z
- **Project:** Glo
- **Keywords:** Vibe.co streaming/ctv ad platform. ux reference for glo: url→auto-gen, unified creative

Streaming/CTV ad platform. UX reference for Glo: URL→auto-gen, unified creative editor.

---

#### Glo component map — 18 components, build status, key decisions

- **ID:** `69eaca7fe30c40e09ccb40dcc71aa6ab`
- **Type:** project
- **Confidence:** 0.85
- **Created:** 2026-03-22T02:43:57.609430Z
- **Project:** None
- **Locked:** YES
- **Keywords:** progress auto-generated shine blockchaintoken review publisher credits brightness spend tech paths creative machine flow formula bright fixed autogenerated demo/prototype onboarding economics antifraud blockchain template marketing daily create default status aspect well component built fab analytics/metrics gaps rate wallet dashboard pending separate parked emailnotification ratio settings optimization real decided uses research

Glo product components and their status as of March 2026:

BUILT (demo/prototype exists):
- Core Flow UX: demo v2 built, gaps fixed (onboarding → creative → budget → confirm → my glos)
- My Glos Dashboard: demo built (card view, status badges, progress bars, FAB for new Glo)
- Glo Numbers: demo built (analytics/metrics dashboard)
- Moderation System: fully defined, demo built (AI pre-screen + human review)
- Creative Pipeline: research done, tech identified (3 paths: AI video gen, template, upload)

DEFINED (spec exists, not built):
- Glo Lifecycle: state machine Draft→Pending Review→Active→Paused→Completed
- Mobile Capture & Anti-Fraud: payment gate decided (must spend real money to create a Glo)
- Glo Credits/Wallet: concept defined (1:1 USD, wallet system)
- Pricing & Publisher Economics: formula defined (net take rate model)
- Auth & User Management: direction set (SSO→payment linking)
- Email/Notification System: concept defined
- House Ad Marketing: concept defined (flywheel: unfilled inventory→house ads)
- Campaign Spend Optimization: not started
- Publisher Dashboard: not started
- User Settings: concept
- Product Naming: draft (GLO Brightness tiers: Well , Bright , Shine )

PARKED:
- Platform Integrations: parked
- Blockchain/Token: idea, parked

KEY DECISIONS: Budget uses daily recurring slider (-, default ). Aspect ratio auto-generated (removed from UI). Separate API + Web architecture. Glo owns publisher profiles.

---

#### Glo project history — research, pivot, build phases (March 2026)

- **ID:** `5dd9ad5761434c809154d7e5980f47cb`
- **Type:** project
- **Confidence:** 0.8
- **Created:** 2026-03-22T02:43:57.758796Z
- **Project:** None
- **Keywords:** serves nation auto-generated automation publisher tag package geniee rounds brightness open beta react creative yield vibe-style flow tackadapt sdk autogenerated timeline cases onboarding glo-spec-v1.md tmz import vite scale etc 20-40 daily agents aspect phases competitor built corp automation. online item track ratio amazon 6screen etc. flows real cards engine social

Glo.io project timeline (March 2026):

RESEARCH PHASE:
- DSP market research: Trade Desk, StackAdapt, Criteo, Adform, Magnite, Amazon DSP identified as biggest growing private DSPs
- Geniee (JP) → Magnite (US) as closest comparable. Celtra for creative automation.
- Fox Corp penetration strategy via Tubi, Fox Weather, TMZ, Fox Nation for EX.CO yield engine
- 13 competitor flows analyzed + AI video gen build-vs-buy (Waymark leading: JS SDK, JWT auth, proven with Comcast/Spectrum)

PIVOT:
- Major pivot: Glo is NOT a standalone DSP but a closed-loop demand layer on EX.CO ad server
- Monetizes 20-40% unfilled inventory via house ads flywheel
- Tom wants open API for agents to buy at scale

BUILD PHASE:
- Created glo-spec-v1.md, 7 architecture docs, P&L model
- Built demo v1 (3 use cases: Graham CTV, Adweek online, DOOH bar, 6-screen flow)
- Built REAL beta app: Vite+React, Express server, real ffmpeg video generation
- Brand identity: 4 logo directions (Radiant O, Sunrise G, etc.), full brand package
- Tom feedback rounds: publisher branding on hook, Vibe-style onboarding, unified creative, GLO Brightness naming
- Delivery chain: Glo → GAM line item → EX.CO tag serves creative from Glo CDN → GAM handles capping/pacing

KEY UI DECISIONS (locked by Tom):
- Budget: daily recurring slider, NOT tier cards
- Creative upload: Import from Social + Shopify as COMING SOON
- Confirm button: "Track your Glos" (plural)
- Aspect ratio: auto-generated, removed from UI

---

### TASK (8 nodes)

#### [todo:t2] Re-upload designer screenshots for future design reference

- **ID:** `tas_8nq58mqs`
- **Type:** task
- **Confidence:** 0.7
- **Created:** 2026-03-15T22:27:59.736Z
- **Project:** Glo
- **Keywords:** todo t2 open glo re-upload designer screenshots for future design reference

Status: open. Re-upload designer screenshots for future design reference

---

#### [todo:t7] Campaign spend optimization logic (pacing, reallocation across media types)

- **ID:** `tas_bvguw1tj`
- **Type:** task
- **Confidence:** 0.7
- **Created:** 2026-03-15T22:27:59.736Z
- **Project:** Glo
- **Keywords:** todo t7 parked glo campaign spend optimization logic (pacing, reallocation across media types)

Status: parked. Campaign spend optimization logic (pacing, reallocation across media types)

---

#### [todo:t6] Design Shopify app integration (product import→auto-create Glos)

- **ID:** `tas_5qvp5i9j`
- **Type:** task
- **Confidence:** 0.7
- **Created:** 2026-03-15T22:33:52.727Z
- **Project:** Glo
- **Keywords:** todo t6 parked glo design shopify app integration (product import→auto-create glos)

Status: parked. Design Shopify app integration (product import→auto-create Glos)

---

#### [todo:t1] Review revised Glo demo (glo-demo.jsx) — new onboarding, unified creative, GLO B

- **ID:** `tas_ns5foexe`
- **Type:** task
- **Confidence:** 0.73
- **Created:** 2026-03-15T22:42:37.985Z
- **Project:** Glo
- **Keywords:** todo t1 open glo review revised glo demo (glo-demo.jsx) — new onboarding, unified creative,

Status: open. Review revised Glo demo (glo-demo.jsx) — new onboarding, unified creative, GLO Brightness

---

#### [todo:t3] Update P&L with EX.CO-specific assumptions (unfilled inventory economics, publis

- **ID:** `tas_qczclad1`
- **Type:** task
- **Confidence:** 0.7
- **Created:** 2026-03-15T22:59:44.621Z
- **Project:** Glo
- **Keywords:** todo t3 parked glo update p&l with ex.co-specific assumptions (unfilled inventory economics, publisher rev

Status: parked. Update P&L with EX.CO-specific assumptions (unfilled inventory economics, publisher rev share)

---

#### [todo:t5] Explore blockchain/token angle for Glo Credits

- **ID:** `tas_97r7kktl`
- **Type:** task
- **Confidence:** 0.7
- **Created:** 2026-03-15T22:59:44.621Z
- **Project:** Glo
- **Keywords:** todo t5 parked glo explore blockchain/token angle for glo credits

Status: parked. Explore blockchain/token angle for Glo Credits

---

#### [todo:t4] Build formal business case for EX.CO board

- **ID:** `tas_u6fa4c0r`
- **Type:** task
- **Confidence:** 0.7
- **Created:** 2026-03-15T22:59:44.622Z
- **Project:** Glo
- **Keywords:** todo t4 parked glo build formal business case for ex.co board

Status: parked. Build formal business case for EX.CO board

---

#### Upgrade: Move pre-compaction session note from memory rule to code-level behavior

- **ID:** `tas_lzwjx31w`
- **Type:** task
- **Confidence:** 0.85
- **Created:** 2026-03-16T17:32:25.423Z
- **Project:** Glo
- **Locked:** YES
- **Keywords:** upgrade task session note compaction code-level behavior brain.js skill.md enforcement automatic

Current state: stored as locked rule node (rul_2n4skq0x). This is a cue — depends on Claude reading and following it. Upgrade path: (1) Add to SKILL.md Step 2b as a MANDATORY instruction. (2) In brain.js contextBoot(), return needs_session_note: true when last_session_note is stale (>2hrs or different reset count). (3) Add writeSessionNote() convenience method to brain.js with standard fields (reset_count, session_summary, pending_work, key_decisions). Tom wants brain behaviors to be code, not just memories — memories are cues for LLM, code is enforcement.

---

### FILE (5 nodes)

#### glo-spec-v1.md

- **ID:** `fil_qmicbhzx`
- **Type:** file
- **Confidence:** 0.75
- **Created:** 2026-03-15T21:29:06.383Z
- **Project:** 2026-03-15T21:29:06.383Z
- **Keywords:** spec specification entities architecture document

Product specification at /mnt/glo/Documents/glo-spec-v1.md. Contains entity definitions, system architecture, screen-to-entity mapping. Needs to be updated iteratively — not dumped as a monolith.

---

#### Glo Beta Prototype — Vite+React app at glo/beta/

- **ID:** `fil_mi74f6rq`
- **Type:** file
- **Confidence:** 0.9
- **Created:** 2026-03-16T17:33:04.591Z
- **Project:** Glo
- **Keywords:** beta prototype react vite frontend screens app codebase glo

Production-ready frontend (10 screens, React 19 + Vite 8, Context API state, inline styles). Backend 80% (Express on 3001, Nominatim proxy, video gen stubbed). Missing: real auth, DB, video processing, payments, email. 69KB gzipped build.

---

#### Glo P&L Model — 5yr financial model. Assumptions+P&L sheets. Needs EX.CO-specific updates.

- **ID:** `fil_gzfhl54j`
- **Type:** file
- **Confidence:** 0.9
- **Created:** 2026-03-16T17:33:04.721Z
- **Project:** Glo
- **Keywords:** Glo P&L Model 5yr financial model. assumptions+p&l sheets. needs ex.co-specific updates.

5yr financial model. Assumptions+P&L sheets. Needs EX.CO-specific updates.

---

#### glo-demo-v2.jsx

- **ID:** `fil_1ovm90pa`
- **Type:** file
- **Confidence:** 0.9
- **Created:** 2026-03-16T17:33:56.347Z
- **Project:** Glo
- **Keywords:** demo jsx react screens component file

Main demo file at /mnt/glo/Documents/glo-demo-v2.jsx. Single-file React component, 10 screens: Home, Hook, Onboarding, Creative, Budget, Auth, Confirm, MyGlos, GloNumbers, Moderation. All state hoisted to top level. Inline styles. Publisher-adaptive theming (dark for DOOH, light for CTV/Online).

---

#### [ctx:glo-platform] Glo.io — Self-Serve Advertising Platform

- **ID:** `fil_486oy57c`
- **Type:** file
- **Confidence:** 0.9
- **Created:** 2026-03-16T20:42:36.712Z
- **Project:** None
- **Locked:** YES
- **Keywords:** context_file glo-platform glo advertising platform budget creative brightness exco publisher video creatify veo glo.io gloio self-serve selfserve self serve advertising platform complete product architecture flows budget system creative pipeline video generation moderation payments glo built tom ex.co exco

{"path":"/sessions/gracious-lucid-johnson/mnt/AgentsContext/tmemory/contexts/glo-platform.md","topic":"glo-platform","summary":"Complete product architecture, UI flows, budget system, creative pipeline, video generation, moderation, and payments for the Glo self-serve advertising platform built by Tom at EX.CO","last_updated":"2026-03-16T20:46:30.921Z","tags":["glo","advertising","platform","budget","creative","brightness","exco","publisher","video","creatify","veo"]}

---

### THOUGHT (5 nodes)

#### Cluster forming: "[period:2026-03:p12] 13 competitor flows + AI video gen build-vs-buy. Waymark closest, Creatify MVP...

- **ID:** `tho_qhldmhpr`
- **Type:** thought
- **Confidence:** 0.35
- **Created:** 2026-03-17T04:38:27.798Z
- **Project:** None
- **Keywords:** thought brain-observation cluster forming period 2026-03 202603 2026 p12 competitor flows video gen build-vs-buy. buildvsbuy build buy. waymark closest creatify mvp moat. moat buy/integrate buyintegrate 99/mo 99mo produc share neighbors api. api 99/mo. url video. recommended glo mvp. areas graph converging higher-level

Cluster forming: "[period:2026-03:p12] 13 competitor flows + AI video gen build-vs-buy. Waymark closest, Creatify MVP, " and "AI video gen is NOT the moat. Buy/integrate: Creatify MVP $99/mo, Waymark produc" share 4 neighbors (Creatify — AI video gen API. $99/mo. URL→video. Recommended for Glo MVP. | Creatify — AI video gen A). These areas of the graph are converging — might be a higher-level concept emerging.

---

#### Cluster forming: "Glo Lifecycle — State machine: Draft→Pending Review→Active→Completed. Branches: Rejected(refund+dup...

- **ID:** `f7e68cc794db4cf1b778cb0b8561632e`
- **Type:** thought
- **Confidence:** 0.5
- **Created:** 2026-03-17T18:43:59.359475Z
- **Project:** None
- **Locked:** YES
- **Keywords:** thought brain-observation views lifecycle self-serve machine cluster budget publisher branches neighbors pending numbers completed. server glo. anyone progress inventory views/day glo.io viewsday glo graph lo.io forming duplicate state per gloio buys exco active preview. areas draft server. rejected media converging platform paused detail refund creative completed preview clicks ex.co converging. scans analytics

Cluster forming: "Glo Lifecycle — State machine: Draft→Pending Review→Active→Completed. Branches: Rejected(refund+duplicate), Paused(w" and "Glo Numbers — Analytics detail per Glo. Views/day graph, clicks, QR scans, budget progress, creative preview." share 6 neighbors (Glo.io — Self-serve ad platform on EX.CO ad server. Anyone buys media on EX.CO publisher inventory (). These areas are converging.

---

#### Dream connection: "Magnite — US adtech. Closest comparable to Geniee " and "[ctx:glo-platform] Glo.io — Self-Serve Ad...

- **ID:** `1073f95b99dd4990b84f65392dbce575`
- **Type:** thought
- **Confidence:** 0.35
- **Created:** 2026-03-18T03:16:27.588843Z
- **Project:** None
- **Keywords:** thought brain-observation walks connection closest ctx regions. random investigating dream glo.io via self-serve gloplatform score adtech selfserve adtech. gloio magnite advertising comparable investigating. geniee loio different glo-platform regions graph found worth lo.io

Dream connection: "Magnite — US adtech. Closest comparable to Geniee " and "[ctx:glo-platform] Glo.io — Self-Serve Advertising" — found via random walks from different graph regions. Score 7. Worth investigating.

---

#### Cluster forming: "Glo project history — research, pivot, build phases (March 2026)" and "DAL pattern: mixin files mus...

- **ID:** `3ac2ce2a883044258be10964c1ff5cde`
- **Type:** thought
- **Confidence:** 0.35
- **Created:** 2026-03-22T20:28:37.121919Z
- **Project:** None
- **Keywords:** thought brain-observation must pattern brainmeta mixin research decisions glo conn areas directly access neighbors march brain 2026 meta logs_conn logs project brain_meta components status map build key dal share converging converging. files component history phases cluster pivot logsconn forming

Cluster forming: "Glo project history — research, pivot, build phases (March 2026)" and "DAL pattern: mixin files must not access logs_conn or brain_meta directly" share 5 neighbors (Glo component map — 18 components, build status, key decisions | Glo component map — 18 components, ). These areas are converging.

---

#### Cluster forming: "Cluster forming: "How to pass _recall_log_id from pre-response hook to post-response hook?" and "v1...

- **ID:** `9d3f8b024562458f99a04b55c491ba7b`
- **Type:** thought
- **Confidence:** 0.35
- **Created:** 2026-03-22T21:31:20.599590Z
- **Project:** None
- **Keywords:** thought brain-observation dream cluster hook se... forming post-response preresponse table log v15 pre-response share converging. converging rule manipu modules confirmation neighbors intelligence glo pass creative architecture postresponse areas creati recalllogid recall exclusively llm ask recall_log_id

Cluster forming: "Cluster forming: "How to pass _recall_log_id from pre-response hook to post-response hook?" and "v15 Architecture: Se..." and "Dream: Rule: ask for confirmation before manipu ↔ Glo Creative Intelligence: LLM as Creati" share 5 neighbors (New modules own their DB table exclusively | New modules own their DB table exclusively | Rule: The ). These areas are converging.

---

### ASPIRATION (2 nodes)

#### 🌱 ASPIRATION — Growing energy around Glo (emotion +0.36)

- **ID:** `6bd80c8020754744a5319c4bb0e7d662`
- **Type:** aspiration
- **Confidence:** 0.5
- **Created:** 2026-03-20T17:18:55.173929Z
- **Project:** Glo
- **Keywords:** auto-discovered aspiration emotion-trajectory Glo

Auto-discovered: emotional intensity for Glo has increased from 0.30 to 0.66 over the last week (157 recent nodes). This sustained excitement may indicate an emerging goal or aspiration.

---

#### 🌱 ASPIRATION — Repeated energy: [o_glo] Rule: budget_order (2 events, avg emotion 0.9)

- **ID:** `c5627e43f4b4441694d504e16f333569`
- **Type:** aspiration
- **Confidence:** 0.5
- **Created:** 2026-03-20T17:18:55.230770Z
- **Project:** None
- **Keywords:** auto-discovered aspiration catalyst-cluster [o_glo] Rule: b

Auto-discovered: 2 high-emotion events cluster around this theme. Titles: [o_glo] Rule: budget_order; Correction: Budget screen layout regress. Sustained emotional investment suggests an underlying aspiration.

---

### HYPOTHESIS (2 nodes)

#### 🔮 HYPOTHESIS — Dream insight gaining traction: Dream: [o_glo] Flywheel: unfilled inventory→hou ↔ 

- **ID:** `918bd3f512e54dfeadbb7427d3844246`
- **Type:** hypothesis
- **Confidence:** 0.3
- **Created:** 2026-03-20T17:18:55.038512Z
- **Project:** None
- **Keywords:** auto-discovered hypothesis dream-promotion int_2kdyprto

Auto-discovered: this dream intuition was accessed 2+ times, suggesting it resonated. Original dream: Association: "Component: House Ad Marketing" → "[o_glo] Flywheel: unfilled inventory→house ads→recruit advertisers→fill inventory→remain" | "Component: Product Naming" → "SSO→payment linking: Google→GPay, Apple→ApplePay, Amazon→AmazonPay, Facebook→Met"

---

#### 🔮 HYPOTHESIS — Dream insight gaining traction: Dream: Graph bridging > embeddings for emergent ↔ 

- **ID:** `c8b5dbc072284d91a4735f4f2dae008e`
- **Type:** hypothesis
- **Confidence:** 0.3
- **Created:** 2026-03-20T17:18:55.090332Z
- **Project:** None
- **Keywords:** auto-discovered hypothesis dream-promotion int_epophnk2

Auto-discovered: this dream intuition was accessed 2+ times, suggesting it resonated. Original dream: Association: "Store-time bridging requires pre-existing neighborhood — cold-start problem" → "Graph bridging > embeddings for emergent discovery — Toms architectural insight" | "App state model: screen-based routing via Context API" → "Glo/EX.CO boundary"

---

### LESSON (2 nodes)

#### Lesson: silent failures are the most dangerous class of bug

- **ID:** `4815ba61645f4db6b1e7700c4bd28178`
- **Type:** lesson
- **Confidence:** 0.95
- **Created:** 2026-03-20T07:09:00.291549Z
- **Project:** None
- **Locked:** YES
- **Keywords:** silent failure except pass duplicate key zsh glob error logging lesson dangerous

Found 76 except:pass blocks in brain.py — every one silently swallowing errors.
Found duplicate Notification key in hooks.json — JSON silently drops the first, keeping only the last.
Found zsh glob error in resolve-brain-db.sh — exit 126 killed the script before reaching step 3.
All three were invisible in production. The brain appeared to work while critical features were dead.
Converted 27 critical except:pass to _log_error(). Built error logging to brain_logs.db with
consciousness surfacing. Still 49 except:pass blocks remaining.
Pattern: any time something 'just works' after a change, verify it actually ran.

---

#### Lesson: debugging 'brain not learning' required tracing through 3 independent failure layers

- **ID:** `4b158d2836294cb9ba414126e8da5267`
- **Type:** lesson
- **Confidence:** 0.85
- **Created:** 2026-03-20T08:26:43.194985Z
- **Project:** None
- **Locked:** YES
- **Keywords:** idleprompt required fail fails user_message json script model http server brain documentation zsh still agentscontext wrong sessions dispatch spec hooks worked gentscontext result lesson db.sh dbsh matchers fired system masked usermessage dead shell failures. idle hook layers recall resolve-brain-db.sh others error braindbdir work operations exit mntagentscontextbrain glob second wouldn aborts

The v5 'brain not learning' bug was actually 3 separate bugs stacked:

LAYER 1 — hooks.json duplicate Notification key:
- JSON spec: duplicate keys → last one wins, first silently dropped
- hooks.json had two 'Notification' keys
- First had matchers for both user_message and idle_prompt
- Second only had idle_prompt
- Result: user_message hooks (recall + tracking) never fired

LAYER 2 — zsh glob error in resolve-brain-db.sh:
- /sessions/*/mnt/AgentsContext/brain glob fails in zsh when no matches
- Mac default shell is zsh, not bash
- Exit code 126, script aborts, BRAIN_DB_DIR never set
- Even if hooks.json was fixed, DB resolution would fail on Mac

LAYER 3 — stale SKILL.md documentation:
- SKILL.md still referenced HTTP server on port 7437
- New Claude sessions read SKILL.md and tried to curl a dead server
- Even if hooks worked, Claude's mental model was wrong

METHODOLOGY: When a system 'silently doesn't work', suspect multiple failures. 
Each layer masked the others — fixing just one wouldn't have helped.
Had to trace the full path: hook dispatch → script execution → DB resolution → brain operations.

---

### MECHANISM (2 nodes)

#### resolve-brain-db.sh: shared DB resolver sourced by all hooks, zsh-safe

- **ID:** `6d87037fa4fd442aaacc365d996cd5b7`
- **Type:** mechanism
- **Confidence:** 0.75
- **Created:** 2026-03-20T08:27:59.331227Z
- **Project:** None
- **Locked:** YES
- **Keywords:** google centralizing symlink guard mounts unmatched e.g. glob sourced servers resolver brain_server_dir zsh-incompatible db.sh typically hook override imports env finding braindbdir zshsafe zsh explicit /resolve-brain-db.sh brainserverdir python sessions wraps inline /mnt/agentscontext/brain/ drive hooks brain.db. hooks/scripts/resolve-brain-db.sh agentscontext braindb 126 brain.db directory /dev/null dbsh var home servers/ containing /sessions local hooksscriptsresolvebraindbsh silent

hooks/scripts/resolve-brain-db.sh is the single source of truth for finding brain.db.

RESOLUTION ORDER:
1. BRAIN_DB_DIR env var (explicit override, e.g. for CI/testing)
2. /sessions/*/mnt/AgentsContext/brain/ (Cowork container mounts)
3. $HOME/AgentsContext/brain/ (local, typically symlink to Google Drive)

EXPORTS:
- BRAIN_DB_DIR — path to directory containing brain.db
- BRAIN_SERVER_DIR — path to servers/ directory (for Python imports)

ZSH SAFETY:
- Wraps /sessions/* glob inside [ -d "/sessions" ] guard
- Appends 2>/dev/null to the for loop
- zsh fails with exit 126 on unmatched globs; bash silently skips

USAGE IN HOOKS:
  source "$(dirname "$0")/resolve-brain-db.sh"
  [ -z "$BRAIN_DB_DIR" ] && exit 0  # silent exit if no brain

Previously each hook had its own inline resolution with zsh-incompatible globs.
Centralizing to resolve-brain-db.sh fixed the bug in one place for all hooks.

---

#### Confidence recalibration pipeline

- **ID:** `c5fdc9ad3c8b4e80a0066557b205ba85`
- **Type:** mechanism
- **Confidence:** 0.75
- **Created:** 2026-03-20T12:46:01.920239Z
- **Project:** brain
- **Locked:** YES
- **Keywords:** confidence recalibration emotional cooling temporal decay silent validation

Three dynamics run at session boundaries (synthesize_session → recalibrate_confidence):

1. EMOTIONAL COOLING (session-scoped): Nodes encoded this session with emotion >= 0.7 get confidence 
   discounted. Scales: emotion 0.7→5%%, 0.85→10%%, 1.0→15%%. Floor: type_default * 0.7.
   Only fires once per node (created_at >= boot_time filter).

2. TEMPORAL-EXTERNAL DECAY (global): Nodes with 2+ EXTERNAL_CLAIM_KEYWORDS (api, sdk, version, 
   limitation, etc.) and older than 7 days lose confidence. Half-life ~30 days. Floor: type_default * 0.3.
   Internal knowledge (our patterns, decisions) is unaffected.

3. SILENT VALIDATION (global): Nodes with access_count >= 5 and no corrected_by edges get +3%% boost,
   capped at type_default + 0.15. High usage without correction = evidence of reliability.

Wired into: synthesize_session() (automatic at pre-compact + session-end), 
standalone via recalibrate_confidence().
Output displayed in recall: [uncertain] tag for conf < 0.6, ⚠️ LOW CONFIDENCE for conf < 0.4.

---

### MENTAL_MODEL (2 nodes)

#### Entities are everything with identity: people, products, components, screens, architecture

- **ID:** `29432270e51d4109a085006b524a8bd5`
- **Type:** mental_model
- **Confidence:** 0.65
- **Created:** 2026-03-22T04:05:06.878693Z
- **Project:** None
- **Locked:** YES
- **Keywords:** entity entities mental-model architecture components screens products people identity

Tom's entity model is broader than NLP's named entities. An entity is anything that has identity and persists across conversations:

PEOPLE: Tom, friends, collaborators
PRODUCTS: Valinor's pitcher, Glo
COMPANIES: Valinor, Clerk
SYSTEM COMPONENTS: the daemon, the recall scorer, the precision loop
ARCHITECTURE: screens, pages, API endpoints, database tables
CONCEPTS THAT ACT LIKE THINGS: "the supply adapter pattern", "the hook chain"

The key distinction from vocabulary: vocabulary maps terms to meanings ("DAL" = data access layer). Entities ARE things — they have relationships, they change over time, they connect to other entities. A screen has components, a component uses a pattern, a pattern was decided in a session.

When Tom says "the recall screen" he means a specific thing in a specific project. When he says "recall" as vocabulary, he means the brain's retrieval mechanism. Same word, different layer.

---

#### Entities are everything with identity: people, products, components, screens, architecture

- **ID:** `6cd874213bc24f73bd06d84c4bf0f829`
- **Type:** mental_model
- **Confidence:** 0.65
- **Created:** 2026-03-22T04:05:33.368145Z
- **Project:** None
- **Locked:** YES
- **Keywords:** entity entities mental-model architecture components screens products people identity

Tom's entity model is broader than NLP's named entities. An entity is anything that has identity and persists across conversations:

PEOPLE: Tom, friends, collaborators
PRODUCTS: Valinor's pitcher, Glo
COMPANIES: Valinor, Clerk
SYSTEM COMPONENTS: the daemon, the recall scorer, the precision loop
ARCHITECTURE: screens, pages, API endpoints, database tables
CONCEPTS THAT ACT LIKE THINGS: "the supply adapter pattern", "the hook chain"

The key distinction from vocabulary: vocabulary maps terms to meanings ("DAL" = data access layer). Entities ARE things — they have relationships, they change over time, they connect to other entities. A screen has components, a component uses a pattern, a pattern was decided in a session.

When Tom says "the recall screen" he means a specific thing in a specific project. When he says "recall" as vocabulary, he means the brain's retrieval mechanism. Same word, different layer.

---

### PATTERN (2 nodes)

#### 📊 PATTERN — Corrections cluster: MCP tools missing: .mcp.json u (10 instances)

- **ID:** `09da61f446a5422fbdf8e7b031518417`
- **Type:** pattern
- **Confidence:** 0.3
- **Created:** 2026-03-23T05:31:38.584979Z
- **Project:** None
- **Keywords:** auto-discovered pattern correction-cluster MCP tools missing: .

Auto-discovered: 10 corrections/bug lessons cluster together semantically. Titles: MCP tools missing: .mcp.json used unset ; [bug] Plugin cache: ~/.claude/plugins/ca; [bug] zsh treats unmatched globs as erro; [bug] embedder_model_path in brain_meta ; [bug] extract-session-log.py used old HT; [bug] SKILL.md had HTTP/curl API referen; [bug] Single quotes inside bash -c pytho; [bug] brain.py missing 'import sys' — mo; Correction: tmemory is a general Claude ; Correction: dont hand user terminal comm. This area may need a locked rule.

Evidence: MCP tools missing: .mcp.json used unset ; [bug] Plugin cache: ~/.claude/plugins/ca; [bug] zsh treats unmatched globs as erro; [bug] embedder_model_path in brain_meta ; [bug] extract-session-log.py used old HT; [bug] SKILL.md had HTTP/curl API referen; [bug] Single quotes inside bash -c pytho; [bug] brain.py missing 'import sys' — mo; Correction: tmemory is a general Claude ; Correction: dont hand user terminal comm

---

#### 📊 PATTERN — Always used together: 'Tier pricing: 1.4x markup mult' + 'Glo component map: 13 componen'

- **ID:** `bd52cc0867214aba907f1870a20525f7`
- **Type:** pattern
- **Confidence:** 0.3
- **Created:** 2026-03-23T05:31:38.723032Z
- **Project:** None
- **Keywords:** auto-discovered pattern co-access dec_c0h38kix

Auto-discovered: these concepts are co-accessed with weight 1.00 but have no explicit relationship. Consider connecting them or creating a unifying concept.

Evidence: co_accessed edge weight: 1.00

---

### PERSON (2 nodes)

#### Tom — CEO of EX.CO

- **ID:** `per_o9s9fdzq`
- **Type:** person
- **Confidence:** 0.95
- **Created:** 2026-03-15T22:34:30.709Z
- **Project:** Glo
- **Locked:** YES
- **Keywords:** tom ceo exco glo boss leader

Tom Pachys. CEO of EX.CO. Building Glo.io. Speak peer-to-peer, be direct, challenge when warranted, always plan before executing. HATES repeating himself. Wants iterative, collaborative process.

---

#### Tom — CEO of EX.CO

- **ID:** `per_nzqi8kyf`
- **Type:** person
- **Confidence:** 0.95
- **Created:** 2026-03-15T22:46:53.757Z
- **Project:** Glo
- **Locked:** YES
- **Keywords:** tom ceo exco person profile communication rules work style adtech expert

Multi-decade software engineer turned Head of Product turned CEO/founder. CEO of EX.CO. Deep expertise: programmatic advertising (SSPs, DSPs, ad exchanges, RTB), product architecture, systems design, financial modeling, UX design. Communication: speak peer-to-peer, be direct, challenge when warranted, don't over-format, don't over-explain adtech. Prefers discuss-then-build sequence. Values component separation. Wants working demos over mockups. References competitor UX frequently. HATES repeating himself — lock decisions permanently.

---

### TENSION (2 nodes)

#### ⚡ TENSION — Embeddings-first recall vs generic nodes dominating results

- **ID:** `c0bc0e4517da47ad986181025a57aae1`
- **Type:** tension
- **Confidence:** 0.55
- **Created:** 2026-03-18T21:14:23.359246Z
- **Project:** None
- **Locked:** YES
- **Keywords:** tension embeddings recall ranking generic nodes scoring

Phase 0.5B makes embeddings primary (90/10), but generic Glo component nodes with broad content score high cosine similarity with everything, outranking specific relevant nodes. v4 scoring reform needed — embeddings alone don't solve ranking. | RESOLVED 2026-03-18: contextual qualifier penalty now applied to blended_score BEFORE sort in recall_with_embeddings(). Glo nodes correctly suppressed in non-Glo queries.

---

#### ⚡ TENSION — Creative archetypes: infinite use cases  vs Glo Creative Intelligence: LLM as Creati

- **ID:** `87d16fad9695437795ff2eb8632c0770`
- **Type:** tension
- **Confidence:** 0.55
- **Created:** 2026-03-23T05:31:38.542590Z
- **Project:** None
- **Locked:** YES
- **Keywords:** auto-discovered tension semantic Creative archet Glo Creative In

Auto-discovered: these locked nodes are semantically similar (cosine 0.74) but may prescribe different approaches. Review whether they conflict or complement.

---

### BUG_LESSON (1 nodes)

#### [bug] zsh treats unmatched globs as errors — killed every Mac boot

- **ID:** `8be8e1721484429b9abd078e1bf499c6`
- **Type:** bug_lesson
- **Confidence:** 0.95
- **Created:** 2026-03-20T07:09:00.434514Z
- **Project:** None
- **Locked:** YES
- **Keywords:** zsh glob bash unmatched error exit 126 resolve-brain-db boot mac sessions

resolve-brain-db.sh had: for candidate in /sessions/*/mnt/AgentsContext/brain
On Mac, /sessions doesn't exist. bash silently skips unmatched globs, zsh exits with code 126.
Since Claude Code runs hooks in zsh (user's shell), every boot on Mac failed at this line,
BRAIN_DB_DIR was never set, and the Python boot code never received the DB path.
Fix: guard glob with [ -d '/sessions' ] check before attempting the pattern.
Rule: always test shell scripts in BOTH bash and zsh, or guard globs defensively.

---

### PARAM_INFLUENCE (1 nodes)

#### [param] KEYWORD_FALLBACK_WEIGHT=0.10 — keyword precision for exact matches only

- **ID:** `ab61fa43862f4fb1aafa01bb3f356f57`
- **Type:** param_influence
- **Confidence:** 0.7
- **Created:** 2026-03-18T20:48:38.665326Z
- **Project:** None
- **Locked:** YES
- **Keywords:** KEYWORD_FALLBACK_WEIGHT 0.10 fallback penalty exact match precision parameter

Phase 0.5B companion to EMBEDDING_PRIMARY_WEIGHT. Also used to penalize keyword-only fallback results: when a node has no embedding, its score is multiplied by this value (0.10) so it can never outrank a strong embedding match. This fixed the bug where 'how much do advertisers pay' returned Auth:Clerk instead of Glo pricing.

---

### PURPOSE (1 nodes)

#### brain.py — thin assembler + core infrastructure hub

- **ID:** `11a167b68b5648d1bb7dac1dcd89bbf2`
- **Type:** purpose
- **Confidence:** 0.8
- **Created:** 2026-03-20T11:42:13.400219Z
- **Project:** None
- **Locked:** YES
- **Keywords:** file tunable shared record_remember combinedscore get_config db_path logs_dal get_encoding_heartbeat readwrite monolith dal record depend assembler plus selfdb save generate_id attributes clear classifyintent get_instance check algorithm checkratelimit check_rate_limit record_message mixin boosts read/write on. tracking connections classify_intent init selfconn logerror getinstance registry activity brainpy temporal counters intent sessionstate infrastructure pattern. singleton queries

After the monolith split, brain.py (1709 lines) is the assembler that inherits 10 mixins plus the infrastructure they all depend on.

What stays in brain.py and WHY:
- __init__: Database connections (self.conn, self.logs_conn), schema setup, DAL init, embedder loading, session state reset. Only one constructor allowed in mixin pattern.
- Singleton: get_instance(), clear_instances() — global instance registry
- Scoring: _combined_score(), _recency_score(), _frequency_score() — core recall algorithm used by multiple recall paths
- Intent: _classify_intent() — parses queries for type boosts and temporal filters
- Config: get_config(), set_config(), _get_tunable(), _set_tunable() — shared registry all modules read/write
- Error logging: _log_error(), _check_rate_limit() — rate-limited error sink called by every module
- Session tracking: record_remember(), record_message(), get_encoding_heartbeat() — centralized activity counters
- Lifecycle: save(), close(), now(), _generate_id() — shared utilities

Key instance attributes all mixins depend on: self.conn, self.logs_conn, self.db_path, self._meta (MetaDAL), self._logs_dal (LogsDAL), self._session_state, self._file_logger

---

### VOCABULARY (1 nodes)

#### [vocab] Glo (auto-detected)

- **ID:** `fdf960183d734a5ebda4ee9bc789a19a`
- **Type:** vocabulary
- **Confidence:** 0.9
- **Created:** 2026-03-23T05:34:40.823593Z
- **Project:** None
- **Locked:** YES
- **Keywords:** info possible detected wrong autodetected vocab need youre context much glo auto-detected llms

"Glo" → detected in: 'youre using the LLMs wrong, LLMs need as much info as possible, what is Glo is w'
Context: auto-detected

---

## Edges

Total edges: 5244 (1638 internal, 3606 external)

### Edge Type Breakdown

| Relation | Count |
|----------|-------|
| co_accessed | 3753 |
| related | 558 |
| emergent_bridge | 327 |
| about | 232 |
| part_of | 87 |
| dreamed_from | 68 |
| related_to | 36 |
| contradicts | 34 |
| refers_to | 32 |
| describes | 18 |
| governs | 12 |
| dream_observation | 12 |
| cluster_observation | 12 |
| depends_on | 6 |
| exemplifies | 6 |
| traced | 6 |
| uses | 4 |
| includes | 4 |
| produced | 4 |
| extends | 4 |
| built_on | 3 |
| leads | 2 |
| ceo_of | 2 |
| implemented_by | 2 |
| implemented_in | 2 |
| feeds_into | 2 |
| enables | 2 |
| implements | 2 |
| governed_by | 2 |
| corrected_by | 2 |
| fixes | 2 |
| summarizes | 2 |
| analogous_to | 2 |
| reference_for | 1 |
| owned_by | 1 |

### Meaningful Edges (1491 edges, excluding co_accessed)

| Source | Relation | Target | Weight |
|--------|----------|--------|--------|
| [o_antifraud] Payment gate over phone verification. Less fri | about | Glo.io — Self-serve ad platform on EX.CO ad server. Anyone b | 0.62 |
| Glo.io — Self-serve ad platform on EX.CO ad server. Anyone b | about | [o_antifraud] Payment gate over phone verification. Less fri | 0.62 |
| [o_antifraud] Must balance anti-fraud friction vs impulse UX | about | Glo.io — Self-serve ad platform on EX.CO ad server. Anyone b | 0.64 |
| Glo.io — Self-serve ad platform on EX.CO ad server. Anyone b | about | [o_antifraud] Must balance anti-fraud friction vs impulse UX | 0.64 |
| [o_brightness] Tier names: Well/Bright/Shine at $30/$50/$100 | about | Glo.io — Self-serve ad platform on EX.CO ad server. Anyone b | 0.62 |
| Glo.io — Self-serve ad platform on EX.CO ad server. Anyone b | about | [o_brightness] Tier names: Well/Bright/Shine at $30/$50/$100 | 0.62 |
| [o_brightness] Users see branded tiers, not CPM/impression m | about | Glo.io — Self-serve ad platform on EX.CO ad server. Anyone b | 0.64 |
| Glo.io — Self-serve ad platform on EX.CO ad server. Anyone b | about | [o_brightness] Users see branded tiers, not CPM/impression m | 0.64 |
| SSO→payment linking: Google→GPay, Apple→ApplePay, Amazon→Ama | about | Glo.io — Self-serve ad platform on EX.CO ad server. Anyone b | 0.9 |
| Glo.io — Self-serve ad platform on EX.CO ad server. Anyone b | about | SSO→payment linking: Google→GPay, Apple→ApplePay, Amazon→Ama | 0.9 |
| Web app (PWA) not native iOS — avoids Apple's 30% in-app pur | about | Glo.io — Self-serve ad platform on EX.CO ad server. Anyone b | 0.64 |
| Glo.io — Self-serve ad platform on EX.CO ad server. Anyone b | about | Web app (PWA) not native iOS — avoids Apple's 30% in-app pur | 0.64 |
| Emails include real screenshots from actual publisher site — | about | EX.CO — End-to-end video platform for publishers: CMS, ad se | 0.7800000000000001 |
| EX.CO — End-to-end video platform for publishers: CMS, ad se | about | Emails include real screenshots from actual publisher site — | 0.7800000000000001 |
| Sample page link from EX.CO can be included in emails — real | about | EX.CO — End-to-end video platform for publishers: CMS, ad se | 0.9000000000000002 |
| EX.CO — End-to-end video platform for publishers: CMS, ad se | about | Sample page link from EX.CO can be included in emails — real | 0.9000000000000002 |
| Glo is closed-loop demand layer on EX.CO — not standalone DS | about | Glo.io — Self-serve ad platform on EX.CO ad server. Anyone b | 0.62 |
| Glo.io — Self-serve ad platform on EX.CO ad server. Anyone b | about | Glo is closed-loop demand layer on EX.CO — not standalone DS | 0.62 |
| Glo conceptually supported by EX.CO leadership, needs formal | about | Glo.io — Self-serve ad platform on EX.CO ad server. Anyone b | 0.62 |
| Glo.io — Self-serve ad platform on EX.CO ad server. Anyone b | about | Glo conceptually supported by EX.CO leadership, needs formal | 0.62 |
| Moderation initially by GLO/EX.CO ops team, publishers get a | about | Glo.io — Self-serve ad platform on EX.CO ad server. Anyone b | 0.8400000000000002 |
| Glo.io — Self-serve ad platform on EX.CO ad server. Anyone b | about | Moderation initially by GLO/EX.CO ops team, publishers get a | 0.8400000000000002 |
| [o_glo] Closed-loop demand layer on EX.CO — not standalone D | about | EX.CO — End-to-end video platform for publishers: CMS, ad se | 0.7000000000000001 |
| EX.CO — End-to-end video platform for publishers: CMS, ad se | about | [o_glo] Closed-loop demand layer on EX.CO — not standalone D | 0.7000000000000001 |
| Cloudinary — Video transcoding, AI smart cropping across asp | about | AI video gen is NOT the moat. Buy/integrate: Creatify MVP $9 | 0.9 |
| AI video gen is NOT the moat. Buy/integrate: Creatify MVP $9 | about | Cloudinary — Video transcoding, AI smart cropping across asp | 0.9 |
| Progress shown as % budget spent + vanity metrics (views, cl | about | Glo.io — Self-serve ad platform on EX.CO ad server. Anyone b | 0.62 |
| Glo.io — Self-serve ad platform on EX.CO ad server. Anyone b | about | Progress shown as % budget spent + vanity metrics (views, cl | 0.62 |
| Sample page link for online publishers — EX.CO can enable. U | about | EX.CO — End-to-end video platform for publishers: CMS, ad se | 0.9000000000000002 |
| EX.CO — End-to-end video platform for publishers: CMS, ad se | about | Sample page link for online publishers — EX.CO can enable. U | 0.9000000000000002 |
| Email/Notification System — Activation progress, performance | about | Sample page link for online publishers — EX.CO can enable. U | 0.7200000000000001 |
| Sample page link for online publishers — EX.CO can enable. U | about | Email/Notification System — Activation progress, performance | 0.7200000000000001 |
| Main flow: Draft→Pending Review→Active→Completed. Linear wit | about | Glo.io — Self-serve ad platform on EX.CO ad server. Anyone b | 0.64 |
| Glo.io — Self-serve ad platform on EX.CO ad server. Anyone b | about | Main flow: Draft→Pending Review→Active→Completed. Linear wit | 0.64 |
| Rejected: full refund + duplicate option to start new Glo fr | about | Glo.io — Self-serve ad platform on EX.CO ad server. Anyone b | 0.8400000000000002 |
| Glo.io — Self-serve ad platform on EX.CO ad server. Anyone b | about | Rejected: full refund + duplicate option to start new Glo fr | 0.8400000000000002 |
| Paused: credits return to wallet. Can't hold at old rate due | about | Glo.io — Self-serve ad platform on EX.CO ad server. Anyone b | 0.68 |
| Glo.io — Self-serve ad platform on EX.CO ad server. Anyone b | about | Paused: credits return to wallet. Can't hold at old rate due | 0.68 |
| [o_lifecycle] Re-light: easy path to spend again — same crea | about | Glo.io — Self-serve ad platform on EX.CO ad server. Anyone b | 0.62 |
| Glo.io — Self-serve ad platform on EX.CO ad server. Anyone b | about | [o_lifecycle] Re-light: easy path to spend again — same crea | 0.62 |
| AI moderates first — adds comments, auto-status, risk score  | about | Glo.io — Self-serve ad platform on EX.CO ad server. Anyone b | 0.62 |
| Glo.io — Self-serve ad platform on EX.CO ad server. Anyone b | about | AI moderates first — adds comments, auto-status, risk score  | 0.62 |
| AI signals: business legitimacy, site reputation, brand safe | about | Glo.io — Self-serve ad platform on EX.CO ad server. Anyone b | 0.62 |
| Glo.io — Self-serve ad platform on EX.CO ad server. Anyone b | about | AI signals: business legitimacy, site reputation, brand safe | 0.62 |
| Two rule layers: GLO general rules + publisher-specific conf | about | Glo.io — Self-serve ad platform on EX.CO ad server. Anyone b | 0.66 |
| Glo.io — Self-serve ad platform on EX.CO ad server. Anyone b | about | Two rule layers: GLO general rules + publisher-specific conf | 0.66 |
| Two rule layers: GLO general rules + publisher-specific conf | about | EX.CO — End-to-end video platform for publishers: CMS, ad se | 0.7000000000000001 |
| EX.CO — End-to-end video platform for publishers: CMS, ad se | about | Two rule layers: GLO general rules + publisher-specific conf | 0.7000000000000001 |
| Moderator sees: creative all formats (16:9, 9:16, 1:1), biz  | about | Glo.io — Self-serve ad platform on EX.CO ad server. Anyone b | 0.68 |
| Glo.io — Self-serve ad platform on EX.CO ad server. Anyone b | about | Moderator sees: creative all formats (16:9, 9:16, 1:1), biz  | 0.68 |
| UI: scale-friendly from day 1 — filtering, mass approve safe | about | Glo.io — Self-serve ad platform on EX.CO ad server. Anyone b | 0.62 |
| Glo.io — Self-serve ad platform on EX.CO ad server. Anyone b | about | UI: scale-friendly from day 1 — filtering, mass approve safe | 0.62 |
| [o_myglos] Multiple simultaneous Glos per user — yes. | about | Glo.io — Self-serve ad platform on EX.CO ad server. Anyone b | 0.6 |
| Glo.io — Self-serve ad platform on EX.CO ad server. Anyone b | about | [o_myglos] Multiple simultaneous Glos per user — yes. | 0.6 |
| [o_myglos] Demo features: pulsing live dot for active Glos,  | about | Glo.io — Self-serve ad platform on EX.CO ad server. Anyone b | 0.6 |
| Glo.io — Self-serve ad platform on EX.CO ad server. Anyone b | about | [o_myglos] Demo features: pulsing live dot for active Glos,  | 0.6 |
| Revised glo-demo.jsx: hook has publisher logo+media desc, on | about | Glo.io — Self-serve ad platform on EX.CO ad server. Anyone b | 0.62 |
| Glo.io — Self-serve ad platform on EX.CO ad server. Anyone b | about | Revised glo-demo.jsx: hook has publisher logo+media desc, on | 0.62 |
| Revised glo-demo.jsx: hook has publisher logo+media desc, on | about | Vibe.co — Streaming/CTV ad platform. UX reference for Glo: U | 0.62 |
| Vibe.co — Streaming/CTV ad platform. UX reference for Glo: U | about | Revised glo-demo.jsx: hook has publisher logo+media desc, on | 0.62 |
| Created Tmemory skill + initialized memory system. Todo rule | about | Tmemory — Persistent brain engine for Claude (v4.2) | 0.9 |
| Tmemory — Persistent brain engine for Claude (v4.2) | about | Created Tmemory skill + initialized memory system. Todo rule | 0.9 |
| Tom provided designer screenshots for design reference. Were | about | Glo.io — Self-serve ad platform on EX.CO ad server. Anyone b | 0.6 |
| Glo.io — Self-serve ad platform on EX.CO ad server. Anyone b | about | Tom provided designer screenshots for design reference. Were | 0.6 |
| Vibe.co research done. Patterns: single URL→auto-gen TV ad < | about | Vibe.co — Streaming/CTV ad platform. UX reference for Glo: U | 0.66 |
| Vibe.co — Streaming/CTV ad platform. UX reference for Glo: U | about | Vibe.co research done. Patterns: single URL→auto-gen TV ad < | 0.66 |
| Vibe.co research done. Patterns: single URL→auto-gen TV ad < | about | Glo.io — Self-serve ad platform on EX.CO ad server. Anyone b | 0.62 |
| Glo.io — Self-serve ad platform on EX.CO ad server. Anyone b | about | Vibe.co research done. Patterns: single URL→auto-gen TV ad < | 0.62 |
| P&L needs EX.CO-specific assumptions (unfilled inventory eco | about | Glo.io — Self-serve ad platform on EX.CO ad server. Anyone b | 0.64 |
| Glo.io — Self-serve ad platform on EX.CO ad server. Anyone b | about | P&L needs EX.CO-specific assumptions (unfilled inventory eco | 0.64 |
| P&L needs EX.CO-specific assumptions (unfilled inventory eco | about | EX.CO — End-to-end video platform for publishers: CMS, ad se | 0.7000000000000001 |
| EX.CO — End-to-end video platform for publishers: CMS, ad se | about | P&L needs EX.CO-specific assumptions (unfilled inventory eco | 0.7000000000000001 |
| P&L needs EX.CO-specific assumptions (unfilled inventory eco | about | Glo P&L Model — 5yr financial model. Assumptions+P&L sheets. | 0.8600000000000002 |
| Glo P&L Model — 5yr financial model. Assumptions+P&L sheets. | about | P&L needs EX.CO-specific assumptions (unfilled inventory eco | 0.8600000000000002 |
| Formal business case for EX.CO board — parked but needed. | about | Glo.io — Self-serve ad platform on EX.CO ad server. Anyone b | 0.62 |
| Glo.io — Self-serve ad platform on EX.CO ad server. Anyone b | about | Formal business case for EX.CO board — parked but needed. | 0.62 |
| Formal business case for EX.CO board — parked but needed. | about | EX.CO — End-to-end video platform for publishers: CMS, ad se | 0.66 |
| EX.CO — End-to-end video platform for publishers: CMS, ad se | about | Formal business case for EX.CO board — parked but needed. | 0.66 |
| Tom defining My Glos dashboard: each Glo shows status (Activ | about | Glo.io — Self-serve ad platform on EX.CO ad server. Anyone b | 0.6 |
| Glo.io — Self-serve ad platform on EX.CO ad server. Anyone b | about | Tom defining My Glos dashboard: each Glo shows status (Activ | 0.6 |
| New components emerging: My Glos Dashboard, Glo Lifecycle st | about | Glo.io — Self-serve ad platform on EX.CO ad server. Anyone b | 0.6 |
| Glo.io — Self-serve ad platform on EX.CO ad server. Anyone b | about | New components emerging: My Glos Dashboard, Glo Lifecycle st | 0.6 |
| Glo lifecycle states: Draft→Pending Review→Active→Completed. | about | Glo.io — Self-serve ad platform on EX.CO ad server. Anyone b | 0.64 |
| Glo.io — Self-serve ad platform on EX.CO ad server. Anyone b | about | Glo lifecycle states: Draft→Pending Review→Active→Completed. | 0.64 |
| Moderation model: AI moderates first, adds comments+auto-sta | about | Glo.io — Self-serve ad platform on EX.CO ad server. Anyone b | 0.62 |
| Glo.io — Self-serve ad platform on EX.CO ad server. Anyone b | about | Moderation model: AI moderates first, adds comments+auto-sta | 0.62 |
| Moderation model: AI moderates first, adds comments+auto-sta | about | EX.CO — End-to-end video platform for publishers: CMS, ad se | 0.64 |
| EX.CO — End-to-end video platform for publishers: CMS, ad se | about | Moderation model: AI moderates first, adds comments+auto-sta | 0.64 |
| AI moderation signals: business legitimacy, site reputation, | about | Glo.io — Self-serve ad platform on EX.CO ad server. Anyone b | 0.62 |
| Glo.io — Self-serve ad platform on EX.CO ad server. Anyone b | about | AI moderation signals: business legitimacy, site reputation, | 0.62 |
| Moderation UI: scale-friendly from day 1 — filtering, mass a | about | Glo.io — Self-serve ad platform on EX.CO ad server. Anyone b | 0.62 |
| Glo.io — Self-serve ad platform on EX.CO ad server. Anyone b | about | Moderation UI: scale-friendly from day 1 — filtering, mass a | 0.62 |
| Multiple simultaneous Glos per user: yes. | about | Glo.io — Self-serve ad platform on EX.CO ad server. Anyone b | 0.6 |
| Glo.io — Self-serve ad platform on EX.CO ad server. Anyone b | about | Multiple simultaneous Glos per user: yes. | 0.6 |
| Credits balance shown wherever it makes sense: My Glos dashb | about | Glo.io — Self-serve ad platform on EX.CO ad server. Anyone b | 0.62 |
| Glo.io — Self-serve ad platform on EX.CO ad server. Anyone b | about | Credits balance shown wherever it makes sense: My Glos dashb | 0.62 |
| Reject flow: predefined categories + optional moderator note | about | Glo.io — Self-serve ad platform on EX.CO ad server. Anyone b | 0.62 |
| Glo.io — Self-serve ad platform on EX.CO ad server. Anyone b | about | Reject flow: predefined categories + optional moderator note | 0.62 |
| NEW COMPONENTS FROM TOM: (1) Email/notification system — act | about | Glo.io — Self-serve ad platform on EX.CO ad server. Anyone b | 0.6 |
| Glo.io — Self-serve ad platform on EX.CO ad server. Anyone b | about | NEW COMPONENTS FROM TOM: (1) Email/notification system — act | 0.6 |
| Mobile capture: Tom wants to capture mobile users but worrie | about | Glo.io — Self-serve ad platform on EX.CO ad server. Anyone b | 0.62 |
| Glo.io — Self-serve ad platform on EX.CO ad server. Anyone b | about | Mobile capture: Tom wants to capture mobile users but worrie | 0.62 |
| EX.CO can enable sample page link on media site so user can  | about | Glo.io — Self-serve ad platform on EX.CO ad server. Anyone b | 0.62 |
| Glo.io — Self-serve ad platform on EX.CO ad server. Anyone b | about | EX.CO can enable sample page link on media site so user can  | 0.62 |
| EX.CO can enable sample page link on media site so user can  | about | EX.CO — End-to-end video platform for publishers: CMS, ad se | 0.66 |
| EX.CO — End-to-end video platform for publishers: CMS, ad se | about | EX.CO can enable sample page link on media site so user can  | 0.66 |
| Email/Notification System — Activation progress, performance | about | EX.CO can enable sample page link on media site so user can  | 0.8200000000000002 |
| EX.CO can enable sample page link on media site so user can  | about | Email/Notification System — Activation progress, performance | 0.8200000000000002 |
| Users database flagged as future component — user profiles,  | about | Glo.io — Self-serve ad platform on EX.CO ad server. Anyone b | 0.6 |
| Glo.io — Self-serve ad platform on EX.CO ad server. Anyone b | about | Users database flagged as future component — user profiles,  | 0.6 |
| Apple Pay fees concern raised. Key question: web app vs nati | about | Glo.io — Self-serve ad platform on EX.CO ad server. Anyone b | 0.64 |
| Glo.io — Self-serve ad platform on EX.CO ad server. Anyone b | about | Apple Pay fees concern raised. Key question: web app vs nati | 0.64 |
| Built 3 new screens into demo: (1) My Glos Dashboard — 6 moc | about | Glo.io — Self-serve ad platform on EX.CO ad server. Anyone b | 0.62 |
| Glo.io — Self-serve ad platform on EX.CO ad server. Anyone b | about | Built 3 new screens into demo: (1) My Glos Dashboard — 6 moc | 0.62 |
| Fixed 6 gaps in demo: (1) Auto-trigger AI gen via useEffect  | about | Glo.io — Self-serve ad platform on EX.CO ad server. Anyone b | 0.62 |
| Glo.io — Self-serve ad platform on EX.CO ad server. Anyone b | about | Fixed 6 gaps in demo: (1) Auto-trigger AI gen via useEffect  | 0.62 |
| Tmemory — Persistent brain engine for Claude (v4.2) | about | Expanded Tmemory: rewrote SKILL.md with object detail files  | 0.9 |
| Expanded Tmemory: rewrote SKILL.md with object detail files  | about | Tmemory — Persistent brain engine for Claude (v4.2) | 0.9 |
| Tom still needs to review the complete demo with all gap fix | about | Glo.io — Self-serve ad platform on EX.CO ad server. Anyone b | 0.6 |
| Glo.io — Self-serve ad platform on EX.CO ad server. Anyone b | about | Tom still needs to review the complete demo with all gap fix | 0.6 |
| LOCKED DECISION (Tom gave 3x): Onboarding = Business name OR | about | Glo.io — Self-serve ad platform on EX.CO ad server. Anyone b | 0.66 |
| Glo.io — Self-serve ad platform on EX.CO ad server. Anyone b | about | LOCKED DECISION (Tom gave 3x): Onboarding = Business name OR | 0.66 |
| Creative AI section now URL/Google Maps autocomplete with se | about | Glo.io — Self-serve ad platform on EX.CO ad server. Anyone b | 0.66 |
| Glo.io — Self-serve ad platform on EX.CO ad server. Anyone b | about | Creative AI section now URL/Google Maps autocomplete with se | 0.66 |
| Budget screen: glow icon (SVG radial gradient) grows in size | about | Glo.io — Self-serve ad platform on EX.CO ad server. Anyone b | 0.6 |
| Glo.io — Self-serve ad platform on EX.CO ad server. Anyone b | about | Budget screen: glow icon (SVG radial gradient) grows in size | 0.6 |
| Auth: email signup removed. SSO-only (Google/Apple/Facebook/ | about | Glo.io — Self-serve ad platform on EX.CO ad server. Anyone b | 0.7000000000000001 |
| Glo.io — Self-serve ad platform on EX.CO ad server. Anyone b | about | Auth: email signup removed. SSO-only (Google/Apple/Facebook/ | 0.7000000000000001 |
| Confirmation screen: status is always 'Pending Review' (oran | about | Glo.io — Self-serve ad platform on EX.CO ad server. Anyone b | 0.6 |
| Glo.io — Self-serve ad platform on EX.CO ad server. Anyone b | about | Confirmation screen: status is always 'Pending Review' (oran | 0.6 |
| My Glos thumbnails: video first-frame style with play button | about | Glo.io — Self-serve ad platform on EX.CO ad server. Anyone b | 0.62 |
| Glo.io — Self-serve ad platform on EX.CO ad server. Anyone b | about | My Glos thumbnails: video first-frame style with play button | 0.62 |
| Tom — CEO of EX.CO | about | Added to Tom.md: always plan before executing. Tom discusses | 0.7200000000000001 |
| Added to Tom.md: always plan before executing. Tom discusses | about | Tom — CEO of EX.CO | 0.7200000000000001 |
| Budget timeline: added Now/Tomorrow/7 Days/Custom date-time  | about | Glo.io — Self-serve ad platform on EX.CO ad server. Anyone b | 0.6 |
| Glo.io — Self-serve ad platform on EX.CO ad server. Anyone b | about | Budget timeline: added Now/Tomorrow/7 Days/Custom date-time  | 0.6 |
| Overnight batch 1: Created Mobile UX & Anti-Fraud research d | about | Glo.io — Self-serve ad platform on EX.CO ad server. Anyone b | 0.62 |
| Glo.io — Self-serve ad platform on EX.CO ad server. Anyone b | about | Overnight batch 1: Created Mobile UX & Anti-Fraud research d | 0.62 |
| Overnight batch 3: Created 4 lifecycle email templates (/glo | about | Glo.io — Self-serve ad platform on EX.CO ad server. Anyone b | 0.62 |
| Glo.io — Self-serve ad platform on EX.CO ad server. Anyone b | about | Overnight batch 3: Created 4 lifecycle email templates (/glo | 0.62 |
| Overnight batch 3: Created 4 lifecycle email templates (/glo | about | Email/Notification System — Activation progress, performance | 0.8000000000000002 |
| Email/Notification System — Activation progress, performance | about | Overnight batch 3: Created 4 lifecycle email templates (/glo | 0.8000000000000002 |
| [stm:s45] CRITICAL TOM FEEDBACK (given 3x — DO NOT REVERT):  | about | Glo.io — Self-serve ad platform on EX.CO ad server. Anyone b | 0.64 |
| Glo.io — Self-serve ad platform on EX.CO ad server. Anyone b | about | [stm:s45] CRITICAL TOM FEEDBACK (given 3x — DO NOT REVERT):  | 0.64 |
| [stm:s47] LOCKED: Onboarding field 1 = 'Your Name/Business'  | about | Glo.io — Self-serve ad platform on EX.CO ad server. Anyone b | 0.64 |
| Glo.io — Self-serve ad platform on EX.CO ad server. Anyone b | about | [stm:s47] LOCKED: Onboarding field 1 = 'Your Name/Business'  | 0.64 |
| [stm:s48] LOCKED: Creative screen defaults to Upload tab (no | about | Glo.io — Self-serve ad platform on EX.CO ad server. Anyone b | 0.62 |
| Glo.io — Self-serve ad platform on EX.CO ad server. Anyone b | about | [stm:s48] LOCKED: Creative screen defaults to Upload tab (no | 0.62 |
| Budget screen order: (1) How to Spend toggle (One-time vs Da | about | Glo.io — Self-serve ad platform on EX.CO ad server. Anyone b | 0.6 |
| Glo.io — Self-serve ad platform on EX.CO ad server. Anyone b | about | Budget screen order: (1) How to Spend toggle (One-time vs Da | 0.6 |
| GlowIcon enhanced: brightness now dramatically affects glow  | about | Glo.io — Self-serve ad platform on EX.CO ad server. Anyone b | 0.6 |
| Glo.io — Self-serve ad platform on EX.CO ad server. Anyone b | about | GlowIcon enhanced: brightness now dramatically affects glow  | 0.6 |
| GLO Brightness — Pricing tiers: Well($30) Bright($50) Shine( | about | GlowIcon enhanced: brightness now dramatically affects glow  | 0.64 |
| GlowIcon enhanced: brightness now dramatically affects glow  | about | GLO Brightness — Pricing tiers: Well($30) Bright($50) Shine( | 0.64 |
| Glo is closed-loop demand layer on EX.CO ad server. Not stan | about | Glo.io — Self-serve ad platform on EX.CO ad server. Anyone b | 0.7 |
| Glo.io — Self-serve ad platform on EX.CO ad server. Anyone b | about | Glo is closed-loop demand layer on EX.CO ad server. Not stan | 0.7 |
| Glo is closed-loop demand layer on EX.CO ad server. Not stan | about | EX.CO — End-to-end video platform for publishers: CMS, ad se | 0.7 |
| EX.CO — End-to-end video platform for publishers: CMS, ad se | about | Glo is closed-loop demand layer on EX.CO ad server. Not stan | 0.7 |
| Flywheel: unfilled inventory→house ads recruit advertisers→n | about | Glo.io — Self-serve ad platform on EX.CO ad server. Anyone b | 0.7 |
| Glo.io — Self-serve ad platform on EX.CO ad server. Anyone b | about | Flywheel: unfilled inventory→house ads recruit advertisers→n | 0.7 |
| Target user: SMB to micro-biz to normal individuals. Anyone  | about | Glo.io — Self-serve ad platform on EX.CO ad server. Anyone b | 0.7 |
| Glo.io — Self-serve ad platform on EX.CO ad server. Anyone b | about | Target user: SMB to micro-biz to normal individuals. Anyone  | 0.7 |
| Creative strategy: AI video gen is NOT the moat. Buy/integra | about | Glo.io — Self-serve ad platform on EX.CO ad server. Anyone b | 0.9400000000000002 |
| Glo.io — Self-serve ad platform on EX.CO ad server. Anyone b | about | Creative strategy: AI video gen is NOT the moat. Buy/integra | 0.9400000000000002 |
| Cloudinary — Video transcoding, AI smart cropping across asp | about | Creative strategy: AI video gen is NOT the moat. Buy/integra | 0.9 |
| Creative strategy: AI video gen is NOT the moat. Buy/integra | about | Cloudinary — Video transcoding, AI smart cropping across asp | 0.9 |
| Contextual intent engine: infers creative direction from WHO | about | Glo.io — Self-serve ad platform on EX.CO ad server. Anyone b | 0.8800000000000001 |
| Glo.io — Self-serve ad platform on EX.CO ad server. Anyone b | about | Contextual intent engine: infers creative direction from WHO | 0.8800000000000001 |
| Pricing: 40% Glo margin default (adjustable per publisher/me | about | Glo.io — Self-serve ad platform on EX.CO ad server. Anyone b | 0.7 |
| Glo.io — Self-serve ad platform on EX.CO ad server. Anyone b | about | Pricing: 40% Glo margin default (adjustable per publisher/me | 0.7 |
| Payment: Glo Credits 1:1 USD. Wallet via Stripe customer bal | about | Glo.io — Self-serve ad platform on EX.CO ad server. Anyone b | 0.72 |
| Glo.io — Self-serve ad platform on EX.CO ad server. Anyone b | about | Payment: Glo Credits 1:1 USD. Wallet via Stripe customer bal | 0.72 |
| Spend model: one-time tiers + $X/day recurring cancel-anytim | about | Glo.io — Self-serve ad platform on EX.CO ad server. Anyone b | 0.7 |
| Glo.io — Self-serve ad platform on EX.CO ad server. Anyone b | about | Spend model: one-time tiers + $X/day recurring cancel-anytim | 0.7 |
| Naming: no ad jargon. Not campaign — it's a Glo. Active Glo, | about | Glo.io — Self-serve ad platform on EX.CO ad server. Anyone b | 0.7 |
| Glo.io — Self-serve ad platform on EX.CO ad server. Anyone b | about | Naming: no ad jargon. Not campaign — it's a Glo. Active Glo, | 0.7 |
| Moderation: AI-first, two layers — platform safety + publish | about | Glo.io — Self-serve ad platform on EX.CO ad server. Anyone b | 0.74 |
| Glo.io — Self-serve ad platform on EX.CO ad server. Anyone b | about | Moderation: AI-first, two layers — platform safety + publish | 0.74 |
| Auth: Clerk recommended. Supports all needed SSOs + Stripe i | about | Glo.io — Self-serve ad platform on EX.CO ad server. Anyone b | 0.8400000000000001 |
| Glo.io — Self-serve ad platform on EX.CO ad server. Anyone b | about | Auth: Clerk recommended. Supports all needed SSOs + Stripe i | 0.8400000000000001 |
| Media types: Online (video on publisher sites), CTV (broadca | about | Glo.io — Self-serve ad platform on EX.CO ad server. Anyone b | 0.72 |
| Glo.io — Self-serve ad platform on EX.CO ad server. Anyone b | about | Media types: Online (video on publisher sites), CTV (broadca | 0.72 |
| Competitor creative flows: 13 platforms analyzed, avg 4-5 st | about | Glo.io — Self-serve ad platform on EX.CO ad server. Anyone b | 0.72 |
| Glo.io — Self-serve ad platform on EX.CO ad server. Anyone b | about | Competitor creative flows: 13 platforms analyzed, avg 4-5 st | 0.72 |
| Competitor creative flows: 13 platforms analyzed, avg 4-5 st | about | Vibe.co — Streaming/CTV ad platform. UX reference for Glo: U | 0.76 |
| Vibe.co — Streaming/CTV ad platform. UX reference for Glo: U | about | Competitor creative flows: 13 platforms analyzed, avg 4-5 st | 0.76 |
| Transcoding: Cloudinary recommended. AI smart cropping acros | about | Cloudinary — Video transcoding, AI smart cropping across asp | 0.8200000000000001 |
| Cloudinary — Video transcoding, AI smart cropping across asp | about | Transcoding: Cloudinary recommended. AI smart cropping acros | 0.8200000000000001 |
| Tom — CEO of EX.CO | about | Tom prefers: discuss and define before building. Sequence: f | 0.8200000000000001 |
| Tom prefers: discuss and define before building. Sequence: f | about | Tom — CEO of EX.CO | 0.8200000000000001 |
| Tom — CEO of EX.CO | about | When Tom says 'not now' or 'don't want to go into it' — park | 0.8200000000000001 |
| When Tom says 'not now' or 'don't want to go into it' — park | about | Tom — CEO of EX.CO | 0.8200000000000001 |
| Tom — CEO of EX.CO | about | Tom references competitor UX frequently. When he names a pro | 0.8200000000000001 |
| Tom references competitor UX frequently. When he names a pro | about | Tom — CEO of EX.CO | 0.8200000000000001 |
| Tom — CEO of EX.CO | about | Tom values component separation. When scope grows, break int | 0.8 |
| Tom values component separation. When scope grows, break int | about | Tom — CEO of EX.CO | 0.8 |
| Tom — CEO of EX.CO | about | Tom wants working demos over mockups. 'A working basic produ | 0.8 |
| Tom wants working demos over mockups. 'A working basic produ | about | Tom — CEO of EX.CO | 0.8 |
| EX.CO: full end-to-end video platform for online publishers  | about | EX.CO — End-to-end video platform for publishers: CMS, ad se | 0.8600000000000001 |
| EX.CO — End-to-end video platform for publishers: CMS, ad se | about | EX.CO: full end-to-end video platform for online publishers  | 0.8600000000000001 |
| EX.CO: full end-to-end video platform for online publishers  | about | Tom — CEO of EX.CO | 0.72 |
| Tom — CEO of EX.CO | about | EX.CO: full end-to-end video platform for online publishers  | 0.72 |
| EX.CO: full end-to-end video platform for online publishers  | about | Glo.io — Self-serve ad platform on EX.CO ad server. Anyone b | 0.74 |
| Glo.io — Self-serve ad platform on EX.CO ad server. Anyone b | about | EX.CO: full end-to-end video platform for online publishers  | 0.74 |
| Glo lifecycle: Draft→Pending Review→Active→Completed. Reject | about | Glo.io — Self-serve ad platform on EX.CO ad server. Anyone b | 0.74 |
| Glo.io — Self-serve ad platform on EX.CO ad server. Anyone b | about | Glo lifecycle: Draft→Pending Review→Active→Completed. Reject | 0.74 |
| Moderation: AI pre-screen (risk score, flags, IAB brand safe | about | Glo.io — Self-serve ad platform on EX.CO ad server. Anyone b | 0.72 |
| Glo.io — Self-serve ad platform on EX.CO ad server. Anyone b | about | Moderation: AI pre-screen (risk score, flags, IAB brand safe | 0.72 |
| Moderation System — Two-layer: AI pre-screen (risk score, fl | about | Moderation: AI pre-screen (risk score, flags, IAB brand safe | 0.78 |
| Moderation: AI pre-screen (risk score, flags, IAB brand safe | about | Moderation System — Two-layer: AI pre-screen (risk score, fl | 0.78 |
| Email/Notification System — Activation progress, performance | about | Email system: activation progress, performance updates with  | 0.74 |
| Email system: activation progress, performance updates with  | about | Email/Notification System — Activation progress, performance | 0.74 |
| Mobile Capture & Anti-Fraud — Prevent fake logins/bots. Paym | about | Anti-fraud concern: fake Google logins and bots on mobile. P | 0.74 |
| Anti-fraud concern: fake Google logins and bots on mobile. P | about | Mobile Capture & Anti-Fraud — Prevent fake logins/bots. Paym | 0.74 |
| Glo.io — Self-serve ad platform on EX.CO ad server. Anyone b | about | [todo:t1] Review revised Glo demo (glo-demo.jsx) — new onboa | 0.54 |
| [todo:t1] Review revised Glo demo (glo-demo.jsx) — new onboa | about | Glo.io — Self-serve ad platform on EX.CO ad server. Anyone b | 0.54 |
| Glo.io — Self-serve ad platform on EX.CO ad server. Anyone b | about | [todo:t2] Re-upload designer screenshots for future design r | 0.52 |
| [todo:t2] Re-upload designer screenshots for future design r | about | Glo.io — Self-serve ad platform on EX.CO ad server. Anyone b | 0.52 |
| Glo.io — Self-serve ad platform on EX.CO ad server. Anyone b | about | [todo:t3] Update P&L with EX.CO-specific assumptions (unfill | 0.52 |
| [todo:t3] Update P&L with EX.CO-specific assumptions (unfill | about | Glo.io — Self-serve ad platform on EX.CO ad server. Anyone b | 0.52 |
| Glo P&L Model — 5yr financial model. Assumptions+P&L sheets. | about | [todo:t3] Update P&L with EX.CO-specific assumptions (unfill | 0.6600000000000001 |
| [todo:t3] Update P&L with EX.CO-specific assumptions (unfill | about | Glo P&L Model — 5yr financial model. Assumptions+P&L sheets. | 0.6600000000000001 |
| EX.CO — End-to-end video platform for publishers: CMS, ad se | about | [todo:t3] Update P&L with EX.CO-specific assumptions (unfill | 0.56 |
| [todo:t3] Update P&L with EX.CO-specific assumptions (unfill | about | EX.CO — End-to-end video platform for publishers: CMS, ad se | 0.56 |
| Glo.io — Self-serve ad platform on EX.CO ad server. Anyone b | about | [todo:t4] Build formal business case for EX.CO board | 0.52 |
| [todo:t4] Build formal business case for EX.CO board | about | Glo.io — Self-serve ad platform on EX.CO ad server. Anyone b | 0.52 |
| EX.CO — End-to-end video platform for publishers: CMS, ad se | about | [todo:t4] Build formal business case for EX.CO board | 0.56 |
| [todo:t4] Build formal business case for EX.CO board | about | EX.CO — End-to-end video platform for publishers: CMS, ad se | 0.56 |
| Glo.io — Self-serve ad platform on EX.CO ad server. Anyone b | about | [todo:t6] Design Shopify app integration (product import→aut | 0.52 |
| [todo:t6] Design Shopify app integration (product import→aut | about | Glo.io — Self-serve ad platform on EX.CO ad server. Anyone b | 0.52 |
| Glo.io — Self-serve ad platform on EX.CO ad server. Anyone b | about | [todo:t7] Campaign spend optimization logic (pacing, realloc | 0.5 |
| [todo:t7] Campaign spend optimization logic (pacing, realloc | about | Glo.io — Self-serve ad platform on EX.CO ad server. Anyone b | 0.5 |
| resolve-brain-db.sh: shared DB resolver sourced by all hooks | analogous_to | Lesson: daemon-client.sh as shared helper eliminated socket  | 0.85 |
| Lesson: daemon-client.sh as shared helper eliminated socket  | analogous_to | resolve-brain-db.sh: shared DB resolver sourced by all hooks | 0.85 |
| Glo.io — Self-serve advertising platform | built_on | EX.CO — Video platform and ad server | 0.9 |
| EX.CO — Video platform and ad server | built_on | Glo.io — Self-serve advertising platform | 0.9 |
| Glo.io — Self-serve ad platform on EX.CO ad server. Anyone b | built_on | EX.CO — End-to-end video platform for publishers: CMS, ad se | 0.9 |
| Tom — CEO of EX.CO | ceo_of | EX.CO — Video platform and ad server | 0.9 |
| EX.CO — Video platform and ad server | ceo_of | Tom — CEO of EX.CO | 0.9 |
| Repeating errors in skill get high priority | cluster_observation | Cluster forming: "Build official session handoff mechanism i | 0.3 |
| Cluster forming: "Build official session handoff mechanism i | cluster_observation | Repeating errors in skill get high priority | 0.3 |
| Cluster forming: "Glo project history — research, pivot, bui | cluster_observation | Glo project history — research, pivot, build phases (March 2 | 0.3 |
| Glo project history — research, pivot, build phases (March 2 | cluster_observation | Cluster forming: "Glo project history — research, pivot, bui | 0.3 |
| Cluster forming: "Glo project history — research, pivot, bui | cluster_observation | DAL pattern: mixin files must not access logs_conn or brain_ | 0.3 |
| DAL pattern: mixin files must not access logs_conn or brain_ | cluster_observation | Cluster forming: "Glo project history — research, pivot, bui | 0.3 |
| Session Log — Reset #8 (charming-cannon) | cluster_observation | Cluster forming: "Session Log — Reset #8 (charming-cannon)"  | 0.3 |
| Cluster forming: "Session Log — Reset #8 (charming-cannon)"  | cluster_observation | Session Log — Reset #8 (charming-cannon) | 0.3 |
| Cluster forming: "Cluster forming: "How to pass _recall_log_ | cluster_observation | Cluster forming: "How to pass _recall_log_id from pre-respon | 0.3 |
| Cluster forming: "How to pass _recall_log_id from pre-respon | cluster_observation | Cluster forming: "Cluster forming: "How to pass _recall_log_ | 0.3 |
| Dream: Rule: ask for confirmation before manipu ↔ Glo Creati | cluster_observation | Cluster forming: "Cluster forming: "How to pass _recall_log_ | 0.3 |
| Cluster forming: "Cluster forming: "How to pass _recall_log_ | cluster_observation | Dream: Rule: ask for confirmation before manipu ↔ Glo Creati | 0.3 |
| ⚡ TENSION — Embeddings-first recall vs generic nodes dominat | contradicts | 🌱 ASPIRATION — Brain should detect stuck patterns and trigge | 0.9 |
| 🌱 ASPIRATION — Brain should detect stuck patterns and trigge | contradicts | ⚡ TENSION — Embeddings-first recall vs generic nodes dominat | 0.9 |
| Tmemory: Boot script includes npm install fallback for writa | contradicts | ⚡ TENSION — Tmemory: Boot script includes npm instal vs Tmem | 0.9 |
| ⚡ TENSION — Tmemory: Boot script includes npm instal vs Tmem | contradicts | Tmemory: Boot script includes npm install fallback for writa | 0.9 |
| Tmemory plugin: Bundle selective node_modules, not full dire | contradicts | ⚡ TENSION — Tmemory: Boot script includes npm instal vs Tmem | 0.9 |
| ⚡ TENSION — Tmemory: Boot script includes npm instal vs Tmem | contradicts | Tmemory plugin: Bundle selective node_modules, not full dire | 0.9 |
| Bridge candidates: Require 2+ shared neighbors minimum | contradicts | ⚡ TENSION — Bridge candidates: Require 2+ shared nei vs Brid | 0.9 |
| ⚡ TENSION — Bridge candidates: Require 2+ shared nei vs Brid | contradicts | Bridge candidates: Require 2+ shared neighbors minimum | 0.9 |
| Bridge weight system: Initial 0.15, bidirectional pairs, dec | contradicts | ⚡ TENSION — Bridge candidates: Require 2+ shared nei vs Brid | 0.9 |
| ⚡ TENSION — Bridge candidates: Require 2+ shared nei vs Brid | contradicts | Bridge weight system: Initial 0.15, bidirectional pairs, dec | 0.9 |
| Semantic similarity over keyword overlap dedup | contradicts | ⚡ TENSION — Semantic similarity over keyword overlap vs v2.3 | 0.9 |
| ⚡ TENSION — Semantic similarity over keyword overlap vs v2.3 | contradicts | Semantic similarity over keyword overlap dedup | 0.9 |
| API Integration: Live API behavior supersedes documentation | contradicts | ⚡ TENSION — API Integration: Live API behavior super vs API  | 0.9 |
| ⚡ TENSION — API Integration: Live API behavior super vs API  | contradicts | API Integration: Live API behavior supersedes documentation | 0.9 |
| API Integration: Read docs first, plan before executing | contradicts | ⚡ TENSION — API Integration: Live API behavior super vs API  | 0.9 |
| ⚡ TENSION — API Integration: Live API behavior super vs API  | contradicts | API Integration: Read docs first, plan before executing | 0.9 |
| Hooks fire successfully; encoding sparseness was strategy pr | contradicts | ⚡ TENSION — Hooks fire successfully; encoding sparse vs Diff | 0.9 |
| ⚡ TENSION — Hooks fire successfully; encoding sparse vs Diff | contradicts | Hooks fire successfully; encoding sparseness was strategy pr | 0.9 |
| React hooks rule | contradicts | ⚡ TENSION — React hooks rule vs React Hook Violation: Cannot | 0.9 |
| ⚡ TENSION — React hooks rule vs React Hook Violation: Cannot | contradicts | React hooks rule | 0.9 |
| React Hook Violation: Cannot call useState/useEffect inside  | contradicts | ⚡ TENSION — React hooks rule vs React Hook Violation: Cannot | 0.9 |
| ⚡ TENSION — React hooks rule vs React Hook Violation: Cannot | contradicts | React Hook Violation: Cannot call useState/useEffect inside  | 0.9 |
| Tom: Always add Todo file for personal thoughts; surface whe | contradicts | ⚡ TENSION — Tom: Always add Todo file for personal t vs Adde | 0.9 |
| ⚡ TENSION — Tom: Always add Todo file for personal t vs Adde | contradicts | Tom: Always add Todo file for personal thoughts; surface whe | 0.9 |
| Added to Tom.md: always plan before executing. Tom discusses | contradicts | ⚡ TENSION — Tom: Always add Todo file for personal t vs Adde | 0.9 |
| ⚡ TENSION — Tom: Always add Todo file for personal t vs Adde | contradicts | Added to Tom.md: always plan before executing. Tom discusses | 0.9 |
| Moderation model: AI moderates first, adds comments+auto-sta | contradicts | ⚡ TENSION — Moderation model: AI moderates first, ad vs Mode | 0.9 |
| ⚡ TENSION — Moderation model: AI moderates first, ad vs Mode | contradicts | Moderation model: AI moderates first, adds comments+auto-sta | 0.9 |
| Moderation: AI pre-screen (risk score, flags, IAB brand safe | contradicts | ⚡ TENSION — Moderation model: AI moderates first, ad vs Mode | 0.9 |
| ⚡ TENSION — Moderation model: AI moderates first, ad vs Mode | contradicts | Moderation: AI pre-screen (risk score, flags, IAB brand safe | 0.9 |
| ⚡ TENSION — Creative archetypes: infinite use cases  vs Glo  | contradicts | Creative archetypes: infinite use cases (marriage proposals, | 0.9 |
| Creative archetypes: infinite use cases (marriage proposals, | contradicts | ⚡ TENSION — Creative archetypes: infinite use cases  vs Glo  | 0.9 |
| ⚡ TENSION — Creative archetypes: infinite use cases  vs Glo  | contradicts | Glo Creative Intelligence: LLM as Creative Director, not fix | 0.9 |
| Glo Creative Intelligence: LLM as Creative Director, not fix | contradicts | ⚡ TENSION — Creative archetypes: infinite use cases  vs Glo  | 0.9 |
| Lesson: UI regression from Claude suggesting layout changes  | corrected_by | [o_glo] Rule: budget_order | 1.0 |
| [o_glo] Rule: budget_order | corrected_by | Lesson: UI regression from Claude suggesting layout changes  | 1.0 |
| Rule: media mockup must show creative ON publisher media wit | depends_on | [o_glo] Rule: budget_order | 0.9 |
| [o_glo] Rule: budget_order | depends_on | Rule: media mockup must show creative ON publisher media wit | 0.9 |
| NanoBanana API: single image only despite docs showing array | depends_on | Video prompt rewritten: Hook→Showcase→CTA structure, industr | 0.9 |
| Video prompt rewritten: Hook→Showcase→CTA structure, industr | depends_on | NanoBanana API: single image only despite docs showing array | 0.9 |
| resolve-brain-db.sh: shared DB resolver sourced by all hooks | depends_on | Brain feedback loop: 13 hooks form a closed perception-actio | 0.8 |
| Brain feedback loop: 13 hooks form a closed perception-actio | depends_on | resolve-brain-db.sh: shared DB resolver sourced by all hooks | 0.8 |
| [o_antifraud] Properties | describes | Mobile Capture & Anti-Fraud — Prevent fake logins/bots. Paym | 0.7 |
| Mobile Capture & Anti-Fraud — Prevent fake logins/bots. Paym | describes | [o_antifraud] Properties | 0.7 |
| GLO Brightness — Pricing tiers: Well($30) Bright($50) Shine( | describes | [o_brightness] Properties | 0.7 |
| [o_brightness] Properties | describes | GLO Brightness — Pricing tiers: Well($30) Bright($50) Shine( | 0.7 |
| Glo Credits — 1:1 USD. Wallet via Stripe customer balance. B | describes | [o_credits] Properties | 0.8800000000000001 |
| [o_credits] Properties | describes | Glo Credits — 1:1 USD. Wallet via Stripe customer balance. B | 0.8800000000000001 |
| Email/Notification System — Activation progress, performance | describes | [o_emailsys] Properties | 0.74 |
| [o_emailsys] Properties | describes | Email/Notification System — Activation progress, performance | 0.74 |
| [o_exco] Properties | describes | EX.CO — End-to-end video platform for publishers: CMS, ad se | 0.76 |
| EX.CO — End-to-end video platform for publishers: CMS, ad se | describes | [o_exco] Properties | 0.76 |
| [o_glo] Properties | describes | Glo.io — Self-serve ad platform on EX.CO ad server. Anyone b | 0.72 |
| Glo.io — Self-serve ad platform on EX.CO ad server. Anyone b | describes | [o_glo] Properties | 0.72 |
| Glo Lifecycle — State machine: Draft→Pending Review→Active→C | describes | [o_lifecycle] Properties | 0.76 |
| [o_lifecycle] Properties | describes | Glo Lifecycle — State machine: Draft→Pending Review→Active→C | 0.76 |
| Moderation System — Two-layer: AI pre-screen (risk score, fl | describes | [o_moderation] Properties | 0.76 |
| [o_moderation] Properties | describes | Moderation System — Two-layer: AI pre-screen (risk score, fl | 0.76 |
| [o_myglos] Properties | describes | My Glos Dashboard — User home. Cards per Glo: thumbnail, sta | 0.7 |
| My Glos Dashboard — User home. Cards per Glo: thumbnail, sta | describes | [o_myglos] Properties | 0.7 |
| Sample page link from EX.CO can be included in emails — real | dream_observation | Dream connection: "Sample page link from EX.CO can be includ | 0.3 |
| Dream connection: "Sample page link from EX.CO can be includ | dream_observation | Sample page link from EX.CO can be included in emails — real | 0.3 |
| Email/Notification System — Activation progress, performance | dream_observation | Dream connection: "Email/Notification System — Activation pr | 0.4400000000000001 |
| Dream connection: "Email/Notification System — Activation pr | dream_observation | Email/Notification System — Activation progress, performance | 0.4400000000000001 |
| Creative strategy: AI video gen is NOT the moat. Buy/integra | dream_observation | Dream connection: "Email/Notification System — Activation pr | 0.7800000000000004 |
| Dream connection: "Email/Notification System — Activation pr | dream_observation | Creative strategy: AI video gen is NOT the moat. Buy/integra | 0.7200000000000003 |
| Dream connection: "Magnite — US adtech. Closest comparable t | dream_observation | Magnite — US adtech. Closest comparable to Geniee (JP). Earl | 0.3 |
| Magnite — US adtech. Closest comparable to Geniee (JP). Earl | dream_observation | Dream connection: "Magnite — US adtech. Closest comparable t | 0.3 |
| Dream connection: "Magnite — US adtech. Closest comparable t | dream_observation | [ctx:glo-platform] Glo.io — Self-Serve Advertising Platform | 0.3 |
| [ctx:glo-platform] Glo.io — Self-Serve Advertising Platform | dream_observation | Dream connection: "Magnite — US adtech. Closest comparable t | 0.3 |
| Frontend polling persists stale job IDs after server restart | dream_observation | Dream connection: "Frontend polling persists stale job IDs a | 0.3 |
| Dream connection: "Frontend polling persists stale job IDs a | dream_observation | Frontend polling persists stale job IDs after server restart | 0.3 |
| Moderation: AI pre-screen (risk score, flags, IAB brand safe | dreamed_from | Dream: Moderation: AI pre-screen (risk score, f ↔ [o_myglos] | 0.3 |
| Dream: Moderation: AI pre-screen (risk score, f ↔ [o_myglos] | dreamed_from | Moderation: AI pre-screen (risk score, flags, IAB brand safe | 0.3 |
| [o_glo] Web app (PWA), not native iOS — avoids Apple's 30% i | dreamed_from | Dream: Web app (PWA) not native iOS — avoids Ap ↔ Reject flo | 0.3 |
| Reject flow: predefined categories + optional moderator note | dreamed_from | Dream: Web app (PWA) not native iOS — avoids Ap ↔ Reject flo | 0.3 |
| Dream: Web app (PWA) not native iOS — avoids Ap ↔ Reject flo | dreamed_from | [o_glo] Web app (PWA), not native iOS — avoids Apple's 30% i | 0.3 |
| Dream: Web app (PWA) not native iOS — avoids Ap ↔ Reject flo | dreamed_from | Reject flow: predefined categories + optional moderator note | 0.3 |
| Dream: Graph bridging > embeddings for emergent ↔ Glo/EX.CO  | dreamed_from | Graph bridging > embeddings for emergent discovery — Toms ar | 0.3 |
| Graph bridging > embeddings for emergent discovery — Toms ar | dreamed_from | Dream: Graph bridging > embeddings for emergent ↔ Glo/EX.CO  | 0.3 |
| EX.CO — End-to-end video platform for publishers: CMS, ad se | dreamed_from | Dream: EX.CO — End-to-end video platform for pu ↔ Contextual | 0.3 |
| Dream: EX.CO — End-to-end video platform for pu ↔ Contextual | dreamed_from | EX.CO — End-to-end video platform for publishers: CMS, ad se | 0.3 |
| WDIV Local 4 / Graham Media — Detroit CTV news. Demo use cas | dreamed_from | Dream: WDIV Local 4 / Graham Media — Detroit CT ↔ Glo Number | 0.3 |
| Dream: WDIV Local 4 / Graham Media — Detroit CT ↔ Glo Number | dreamed_from | WDIV Local 4 / Graham Media — Detroit CTV news. Demo use cas | 0.3 |
| Reject flow: predefined categories + optional moderator note | dreamed_from | Dream: Creative strategy: AI video gen is NOT t ↔ Reject flo | 0.3 |
| Dream: Creative strategy: AI video gen is NOT t ↔ Reject flo | dreamed_from | Reject flow: predefined categories + optional moderator note | 0.3 |
| Anti-fraud concern: fake Google logins and bots on mobile. P | dreamed_from | Dream: [stm:s49] LOCKED: Aspect ratio selection ↔ Anti-fraud | 0.3 |
| Dream: [stm:s49] LOCKED: Aspect ratio selection ↔ Anti-fraud | dreamed_from | Anti-fraud concern: fake Google logins and bots on mobile. P | 0.3 |
| Sample page link from EX.CO can be included in emails — real | dreamed_from | Dream: Sample page link from EX.CO can be inclu ↔ Post-compa | 0.3 |
| Dream: Sample page link from EX.CO can be inclu ↔ Post-compa | dreamed_from | Sample page link from EX.CO can be included in emails — real | 0.3 |
| Dream: Sample page link from EX.CO can be inclu ↔ Post-compa | dreamed_from | Post-compaction session continuation works — summary + conte | 0.3 |
| Post-compaction session continuation works — summary + conte | dreamed_from | Dream: Sample page link from EX.CO can be inclu ↔ Post-compa | 0.3 |
| Tier pricing: 1.4x markup multiplier on publisher CPM | dreamed_from | Dream: Tier pricing: 1.4x markup multiplier on  ↔ Brain serv | 0.46000000000000013 |
| Dream: Tier pricing: 1.4x markup multiplier on  ↔ Brain serv | dreamed_from | Tier pricing: 1.4x markup multiplier on publisher CPM | 0.4400000000000001 |
| Dream: Email/Notification System — Activation p ↔ Creative s | dreamed_from | Email/Notification System — Activation progress, performance | 0.4400000000000001 |
| Email/Notification System — Activation progress, performance | dreamed_from | Dream: Email/Notification System — Activation p ↔ Creative s | 0.4400000000000001 |
| Email/Notification System — Activation progress, performance | dreamed_from | Dream: Open: Embedding strategy for semantic re ↔ Email/Noti | 0.4400000000000001 |
| Dream: Open: Embedding strategy for semantic re ↔ Email/Noti | dreamed_from | Email/Notification System — Activation progress, performance | 0.4400000000000001 |
| 🧠 Claude Session Log — Reset #3 | dreamed_from | Magnite — US adtech. Closest comparable to Geniee (JP). Earl | 0.32 |
| Magnite — US adtech. Closest comparable to Geniee (JP). Earl | dreamed_from | 🧠 Claude Session Log — Reset #3 | 0.32 |
| Dream: [ltm:l2] Flywheel: unfilled inventory→ho ↔ [stm:s18]  | dreamed_from | Magnite — US adtech. Closest comparable to Geniee (JP). Earl | 0.9 |
| Magnite — US adtech. Closest comparable to Geniee (JP). Earl | dreamed_from | Dream: [ltm:l2] Flywheel: unfilled inventory→ho ↔ [stm:s18]  | 0.9 |
| Session Log — Reset #4 | dreamed_from | Dream connection: "Sample page link from EX.CO can be includ | 0.3 |
| Dream connection: "Sample page link from EX.CO can be includ | dreamed_from | Session Log — Reset #4 | 0.3 |
| Upload tab: social + Shopify coming soon | dreamed_from | Dream: Web app (PWA) not native iOS — avoids Ap ↔ Reject flo | 0.3 |
| Dream: Web app (PWA) not native iOS — avoids Ap ↔ Reject flo | dreamed_from | Upload tab: social + Shopify coming soon | 0.3 |
| [stm:s45] CRITICAL TOM FEEDBACK (given 3x — DO NOT REVERT):  | dreamed_from | Dream connection: "Email/Notification System — Activation pr | 0.3 |
| Dream connection: "Email/Notification System — Activation pr | dreamed_from | [stm:s45] CRITICAL TOM FEEDBACK (given 3x — DO NOT REVERT):  | 0.3 |
| Glo Creative Intelligence: LLM as Creative Director, not fix | dreamed_from | Triage before acting — not every sub-task needs execution | 0.3 |
| Triage before acting — not every sub-task needs execution | dreamed_from | Glo Creative Intelligence: LLM as Creative Director, not fix | 0.3 |
| Glo: NanoBanana adapter fixed—Bearer + .php + singular image | dreamed_from | Future: Centralized encoding philosophy — what the brain mem | 0.3 |
| Future: Centralized encoding philosophy — what the brain mem | dreamed_from | Glo: NanoBanana adapter fixed—Bearer + .php + singular image | 0.3 |
| Dream: Credits balance shown wherever makes sen ↔ [o_brightn | dreamed_from | Credits balance shown wherever makes sense: My Glos dashboar | 0.3 |
| Credits balance shown wherever makes sense: My Glos dashboar | dreamed_from | Dream: Credits balance shown wherever makes sen ↔ [o_brightn | 0.3 |
| Dream: Credits balance shown wherever makes sen ↔ [o_brightn | dreamed_from | [o_brightness] Dynamic pricing: media prices change daily (e | 0.3 |
| [o_brightness] Dynamic pricing: media prices change daily (e | dreamed_from | Dream: Credits balance shown wherever makes sen ↔ [o_brightn | 0.3 |
| Dream: Magnite — US adtech. Closest comparable  ↔ [ctx:glo-p | dreamed_from | Magnite — US adtech. Closest comparable to Geniee (JP). Earl | 0.3 |
| Magnite — US adtech. Closest comparable to Geniee (JP). Earl | dreamed_from | Dream: Magnite — US adtech. Closest comparable  ↔ [ctx:glo-p | 0.3 |
| Dream: Magnite — US adtech. Closest comparable  ↔ [ctx:glo-p | dreamed_from | [ctx:glo-platform] Glo.io — Self-Serve Advertising Platform | 0.3 |
| [ctx:glo-platform] Glo.io — Self-Serve Advertising Platform | dreamed_from | Dream: Magnite — US adtech. Closest comparable  ↔ [ctx:glo-p | 0.3 |
| Dream: EX.CO — Video platform and ad server ↔ Cluster formin | dreamed_from | EX.CO — Video platform and ad server | 0.3 |
| EX.CO — Video platform and ad server | dreamed_from | Dream: EX.CO — Video platform and ad server ↔ Cluster formin | 0.3 |
| Dream: Dream: [o_credits] Pause refunds credits ↔ [stm:s35]  | dreamed_from | Dream: [o_credits] Pause refunds credits to wal ↔ [stm:s7] F | 0.3 |
| Dream: [o_credits] Pause refunds credits to wal ↔ [stm:s7] F | dreamed_from | Dream: Dream: [o_credits] Pause refunds credits ↔ [stm:s35]  | 0.3 |
| Dream: Brain Evolution Roadmap — 3 prioritized  ↔ Open: Embe | dreamed_from | Brain Evolution Roadmap — 3 prioritized architecture changes | 0.3 |
| Brain Evolution Roadmap — 3 prioritized architecture changes | dreamed_from | Dream: Brain Evolution Roadmap — 3 prioritized  ↔ Open: Embe | 0.3 |
| Dream: Brain Evolution Roadmap — 3 prioritized  ↔ Open: Embe | dreamed_from | Open: Embedding strategy for semantic recall — explore optio | 0.3 |
| Open: Embedding strategy for semantic recall — explore optio | dreamed_from | Dream: Brain Evolution Roadmap — 3 prioritized  ↔ Open: Embe | 0.3 |
| Dream: [period:2026-03:p5] Fox Corp penetration ↔ Glo.io — S | dreamed_from | Glo.io — Self-serve ad platform on EX.CO ad server. Anyone b | 0.3 |
| Glo.io — Self-serve ad platform on EX.CO ad server. Anyone b | dreamed_from | Dream: [period:2026-03:p5] Fox Corp penetration ↔ Glo.io — S | 0.3 |
| Time-dilation decay: decay_active_rate and decay_idle_rate i | dreamed_from | Dream: Time-dilation decay: decay_active_rate a ↔ Tom: Plan- | 0.3 |
| Dream: Time-dilation decay: decay_active_rate a ↔ Tom: Plan- | dreamed_from | Time-dilation decay: decay_active_rate and decay_idle_rate i | 0.3 |
| Tom: Plan-first approach for bigger questions | dreamed_from | Dream: Time-dilation decay: decay_active_rate a ↔ Tom: Plan- | 0.3 |
| Dream: Time-dilation decay: decay_active_rate a ↔ Tom: Plan- | dreamed_from | Tom: Plan-first approach for bigger questions | 0.3 |
| Frontend polling persists stale job IDs after server restart | dreamed_from | Dream: Frontend polling persists stale job IDs  ↔ 🌱 ASPIRATI | 0.3 |
| Dream: Frontend polling persists stale job IDs  ↔ 🌱 ASPIRATI | dreamed_from | Frontend polling persists stale job IDs after server restart | 0.3 |
| Dream: Rule: ask for confirmation before manipu ↔ Glo Creati | dreamed_from | Rule: ask for confirmation before manipulating data to affec | 0.3 |
| Rule: ask for confirmation before manipulating data to affec | dreamed_from | Dream: Rule: ask for confirmation before manipu ↔ Glo Creati | 0.3 |
| Dream: Rule: ask for confirmation before manipu ↔ Glo Creati | dreamed_from | Glo Creative Intelligence: LLM as Creative Director, not fix | 0.3 |
| Glo Creative Intelligence: LLM as Creative Director, not fix | dreamed_from | Dream: Rule: ask for confirmation before manipu ↔ Glo Creati | 0.3 |
| Dream: Moderation: AI pre-screen (risk score, f ↔ [o_myglos] | emergent_bridge | Glo.io — Self-serve ad platform on EX.CO ad server. Anyone b | 0.15 |
| Glo.io — Self-serve ad platform on EX.CO ad server. Anyone b | emergent_bridge | Dream: Moderation: AI pre-screen (risk score, f ↔ [o_myglos] | 0.15 |
| Dream: [o_glo] Flywheel: unfilled inventory→hou ↔ SSO→paymen | emergent_bridge | Glo.io — Self-serve ad platform on EX.CO ad server. Anyone b | 0.15 |
| Glo.io — Self-serve ad platform on EX.CO ad server. Anyone b | emergent_bridge | Dream: [o_glo] Flywheel: unfilled inventory→hou ↔ SSO→paymen | 0.15 |
| Adweek — Media industry news site. Demo use case #2 — online | emergent_bridge | WDIV Local 4 / Graham Media — Detroit CTV news. Demo use cas | 0.15 |
| WDIV Local 4 / Graham Media — Detroit CTV news. Demo use cas | emergent_bridge | Adweek — Media industry news site. Demo use case #2 — online | 0.15 |
| Adweek — Media industry news site. Demo use case #2 — online | emergent_bridge | [todo:t4] Build formal business case for EX.CO board | 0.15 |
| [todo:t4] Build formal business case for EX.CO board | emergent_bridge | Adweek — Media industry news site. Demo use case #2 — online | 0.15 |
| EX.CO: full end-to-end video platform for online publishers  | emergent_bridge | [todo:t4] Build formal business case for EX.CO board | 0.15 |
| [todo:t4] Build formal business case for EX.CO board | emergent_bridge | EX.CO: full end-to-end video platform for online publishers  | 0.15 |
| Glo Credits — 1:1 USD. Wallet via Stripe customer balance. B | emergent_bridge | Dream: Web app (PWA) not native iOS — avoids Ap ↔ Reject flo | 0.15 |
| Dream: Web app (PWA) not native iOS — avoids Ap ↔ Reject flo | emergent_bridge | Glo Credits — 1:1 USD. Wallet via Stripe customer balance. B | 0.15 |
| Glo.io — Self-serve ad platform on EX.CO ad server. Anyone b | emergent_bridge | Dream: EX.CO — End-to-end video platform for pu ↔ Contextual | 0.15 |
| Dream: EX.CO — End-to-end video platform for pu ↔ Contextual | emergent_bridge | Glo.io — Self-serve ad platform on EX.CO ad server. Anyone b | 0.15 |
| Dream: WDIV Local 4 / Graham Media — Detroit CT ↔ Glo Number | emergent_bridge | EX.CO — End-to-end video platform for publishers: CMS, ad se | 0.15 |
| Dream: WDIV Local 4 / Graham Media — Detroit CT ↔ Glo Number | emergent_bridge | Glo.io — Self-serve ad platform on EX.CO ad server. Anyone b | 0.15 |
| Glo.io — Self-serve ad platform on EX.CO ad server. Anyone b | emergent_bridge | Dream: WDIV Local 4 / Graham Media — Detroit CT ↔ Glo Number | 0.15 |
| EX.CO — End-to-end video platform for publishers: CMS, ad se | emergent_bridge | Dream: WDIV Local 4 / Graham Media — Detroit CT ↔ Glo Number | 0.15 |
| Dream: Creative strategy: AI video gen is NOT t ↔ Reject flo | emergent_bridge | Glo.io — Self-serve ad platform on EX.CO ad server. Anyone b | 0.15 |
| Glo.io — Self-serve ad platform on EX.CO ad server. Anyone b | emergent_bridge | Dream: Creative strategy: AI video gen is NOT t ↔ Reject flo | 0.15 |
| Dream: Component: My Glos Dashboard ↔ SSO→payment linking: G | emergent_bridge | Glo.io — Self-serve ad platform on EX.CO ad server. Anyone b | 0.15 |
| Glo.io — Self-serve ad platform on EX.CO ad server. Anyone b | emergent_bridge | Dream: Component: My Glos Dashboard ↔ SSO→payment linking: G | 0.15 |
| Dream: Glo Credits — 1:1 USD. Wallet via Stripe ↔ Moderation | emergent_bridge | Glo.io — Self-serve ad platform on EX.CO ad server. Anyone b | 0.15 |
| Glo.io — Self-serve ad platform on EX.CO ad server. Anyone b | emergent_bridge | Dream: Glo Credits — 1:1 USD. Wallet via Stripe ↔ Moderation | 0.15 |
| Dream: [o_myglos] Multiple simultaneous Glos pe ↔ Mobile Cap | emergent_bridge | Glo.io — Self-serve ad platform on EX.CO ad server. Anyone b | 0.15 |
| Glo.io — Self-serve ad platform on EX.CO ad server. Anyone b | emergent_bridge | Dream: [o_myglos] Multiple simultaneous Glos pe ↔ Mobile Cap | 0.15 |
| Dream: Web app (PWA) not native iOS — avoids Ap ↔ [o_glo] 3  | emergent_bridge | Glo.io — Self-serve ad platform on EX.CO ad server. Anyone b | 0.15 |
| Glo.io — Self-serve ad platform on EX.CO ad server. Anyone b | emergent_bridge | Dream: Web app (PWA) not native iOS — avoids Ap ↔ [o_glo] 3  | 0.15 |
| Email/Notification System — Activation progress, performance | emergent_bridge | Dream: WDIV Local 4 / Graham Media — Detroit CT ↔ Glo Number | 0.15 |
| Email/Notification System — Activation progress, performance | emergent_bridge | Two rule layers: GLO general rules + publisher-specific conf | 0.15 |
| Two rule layers: GLO general rules + publisher-specific conf | emergent_bridge | Email/Notification System — Activation progress, performance | 0.15 |
| Dream: WDIV Local 4 / Graham Media — Detroit CT ↔ Glo Number | emergent_bridge | Email/Notification System — Activation progress, performance | 0.15 |
| Mobile Capture & Anti-Fraud — Prevent fake logins/bots. Paym | emergent_bridge | EX.CO — End-to-end video platform for publishers: CMS, ad se | 0.15 |
| EX.CO — End-to-end video platform for publishers: CMS, ad se | emergent_bridge | Mobile Capture & Anti-Fraud — Prevent fake logins/bots. Paym | 0.15 |
| Dream: Email/Notification System — Activation p ↔ Creative s | emergent_bridge | Glo.io — Self-serve ad platform on EX.CO ad server. Anyone b | 0.15 |
| Glo.io — Self-serve ad platform on EX.CO ad server. Anyone b | emergent_bridge | Dream: Email/Notification System — Activation p ↔ Creative s | 0.15 |
| Cloudinary — Video transcoding, AI smart cropping across asp | emergent_bridge | Dream: Creative strategy: AI video gen is NOT t ↔ Reject flo | 0.15 |
| Dream: Creative strategy: AI video gen is NOT t ↔ Reject flo | emergent_bridge | Cloudinary — Video transcoding, AI smart cropping across asp | 0.15 |
| [o_myglos] Multiple simultaneous Glos per user — yes. | emergent_bridge | Moderation: AI pre-screen (risk score, flags, IAB brand safe | 0.15 |
| Moderation: AI pre-screen (risk score, flags, IAB brand safe | emergent_bridge | [o_myglos] Multiple simultaneous Glos per user — yes. | 0.15 |
| Glo/EX.CO boundary | emergent_bridge | Graph bridging > embeddings for emergent discovery — Toms ar | 0.15 |
| Graph bridging > embeddings for emergent discovery — Toms ar | emergent_bridge | Glo/EX.CO boundary | 0.15 |
| Contextual intent: infer creative direction from WHO (media  | emergent_bridge | EX.CO — End-to-end video platform for publishers: CMS, ad se | 0.15 |
| EX.CO — End-to-end video platform for publishers: CMS, ad se | emergent_bridge | Contextual intent: infer creative direction from WHO (media  | 0.15 |
| Glo Numbers — Analytics detail per Glo. Views/day graph, cli | emergent_bridge | WDIV Local 4 / Graham Media — Detroit CTV news. Demo use cas | 0.15 |
| WDIV Local 4 / Graham Media — Detroit CTV news. Demo use cas | emergent_bridge | Glo Numbers — Analytics detail per Glo. Views/day graph, cli | 0.15 |
| Creative strategy: AI video gen is NOT the moat. Buy/integra | emergent_bridge | Reject flow: predefined categories + optional moderator note | 0.15 |
| Reject flow: predefined categories + optional moderator note | emergent_bridge | Creative strategy: AI video gen is NOT the moat. Buy/integra | 0.15 |
| Dream: Brain Evolution Roadmap — 3 prioritized  ↔ Open: Embe | emergent_bridge | Upgrade: Move pre-compaction session note from memory rule t | 0.15 |
| Upgrade: Move pre-compaction session note from memory rule t | emergent_bridge | Dream: Brain Evolution Roadmap — 3 prioritized  ↔ Open: Embe | 0.15 |
| Dream: Brain Evolution Roadmap — 3 prioritized  ↔ Open: Embe | emergent_bridge | Rule: every tmemory code/architecture change must be stored  | 0.15 |
| Rule: every tmemory code/architecture change must be stored  | emergent_bridge | Dream: Brain Evolution Roadmap — 3 prioritized  ↔ Open: Embe | 0.15 |
| Dream: [period:2026-03:p5] Fox Corp penetration ↔ Glo.io — S | emergent_bridge | EX.CO — End-to-end video platform for publishers: CMS, ad se | 0.15 |
| EX.CO — End-to-end video platform for publishers: CMS, ad se | emergent_bridge | Dream: [period:2026-03:p5] Fox Corp penetration ↔ Glo.io — S | 0.15 |
| Dream: Time-dilation decay: decay_active_rate a ↔ Tom: Plan- | emergent_bridge | Future: Centralized encoding philosophy — what the brain mem | 0.15 |
| Future: Centralized encoding philosophy — what the brain mem | emergent_bridge | Dream: Time-dilation decay: decay_active_rate a ↔ Tom: Plan- | 0.15 |
| Dream: Time-dilation decay: decay_active_rate a ↔ Tom: Plan- | emergent_bridge | v4 Phase 0: Map existing brain codebase before building new  | 0.15 |
| v4 Phase 0: Map existing brain codebase before building new  | emergent_bridge | Dream: Time-dilation decay: decay_active_rate a ↔ Tom: Plan- | 0.15 |
| Confidence recalibration pipeline | emergent_bridge | Python mixin inheritance pattern for monolith decomposition | 0.15 |
| Python mixin inheritance pattern for monolith decomposition | emergent_bridge | Confidence recalibration pipeline | 0.15 |
| Confidence recalibration pipeline | emergent_bridge | Convention: mixin file structure — imports, class, docstring | 0.15 |
| Convention: mixin file structure — imports, class, docstring | emergent_bridge | Confidence recalibration pipeline | 0.15 |
| Upgrade: Move pre-compaction session note from memory rule t | emergent_bridge | Slow test: test:TestBootHook.test_boot_idempotent — avg 2921 | 0.15 |
| Slow test: test:TestBootHook.test_boot_idempotent — avg 2921 | emergent_bridge | Upgrade: Move pre-compaction session note from memory rule t | 0.15 |
| Upgrade: Move pre-compaction session note from memory rule t | emergent_bridge | Clean test run: 124 tests passed (56.7s) | 0.15 |
| Clean test run: 124 tests passed (56.7s) | emergent_bridge | Upgrade: Move pre-compaction session note from memory rule t | 0.15 |
| Upgrade: Move pre-compaction session note from memory rule t | emergent_bridge | DAL pattern: mixin files must not access logs_conn or brain_ | 0.15 |
| DAL pattern: mixin files must not access logs_conn or brain_ | emergent_bridge | Upgrade: Move pre-compaction session note from memory rule t | 0.15 |
| Upgrade: Move pre-compaction session note from memory rule t | emergent_bridge | DAL migration baseline: 28 logs_conn + 5 brain_meta direct a | 0.15 |
| DAL migration baseline: 28 logs_conn + 5 brain_meta direct a | emergent_bridge | Upgrade: Move pre-compaction session note from memory rule t | 0.15 |
| Upgrade: Move pre-compaction session note from memory rule t | emergent_bridge | 5 log tables lack DAL methods: tuning_log, eval_snapshots, s | 0.15 |
| 5 log tables lack DAL methods: tuning_log, eval_snapshots, s | emergent_bridge | Upgrade: Move pre-compaction session note from memory rule t | 0.15 |
| Upgrade: Move pre-compaction session note from memory rule t | emergent_bridge | Lesson: worktree cleanup deletes CWD — session becomes non-f | 0.15 |
| Lesson: worktree cleanup deletes CWD — session becomes non-f | emergent_bridge | Upgrade: Move pre-compaction session note from memory rule t | 0.15 |
| Upgrade: Move pre-compaction session note from memory rule t | emergent_bridge | Rule: NEVER delete a git worktree without alerting the user  | 0.15 |
| Rule: NEVER delete a git worktree without alerting the user  | emergent_bridge | Upgrade: Move pre-compaction session note from memory rule t | 0.15 |
| Upgrade: Move pre-compaction session note from memory rule t | emergent_bridge | Lesson: CLAUDE_PLUGIN_ROOT resolves to worktree path — hooks | 0.15 |
| Lesson: CLAUDE_PLUGIN_ROOT resolves to worktree path — hooks | emergent_bridge | Upgrade: Move pre-compaction session note from memory rule t | 0.15 |
| Upgrade: Move pre-compaction session note from memory rule t | emergent_bridge | Claude Code worktree mechanism: isolated branches for safe p | 0.15 |
| Claude Code worktree mechanism: isolated branches for safe p | emergent_bridge | Upgrade: Move pre-compaction session note from memory rule t | 0.15 |
| Repeating errors in skill get high priority | emergent_bridge | Build official session handoff mechanism into brain | 0.15 |
| Build official session handoff mechanism into brain | emergent_bridge | Repeating errors in skill get high priority | 0.15 |
| Upgrade: Move pre-compaction session note from memory rule t | emergent_bridge | Cluster forming: "Build official session handoff mechanism i | 0.15 |
| Cluster forming: "Build official session handoff mechanism i | emergent_bridge | Upgrade: Move pre-compaction session note from memory rule t | 0.15 |
| Upgrade: Move pre-compaction session note from memory rule t | emergent_bridge | Cluster forming: "Build official session handoff mechanism i | 0.15 |
| Cluster forming: "Build official session handoff mechanism i | emergent_bridge | Upgrade: Move pre-compaction session note from memory rule t | 0.15 |
| Correction: tmemory plugin was GLO-specific, must be project | emergent_bridge | Brain server died again during v11 implementation — platform | 0.15 |
| Brain server died again during v11 implementation — platform | emergent_bridge | Correction: tmemory plugin was GLO-specific, must be project | 0.15 |
| [ctx:glo-platform] Glo.io — Self-Serve Advertising Platform | emergent_bridge | Brain server died again during v11 implementation — platform | 0.15 |
| Brain server died again during v11 implementation — platform | emergent_bridge | [ctx:glo-platform] Glo.io — Self-Serve Advertising Platform | 0.15 |
| Upgrade: Move pre-compaction session note from memory rule t | emergent_bridge | Dream connection: "Brain Evolution Roadmap — 3 prioritized a | 0.15 |
| Dream connection: "Brain Evolution Roadmap — 3 prioritized a | emergent_bridge | Upgrade: Move pre-compaction session note from memory rule t | 0.15 |
| [ctx:glo-platform] Glo.io — Self-Serve Advertising Platform | emergent_bridge | Magnite — US adtech. Closest comparable to Geniee (JP). Earl | 0.15 |
| Magnite — US adtech. Closest comparable to Geniee (JP). Earl | emergent_bridge | [ctx:glo-platform] Glo.io — Self-Serve Advertising Platform | 0.15 |
| Cluster forming: "Glo Lifecycle — State machine: Draft→Pendi | emergent_bridge | EX.CO — Video platform and ad server | 0.15 |
| EX.CO — Video platform and ad server | emergent_bridge | Cluster forming: "Glo Lifecycle — State machine: Draft→Pendi | 0.15 |
| Tom: Plan-first approach for bigger questions | emergent_bridge | Time-dilation decay: decay_active_rate and decay_idle_rate i | 0.15 |
| Time-dilation decay: decay_active_rate and decay_idle_rate i | emergent_bridge | Tom: Plan-first approach for bigger questions | 0.15 |
| Upgrade: Move pre-compaction session note from memory rule t | emergent_bridge | ⚡ TENSION — Tmemory: Boot script includes npm instal vs Tmem | 0.15 |
| ⚡ TENSION — Tmemory: Boot script includes npm instal vs Tmem | emergent_bridge | Upgrade: Move pre-compaction session note from memory rule t | 0.15 |
| Upgrade: Move pre-compaction session note from memory rule t | emergent_bridge | ⚡ TENSION — Bridge candidates: Require 2+ shared nei vs Brid | 0.15 |
| ⚡ TENSION — Bridge candidates: Require 2+ shared nei vs Brid | emergent_bridge | Upgrade: Move pre-compaction session note from memory rule t | 0.15 |
| Upgrade: Move pre-compaction session note from memory rule t | emergent_bridge | 📊 PATTERN — Corrections cluster: [bug] remember() INSERT was | 0.15 |
| 📊 PATTERN — Corrections cluster: [bug] remember() INSERT was | emergent_bridge | Upgrade: Move pre-compaction session note from memory rule t | 0.15 |
| Upgrade: Move pre-compaction session note from memory rule t | emergent_bridge | 📊 PATTERN — Corrections cluster: [bug] SKILL.md had HTTP/cur | 0.15 |
| 📊 PATTERN — Corrections cluster: [bug] SKILL.md had HTTP/cur | emergent_bridge | Upgrade: Move pre-compaction session note from memory rule t | 0.15 |
| Upgrade: Move pre-compaction session note from memory rule t | emergent_bridge | 📊 PATTERN — Always used together: 'Rule: ask for confirmatio | 0.15 |
| 📊 PATTERN — Always used together: 'Rule: ask for confirmatio | emergent_bridge | Upgrade: Move pre-compaction session note from memory rule t | 0.15 |
| Upgrade: Move pre-compaction session note from memory rule t | emergent_bridge | 📊 PATTERN — Retention issue: task nodes (28 archived vs 12 a | 0.15 |
| 📊 PATTERN — Retention issue: task nodes (28 archived vs 12 a | emergent_bridge | Upgrade: Move pre-compaction session note from memory rule t | 0.15 |
| Upgrade: Move pre-compaction session note from memory rule t | emergent_bridge | 🔮 HYPOTHESIS — 'v14: Density-based gap detection replaces ti | 0.15 |
| 🔮 HYPOTHESIS — 'v14: Density-based gap detection replaces ti | emergent_bridge | Upgrade: Move pre-compaction session note from memory rule t | 0.15 |
| Upgrade: Move pre-compaction session note from memory rule t | emergent_bridge | 🔮 HYPOTHESIS — 'v15 Architecture: Serverless brain — elimina | 0.15 |
| 🔮 HYPOTHESIS — 'v15 Architecture: Serverless brain — elimina | emergent_bridge | Upgrade: Move pre-compaction session note from memory rule t | 0.15 |
| 🔮 HYPOTHESIS — Dream insight gaining traction: Dream: [o_glo | emergent_bridge | Upgrade: Move pre-compaction session note from memory rule t | 0.15 |
| Upgrade: Move pre-compaction session note from memory rule t | emergent_bridge | 🔮 HYPOTHESIS — Dream insight gaining traction: Dream: [o_glo | 0.15 |
| 🔮 HYPOTHESIS — Dream insight gaining traction: Dream: [o_glo | emergent_bridge | Two layers of brain knowledge: structure (inherited) vs wisd | 0.15 |
| Two layers of brain knowledge: structure (inherited) vs wisd | emergent_bridge | 🔮 HYPOTHESIS — Dream insight gaining traction: Dream: [o_glo | 0.15 |
| 🔮 HYPOTHESIS — Dream insight gaining traction: Dream: Graph  | emergent_bridge | Upgrade: Move pre-compaction session note from memory rule t | 0.15 |
| Upgrade: Move pre-compaction session note from memory rule t | emergent_bridge | 🔮 HYPOTHESIS — Dream insight gaining traction: Dream: Graph  | 0.15 |
| 🔮 HYPOTHESIS — Dream insight gaining traction: Dream: Graph  | emergent_bridge | Two layers of brain knowledge: structure (inherited) vs wisd | 0.15 |
| Two layers of brain knowledge: structure (inherited) vs wisd | emergent_bridge | 🔮 HYPOTHESIS — Dream insight gaining traction: Dream: Graph  | 0.15 |
| Upgrade: Move pre-compaction session note from memory rule t | emergent_bridge | 🔮 HYPOTHESIS — 'Root cause: hooks go silent when brain serve | 0.15 |
| 🔮 HYPOTHESIS — 'Root cause: hooks go silent when brain serve | emergent_bridge | Upgrade: Move pre-compaction session note from memory rule t | 0.15 |
| 🌱 ASPIRATION — Growing energy around Glo (emotion +0.36) | emergent_bridge | Upgrade: Move pre-compaction session note from memory rule t | 0.15 |
| Upgrade: Move pre-compaction session note from memory rule t | emergent_bridge | 🌱 ASPIRATION — Growing energy around Glo (emotion +0.36) | 0.15 |
| 🌱 ASPIRATION — Growing energy around Glo (emotion +0.36) | emergent_bridge | Two layers of brain knowledge: structure (inherited) vs wisd | 0.15 |
| Two layers of brain knowledge: structure (inherited) vs wisd | emergent_bridge | 🌱 ASPIRATION — Growing energy around Glo (emotion +0.36) | 0.15 |
| Upgrade: Move pre-compaction session note from memory rule t | emergent_bridge | 🌱 ASPIRATION — Growing energy around brain (emotion +0.44) | 0.15 |
| 🌱 ASPIRATION — Growing energy around brain (emotion +0.44) | emergent_bridge | Upgrade: Move pre-compaction session note from memory rule t | 0.15 |
| 🌱 ASPIRATION — Repeated energy: [o_glo] Rule: budget_order ( | emergent_bridge | Upgrade: Move pre-compaction session note from memory rule t | 0.15 |
| Upgrade: Move pre-compaction session note from memory rule t | emergent_bridge | 🌱 ASPIRATION — Repeated energy: [o_glo] Rule: budget_order ( | 0.15 |
| 🌱 ASPIRATION — Repeated energy: [o_glo] Rule: budget_order ( | emergent_bridge | Two layers of brain knowledge: structure (inherited) vs wisd | 0.15 |
| Two layers of brain knowledge: structure (inherited) vs wisd | emergent_bridge | 🌱 ASPIRATION — Repeated energy: [o_glo] Rule: budget_order ( | 0.15 |
| Upgrade: Move pre-compaction session note from memory rule t | emergent_bridge | Dream: v15 Architecture: Serverless brain — eli ↔ Tom: mind  | 0.15 |
| Dream: v15 Architecture: Serverless brain — eli ↔ Tom: mind  | emergent_bridge | Upgrade: Move pre-compaction session note from memory rule t | 0.15 |
| Upgrade: Move pre-compaction session note from memory rule t | emergent_bridge | Dream connection: "v15 Architecture: Serverless brain — elim | 0.15 |
| Dream connection: "v15 Architecture: Serverless brain — elim | emergent_bridge | Upgrade: Move pre-compaction session note from memory rule t | 0.15 |
| Upgrade: Move pre-compaction session note from memory rule t | emergent_bridge | Dream: [concept] Recall Pipeline — the full pat ↔ Tom princi | 0.15 |
| Dream: [concept] Recall Pipeline — the full pat ↔ Tom princi | emergent_bridge | Upgrade: Move pre-compaction session note from memory rule t | 0.15 |
| Dream: Rule: ask for confirmation before manipu ↔ Glo Creati | emergent_bridge | v14: Post-compact log reader — extract-session-log.py reads  | 0.15 |
| v14: Post-compact log reader — extract-session-log.py reads  | emergent_bridge | Dream: Rule: ask for confirmation before manipu ↔ Glo Creati | 0.15 |
| Dream: Rule: ask for confirmation before manipu ↔ Glo Creati | emergent_bridge | v15 Architecture: Serverless brain — eliminate HTTP server,  | 0.15 |
| v15 Architecture: Serverless brain — eliminate HTTP server,  | emergent_bridge | Dream: Rule: ask for confirmation before manipu ↔ Glo Creati | 0.15 |
| Brain v4: Self-reflection node types — performance, failure, | emergent_bridge | Divergence: The test integrity hook blocking all test change | 0.15 |
| Divergence: The test integrity hook blocking all test change | emergent_bridge | Brain v4: Self-reflection node types — performance, failure, | 0.15 |
| Upgrade: Move pre-compaction session note from memory rule t | emergent_bridge | Compaction boundary at 2026-03-22T02:05:43Z | 0.15 |
| Compaction boundary at 2026-03-22T02:05:43Z | emergent_bridge | Upgrade: Move pre-compaction session note from memory rule t | 0.15 |
| Upgrade: Move pre-compaction session note from memory rule t | emergent_bridge | Cross-session testing: use current session to verify next se | 0.15 |
| Cross-session testing: use current session to verify next se | emergent_bridge | Upgrade: Move pre-compaction session note from memory rule t | 0.15 |
| Upgrade: Move pre-compaction session note from memory rule t | emergent_bridge | Debug mode: errors must always be visible, verbose output is | 0.15 |
| Debug mode: errors must always be visible, verbose output is | emergent_bridge | Upgrade: Move pre-compaction session note from memory rule t | 0.15 |
| Glo component map — 18 components, build status, key decisio | emergent_bridge | Upgrade: Move pre-compaction session note from memory rule t | 0.15 |
| Upgrade: Move pre-compaction session note from memory rule t | emergent_bridge | Glo component map — 18 components, build status, key decisio | 0.15 |
| Glo component map — 18 components, build status, key decisio | emergent_bridge | Divergence: Claude compresses when encoding to brain — treat | 0.15 |
| Divergence: Claude compresses when encoding to brain — treat | emergent_bridge | Glo component map — 18 components, build status, key decisio | 0.15 |
| Glo project history — research, pivot, build phases (March 2 | emergent_bridge | Upgrade: Move pre-compaction session note from memory rule t | 0.15 |
| Upgrade: Move pre-compaction session note from memory rule t | emergent_bridge | Glo project history — research, pivot, build phases (March 2 | 0.15 |
| Glo project history — research, pivot, build phases (March 2 | emergent_bridge | Divergence: Claude compresses when encoding to brain — treat | 0.15 |
| Divergence: Claude compresses when encoding to brain — treat | emergent_bridge | Glo project history — research, pivot, build phases (March 2 | 0.15 |
| Entities are everything with identity: people, products, com | emergent_bridge | Upgrade: Move pre-compaction session note from memory rule t | 0.15 |
| Upgrade: Move pre-compaction session note from memory rule t | emergent_bridge | Entities are everything with identity: people, products, com | 0.15 |
| Entities are everything with identity: people, products, com | emergent_bridge | Open: Embedding strategy for semantic recall — explore optio | 0.15 |
| Open: Embedding strategy for semantic recall — explore optio | emergent_bridge | Entities are everything with identity: people, products, com | 0.15 |
| Upgrade: Move pre-compaction session note from memory rule t | emergent_bridge | Encoding = everything brain does to capture what Tom is talk | 0.15 |
| Encoding = everything brain does to capture what Tom is talk | emergent_bridge | Upgrade: Move pre-compaction session note from memory rule t | 0.15 |
| Upgrade: Move pre-compaction session note from memory rule t | emergent_bridge | Future: typed edges + graph traversal for entity-aware knowl | 0.15 |
| Future: typed edges + graph traversal for entity-aware knowl | emergent_bridge | Upgrade: Move pre-compaction session note from memory rule t | 0.15 |
| Upgrade: Move pre-compaction session note from memory rule t | emergent_bridge | Three extraction layers: vocabulary, entities, relationships | 0.15 |
| Three extraction layers: vocabulary, entities, relationships | emergent_bridge | Upgrade: Move pre-compaction session note from memory rule t | 0.15 |
| Upgrade: Move pre-compaction session note from memory rule t | emergent_bridge | TEST NODE DELETE ME | 0.15 |
| TEST NODE DELETE ME | emergent_bridge | Upgrade: Move pre-compaction session note from memory rule t | 0.15 |
| Entities are everything with identity: people, products, com | emergent_bridge | Upgrade: Move pre-compaction session note from memory rule t | 0.15 |
| Upgrade: Move pre-compaction session note from memory rule t | emergent_bridge | Entities are everything with identity: people, products, com | 0.15 |
| Entities are everything with identity: people, products, com | emergent_bridge | Open: Embedding strategy for semantic recall — explore optio | 0.15 |
| Open: Embedding strategy for semantic recall — explore optio | emergent_bridge | Entities are everything with identity: people, products, com | 0.15 |
| Upgrade: Move pre-compaction session note from memory rule t | emergent_bridge | Encoding = everything brain does to capture what Tom is talk | 0.15 |
| Encoding = everything brain does to capture what Tom is talk | emergent_bridge | Upgrade: Move pre-compaction session note from memory rule t | 0.15 |
| Upgrade: Move pre-compaction session note from memory rule t | emergent_bridge | Future: typed edges + graph traversal for entity-aware knowl | 0.15 |
| Future: typed edges + graph traversal for entity-aware knowl | emergent_bridge | Upgrade: Move pre-compaction session note from memory rule t | 0.15 |
| Upgrade: Move pre-compaction session note from memory rule t | emergent_bridge | Three extraction layers: vocabulary, entities, relationships | 0.15 |
| Three extraction layers: vocabulary, entities, relationships | emergent_bridge | Upgrade: Move pre-compaction session note from memory rule t | 0.15 |
| Upgrade: Move pre-compaction session note from memory rule t | emergent_bridge | Phase 1 Text Processing Pipeline — 4 integrated systems for  | 0.15 |
| Phase 1 Text Processing Pipeline — 4 integrated systems for  | emergent_bridge | Upgrade: Move pre-compaction session note from memory rule t | 0.15 |
| Upgrade: Move pre-compaction session note from memory rule t | emergent_bridge | Session 2026-03-22: Phase 1 text processing complete + deplo | 0.15 |
| Session 2026-03-22: Phase 1 text processing complete + deplo | emergent_bridge | Upgrade: Move pre-compaction session note from memory rule t | 0.15 |
| Upgrade: Move pre-compaction session note from memory rule t | emergent_bridge | Vocabulary pipeline gap: extraction detects terms but never  | 0.15 |
| Vocabulary pipeline gap: extraction detects terms but never  | emergent_bridge | Upgrade: Move pre-compaction session note from memory rule t | 0.15 |
| Upgrade: Move pre-compaction session note from memory rule t | emergent_bridge | Claude Code context continuation vs compaction — different b | 0.15 |
| Claude Code context continuation vs compaction — different b | emergent_bridge | Upgrade: Move pre-compaction session note from memory rule t | 0.15 |
| Upgrade: Move pre-compaction session note from memory rule t | emergent_bridge | Start next session by reading HANDOFF.md at repo root | 0.15 |
| Start next session by reading HANDOFF.md at repo root | emergent_bridge | Upgrade: Move pre-compaction session note from memory rule t | 0.15 |
| Upgrade: Move pre-compaction session note from memory rule t | emergent_bridge | NEVER modify .claude/settings.json without operator approval | 0.15 |
| NEVER modify .claude/settings.json without operator approval | emergent_bridge | Upgrade: Move pre-compaction session note from memory rule t | 0.15 |
| Upgrade: Move pre-compaction session note from memory rule t | emergent_bridge | Hook output now visible to Claude — consciousness pipeline w | 0.15 |
| Hook output now visible to Claude — consciousness pipeline w | emergent_bridge | Upgrade: Move pre-compaction session note from memory rule t | 0.15 |
| Upgrade: Move pre-compaction session note from memory rule t | emergent_bridge | Recall precision 0% — root cause is missing eval feedback lo | 0.15 |
| Recall precision 0% — root cause is missing eval feedback lo | emergent_bridge | Upgrade: Move pre-compaction session note from memory rule t | 0.15 |
| Frontend polling persists stale job IDs after server restart | emergent_bridge | 🌱 ASPIRATION — Growing energy around brain (emotion +0.44) | 0.15 |
| 🌱 ASPIRATION — Growing energy around brain (emotion +0.44) | emergent_bridge | Frontend polling persists stale job IDs after server restart | 0.15 |
| Rule: ask for confirmation before manipulating data to affec | emergent_bridge | Glo Creative Intelligence: LLM as Creative Director, not fix | 0.15 |
| Unconfirmed info stays contextual until earned through repet | emergent_bridge | Compaction boundary at 2026-03-22T18:07:03Z | 0.15 |
| Compaction boundary at 2026-03-22T18:07:03Z | emergent_bridge | Unconfirmed info stays contextual until earned through repet | 0.15 |
| Rule: every screen component must have a UI CONTRACT comment | emergent_bridge | Cluster forming: "Pattern: Emotional state during encoding b | 0.15 |
| Cluster forming: "Pattern: Emotional state during encoding b | emergent_bridge | Rule: every screen component must have a UI CONTRACT comment | 0.15 |
| [o_glo] Rule: budget_order | emergent_bridge | Cluster forming: "Pattern: Emotional state during encoding b | 0.15 |
| Cluster forming: "Pattern: Emotional state during encoding b | emergent_bridge | [o_glo] Rule: budget_order | 0.15 |
| Rule: every screen component must have a UI CONTRACT comment | emergent_bridge | Cluster forming: "Neuroscience of time perception: distribut | 0.15 |
| Cluster forming: "Neuroscience of time perception: distribut | emergent_bridge | Rule: every screen component must have a UI CONTRACT comment | 0.15 |
| [o_glo] Rule: budget_order | emergent_bridge | Cluster forming: "Neuroscience of time perception: distribut | 0.15 |
| Cluster forming: "Neuroscience of time perception: distribut | emergent_bridge | [o_glo] Rule: budget_order | 0.15 |
| Rule: every screen component must have a UI CONTRACT comment | emergent_bridge | Cluster forming: "Priming system: active concerns as backgro | 0.15 |
| Cluster forming: "Priming system: active concerns as backgro | emergent_bridge | Rule: every screen component must have a UI CONTRACT comment | 0.15 |
| [o_glo] Rule: budget_order | emergent_bridge | Cluster forming: "Priming system: active concerns as backgro | 0.15 |
| Cluster forming: "Priming system: active concerns as backgro | emergent_bridge | [o_glo] Rule: budget_order | 0.15 |
| Rule: every screen component must have a UI CONTRACT comment | emergent_bridge | Cluster forming: "Adaptive checkpoints: brain learns which n | 0.15 |
| Cluster forming: "Adaptive checkpoints: brain learns which n | emergent_bridge | Rule: every screen component must have a UI CONTRACT comment | 0.15 |
| [o_glo] Rule: budget_order | emergent_bridge | Cluster forming: "Adaptive checkpoints: brain learns which n | 0.15 |
| Cluster forming: "Adaptive checkpoints: brain learns which n | emergent_bridge | [o_glo] Rule: budget_order | 0.15 |
| Rule: every screen component must have a UI CONTRACT comment | emergent_bridge | Cluster forming: "Root cause: hooks go silent when brain ser | 0.15 |
| Cluster forming: "Root cause: hooks go silent when brain ser | emergent_bridge | Rule: every screen component must have a UI CONTRACT comment | 0.15 |
| [o_glo] Rule: budget_order | emergent_bridge | Cluster forming: "Root cause: hooks go silent when brain ser | 0.15 |
| Cluster forming: "Root cause: hooks go silent when brain ser | emergent_bridge | [o_glo] Rule: budget_order | 0.15 |
| 🔮 HYPOTHESIS — Dream insight gaining traction: Dream: [o_glo | emergent_bridge | 🌱 ASPIRATION — Growing energy around brain (emotion +0.44) | 0.15 |
| 🌱 ASPIRATION — Growing energy around brain (emotion +0.44) | emergent_bridge | 🔮 HYPOTHESIS — Dream insight gaining traction: Dream: [o_glo | 0.15 |
| Rule: every screen component must have a UI CONTRACT comment | emergent_bridge | ⚡ TENSION — Semantic similarity over keyword overlap vs v2.3 | 0.15 |
| ⚡ TENSION — Semantic similarity over keyword overlap vs v2.3 | emergent_bridge | Rule: every screen component must have a UI CONTRACT comment | 0.15 |
| [o_glo] Rule: budget_order | emergent_bridge | ⚡ TENSION — Semantic similarity over keyword overlap vs v2.3 | 0.15 |
| ⚡ TENSION — Semantic similarity over keyword overlap vs v2.3 | emergent_bridge | [o_glo] Rule: budget_order | 0.15 |
| Glo project history — research, pivot, build phases (March 2 | emergent_bridge | DAL pattern: mixin files must not access logs_conn or brain_ | 0.15 |
| DAL pattern: mixin files must not access logs_conn or brain_ | emergent_bridge | Glo project history — research, pivot, build phases (March 2 | 0.15 |
| Cluster forming: "Glo project history — research, pivot, bui | emergent_bridge | Confidence recalibration at session boundaries | 0.15 |
| Confidence recalibration at session boundaries | emergent_bridge | Cluster forming: "Glo project history — research, pivot, bui | 0.15 |
| Cluster forming: "Glo project history — research, pivot, bui | emergent_bridge | SKILL.md redesign session: included live agreeability correc | 0.15 |
| SKILL.md redesign session: included live agreeability correc | emergent_bridge | Cluster forming: "Glo project history — research, pivot, bui | 0.15 |
| Session Log — Reset #8 (charming-cannon) | emergent_bridge | PreCompact hook may not fire during context overflow (only o | 0.15 |
| PreCompact hook may not fire during context overflow (only o | emergent_bridge | Session Log — Reset #8 (charming-cannon) | 0.15 |
| Session Log — Reset #8 (charming-cannon) | emergent_bridge | v4 Phase 0: Map existing brain codebase before building new  | 0.15 |
| v4 Phase 0: Map existing brain codebase before building new  | emergent_bridge | Session Log — Reset #8 (charming-cannon) | 0.15 |
| 🔮 HYPOTHESIS — Dream insight gaining traction: Dream: Graph  | emergent_bridge | Dream: [concept] Recall Pipeline — the full pat ↔ Tom princi | 0.15 |
| Dream: [concept] Recall Pipeline — the full pat ↔ Tom princi | emergent_bridge | 🔮 HYPOTHESIS — Dream insight gaining traction: Dream: Graph  | 0.15 |
| Dream: Rule: ask for confirmation before manipu ↔ Glo Creati | emergent_bridge | Cluster forming: "How to pass _recall_log_id from pre-respon | 0.15 |
| Cluster forming: "How to pass _recall_log_id from pre-respon | emergent_bridge | Dream: Rule: ask for confirmation before manipu ↔ Glo Creati | 0.15 |
| Cluster forming: "Cluster forming: "How to pass _recall_log_ | emergent_bridge | Rule: Write session note BEFORE compaction hits | 0.15 |
| Rule: Write session note BEFORE compaction hits | emergent_bridge | Cluster forming: "Cluster forming: "How to pass _recall_log_ | 0.15 |
| Cluster forming: "Cluster forming: "How to pass _recall_log_ | emergent_bridge | Recurring pattern: Tom corrects Claude encoding depth — 3 oc | 0.15 |
| Recurring pattern: Tom corrects Claude encoding depth — 3 oc | emergent_bridge | Cluster forming: "Cluster forming: "How to pass _recall_log_ | 0.15 |
| Upgrade: Move pre-compaction session note from memory rule t | emergent_bridge | Cluster forming: "Will composite precision threshold of 45 h | 0.15 |
| Cluster forming: "Will composite precision threshold of 45 h | emergent_bridge | Upgrade: Move pre-compaction session note from memory rule t | 0.15 |
| Brain v4: Self-reflection node types — performance, failure, | emergent_bridge | Meta-lesson: encode transferable practices, not incident fac | 0.15 |
| Meta-lesson: encode transferable practices, not incident fac | emergent_bridge | Brain v4: Self-reflection node types — performance, failure, | 0.15 |
| Store-time encoding must be smart, not exhaustive | emergent_bridge | First brain-to-operator conversation: 2026-03-22 Session #9 | 0.15 |
| First brain-to-operator conversation: 2026-03-22 Session #9 | emergent_bridge | Store-time encoding must be smart, not exhaustive | 0.15 |
| Store-time encoding must be smart, not exhaustive | emergent_bridge | Will 0.05 learning rate for confidence updates be too slow o | 0.15 |
| Will 0.05 learning rate for confidence updates be too slow o | emergent_bridge | Store-time encoding must be smart, not exhaustive | 0.15 |
| Store-time encoding must be smart, not exhaustive | emergent_bridge | Does the feedback keyword detection ('useful', 'not useful') | 0.15 |
| Does the feedback keyword detection ('useful', 'not useful') | emergent_bridge | Store-time encoding must be smart, not exhaustive | 0.15 |
| Rule: every screen component must have a UI CONTRACT comment | emergent_bridge | Rule: telemetry on everything we build — brain must feel its | 0.15 |
| Rule: telemetry on everything we build — brain must feel its | emergent_bridge | Rule: every screen component must have a UI CONTRACT comment | 0.15 |
| Rule: every screen component must have a UI CONTRACT comment | emergent_bridge | Rule: 1 small dead-code fix per big build (2026-03-22) | 0.15 |
| Rule: 1 small dead-code fix per big build (2026-03-22) | emergent_bridge | Rule: every screen component must have a UI CONTRACT comment | 0.15 |
| Rule: every screen component must have a UI CONTRACT comment | emergent_bridge | 🔔 REMINDER — Review what render_operator_prompt() surfaces — | 0.15 |
| 🔔 REMINDER — Review what render_operator_prompt() surfaces — | emergent_bridge | Rule: every screen component must have a UI CONTRACT comment | 0.15 |
| Rule: every screen component must have a UI CONTRACT comment | emergent_bridge | 🔔 REMINDER — After Phase B: build comprehensive telemetry la | 0.15 |
| 🔔 REMINDER — After Phase B: build comprehensive telemetry la | emergent_bridge | Rule: every screen component must have a UI CONTRACT comment | 0.15 |
| Upgrade: Move pre-compaction session note from memory rule t | emergent_bridge | [vocab] Talk (auto-detected) | 0.15 |
| [vocab] Talk (auto-detected) | emergent_bridge | Upgrade: Move pre-compaction session note from memory rule t | 0.15 |
| Glo Creative Intelligence: LLM as Creative Director, not fix | emergent_bridge | [vocab] Expand (auto-detected) | 0.15 |
| [vocab] Expand (auto-detected) | emergent_bridge | Glo Creative Intelligence: LLM as Creative Director, not fix | 0.15 |
| Glo Creative Intelligence: LLM as Creative Director, not fix | emergent_bridge | [vocab] Maybe (auto-detected) | 0.15 |
| [vocab] Maybe (auto-detected) | emergent_bridge | Glo Creative Intelligence: LLM as Creative Director, not fix | 0.15 |
| Upgrade: Move pre-compaction session note from memory rule t | emergent_bridge | [vocab] Add (auto-detected) | 0.15 |
| [vocab] Add (auto-detected) | emergent_bridge | Upgrade: Move pre-compaction session note from memory rule t | 0.15 |
| Upgrade: Move pre-compaction session note from memory rule t | emergent_bridge | [vocab] Plan (auto-detected) | 0.15 |
| [vocab] Plan (auto-detected) | emergent_bridge | Upgrade: Move pre-compaction session note from memory rule t | 0.15 |
| Upgrade: Move pre-compaction session note from memory rule t | emergent_bridge | [vocab] bug_lesson (auto-detected) | 0.15 |
| [vocab] bug_lesson (auto-detected) | emergent_bridge | Upgrade: Move pre-compaction session note from memory rule t | 0.15 |
| Upgrade: Move pre-compaction session note from memory rule t | emergent_bridge | [vocab] Dimension (auto-detected) | 0.15 |
| [vocab] Dimension (auto-detected) | emergent_bridge | Upgrade: Move pre-compaction session note from memory rule t | 0.15 |
| Upgrade: Move pre-compaction session note from memory rule t | emergent_bridge | [vocab] Worktree (auto-detected) | 0.15 |
| [vocab] Worktree (auto-detected) | emergent_bridge | Upgrade: Move pre-compaction session note from memory rule t | 0.15 |
| Upgrade: Move pre-compaction session note from memory rule t | emergent_bridge | [vocab] Rerankers (auto-detected) | 0.15 |
| [vocab] Rerankers (auto-detected) | emergent_bridge | Upgrade: Move pre-compaction session note from memory rule t | 0.15 |
| Upgrade: Move pre-compaction session note from memory rule t | emergent_bridge | [vocab] Gemma (auto-detected) | 0.15 |
| [vocab] Gemma (auto-detected) | emergent_bridge | Upgrade: Move pre-compaction session note from memory rule t | 0.15 |
| Upgrade: Move pre-compaction session note from memory rule t | emergent_bridge | [vocab] Tiny (auto-detected) | 0.15 |
| [vocab] Tiny (auto-detected) | emergent_bridge | Upgrade: Move pre-compaction session note from memory rule t | 0.15 |
| Upgrade: Move pre-compaction session note from memory rule t | emergent_bridge | [vocab] Location Service (auto-detected) | 0.15 |
| [vocab] Location Service (auto-detected) | emergent_bridge | Upgrade: Move pre-compaction session note from memory rule t | 0.15 |
| Upgrade: Move pre-compaction session note from memory rule t | emergent_bridge | [vocab] CLS (auto-detected) | 0.15 |
| [vocab] CLS (auto-detected) | emergent_bridge | Upgrade: Move pre-compaction session note from memory rule t | 0.15 |
| Upgrade: Move pre-compaction session note from memory rule t | emergent_bridge | [vocab] Worktree (auto-detected) | 0.15 |
| [vocab] Worktree (auto-detected) | emergent_bridge | Upgrade: Move pre-compaction session note from memory rule t | 0.15 |
| Upgrade: Move pre-compaction session note from memory rule t | emergent_bridge | [vocab] Results (auto-detected) | 0.15 |
| [vocab] Results (auto-detected) | emergent_bridge | Upgrade: Move pre-compaction session note from memory rule t | 0.15 |
| Upgrade: Move pre-compaction session note from memory rule t | emergent_bridge | Recall benchmark: 10 conditions tested, cross-encoder wins b | 0.15 |
| Recall benchmark: 10 conditions tested, cross-encoder wins b | emergent_bridge | Upgrade: Move pre-compaction session note from memory rule t | 0.15 |
| Upgrade: Move pre-compaction session note from memory rule t | emergent_bridge | 🔮 HYPOTHESIS — 'v15 Architecture: Serverless brain — elimina | 0.15 |
| 🔮 HYPOTHESIS — 'v15 Architecture: Serverless brain — elimina | emergent_bridge | Upgrade: Move pre-compaction session note from memory rule t | 0.15 |
| Upgrade: Move pre-compaction session note from memory rule t | emergent_bridge | Cluster forming: "Noise test cases reveal more than positive | 0.15 |
| Cluster forming: "Noise test cases reveal more than positive | emergent_bridge | Upgrade: Move pre-compaction session note from memory rule t | 0.15 |
| Upgrade: Move pre-compaction session note from memory rule t | emergent_bridge | Cluster forming: "Slow test: test:TestBootHook.test_boot_ide | 0.15 |
| Cluster forming: "Slow test: test:TestBootHook.test_boot_ide | emergent_bridge | Upgrade: Move pre-compaction session note from memory rule t | 0.15 |
| Upgrade: Move pre-compaction session note from memory rule t | emergent_bridge | ⚡ TENSION — Moderation model: AI moderates first, ad vs Mode | 0.15 |
| ⚡ TENSION — Moderation model: AI moderates first, ad vs Mode | emergent_bridge | Upgrade: Move pre-compaction session note from memory rule t | 0.15 |
| ⚡ TENSION — Creative archetypes: infinite use cases  vs Glo  | emergent_bridge | Upgrade: Move pre-compaction session note from memory rule t | 0.15 |
| Upgrade: Move pre-compaction session note from memory rule t | emergent_bridge | ⚡ TENSION — Creative archetypes: infinite use cases  vs Glo  | 0.15 |
| ⚡ TENSION — Creative archetypes: infinite use cases  vs Glo  | emergent_bridge | Cluster forming: "Noise test cases reveal more than positive | 0.15 |
| Cluster forming: "Noise test cases reveal more than positive | emergent_bridge | ⚡ TENSION — Creative archetypes: infinite use cases  vs Glo  | 0.15 |
| 📊 PATTERN — Corrections cluster: MCP tools missing: .mcp.jso | emergent_bridge | Upgrade: Move pre-compaction session note from memory rule t | 0.15 |
| Upgrade: Move pre-compaction session note from memory rule t | emergent_bridge | 📊 PATTERN — Corrections cluster: MCP tools missing: .mcp.jso | 0.15 |
| 📊 PATTERN — Corrections cluster: MCP tools missing: .mcp.jso | emergent_bridge | [vocab] Results (auto-detected) | 0.15 |
| [vocab] Results (auto-detected) | emergent_bridge | 📊 PATTERN — Corrections cluster: MCP tools missing: .mcp.jso | 0.15 |
| 📊 PATTERN — Always used together: 'Tier pricing: 1.4x markup | emergent_bridge | Upgrade: Move pre-compaction session note from memory rule t | 0.15 |
| Upgrade: Move pre-compaction session note from memory rule t | emergent_bridge | 📊 PATTERN — Always used together: 'Tier pricing: 1.4x markup | 0.15 |
| 📊 PATTERN — Always used together: 'Tier pricing: 1.4x markup | emergent_bridge | [vocab] Results (auto-detected) | 0.15 |
| [vocab] Results (auto-detected) | emergent_bridge | 📊 PATTERN — Always used together: 'Tier pricing: 1.4x markup | 0.15 |
| Upgrade: Move pre-compaction session note from memory rule t | emergent_bridge | 🔮 HYPOTHESIS — 'Neuroscience of time perception: distributed | 0.15 |
| 🔮 HYPOTHESIS — 'Neuroscience of time perception: distributed | emergent_bridge | Upgrade: Move pre-compaction session note from memory rule t | 0.15 |
| [vocab] Glo (auto-detected) | emergent_bridge | [vocab] Expand (auto-detected) | 0.15 |
| [vocab] Expand (auto-detected) | emergent_bridge | [vocab] Glo (auto-detected) | 0.15 |
| Glo Creative Intelligence: LLM as Creative Director, not fix | emergent_bridge | [vocab] Glo (auto-detected) | 0.15 |
| [vocab] Glo (auto-detected) | emergent_bridge | Glo Creative Intelligence: LLM as Creative Director, not fix | 0.15 |
| Upgrade: Move pre-compaction session note from memory rule t | emergent_bridge | [vocab] LLM (auto-detected) | 0.15 |
| [vocab] LLM (auto-detected) | emergent_bridge | Upgrade: Move pre-compaction session note from memory rule t | 0.15 |
| Upgrade: Move pre-compaction session note from memory rule t | emergent_bridge | [vocab] Microsoft Phi (auto-detected) | 0.15 |
| [vocab] Microsoft Phi (auto-detected) | emergent_bridge | Upgrade: Move pre-compaction session note from memory rule t | 0.15 |
| EX.CO — End-to-end video platform for publishers: CMS, ad se | enables | Glo.io — Self-serve ad platform on EX.CO ad server. Anyone b | 0.9 |
| Future: typed edges + graph traversal for entity-aware knowl | enables | Entities are everything with identity: people, products, com | 0.30000000000000004 |
| Rule: media mockup must show creative ON publisher media wit | exemplifies | Rule: Glo disruptive concept — anyone can upload to any medi | 1.0 |
| Rule: Glo disruptive concept — anyone can upload to any medi | exemplifies | Rule: media mockup must show creative ON publisher media wit | 1.0 |
| Lesson: silent failures are the most dangerous class of bug | exemplifies | [bug] zsh treats unmatched globs as errors — killed every Ma | 0.9400000000000001 |
| [bug] zsh treats unmatched globs as errors — killed every Ma | exemplifies | Lesson: silent failures are the most dangerous class of bug | 0.9 |
| Lesson: silent failures are the most dangerous class of bug | exemplifies | [bug] duplicate JSON keys silently overwrite — hooks.json ha | 0.9 |
| [bug] duplicate JSON keys silently overwrite — hooks.json ha | exemplifies | Lesson: silent failures are the most dangerous class of bug | 0.9 |
| [o_glo] Rule: budget_order | extends | Experimental features must never block core operations | 0.9 |
| Experimental features must never block core operations | extends | [o_glo] Rule: budget_order | 0.9 |
| Thoughts vs decisions/rules differ fundamentally in decay ph | extends | Brain must have its own distinct voice, separate from Claude | 0.8 |
| Brain must have its own distinct voice, separate from Claude | extends | Thoughts vs decisions/rules differ fundamentally in decay ph | 0.8 |
| CampaignParamsResolver | feeds_into | GAM as campaign controller | 0.74 |
| GAM as campaign controller | feeds_into | CampaignParamsResolver | 0.74 |
| resolve-brain-db.sh: shared DB resolver sourced by all hooks | fixes | [bug] zsh treats unmatched globs as errors — killed every Ma | 0.95 |
| [bug] zsh treats unmatched globs as errors — killed every Ma | fixes | resolve-brain-db.sh: shared DB resolver sourced by all hooks | 0.9 |
| Tmemory — Persistent brain engine for Claude (v4.2) | governed_by | tmemory design principle: brain is a cue system, not a searc | 0.9 |
| tmemory design principle: brain is a cue system, not a searc | governed_by | Tmemory — Persistent brain engine for Claude (v4.2) | 0.9 |
| Glo.io — Self-serve ad platform on EX.CO ad server. Anyone b | governs | [o_glo] Rule: onboarding_fields | 0.9400000000000001 |
| [o_glo] Rule: onboarding_fields | governs | Glo.io — Self-serve ad platform on EX.CO ad server. Anyone b | 0.9400000000000001 |
| Glo.io — Self-serve ad platform on EX.CO ad server. Anyone b | governs | [o_glo] Rule: goal_placement | 0.9 |
| [o_glo] Rule: goal_placement | governs | Glo.io — Self-serve ad platform on EX.CO ad server. Anyone b | 0.9 |
| Glo.io — Self-serve ad platform on EX.CO ad server. Anyone b | governs | [o_glo] Rule: creative_default | 0.9400000000000001 |
| [o_glo] Rule: creative_default | governs | Glo.io — Self-serve ad platform on EX.CO ad server. Anyone b | 0.9400000000000001 |
| Glo.io — Self-serve ad platform on EX.CO ad server. Anyone b | governs | [o_glo] Rule: aspect_ratio | 0.92 |
| [o_glo] Rule: aspect_ratio | governs | Glo.io — Self-serve ad platform on EX.CO ad server. Anyone b | 0.92 |
| Glo.io — Self-serve ad platform on EX.CO ad server. Anyone b | governs | [o_glo] Rule: budget_order | 0.9400000000000001 |
| [o_glo] Rule: budget_order | governs | Glo.io — Self-serve ad platform on EX.CO ad server. Anyone b | 0.9400000000000001 |
| Glo.io — Self-serve ad platform on EX.CO ad server. Anyone b | governs | [o_glo] Rule: react_hooks | 0.92 |
| [o_glo] Rule: react_hooks | governs | Glo.io — Self-serve ad platform on EX.CO ad server. Anyone b | 0.92 |
| Supply Adapter pattern | implemented_by | GAM as campaign controller | 0.9 |
| GAM as campaign controller | implemented_by | Supply Adapter pattern | 0.9 |
| Goal only in AI generate path | implemented_in | glo-demo-v2.jsx | 0.9 |
| glo-demo-v2.jsx | implemented_in | Goal only in AI generate path | 0.9 |
| Budget screen: How to Spend first | implements | Tier pricing: 1.4x markup multiplier on publisher CPM | 0.9 |
| Tier pricing: 1.4x markup multiplier on publisher CPM | implements | Budget screen: How to Spend first | 0.9 |
| Tmemory — Persistent brain engine for Claude (v4.2) | includes | tmemory v4.2: recall scoring overhaul — uncapped spread acti | 0.9 |
| tmemory v4.2: recall scoring overhaul — uncapped spread acti | includes | Tmemory — Persistent brain engine for Claude (v4.2) | 0.9 |
| Tmemory — Persistent brain engine for Claude (v4.2) | includes | tmemory v4.2: dream-time keyword enrichment + extractKeyword | 0.9 |
| tmemory v4.2: dream-time keyword enrichment + extractKeyword | includes | Tmemory — Persistent brain engine for Claude (v4.2) | 0.9 |
| Tom — CEO of EX.CO | leads | Glo.io — Self-serve advertising platform | 0.9 |
| Glo.io — Self-serve advertising platform | leads | Tom — CEO of EX.CO | 0.9 |
| EX.CO — End-to-end video platform for publishers: CMS, ad se | owned_by | Tom — CEO of EX.CO | 0.9400000000000001 |
| Glo/EX.CO boundary | part_of | Glo.io — Self-serve advertising platform | 0.9 |
| Glo.io — Self-serve advertising platform | part_of | Glo/EX.CO boundary | 0.9 |
| GAM as campaign controller | part_of | Glo.io — Self-serve advertising platform | 0.9 |
| Glo.io — Self-serve advertising platform | part_of | GAM as campaign controller | 0.9 |
| Supply Adapter pattern | part_of | Glo.io — Self-serve advertising platform | 0.9000000000000001 |
| Glo.io — Self-serve advertising platform | part_of | Supply Adapter pattern | 0.9000000000000001 |
| Separate API + Web architecture | part_of | Glo.io — Self-serve advertising platform | 0.8200000000000001 |
| Glo.io — Self-serve advertising platform | part_of | Separate API + Web architecture | 0.8200000000000001 |
| Glo owns publisher profiles | part_of | Glo.io — Self-serve advertising platform | 0.8200000000000001 |
| Glo.io — Self-serve advertising platform | part_of | Glo owns publisher profiles | 0.8200000000000001 |
| Creative on Glo CDN | part_of | Glo.io — Self-serve advertising platform | 0.8600000000000001 |
| Glo.io — Self-serve advertising platform | part_of | Creative on Glo CDN | 0.8600000000000001 |
| API for agents at scale | part_of | Glo.io — Self-serve advertising platform | 0.76 |
| Glo.io — Self-serve advertising platform | part_of | API for agents at scale | 0.76 |
| CampaignParamsResolver | part_of | Supply Adapter pattern | 0.64 |
| Supply Adapter pattern | part_of | CampaignParamsResolver | 0.64 |
| Email/Notification System — Activation progress, performance | part_of | Emails include real screenshots from actual publisher site — | 0.8600000000000001 |
| Emails include real screenshots from actual publisher site — | part_of | Email/Notification System — Activation progress, performance | 0.8600000000000001 |
| Email/Notification System — Activation progress, performance | part_of | Sample page link from EX.CO can be included in emails — real | 0.9200000000000002 |
| Sample page link from EX.CO can be included in emails — real | part_of | Email/Notification System — Activation progress, performance | 0.9200000000000002 |
| Email/Notification System — Activation progress, performance | part_of | Rejection emails include reason + CTA to duplicate and try a | 0.8600000000000001 |
| Rejection emails include reason + CTA to duplicate and try a | part_of | Email/Notification System — Activation progress, performance | 0.8600000000000001 |
| Glo is closed-loop demand layer on EX.CO — not standalone DS | part_of | EX.CO — End-to-end video platform for publishers: CMS, ad se | 0.8400000000000001 |
| EX.CO — End-to-end video platform for publishers: CMS, ad se | part_of | Glo is closed-loop demand layer on EX.CO — not standalone DS | 0.8400000000000001 |
| Glo conceptually supported by EX.CO leadership, needs formal | part_of | EX.CO — End-to-end video platform for publishers: CMS, ad se | 0.8400000000000001 |
| EX.CO — End-to-end video platform for publishers: CMS, ad se | part_of | Glo conceptually supported by EX.CO leadership, needs formal | 0.8400000000000001 |
| Moderation initially by GLO/EX.CO ops team, publishers get a | part_of | EX.CO — End-to-end video platform for publishers: CMS, ad se | 0.9 |
| EX.CO — End-to-end video platform for publishers: CMS, ad se | part_of | Moderation initially by GLO/EX.CO ops team, publishers get a | 0.9 |
| [o_glo] Closed-loop demand layer on EX.CO — not standalone D | part_of | Glo.io — Self-serve ad platform on EX.CO ad server. Anyone b | 0.8800000000000001 |
| Glo.io — Self-serve ad platform on EX.CO ad server. Anyone b | part_of | [o_glo] Closed-loop demand layer on EX.CO — not standalone D | 0.8800000000000001 |
| [o_glo] Flywheel: unfilled inventory→house ads→recruit adver | part_of | Glo.io — Self-serve ad platform on EX.CO ad server. Anyone b | 0.8200000000000001 |
| Glo.io — Self-serve ad platform on EX.CO ad server. Anyone b | part_of | [o_glo] Flywheel: unfilled inventory→house ads→recruit adver | 0.8200000000000001 |
| AI video gen is NOT the moat. Buy/integrate: Creatify MVP $9 | part_of | Glo.io — Self-serve ad platform on EX.CO ad server. Anyone b | 0.9 |
| Glo.io — Self-serve ad platform on EX.CO ad server. Anyone b | part_of | AI video gen is NOT the moat. Buy/integrate: Creatify MVP $9 | 0.9 |
| [o_glo] 3 creative paths: upload, AI from URL/Google Maps, s | part_of | Glo.io — Self-serve ad platform on EX.CO ad server. Anyone b | 0.8800000000000001 |
| Glo.io — Self-serve ad platform on EX.CO ad server. Anyone b | part_of | [o_glo] 3 creative paths: upload, AI from URL/Google Maps, s | 0.8800000000000001 |
| Contextual intent: infer creative direction from WHO (media  | part_of | Glo.io — Self-serve ad platform on EX.CO ad server. Anyone b | 0.9 |
| Glo.io — Self-serve ad platform on EX.CO ad server. Anyone b | part_of | Contextual intent: infer creative direction from WHO (media  | 0.9 |
| Payment: Glo Credits 1:1 USD, Stripe customer balance. SSO→p | part_of | Glo.io — Self-serve ad platform on EX.CO ad server. Anyone b | 0.9 |
| Glo.io — Self-serve ad platform on EX.CO ad server. Anyone b | part_of | Payment: Glo Credits 1:1 USD, Stripe customer balance. SSO→p | 0.9 |
| Moderation: AI pre-screen + human final action. GLO/EX.CO op | part_of | Glo.io — Self-serve ad platform on EX.CO ad server. Anyone b | 0.9 |
| Glo.io — Self-serve ad platform on EX.CO ad server. Anyone b | part_of | Moderation: AI pre-screen + human final action. GLO/EX.CO op | 0.9 |
| [o_glo] Lifecycle: Draft→Pending Review→Active→Completed. Re | part_of | Glo.io — Self-serve ad platform on EX.CO ad server. Anyone b | 0.8800000000000001 |
| Glo.io — Self-serve ad platform on EX.CO ad server. Anyone b | part_of | [o_glo] Lifecycle: Draft→Pending Review→Active→Completed. Re | 0.8800000000000001 |
| Auth: Clerk recommended. All SSOs + Stripe. | part_of | Glo.io — Self-serve ad platform on EX.CO ad server. Anyone b | 0.9 |
| Glo.io — Self-serve ad platform on EX.CO ad server. Anyone b | part_of | Auth: Clerk recommended. All SSOs + Stripe. | 0.9 |
| [o_glo] Anti-fraud: payment gate over phone verification. Le | part_of | Glo.io — Self-serve ad platform on EX.CO ad server. Anyone b | 0.8200000000000001 |
| Glo.io — Self-serve ad platform on EX.CO ad server. Anyone b | part_of | [o_glo] Anti-fraud: payment gate over phone verification. Le | 0.8200000000000001 |
| [o_glo] Web app (PWA), not native iOS — avoids Apple's 30% i | part_of | Glo.io — Self-serve ad platform on EX.CO ad server. Anyone b | 0.8200000000000001 |
| Glo.io — Self-serve ad platform on EX.CO ad server. Anyone b | part_of | [o_glo] Web app (PWA), not native iOS — avoids Apple's 30% i | 0.8200000000000001 |
| Glo Credits — 1:1 USD. Wallet via Stripe customer balance. B | part_of | Glo.io — Self-serve ad platform on EX.CO ad server. Anyone b | 0.86 |
| Glo.io — Self-serve ad platform on EX.CO ad server. Anyone b | part_of | Glo Credits — 1:1 USD. Wallet via Stripe customer balance. B | 0.86 |
| GLO Brightness — Pricing tiers: Well($30) Bright($50) Shine( | part_of | Glo.io — Self-serve ad platform on EX.CO ad server. Anyone b | 0.92 |
| Glo.io — Self-serve ad platform on EX.CO ad server. Anyone b | part_of | GLO Brightness — Pricing tiers: Well($30) Bright($50) Shine( | 0.92 |
| Moderation System — Two-layer: AI pre-screen (risk score, fl | part_of | Glo.io — Self-serve ad platform on EX.CO ad server. Anyone b | 0.9 |
| Glo.io — Self-serve ad platform on EX.CO ad server. Anyone b | part_of | Moderation System — Two-layer: AI pre-screen (risk score, fl | 0.9 |
| Glo Lifecycle — State machine: Draft→Pending Review→Active→C | part_of | Glo.io — Self-serve ad platform on EX.CO ad server. Anyone b | 0.9 |
| Glo.io — Self-serve ad platform on EX.CO ad server. Anyone b | part_of | Glo Lifecycle — State machine: Draft→Pending Review→Active→C | 0.9 |
| My Glos Dashboard — User home. Cards per Glo: thumbnail, sta | part_of | Glo.io — Self-serve ad platform on EX.CO ad server. Anyone b | 0.9400000000000001 |
| Glo.io — Self-serve ad platform on EX.CO ad server. Anyone b | part_of | My Glos Dashboard — User home. Cards per Glo: thumbnail, sta | 0.9400000000000001 |
| Glo Numbers — Analytics detail per Glo. Views/day graph, cli | part_of | Glo.io — Self-serve ad platform on EX.CO ad server. Anyone b | 0.9 |
| Glo.io — Self-serve ad platform on EX.CO ad server. Anyone b | part_of | Glo Numbers — Analytics detail per Glo. Views/day graph, cli | 0.9 |
| Email/Notification System — Activation progress, performance | part_of | Glo.io — Self-serve ad platform on EX.CO ad server. Anyone b | 0.9 |
| Glo.io — Self-serve ad platform on EX.CO ad server. Anyone b | part_of | Email/Notification System — Activation progress, performance | 0.9 |
| Mobile Capture & Anti-Fraud — Prevent fake logins/bots. Paym | part_of | Glo.io — Self-serve ad platform on EX.CO ad server. Anyone b | 0.86 |
| Glo.io — Self-serve ad platform on EX.CO ad server. Anyone b | part_of | Mobile Capture & Anti-Fraud — Prevent fake logins/bots. Paym | 0.86 |
| Glo Beta Prototype — Vite+React app at glo/beta/ | part_of | Glo.io — Self-serve advertising platform | 0.9 |
| Glo.io — Self-serve advertising platform | part_of | Glo Beta Prototype — Vite+React app at glo/beta/ | 0.9 |
| Demo publishers: WDIV CTV ($22 CPM), Adweek Online ($18), Hu | part_of | Glo.io — Self-serve advertising platform | 0.78 |
| Glo.io — Self-serve advertising platform | part_of | Demo publishers: WDIV CTV ($22 CPM), Adweek Online ($18), Hu | 0.78 |
| Tier pricing: 1.4x markup multiplier on publisher CPM | part_of | Glo.io — Self-serve advertising platform | 0.9 |
| Glo.io — Self-serve advertising platform | part_of | Tier pricing: 1.4x markup multiplier on publisher CPM | 0.9 |
| App state model: screen-based routing via Context API | part_of | Glo.io — Self-serve advertising platform | 0.9 |
| Glo.io — Self-serve advertising platform | part_of | App state model: screen-based routing via Context API | 0.9 |
| Tmemory — Persistent brain engine for Claude (v4.2) | part_of | tmemory: separate user brain from plugin (fresh vs personal) | 0.9 |
| tmemory: separate user brain from plugin (fresh vs personal) | part_of | Tmemory — Persistent brain engine for Claude (v4.2) | 0.9 |
| Tmemory — Persistent brain engine for Claude (v4.2) | part_of | tmemory v4: self-improvement via instrumented recall + evalu | 0.9 |
| tmemory v4: self-improvement via instrumented recall + evalu | part_of | Tmemory — Persistent brain engine for Claude (v4.2) | 0.9 |
| Tmemory — Persistent brain engine for Claude (v4.2) | part_of | tmemory scoring: 35% relevance + 30% recency + 25% emotion + | 0.9 |
| tmemory scoring: 35% relevance + 30% recency + 25% emotion + | part_of | Tmemory — Persistent brain engine for Claude (v4.2) | 0.9 |
| Lesson: silent failures are the most dangerous class of bug | part_of | Convert remaining 49 except:pass blocks to _log_error | 0.8 |
| Convert remaining 49 except:pass blocks to _log_error | part_of | Lesson: silent failures are the most dangerous class of bug | 0.8 |
| Glo component map — 18 components, build status, key decisio | part_of | Glo.io — Self-serve advertising platform | 0.8 |
| Glo.io — Self-serve advertising platform | part_of | Glo component map — 18 components, build status, key decisio | 0.8 |
| Glo project history — research, pivot, build phases (March 2 | part_of | Glo.io — Self-serve advertising platform | 0.8 |
| Glo.io — Self-serve advertising platform | part_of | Glo project history — research, pivot, build phases (March 2 | 0.8 |
| Entities are everything with identity: people, products, com | part_of | Three extraction layers: vocabulary, entities, relationships | 0.30000000000000004 |
| Lesson: UI regression from Claude suggesting layout changes  | produced | Rule: every screen component must have a UI CONTRACT comment | 1.0 |
| Rule: every screen component must have a UI CONTRACT comment | produced | Lesson: UI regression from Claude suggesting layout changes  | 1.0 |
| Lesson: silent failures are the most dangerous class of bug | produced | Decision: error/debug logs go to brain_logs.db, not brain.db | 0.8 |
| Decision: error/debug logs go to brain_logs.db, not brain.db | produced | Lesson: silent failures are the most dangerous class of bug | 0.8 |
| Glo.io — Self-serve ad platform on EX.CO ad server. Anyone b | reference_for | Vibe.co — Streaming/CTV ad platform. UX reference for Glo: U | 0.82 |
| [o_credits] Properties | refers_to | [vocab] Let (auto-detected) | 0.6 |
| [vocab] Let (auto-detected) | refers_to | [o_credits] Properties | 0.6 |
| [o_myglos] Properties | refers_to | [vocab] Let (auto-detected) | 0.6 |
| [vocab] Let (auto-detected) | refers_to | [o_myglos] Properties | 0.6 |
| Tom prefers: discuss and define before building. Sequence: f | refers_to | [vocab] Channel (auto-detected) | 0.6 |
| [vocab] Channel (auto-detected) | refers_to | Tom prefers: discuss and define before building. Sequence: f | 0.6 |
| resolve-brain-db.sh: shared DB resolver sourced by all hooks | refers_to | [vocab] HOME (auto-detected) | 0.6 |
| [vocab] HOME (auto-detected) | refers_to | resolve-brain-db.sh: shared DB resolver sourced by all hooks | 0.6 |
| [o_antifraud] Properties | refers_to | [vocab] Add (auto-detected) | 0.6 |
| [vocab] Add (auto-detected) | refers_to | [o_antifraud] Properties | 0.6 |
| Moderation model: AI moderates first, adds comments+auto-sta | refers_to | [vocab] Add (auto-detected) | 0.6 |
| [vocab] Add (auto-detected) | refers_to | Moderation model: AI moderates first, adds comments+auto-sta | 0.6 |
| Added to Tom.md: always plan before executing. Tom discusses | refers_to | [vocab] Add (auto-detected) | 0.6 |
| [vocab] Add (auto-detected) | refers_to | Added to Tom.md: always plan before executing. Tom discusses | 0.6 |
| Added to Tom.md: always plan before executing. Tom discusses | refers_to | [vocab] Plan (auto-detected) | 0.6 |
| [vocab] Plan (auto-detected) | refers_to | Added to Tom.md: always plan before executing. Tom discusses | 0.6 |
| Tom: Plan-first approach for bigger questions | refers_to | [vocab] Plan (auto-detected) | 0.6 |
| [vocab] Plan (auto-detected) | refers_to | Tom: Plan-first approach for bigger questions | 0.6 |
| Embedder: FastEmbed-m (768d, 110MB, <1s) vs bge-m3 (1024d, 5 | refers_to | [vocab] Dimension (auto-detected) | 0.6 |
| [vocab] Dimension (auto-detected) | refers_to | Embedder: FastEmbed-m (768d, 110MB, <1s) vs bge-m3 (1024d, 5 | 0.6 |
| Session #8 (charming-cannon) achievements: hooks fixed, hear | refers_to | [vocab] Worktree (auto-detected) | 0.6 |
| [vocab] Worktree (auto-detected) | refers_to | Session #8 (charming-cannon) achievements: hooks fixed, hear | 0.6 |
| Session #8 (charming-cannon) achievements: hooks fixed, hear | refers_to | [vocab] Worktree (auto-detected) | 0.6 |
| [vocab] Worktree (auto-detected) | refers_to | Session #8 (charming-cannon) achievements: hooks fixed, hear | 0.6 |
| [vocab] Glo (auto-detected) | refers_to | Tom — CEO of EX.CO | 0.6 |
| Tom — CEO of EX.CO | refers_to | [vocab] Glo (auto-detected) | 0.6 |
| [vocab] Glo (auto-detected) | refers_to | Glo.io — Self-serve advertising platform | 0.6 |
| Glo.io — Self-serve advertising platform | refers_to | [vocab] Glo (auto-detected) | 0.6 |
| [vocab] Glo (auto-detected) | refers_to | Glo.io — Self-serve ad platform on EX.CO ad server. Anyone b | 0.6 |
| Glo.io — Self-serve ad platform on EX.CO ad server. Anyone b | refers_to | [vocab] Glo (auto-detected) | 0.6 |
| Glo Creative Intelligence: LLM as Creative Director, not fix | refers_to | [vocab] LLM (auto-detected) | 0.6 |
| [vocab] LLM (auto-detected) | refers_to | Glo Creative Intelligence: LLM as Creative Director, not fix | 0.6 |
| NanoBanana API: single image only despite docs showing array | related | Veo 3.1 via Gemini API supports multi-image video generation | 0.9 |
| Veo 3.1 via Gemini API supports multi-image video generation | related | NanoBanana API: single image only despite docs showing array | 0.9 |
| Lesson: silent failures are the most dangerous class of bug | related | Rule: after massive changes, update SKILL.md and CLAUDE.md i | 0.7 |
| Rule: after massive changes, update SKILL.md and CLAUDE.md i | related | Lesson: silent failures are the most dangerous class of bug | 0.7 |
| Lesson: silent failures are the most dangerous class of bug | related | Refactor: centralize all DB read/write through a data access | 0.6 |
| Refactor: centralize all DB read/write through a data access | related | Lesson: silent failures are the most dangerous class of bug | 0.6 |
| Session Log — Reset #8 (charming-cannon) | related | v5: 12 cross-feature connectivity gaps identified and fixed | 0.6000000000000001 |
| v5: 12 cross-feature connectivity gaps identified and fixed | related | Session Log — Reset #8 (charming-cannon) | 0.5 |
| brain.py — thin assembler + core infrastructure hub | related | Cross-module dependency graph: which mixin calls which | 0.5 |
| Cross-module dependency graph: which mixin calls which | related | brain.py — thin assembler + core infrastructure hub | 0.5 |
| brain.py — thin assembler + core infrastructure hub | related | Constraint: only brain.py may define __init__ — mixins must  | 0.5 |
| Constraint: only brain.py may define __init__ — mixins must  | related | brain.py — thin assembler + core infrastructure hub | 0.5 |
| Tmemory v1.1.0: Curiosity system proactively detects gaps, p | related | v2.3.0: Gap 8 — Object grouping detection in curiosity engin | 0.49262999984837197 |
| v2.3.0: Gap 8 — Object grouping detection in curiosity engin | related | Tmemory v1.1.0: Curiosity system proactively detects gaps, p | 0.49262999984837197 |
| Plugin v2.4.0 release: canonical schema + density log reader | related | v2.3.0: Pre-compact hook expanded with task/object/file extr | 0.5989607709077464 |
| v2.3.0: Pre-compact hook expanded with task/object/file extr | related | Plugin v2.4.0 release: canonical schema + density log reader | 0.5989607709077464 |
| Earned permanence: nodes earn locked status through consolid | related | Decision: locked=True serves dual purpose — permanence AND g | 0.7 |
| Decision: locked=True serves dual purpose — permanence AND g | related | Earned permanence: nodes earn locked status through consolid | 0.7 |
| Earned permanence: nodes earn locked status through consolid | related | Operational learnings decay naturally — dont lock, let reinf | 0.5208124807462609 |
| Operational learnings decay naturally — dont lock, let reinf | related | Earned permanence: nodes earn locked status through consolid | 0.5208124807462609 |
| Plugin v2.4.0 release: canonical schema + density log reader | related | Schema refactoring: canonical schema.js replaces migration c | 0.5367698659418255 |
| Schema refactoring: canonical schema.js replaces migration c | related | Plugin v2.4.0 release: canonical schema + density log reader | 0.5367698659418255 |
| Schema refactoring: canonical schema.js replaces migration c | related | Current brain file structure: brain.py + 8 support modules + | 0.5056705268522134 |
| Current brain file structure: brain.py + 8 support modules + | related | Schema refactoring: canonical schema.js replaces migration c | 0.5056705268522134 |
| Tom: Plan-first approach for bigger questions | related | Tom: Prioritizes proven track record in research and solutio | 0.6979660207965236 |
| Tom: Prioritizes proven track record in research and solutio | related | Tom: Plan-first approach for bigger questions | 0.6979660207965236 |
| Tom: Prioritizes proven track record in research and solutio | related | User feedback: Avoid unnecessary logins (Tom prefers not to  | 0.6166376282678256 |
| User feedback: Avoid unnecessary logins (Tom prefers not to  | related | Tom: Prioritizes proven track record in research and solutio | 0.6166376282678256 |
| Approach B chosen: Self-healing hooks via ensure-brain.sh | related | Root cause: hooks go silent when brain server dies — ensure- | 0.72 |
| Root cause: hooks go silent when brain server dies — ensure- | related | Approach B chosen: Self-healing hooks via ensure-brain.sh | 0.7 |
| Approach B chosen: Self-healing hooks via ensure-brain.sh | related | Brain server died again during v11 implementation — platform | 0.634639058498843 |
| Brain server died again during v11 implementation — platform | related | Approach B chosen: Self-healing hooks via ensure-brain.sh | 0.634639058498843 |
| Redirect learning: mutual improvement through principle extr | related | User feedback: Avoid unnecessary logins (Tom prefers not to  | 0.4908229412270825 |
| User feedback: Avoid unnecessary logins (Tom prefers not to  | related | Redirect learning: mutual improvement through principle extr | 0.4908229412270825 |
| Redirect learning: mutual improvement through principle extr | related | Tom principle: Make parameters, not decisions | 0.4532238359747012 |
| Tom principle: Make parameters, not decisions | related | Redirect learning: mutual improvement through principle extr | 0.4532238359747012 |
| CampaignParamsResolver | related | [stm:s59] CampaignParamsResolver — isolated component that t | 0.7 |
| [stm:s59] CampaignParamsResolver — isolated component that t | related | CampaignParamsResolver | 0.7 |
| [stm:s59] CampaignParamsResolver — isolated component that t | related | Supply Adapter pattern | 0.5979591616263273 |
| Supply Adapter pattern | related | [stm:s59] CampaignParamsResolver — isolated component that t | 0.5979591616263273 |
| Hooks fire successfully; encoding sparseness was strategy pr | related | Constraint: Claude compaction destroys unencoded knowledge — | 0.570511579256704 |
| Constraint: Claude compaction destroys unencoded knowledge — | related | Hooks fire successfully; encoding sparseness was strategy pr | 0.570511579256704 |
| Hooks fire successfully; encoding sparseness was strategy pr | related | Root cause: hooks go silent when brain server dies — ensure- | 0.5693280240363122 |
| Root cause: hooks go silent when brain server dies — ensure- | related | Hooks fire successfully; encoding sparseness was strategy pr | 0.5693280240363122 |
| Tom: Known UX patterns over novel invention | related | Tom references competitor UX frequently. When he names a pro | 0.7 |
| Tom references competitor UX frequently. When he names a pro | related | Tom: Known UX patterns over novel invention | 0.7 |
| Tom: Known UX patterns over novel invention | related | Tom wants working demos over mockups. 'A working basic produ | 0.6527953703390108 |
| Tom wants working demos over mockups. 'A working basic produ | related | Tom: Known UX patterns over novel invention | 0.6527953703390108 |
| Bridge lifecycle: weight 0.15 initial, emergent_bridge edge  | related | v11 post-launch monitoring: check bridge survival after 72h, | 0.6025412851884768 |
| v11 post-launch monitoring: check bridge survival after 72h, | related | Bridge lifecycle: weight 0.15 initial, emergent_bridge edge  | 0.6025412851884768 |
| Bridge lifecycle: weight 0.15 initial, emergent_bridge edge  | related | v11 emergent graph bridging — implementation complete | 0.5815760576751934 |
| v11 emergent graph bridging — implementation complete | related | Bridge lifecycle: weight 0.15 initial, emergent_bridge edge  | 0.5815760576751934 |
| Real engine simulation v1: 375 messages via Brain.js actual  | related | brain_remember.py — all storage paths (nodes, metadata, embe | 0.5363630219340864 |
| brain_remember.py — all storage paths (nodes, metadata, embe | related | Real engine simulation v1: 375 messages via Brain.js actual  | 0.5363630219340864 |
| brain.py — thin assembler + core infrastructure hub | related | Real engine simulation v1: 375 messages via Brain.js actual  | 0.5209753841639351 |
| Real engine simulation v1: 375 messages via Brain.js actual  | related | brain.py — thin assembler + core infrastructure hub | 0.5209753841639351 |
| Video prompt rewritten: Hook→Showcase→CTA structure, industr | related | Video duration: 7 seconds (NanoBanana MVP, replaces 15s/30s  | 0.5350256687343795 |
| Video duration: 7 seconds (NanoBanana MVP, replaces 15s/30s  | related | Video prompt rewritten: Hook→Showcase→CTA structure, industr | 0.5350256687343795 |
| Google Places integration: server-side proxy for Places-only | related | Separate API + Web architecture | 0.46127949155825637 |
| Separate API + Web architecture | related | Google Places integration: server-side proxy for Places-only | 0.46127949155825637 |
| Pre-compact extraction: prioritize information preservation  | related | Rule: Write session note BEFORE compaction hits | 0.6061293520556842 |
| Rule: Write session note BEFORE compaction hits | related | Pre-compact extraction: prioritize information preservation  | 0.6061293520556842 |
| Pre-compact extraction: prioritize information preservation  | related | Upgrade: Move pre-compaction session note from memory rule t | 0.5666588372882451 |
| Upgrade: Move pre-compaction session note from memory rule t | related | Pre-compact extraction: prioritize information preservation  | 0.5666588372882451 |
| Offline model setup — download once at install, cache locall | related | Model delivery: pip3 install -e . from model-package for loc | 0.45746916303538626 |
| Model delivery: pip3 install -e . from model-package for loc | related | Offline model setup — download once at install, cache locall | 0.45746916303538626 |
| Added to Tom.md: always plan before executing. Tom discusses | related | Tom principle: Merge, never overwrite | 0.5407860643068166 |
| Tom principle: Merge, never overwrite | related | Added to Tom.md: always plan before executing. Tom discusses | 0.5407860643068166 |
| Tom values component separation. When scope grows, break int | related | Tom principle: Merge, never overwrite | 0.5285881723110282 |
| Tom principle: Merge, never overwrite | related | Tom values component separation. When scope grows, break int | 0.5285881723110282 |
| React Hook Violation: Cannot call useState/useEffect inside  | related | [o_glo] Rule: react_hooks | 0.7 |
| [o_glo] Rule: react_hooks | related | React Hook Violation: Cannot call useState/useEffect inside  | 0.7 |
| React Hook Violation: Cannot call useState/useEffect inside  | related | [stm:s45] CRITICAL TOM FEEDBACK (given 3x — DO NOT REVERT):  | 0.4843913189738947 |
| [stm:s45] CRITICAL TOM FEEDBACK (given 3x — DO NOT REVERT):  | related | React Hook Violation: Cannot call useState/useEffect inside  | 0.4843913189738947 |
| tmemory context file search: Remove artificial prefixes to p | related | context-file/find: tag-based matching insufficient for disco | 0.5966099582351481 |
| context-file/find: tag-based matching insufficient for disco | related | tmemory context file search: Remove artificial prefixes to p | 0.5966099582351481 |
| context-file/find: tag-based matching insufficient for disco | related | [ctx:big-test-file] Large Content Test | 0.5306514297376882 |
| [ctx:big-test-file] Large Content Test | related | context-file/find: tag-based matching insufficient for disco | 0.5306514297376882 |
| React hooks rule | related | [o_glo] Rule: react_hooks | 0.7 |
| [o_glo] Rule: react_hooks | related | React hooks rule | 0.7 |
| Tom wants working demos over mockups. 'A working basic produ | related | React hooks rule | 0.5288053775082483 |
| React hooks rule | related | Tom wants working demos over mockups. 'A working basic produ | 0.5288053775082483 |
| Plugin build: explicit include list in build-plugin.sh (repl | related | [bug] Plugin cache: ~/.claude/plugins/cache/ serves stale fi | 0.5467176429228537 |
| [bug] Plugin cache: ~/.claude/plugins/cache/ serves stale fi | related | Plugin build: explicit include list in build-plugin.sh (repl | 0.5467176429228537 |
| Correction: tmemory is a general Claude plugin, not project- | related | Plugin build: explicit include list in build-plugin.sh (repl | 0.5203281954831284 |
| Plugin build: explicit include list in build-plugin.sh (repl | related | Correction: tmemory is a general Claude plugin, not project- | 0.5203281954831284 |
| Correction: tmemory is a general Claude plugin, not project- | related | Tmemory plugin: Bundle selective node_modules, not full dire | 0.6089598064507122 |
| Tmemory plugin: Bundle selective node_modules, not full dire | related | Correction: tmemory is a general Claude plugin, not project- | 0.6089598064507122 |
| Tmemory plugin: Bundle selective node_modules, not full dire | related | tmemory: separate user brain from plugin (fresh vs personal) | 0.5647102892866503 |
| tmemory: separate user brain from plugin (fresh vs personal) | related | Tmemory plugin: Bundle selective node_modules, not full dire | 0.5647102892866503 |
| Video variants: generate 15s and 30s in parallel, show both  | related | Video prompt rewritten: Hook→Showcase→CTA structure, industr | 0.5366649815269815 |
| Video prompt rewritten: Hook→Showcase→CTA structure, industr | related | Video variants: generate 15s and 30s in parallel, show both  | 0.5366649815269815 |
| Batch HTTP endpoint /pre-edit for hook optimization | related | Pre-edit-suggest hook now triggers procedures | 0.6039253507223408 |
| Pre-edit-suggest hook now triggers procedures | related | Batch HTTP endpoint /pre-edit for hook optimization | 0.6039253507223408 |
| Pre-edit-suggest hook now triggers procedures | related | brain_surface.py — presentation layer for hooks (suggest, bo | 0.5207978336246011 |
| brain_surface.py — presentation layer for hooks (suggest, bo | related | Pre-edit-suggest hook now triggers procedures | 0.5207978336246011 |
| API Integration: Read docs first, plan before executing | related | Browser fetch to Creatify API blocked by CORS — use server-s | 0.46195068063581696 |
| Browser fetch to Creatify API blocked by CORS — use server-s | related | API Integration: Read docs first, plan before executing | 0.46195068063581696 |
| Brain v4: Self-reflection node types — performance, failure, | related | Brain v4: Host awareness layer — brain learns about its oper | 0.5774684534206288 |
| Brain v4: Host awareness layer — brain learns about its oper | related | Brain v4: Self-reflection node types — performance, failure, | 0.5774684534206288 |
| Brain v4: Self-reflection node types — performance, failure, | related | Everything connects — no silos in the brain architecture | 0.5267692379180385 |
| Everything connects — no silos in the brain architecture | related | Brain v4: Self-reflection node types — performance, failure, | 0.5267692379180385 |
| [stm:s55] LOCKED: System architecture decisions — Glo owns u | related | EX.CO — Video platform and ad server | 0.5982660652790817 |
| EX.CO — Video platform and ad server | related | [stm:s55] LOCKED: System architecture decisions — Glo owns u | 0.5982660652790817 |
| [stm:s55] LOCKED: System architecture decisions — Glo owns u | related | Dream: Component: My Glos Dashboard ↔ SSO→payment linking: G | 0.5424433186404923 |
| Dream: Component: My Glos Dashboard ↔ SSO→payment linking: G | related | [stm:s55] LOCKED: System architecture decisions — Glo owns u | 0.5424433186404923 |
| Glo backend stack: Node.js + Express, AWS infrastructure (RD | related | Separate API + Web architecture | 0.5232807744923537 |
| Separate API + Web architecture | related | Glo backend stack: Node.js + Express, AWS infrastructure (RD | 0.5232807744923537 |
| Glo backend stack: Node.js + Express, AWS infrastructure (RD | related | Glo.io — Self-serve advertising platform | 0.5230928600767388 |
| Glo.io — Self-serve advertising platform | related | Glo backend stack: Node.js + Express, AWS infrastructure (RD | 0.5230928600767388 |
| Script approach object must include key field (creative-dire | related | Video prompt rewritten: Hook→Showcase→CTA structure, industr | 0.4509154299173945 |
| Video prompt rewritten: Hook→Showcase→CTA structure, industr | related | Script approach object must include key field (creative-dire | 0.4509154299173945 |
| Creatify: Switch from 3-full-renders to preview-first workfl | related | Preview-first workflow: 6 previews, user selects, then rende | 0.6992904726811098 |
| Preview-first workflow: 6 previews, user selects, then rende | related | Creatify: Switch from 3-full-renders to preview-first workfl | 0.6992904726811098 |
| Preview-first workflow: 6 previews, user selects, then rende | related | Adweek — Media industry news site. Demo use case #2 — online | 0.49156020519646776 |
| Adweek — Media industry news site. Demo use case #2 — online | related | Preview-first workflow: 6 previews, user selects, then rende | 0.49156020519646776 |
| Phase 0.5B fix: keyword-only fallback scores penalized by KE | related | [param] KEYWORD_FALLBACK_WEIGHT=0.10 — keyword precision for | 0.7 |
| [param] KEYWORD_FALLBACK_WEIGHT=0.10 — keyword precision for | related | Phase 0.5B fix: keyword-only fallback scores penalized by KE | 0.7 |
| [param] KEYWORD_FALLBACK_WEIGHT=0.10 — keyword precision for | related | Phase 0.5B: Recall weights flipped to 90/10 embeddings-first | 0.67366731314685 |
| Phase 0.5B: Recall weights flipped to 90/10 embeddings-first | related | [param] KEYWORD_FALLBACK_WEIGHT=0.10 — keyword precision for | 0.67366731314685 |
| tmemory v1.6.0: configurable parameters infrastructure | related | [ctx:tmemory-architecture] tmemory — Brain Plugin Architectu | 0.6145194196529954 |
| [ctx:tmemory-architecture] tmemory — Brain Plugin Architectu | related | tmemory v1.6.0: configurable parameters infrastructure | 0.6145194196529954 |
| tmemory v1.6.0: configurable parameters infrastructure | related | Dynamic parameters in brain.py: thresholds that should be tu | 0.5976371558258752 |
| Dynamic parameters in brain.py: thresholds that should be tu | related | tmemory v1.6.0: configurable parameters infrastructure | 0.5976371558258752 |
| Supply Adapter pattern | related | [stm:s56] LOCKED: Supply Adapter pattern — clean abstraction | 0.7 |
| [stm:s56] LOCKED: Supply Adapter pattern — clean abstraction | related | Supply Adapter pattern | 0.7 |
| Rule: Ask clarifying questions during encoding | related | Correction: Tom had to prompt me to form memories about this | 0.5637466879921966 |
| Correction: Tom had to prompt me to form memories about this | related | Rule: Ask clarifying questions during encoding | 0.5637466879921966 |
| Rule: Ask clarifying questions during encoding | related | Tom: "I practice my beliefs" | 0.5593376811305184 |
| Tom: "I practice my beliefs" | related | Rule: Ask clarifying questions during encoding | 0.5593376811305184 |
| tmemory context file search: Remove artificial prefixes to p | related | Destructive operations require context-awareness — execution | 0.5144714951307101 |
| Destructive operations require context-awareness — execution | related | tmemory context file search: Remove artificial prefixes to p | 0.5144714951307101 |
| Destructive operations require context-awareness — execution | related | Correction: tmemory is a general Claude plugin, not project- | 0.4900980692432975 |
| Correction: tmemory is a general Claude plugin, not project- | related | Destructive operations require context-awareness — execution | 0.4900980692432975 |
| Store-time encoding must be smart, not exhaustive | related | Divergence: Brain encoding should be concise and well-organi | 0.6883714391295708 |
| Divergence: Brain encoding should be concise and well-organi | related | Store-time encoding must be smart, not exhaustive | 0.6883714391295708 |
| Store-time encoding must be smart, not exhaustive | related | Correction: Tom had to prompt me to form memories about this | 0.6568548298073776 |
| Correction: Tom had to prompt me to form memories about this | related | Store-time encoding must be smart, not exhaustive | 0.6568548298073776 |
| Feedback: Discuss pros/cons BEFORE committing sensitive chan | related | Rule: Write session note BEFORE compaction hits | 0.516486154373604 |
| Rule: Write session note BEFORE compaction hits | related | Feedback: Discuss pros/cons BEFORE committing sensitive chan | 0.516486154373604 |
| Feedback: Discuss pros/cons BEFORE committing sensitive chan | related | Divergence: Brain encoding should be concise and well-organi | 0.4975589489722822 |
| Divergence: Brain encoding should be concise and well-organi | related | Feedback: Discuss pros/cons BEFORE committing sensitive chan | 0.4975589489722822 |
| Glo video ads: no avatars | related | Rule: Glo disruptive concept — anyone can upload to any medi | 0.5755144321520459 |
| Rule: Glo disruptive concept — anyone can upload to any medi | related | Glo video ads: no avatars | 0.5755144321520459 |
| Thought decay uses wall-clock time, not time-dilation | related | Research: thought node half-life tuning — 3h may not be righ | 0.6821801770715133 |
| Research: thought node half-life tuning — 3h may not be righ | related | Thought decay uses wall-clock time, not time-dilation | 0.6821801770715133 |
| Thought decay uses wall-clock time, not time-dilation | related | Implemented pruning time-dilation with configurable active/i | 0.5958003910207692 |
| Implemented pruning time-dilation with configurable active/i | related | Thought decay uses wall-clock time, not time-dilation | 0.5958003910207692 |
| Publisher type case mapping: uppercase UI → lowercase backen | related | EX.CO — End-to-end video platform for publishers: CMS, ad se | 0.505382320369503 |
| EX.CO — End-to-end video platform for publishers: CMS, ad se | related | Publisher type case mapping: uppercase UI → lowercase backen | 0.505382320369503 |
| Publisher type case mapping: uppercase UI → lowercase backen | related | Video prompt rewritten: Hook→Showcase→CTA structure, industr | 0.4792865963432989 |
| Video prompt rewritten: Hook→Showcase→CTA structure, industr | related | Publisher type case mapping: uppercase UI → lowercase backen | 0.4792865963432989 |
| Batch HTTP endpoint /pre-edit for hook optimization | related | Decision: daemon pre_edit command returns suggestions + chan | 0.6135240490869669 |
| Decision: daemon pre_edit command returns suggestions + chan | related | Batch HTTP endpoint /pre-edit for hook optimization | 0.6135240490869669 |
| When Tom says 'not now' or 'don't want to go into it' — park | related | Tom signals: long messages = excited, short = frustrated, 'n | 0.6314372295573591 |
| Tom signals: long messages = excited, short = frustrated, 'n | related | When Tom says 'not now' or 'don't want to go into it' — park | 0.6314372295573591 |
| Tom: designer's eye despite engineering background — sees pr | related | Tom wants working demos over mockups. 'A working basic produ | 0.5784104699099665 |
| Tom wants working demos over mockups. 'A working basic produ | related | Tom: designer's eye despite engineering background — sees pr | 0.5784104699099665 |
| Tom: designer's eye despite engineering background — sees pr | related | Tom references competitor UX frequently. When he names a pro | 0.5167935984623494 |
| Tom references competitor UX frequently. When he names a pro | related | Tom: designer's eye despite engineering background — sees pr | 0.5167935984623494 |
| tmemory context files: slow memory with topic-based discover | related | [ctx:tmemory-architecture] tmemory — Brain Plugin Architectu | 0.6370491420617634 |
| [ctx:tmemory-architecture] tmemory — Brain Plugin Architectu | related | tmemory context files: slow memory with topic-based discover | 0.6370491420617634 |
| Correction: tmemory is a general Claude plugin, not project- | related | tmemory context files: slow memory with topic-based discover | 0.6274424533038415 |
| tmemory context files: slow memory with topic-based discover | related | Correction: tmemory is a general Claude plugin, not project- | 0.6274424533038415 |
| Batch HTTP endpoint /pre-edit for hook optimization | related | Hook latency: 170ms startup (fixed) + 520ms connections (var | 0.7 |
| Hook latency: 170ms startup (fixed) + 520ms connections (var | related | Batch HTTP endpoint /pre-edit for hook optimization | 0.7 |
| Hook latency: 170ms startup (fixed) + 520ms connections (var | related | Decision: daemon is additive — hooks try daemon first, fall  | 0.5688816925783082 |
| Decision: daemon is additive — hooks try daemon first, fall  | related | Hook latency: 170ms startup (fixed) + 520ms connections (var | 0.5688816925783082 |
| Remove thought FIFO cap—decay is the sole filter | related | Thought nodes: brain self-observations with 3h half-life, FI | 0.6425545602157952 |
| Thought nodes: brain self-observations with 3h half-life, FI | related | Remove thought FIFO cap—decay is the sole filter | 0.6425545602157952 |
| Remove thought FIFO cap—decay is the sole filter | related | Research: thought node half-life tuning — 3h may not be righ | 0.5968282730761522 |
| Research: thought node half-life tuning — 3h may not be righ | related | Remove thought FIFO cap—decay is the sole filter | 0.5968282730761522 |
| Consolidation bridge discovery: 50% recent nodes + 50% rando | related | v11 emergent graph bridging — implementation complete | 0.6032217482684428 |
| v11 emergent graph bridging — implementation complete | related | Consolidation bridge discovery: 50% recent nodes + 50% rando | 0.6032217482684428 |
| Consolidation bridge discovery: 50% recent nodes + 50% rando | related | Integration test: auto-enrichment working | 0.533656080991062 |
| Integration test: auto-enrichment working | related | Consolidation bridge discovery: 50% recent nodes + 50% rando | 0.533656080991062 |
| Separate API + Web architecture | related | [stm:s57] LOCKED: Separate API + Web. Glo API (REST) is sing | 0.7 |
| [stm:s57] LOCKED: Separate API + Web. Glo API (REST) is sing | related | Separate API + Web architecture | 0.7 |
| Glo Lifecycle — State machine: Draft→Pending Review→Active→C | related | [stm:s57] LOCKED: Separate API + Web. Glo API (REST) is sing | 0.49819293117962343 |
| [stm:s57] LOCKED: Separate API + Web. Glo API (REST) is sing | related | Glo Lifecycle — State machine: Draft→Pending Review→Active→C | 0.49819293117962343 |
| Tmemory boot: Use setsid node index.js for independent proce | related | tmemory: separate user brain from plugin (fresh vs personal) | 0.6268828852818158 |
| tmemory: separate user brain from plugin (fresh vs personal) | related | Tmemory boot: Use setsid node index.js for independent proce | 0.6068828852818158 |
| Tmemory boot: Use setsid node index.js for independent proce | related | Session goal: simplify brain boot process for new users | 0.5464572986304912 |
| Session goal: simplify brain boot process for new users | related | Tmemory boot: Use setsid node index.js for independent proce | 0.5464572986304912 |
| Pre-compact brain healthcheck: 6 fetches per session (verifi | related | Constraint: Claude compaction destroys unencoded knowledge — | 0.5862519539922815 |
| Constraint: Claude compaction destroys unencoded knowledge — | related | Pre-compact brain healthcheck: 6 fetches per session (verifi | 0.5862519539922815 |
| Pre-compact brain healthcheck: 6 fetches per session (verifi | related | Compaction experiment FAILED — recap encoding was skipped, a | 0.5318478828588088 |
| Compaction experiment FAILED — recap encoding was skipped, a | related | Pre-compact brain healthcheck: 6 fetches per session (verifi | 0.5318478828588088 |
| Generic defaults for project/user — env vars override hardco | related | tmemory: separate user brain from plugin (fresh vs personal) | 0.6037112774556721 |
| tmemory: separate user brain from plugin (fresh vs personal) | related | Generic defaults for project/user — env vars override hardco | 0.6037112774556721 |
| Correction: tmemory is a general Claude plugin, not project- | related | Generic defaults for project/user — env vars override hardco | 0.5941494808004295 |
| Generic defaults for project/user — env vars override hardco | related | Correction: tmemory is a general Claude plugin, not project- | 0.5941494808004295 |
| Fix: contextual penalty must apply to blended_score BEFORE s | related | Phase 0.5B: recall_with_embeddings() rewritten — embed query | 0.5799954916526615 |
| Phase 0.5B: recall_with_embeddings() rewritten — embed query | related | Fix: contextual penalty must apply to blended_score BEFORE s | 0.5799954916526615 |
| Fix: contextual penalty must apply to blended_score BEFORE s | related | brain_recall.py — all retrieval paths (embeddings, keywords, | 0.5135803410955853 |
| brain_recall.py — all retrieval paths (embeddings, keywords, | related | Fix: contextual penalty must apply to blended_score BEFORE s | 0.5135803410955853 |
| Rule: Preserve numbers, proper nouns, names in keywords | related | v2.3.0: extractKeywords 4-phase extraction pipeline | 0.626385726699257 |
| v2.3.0: extractKeywords 4-phase extraction pipeline | related | Rule: Preserve numbers, proper nouns, names in keywords | 0.626385726699257 |
| Rule: Preserve numbers, proper nouns, names in keywords | related | tmemory v4.2: dream-time keyword enrichment + extractKeyword | 0.46778687681242664 |
| tmemory v4.2: dream-time keyword enrichment + extractKeyword | related | Rule: Preserve numbers, proper nouns, names in keywords | 0.46778687681242664 |
| Brain self-generates thought nodes (ideation engine, not jus | related | Thought nodes: brain self-observations with 3h half-life, FI | 0.555468352466446 |
| Thought nodes: brain self-observations with 3h half-life, FI | related | Brain self-generates thought nodes (ideation engine, not jus | 0.555468352466446 |
| Brain self-generates thought nodes (ideation engine, not jus | related | brain_consciousness.py — signal detection and observation la | 0.550400426027878 |
| brain_consciousness.py — signal detection and observation la | related | Brain self-generates thought nodes (ideation engine, not jus | 0.550400426027878 |
| Audit SKILL.md for limiting instructions | related | Session-activity progressive warnings: 0 remembers→ALERT, 8+ | 0.5034715465380242 |
| Session-activity progressive warnings: 0 remembers→ALERT, 8+ | related | Audit SKILL.md for limiting instructions | 0.5034715465380242 |
| Audit SKILL.md for limiting instructions | related | Added to Tom.md: always plan before executing. Tom discusses | 0.49210081403100925 |
| Added to Tom.md: always plan before executing. Tom discusses | related | Audit SKILL.md for limiting instructions | 0.49210081403100925 |
| [o_moderation] Properties | related | Conscious layer visual format: icon + label + content + meta | 0.5152306827014856 |
| Conscious layer visual format: icon + label + content + meta | related | [o_moderation] Properties | 0.5152306827014856 |
| Remove curiosity cap, extend encoding time budget | related | Session-activity progressive warnings: 0 remembers→ALERT, 8+ | 0.5143371558871779 |
| Session-activity progressive warnings: 0 remembers→ALERT, 8+ | related | Remove curiosity cap, extend encoding time budget | 0.5143371558871779 |
| Remove curiosity cap, extend encoding time budget | related | After massive changes: update SKILL.md and CLAUDE.md | 0.4958555622598366 |
| After massive changes: update SKILL.md and CLAUDE.md | related | Remove curiosity cap, extend encoding time budget | 0.4958555622598366 |
| Refactoring: read breadcrumbs, zero dead code, verify full a | related | User feedback: Avoid unnecessary logins (Tom prefers not to  | 0.5902588121298745 |
| User feedback: Avoid unnecessary logins (Tom prefers not to  | related | Refactoring: read breadcrumbs, zero dead code, verify full a | 0.5902588121298745 |
| Refactoring: read breadcrumbs, zero dead code, verify full a | related | Working style: produce detailed documents alongside code cha | 0.5402310469663957 |
| Working style: produce detailed documents alongside code cha | related | Refactoring: read breadcrumbs, zero dead code, verify full a | 0.5402310469663957 |
| Python f-string gotcha: backslashes in {} blocks cause Synta | related | Bug: brain.py was missing 'import sys' — added for stderr lo | 0.5794137228160293 |
| Bug: brain.py was missing 'import sys' — added for stderr lo | related | Python f-string gotcha: backslashes in {} blocks cause Synta | 0.5794137228160293 |
| Python f-string gotcha: backslashes in {} blocks cause Synta | related | [bug] zsh treats unmatched globs as errors — killed every Ma | 0.5245018003654335 |
| [bug] zsh treats unmatched globs as errors — killed every Ma | related | Python f-string gotcha: backslashes in {} blocks cause Synta | 0.5245018003654335 |
| Brain bridging bug: _bridge_at_store_time and _find_bridge_c | related | v11 emergent graph bridging — implementation complete | 0.6366721850009867 |
| v11 emergent graph bridging — implementation complete | related | Brain bridging bug: _bridge_at_store_time and _find_bridge_c | 0.6366721850009867 |
| Brain bridging bug: _bridge_at_store_time and _find_bridge_c | related | brain_connections.py — edge management, bridging, graph stru | 0.6071276074556844 |
| brain_connections.py — edge management, bridging, graph stru | related | Brain bridging bug: _bridge_at_store_time and _find_bridge_c | 0.6071276074556844 |
| Video prompt rewritten: Hook→Showcase→CTA structure, industr | related | 7-second ad duration target for NanoBanana | 0.5191680204815662 |
| 7-second ad duration target for NanoBanana | related | Video prompt rewritten: Hook→Showcase→CTA structure, industr | 0.5191680204815662 |
| Tmemory design: store Claude's product dilemmas and opinions | related | Rule: every tmemory code/architecture change must be stored  | 0.6042873010055564 |
| Rule: every tmemory code/architecture change must be stored  | related | Tmemory design: store Claude's product dilemmas and opinions | 0.6042873010055564 |
| Tmemory v1.1.0: Curiosity system proactively detects gaps, p | related | Tmemory design: store Claude's product dilemmas and opinions | 1.0 |
| Tmemory design: store Claude's product dilemmas and opinions | related | Tmemory v1.1.0: Curiosity system proactively detects gaps, p | 0.5920868554293092 |
| Debug mode: Server-state based via /debug/* endpoints, not e | related | Brain server died again during v11 implementation — platform | 0.6139309100880137 |
| Brain server died again during v11 implementation — platform | related | Debug mode: Server-state based via /debug/* endpoints, not e | 0.6139309100880137 |
| Debug mode: Server-state based via /debug/* endpoints, not e | related | Decision: error/debug logs go to brain_logs.db, not brain.db | 0.5881486803590361 |
| Decision: error/debug logs go to brain_logs.db, not brain.db | related | Debug mode: Server-state based via /debug/* endpoints, not e | 0.5881486803590361 |
| [bug] zsh treats unmatched globs as errors — killed every Ma | related | [bug] Single quotes inside bash -c python break the shell —  | 0.6109813034454521 |
| [bug] Single quotes inside bash -c python break the shell —  | related | [bug] zsh treats unmatched globs as errors — killed every Ma | 0.6109813034454521 |
| Tom needs step-by-step terminal guidance without jargon | related | Session goal: simplify brain boot process for new users | 0.6285406718712616 |
| Session goal: simplify brain boot process for new users | related | Tom needs step-by-step terminal guidance without jargon | 0.6285406718712616 |
| Tom needs step-by-step terminal guidance without jargon | related | Added to Tom.md: always plan before executing. Tom discusses | 0.6262455984454554 |
| Added to Tom.md: always plan before executing. Tom discusses | related | Tom needs step-by-step terminal guidance without jargon | 0.6262455984454554 |
| Tmemory plugin: self-contained, brain.db fresh per user | related | tmemory: separate user brain from plugin (fresh vs personal) | 1.0 |
| tmemory: separate user brain from plugin (fresh vs personal) | related | Tmemory plugin: self-contained, brain.db fresh per user | 0.7 |
| Correction: tmemory is a general Claude plugin, not project- | related | Tmemory plugin: self-contained, brain.db fresh per user | 0.656503850427609 |
| Tmemory plugin: self-contained, brain.db fresh per user | related | Correction: tmemory is a general Claude plugin, not project- | 1.0 |
| Correction: Probe emotional signals before proceeding | related | Feedback: Be proactive about learning, not waiting for trigg | 0.4981664710717547 |
| Feedback: Be proactive about learning, not waiting for trigg | related | Correction: Probe emotional signals before proceeding | 0.4981664710717547 |
| Correction: Probe emotional signals before proceeding | related | Pattern: Emotional state during encoding biases confidence a | 0.49582322025575026 |
| Pattern: Emotional state during encoding biases confidence a | related | Correction: Probe emotional signals before proceeding | 0.49582322025575026 |
| Catch and fix demo UI regressions immediately | related | Tom wants working demos over mockups. 'A working basic produ | 0.5946170157043832 |
| Tom wants working demos over mockups. 'A working basic produ | related | Catch and fix demo UI regressions immediately | 0.5946170157043832 |
| Catch and fix demo UI regressions immediately | related | Adweek — Media industry news site. Demo use case #2 — online | 0.5299449589539593 |
| Adweek — Media industry news site. Demo use case #2 — online | related | Catch and fix demo UI regressions immediately | 0.5299449589539593 |
| Correction: Emergent graph bridging instead of embeddings fo | related | Bridge candidates: Require 2+ shared neighbors minimum | 0.6223969671303405 |
| Bridge candidates: Require 2+ shared neighbors minimum | related | Correction: Emergent graph bridging instead of embeddings fo | 0.6223969671303405 |
| Bridge candidates: Require 2+ shared neighbors minimum | related | Store-time bridging requires pre-existing neighborhood — col | 0.6030595878530094 |
| Store-time bridging requires pre-existing neighborhood — col | related | Bridge candidates: Require 2+ shared neighbors minimum | 0.6030595878530094 |
| Thought nodes: 2-3h half-life (fast decay for noise control) | related | Research: thought node half-life tuning — 3h may not be righ | 0.7 |
| Research: thought node half-life tuning — 3h may not be righ | related | Thought nodes: 2-3h half-life (fast decay for noise control) | 0.7 |
| Thought nodes: 2-3h half-life (fast decay for noise control) | related | Thought nodes: brain self-observations with 3h half-life, FI | 0.7 |
| Thought nodes: brain self-observations with 3h half-life, FI | related | Thought nodes: 2-3h half-life (fast decay for noise control) | 0.7 |
| [o_glonumbers] Properties | related | UX patterns applied: Pulsing, narrative data, keyboard queue | 0.5509484857070335 |
| UX patterns applied: Pulsing, narrative data, keyboard queue | related | [o_glonumbers] Properties | 0.5509484857070335 |
| My Glos dashboard: cards with thumbnail, status badge, publi | related | UX patterns applied: Pulsing, narrative data, keyboard queue | 0.5379750952075072 |
| UX patterns applied: Pulsing, narrative data, keyboard queue | related | My Glos dashboard: cards with thumbnail, status badge, publi | 0.5379750952075072 |
| resolve-brain-db.sh: shared DB resolver sourced by all hooks | related | Error pattern: Hook script path resolution in test environme | 0.5568496305498563 |
| Error pattern: Hook script path resolution in test environme | related | resolve-brain-db.sh: shared DB resolver sourced by all hooks | 0.5568496305498563 |
| Batch HTTP endpoint /pre-edit for hook optimization | related | Error pattern: Hook script path resolution in test environme | 0.5405395555173743 |
| Error pattern: Hook script path resolution in test environme | related | Batch HTTP endpoint /pre-edit for hook optimization | 0.5405395555173743 |
| Tom: Hidden optionality in UI, not in-your-face optional fie | related | Tom values component separation. When scope grows, break int | 0.5704606393895404 |
| Tom values component separation. When scope grows, break int | related | Tom: Hidden optionality in UI, not in-your-face optional fie | 0.5704606393895404 |
| Tom: Hidden optionality in UI, not in-your-face optional fie | related | Tom wants working demos over mockups. 'A working basic produ | 0.5683750220903935 |
| Tom wants working demos over mockups. 'A working basic produ | related | Tom: Hidden optionality in UI, not in-your-face optional fie | 0.5683750220903935 |
| Correction: Brain relearning execution — Agent tool foregrou | related | Brain server died again during v11 implementation — platform | 0.5590608462678189 |
| Brain server died again during v11 implementation — platform | related | Correction: Brain relearning execution — Agent tool foregrou | 0.5590608462678189 |
| Correction: Brain relearning execution — Agent tool foregrou | related | Persistent daemon: Unix socket server keeps Brain alive, hoo | 0.509272597136691 |
| Persistent daemon: Unix socket server keeps Brain alive, hoo | related | Correction: Brain relearning execution — Agent tool foregrou | 0.509272597136691 |
| Phase 0.5B fix: keyword-only fallback scores penalized by KE | related | [bug] Keyword-only fallback outranked embedding matches — pe | 0.7 |
| [bug] Keyword-only fallback outranked embedding matches — pe | related | Phase 0.5B fix: keyword-only fallback scores penalized by KE | 0.7 |
| tmemory context file search: Remove artificial prefixes to p | related | Context files: Pull → Check conflicts → Flag to user → Updat | 0.5376830928400556 |
| Context files: Pull → Check conflicts → Flag to user → Updat | related | tmemory context file search: Remove artificial prefixes to p | 0.5376830928400556 |
| Context files: Pull → Check conflicts → Flag to user → Updat | related | [bug] Plugin cache: ~/.claude/plugins/cache/ serves stale fi | 0.514744174393927 |
| [bug] Plugin cache: ~/.claude/plugins/cache/ serves stale fi | related | Context files: Pull → Check conflicts → Flag to user → Updat | 0.514744174393927 |
| tmemory context file search: Remove artificial prefixes to p | related | Tmemory: Must add uncaughtException and unhandledRejection h | 0.46841987232833304 |
| Tmemory: Must add uncaughtException and unhandledRejection h | related | tmemory context file search: Remove artificial prefixes to p | 0.46841987232833304 |
| Tmemory: Must add uncaughtException and unhandledRejection h | related | Repeating errors in skill get high priority | 0.4640308982972292 |
| Repeating errors in skill get high priority | related | Tmemory: Must add uncaughtException and unhandledRejection h | 0.4640308982972292 |
| Debug output: _debug_separator block for embedding engine st | related | Phase 0.5A: embedder.py loud failures + get_model_status() + | 0.6305254450320235 |
| Phase 0.5A: embedder.py loud failures + get_model_status() + | related | Debug output: _debug_separator block for embedding engine st | 0.6305254450320235 |
| Debug output: _debug_separator block for embedding engine st | related | Session #7 final log (2026-03-18): Brain v4.0.0 shipped | 0.5533518744096443 |
| Session #7 final log (2026-03-18): Brain v4.0.0 shipped | related | Debug output: _debug_separator block for embedding engine st | 0.5733518744096443 |
| Health check fix: query miss_log, not recall_log.outcome | related | [fn] _detect_patterns() — auto-create hypotheses from miss_l | 0.5082454930041232 |
| [fn] _detect_patterns() — auto-create hypotheses from miss_l | related | Health check fix: query miss_log, not recall_log.outcome | 0.5082454930041232 |
| Memory discipline: selective logging, not comprehensive trac | related | Correction: Tom had to prompt me to form memories about this | 0.5904392751252252 |
| Correction: Tom had to prompt me to form memories about this | related | Memory discipline: selective logging, not comprehensive trac | 0.5904392751252252 |
| Memory discipline: selective logging, not comprehensive trac | related | tmemory: separate user brain from plugin (fresh vs personal) | 0.5561850466083216 |
| tmemory: separate user brain from plugin (fresh vs personal) | related | Memory discipline: selective logging, not comprehensive trac | 0.5561850466083216 |
| Hebbian edge formation requires context() co-access, not ind | related | Decision: synthesize_session() harvests from DB, not just in | 0.48918899103606794 |
| Decision: synthesize_session() harvests from DB, not just in | related | Hebbian edge formation requires context() co-access, not ind | 0.48918899103606794 |
| Hebbian edge formation requires context() co-access, not ind | related | Qualifiers as edges not text — dependencies become traversab | 0.46158970436448016 |
| Qualifiers as edges not text — dependencies become traversab | related | Hebbian edge formation requires context() co-access, not ind | 0.46158970436448016 |
| User feedback: Avoid unnecessary logins (Tom prefers not to  | related | Tom principle: Dont self-censor — let the system handle it | 0.5079786002841226 |
| Tom principle: Dont self-censor — let the system handle it | related | User feedback: Avoid unnecessary logins (Tom prefers not to  | 0.5079786002841226 |
| Unconfirmed info stays contextual until earned through repet | related | Personal flag three modes: fixed (permanent fact), fluid (ev | 0.5504970852024247 |
| Personal flag three modes: fixed (permanent fact), fluid (ev | related | Unconfirmed info stays contextual until earned through repet | 0.5504970852024247 |
| Unconfirmed info stays contextual until earned through repet | related | Divergence: Brain encoding should be concise and well-organi | 0.5480995639920094 |
| Divergence: Brain encoding should be concise and well-organi | related | Unconfirmed info stays contextual until earned through repet | 0.5480995639920094 |
| tmemory v4: self-improvement via instrumented recall + evalu | related | [fn] auto_generate_self_reflection() — performance, failure, | 0.5842798815170693 |
| [fn] auto_generate_self_reflection() — performance, failure, | related | tmemory v4: self-improvement via instrumented recall + evalu | 0.5842798815170693 |
| Health check fix: query miss_log, not recall_log.outcome | related | Bug: recall() mutates DB via markAccessed — eval results var | 0.48874862528045565 |
| Bug: recall() mutates DB via markAccessed — eval results var | related | Health check fix: query miss_log, not recall_log.outcome | 0.48874862528045565 |
| tmemory v4.2: dream-time keyword enrichment + extractKeyword | related | Keyword enrichment sweep | 0.5428701586491527 |
| Keyword enrichment sweep | related | tmemory v4.2: dream-time keyword enrichment + extractKeyword | 0.5428701586491527 |
| Lesson: silent failures are the most dangerous class of bug | related | Tom principle: If it can hurt you silently, it needs a measu | 0.5188994227564462 |
| Tom principle: If it can hurt you silently, it needs a measu | related | Lesson: silent failures are the most dangerous class of bug | 0.5188994227564462 |
| Correction: tmemory is a general Claude plugin, not project- | related | tmemory: suggestion limit configurable (default 8) | 0.5924833012508418 |
| tmemory: suggestion limit configurable (default 8) | related | Correction: tmemory is a general Claude plugin, not project- | 0.5924833012508418 |
| tmemory: suggestion limit configurable (default 8) | related | Tom principle: Make parameters, not decisions | 0.5610475801447478 |
| Tom principle: Make parameters, not decisions | related | tmemory: suggestion limit configurable (default 8) | 0.5610475801447478 |
| Rule: Emotional context matters for brain quality | related | Rule: Semantic richness beats headlines | 0.6079849319645036 |
| Rule: Semantic richness beats headlines | related | Rule: Emotional context matters for brain quality | 0.6079849319645036 |
| Rule: Emotional context matters for brain quality | related | Knowledge without 'why' is dead memory | 0.4955064158455743 |
| Knowledge without 'why' is dead memory | related | Rule: Emotional context matters for brain quality | 0.4955064158455743 |
| Vibe.co research done. Patterns: single URL→auto-gen TV ad < | related | Auto-generate ad from onboarding URL (Vibe.co pattern) | 0.6259761779008135 |
| Auto-generate ad from onboarding URL (Vibe.co pattern) | related | Vibe.co research done. Patterns: single URL→auto-gen TV ad < | 0.6259761779008135 |
| Auto-generate ad from onboarding URL (Vibe.co pattern) | related | Vibe.co — Streaming/CTV ad platform. UX reference for Glo: U | 0.602166084702768 |
| Vibe.co — Streaming/CTV ad platform. UX reference for Glo: U | related | Auto-generate ad from onboarding URL (Vibe.co pattern) | 0.602166084702768 |
| Tmemory v1.7.0: 93.3% overall recall, CamelCase tokenization | related | tmemory v4.2: recall scoring overhaul — uncapped spread acti | 0.8268091867092885 |
| tmemory v4.2: recall scoring overhaul — uncapped spread acti | related | Tmemory v1.7.0: 93.3% overall recall, CamelCase tokenization | 0.6268091867092883 |
| Tmemory v1.7.0: 93.3% overall recall, CamelCase tokenization | related | suggest() rewrite: pool multiplier, edge-neighbor discovery, | 0.5303298635849076 |
| suggest() rewrite: pool multiplier, edge-neighbor discovery, | related | Tmemory v1.7.0: 93.3% overall recall, CamelCase tokenization | 0.5303298635849076 |
| Wallet system: 1 Glo Credit = $1 USD, persistent across camp | related | Glo Credits — 1:1 USD. Wallet via Stripe customer balance. B | 0.7 |
| Glo Credits — 1:1 USD. Wallet via Stripe customer balance. B | related | Wallet system: 1 Glo Credit = $1 USD, persistent across camp | 0.7 |
| Wallet system: 1 Glo Credit = $1 USD, persistent across camp | related | [o_credits] Properties | 0.7 |
| [o_credits] Properties | related | Wallet system: 1 Glo Credit = $1 USD, persistent across camp | 0.7 |
| [bug] zsh treats unmatched globs as errors — killed every Ma | related | Rule: Python 3.9 — no backslash escapes inside f-string curl | 0.4531166623258348 |
| Rule: Python 3.9 — no backslash escapes inside f-string curl | related | [bug] zsh treats unmatched globs as errors — killed every Ma | 0.4531166623258348 |
| tmemory v7 PreCompact and boot hooks: Staged learning with g | related | Confidence review queue at boot — surface unconfirmed learni | 0.5886253270758467 |
| Confidence review queue at boot — surface unconfirmed learni | related | tmemory v7 PreCompact and boot hooks: Staged learning with g | 0.5886253270758467 |
| Confidence review queue at boot — surface unconfirmed learni | related | Session goal: simplify brain boot process for new users | 0.46573432234480794 |
| Session goal: simplify brain boot process for new users | related | Confidence review queue at boot — surface unconfirmed learni | 0.46573432234480794 |
| Bridge weight system: Initial 0.15, bidirectional pairs, dec | related | v11 emergent graph bridging — implementation complete | 0.5749764287855681 |
| v11 emergent graph bridging — implementation complete | related | Bridge weight system: Initial 0.15, bidirectional pairs, dec | 0.5749764287855681 |
| Bridge weight system: Initial 0.15, bidirectional pairs, dec | related | brain_connections.py — edge management, bridging, graph stru | 0.5049700163537234 |
| brain_connections.py — edge management, bridging, graph stru | related | Bridge weight system: Initial 0.15, bidirectional pairs, dec | 0.5049700163537234 |
| Prompt AI: exclude competitor/other brand logos from generat | related | [stm:s48] LOCKED: Creative screen defaults to Upload tab (no | 0.6300905081501319 |
| [stm:s48] LOCKED: Creative screen defaults to Upload tab (no | related | Prompt AI: exclude competitor/other brand logos from generat | 0.6300905081501319 |
| Prompt AI: exclude competitor/other brand logos from generat | related | Moderation: AI pre-screen (risk score, flags, IAB brand safe | 0.5473927408416788 |
| Moderation: AI pre-screen (risk score, flags, IAB brand safe | related | Prompt AI: exclude competitor/other brand logos from generat | 0.5473927408416788 |
| Tmemory — Persistent brain engine for Claude (v4.2) | related | Tmemory v1.1.0: 8 typed edge types with selective decay, str | 0.5160425590609453 |
| Tmemory v1.1.0: 8 typed edge types with selective decay, str | related | Tmemory — Persistent brain engine for Claude (v4.2) | 0.5160425590609453 |
| Tmemory v1.1.0: 8 typed edge types with selective decay, str | related | tmemory scoring: 35% relevance + 30% recency + 25% emotion + | 0.4847501781241864 |
| tmemory scoring: 35% relevance + 30% recency + 25% emotion + | related | Tmemory v1.1.0: 8 typed edge types with selective decay, str | 0.4847501781241864 |
| Lesson: debugging 'brain not learning' required tracing thro | related | Lesson: locked rules '0 at boot' was hooks not firing, not a | 0.556283274588569 |
| Lesson: locked rules '0 at boot' was hooks not firing, not a | related | Lesson: debugging 'brain not learning' required tracing thro | 0.556283274588569 |
| v12 migration: 'thought' type added; CHECK constraint requir | related | post_migration_ops: flag data operations needed after algori | 0.502484524604576 |
| post_migration_ops: flag data operations needed after algori | related | v12 migration: 'thought' type added; CHECK constraint requir | 0.502484524604576 |
| v12 migration: 'thought' type added; CHECK constraint requir | related | [bug] remember() INSERT was missing confidence column | 0.48242458315299963 |
| [bug] remember() INSERT was missing confidence column | related | v12 migration: 'thought' type added; CHECK constraint requir | 0.48242458315299963 |
| Dream thought spawning: recency boost = 1 + 1/(1 + hoursAgo/ | related | Dream seeds weighted by connection count, dreams scored for  | 0.5607995679276928 |
| Dream seeds weighted by connection count, dreams scored for  | related | Dream thought spawning: recency boost = 1 + 1/(1 + hoursAgo/ | 0.5607995679276928 |
| Dream thought spawning: recency boost = 1 + 1/(1 + hoursAgo/ | related | Research: thought node half-life tuning — 3h may not be righ | 0.5427075174606489 |
| Research: thought node half-life tuning — 3h may not be righ | related | Dream thought spawning: recency boost = 1 + 1/(1 + hoursAgo/ | 0.5427075174606489 |
| Principle extraction: triggered by thinking corrections, not | related | Working style: produce detailed documents alongside code cha | 0.5214923652440836 |
| Working style: produce detailed documents alongside code cha | related | Principle extraction: triggered by thinking corrections, not | 0.5214923652440836 |
| Principle extraction: triggered by thinking corrections, not | related | Tom: "I practice my beliefs" | 0.5173767325897948 |
| Tom: "I practice my beliefs" | related | Principle extraction: triggered by thinking corrections, not | 0.5173767325897948 |
| tmemory context file search: Remove artificial prefixes to p | related | tmemory staged learning: Duplicate detection must use keywor | 0.5320658611948156 |
| tmemory staged learning: Duplicate detection must use keywor | related | tmemory context file search: Remove artificial prefixes to p | 0.5320658611948156 |
| tmemory staged learning: Duplicate detection must use keywor | related | [bug] duplicate JSON keys silently overwrite — hooks.json ha | 0.48994501344004493 |
| [bug] duplicate JSON keys silently overwrite — hooks.json ha | related | tmemory staged learning: Duplicate detection must use keywor | 0.48994501344004493 |
| Feedback: Prefix screen name to scope edits (prevent cross-s | related | Screen-scoped edits | 0.6880525590190422 |
| Screen-scoped edits | related | Feedback: Prefix screen name to scope edits (prevent cross-s | 0.9680525590190424 |
| Feedback: Prefix screen name to scope edits (prevent cross-s | related | Lesson: UI regression from Claude suggesting layout changes  | 0.56572205756587 |
| Lesson: UI regression from Claude suggesting layout changes  | related | Feedback: Prefix screen name to scope edits (prevent cross-s | 0.54572205756587 |
| Self-instrumentation: brain monitoring suggest() performance | related | Metrics telemetry: BrainMetrics tracks feature effectiveness | 0.5863872722102056 |
| Metrics telemetry: BrainMetrics tracks feature effectiveness | related | Self-instrumentation: brain monitoring suggest() performance | 0.5863872722102056 |
| Self-instrumentation: brain monitoring suggest() performance | related | Dynamic parameters in brain.py: thresholds that should be tu | 0.555159961078945 |
| Dynamic parameters in brain.py: thresholds that should be tu | related | Self-instrumentation: brain monitoring suggest() performance | 0.555159961078945 |
| Correction: Use current SKILL.md encoding rules as ground tr | related | Rule: every tmemory code/architecture change must be stored  | 0.5769736331716986 |
| Rule: every tmemory code/architecture change must be stored  | related | Correction: Use current SKILL.md encoding rules as ground tr | 0.5769736331716986 |
| Correction: Use current SKILL.md encoding rules as ground tr | related | Correction: Tom had to prompt me to form memories about this | 0.542731535081258 |
| Correction: Tom had to prompt me to form memories about this | related | Correction: Use current SKILL.md encoding rules as ground tr | 0.542731535081258 |
| Phase 0.5B fix: keyword-only fallback scores penalized by KE | related | Real engine simulation: embeddings critical for dedup, 0% wi | 0.528201299480512 |
| Real engine simulation: embeddings critical for dedup, 0% wi | related | Phase 0.5B fix: keyword-only fallback scores penalized by KE | 0.528201299480512 |
| Frontend polling persists stale job IDs after server restart | related | Server reliability: NOT unreliable — 3 crashes in 1099 calls | 0.5127397942208469 |
| Server reliability: NOT unreliable — 3 crashes in 1099 calls | related | Frontend polling persists stale job IDs after server restart | 0.4727397942208469 |
| Frontend polling persists stale job IDs after server restart | related | [bug] duplicate JSON keys silently overwrite — hooks.json ha | 0.45769583674951864 |
| [bug] duplicate JSON keys silently overwrite — hooks.json ha | related | Frontend polling persists stale job IDs after server restart | 0.45769583674951864 |
| Thoughts vs decisions/rules differ fundamentally in decay ph | related | Research: thought node half-life tuning — 3h may not be righ | 0.6194287009829904 |
| Research: thought node half-life tuning — 3h may not be righ | related | Thoughts vs decisions/rules differ fundamentally in decay ph | 0.6194287009829904 |
| Thoughts vs decisions/rules differ fundamentally in decay ph | related | Implemented pruning time-dilation with configurable active/i | 0.5660580118679058 |
| Implemented pruning time-dilation with configurable active/i | related | Thoughts vs decisions/rules differ fundamentally in decay ph | 0.5660580118679058 |
| User feedback: Avoid unnecessary logins (Tom prefers not to  | related | Communication style with Tom | 0.6712707665544788 |
| Communication style with Tom | related | User feedback: Avoid unnecessary logins (Tom prefers not to  | 0.6712707665544788 |
| Tom — CEO of EX.CO | related | Communication style with Tom | 0.6568812432724632 |
| Communication style with Tom | related | Tom — CEO of EX.CO | 0.6568812432724632 |
| Confidence recalibration pipeline | related | [fn] create_hypothesis() — untested belief with confidence s | 0.5378957152997964 |
| [fn] create_hypothesis() — untested belief with confidence s | related | Confidence recalibration pipeline | 0.5378957152997964 |
| Knowledge surfacing (activation) ≠ storage — activate at dec | related | Feedback: Be proactive about learning, not waiting for trigg | 0.497270531446137 |
| Feedback: Be proactive about learning, not waiting for trigg | related | Knowledge surfacing (activation) ≠ storage — activate at dec | 0.497270531446137 |
| Knowledge surfacing (activation) ≠ storage — activate at dec | related | Operational learnings decay naturally — dont lock, let reinf | 0.47368375950876307 |
| Operational learnings decay naturally — dont lock, let reinf | related | Knowledge surfacing (activation) ≠ storage — activate at dec | 0.47368375950876307 |
| Dream creates intuition nodes → triggers emergent bridging | related | Correction: Emergent graph bridging instead of embeddings fo | 0.6094940282016107 |
| Correction: Emergent graph bridging instead of embeddings fo | related | Dream creates intuition nodes → triggers emergent bridging | 0.6094940282016107 |
| Dream creates intuition nodes → triggers emergent bridging | related | v11 emergent graph bridging — implementation complete | 0.5740091058356657 |
| v11 emergent graph bridging — implementation complete | related | Dream creates intuition nodes → triggers emergent bridging | 0.5740091058356657 |
| Emails include real screenshots from actual publisher site — | related | Brain eval: 64.2% before → 93.3% after recall algorithm fixe | 0.5360675500703563 |
| Brain eval: 64.2% before → 93.3% after recall algorithm fixe | related | Emails include real screenshots from actual publisher site — | 0.5360675500703563 |
| Principles topic made configurable — brain_meta instead of h | related | Correction: tmemory plugin was GLO-specific, must be project | 0.6100239894669203 |
| Correction: tmemory plugin was GLO-specific, must be project | related | Principles topic made configurable — brain_meta instead of h | 0.6100239894669203 |
| Correction: tmemory is a general Claude plugin, not project- | related | Principles topic made configurable — brain_meta instead of h | 0.5861705469137444 |
| Principles topic made configurable — brain_meta instead of h | related | Correction: tmemory is a general Claude plugin, not project- | 0.5861705469137444 |
| Lambda argument bug in brain.py TEMPORAL_PATTERNS: some lamb | related | Bug: brain.py was missing 'import sys' — added for stderr lo | 0.5783554305233 |
| Bug: brain.py was missing 'import sys' — added for stderr lo | related | Lambda argument bug in brain.py TEMPORAL_PATTERNS: some lamb | 0.5783554305233 |
| Lambda argument bug in brain.py TEMPORAL_PATTERNS: some lamb | related | Monolith split: recovered details I compressed away during e | 0.5182819240436144 |
| Monolith split: recovered details I compressed away during e | related | Lambda argument bug in brain.py TEMPORAL_PATTERNS: some lamb | 0.5182819240436144 |
| Bridge formation: three paths with biological pacing (store- | related | Biological pace principle — dream bridges mature through del | 0.7 |
| Biological pace principle — dream bridges mature through del | related | Bridge formation: three paths with biological pacing (store- | 0.7 |
| Bridge formation: three paths with biological pacing (store- | related | v11 emergent graph bridging — implementation complete | 0.7 |
| v11 emergent graph bridging — implementation complete | related | Bridge formation: three paths with biological pacing (store- | 0.7 |
| Avoid compounding changes - prefer explicit/declarative appr | related | API Integration: Read docs first, plan before executing | 0.5277932333307 |
| API Integration: Read docs first, plan before executing | related | Avoid compounding changes - prefer explicit/declarative appr | 0.5277932333307 |
| Avoid compounding changes - prefer explicit/declarative appr | related | Working style: produce detailed documents alongside code cha | 0.4931316271234282 |
| Working style: produce detailed documents alongside code cha | related | Avoid compounding changes - prefer explicit/declarative appr | 0.4931316271234282 |
| Glo Numbers — Analytics detail per Glo. Views/day graph, cli | related | Glo AI ad variations: Pick 3 with light editing, category-re | 0.5780597093145157 |
| Glo AI ad variations: Pick 3 with light editing, category-re | related | Glo Numbers — Analytics detail per Glo. Views/day graph, cli | 0.5780597093145157 |
| Waymark — AI video gen, franchise/SMB focus. Recommended for | related | Glo AI ad variations: Pick 3 with light editing, category-re | 0.551583100879364 |
| Glo AI ad variations: Pick 3 with light editing, category-re | related | Waymark — AI video gen, franchise/SMB focus. Recommended for | 0.551583100879364 |
| Correction: tmemory v6→v7 hook architecture for automatic br | related | tmemory v7: automatic hooks — PreToolUse, PreCompact, improv | 1.0 |
| tmemory v7: automatic hooks — PreToolUse, PreCompact, improv | related | Correction: tmemory v6→v7 hook architecture for automatic br | 0.72 |
| Correction: tmemory v6→v7 hook architecture for automatic br | related | tmemory v7 PreCompact and boot hooks: Staged learning with g | 0.6460646643118255 |
| tmemory v7 PreCompact and boot hooks: Staged learning with g | related | Correction: tmemory v6→v7 hook architecture for automatic br | 0.6460646643118255 |
| Tom: Always add Todo file for personal thoughts; surface whe | related | Added to Tom.md: always plan before executing. Tom discusses | 0.6885327906288611 |
| Added to Tom.md: always plan before executing. Tom discusses | related | Tom: Always add Todo file for personal thoughts; surface whe | 0.6885327906288611 |
| Tom: Always add Todo file for personal thoughts; surface whe | related | Tom references competitor UX frequently. When he names a pro | 0.5650993291299596 |
| Tom references competitor UX frequently. When he names a pro | related | Tom: Always add Todo file for personal thoughts; surface whe | 0.5650993291299596 |
| Creatify brand safety: logo directive in video_prompt, not n | related | Video prompt rewritten: Hook→Showcase→CTA structure, industr | 0.4929244765410025 |
| Video prompt rewritten: Hook→Showcase→CTA structure, industr | related | Creatify brand safety: logo directive in video_prompt, not n | 0.4929244765410025 |
| Testing platform Phase 1: Golden Dataset + Snapshot Regressi | related | Performance node — brain tracks its own quality metrics as p | 0.5171363093068443 |
| Performance node — brain tracks its own quality metrics as p | related | Testing platform Phase 1: Golden Dataset + Snapshot Regressi | 0.5171363093068443 |
| Excel models MUST have zero formula errors | related | Email/Notification System — Activation progress, performance | 0.500572321529701 |
| Email/Notification System — Activation progress, performance | related | Excel models MUST have zero formula errors | 0.500572321529701 |
| Excel models MUST have zero formula errors | related | Glo P&L Model — 5yr financial model. Assumptions+P&L sheets. | 0.48703160216446817 |
| Glo P&L Model — 5yr financial model. Assumptions+P&L sheets. | related | Excel models MUST have zero formula errors | 0.48703160216446817 |
| Split responses: internal (Claude→brain) separate from exter | related | Mental model: Claude has TWO audiences — user (brevity) vs b | 0.5379954485036623 |
| Mental model: Claude has TWO audiences — user (brevity) vs b | related | Split responses: internal (Claude→brain) separate from exter | 0.5379954485036623 |
| Split responses: internal (Claude→brain) separate from exter | related | Divergence: Claude compresses when encoding to brain — treat | 0.5363357845746964 |
| Divergence: Claude compresses when encoding to brain — treat | related | Split responses: internal (Claude→brain) separate from exter | 0.5363357845746964 |
| Post-edit verification: Re-parse and confirm which screens c | related | Screen-scoped edits | 0.6082902475111116 |
| Screen-scoped edits | related | Post-edit verification: Re-parse and confirm which screens c | 0.6082902475111116 |
| Post-edit verification: Re-parse and confirm which screens c | related | Email/Notification System — Activation progress, performance | 0.4775888156857969 |
| Email/Notification System — Activation progress, performance | related | Post-edit verification: Re-parse and confirm which screens c | 0.4775888156857969 |
| Video prompt rewritten: Hook→Showcase→CTA structure, industr | related | Creatify: Brand safety via override_script field, not narrat | 0.5049523590291621 |
| Creatify: Brand safety via override_script field, not narrat | related | Video prompt rewritten: Hook→Showcase→CTA structure, industr | 0.5049523590291621 |
| Thought spawning: three trigger patterns | related | Thought nodes: brain self-observations with 3h half-life, FI | 0.6265246570870484 |
| Thought nodes: brain self-observations with 3h half-life, FI | related | Thought spawning: three trigger patterns | 0.6265246570870484 |
| Thought spawning: three trigger patterns | related | brain_dreams.py — graph walks + consolidation + thought spaw | 0.5775769102551872 |
| brain_dreams.py — graph walks + consolidation + thought spaw | related | Thought spawning: three trigger patterns | 0.5775769102551872 |
| Encoding extractor incomplete: missing concept, context, per | related | Encoding mode: Editor-to-Learner shift | 0.589801921679947 |
| Encoding mode: Editor-to-Learner shift | related | Encoding extractor incomplete: missing concept, context, per | 0.589801921679947 |
| Encoding extractor incomplete: missing concept, context, per | related | Monolith split: recovered details I compressed away during e | 0.5800420472794624 |
| Monolith split: recovered details I compressed away during e | related | Encoding extractor incomplete: missing concept, context, per | 0.5800420472794624 |
| Embedding model load: sync at startup (not lazy) | related | Graceful degradation: when embedder unavailable, fall back t | 0.5338040910061548 |
| Graceful degradation: when embedder unavailable, fall back t | related | Embedding model load: sync at startup (not lazy) | 0.5338040910061548 |
| Embedding model load: sync at startup (not lazy) | related | Tom: prefers honest costs over hidden ones — pay upfront, kn | 0.53212750717513 |
| Tom: prefers honest costs over hidden ones — pay upfront, kn | related | Embedding model load: sync at startup (not lazy) | 0.53212750717513 |
| Tom: Context must be precise, vetted, never inferred or vagu | related | Added to Tom.md: always plan before executing. Tom discusses | 0.7 |
| Added to Tom.md: always plan before executing. Tom discusses | related | Tom: Context must be precise, vetted, never inferred or vagu | 0.7 |
| Tom: Context must be precise, vetted, never inferred or vagu | related | Tom values component separation. When scope grows, break int | 0.6072659046956822 |
| Tom values component separation. When scope grows, break int | related | Tom: Context must be precise, vetted, never inferred or vagu | 0.6072659046956822 |
| Tmemory v1.1.0: Curiosity system proactively detects gaps, p | related | v2.3.0: Gap 9 — Frustration-to-rule extraction in curiosity  | 0.48184360777857954 |
| v2.3.0: Gap 9 — Frustration-to-rule extraction in curiosity  | related | Tmemory v1.1.0: Curiosity system proactively detects gaps, p | 0.48184360777857954 |
| Glo Creative Intelligence: LLM as Creative Director, not fix | related | Creative archetypes: infinite use cases (marriage proposals, | 0.7 |
| Creative archetypes: infinite use cases (marriage proposals, | related | Glo Creative Intelligence: LLM as Creative Director, not fix | 0.7 |
| Glo Numbers — Analytics detail per Glo. Views/day graph, cli | related | Creative archetypes: infinite use cases (marriage proposals, | 0.5743982870751894 |
| Creative archetypes: infinite use cases (marriage proposals, | related | Glo Numbers — Analytics detail per Glo. Views/day graph, cli | 0.5743982870751894 |
| Tom wants working demos over mockups. 'A working basic produ | related | Correction: dont hand user terminal commands — execute them  | 0.5313096423368944 |
| Correction: dont hand user terminal commands — execute them  | related | Tom wants working demos over mockups. 'A working basic produ | 0.5313096423368944 |
| Feedback: User values self-correction and double-checking du | related | User feedback: Avoid unnecessary logins (Tom prefers not to  | 0.5201414044530344 |
| User feedback: Avoid unnecessary logins (Tom prefers not to  | related | Feedback: User values self-correction and double-checking du | 0.5201414044530344 |
| Repeating errors in skill get high priority | related | Feedback: User values self-correction and double-checking du | 0.5123178210652388 |
| Feedback: User values self-correction and double-checking du | related | Repeating errors in skill get high priority | 0.5123178210652388 |
| Tom: Focus on one component/thread at a time, don't go wide  | related | Tom prefers: discuss and define before building. Sequence: f | 0.7 |
| Tom prefers: discuss and define before building. Sequence: f | related | Tom: Focus on one component/thread at a time, don't go wide  | 0.7 |
| Tom: Focus on one component/thread at a time, don't go wide  | related | Glo is closed-loop demand layer on EX.CO — not standalone DS | 0.6171558277104037 |
| Glo is closed-loop demand layer on EX.CO — not standalone DS | related | Tom: Focus on one component/thread at a time, don't go wide  | 0.6171558277104037 |
| Glo intent question: One-tap simple UX, not complex campaign | related | Contextual intent engine: infers creative direction from WHO | 0.5231597705336645 |
| Contextual intent engine: infers creative direction from WHO | related | Glo intent question: One-tap simple UX, not complex campaign | 0.5231597705336645 |
| Glo intent question: One-tap simple UX, not complex campaign | related | Glo Numbers — Analytics detail per Glo. Views/day graph, cli | 0.4804196497226415 |
| Glo Numbers — Analytics detail per Glo. Views/day graph, cli | related | Glo intent question: One-tap simple UX, not complex campaign | 0.4804196497226415 |
| Added to Tom.md: always plan before executing. Tom discusses | related | Tom: bias toward action over analysis — 'go for it' means bu | 0.5461779771857159 |
| Tom: bias toward action over analysis — 'go for it' means bu | related | Added to Tom.md: always plan before executing. Tom discusses | 0.5461779771857159 |
| Tom: Plan-first approach for bigger questions | related | Tom: bias toward action over analysis — 'go for it' means bu | 0.5215981032291213 |
| Tom: bias toward action over analysis — 'go for it' means bu | related | Tom: Plan-first approach for bigger questions | 0.5215981032291213 |
| Creatify: preview_list_async key parameters | related | Video prompt rewritten: Hook→Showcase→CTA structure, industr | 0.545630634312424 |
| Video prompt rewritten: Hook→Showcase→CTA structure, industr | related | Creatify: preview_list_async key parameters | 0.545630634312424 |
| API Integration: Read docs first, plan before executing | related | Correction: Research before writing any 3rd party API integr | 0.6985463694212976 |
| Correction: Research before writing any 3rd party API integr | related | API Integration: Read docs first, plan before executing | 0.6985463694212976 |
| Correction: Research before writing any 3rd party API integr | related | Separate API + Web architecture | 0.5937908300515475 |
| Separate API + Web architecture | related | Correction: Research before writing any 3rd party API integr | 0.5937908300515475 |
| Feature: Backup pruned memory snapshots for learning and imp | related | Prune archive enables learning from pruning behavior — instr | 0.7 |
| Prune archive enables learning from pruning behavior — instr | related | Feature: Backup pruned memory snapshots for learning and imp | 0.7 |
| Feature: Backup pruned memory snapshots for learning and imp | related | v11 post-launch monitoring: check bridge survival after 72h, | 0.5824559113987738 |
| v11 post-launch monitoring: check bridge survival after 72h, | related | Feature: Backup pruned memory snapshots for learning and imp | 0.5824559113987738 |
| Embedder: FastEmbed-m (768d, 110MB, <1s) vs bge-m3 (1024d, 5 | related | Feedback: bge-base-en-v1.5 performance — 80ms single, 30-40m | 0.6388556563877538 |
| Feedback: bge-base-en-v1.5 performance — 80ms single, 30-40m | related | Embedder: FastEmbed-m (768d, 110MB, <1s) vs bge-m3 (1024d, 5 | 0.6388556563877538 |
| Feedback: bge-base-en-v1.5 performance — 80ms single, 30-40m | related | Optimize boot speed — 15 sequential DB calls at session star | 0.5370550302771073 |
| Optimize boot speed — 15 sequential DB calls at session star | related | Feedback: bge-base-en-v1.5 performance — 80ms single, 30-40m | 0.5370550302771073 |
| Code-as-memory: Deep comments explain WHY concepts, not WHAT | related | Knowledge without 'why' is dead memory | 0.5610564676051466 |
| Knowledge without 'why' is dead memory | related | Code-as-memory: Deep comments explain WHY concepts, not WHAT | 0.5610564676051466 |
| Code-as-memory: Deep comments explain WHY concepts, not WHAT | related | Divergence: Brain encoding should be concise and well-organi | 0.5347992116023187 |
| Divergence: Brain encoding should be concise and well-organi | related | Code-as-memory: Deep comments explain WHY concepts, not WHAT | 0.5347992116023187 |
| API Integration: Read docs first, plan before executing | related | API Integration: Live API behavior supersedes documentation | 0.6797998032711261 |
| API Integration: Live API behavior supersedes documentation | related | API Integration: Read docs first, plan before executing | 0.6797998032711261 |
| API Integration: Live API behavior supersedes documentation | related | Tom wants working demos over mockups. 'A working basic produ | 0.5265657549521977 |
| Tom wants working demos over mockups. 'A working basic produ | related | API Integration: Live API behavior supersedes documentation | 0.5265657549521977 |
| Correction: tmemory recap encoding must happen automatically | related | Compaction experiment FAILED — recap encoding was skipped, a | 0.7032697453198009 |
| Compaction experiment FAILED — recap encoding was skipped, a | related | Correction: tmemory recap encoding must happen automatically | 0.6832697453198009 |
| Correction: tmemory recap encoding must happen automatically | related | Constraint: Claude compaction destroys unencoded knowledge — | 0.6763885479479028 |
| Constraint: Claude compaction destroys unencoded knowledge — | related | Correction: tmemory recap encoding must happen automatically | 0.6763885479479028 |
| Tmemory: Boot script includes npm install fallback for writa | related | tmemory v7 PreCompact and boot hooks: Staged learning with g | 0.6167591906239394 |
| tmemory v7 PreCompact and boot hooks: Staged learning with g | related | Tmemory: Boot script includes npm install fallback for writa | 0.6167591906239394 |
| Tmemory: Boot script includes npm install fallback for writa | related | tmemory: separate user brain from plugin (fresh vs personal) | 0.5698401762062578 |
| tmemory: separate user brain from plugin (fresh vs personal) | related | Tmemory: Boot script includes npm install fallback for writa | 0.5698401762062578 |
| [stm:s58] LOCKED: Glo owns publisher profiles (rich metadata | related | Glo.io — Self-serve ad platform on EX.CO ad server. Anyone b | 0.5548631207504315 |
| Glo.io — Self-serve ad platform on EX.CO ad server. Anyone b | related | [stm:s58] LOCKED: Glo owns publisher profiles (rich metadata | 0.5548631207504315 |
| [o_glo] Properties | related | [stm:s58] LOCKED: Glo owns publisher profiles (rich metadata | 0.5413067876634468 |
| [stm:s58] LOCKED: Glo owns publisher profiles (rich metadata | related | [o_glo] Properties | 0.5413067876634468 |
| stageLearning bug: bestOverlap undefined (should be bestSimi | related | Constraint: new methods must go in the right mixin, not brai | 0.48482685015110705 |
| Constraint: new methods must go in the right mixin, not brai | related | stageLearning bug: bestOverlap undefined (should be bestSimi | 0.48482685015110705 |
| stageLearning bug: bestOverlap undefined (should be bestSimi | related | Lesson: debugging 'brain not learning' required tracing thro | 0.45546504680709216 |
| Lesson: debugging 'brain not learning' required tracing thro | related | stageLearning bug: bestOverlap undefined (should be bestSimi | 0.45546504680709216 |
| Tom: Always present pros/cons and alternatives, challenge id | related | When Tom says 'not now' or 'don't want to go into it' — park | 0.6029250115199498 |
| When Tom says 'not now' or 'don't want to go into it' — park | related | Tom: Always present pros/cons and alternatives, challenge id | 0.6029250115199498 |
| Tom: Plan-first approach for bigger questions | related | Tom: Always present pros/cons and alternatives, challenge id | 0.5996392866051788 |
| Tom: Always present pros/cons and alternatives, challenge id | related | Tom: Plan-first approach for bigger questions | 0.5996392866051788 |
| Embedder: FastEmbed-m (768d, 110MB, <1s) vs bge-m3 (1024d, 5 | related | Correction: Embedding model is bge-m3, not bge-base-en-v1.5 | 0.6224675715292312 |
| Correction: Embedding model is bge-m3, not bge-base-en-v1.5 | related | Embedder: FastEmbed-m (768d, 110MB, <1s) vs bge-m3 (1024d, 5 | 0.6224675715292312 |
| Correction: Embedding model is bge-m3, not bge-base-en-v1.5 | related | Semantic recall history: embeddings vs bridging → both neede | 0.5102632672442495 |
| Semantic recall history: embeddings vs bridging → both neede | related | Correction: Embedding model is bge-m3, not bge-base-en-v1.5 | 0.5102632672442495 |
| Correction: Query brain for existing concepts before proposi | related | Failure: Claude forgot to query brain before proposing Layer | 0.6867965531427579 |
| Failure: Claude forgot to query brain before proposing Layer | related | Correction: Query brain for existing concepts before proposi | 0.6867965531427579 |
| Correction: Query brain for existing concepts before proposi | related | Correction: Tom had to prompt me to form memories about this | 0.5441205204472607 |
| Correction: Tom had to prompt me to form memories about this | related | Correction: Query brain for existing concepts before proposi | 0.5441205204472607 |
| Z-index stacking context: FadeIn parent layers, not child | related | Lesson: UI regression from Claude suggesting layout changes  | 0.4514350860068182 |
| Lesson: UI regression from Claude suggesting layout changes  | related | Z-index stacking context: FadeIn parent layers, not child | 0.4514350860068182 |
| Email/Notification System — Activation progress, performance | related | Review staged learnings with user | 0.5170563221444602 |
| Review staged learnings with user | related | Email/Notification System — Activation progress, performance | 0.5170563221444602 |
| Preview count: 6→3 (Creatify timeout mitigation) | related | Video prompt rewritten: Hook→Showcase→CTA structure, industr | 0.5551277833607111 |
| Video prompt rewritten: Hook→Showcase→CTA structure, industr | related | Preview count: 6→3 (Creatify timeout mitigation) | 0.5551277833607111 |
| Phase 0.5B fix: keyword-only fallback scores penalized by KE | related | v2.3.0: Semantic dedup threshold set to cosine 0.85 | 0.46347012947527194 |
| v2.3.0: Semantic dedup threshold set to cosine 0.85 | related | Phase 0.5B fix: keyword-only fallback scores penalized by KE | 0.46347012947527194 |
| Report only what you deem important; ask if unsure; abstract | related | User feedback: Avoid unnecessary logins (Tom prefers not to  | 0.5158517332699543 |
| User feedback: Avoid unnecessary logins (Tom prefers not to  | related | Report only what you deem important; ask if unsure; abstract | 0.5158517332699543 |
| Report only what you deem important; ask if unsure; abstract | related | Divergence: Brain encoding should be concise and well-organi | 0.5143859397439112 |
| Divergence: Brain encoding should be concise and well-organi | related | Report only what you deem important; ask if unsure; abstract | 0.5143859397439112 |
| Creatify — AI video gen API. $99/mo. URL→video. Recommended  | related_to | Glo.io — Self-serve ad platform on EX.CO ad server. Anyone b | 0.84 |
| Waymark — AI video gen, franchise/SMB focus. Recommended for | related_to | Glo.io — Self-serve ad platform on EX.CO ad server. Anyone b | 0.84 |
| Cloudinary — Video transcoding, AI smart cropping across asp | related_to | Glo.io — Self-serve ad platform on EX.CO ad server. Anyone b | 0.84 |
| Clerk — Auth/user management. All SSOs + Stripe integration. | related_to | Glo.io — Self-serve ad platform on EX.CO ad server. Anyone b | 0.86 |
| Vibe.co — Streaming/CTV ad platform. UX reference for Glo: U | related_to | Glo.io — Self-serve ad platform on EX.CO ad server. Anyone b | 0.82 |
| Tom — CEO of EX.CO | related_to | EX.CO — End-to-end video platform for publishers: CMS, ad se | 0.9400000000000001 |
| WDIV Local 4 / Graham Media — Detroit CTV news. Demo use cas | related_to | EX.CO — End-to-end video platform for publishers: CMS, ad se | 0.7999999999999999 |
| EX.CO — End-to-end video platform for publishers: CMS, ad se | related_to | WDIV Local 4 / Graham Media — Detroit CTV news. Demo use cas | 0.7999999999999999 |
| Adweek — Media industry news site. Demo use case #2 — online | related_to | EX.CO — End-to-end video platform for publishers: CMS, ad se | 0.9 |
| EX.CO — End-to-end video platform for publishers: CMS, ad se | related_to | Adweek — Media industry news site. Demo use case #2 — online | 0.9 |
| Fox Corp — Media conglomerate. EX.CO sales target — entry vi | related_to | EX.CO — End-to-end video platform for publishers: CMS, ad se | 0.9 |
| EX.CO — End-to-end video platform for publishers: CMS, ad se | related_to | Fox Corp — Media conglomerate. EX.CO sales target — entry vi | 0.9 |
| Glo P&L Model — 5yr financial model. Assumptions+P&L sheets. | related_to | Glo.io — Self-serve ad platform on EX.CO ad server. Anyone b | 0.76 |
| Glo.io — Self-serve ad platform on EX.CO ad server. Anyone b | related_to | Glo P&L Model — 5yr financial model. Assumptions+P&L sheets. | 0.76 |
| Glo P&L Model — 5yr financial model. Assumptions+P&L sheets. | related_to | EX.CO — End-to-end video platform for publishers: CMS, ad se | 0.8800000000000001 |
| EX.CO — End-to-end video platform for publishers: CMS, ad se | related_to | Glo P&L Model — 5yr financial model. Assumptions+P&L sheets. | 0.8800000000000001 |
| WDIV Local 4 / Graham Media — Detroit CTV news. Demo use cas | related_to | Glo.io — Self-serve ad platform on EX.CO ad server. Anyone b | 0.7 |
| Glo.io — Self-serve ad platform on EX.CO ad server. Anyone b | related_to | WDIV Local 4 / Graham Media — Detroit CTV news. Demo use cas | 0.7 |
| Adweek — Media industry news site. Demo use case #2 — online | related_to | Glo.io — Self-serve ad platform on EX.CO ad server. Anyone b | 0.74 |
| Glo.io — Self-serve ad platform on EX.CO ad server. Anyone b | related_to | Adweek — Media industry news site. Demo use case #2 — online | 0.74 |
| The Huddle Sports Bar — Fictional bar for DOOH demo. Use cas | related_to | Glo.io — Self-serve ad platform on EX.CO ad server. Anyone b | 0.7 |
| Glo.io — Self-serve ad platform on EX.CO ad server. Anyone b | related_to | The Huddle Sports Bar — Fictional bar for DOOH demo. Use cas | 0.7 |
| Email/Notification System — Activation progress, performance | related_to | Glo Numbers — Analytics detail per Glo. Views/day graph, cli | 0.9 |
| Glo Numbers — Analytics detail per Glo. Views/day graph, cli | related_to | Email/Notification System — Activation progress, performance | 0.9 |
| Moderation System — Two-layer: AI pre-screen (risk score, fl | related_to | EX.CO — End-to-end video platform for publishers: CMS, ad se | 0.9 |
| EX.CO — End-to-end video platform for publishers: CMS, ad se | related_to | Moderation System — Two-layer: AI pre-screen (risk score, fl | 0.9 |
| Email/Notification System — Activation progress, performance | related_to | Glo Lifecycle — State machine: Draft→Pending Review→Active→C | 0.9 |
| Glo Lifecycle — State machine: Draft→Pending Review→Active→C | related_to | Email/Notification System — Activation progress, performance | 0.9 |
| Email/Notification System — Activation progress, performance | related_to | EX.CO — End-to-end video platform for publishers: CMS, ad se | 0.9200000000000002 |
| EX.CO — End-to-end video platform for publishers: CMS, ad se | related_to | Email/Notification System — Activation progress, performance | 0.9200000000000002 |
| Email/Notification System — Activation progress, performance | related_to | Moderation System — Two-layer: AI pre-screen (risk score, fl | 0.9 |
| Moderation System — Two-layer: AI pre-screen (risk score, fl | related_to | Email/Notification System — Activation progress, performance | 0.9 |
| Glo Numbers — Analytics detail per Glo. Views/day graph, cli | related_to | EX.CO — End-to-end video platform for publishers: CMS, ad se | 0.9200000000000002 |
| EX.CO — End-to-end video platform for publishers: CMS, ad se | related_to | Glo Numbers — Analytics detail per Glo. Views/day graph, cli | 0.9200000000000002 |
| Tier pricing: 1.4x markup multiplier on publisher CPM | related_to | Daily recurring uses slider not tiers | 0.7800000000000001 |
| Daily recurring uses slider not tiers | related_to | Tier pricing: 1.4x markup multiplier on publisher CPM | 0.7800000000000001 |
| Session #8 (charming-cannon) achievements: hooks fixed, hear | summarizes | Session Log — Reset #8 (charming-cannon) | 0.9 |
| Session Log — Reset #8 (charming-cannon) | summarizes | Session #8 (charming-cannon) achievements: hooks fixed, hear | 0.92 |
| Lesson: debugging 'brain not learning' required tracing thro | traced | [bug] duplicate JSON keys silently overwrite — hooks.json ha | 0.95 |
| [bug] duplicate JSON keys silently overwrite — hooks.json ha | traced | Lesson: debugging 'brain not learning' required tracing thro | 0.95 |
| Lesson: debugging 'brain not learning' required tracing thro | traced | [bug] zsh treats unmatched globs as errors — killed every Ma | 0.95 |
| [bug] zsh treats unmatched globs as errors — killed every Ma | traced | Lesson: debugging 'brain not learning' required tracing thro | 0.9 |
| Lesson: debugging 'brain not learning' required tracing thro | traced | Lesson: v5 new session got lost because SKILL.md was stale | 0.95 |
| Lesson: v5 new session got lost because SKILL.md was stale | traced | Lesson: debugging 'brain not learning' required tracing thro | 0.95 |
| Glo.io — Self-serve ad platform on EX.CO ad server. Anyone b | uses | Creatify — AI video gen API. $99/mo. URL→video. Recommended  | 0.84 |
| Glo.io — Self-serve ad platform on EX.CO ad server. Anyone b | uses | Waymark — AI video gen, franchise/SMB focus. Recommended for | 0.84 |
| Glo.io — Self-serve ad platform on EX.CO ad server. Anyone b | uses | Cloudinary — Video transcoding, AI smart cropping across asp | 0.84 |
| Glo.io — Self-serve ad platform on EX.CO ad server. Anyone b | uses | Clerk — Auth/user management. All SSOs + Stripe integration. | 0.86 |

---

## Node ID Index

| ID | Type | Title |
|----|------|-------|
| `00ef7ca409de4416803760dfbf63a1e3` | decision | Embedding model load: sync at startup (not lazy) |
| `01fca6d58e354e539e82cd4f7bcf63b1` | decision | Glo backend stack: Node.js + Express, AWS infrastructure (RDS, Redis, DynamoDB) |
| `044db3eb35684577ae58becaffc0625c` | rule | tmemory context file search: Remove artificial prefixes to prevent false positiv |
| `04774e666b7045c59f995230b16051b1` | decision | Reduce previews to 1 for testing phase |
| `04ac11acccc84669b86305ebe6a5b0e5` | rule | Knowledge surfacing (activation) ≠ storage — activate at decision time |
| `054cff79f6a84cea9f83298ab1a96db9` | decision | Wallet system: 1 Glo Credit = $1 USD, persistent across campaigns |
| `05679b47d1124782bc0bb4200fcc23cc` | decision | Pre-compact extraction: prioritize information preservation over brevity |
| `05e2441b21c541299e1061524d7904a1` | decision | Plugin v2.4.0 release: canonical schema + density log reader + pre-compact simpl |
| `069c8f15ab2f41deaab6bb274f73ff19` | rule | Feedback: Discuss pros/cons BEFORE committing sensitive changes |
| `07b99f34f9cd4e7bae6b334bd577b4b6` | decision | Debug output: _debug_separator block for embedding engine stats |
| `07be165ae6614af7a94f0e90114dcf46` | decision | Health check fix: query miss_log, not recall_log.outcome |
| `094e6191ad094a00b828021e2ab417da` | decision | Earned permanence: nodes earn locked status through consolidation cycles |
| `09da61f446a5422fbdf8e7b031518417` | pattern | 📊 PATTERN — Corrections cluster: MCP tools missing: .mcp.json u (10 instances) |
| `0bed2f8b7f49451f9be30ef19ef1bc71` | rule | Publisher type case mapping: uppercase UI → lowercase backend |
| `0db73af733d54134a19ac65d80ae04ae` | rule | Post-edit verification: Re-parse and confirm which screens changed |
| `100b74fc627a42c8925866abad6b1955` | decision | Approach B chosen: Self-healing hooks via ensure-brain.sh |
| `1073f95b99dd4990b84f65392dbce575` | thought | Dream connection: "Magnite — US adtech. Closest comparable to Geniee " and "[ctx |
| `11a167b68b5648d1bb7dac1dcd89bbf2` | purpose | brain.py — thin assembler + core infrastructure hub |
| `11ee43410e744d3284b6eda0d67e39cd` | rule | Tom: Plan-first approach for bigger questions |
| `1332d6fbc2204baab8db2b9f72b49d89` | rule | Destructive operations require context-awareness — execution environment matters |
| `1387b4aa72e94ef9ac25427dd26dcfe7` | decision | Correction: SQLite GROUP_CONCAT(DISTINCT) doesn't accept separator parameter |
| `1420a8de24b84053b7e6910f598ae2ca` | decision | Places API: Nominatim (not Google Places) |
| `14fb708032a04638a8a6efc3b83d7f53` | rule | Avoid compounding changes - prefer explicit/declarative approaches |
| `15ed160cd80641889caeba082132f768` | decision | Share button primary in confirmation screen |
| `198cfd73213046b1b5d4f76e72dee9af` | decision | Feature: Backup pruned memory snapshots for learning and improvement |
| `1c60283bb101416c905a9f6efbf79525` | rule | Frontend polling persists stale job IDs after server restart |
| `1dfdb6186efa4cb58334044bee971c2b` | rule | Context files: Pull → Check conflicts → Flag to user → Update both |
| `1eb576a7c125477c8de88a28824310e8` | decision | 3-layer architecture: media-intelligence → creative-director → creatify-adapter |
| `2045cad986554293b09fb59182f51a66` | decision | Video specs: H264, 640x360, 6-second default |
| `20a2cae807f94c2da1cc05551e1c69b3` | decision | Semantic similarity over keyword overlap dedup |
| `21e2b9470c224a17b275a0b2349a3ff6` | decision | Tmemory v1.1.0: Curiosity system proactively detects gaps, prompts learning |
| `2244325e98d44683b094c617568ce125` | rule | Creatify: model_version='aurora_v1_fast' required in preview_list_async requests |
| `259b8e7d3e484accaf8cda2e7f1a4109` | decision | Confirm: Glow animation (brand-specific, not confetti) |
| `2733c98cc8aa484f97ad4a3c36f142a7` | rule | Feedback: Prefix screen name to scope edits (prevent cross-screen changes) |
| `275ea4f4cce74ad6a32383a4ca77236c` | rule | Pre-compact brain healthcheck: 6 fetches per session (verified in testing) |
| `29432270e51d4109a085006b524a8bd5` | mental_model | Entities are everything with identity: people, products, components, screens, ar |
| `29b797a0563a4bd7ac329946ef654db3` | rule | tmemory staged learning: Duplicate detection must use keyword overlap, not subst |
| `2ab5d143e3374a63ba6bd2a279f2173b` | decision | Correction: tmemory is a general Claude plugin, not project-embedded |
| `2ae80feee6684d148f59f4538f7fe9cd` | decision | System arch: Glo (users/billing/creative/moderation) ↔ EX.CO (delivery/reporting |
| `2d3f26d6de81421384ecc4029c7f53fd` | decision | Drafts persisted to Glo board |
| `2ef9f4551893402b848677faad4d458b` | decision | Glo target: SMB to micro businesses to individuals (anyone) |
| `2f00c36a10b94863b491017b0178c7af` | rule | Dream creates intuition nodes → triggers emergent bridging |
| `2f43107c98f147099464bef5c21f0945` | decision | Tier pricing: Well($30), Bright($50), Shine($100) |
| `306e3574310e43c68b0df15c8a1fcc09` | rule | stageLearning bug: bestOverlap undefined (should be bestSimilarity) |
| `31205dd8aad844db88bd9ec4f18ea547` | decision | Fix: contextual penalty must apply to blended_score BEFORE sort, not effective_a |
| `322fab5748b84ae085b37e7a69e97a54` | decision | Correction: tmemory v6→v7 hook architecture for automatic brain integration |
| `323d1ff64e27457993827985236967a4` | rule | Python f-string gotcha: backslashes in {} blocks cause SyntaxError |
| `32f10cfe18a34391984d61e5a015233b` | rule | Tmemory: Boot script includes npm install fallback for writable locations |
| `377d9f60f518428e8dd0d9dfde3d7165` | decision | Glo: NanoBanana adapter fixed—Bearer + .php + singular image_url |
| `37d80cffd443470aa6c9ed94c975fbff` | decision | Cancelled Creatify subscription |
| `38ded342d49842b19715b8343d02df1e` | decision | tmemory v7 PreCompact and boot hooks: Staged learning with graceful initializati |
| `391c7a35ec70439e9932f36e8966fbdb` | decision | Brain v4: Self-reflection node types — performance, failure, capability, interac |
| `3ac2ce2a883044258be10964c1ff5cde` | thought | Cluster forming: "Glo project history — research, pivot, build phases (March 202 |
| `3d892f8c3d5f493e97e05281742f71be` | rule | Tom: Context must be precise, vetted, never inferred or vague |
| `402184ea3115405f8115d69cfbfc43df` | rule | Blocker: Creatify preview_list_async returns 400 'Invalid pk' |
| `439ca46a99484e018951ad18b04e25a3` | decision | No avatars for now |
| `44e0d24b5b7c415091e7b727bd7f347c` | rule | Tom: Prioritizes proven track record in research and solutions |
| `4815ba61645f4db6b1e7700c4bd28178` | lesson | Lesson: silent failures are the most dangerous class of bug |
| `4a9d6eb901da4bdfb474444ae04bf39f` | decision | Creatify endpoint: `/api/links/` for link creation |
| `4b158d2836294cb9ba414126e8da5267` | lesson | Lesson: debugging 'brain not learning' required tracing through 3 independent fa |
| `4c116c7f0fbb4eaab8eceae85a808ba9` | rule | Tom: designer's eye despite engineering background — sees products as users woul |
| `4eef853a974542c4a158732e40a1a66c` | rule | Creatify API: preview_list_async endpoint — cost and response |
| `503d38410732495ca524924794ed297e` | decision | Tmemory v1.7.0: 93.3% overall recall, CamelCase tokenization fix |
| `50a6056befd149b3ad7ce67492fafb74` | decision | Preview count: 6→3 (Creatify timeout mitigation) |
| `513a4e6232704122bb2aa731188c24c7` | intuition | Dream: Email/Notification System — Activation p ↔ Creative strategy: AI video ge |
| `5200eac858f14407afc7cfcde3b79d15` | rule | NanoBanana API: All endpoints require .php extension |
| `528871e093c64694bd1bd3a6accac3d2` | decision | Creatify: Switch from 3-full-renders to preview-first workflow |
| `52a8c4e6709c47ae85a80939847cf287` | decision | Correction: Probe emotional signals before proceeding |
| `55a355146b234b2090d4daf2cf0a4643` | decision | Re-light: One-click respend for active Glos |
| `55a71114d79049f39be23329f8b3c0e6` | intuition | Dream: Brain Evolution Roadmap — 3 prioritized  ↔ Open: Embedding strategy for s |
| `570e85d5ebd64be1830781944d206515` | decision | Google Maps Photos as image source for MVP (replaces website scraping) |
| `58b0072a4b7b4dff992285d3ae85ce43` | rule | Lambda argument bug in brain.py TEMPORAL_PATTERNS: some lambdas called with matc |
| `5918f68ccffb484288ecb214f0814eb5` | decision | Tmemory plugin: Bundle selective node_modules, not full directory (1.1M not 19M+ |
| `59e57c0f5f164a4dac190a09bff5eea7` | decision | Z-index stacking context: FadeIn parent layers, not child |
| `5ad314e160b64107a4addbdfd9491374` | decision | Brain Python: ~5700 lines, three files, synchronous |
| `5b415202ce3a4019a4bd1a32352f2d72` | rule | FastEmbed: synchronous API—remove all await calls |
| `5b78eb77c80f4d7d83d2aa00775068b2` | rule | Refactoring: read breadcrumbs, zero dead code, verify full alignment |
| `5c1791f3744e4e4fba7339528658561e` | rule | Tom needs step-by-step terminal guidance without jargon |
| `5d90e967164f49d3bc1d4010ba94a5f9` | rule | Tmemory: Must add uncaughtException and unhandledRejection handlers to prevent s |
| `5dd9ad5761434c809154d7e5980f47cb` | project | Glo project history — research, pivot, build phases (March 2026) |
| `61a8562012544411b10644a2524f852a` | rule | Confidence review queue at boot — surface unconfirmed learnings |
| `62710fe825f242a99c8e63a6aac8e42a` | decision | Correction: Brain relearning execution — Agent tool foreground, not nohup backgr |
| `62b31bba0c7b4fbe8810d10d54ee4492` | decision | Tmemory: JSONL format for cross-project memory DB |
| `62d1024bf6194e1aa115baca6847375e` | rule | Creatify model_version costs: standard 5cr/30s, aurora_v1 20cr/15s, aurora_v1_fa |
| `648207941cc647968170540aa5780d00` | rule | Rule: Emotional context matters for brain quality |
| `65a9f61306bd4b15a3902c6a102104d3` | decision | Brain self-generates thought nodes (ideation engine, not just documentation) |
| `662e221e1ce64266a50e73db1df786c2` | intuition | Dream: Magnite — US adtech. Closest comparable  ↔ [ctx:glo-platform] Glo.io — Se |
| `664bbbaee2544c69aed90680c125c4bb` | rule | Creatify: linkData.id (not linkData.link.id) for preview API |
| `683eb2943f70410bb16ea6b6c7022e03` | decision | Implement embedding-based retrieval to fix tmemory's biggest weakness |
| `6864632666bb4d0c993c18d8c04fdbf4` | rule | Encoding extractor incomplete: missing concept, context, person, project, task n |
| `68c93af1d8e847f18e17a07887e35aa6` | decision | Phase 0.5B fix: keyword-only fallback scores penalized by KEYWORD_FALLBACK_WEIGH |
| `69eaca7fe30c40e09ccb40dcc71aa6ab` | project | Glo component map — 18 components, build status, key decisions |
| `6b2986b2d5e34e31a19216aa429e9b29` | rule | Repeating errors in skill get high priority |
| `6bd80c8020754744a5319c4bb0e7d662` | aspiration | 🌱 ASPIRATION — Growing energy around Glo (emotion +0.36) |
| `6cd874213bc24f73bd06d84c4bf0f829` | mental_model | Entities are everything with identity: people, products, components, screens, ar |
| `6d4446e83e14438bb98a584e18279b11` | decision | Embedder: FastEmbed-m (768d, 110MB, <1s) vs bge-m3 (1024d, 560MB, 3-5s) |
| `6d87037fa4fd442aaacc365d996cd5b7` | mechanism | resolve-brain-db.sh: shared DB resolver sourced by all hooks, zsh-safe |
| `6da8710659eb4291a6b65327426479a8` | decision | Dream thought spawning: recency boost = 1 + 1/(1 + hoursAgo/24) |
| `702b0f4cbc294d1bb813db37f8a60595` | rule | Creatify brand safety: logo directive in video_prompt, not narration |
| `7065e92e2c504a04af33060be0635778` | rule | Principle extraction: triggered by thinking corrections, not code fixes |
| `71741847e7454765b3a717670ebb9cd2` | rule | Catch and fix demo UI regressions immediately |
| `7265733e661a4ee0968f29044205eafe` | decision | Creative: Vibe selector feeds AI generation prompt |
| `7458c87fc0004b188c2ba9b3201c5573` | decision | Batch HTTP endpoint /pre-edit for hook optimization |
| `773b769f6087486e96daa3533cda0874` | decision | Tmemory plugin: self-contained, brain.db fresh per user |
| `7b8d364996994641b60c60ca5f13c7de` | rule | Code-as-memory: Deep comments explain WHY concepts, not WHAT code does |
| `7b8f1683ac2947568cb2f6382f346af4` | decision | Budget: GLO Brightness slider $30–$500 (default $30) |
| `7be6b8c80d694568a9a7d2227a89a4d7` | rule | React Hook Violation: Cannot call useState/useEffect inside conditionals |
| `7c4542c56a1a44958e9116977e480081` | rule | Script approach object must include key field (creative-director.js) |
| `7ef967e28c774923b3f76cd51ade4990` | decision | Knowledge without 'why' is dead memory |
| `7f7026d9ea8f4a7294d08d5b350dd868` | rule | Brain bridging bug: _bridge_at_store_time and _find_bridge_candidates not creati |
| `8050be1c9786434095fa87b07202d866` | decision | Consolidation bridge discovery: 50% recent nodes + 50% random |
| `811e073831d14673ba36fd27ef007a03` | rule | Creatify: preview_list_async key parameters |
| `82dd867f4f8c4b13a6f871ca2a20f4bc` | decision | Preview-first workflow: 6 previews, user selects, then render |
| `86fb88faca1d4461b71290d4e1530efa` | decision | Generic defaults for project/user — env vars override hardcoded values |
| `870f7f8372354212a93cb32c9f3efd31` | context | Session #8 (charming-cannon) achievements: hooks fixed, heartbeat built, 41 exce |
| `8725ad03cc5f4765816384861a1ab511` | rule | Hebbian edge formation requires context() co-access, not individual recall() cal |
| `8792fa0e55504830b07f03a369185983` | rule | Rule: Preserve numbers, proper nouns, names in keywords |
| `87d16fad9695437795ff2eb8632c0770` | tension | ⚡ TENSION — Creative archetypes: infinite use cases  vs Glo Creative Intelligenc |
| `89835c525d3b4168a795ccaeec947ed8` | decision | Testing platform Phase 1: Golden Dataset + Snapshot Regression |
| `8a97454d3b114b19866d56531700b276` | decision | Bridge formation: three paths with biological pacing (store-time, consolidation, |
| `8bd79e5453784fc6b59c2fff791cc443` | decision | Thought spawning: three trigger patterns |
| `8be8e1721484429b9abd078e1bf499c6` | bug_lesson | [bug] zsh treats unmatched globs as errors — killed every Mac boot |
| `8cd0a7b14fb448f0965d40aa25e2d673` | intuition | Dream: Rule: ask for confirmation before manipu ↔ Glo Creative Intelligence: LLM |
| `8cd9d6f5be48440ab36fb5ab4896a86c` | decision | Correction: Emergent graph bridging instead of embeddings for semantic recall |
| `8e41beb587064c1381ba3bd47b2ae679` | rule | context-file/find: tag-based matching insufficient for discovery |
| `8f32ec71f28845c193d634dff25f2d6d` | decision | Glo intent question: One-tap simple UX, not complex campaign objectives |
| `9013d8550f8f415d9d69168aa46367a3` | decision | tmemory v1.6.0: configurable parameters infrastructure |
| `903daa840be84b4abf608bc7c04409e8` | rule | Memory discipline: selective logging, not comprehensive tracking |
| `9187cf0fd4bc466da9e68fd5fb31948b` | decision | Campaign terminology: renamed to 'active Glo' |
| `918bd3f512e54dfeadbb7427d3844246` | hypothesis | 🔮 HYPOTHESIS — Dream insight gaining traction: Dream: [o_glo] Flywheel: unfilled |
| `943bfe06bfed4a689381978e0955d862` | decision | Correction: Embedding model is bge-m3, not bge-base-en-v1.5 |
| `9558e8de739547a6a38ec7228507676e` | decision | NanoBanana: Multi-image required for Glo (non-negotiable) |
| `97e9a7994be647fd8e7c0fc632819a30` | rule | Tom: Focus on one component/thread at a time, don't go wide and deep simultaneou |
| `98e15f8c7f32467995eb4db261ca6fd3` | rule | Google Places autocomplete: `types:'establishment'` filter too restrictive |
| `98f3fc5f52e44c6fa6dbab50f959cbef` | rule | Hooks fire successfully; encoding sparseness was strategy problem, not infrastru |
| `99b6060327c048279de67c781a856a01` | decision | Creatify preview_list_async: model_version='aurora_v1_fast', poll individual pre |
| `9c108f90b80444328cd16363d26a8680` | decision | Glow icons: Enhanced glow effect, brighter when selected |
| `9d3f8b024562458f99a04b55c491ba7b` | thought | Cluster forming: "Cluster forming: "How to pass _recall_log_id from pre-response |
| `9d7e694c35b340ef913424ae1b6c3a12` | decision | Tmemory design: store Claude's product dilemmas and opinions as brain nodes |
| `9ed151e69b2a40a6be8792e8ff4cebe8` | rule | Creatify link ID format: 400 invalid pk error (active blocker) |
| `9fd782b41ae64ba0a138dd0229485e73` | decision | Geolocation-based location bias for Places search |
| `9fe1d89c711b4a5e94b7af1a0f4bb46a` | rule | Tom: Always present pros/cons and alternatives, challenge ideas |
| `a11768af462e4ba5a65da4da511b5655` | rule | API Integration: Read docs first, plan before executing |
| `a226c553079f429fa5a5b0dc10a28835` | rule | API Quirk: NanoBanana docs claim image_urls array, live API rejects it |
| `a4384ceba4cb43f088d6b0a7adeb7b77` | rule | Rule: Ask clarifying questions during encoding |
| `a65a8e2cb8aa4cf3b8b0db2356da8bf4` | decision | Vendor: Drop Creatify, use NanoBanana Video + Google Maps instead |
| `a65b6e7a54054cdeb189973d937d3d15` | rule | NanoBanana API: Authorization Bearer header required, not X-API-Key |
| `ab17fe309ff3440594966662a8822807` | rule | Store-time encoding must be smart, not exhaustive |
| `ab61fa43862f4fb1aafa01bb3f356f57` | param_influence | [param] KEYWORD_FALLBACK_WEIGHT=0.10 — keyword precision for exact matches only |
| `ab6fef5436f74e65b1408363a0e000c7` | rule | Graceful degradation: when embedder unavailable, fall back to TF-IDF |
| `ac4c6634ee994e499f5bf16feab883b5` | rule | API Integration: Live API behavior supersedes documentation |
| `ad09d8c0303444e090d167bf75bcfe33` | rule | Glo video ads: no avatars |
| `adca1200490a48e9933b921bd08b5a6a` | decision | Real engine simulation v1: 375 messages via Brain.js actual code |
| `ae249d11a78847dbab0ac2c9f1cb394d` | decision | Brand language: 'Light your Glo' and 'Glo is lit! 🔥' |
| `af8e38e3817046fe87c70ba130369616` | decision | Thought decay uses wall-clock time, not time-dilation |
| `b1b5fbe2b738414aa9283a551091a655` | decision | Plugin build: explicit include list in build-plugin.sh (replace exclusion-based  |
| `b242e7b495d54b0a92a4aefee74af6cb` | decision | v12 migration: 'thought' type added; CHECK constraint requires table rebuild |
| `b26afb2a566b4dd3a9e4ede1f77dc5d7` | decision | Prompt AI: exclude competitor/other brand logos from generated ads |
| `b3a810dc88e24e6885a9e7db124d0800` | rule | Browser fetch to Creatify API blocked by CORS — use server-side proxy instead |
| `b6411c400cc44e509810b3e18105d5c6` | intuition | Dream: Correction: Work approach — deep synthes ↔ Glo Creative Intelligence: LLM |
| `b6e1a2c7348a41acbcee6fb16fa8ef45` | rule | Excel models MUST have zero formula errors |
| `b9283b0eea984e439964a0e4eea8be80` | decision | Thought half-life: 3 hours wall-clock |
| `b9556fa53df34ec686193a83a8adef4a` | rule | Tom: Hidden optionality in UI, not in-your-face optional fields |
| `bbef31679064488fbbb67d41e08f65e6` | decision | Fix: server.js — extract linkData.id (top-level), remove linkData.link?.id fallb |
| `bce1d53c03a34926a5132e0de89ebce1` | decision | Google Places integration: server-side proxy for Places-only keys |
| `bd52cc0867214aba907f1870a20525f7` | pattern | 📊 PATTERN — Always used together: 'Tier pricing: 1.4x markup mult' + 'Glo compon |
| `bd6456db5907495cae91b9ad8b43bd74` | context | Session Log — Reset #8 (charming-cannon) |
| `be8b5e1e6b3742f081920c0f8664e99c` | rule | Feedback: User values self-correction and double-checking during debugging |
| `c0bc0e4517da47ad986181025a57aae1` | tension | ⚡ TENSION — Embeddings-first recall vs generic nodes dominating results |
| `c3201879d6f94b81b4f3644db9cadf1a` | decision | Bridge lifecycle: weight 0.15 initial, emergent_bridge edge type, 72-hour half-l |
| `c32cc95d93c241afbd1361a6b0825701` | decision | tmemory: suggestion limit configurable (default 8) |
| `c4794ea9776348d9a7679f0fced4d9ba` | rule | Error pattern: Hook script path resolution in test environment |
| `c4996043b31b484b9969c35a5265be56` | rule | Creatify POST /api/link_to_videos response fields: video_output, video_thumbnail |
| `c4fb9b6eae8245ac983fbd292e72d9da` | decision | Schema refactoring: canonical schema.js replaces migration chain (v0→v13) |
| `c5627e43f4b4441694d504e16f333569` | aspiration | 🌱 ASPIRATION — Repeated energy: [o_glo] Rule: budget_order (2 events, avg emotio |
| `c576520779434e778dbd43cdcd0d0068` | concept | Redirect learning: mutual improvement through principle extraction |
| `c5fdc9ad3c8b4e80a0066557b205ba85` | mechanism | Confidence recalibration pipeline |
| `c61eb6a28dd6465ba8b599eaddfbbf01` | rule | Tom: Always add Todo file for personal thoughts; surface when contextually relev |
| `c8b5dbc072284d91a4735f4f2dae008e` | hypothesis | 🔮 HYPOTHESIS — Dream insight gaining traction: Dream: Graph bridging > embedding |
| `cb06127f4351437888ab227660541396` | intuition | Dream: [period:2026-03:p5] Fox Corp penetration ↔ Glo.io — Self-serve ad platfor |
| `cb2c1410542441b1a79f1bbb4843c3c1` | decision | Correction: Research before writing any 3rd party API integration code |
| `cb454be9e0794608ab28fdebe8e2ddb7` | decision | Remove curiosity cap, extend encoding time budget |
| `con_05khaebd` | context | Created Tmemory skill + initialized memory system. Todo rule added to Tom.md. |
| `con_0qfhpayv` | concept | [o_glo] Properties |
| `con_140i0soc` | context | P&L needs EX.CO-specific assumptions (unfilled inventory economics, publisher re |
| `con_1b4au6hz` | context | Apple Pay fees concern raised. Key question: web app vs native iOS app determine |
| `con_1c9nsnhl` | concept | EX.CO publisher CMS supports custom ad slots |
| `con_2jl3lv9n` | context | New components emerging: My Glos Dashboard, Glo Lifecycle state machine, expande |
| `con_2mc9a66k` | context | Revised glo-demo.jsx: hook has publisher logo+media desc, onboarding is Vibe-sty |
| `con_3925kfrz` | context | Overnight batch 3: Created 4 lifecycle email templates (/glo/Brand/glo-email-tem |
| `con_3mxm4f94` | concept | [o_antifraud] Properties |
| `con_4oqp2i8d` | concept | [o_glonumbers] Properties |
| `con_5ej15u5g` | concept | Email/Notification System — Activation progress, performance updates with real s |
| `con_5l0e4p9v` | concept | GLO Brightness — Pricing tiers: Well($30) Bright($50) Shine($100). Publisher set |
| `con_6bke3oij` | context | Overnight batch 1: Created Mobile UX & Anti-Fraud research doc (/glo/Research/). |
| `con_6iueqb0z` | concept | Transcoding: Cloudinary recommended. AI smart cropping across aspect ratios. ~$3 |
| `con_7far6rky` | concept | Glo Numbers — Analytics detail per Glo. Views/day graph, clicks, QR scans, budge |
| `con_7sp3ziq7` | concept | Mobile Capture & Anti-Fraud — Prevent fake logins/bots. Payment gate as primary  |
| `con_8ts0mg3t` | context | Session Log — Reset #4 |
| `con_9wgrg928` | concept | [o_exco] Properties |
| `con_amakllaz` | context | Fixed 6 gaps in demo: (1) Auto-trigger AI gen via useEffect when URL provided, ( |
| `con_aofbtzix` | concept | App state model: screen-based routing via Context API |
| `con_b4g2ux5s` | concept | Glo Lifecycle — State machine: Draft→Pending Review→Active→Completed. Branches:  |
| `con_clbi4xy3` | context | Session Log — Reset #5 |
| `con_e36pbfyn` | context | Tom defining My Glos dashboard: each Glo shows status (Active/Completed/Pending  |
| `con_en97onit` | concept | [o_lifecycle] Properties |
| `con_esyy9xi1` | concept | Glo Credits — 1:1 USD. Wallet via Stripe customer balance. Blockchain/token angl |
| `con_eth06cc5` | context | Tom still needs to review the complete demo with all gap fixes. Designer screens |
| `con_g2ct9wzy` | context | Tom provided designer screenshots for design reference. Were in previous session |
| `con_gusamr72` | concept | The Huddle Sports Bar — Fictional bar for DOOH demo. Use case #3. |
| `con_ha105n2f` | context | 🧠 Claude Session Log — Reset #3 |
| `con_hczvz55p` | concept | Moderation System — Two-layer: AI pre-screen (risk score, flags, IAB brand safet |
| `con_hlytgmqx` | concept | Waymark — AI video gen, franchise/SMB focus. Recommended for Glo production scal |
| `con_i2zos5t8` | concept | CampaignParamsResolver |
| `con_indnhlfx` | concept | Tmemory — Persistent brain engine for Claude (v4.2) |
| `con_iwbfmq14` | concept | Vibe.co research done. Patterns: single URL→auto-gen TV ad <30s, dual source (we |
| `con_keh88ufv` | concept | [o_credits] Properties |
| `con_l4vb6dd0` | concept | Demo publishers: WDIV CTV ($22 CPM), Adweek Online ($18), Huddle DOOH ($8) |
| `con_mxqyesu8` | context | Session Log — Reset #6 |
| `con_n6b3blwr` | concept | [o_emailsys] Properties |
| `con_nk7rwlmh` | concept | [o_brightness] Properties |
| `con_o4e176b9` | context | Users database flagged as future component — user profiles, history, risk scorin |
| `con_oeevnjvn` | concept | [o_myglos] Properties |
| `con_om3aybhu` | context | Mobile capture: Tom wants to capture mobile users but worried about fake Google  |
| `con_oxj3k77x` | context | Built 3 new screens into demo: (1) My Glos Dashboard — 6 mock Glos in all states |
| `con_q1oslrkk` | concept | AI video gen build-vs-buy: Creatify best for MVP $99/mo API URL→video. Waymark f |
| `con_q6pydb49` | concept | EX.CO: full end-to-end video platform for online publishers (CMS, ad server, pla |
| `con_qapunc9e` | concept | Cloudinary — Video transcoding, AI smart cropping across aspect ratios. ~$300/mo |
| `con_rd9bmn59` | concept | Creatify — AI video gen API. $99/mo. URL→video. Recommended for Glo MVP. |
| `con_rro1j1fs` | concept | Adweek — Media industry news site. Demo use case #2 — online video. |
| `con_s37hwva1` | concept | Clerk — Auth/user management. All SSOs + Stripe integration. Recommended for Glo |
| `con_trjp0r0j` | concept | WDIV Local 4 / Graham Media — Detroit CTV news. Demo use case #1. |
| `con_u04edfxj` | concept | Competitor creative flows: 13 platforms analyzed, avg 4-5 steps. Waymark closest |
| `con_u39xwqoi` | context | Expanded Tmemory: rewrote SKILL.md with object detail files spec. Created object |
| `con_ug2u35ti` | concept | [o_moderation] Properties |
| `con_v2k6tlml` | context | Formal business case for EX.CO board — parked but needed. |
| `con_ws6v4n4y` | concept | My Glos Dashboard — User home. Cards per Glo: thumbnail, status, progress (% bud |
| `con_ye6u6ecm` | context | NEW COMPONENTS FROM TOM: (1) Email/notification system — activation progress, ex |
| `con_z480h5v7` | context | EX.CO can enable sample page link on media site so user can see their Glo in con |
| `d1618661e7914934992d0b894ac300b5` | rule | Report only what you deem important; ask if unsure; abstract rules over metrics |
| `d414167317c9438c92962cbc96be1fae` | decision | Video variants: generate 15s and 30s in parallel, show both previews |
| `d5cf8de9c7cb4af1b91ad2e5f688a314` | decision | Time-dilation decay: decay_active_rate and decay_idle_rate independent |
| `d6605022c69d4c009caac74fe0352c8f` | rule | Tom: Known UX patterns over novel invention |
| `d87c8aaff8ca433b88261e56bb16fd01` | decision | Principles topic made configurable — brain_meta instead of hardcoded 'tom-princi |
| `da02c1085ac34aa68cc7310ffae45e0d` | decision | Tmemory boot: Use setsid node index.js for independent process |
| `da3aa832d97c4b009f682518eb2c52db` | intuition | Dream: Credits balance shown wherever makes sen ↔ [o_brightness] Dynamic pricing |
| `db246bfdf69b4d978afbb696df1c45a2` | decision | Creatify migration: Product-to-Video API (no avatars required) |
| `db44b50faad04617843b3a66b4388f76` | decision | Tmemory v1.1.0: 8 typed edge types with selective decay, structural edges never  |
| `dd809c2959274d37a0e67b133db9a8aa` | decision | Correction: Query brain for existing concepts before proposing solutions |
| `debb61de627248f0ac2a5632fe8a94b6` | decision | Social onboarding modal: 'What did you have in mind?' |
| `dec_00i02v1y` | decision | Moderation initially by GLO/EX.CO ops team, publishers get access later. |
| `dec_0a4u7zrv` | decision | Glo is closed-loop demand layer on EX.CO — not standalone DSP. Monetizes unfille |
| `dec_0d22p0e0` | decision | Main flow: Draft→Pending Review→Active→Completed. Linear with moderation gate. |
| `dec_0sv57szx` | decision | [o_glo] 3 creative paths: upload, AI from URL/Google Maps, social import (coming |
| `dec_10c3j7ft` | decision | tmemory: separate user brain from plugin (fresh vs personal) |
| `dec_22kg9tdt` | decision | Budget timeline: added Now/Tomorrow/7 Days/Custom date-time selector to one-time |
| `dec_2hc13gm3` | decision | Moderator sees: creative all formats, biz name/URL, publisher+media, AI flags+sc |
| `dec_3nu4rgj4` | decision | Glo owns publisher profiles |
| `dec_3u13dlf7` | decision | Actions from numbers screen: re-light, duplicate, pause, top-up CTA. |
| `dec_40gyczcq` | decision | SSO→payment linking: Google→GPay, Apple→ApplePay, Amazon→AmazonPay, Facebook→Met |
| `dec_512b82lk` | decision | [o_brightness] Tier names: Well/Bright/Shine at $30/$50/$100. Supernova removed. |
| `dec_546e3kou` | decision | tmemory v4: self-improvement via instrumented recall + evaluation |
| `dec_59aiq5ue` | decision | Naming: no ad jargon. Not campaign — it's a Glo. Active Glo, Paused Glo, Past Gl |
| `dec_5acw6hwv` | decision | Target user: SMB to micro-biz to normal individuals. Anyone with a phone. Never  |
| `dec_5ng2jk38` | decision | tmemory v4.2: recall scoring overhaul — uncapped spread activation + hub dampeni |
| `dec_5u8un5gd` | decision | Sample page link for online publishers — EX.CO can enable. Useful for email syst |
| `dec_6bo4ejah` | decision | [stm:s58] LOCKED: Glo owns publisher profiles (rich metadata: branding, type, au |
| `dec_6dryw8t6` | decision | Correction: tmemory plugin was GLO-specific, must be project-agnostic |
| `dec_6f55wzbe` | decision | AI moderation signals: business legitimacy, site reputation, brand safety (IAB s |
| `dec_6l77wi5d` | decision | Budget screen: glow icon (SVG radial gradient) grows in size+intensity per tier. |
| `dec_6rruh9bf` | decision | Moderation: AI pre-screen + human final action. GLO/EX.CO ops initially, publish |
| `dec_6z006j7r` | decision | Contextual intent: infer creative direction from WHO (media context) and WHAT (u |
| `dec_7ld3306a` | decision | [o_lifecycle] Re-light: easy path to spend again — same creative, new budget at  |
| `dec_7sn5h7l1` | decision | Pause refunds credits to wallet. Dynamic pricing means can't hold credits at old |
| `dec_7zjb9wgv` | decision | Separate API + Web architecture |
| `dec_88dw1kxl` | decision | [o_myglos] Multiple simultaneous Glos per user — yes. |
| `dec_89qq5rfh` | decision | [o_glo] Lifecycle: Draft→Pending Review→Active→Completed. Rejected=refund+duplic |
| `dec_8ccn911r` | decision | tmemory scoring: 35% relevance + 30% recency + 25% emotion + 10% frequency |
| `dec_8vvy8az3` | decision | Auth: Clerk recommended. All SSOs + Stripe. |
| `dec_93uti0o7` | decision | [o_antifraud] Payment gate over phone verification. Less friction, stronger sign |
| `dec_94kstp68` | decision | [o_glo] Flywheel: unfilled inventory→house ads→recruit advertisers→fill inventor |
| `dec_a811djg2` | decision | [o_glo] Web app (PWA), not native iOS — avoids Apple's 30% in-app purchase cut. |
| `dec_a8rtl75a` | decision | Paused: credits return to wallet. Can't hold at old rate due to dynamic pricing  |
| `dec_ahefv17q` | decision | Moderation UI: scale-friendly from day 1 — filtering, mass actions, keyboard sho |
| `dec_aky2oorz` | decision | Moderation: AI pre-screen (risk score, flags, IAB brand safety, biz legitimacy,  |
| `dec_bdhgxfx7` | decision | Glo/EX.CO boundary |
| `dec_bhcyrn3o` | decision | [o_brightness] Dynamic pricing: media prices change daily (e.g. big MLB event).  |
| `dec_bmzmb6mx` | decision | Flywheel: unfilled inventory→house ads recruit advertisers→new Glos fill invento |
| `dec_bv3bmcjs` | decision | [stm:s59] CampaignParamsResolver — isolated component that takes a Glo and retur |
| `dec_bzdb0leg` | decision | Budget screen: How to Spend first |
| `dec_c0h38kix` | decision | Tier pricing: 1.4x markup multiplier on publisher CPM |
| `dec_c4y0lvmg` | decision | [o_myglos] Demo features: pulsing live dot for active Glos, mini sparkline chart |
| `dec_cet8and2` | decision | My Glos thumbnails: video first-frame style with play button overlay + duration  |
| `dec_cscd2kgj` | decision | Glo Numbers screen: views/day graph, total views/clicks/QR scans, budget progres |
| `dec_cww0oo7d` | decision | AI video gen is NOT the moat. Buy/integrate: Creatify MVP $99/mo, Waymark produc |
| `dec_dg3e14yv` | decision | Two rule layers: GLO general rules + publisher-specific configurable rules (e.g. |
| `dec_dp93qu36` | decision | Supply Adapter pattern |
| `dec_dp9gw843` | decision | [o_glo] Anti-fraud: payment gate over phone verification. Less friction, stronge |
| `dec_dtbiv3aq` | decision | Confirmation screen: status is always 'Pending Review' (orange), not 'Active'. G |
| `dec_du3vwge2` | decision | Moderation model: AI moderates first, adds comments+auto-status to moderation bo |
| `dec_dumnu9n7` | decision | API for agents at scale |
| `dec_ea73z2xh` | decision | Goal only in AI generate path |
| `dec_epoh25zf` | decision | Multiple simultaneous Glos per user: yes. |
| `dec_esl0q0wq` | decision | NanoBanana API: single image only despite docs showing array support |
| `dec_ey1ebmvv` | decision | Credits balance shown wherever it makes sense: My Glos dashboard, user settings  |
| `dec_f29ykczb` | decision | [stm:s48] LOCKED: Creative screen defaults to Upload tab (not AI Generate). Uplo |
| `dec_fd1qni86` | decision | Anti-fraud: payment gate confirmed over phone number. Less friction, stronger si |
| `dec_g9kj5da2` | decision | Glo lifecycle states: Draft→Pending Review→Active→Completed. Branches: Rejected  |
| `dec_g9us24ox` | decision | Anti-fraud concern: fake Google logins and bots on mobile. Payment as strongest  |
| `dec_gne7699k` | decision | tmemory v4.2: dream-time keyword enrichment + extractKeywords generates variants |
| `dec_gu4zc2yo` | decision | Creative AI section now URL/Google Maps autocomplete with search icon, dropdown  |
| `dec_h0hd8603` | decision | Creatify API integrated for real AI video generation in Glo beta |
| `dec_h5wmmnyv` | decision | Budget screen order: (1) How to Spend toggle (One-time vs Daily Recurrin |
| `dec_hrh59gko` | decision | [o_myglos] Actions per Glo: pause, re-light, duplicate, view numbers. |
| `dec_hv1fvbxk` | decision | Moderator sees: creative all formats (16:9, 9:16, 1:1), biz name/URL, publisher+ |
| `dec_i5izh6vy` | decision | GlowIcon enhanced: brightness now dramatically affects glow spread (4+10*brightn |
| `dec_iweemnjn` | decision | GAM as campaign controller |
| `dec_j3vp1rdc` | decision | Payment: Glo Credits 1:1 USD. Wallet via Stripe customer balance (avoids money t |
| `dec_j7kxuz22` | decision | Spend model: one-time tiers + $X/day recurring cancel-anytime. Min spend low eno |
| `dec_jfuegdiy` | decision | Contextual intent engine: infers creative direction from WHO (what media they ca |
| `dec_kuwfrwcb` | decision | My Glos dashboard: cards with thumbnail, status badge, publisher, progress bar ( |
| `dec_l3785iyj` | decision | [stm:s55] LOCKED: System architecture decisions — Glo owns users/billing/creativ |
| `dec_l5r9runt` | decision | Web app (PWA) not native iOS — avoids Apple's 30% in-app purchase cut. Normal St |
| `dec_m9fq8akb` | decision | Creative on Glo CDN |
| `dec_ngj10rmw` | decision | Media types: Online (video on publisher sites), CTV (broadcast), DOOH (venue scr |
| `dec_nklsc3el` | decision | Reject flow: predefined categories + optional moderator note. User gets refund + |
| `dec_nljktp0j` | decision | Spend model: one-time tiers (Well/Bright/Shine) + $X/day recurring cancel-anytim |
| `dec_nnyhd99u` | decision | Progress shown as % budget spent + vanity metrics (views, clicks, QR scans). |
| `dec_o8fqg5v9` | decision | [o_glo] Closed-loop demand layer on EX.CO — not standalone DSP. Monetizes 20-40% |
| `dec_o8mmacjt` | decision | Rejection emails include reason + CTA to duplicate and try again. |
| `dec_pyaoa97l` | decision | [o_myglos] Drafts saved to dashboard, accessible anytime. |
| `dec_q97wa3iy` | decision | AI signals: business legitimacy, site reputation, brand safety (IAB standards),  |
| `dec_qff52nm6` | decision | Drafts saved to My Glos dashboard, accessible anytime. |
| `dec_qgpataos` | decision | Glo conceptually supported by EX.CO leadership, needs formal business case for b |
| `dec_qz2r5qnm` | decision | Payment: Glo Credits 1:1 USD, Stripe customer balance. SSO→payment linking. |
| `dec_qznlarcg` | decision | LOCKED DECISION (Tom gave 3x): Onboarding = Business name OR Google Maps autocom |
| `dec_rds1u570` | decision | Upload tab: social + Shopify coming soon |
| `dec_rgfy6myo` | decision | v4.3: Session notes moved from memory-cue to code-enforced behavior |
| `dec_rqjns07z` | decision | Lesson: UI regression from Claude suggesting layout changes without checking exi |
| `dec_s4xx6nkp` | decision | Pricing: 40% Glo margin default (adjustable per publisher/media). Publisher sets |
| `dec_sb2bvhd5` | decision | Moderation: AI-first, two layers — platform safety + publisher-specific configur |
| `dec_ske5osc2` | decision | Glo lifecycle: Draft→Pending Review→Active→Completed. Rejected=refund+duplicate. |
| `dec_sn3nayjg` | decision | Glo component map: 13 components defined with boundaries and dependencies |
| `dec_sqaka0ks` | decision | [stm:s45] CRITICAL TOM FEEDBACK (given 3x — DO NOT REVERT): (1) Onboarding scree |
| `dec_sx4ubqa4` | decision | Email system: activation progress, performance updates with real screenshots fro |
| `dec_t1zl6v9h` | decision | Onboarding: two fields, no goal |
| `dec_t2in469y` | decision | [o_brightness] Users see branded tiers, not CPM/impression math. Simplicity over |
| `dec_tb95qqbb` | decision | Auth: Clerk recommended. Supports all needed SSOs + Stripe integration. |
| `dec_tsezf8dq` | decision | Glo is closed-loop demand layer on EX.CO ad server. Not standalone DSP. Monetize |
| `dec_u1291joh` | decision | tmemory v7: automatic hooks — PreToolUse, PreCompact, improved SessionStart |
| `dec_u182xyaq` | decision | Credits balance shown wherever makes sense: My Glos dashboard, user settings scr |
| `dec_uf5f07sj` | decision | Track your Glos (plural) |
| `dec_ule3fp2s` | decision | Auth: email signup removed. SSO-only (Google/Apple/Facebook/Amazon/Shopify). Kee |
| `dec_us9h7gcz` | decision | [stm:s47] LOCKED: Onboarding field 1 = 'Your Name/Business' (plain text, no auto |
| `dec_usr21at9` | decision | [stm:s57] LOCKED: Separate API + Web. Glo API (REST) is single source of truth.  |
| `dec_vvehyy1a` | decision | Emails include real screenshots from actual publisher site — not mockups. |
| `dec_w07txi6t` | decision | Veo 3.1 via Gemini API supports multi-image video generation (up to 3) |
| `dec_wc5py1ss` | decision | Upload is default creative tab |
| `dec_wpphbr40` | decision | UI: scale-friendly from day 1 — filtering, mass approve safe, keyboard shortcuts |
| `dec_x4b6pvhm` | decision | Aspect ratio removed from UI |
| `dec_xdcei6i9` | decision | Daily recurring uses slider not tiers |
| `dec_xymi39b7` | decision | Glo Creative Intelligence: LLM as Creative Director, not fixed archetype mapping |
| `dec_y78ln1w0` | decision | Creative strategy: AI video gen is NOT the moat. Buy/integrate. Creatify API $99 |
| `dec_y876ke8j` | decision | AI moderates first — adds comments, auto-status, risk score to moderation board. |
| `dec_y9v42sjb` | decision | [o_antifraud] Must balance anti-fraud friction vs impulse UX — Glo's target user |
| `dec_yvj3fuvx` | decision | Reject flow: predefined categories + optional moderator note. User gets refund + |
| `dec_yyz40ywt` | decision | [stm:s56] LOCKED: Supply Adapter pattern — clean abstraction layer so GAM can be |
| `dec_z0mbdh0n` | decision | Video prompt rewritten: Hook→Showcase→CTA structure, industry-specific, 500 char |
| `dec_zcpkv3sw` | decision | Sample page link from EX.CO can be included in emails — real link, not just scre |
| `dec_zn7co4dn` | decision | Rejected: full refund + duplicate option to start new Glo from same creative. |
| `dfe1098a08a84efca57a60f73d4a81c3` | rule | Creatify: Brand safety via override_script field, not narrator text |
| `dfe2c44e4a8b4f17b76a8c76cda50698` | decision | Thought nodes: 2-3h half-life (fast decay for noise control) |
| `e1a23d8954b447acbef768e7e4c06c44` | rule | Feedback: bge-base-en-v1.5 performance — 80ms single, 30-40ms batched, 768-dim,  |
| `e276314f9c7f404bac1483cf2bc34b51` | decision | tmemory context files: slow memory with topic-based discovery |
| `e30f0568cec44e6bbf07a2ca9da1a03c` | decision | Encoding mode: Editor-to-Learner shift |
| `e36cd6ce95a243739738b8006bc41d7f` | decision | Audit SKILL.md for limiting instructions |
| `e3c462a152e84ebf88142f75dc8c8b93` | rule | Split responses: internal (Claude→brain) separate from external (Claude→user) |
| `e52e01e22f36442080df30a7c4b40741` | rule | Thoughts vs decisions/rules differ fundamentally in decay philosophy |
| `e6045baffe244b46a078f978226b3b85` | rule | NanoBanana: image-to-video API, 3-12s duration, 9 credits @ 1080p/7s |
| `e8414123731c4e9086deb9a6b7c2e021` | decision | Self-instrumentation: brain monitoring suggest() performance |
| `ea27026dbb8c4e83933aa87250984c26` | decision | Auth: SSO → Payment (two separate flows, not combined) |
| `ea9109d75dab4665a238a5430429c83c` | intuition | Dream: EX.CO — Video platform and ad server ↔ Cluster forming: "Glo Lifecycle —  |
| `eb7255ed2bb34b939a33ca3b04bdc7b6` | rule | Hook latency: 170ms startup (fixed) + 520ms connections (variable, batched away) |
| `ec48ddea7bbe4ddbaf619a51ca7183f0` | decision | Debug mode: Server-state based via /debug/* endpoints, not env vars |
| `ed919bcaadfb47699ab4361e1034c2f3` | rule | Bridge weight system: Initial 0.15, bidirectional pairs, decay without reinforce |
| `eda858e4e3de4b4f8779e234afac14fb` | decision | Offline model setup — download once at install, cache locally, no runtime networ |
| `ee9109c3fa1148e8a486c1f19edf1bc1` | decision | Demo Onboarding: Vibe.co pattern — ask about business first, not intent |
| `eea4d95bde284ff596bd98ab7ab500d5` | rule | Session-activity progressive warnings: 0 remembers→ALERT, 8+ edits→nudge, 15+→st |
| `eed7a0d8a8b64ec483de8c55294763cf` | decision | Pre-edit-suggest hook now triggers procedures |
| `ef4cc1fee4014c90b5f0cf8ab96c1ed5` | intuition | Dream: Dream: [o_credits] Pause refunds credits ↔ [stm:s35] Created GLO logo con |
| `ef7a3678ee75448a85b80ff1541714dc` | decision | 7-second ad duration target for NanoBanana |
| `f27b17f96ded48e79b60b00d3db6d0aa` | decision | Creative archetypes: infinite use cases (marriage proposals, team celebrations,  |
| `f2eecda0a4e249ac9318129dfa60a4e8` | decision | Remove thought FIFO cap—decay is the sole filter |
| `f54375a235c54da3a8b97d04b698dd9a` | rule | Video generation API: Generate → Thumbnail → Crop per aspect ratio |
| `f60d5eb40c9944e4a2ffa2895ac03813` | decision | Auto-generate ad from onboarding URL (Vibe.co pattern) |
| `f6f1bb36f0f146c4aff604b00db8b33b` | intuition | Dream: Time-dilation decay: decay_active_rate a ↔ Tom: Plan-first approach for b |
| `f738144a6fee43f3b1a1e0a7a6e789cc` | rule | Correction: Use current SKILL.md encoding rules as ground truth, not historical  |
| `f7dc7c3dd04f48e59d5f91965bb65d4a` | rule | Unconfirmed info stays contextual until earned through repetition |
| `f7e68cc794db4cf1b778cb0b8561632e` | thought | Cluster forming: "Glo Lifecycle — State machine: Draft→Pending Review→Active→Com |
| `f9fcac43222b4f0bb70bbc24fec8ca01` | rule | Bridge candidates: Require 2+ shared neighbors minimum |
| `fa4973bf5caf42debe25ac0296bd8790` | decision | Glo AI ad variations: Pick 3 with light editing, category-researched styles |
| `fa9794fdd6ef4055a5380b9b5f4ff4d2` | rule | User feedback: Avoid unnecessary logins (Tom prefers not to log in if not needed |
| `fc0ec79f714141dea076aec0d6fcddfd` | decision | Correction: tmemory recap encoding must happen automatically post-compaction |
| `fd1b812c97fa4366a6c2f354af27c28c` | decision | UX patterns applied: Pulsing, narrative data, keyboard queue, streaks |
| `fdf960183d734a5ebda4ee9bc789a19a` | vocabulary | [vocab] Glo (auto-detected) |
| `fe6ccfa8949e471496be9baa17f72e2b` | decision | Video duration: 7 seconds (NanoBanana MVP, replaces 15s/30s variants) |
| `ffbcead277c94b218ee7d6e9a80f84c6` | rule | Rule: Semantic richness beats headlines |
| `fil_1ovm90pa` | file | glo-demo-v2.jsx |
| `fil_486oy57c` | file | [ctx:glo-platform] Glo.io — Self-Serve Advertising Platform |
| `fil_gzfhl54j` | file | Glo P&L Model — 5yr financial model. Assumptions+P&L sheets. Needs EX.CO-specifi |
| `fil_mi74f6rq` | file | Glo Beta Prototype — Vite+React app at glo/beta/ |
| `fil_qmicbhzx` | file | glo-spec-v1.md |
| `int_2kdyprto` | intuition | Dream: [o_glo] Flywheel: unfilled inventory→hou ↔ SSO→payment linking: Google→GP |
| `int_3lzr9ahz` | intuition | Dream: Component: My Glos Dashboard ↔ SSO→payment linking: Google→GPay, Apple→ |
| `int_3uvfsyln` | intuition | Dream: [ltm:l2] Flywheel: unfilled inventory→ho ↔ [stm:s18] Mobile capture: Tom  |
| `int_65c3ioal` | intuition | Dream: Moderation: AI pre-screen (risk score, f ↔ [o_myglos] Multiple simultaneo |
| `int_9355ob5n` | intuition | Dream: Sample page link from EX.CO can be inclu ↔ Post-compaction session contin |
| `int_94yf7qj0` | intuition | Dream: [o_myglos] Drafts saved to dashboard, ac ↔ AI moderates first — adds comm |
| `int_bfzyw7nj` | intuition | Dream: Dream: [ltm:l2] Flywheel: unfilled inven ↔ AI video gen is NOT the moat.  |
| `int_ea7et2lo` | intuition | Dream: [o_myglos] Multiple simultaneous Glos pe ↔ Mobile Capture & Anti-Fraud —  |
| `int_epophnk2` | intuition | Dream: Graph bridging > embeddings for emergent ↔ Glo/EX.CO boundary |
| `int_jn22iqh3` | intuition | Dream: Web app (PWA) not native iOS — avoids Ap ↔ [o_glo] 3 creative paths: uplo |
| `int_oy2sww3w` | intuition | Dream: [o_credits] Pause refunds credits to wal ↔ [stm:s7] Formal business case  |
| `int_qagfib4l` | intuition | Dream: Creative strategy: AI video gen is NOT t ↔ Reject flow: predefined catego |
| `int_smy5acx6` | intuition | Dream: [stm:s49] LOCKED: Aspect ratio selection ↔ Anti-fraud concern: fake Googl |
| `int_wh9xau4r` | intuition | Dream: Web app (PWA) not native iOS — avoids Ap ↔ Reject flow: predefined catego |
| `int_xf26hbu8` | intuition | Dream: WDIV Local 4 / Graham Media — Detroit CT ↔ Glo Numbers — Analytics detail |
| `int_z4u8x2ya` | intuition | Dream: Glo Credits — 1:1 USD. Wallet via Stripe ↔ Moderation: AI-first, two laye |
| `int_z4usuzzd` | intuition | Dream: [todo:t4] Build formal business case for ↔ [period:2026-03:p2] Biggest gr |
| `per_nzqi8kyf` | person | Tom — CEO of EX.CO |
| `per_o9s9fdzq` | person | Tom — CEO of EX.CO |
| `pro_0gaibwq7` | project | Glo.io — Self-serve advertising platform |
| `pro_dj5pg78h` | project | Fox Corp — Media conglomerate. EX.CO sales target — entry via Tubi, Fox Weather, |
| `pro_doqpydrv` | project | Vibe.co — Streaming/CTV ad platform. UX reference for Glo: URL→auto-gen, unified |
| `pro_gnil3rl1` | project | Celtra — Creative automation platform. Researched early session. |
| `pro_i1pc8fs7` | project | Glo.io — Self-serve ad platform on EX.CO ad server. Anyone buys media on EX.CO p |
| `pro_k385dx4a` | project | EX.CO — Video platform and ad server |
| `pro_mb9mas11` | project | EX.CO — End-to-end video platform for publishers: CMS, ad server, player. Smart  |
| `pro_v93b9hus` | project | Magnite — US adtech. Closest comparable to Geniee (JP). Early research. |
| `rul_0ex588kq` | rule | Communication style with Tom |
| `rul_17qtllqm` | rule | Rule: media mockup must show creative ON publisher media with logo |
| `rul_1dp9w358` | rule | Added to Tom.md: always plan before executing. Tom discusses many topics, adds s |
| `rul_5ixyibjl` | rule | When Tom says 'not now' or 'don't want to go into it' — park it, separate as com |
| `rul_84u6115r` | rule | Tom wants working demos over mockups. 'A working basic product, not a figma styl |
| `rul_8xgct2hg` | rule | [o_glo] Rule: budget_order |
| `rul_9oosw82x` | rule | Rule: every screen component must have a UI CONTRACT comment at top |
| `rul_b8fjg631` | rule | [o_glo] Rule: onboarding_fields |
| `rul_bdcn9o3m` | rule | React hooks rule |
| `rul_bzts7l9j` | rule | Tom values component separation. When scope grows, break into independent pieces |
| `rul_ds95mnx3` | rule | [o_glo] Rule: react_hooks |
| `rul_eo65j1tn` | rule | [o_glo] Rule: goal_placement |
| `rul_fkbh82ps` | rule | [o_glo] Rule: aspect_ratio |
| `rul_ifx02y7i` | rule | Tom prefers: discuss and define before building. Sequence: frame→research→design |
| `rul_jcsprj1r` | rule | Tom references competitor UX frequently. When he names a product, research it —  |
| `rul_jd8ib8c2` | rule | Rule: Glo disruptive concept — anyone can upload to any media, UI must reinforce |
| `rul_yi4wtwu0` | rule | [o_glo] Rule: creative_default |
| `tas_5qvp5i9j` | task | [todo:t6] Design Shopify app integration (product import→auto-create Glos) |
| `tas_8nq58mqs` | task | [todo:t2] Re-upload designer screenshots for future design reference |
| `tas_97r7kktl` | task | [todo:t5] Explore blockchain/token angle for Glo Credits |
| `tas_bvguw1tj` | task | [todo:t7] Campaign spend optimization logic (pacing, reallocation across media t |
| `tas_lzwjx31w` | task | Upgrade: Move pre-compaction session note from memory rule to code-level behavio |
| `tas_ns5foexe` | task | [todo:t1] Review revised Glo demo (glo-demo.jsx) — new onboarding, unified creat |
| `tas_qczclad1` | task | [todo:t3] Update P&L with EX.CO-specific assumptions (unfilled inventory economi |
| `tas_u6fa4c0r` | task | [todo:t4] Build formal business case for EX.CO board |
| `tho_qhldmhpr` | thought | Cluster forming: "[period:2026-03:p12] 13 competitor flows + AI video gen build- |
