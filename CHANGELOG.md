# Changelog

All notable changes to **Anchor** (the `brain` plugin). The format roughly follows
[Keep a Changelog](https://keepachangelog.com); versioning is [semver](https://semver.org).

> Per-version detail for releases before 9.6.0 lives in git history and the prior
> `plugin.json` descriptions. This file is the going-forward home for release notes —
> the `description` field stays a stable one-liner.

## [9.6.0] — 2026-05-31
### Added
- **Self-channel rules of engagement** — the channel between parallel streams (concurrent
  sessions of the same identity) gains a behavioral layer plus the substrate behind it:
  - third-person **containment render** (`⚡ <who> says: "…"` + a standing attribution
    footer) so another stream's action can't bleed into this stream's self-model;
  - **`self_outbox`** — sender-side delivery visibility (per-recipient delivered/pending),
    so silence reads as "delivered vs never," not a guess;
  - **presence active / dormant / lost** classification, surfacing recently-gone streams
    instead of silently dropping them;
  - **graceful `self_send` addressing** — id-canonical, with the 8-char short resolving as
    a prefix against the live roster (no self-naming);
  - the operative rules in `skills/brain/SKILL.md`.

### Changed
- **Plugin metadata aligned to the Claude Code plugin standard**: `displayName` ("Anchor"),
  a concise stable `description` ("Persistent identity…"), `homepage` / `repository` /
  `license`, and trimmed keywords. Release notes moved out of the `description` field into
  this changelog.
- `build-plugin.sh` warns if `BRAIN_DEV_MODE` is set in the building shell (end-user
  artifacts must run without the developer safety-net opt-out).

## [9.5.x] and earlier
Highlights (full detail in git history): episodic-references write path (`source_refs`,
`co_anchored` engram edges); S2 graph-integration units (consolidation, community, healer,
aspect-integration); the unified aspect taxonomy; the agentic surface recall path; and the
Frame structured prior.
