# Migrating from the old `brain` plugin to `entity`

**For the person:** open a Claude Code session and say
*"Read MIGRATING.md from the entity repo and walk me through it."*
Then answer its questions. It takes about ten minutes, most of it waiting for
the new runtime to build. Nothing gets deleted at any step.

**For Claude, reading this on the user's behalf — the rules:**

- Work one phase at a time. Tell the user what a phase does before running it,
  and ask before every phase that changes anything.
- Never run `claude plugin uninstall` without `--keep-data`. Never `rm`
  anything. Never move brain files by hand — one script does that, safely.
- If a check fails, stop and show the user what you saw. Do not improvise a
  fix; the old plugin and its data are still intact at every point below.

---

## Why this is a migration, not an update

The plugin was renamed. Claude Code identifies a plugin by
`<plugin>@<marketplace>`, and both halves changed (`brain@brain` →
`entity@anchor`), so `/plugin update` cannot carry anyone across. The new
plugin has to be installed alongside, and the old one removed.

The one real hazard: on older installs the brain's files live inside a folder
that a default `claude plugin uninstall` deletes. Phase 1 moves them out
before anything is uninstalled. Everything else is bookkeeping.

---

## Phase 0 — Where is the brain, and how big is it?

Run:

```bash
cat "${XDG_CONFIG_HOME:-$HOME/.config}/brain/resolved.env"
```

Note two values: `BRAIN_DB_DIR` (where the brain lives) and `PLUGIN_ROOT`
(where the old plugin is installed).

Classify `BRAIN_DB_DIR`:

| It contains | Meaning | Phase 1 |
|---|---|---|
| `/.claude/plugins/data/` | **parked** inside a plugin-owned folder — at risk on uninstall | **required** |
| anything else (`~/.local/share/brain`, a legacy pre-plugin folder, a custom path) | already outside plugin control — safe | skip to Phase 2 |

Also note the memory count from this session's boot banner — the line that
reads *"I have N memories"*. Write N down. It is the number every later
check compares against.

Confirm the numbers with the user before continuing.

---

## Phase 1 — Move a parked brain to safety (only if parked)

The relocation tool ships with plugin version 9.7.2 or later. Make sure the
old plugin is current, then restart Claude Code so the new hooks load:

```bash
claude plugin update brain
```

In the fresh session, the boot banner shows a notice titled *"your brain lives
in a folder `claude plugin uninstall` deletes"* and names the exact command.
It is this one (substitute the `PLUGIN_ROOT` from Phase 0):

```bash
bash "<PLUGIN_ROOT>/hooks/scripts/relocate-brain.sh"
```

What it does, so the user knows: stops the daemon under a maintenance lock,
copies the brain to `~/.local/share/brain/`, integrity-checks both copies,
swaps the new location in with one atomic rename, **keeps the original beside
its old path as an inert spare** (that spare is the backup — leave it), and
restarts the services pointing at the new location.

Then start a new session and check:

1. `cat "${XDG_CONFIG_HOME:-$HOME/.config}/brain/resolved.env"` — `BRAIN_DB_DIR`
   now points under `.local/share/brain`.
2. The boot banner's memory count equals N.

If the script is missing (the update did not reach 9.7.2) or either check
fails: **stop here**. The brain is untouched at its original path; ask the
maintainer before doing anything else.

---

## Phase 2 — Remove the old plugin, keeping its data folder

```bash
claude plugin uninstall brain --keep-data
```

`--keep-data` is not optional. It preserves the plugin's data folder — which
after Phase 1 holds only the inert spare, and on a never-parked install holds
nothing of value, but there is no reason to delete either.

Expected side effect: the background daemon stops, because the folder it was
launched from is gone. Memory is offline until Phase 3 finishes. That is
normal.

---

## Phase 3 — Install `entity`

```bash
claude plugin marketplace add tpac/entity
claude plugin install entity@anchor
```

Start a new Claude Code session. First boot builds an isolated runtime and
downloads the embedding model (a couple of minutes; the banner reports
progress) and re-installs the background service.

The new plugin finds the existing brain through the same `resolved.env` file
from Phase 0 — it does not create a fresh one when a brain is already
recorded there. If the boot banner nevertheless offers a choice about an
existing brain, pick the option that **connects to the existing brain** at
the `BRAIN_DB_DIR` you wrote down. Never pick "start fresh".

API key: if it was stored in the old plugin's settings, the old plugin already
mirrored it to `~/.config/brain/env`, and the new plugin reads it from there.
If Claude Code asks for a key anyway, enter it in the new plugin's settings.

---

## Phase 4 — Verify

In a fresh session, all three must hold:

1. The boot banner's memory count equals N.
2. The brain tools carry the new prefix: `mcp__plugin_entity_brain__recall`
   and friends (39 tools).
3. `/entity:brain` exists as a slash command (the old `/brain:brain` is gone).

If all three hold, the migration is complete. From here on, every release is
an ordinary `claude plugin update entity`.

---

## If something went wrong

Nothing above deletes data, so the way back is short:

- **After Phase 1:** the original brain is still at its old path, renamed as
  a spare beside it. The relocation can be re-run or the spare pointed at.
- **After Phase 2:** the old plugin can be reinstalled from its marketplace;
  `--keep-data` left its folder in place.
- **After Phase 3:** if the new plugin created a fresh brain by mistake, do
  not use it. Set `BRAIN_DB_DIR` in `~/.config/brain/env` to the path from
  Phase 0 (or the Phase 1 location) and start a new session.

Bring the failing check's output to the maintainer. The memory count N is the
fact that settles whether anything was lost.
