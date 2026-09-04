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
  fix; the brain's files are intact at every point below.

---

## Why this is a migration, not an update

The plugin was renamed. Claude Code identifies a plugin by
`<plugin>@<marketplace>`, and both halves changed (`brain@brain` →
`entity@anchor`), so `/plugin update` cannot carry anyone across. The old
plugin has to be removed and the new one installed.

The one real hazard: on older installs the brain's files live inside a folder
that a default `claude plugin uninstall` deletes. `--keep-data` in Phase 1 is
what protects them; Phase 3 then moves them somewhere no plugin operation
ever touches. Everything else is bookkeeping.

---

## Phase 0 — Where is the brain, and how big is it?

Find the brain. Try these in order and stop at the first that answers:

```bash
cat "${XDG_CONFIG_HOME:-$HOME/.config}/brain/resolved.env"   # BRAIN_DB_DIR = the live brain
cat "${XDG_CONFIG_HOME:-$HOME/.config}/brain/env"            # a BRAIN_DB_DIR the user set by hand
ls ~/.claude/plugins/data/*/brain/brain.db ~/.local/share/brain/brain.db 2>/dev/null
```

Write down the folder that holds `brain.db`. Call it **the brain path**.
If the old plugin was pointed at the brain through its *settings* field
("Existing brain location"), note that too — Phase 2 needs it.

Classify the brain path:

| It contains | Meaning |
|---|---|
| `/.claude/plugins/data/` | **parked** inside a plugin-owned folder — Phase 3 will move it |
| anything else | already outside plugin control — Phase 3 is a no-op |

Now record how big the brain is, as **N**, using whichever of these the
install offers:

1. The boot banner line *"I have N memories"* (plugin 9.7.1 or later).
2. The node count on the local dashboard, `http://localhost:47303`.
3. Failing both: `ls -l <brain path>/brain.db` — record the size in bytes.

Confirm the brain path and N with the user before continuing. N is the fact
every later check compares against.

---

## Phase 1 — Remove the old plugin, keeping its data folder

```bash
claude plugin uninstall brain --keep-data
```

`--keep-data` is not optional. It preserves the plugin's data folder, which on
a parked install **is where the brain lives**.

Then remove the marketplace the old plugin came from, so no copy of the old
code remains for the background service to keep launching:

```bash
claude plugin marketplace list          # find the one that carried `brain`
claude plugin marketplace remove <that-marketplace>
```

Expected side effect: memory is offline until Phase 2 finishes. That is
normal.

---

## Phase 2 — Install `entity`

```bash
claude plugin marketplace add tpac/entity
claude plugin install entity@anchor
```

Start a new Claude Code session. First boot builds an isolated runtime and
downloads the embedding model (a couple of minutes; the banner reports
progress) and installs the background service pointing at the new plugin.

The new plugin finds the existing brain through `~/.config/brain/resolved.env`
— it does not create a fresh one when a brain is recorded there. Two cases
need a hand:

- If the old install used the plugin's **settings field** to point at the
  brain, that setting did not carry over. Enter the brain path from Phase 0
  in `entity`'s settings ("Existing brain location"), then start a new
  session.
- If the boot banner offers a choice about an existing brain, pick the option
  that **connects to the existing brain** at the brain path. Never pick
  "start fresh".

API key: if it was stored in the old plugin's settings, the old plugin already
mirrored it to `~/.config/brain/env`, and the new plugin reads it from there.
If Claude Code asks for a key anyway, enter it in the new plugin's settings.

---

## Phase 3 — Move a parked brain to safety (only if parked)

If Phase 0 classified the brain as parked, the new session's boot banner shows
a notice titled *"your brain lives in a folder `claude plugin uninstall`
deletes"* and names the exact command. It is the new plugin's own script:

```bash
cat "${XDG_CONFIG_HOME:-$HOME/.config}/brain/resolved.env"     # PLUGIN_ROOT = the entity install
bash "<PLUGIN_ROOT>/hooks/scripts/relocate-brain.sh"
```

What it does, so the user knows: stops the daemon under a maintenance lock,
copies the brain to `~/.local/share/brain/`, integrity-checks both copies,
swaps the new location in with one atomic rename, **keeps the original beside
its old path as an inert spare** (that spare is the backup — leave it), and
restarts the services pointing at the new location.

Then start a new session. `resolved.env` should now show `BRAIN_DB_DIR` under
`.local/share/brain`, and the parked notice should be gone.

If the brain was not parked, there is nothing to do here.

---

## Phase 4 — Verify

In a fresh session, all of these must hold:

1. **Size.** The boot banner's memory count is **at least N**. A few more
   than N is expected — the brain keeps encoding across the sessions this
   migration took. What would signal loss is a count near zero, or one the
   size of a fresh seed pack (a few dozen). If N was a byte size, the new
   `brain.db` is at least that large.
2. **Tools.** The brain tools carry the new prefix:
   `mcp__plugin_entity_brain__recall` and friends (39 tools).
3. **Skill.** `/entity:brain` exists as a slash command (the old
   `/brain:brain` is gone).
4. **Service** (macOS). The background service points at the new plugin:

   ```bash
   grep -A3 ProgramArguments ~/Library/LaunchAgents/com.brain.daemon.plist
   ```

   The path shown must be inside the `entity` install (the `PLUGIN_ROOT`
   from Phase 3), not the old plugin's folder.

If all hold, the migration is complete. From here on, every release is an
ordinary `claude plugin update entity`.

---

## If something went wrong

Nothing above deletes data, so the way back is short:

- **After Phase 1:** the brain is exactly where it was; `--keep-data` left its
  folder in place. The old plugin can be reinstalled from its marketplace if
  you need to pause here.
- **After Phase 2:** if the new plugin created a fresh brain by mistake, do
  not use it. Set `BRAIN_DB_DIR` in `~/.config/brain/env` to the brain path
  from Phase 0 and start a new session.
- **After Phase 3:** the original brain is still at its old path, renamed as
  a spare beside it. The relocation can be re-run or the spare pointed at.

Bring the failing check's output to the maintainer. N is the fact that settles
whether anything was lost.
