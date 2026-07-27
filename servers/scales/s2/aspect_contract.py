"""S2 Aspect Integration — Contract and Configuration.

Classifies distinct node types and edge relations into the 15 required
aspects defined in `aspects_v1.json`. Closed-list classification — encoder
can only route to existing aspects (or `noise`/`generic_relation` as
catch-alls), never proposes new ones. New aspects are added by humans
editing `aspects_v1.json`.

The unit is self-contained: reads + writes JSON files only, never mutates
brain state. No suppression machinery — closed list means every string
gets a home, so SKIP isn't a valid encoder action.
"""

import os


ASPECT = {
    # LLM config
    'model': 'claude-sonnet-4-6',  # Sonnet for classification quality
    'max_tokens': 8192,

    # Batch sizing — chosen for clean per-item attention without losing
    # the cross-string visibility that helps consistency (e.g., `corrects`
    # and `corrected_by` classified together).
    'max_candidates_per_call': 30,

    # Decoder filters
    # 1 = classify EVERY string, singletons included. An unaspected string is
    # invisible to every aspect-driven consumer (correction_enrich, the wisdom
    # Frame pull, noise filtering); a one-off typo filed under `noise` is the
    # cheaper failure. Most strings arrive exactly once, so this is the floor,
    # not a tuning knob.
    'min_count_threshold': 1,
    'examples_per_candidate': 3,     # nodes/edges shown per candidate string
}


# Runtime aspect state lives next to brain.db (per-operator, not in repo).
# Repo-bundled seed (SEED_ASPECTS_JSON_PATH) is the first-boot baseline;
# AspectRegistry copies seed → user dir on first load when missing. After
# that, all encoder writes stay in the user dir — the repo seed is read-
# only and never touched by runtime.
#
# Paths are resolved at CALL time, not import time. BRAIN_DB_DIR (and the
# explicit overrides) are read on every call so a later `os.environ` change
# takes effect — IsolatedBrain sets BRAIN_DB_DIR in __enter__, AFTER this
# module is imported, and a module-level constant would freeze the live
# user-dir path and leak heals into it (observed 2026-06-16).
_DEFAULT_DB_DIR = os.path.join(os.path.expanduser('~'), 'AgentsContext', 'brain')


def aspects_json_path() -> str:
    """Path to the per-operator working aspects file (call-time resolved)."""
    return os.environ.get(
        'ASPECTS_JSON_PATH',
        os.path.join(os.environ.get('BRAIN_DB_DIR', _DEFAULT_DB_DIR),
                     'aspects_v1.json'))


def aspects_proposed_path() -> str:
    """Path to the per-cycle audit artifact (call-time resolved)."""
    return os.environ.get(
        'ASPECTS_PROPOSED_PATH',
        os.path.join(os.environ.get('BRAIN_DB_DIR', _DEFAULT_DB_DIR),
                     'aspects_proposed.json'))


# Repo seed — frozen baseline shipped with the plugin. Never written.
SEED_ASPECTS_JSON_PATH = os.path.join(
    os.path.dirname(__file__), 'aspects_v1.json')


def ensure_aspects_user_copy(log_fn=None) -> bool:
    """Seed the user-dir aspects file from the repo seed, and SELF-HEAL.

    Three jobs, all idempotent and safe to call on every boot:
      1. First boot — working copy missing → copy the whole seed.
      2. Missing aspect — the seed has a REQUIRED aspect the working copy
         lacks → add the whole aspect from the seed.
      3. Missing member — a REQUIRED aspect's seed `node_types` /
         `edge_relations` list names a string the working copy's list lacks
         → APPEND it. This is how a curated membership fix (e.g. multi-homing
         a replacement verb into correction_improvement, which recall walks)
         reaches an existing brain. Without it a seed member edit propagates
         to fresh installs only, and every existing brain keeps the defect
         while the seed-based contract tests pass.

    ADDITIVE ONLY, in both directions of scope:
      · Members are appended, never reordered or removed — operator- and
        AspectIntegration-grown lists survive, and append-at-end cannot
        evict anything from the first-8 window `render_edge_aspects_block`
        shows the encoders.
      · A seed REMOVAL does not propagate. Retiring a member (or moving one
        between aspects) still needs a supervised migration; this heals
        omissions, not disagreements.
      · Only REQUIRED aspects are touched. Emergent/unlocked aspects are the
        classifier's to own.

    Never writes the seed itself (tests may point the working path at it).
    Returns True if the file was created or modified. `log_fn(message)` — when
    given — is called with a one-line summary of what was healed: a member heal
    silently changes which edges `correction_enrich` walks, so it announces
    itself rather than being inferred from behaviour later.
    """
    import shutil
    json_path = aspects_json_path()
    if not os.path.exists(SEED_ASPECTS_JSON_PATH):
        return False
    # Never heal the seed into itself.
    if os.path.abspath(json_path) == os.path.abspath(SEED_ASPECTS_JSON_PATH):
        return False
    if not os.path.exists(json_path):
        os.makedirs(os.path.dirname(json_path), exist_ok=True)
        shutil.copy2(SEED_ASPECTS_JSON_PATH, json_path)
        return True

    # Working copy exists — self-heal any missing REQUIRED aspect from the seed.
    import json
    from servers.aspects import REQUIRED_ASPECTS  # local: avoid import cycle
    try:
        with open(SEED_ASPECTS_JSON_PATH) as f:
            seed = json.load(f)
        with open(json_path) as f:
            cur = json.load(f)
    except (OSError, json.JSONDecodeError):
        return False
    missing = [n for n in REQUIRED_ASPECTS if n in seed and n not in cur]
    # Missing MEMBERS of required aspects that both files carry (job 3).
    # Shape-guarded the same way AspectRegistry._load guards: a working copy
    # can carry a malformed entry (hand-edit, partial write), and an unguarded
    # deref here would abort the WHOLE heal — including job 2 above, whose
    # failure leaves survivor_lineage empty and silently disables the
    # absorbed_into archive exemption. A malformed entry is skipped, not fatal.
    member_heals = []          # (aspect, category, [strings]) — for the caller's log
    skipped_malformed = []
    for n in REQUIRED_ASPECTS:
        if n not in seed or n not in cur:
            continue           # job 1/2 territory, or not seeded at all
        if not isinstance(cur[n], dict) or not isinstance(seed[n], dict):
            skipped_malformed.append(n)
            continue
        for category in ('node_types', 'edge_relations'):
            have = cur[n].get(category)
            want = seed[n].get(category)
            if have is None:
                have = []      # absent key is a legitimate empty list
            if not isinstance(have, list) or not isinstance(want or [], list):
                skipped_malformed.append('%s.%s' % (n, category))
                continue
            gap = [s for s in (want or []) if s not in have]
            if gap:
                member_heals.append((n, category, gap))
    if not missing and not member_heals:
        if skipped_malformed and log_fn:
            try:
                log_fn('working copy has malformed entries, skipped: %s'
                       % ', '.join(skipped_malformed))
            except Exception:
                pass
        return False
    # NOTE (accepted, narrow): this read-modify-write can lose to a concurrent
    # AspectIntegration os.replace that carries a pre-heal snapshot. The window
    # is tiny — self-heal only writes when the working copy is actually behind
    # the seed, and AspectIntegration runs during S2 idle maintenance, not at
    # Brain.__init__. If it ever loses, the next boot self-heals again
    # (idempotent), so it's self-correcting, not durable loss.
    for n in missing:
        cur[n] = seed[n]
    for n, category, gap in member_heals:
        cur[n].setdefault(category, []).extend(gap)   # append — never reorder
    parts = ['+aspect %s' % n for n in missing]
    parts += ['%s.%s += %s' % (n, category, ','.join(gap))
              for n, category, gap in member_heals]
    if skipped_malformed:
        parts.append('SKIPPED malformed: %s' % ', '.join(skipped_malformed))
    summary = 'healed working copy from seed: ' + '; '.join(parts)
    # Atomic write — temp file + os.replace, so a crash mid-write can't leave a
    # truncated/0-byte aspects file. A corrupt working copy loads as an empty
    # registry → relations_in(['survivor_lineage']) returns () → the
    # absorbed_into exemption silently disables and the reaper scrubs redirect
    # edges. Mirrors AspectEncoder._write_aspects.
    import tempfile
    d = os.path.dirname(json_path)
    fd, tmp = tempfile.mkstemp(prefix='aspects_v1_', suffix='.json.tmp', dir=d)
    try:
        with os.fdopen(fd, 'w') as f:
            json.dump(cur, f, indent=2, ensure_ascii=False)
            f.write('\n')
        os.replace(tmp, json_path)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    # Announce AFTER the write lands, so a failed write can't leave a log line
    # claiming a heal that didn't happen. Two channels on purpose: stdout is the
    # only one that reliably reaches a log at Brain.__init__ time (a short-lived
    # process whose logs-db write hits `database is locked` degrades to a stderr
    # nobody reads), and log_fn gives the daemon a queryable row.
    print('[aspects] %s' % summary, flush=True)
    if log_fn:
        try:
            log_fn(summary)
        except Exception:
            pass                                      # never break boot on a log
    return True
