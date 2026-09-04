#!/bin/bash
# 5.7 — the release command. Builds the public tree, runs EVERY gate, and
# squash-commits it into a fresh repo with a tag. Dry-run by default: the push
# is the one-way door, and it opens only with --publish <url>, on a clean
# main, with every step run and green.
# DEV TOOL: lives in scripts/, never ships.
#
#   Usage: release.sh <version> [--publish <git-url>] [--skip <step>]...
#   Env:   RELEASE_AUTHOR_EMAIL  REQUIRED — the e-mail on the public commit.
#                                git's configured identity is never used: on
#                                the dev machine it is a work address, and the
#                                commit is the one artifact the export gates
#                                never see.
#          RELEASE_PREVIOUS      REQUIRED — the previous release's tree (a
#                                checkout of the public repo at its tag), or
#                                the word `none` for a first release. Unset
#                                refuses: forgetting the upgrade test must be
#                                impossible, skipping it must be stated.
#          RELEASE_STAGE         staging dir (default: <repo>/dist/release)
#
# Steps, in order — the first red refuses everything after it. Cheap gates
# run before the long suite so a red costs seconds, not forty minutes:
#   preflight    version shape · author identity · predecessor named · publish
#                preconditions
#   export       scripts/export-public-tree.sh → gate A (denylist), B (scrub),
#                C (version, pinned to <version> via EXPECT_VERSION),
#                D (credential shapes, forbidden files)
#   deploy-gate  tests/test_deploy_contract.py — version lockstep, shipped-file
#                reachability, the live-tree export + collection ratchet
#   smoke        scripts/install-smoke.sh — a stranger's first run, on a copy
#   upgrade      scripts/upgrade-smoke.sh — RELEASE_PREVIOUS's user updates to
#                this release and keeps their brain (SKIP on a first release)
#   suite        the full suite against a COPY of the export, never the tree
#                itself (.pytest_cache/ and __pycache__/ would ship)
#   repo         git init → one commit → tag; the commit's author and messages
#                are the release's OWN additions, outside every export gate,
#                so they are asserted and scrubbed here
# --skip <step> is for iterating on a dry run; it is printed in the report and
# --publish refuses it. Nothing is ever pushed without --publish.
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
say()  { printf '%s\n' "[release] $*"; }
fail() { printf '%s\n' "[release] REFUSED: $*" >&2; exit 1; }

STEPS="preflight export deploy-gate smoke upgrade suite repo"
VERSION="${1:-}"; shift || true
PUBLISH_URL=""; SKIPS=" "
while [ $# -gt 0 ]; do
  case "$1" in
    --publish) PUBLISH_URL="${2:?--publish needs a git url}"; shift 2 ;;
    --skip)
      case " $STEPS " in
        *" ${2:-} "*) ;;
        *) fail "--skip: unknown step '${2:-}' (steps: $STEPS)" ;;
      esac
      SKIPS="$SKIPS$2 "; shift 2 ;;
    *) fail "unknown argument: $1" ;;
  esac
done
STAGE="${RELEASE_STAGE:-$REPO/dist/release}"
EXPORT="$REPO/scripts/export-public-tree.sh"

# ── report bookkeeping (bash 3.2: no associative arrays) ───────────────────
RESULTS=""
_record() { RESULTS="$RESULTS$1 $2 ${3:--}"$'\n'; }
_report() {
  printf '\n[release] %s  version=%s  branch=%s%s\n' \
    "$([ -n "$PUBLISH_URL" ] && echo PUBLISH || echo 'DRY RUN')" "$VERSION" "$BRANCH" \
    "$([ -n "$DIRTY" ] && echo ' (dirty tree)')"
  printf '%-12s %-8s %s\n' STEP RESULT SECONDS
  printf '%s' "$RESULTS" | while read -r s r t; do
    printf '%-12s %-8s %s\n' "$s" "$r" "$t"
  done
}
_skipped() { case "$SKIPS" in *" $1 "*) return 0 ;; *) return 1 ;; esac; }
# Run one step: log to $STAGE/logs/<step>.log; a red step ends the run.
_step() {
  local name="$1"; shift
  if _skipped "$name"; then _record "$name" SKIP; say "$name: SKIPPED"; return 0; fi
  local t0=$SECONDS log="$STAGE/logs/$name.log" rc
  say "$name ..."
  # The body runs in a subshell with errexit of its own, and NOT as the
  # condition of an `if`: bash suppresses `set -e` throughout a function
  # tested by `if`/`&&`/`||`, so only its last command would decide the step.
  # Here every command in the body decides.
  set +e
  ( set -e; "$@" ) >"$log" 2>&1
  rc=$?
  set -e
  if [ "$rc" -eq 0 ]; then
    _record "$name" PASS $((SECONDS - t0)); say "$name: PASS ($((SECONDS - t0))s)"
  else
    _record "$name" FAIL $((SECONDS - t0))
    tail -40 "$log" >&2
    _report >&2
    fail "$name is red (full log: $log) — nothing after it ran, nothing was pushed"
  fi
}
# Steps that consume the export must find one — a skipped or failed export
# must not turn them into a scan of nothing that reports PASS.
_need_tree() {
  [ -f "$STAGE/tree/.claude-plugin/plugin.json" ] \
    || { echo "no export tree at $STAGE/tree (export skipped or failed)"; return 1; }
}

# ═══ preflight ═════════════════════════════════════════════════════════════
BRANCH="$(git -C "$REPO" rev-parse --abbrev-ref HEAD)"
DIRTY="$(git -C "$REPO" status --porcelain)"
printf '%s' "$VERSION" | grep -qE '^[0-9]+\.[0-9]+\.[0-9]+$' \
  || fail "version must be X.Y.Z (got '${VERSION:-}')"
[ -n "${RELEASE_AUTHOR_EMAIL:-}" ] \
  || fail "RELEASE_AUTHOR_EMAIL is not set — the public commit needs an explicit author e-mail (git's configured identity is deliberately not used)"
case "${RELEASE_PREVIOUS:-}" in
  "")   fail "RELEASE_PREVIOUS is not set — name the previous release's tree, or 'none' for a first release" ;;
  none) ;;
  *)    [ -f "$RELEASE_PREVIOUS/.claude-plugin/plugin.json" ] \
          || fail "RELEASE_PREVIOUS=$RELEASE_PREVIOUS is not a plugin tree" ;;
esac
PY="$REPO/venv/bin/python"
[ -x "$PY" ] || fail "bundled python missing at $PY — run hooks/scripts/ensure-runtime.sh"
AUTHOR_NAME="$(cd "$REPO" && "$PY" -c 'import json;print(json.load(open(".claude-plugin/plugin.json"))["author"]["name"])')"
REPO_URL="$(cd "$REPO" && "$PY" -c 'import json;print(json.load(open(".claude-plugin/plugin.json"))["repository"])')"
if [ -n "$PUBLISH_URL" ]; then
  # A mis-aimed push is the one way history could leak; the manifest names
  # the only legitimate target — and having vetted the URL by equality, the
  # scrub below need not (and could not: the org name is a scrub pattern).
  _norm() { printf '%s' "$1" | sed -E 's#^git@github\.com:#https://github.com/#; s#\.git$##; s#/$##'; }
  [ "$(_norm "$PUBLISH_URL")" = "$(_norm "$REPO_URL")" ] \
    || fail "--publish target '$PUBLISH_URL' is not plugin.json's repository '$REPO_URL'"
  [ "$SKIPS" = " " ] || fail "--publish with --skip: every step must run green before the door opens"
  [ "$BRANCH" = "main" ] || fail "--publish from '$BRANCH' — releases come off main"
  [ -z "$DIRTY" ] || fail "--publish with a dirty tree — the export copies the WORKING tree, uncommitted edits would ship"
  if [ "$RELEASE_PREVIOUS" = none ]; then
    # 'none' is truthful only against a virgin remote: any ref there means a
    # release exists whose users this one must not break. No prompt — an
    # unreachable or private remote must refuse, never hang.
    refs="$(GIT_TERMINAL_PROMPT=0 git ls-remote "$PUBLISH_URL" 2>&1)" \
      || fail "cannot reach $PUBLISH_URL to confirm a first release: $refs"
    [ -z "$refs" ] \
      || fail "RELEASE_PREVIOUS=none, but $PUBLISH_URL already has refs — this is not a first release; name the previous release's tree"
  fi
fi
# A declared first release has nothing to upgrade from: the step is skipped
# by the same bookkeeping as --skip, after the --publish check above so the
# two cannot be confused.
[ "$RELEASE_PREVIOUS" != none ] || SKIPS="${SKIPS}upgrade "
SUBJECT="Entity v$VERSION"
BODY="Release build of the Entity plugin — see README.md for what it is and how to install it.
Produced by the release command from the private development repository; history is squashed by design."
TAGMSG="Entity v$VERSION"

# The release's own additions run through gate B exactly as shipped files do
# (the --scrub-only door). Not in this file: the author NAME, asserted equal
# to plugin.json's author instead (the manifest's own allowlist entry vets
# it), and the publish URL, asserted equal to plugin.json's repository above.
if [ -e "$STAGE" ] && [ ! -e "$STAGE/.release-stage" ] && [ -n "$(ls -A "$STAGE" 2>/dev/null)" ]; then
  fail "$STAGE exists and is not a previous release stage — refusing to remove it"
fi
rm -rf "$STAGE"; mkdir -p "$STAGE/logs" "$STAGE/meta"; : >"$STAGE/.release-stage"
printf '%s\n%s\n%s\n%s\n' "$RELEASE_AUTHOR_EMAIL" "$SUBJECT" "$BODY" "$TAGMSG" \
  >"$STAGE/meta/RELEASE-METADATA"
_preflight_scrub() { bash "$EXPORT" --scrub-only "$STAGE/meta"; }
_step preflight _preflight_scrub

# ═══ export — gates A, B, C, D ════════════════════════════════════════════
_export() { EXPECT_VERSION="$VERSION" bash "$EXPORT" "$STAGE/tree"; }
_step export _export
# Surface each gate's one-liner (gate D says whether gitleaks ran) — the step
# log is otherwise shown only on FAIL.
{ [ -f "$STAGE/logs/export.log" ] && grep '^\[export-public\] gate' "$STAGE/logs/export.log" | sed 's/^\[export-public\]/[release]  /'; } || true

# ═══ deploy gate — the dev repo's own contract tests ══════════════════════
_deploy_gate() { (cd "$REPO" && ./dev pytest -q tests/test_deploy_contract.py); }
_step deploy-gate _deploy_gate

# ═══ smoke — a stranger's first run ═══════════════════════════════════════
_smoke() { _need_tree; bash "$REPO/scripts/install-smoke.sh" "$STAGE/tree"; }
_step smoke _smoke

# ═══ upgrade — the previous release's user keeps their brain ══════════════
_upgrade() { _need_tree; bash "$REPO/scripts/upgrade-smoke.sh" "$RELEASE_PREVIOUS" "$STAGE/tree"; }
_step upgrade _upgrade

# ═══ suite — against a COPY of the export ═════════════════════════════════
# The export carries no venv and no pytest.ini: the dev venv runs it (conftest
# warns and proceeds) and the per-test timeout is passed explicitly.
_suite() {
  _need_tree
  rm -rf "$STAGE/suite"; cp -R "$STAGE/tree" "$STAGE/suite"
  (cd "$STAGE/suite" && "$REPO/venv/bin/python" -m pytest -q -p no:cacheprovider -o timeout=120 tests)
}
_step suite _suite

# ═══ repo — one commit, one tag, identity asserted on the objects ═════════
_repo() {
  _need_tree
  local repo="$STAGE/repo"
  rm -rf "$repo"; cp -R "$STAGE/tree" "$repo"
  git -C "$repo" init -q
  git -C "$repo" symbolic-ref HEAD refs/heads/main
  git -C "$repo" config user.name "$AUTHOR_NAME"
  git -C "$repo" config user.email "$RELEASE_AUTHOR_EMAIL"
  git -C "$repo" config commit.gpgsign false
  # The export ships no .gitignore, and git would still honor the operator's
  # GLOBAL excludes — a personal ignore pattern must not decide what the
  # public gets. Everything in the tree is the commit, by construction.
  git -C "$repo" -c core.excludesFile=/dev/null add -A -f
  git -C "$repo" commit -q -m "$SUBJECT" -m "$BODY"
  git -C "$repo" tag -a "v$VERSION" -m "$TAGMSG"
  # what was committed is exactly the export, and only the export
  [ "$(git -C "$repo" rev-list --count HEAD)" = 1 ] || { echo "more than one commit"; return 1; }
  local missing
  missing="$(comm -23 <(cd "$STAGE/tree" && find . -type f | sed 's|^\./||' | sort) \
                      <(git -C "$repo" ls-files | sort))"
  [ -z "$missing" ] || { printf 'exported files missing from the commit:\n%s\n' "$missing"; return 1; }
  # the identity on the OBJECTS, not the config we asked for
  local an ae cn ce
  an="$(git -C "$repo" log -1 --format=%an)"; ae="$(git -C "$repo" log -1 --format=%ae)"
  cn="$(git -C "$repo" log -1 --format=%cn)"; ce="$(git -C "$repo" log -1 --format=%ce)"
  [ "$an" = "$AUTHOR_NAME" ] && [ "$cn" = "$AUTHOR_NAME" ] || { echo "commit names '$an'/'$cn' != '$AUTHOR_NAME'"; return 1; }
  [ "$ae" = "$RELEASE_AUTHOR_EMAIL" ] && [ "$ce" = "$RELEASE_AUTHOR_EMAIL" ] || { echo "commit e-mails '$ae'/'$ce' != '$RELEASE_AUTHOR_EMAIL'"; return 1; }
  git -C "$repo" log -1 --format='%h %an <%ae> %s'
}
_step repo _repo

# ═══ report, and the door ═════════════════════════════════════════════════
_report
if [ -d "$STAGE/repo/.git" ]; then
  say "fresh repo: $STAGE/repo  (tag v$VERSION, $(git -C "$STAGE/repo" ls-files | wc -l | tr -d ' ') files)"
fi
# Said on every run, because it is the one thing the command cannot do: Claude
# Code has no downgrade from a marketplace repo, and schema migrations are
# one-way. The protection is the daemon's pre-destructive backups plus a fast
# follow-up release.
cat <<EOF
[release] Rollback is FIX-FORWARD. Claude Code offers no downgrade from a marketplace repo and
[release] the brain's schema migrations are one-way; a bad release is undone by the next release,
[release] and the user's data is protected by the daemon's pre-destructive backups, not by a revert.
EOF
if [ -z "$PUBLISH_URL" ]; then
  cat <<EOF
[release] DRY RUN — nothing pushed. To publish, re-run on a clean main with
[release]   RELEASE_AUTHOR_EMAIL=... $0 $VERSION --publish $REPO_URL
[release] which will run every step again and then:
[release]   git -C $STAGE/repo remote add origin $REPO_URL
[release]   git -C $STAGE/repo push -u origin main
[release]   git -C $STAGE/repo push origin v$VERSION
EOF
  exit 0
fi
say "publishing to $PUBLISH_URL"
git -C "$STAGE/repo" remote add origin "$PUBLISH_URL"
git -C "$STAGE/repo" push -u origin main
git -C "$STAGE/repo" push origin "v$VERSION"
say "PUBLISHED v$VERSION — the door is open; history begins at $(git -C "$STAGE/repo" rev-parse --short HEAD)"
