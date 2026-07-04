"""Pre-response recall — thin wrapper that calls daemon for recall + judge.

The daemon handles everything: Layer 1 recall, Layer 2 Haiku judge, Layer 3 graph expansion.
This script just passes the user message to the daemon and prints the result.

Flow:
1. Send user message to daemon via "hook_recall" command
2. Daemon does recall → judge → graph expand → formats additionalContext
3. This script prints the result (additionalContext or approve)
"""
import sys, os, json, time

_t0 = time.time()
sys.path.insert(0, os.path.dirname(__file__))
from hook_common import (get_hook_input, daemon_available, daemon_call_raw,
                         daemon_unavailable_error, brain_debug, run_hook)
from datetime import datetime as _dt
def _ts(): return _dt.now().strftime("%H:%M:%S.%f")[:-3]
sys.stderr.write("[recall-hook %s] import: %dms\n" % (_ts(), (time.time() - _t0) * 1000))

APPROVE = json.dumps({"decision": "approve"})

# Answers with fewer meaningful chars than this register the turn but skip the
# recall + Haiku surface (register_only). A bare answer carries no recall signal.
SHORT_MESSAGE_MAX_LEN = 5

hook_input = get_hook_input()
user_message = hook_input.get("prompt", "") or hook_input.get("message", "")

# Slash / bang / empty (incl. whitespace-only) are genuinely non-conversational —
# slash commands, /watch wakeups, bang. Skip the daemon entirely; they correctly
# read as heartbeats at Stop (no user_message trace, never encoded).
if not user_message.strip() or user_message.startswith("/") or user_message.startswith("!"):
    brain_debug("recall: skipped (slash/bang/empty)")
    print(APPROVE)
    sys.exit(0)

# Short real answers ("yes", "ok", "no") ARE conversational but carry no recall
# signal. Register the turn (user_message trace + conversational classification)
# WITHOUT the recall + Haiku surface, via register_only. Dropping them entirely
# (the old `len < 5` skip) misfiled the turn as a heartbeat and lost the
# operator's words — often the highest-signal turns (approvals/decisions).
# Measure stripped length so " ok " counts as 2, not 4. See
# daemon_hooks.hook_recall register-only fast path.
register_only = len(user_message.strip()) < SHORT_MESSAGE_MAX_LEN

# Harness-injected background-task completions arrive through THIS same
# UserPromptSubmit channel — the harness packages a <task-notification> as a
# prompt, so it passes the slash/bang/short gate above and would otherwise run
# the full recall + Haiku surface. That's pure waste here (I'm mid-task with
# full context) AND actively harmful: every surfaced candidate gets marked
# accessed under this session_id, so machine chatter pollutes synaptic fatigue
# and dampens my NEXT real prompt. Route them register_only — keep the
# user_message trace (turn stays conversational, so my substantive response to
# the results is still encoded at Stop) but skip recall + Haiku + fatigue.
# Full-skip (slash/bang path) would misclassify the turn as a heartbeat and
# drop that response from encoding.
if "<task-notification>" in user_message:
    register_only = True


def main():
    t0 = time.time()
    if not daemon_available():
        # Register-only is best-effort: there's nothing to recall, so a down
        # daemon must fail SILENT (approve) — not surface the recall-unavailable
        # banner for a bare "yes". Worst case the turn goes unregistered, which
        # is exactly the old pre-fix behavior, not a regression.
        if register_only:
            print(APPROVE)
            sys.exit(0)
        err = daemon_unavailable_error("recall")
        print(json.dumps({"hookSpecificOutput": {"hookEventName": "UserPromptSubmit", "additionalContext": err}}))
        sys.exit(0)

    # Call daemon — it handles Layer 1 + Layer 2 judge + Layer 3 graph expand.
    # register_only does no Haiku/recall (just a trace write), so it fails fast
    # rather than carrying the 20s Haiku-tail budget on the prompt path.
    resp = daemon_call_raw("hook_recall", {
        "prompt": hook_input.get("prompt", ""),
        "message": hook_input.get("message", ""),
        "session_id": hook_input.get("session_id", ""),
        "register_only": register_only,  # short answers: register turn, skip recall+Haiku
    }, timeout=4.0 if register_only else 20.0)  # 20s covers Haiku tail latency
                       # under load (2026-05-02, 14→20). See FRAME-DESIGN.md
                       # and node 2340b053. register_only needs none of it.

    if not resp.get("ok"):
        # Register-only failure is best-effort too — approve silently rather than
        # surface RECALL FAILED for a turn that had nothing to recall.
        if register_only:
            print(APPROVE)
            sys.exit(0)
        err_msg = resp.get("error", "unknown error")
        # daemon_call_raw already logged this failure to hook_errors (single
        # source of truth). We only render the user-facing message here.
        print(json.dumps({"hookSpecificOutput": {"hookEventName": "UserPromptSubmit", "additionalContext":
            "[BRAIN]\n⚠️ RECALL FAILED: %s\nThe brain could not search for relevant memories.\n[/BRAIN]" % err_msg}}))
        sys.exit(0)

    result = resp.get("result", {})
    elapsed = int((time.time() - t0) * 1000)

    # The daemon returns either additionalContext (judge completed) or approve (no results/judge failed)
    result_json = result.get("json", {})
    if "additionalContext" in result_json:
        context = result_json["additionalContext"]
        brain_debug("recall: daemon returned context (%d chars) in %dms" % (len(context), elapsed))
        sys.stderr.write("[recall-hook %s] total: %dms, printing and exiting\n" % (_ts(), (time.time() - _t0) * 1000))
        sys.stdout.write(json.dumps({"hookSpecificOutput": {"hookEventName": "UserPromptSubmit", "additionalContext": context}}))
        sys.stdout.flush()
        os._exit(0)  # Fast exit — skip Python cleanup
    else:
        brain_debug("recall: daemon returned approve in %dms" % elapsed)
        sys.stdout.write(json.dumps(result_json))
        sys.stdout.flush()
        os._exit(0)


run_hook("recall", main, on_error=lambda: print(APPROVE))
