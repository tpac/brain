#!/bin/bash
# brain v14 (serverless) — PreToolUse hook for Edit|Write
# Direct Python brain module call. No HTTP server needed.
#
# Input: JSON on stdin with tool_name, tool_input.file_path
# Output: JSON with decision + reason containing brain suggestions

source "$(dirname "$0")/resolve-brain-db.sh"

if [ -z "$BRAIN_DB_DIR" ] || [ ! -f "$BRAIN_DB_DIR/brain.db" ]; then
  echo '{"decision":"approve"}'
  exit 0
fi

# Read hook input from stdin
export HOOK_INPUT=$(cat)

# ── Try daemon first (fast path: ~50ms vs ~2000ms) ──
source "$(dirname "$0")/daemon-client.sh"
if daemon_available; then
  DAEMON_RESULT=$(python3 -c '
import json, sys, os, socket, time

hook_input = json.loads(os.environ.get("HOOK_INPUT", "{}"))
tool_input = hook_input.get("tool_input", {})
file_path = tool_input.get("file_path", "")
tool_name = hook_input.get("tool_name", "Edit")

if not file_path:
    print(json.dumps({"decision": "approve"}))
    sys.exit(0)

filename = os.path.basename(file_path)

# Skip non-source files
skip_exts = [".log", ".map", ".lock", ".json"]
if any(filename.endswith(ext) for ext in skip_exts) and filename != "package.json":
    print(json.dumps({"decision": "approve"}))
    sys.exit(0)

sock_path = "'"$BRAIN_SOCKET_PATH"'"
t_start = time.time()

try:
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock.settimeout(10.0)
    sock.connect(sock_path)

    msg = json.dumps({"cmd": "pre_edit", "args": {"file": filename, "tool_name": tool_name}}) + "\n"
    sock.sendall(msg.encode())
    data = b""
    while b"\n" not in data:
        chunk = sock.recv(65536)
        if not chunk: break
        data += chunk
    resp = json.loads(data.decode().strip())
    sock.close()

    if not resp.get("ok"):
        raise Exception("Daemon pre_edit failed")

    result = resp.get("result", {})
    suggestions = result.get("suggestions", [])
    procedures = result.get("procedures", [])
    context_files = result.get("context_files", [])
    change_impacts = result.get("change_impacts", [])
    encoding = result.get("encoding", {})

    # ── Encoding health warning ──
    encoding_warning = ""
    health = encoding.get("health", "OK")
    edits_gap = encoding.get("edits_since_last_remember", 0)
    mins_since = encoding.get("minutes_since_last_remember", 0)
    session_min = encoding.get("session_minutes", 0)

    if health == "NONE" and session_min > 3:
        encoding_warning = (
            "ENCODING ALERT: You have not stored ANY learnings in the brain this session. "
            "If decisions were made, corrections happened, or the user gave feedback — "
            "call /remember NOW before continuing."
        )
    elif health == "STALE":
        if edits_gap > 15:
            encoding_warning = (
                "ENCODING WARNING: %d edits since your last /remember call "
                "(%d min ago). Store decisions and corrections now." % (edits_gap, mins_since)
            )
        elif edits_gap > 8:
            encoding_warning = (
                "ENCODING CHECK: %d edits since last /remember. "
                "Anything worth storing?" % edits_gap
            )

    if not suggestions and not procedures and not context_files and not change_impacts:
        if encoding_warning:
            print(json.dumps({"decision": "approve", "reason": encoding_warning}))
        else:
            print(json.dumps({"decision": "approve"}))
        sys.exit(0)

    # ── Format output (same as direct Python path) ──
    lines = ["BRAIN AUTO-SUGGEST for %s:" % filename, ""]

    ENGINEERING_TYPES = {"purpose", "mechanism", "impact", "constraint", "convention", "lesson", "vocabulary"}
    CODE_COGNITION_TYPES = {"fn_reasoning", "param_influence", "code_concept", "arch_constraint", "causal_chain", "bug_lesson", "comment_anchor"}

    eng_nodes = [s for s in suggestions if s.get("type") in ENGINEERING_TYPES]
    code_nodes = [s for s in suggestions if s.get("type") in CODE_COGNITION_TYPES]
    other_nodes = [s for s in suggestions if s.get("type") not in ENGINEERING_TYPES and s.get("type") not in CODE_COGNITION_TYPES and s.get("type") != "procedure" and not (s.get("type") == "file" and "[ctx:" in s.get("title", ""))]

    if change_impacts:
        lines.append("CHANGE IMPACT WARNING:")
        lines.append("")
        for ci in change_impacts[:5]:
            ci_title = ci.get("title", "")[:80]
            ci_content = ci.get("content", "")
            if len(ci_content) > 300: ci_content = ci_content[:300] + "..."
            lines.append("  [impact] " + ci_title)
            lines.append("    " + ci_content)
            lines.append("")

    if eng_nodes:
        lines.append("ENGINEERING MEMORY (read carefully — these describe what you are about to edit):")
        lines.append("")
        for s in eng_nodes:
            typ = s.get("type", "?")
            title = s.get("title", "")[:80]
            content = s.get("content", "")
            locked = "LOCKED " if s.get("locked") else ""
            if len(content) > 350: content = content[:350] + "..."
            lines.append("  [%s] %s%s" % (typ, locked, title))
            lines.append("    " + content)
            lines.append("")

    if code_nodes:
        lines.append("CODE KNOWLEDGE:")
        lines.append("")
        for s in code_nodes:
            typ = s.get("type", "?")
            title = s.get("title", "")[:80]
            content = s.get("content", "")
            locked = "LOCKED " if s.get("locked") else ""
            if len(content) > 350: content = content[:350] + "..."
            lines.append("  [%s] %s%s" % (typ, locked, title))
            lines.append("    " + content)
            lines.append("")

    if other_nodes:
        if code_nodes: lines.append("OTHER RULES & DECISIONS:")
        lines.append("")
        for s in other_nodes:
            typ = s.get("type", "?")
            title = s.get("title", "")[:80]
            content = s.get("content", "")
            locked = "LOCKED " if s.get("locked") else ""
            if len(content) > 250: content = content[:250] + "..."
            lines.append("  [%s] %s%s" % (typ, locked, title))
            lines.append("    " + content)
            lines.append("")

    if procedures:
        lines.append("TRIGGERED PROCEDURES:")
        for p in procedures[:3]:
            ptitle = p.get("title", "")
            lines.append("  [procedure] " + ptitle)
            psteps = p.get("steps", "")
            if len(psteps) > 300: psteps = psteps[:300] + "..."
            lines.append("    " + psteps)
            lines.append("")

    if context_files:
        lines.append("CONTEXT FILES:")
        for cf in context_files[:2]:
            cftopic = cf.get("topic", "")
            cftitle = cf.get("title", "")
            lines.append("  [%s] %s" % (cftopic, cftitle))
        lines.append("")

    if encoding_warning:
        lines.append("")
        lines.append(encoding_warning)
        lines.append("")

    locked_ids = [s.get("id", "") for s in suggestions if s.get("locked")]
    if locked_ids:
        lines.append("BRAIN->HOST: If you follow locked rules above, call brain.log_communication(node_id, '"'"'high_priority'"'"', True).")
        lines.append("")

    lines.append("Review these constraints before proceeding with the edit.")
    context = "\n".join(lines)
    print(json.dumps({"decision": "approve", "reason": context}))

except Exception as e:
    print("DAEMON_FALLBACK")
    sys.exit(1)
' 2>/dev/null)

  if [ $? -eq 0 ] && [ -n "$DAEMON_RESULT" ] && [ "$DAEMON_RESULT" != "DAEMON_FALLBACK" ]; then
    echo "$DAEMON_RESULT"
    exit 0
  fi
  # Fall through to direct Python
fi

# ── Direct Python fallback (cold path: ~2000ms) ──
exec python3 -c '
import json, sys, os, time

server_dir = os.environ.get("BRAIN_SERVER_DIR", "")
db_dir = os.environ.get("BRAIN_DB_DIR", "")
db_path = os.path.join(db_dir, "brain.db")

# Add servers dir to Python path
if server_dir:
    parent = os.path.dirname(server_dir)
    sys.path.insert(0, parent)

t_start = time.time()

# Parse hook input
try:
    hook_input = json.loads(os.environ.get("HOOK_INPUT", "{}"))
except Exception:
    print(json.dumps({"decision": "approve"}))
    sys.exit(0)

tool_input = hook_input.get("tool_input", {})
file_path = tool_input.get("file_path", "")
tool_name = hook_input.get("tool_name", "Edit")

if not file_path:
    print(json.dumps({"decision": "approve"}))
    sys.exit(0)

filename = os.path.basename(file_path)

# Skip non-source files
skip_exts = [".log", ".map", ".lock", ".json"]
if any(filename.endswith(ext) for ext in skip_exts) and filename != "package.json":
    print(json.dumps({"decision": "approve"}))
    sys.exit(0)

# Initialize brain
try:
    from servers.brain import Brain
    brain = Brain(db_path)
except Exception:
    print(json.dumps({"decision": "approve"}))
    sys.exit(0)

# Single batch call — replaces 8 HTTP requests
try:
    data = brain.pre_edit(file=filename, tool_name=tool_name)
except Exception:
    brain.close()
    print(json.dumps({"decision": "approve"}))
    sys.exit(0)

suggestions = data.get("suggestions", [])
procedures = data.get("procedures", [])
context_files = data.get("context_files", [])
encoding = data.get("encoding", {})
debug_enabled = data.get("debug_enabled", False)

# v5: Change impact maps — "if you modify this, also check..."
change_impacts = []
try:
    change_impacts = brain.get_change_impact(filename)
except Exception:
    pass

# ── Encoding health warning ──
encoding_warning = ""
health = encoding.get("health", "OK")
edits_gap = encoding.get("edits_since_last_remember", 0)
mins_since = encoding.get("minutes_since_last_remember", 0)
session_min = encoding.get("session_minutes", 0)

if health == "NONE" and session_min > 3:
    encoding_warning = (
        "ENCODING ALERT: You have not stored ANY learnings in the brain this session. "
        "If decisions were made, corrections happened, or the user gave feedback — "
        "call /remember NOW before continuing. The brain cannot learn from what you dont store."
    )
elif health == "STALE":
    if edits_gap > 15:
        encoding_warning = (
            f"ENCODING WARNING: {edits_gap} edits since your last /remember call "
            f"({mins_since} min ago). If anything worth remembering happened in that span — "
            "a decision, a correction, a pattern, feedback — store it now."
        )
    elif edits_gap > 8:
        encoding_warning = (
            f"ENCODING CHECK: {edits_gap} edits since last /remember. "
            "Anything worth storing? Decisions, corrections, lessons?"
        )

# ── Early exit if nothing to show ──
if not suggestions and not procedures and not context_files and not change_impacts:
    if encoding_warning:
        print(json.dumps({"decision": "approve", "reason": encoding_warning}))
    else:
        print(json.dumps({"decision": "approve"}))
    brain.save()
    brain.close()
    sys.exit(0)

# ── Format into readable context ──
lines = [f"BRAIN AUTO-SUGGEST for {filename}:", ""]

# v5: Engineering memory types surface first
ENGINEERING_TYPES = {"purpose", "mechanism", "impact", "constraint", "convention", "lesson", "vocabulary"}
# v4: Code cognition types (legacy, still supported)
CODE_COGNITION_TYPES = {"fn_reasoning", "param_influence", "code_concept", "arch_constraint", "causal_chain", "bug_lesson", "comment_anchor"}

eng_nodes = [s for s in suggestions if s.get("type") in ENGINEERING_TYPES]
code_nodes = [s for s in suggestions if s.get("type") in CODE_COGNITION_TYPES]
other_nodes = [s for s in suggestions if s.get("type") not in ENGINEERING_TYPES and s.get("type") not in CODE_COGNITION_TYPES and s.get("type") != "procedure" and not (s.get("type") == "file" and "[ctx:" in s.get("title", ""))]

# v5: Change impact warnings — most critical, show first
if change_impacts:
    lines.append("CHANGE IMPACT WARNING:")
    lines.append("")
    for ci in change_impacts[:5]:
        ci_title = ci.get("title", "")[:80]
        ci_content = ci.get("content", "")
        if len(ci_content) > 300:
            ci_content = ci_content[:300] + "..."
        lines.append("  [impact] " + ci_title)
        lines.append("    " + ci_content)
        lines.append("")

# v5: Engineering memory (purpose, mechanism, constraint, convention, lesson)
if eng_nodes:
    lines.append("ENGINEERING MEMORY (read carefully — these describe what you are about to edit):")
    lines.append("")
    for s in eng_nodes:
        typ = s.get("type", "?")
        title = s.get("title", "")[:80]
        content = s.get("content", "")
        locked = "LOCKED " if s.get("locked") else ""
        if len(content) > 350:
            content = content[:350] + "..."
        lines.append("  [" + typ + "] " + locked + title)
        lines.append("    " + content)
        lines.append("")

if code_nodes:
    lines.append("CODE KNOWLEDGE (read carefully — these describe what you are about to edit):")
    lines.append("")
    for s in code_nodes:
        typ = s.get("type", "?")
        title = s.get("title", "")[:80]
        content = s.get("content", "")
        locked = "LOCKED " if s.get("locked") else ""
        if len(content) > 350:
            content = content[:350] + "..."
        lines.append(f"  [{typ}] {locked}{title}")
        lines.append(f"    {content}")
        lines.append("")

if other_nodes:
    if code_nodes:
        lines.append("OTHER RULES & DECISIONS:")
        lines.append("")
    for s in other_nodes:
        typ = s.get("type", "?")
        title = s.get("title", "")[:80]
        content = s.get("content", "")
        locked = "LOCKED " if s.get("locked") else ""
        if len(content) > 250:
            content = content[:250] + "..."
        lines.append(f"  [{typ}] {locked}{title}")
        lines.append(f"    {content}")
        lines.append("")

if procedures:
    lines.append("TRIGGERED PROCEDURES:")
    for p in procedures[:3]:
        ptitle = p.get("title", "")
        lines.append(f"  [procedure] {ptitle}")
        psteps = p.get("steps", "")
        if len(psteps) > 300:
            psteps = psteps[:300] + "..."
        lines.append(f"    {psteps}")
        lines.append("")

if context_files:
    lines.append("CONTEXT FILES (read before editing — may contain detailed requirements):")
    for cf in context_files[:2]:
        cftopic = cf.get("topic", "")
        cftitle = cf.get("title", "")
        cfupdated = str(cf.get("last_updated", ""))[:10]
        cfsummary = str(cf.get("summary", ""))[:150]
        lines.append(f"  [{cftopic}] {cftitle} (updated {cfupdated})")
        lines.append(f"    {cfsummary}")
        lines.append("")
    lines.append("IMPORTANT: If the context file conflicts with current work, flag the conflict.")
    lines.append("")

if encoding_warning:
    lines.append("")
    lines.append(encoding_warning)
    lines.append("")

# v4: Embed communication tracking instruction
# The host should report back whether it followed the brain guidance
locked_ids = [s.get("id", "") for s in suggestions if s.get("locked")]
if locked_ids:
    lines.append("BRAIN->HOST: If you follow locked rules above, call brain.log_communication(node_id, 'high_priority', True).")
    lines.append("If you must deviate, call brain.log_communication(node_id, 'high_priority', False, reason).")
    lines.append("")

lines.append("Review these constraints before proceeding with the edit.")
context = "\n".join(lines)

# Debug logging
if debug_enabled:
    latency = round((time.time() - t_start) * 1000, 1)
    node_ids = [s.get("id", "") for s in suggestions if s.get("type") != "procedure"]
    node_ids += [p.get("id", "") for p in procedures]
    brain.log_debug(
        event_type="pre_edit",
        source="hook",
        file_target=filename,
        suggestions_served=len([s for s in suggestions if s.get("type") != "procedure"]),
        procedures_served=len(procedures),
        node_ids_served=json.dumps(node_ids),
        latency_ms=latency,
        metadata=json.dumps({"tool": tool_name, "timings": data.get("timings", {})})
    )

brain.save()
brain.close()
print(json.dumps({"decision": "approve", "reason": context}))
'
