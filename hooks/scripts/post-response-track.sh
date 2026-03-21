#!/bin/bash
# brain v5.3 — Post-response tracker: vocab gap detection + 5-focus rotating encoding checkpoints
# Fires on TWO events:
#   - UserPromptSubmit (input has "prompt" field) — vocab gap detection + heartbeat
#   - Stop (input has "last_assistant_message" field) — encoding heartbeat only
#
# Previously also registered on Notification(user_message) which was INVALID —
# that event type doesn't exist, so the hook was silently dead.
#
# Lightweight: regex extraction + SQLite lookups, no 3rd-party NLP.

# ── Resolve brain DB ──
source "$(dirname "$0")/resolve-brain-db.sh"

if [ -z "$BRAIN_DB_DIR" ] || [ ! -f "$BRAIN_DB_DIR/brain.db" ]; then
  exit 0
fi
export HOOK_INPUT=$(cat)

# ── Try daemon first (fast path) ──
SOCKET_PATH="/tmp/brain-daemon-$(id -u).sock"
if [ -S "$SOCKET_PATH" ]; then
  DAEMON_OK=$(python3 -c '
import json, sys, os, socket, re

hook_input = json.loads(os.environ.get("HOOK_INPUT", "{}"))
# UserPromptSubmit provides "prompt", Notification provided "message"
user_message = hook_input.get("prompt", "") or hook_input.get("message", "")

sock_path = "'"$SOCKET_PATH"'"

# Record message + get heartbeat
try:
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock.settimeout(5.0)
    sock.connect(sock_path)
    msg = json.dumps({"cmd": "record_message", "args": {}}) + "\n"
    sock.sendall(msg.encode())
    data = b""
    while b"\n" not in data:
        chunk = sock.recv(65536)
        if not chunk: break
        data += chunk
    resp = json.loads(data.decode().strip())
    sock.close()

    if resp.get("ok"):
        nudge = resp.get("result", {}).get("nudge")
        if nudge:
            severity = nudge.get("severity", "gentle")
            nudge_msg = nudge.get("message", "")

            # 5-focus rotating checkpoint (eval-tested best strategy)
            checkpoint_cycle = [
                "UNCERTAINTY: What don\u0027t you fully understand? Encode brain.remember_uncertainty(). Honest \u0027I don\u0027t know\u0027 > thin facts.",
                "CONNECTIONS: Use brain.connect() between related nodes. brain.remember_impact() for dependencies. Orphan nodes die.",
                "DECISIONS + LESSONS: brain.remember(type=\u0027decision\u0027, locked=True) with reasoning. brain.remember_lesson() for mistakes.",
                "BLAST RADIUS: brain.remember_impact(if_changed, must_check, because). Map what breaks if this changes.",
                "PATTERNS: brain.remember_convention() for patterns. brain.remember(type=\u0027mental_model\u0027) for architecture.",
            ]

            # Get + rotate checkpoint index via daemon
            try:
                sock_idx = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                sock_idx.settimeout(3.0)
                sock_idx.connect(sock_path)
                sock_idx.sendall(json.dumps({"cmd": "get_config", "args": {"key": "checkpoint_index", "default": "0"}}).encode() + b"\n")
                idx_data = b""
                while b"\n" not in idx_data:
                    chunk = sock_idx.recv(4096)
                    if not chunk: break
                    idx_data += chunk
                idx_resp = json.loads(idx_data.decode().strip())
                sock_idx.close()
                idx = int(idx_resp.get("result", "0")) if idx_resp.get("ok") else 0
            except Exception:
                idx = 0

            focus = checkpoint_cycle[idx % len(checkpoint_cycle)]

            # Rotate index
            try:
                sock_set = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                sock_set.settimeout(3.0)
                sock_set.connect(sock_path)
                sock_set.sendall(json.dumps({"cmd": "set_config", "args": {"key": "checkpoint_index", "value": str((idx + 1) % len(checkpoint_cycle))}}).encode() + b"\n")
                sock_set.recv(4096)
                sock_set.close()
            except Exception:
                pass

            print("")
            print("ENCODING CHECKPOINT: " + nudge_msg)
            print(focus)
            print("")

    # Vocab gap detection (if we have a message)
    if user_message and len(user_message) >= 10:
        candidates = set()
        quoted = re.findall(r"[\"]([\w\s-]{3,30})[\"]", user_message)
        hyphenated = [h for h in re.findall(r"\b([\w]+-[\w]+(?:-[\w]+)?)\b", user_message) if len(h) > 4]
        for term in quoted + hyphenated:
            term = term.strip().lower()
            if len(term) >= 3 and len(term) <= 40:
                candidates.add(term)
        # Vocab check is lightweight via daemon — skip for now, main value is heartbeat

    # Save
    sock2 = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock2.settimeout(5.0)
    sock2.connect(sock_path)
    sock2.sendall(json.dumps({"cmd": "save", "args": {}}).encode() + b"\n")
    sock2.recv(65536)
    sock2.close()

    print("OK")
except Exception as e:
    print("FAIL:" + str(e))
    sys.exit(1)
' 2>/dev/null)

  if [ "$DAEMON_OK" = "OK" ] || echo "$DAEMON_OK" | grep -q "ENCODING"; then
    # Daemon handled it — output any nudges
    echo "$DAEMON_OK" | grep -v "^OK$" | grep -v "^FAIL"
    exit 0
  fi
  # Fall through to direct Python
fi

# ── Direct Python fallback ──
python3 -c '
import json, sys, os, re

server_dir = os.environ.get("BRAIN_SERVER_DIR", "")
db_dir = os.environ.get("BRAIN_DB_DIR", "")
db_path = os.path.join(db_dir, "brain.db")

if server_dir:
    parent = os.path.dirname(server_dir)
    sys.path.insert(0, parent)

try:
    hook_input = json.loads(os.environ.get("HOOK_INPUT", "{}"))
except Exception:
    sys.exit(0)

# UserPromptSubmit provides "prompt", Stop provides "last_assistant_message"
# Vocab gap detection needs the user message (from prompt field)
user_message = hook_input.get("prompt", "") or hook_input.get("message", "")
event_name = hook_input.get("hook_event_name", "")

has_user_message = user_message and len(user_message) >= 10

# If no user message and not a Stop event, nothing to do
if not has_user_message and event_name != "Stop":
    sys.exit(0)

# ── Load brain ──
try:
    from servers.brain import Brain
    brain = Brain(db_path)
except Exception:
    sys.exit(0)

# ── Vocab gap detection (only when we have user message) ──
if has_user_message:
    try:
        # Extract candidate terms
        quoted = re.findall(r"[\"]([\w\s-]{3,30})[\"]", user_message)
        the_patterns = re.findall(r"\bthe\s+([\w][\w\s-]{2,25}(?:hook|script|table|file|function|method|class|module|layer|loop|sequence|pipeline|system|engine|server|db|database|config|schema|signal|node|type|map|graph|cache|queue|log|test|spec))\b", user_message, re.IGNORECASE)
        action_context = re.findall(r"\b(?:fix|update|change|modify|check|look at|review|debug|test|refactor|rewrite|add|remove|delete|move|rename|split|merge|clean)\s+(?:the\s+)?([\w]+(?:\s+[\w]+){0,3}?)(?:\s*(?:and|or|but|also|then|,|;|\.|!|\?)|\s*$)", user_message, re.IGNORECASE)
        action_context = [t.strip() for t in action_context if len(t.strip()) >= 3]
        hyphenated = re.findall(r"\b([\w]+-[\w]+(?:-[\w]+)?)\b", user_message)
        hyphenated = [h for h in hyphenated if len(h) > 4 and h.lower() not in (
            "re-run", "re-do", "non-null", "non-zero", "up-to-date", "e-g",
            "pre-existing", "co-authored-by",
        )]

        candidates = set()
        skip_words = {"the", "a", "an", "this", "that", "it", "them", "is", "are",
                      "was", "were", "be", "been", "do", "does", "did", "have", "has",
                      "can", "could", "will", "would", "should", "may", "might",
                      "yes", "no", "ok", "sure", "please", "thanks", "code", "file",
                      "thing", "stuff", "something", "everything", "nothing"}
        for term in quoted + the_patterns + action_context + hyphenated:
            term = term.strip().lower()
            if len(term) < 3 or len(term) > 40:
                continue
            words = term.split()
            if all(w in skip_words for w in words):
                continue
            candidates.add(term)

        if candidates:
            # Check against brain vocabulary
            try:
                vocab_rows = brain.conn.execute(
                    "SELECT LOWER(title), LOWER(content) FROM nodes WHERE type = ? AND archived = 0", ("vocabulary",)
                ).fetchall()
            except Exception:
                vocab_rows = []

            known_terms = set()
            for title, content in vocab_rows:
                if " \u2192 " in title:
                    known_terms.add(title.split(" \u2192 ")[0].strip())
                elif " -> " in title:
                    known_terms.add(title.split(" -> ")[0].strip())
                else:
                    known_terms.add(title.strip())

            try:
                all_titles = brain.conn.execute("SELECT LOWER(title) FROM nodes WHERE archived = 0").fetchall()
                title_set = {r[0] for r in all_titles}
            except Exception:
                title_set = set()

            unmapped = []
            for term in candidates:
                if term in known_terms:
                    continue
                if term in title_set:
                    continue
                if any(term in kt or kt in term for kt in known_terms):
                    continue
                words = term.split()
                if len(words) == 1 and any(term in t for t in title_set):
                    continue
                unmapped.append(term)

            if unmapped:
                existing = brain.get_config("vocabulary_gaps", "[]")
                try:
                    gaps = json.loads(existing)
                except Exception:
                    gaps = []
                existing_terms = {g.get("term") if isinstance(g, dict) else g for g in gaps}
                for term in unmapped[:5]:
                    if term not in existing_terms:
                        gaps.append({"term": term, "message_preview": user_message[:80]})
                gaps = gaps[-20:]
                brain.set_config("vocabulary_gaps", json.dumps(gaps))
    except Exception:
        pass

# ── Encoding heartbeat with rotating checkpoints ──
try:
    brain.record_message()
    nudge = brain.get_encoding_heartbeat()
    if nudge:
        severity = nudge.get("severity", "gentle")
        msg = nudge.get("message", "")

        # Rotating checkpoint: 5-focus cycle (eval-tested best strategy)
        checkpoint_cycle = [
            "UNCERTAINTY: What don\u0027t you fully understand from the last few exchanges? Encode at least one brain.remember_uncertainty(). Honest \u0027I don\u0027t know\u0027 is more valuable than thin facts.",
            "CONNECTIONS: What connections did you discover? Use brain.connect() between related nodes and brain.remember_impact(if_changed, must_check, because) for dependencies. Orphan nodes die.",
            "DECISIONS + LESSONS: What was decided or learned? brain.remember(type=\u0027decision\u0027, locked=True) with full reasoning. brain.remember_lesson() for any bugs or mistakes. Include WHY, not just WHAT.",
            "BLAST RADIUS: What could break if this code changes? brain.remember_impact(if_changed=\u0027component\u0027, must_check=[\u0027dependents\u0027], because=\u0027reason\u0027). Map the ripple effects you noticed.",
            "PATTERNS: What patterns or conventions did you observe? brain.remember_convention() for coding patterns. brain.remember(type=\u0027mental_model\u0027) for architectural insights. Name the pattern.",
        ]

        # Get checkpoint index from brain config (rotates 0, 1, 2, 0, 1, 2...)
        try:
            idx = int(brain.get_config("checkpoint_index", "0"))
        except Exception:
            idx = 0
        focus = checkpoint_cycle[idx % len(checkpoint_cycle)]
        brain.set_config("checkpoint_index", str((idx + 1) % len(checkpoint_cycle)))

        # Get session encoding stats for context
        try:
            session_id = brain.get_config("current_session_id", "")
            node_count = brain.conn.execute(
                "SELECT COUNT(*) FROM nodes WHERE created_at > datetime(\u0027now\u0027, \u0027-2 hours\u0027)"
            ).fetchone()[0]
            uncert_count = brain.conn.execute(
                "SELECT COUNT(*) FROM nodes WHERE type = \u0027uncertainty\u0027 AND created_at > datetime(\u0027now\u0027, \u0027-2 hours\u0027)"
            ).fetchone()[0]
            connect_count = brain.conn.execute(
                "SELECT COUNT(*) FROM edges WHERE created_at > datetime(\u0027now\u0027, \u0027-2 hours\u0027)"
            ).fetchone()[0]
            stats = f"Session stats: {node_count} nodes, {uncert_count} uncertainties, {connect_count} connections."
        except Exception:
            stats = ""

        if severity == "urgent":
            print("")
            print("ENCODING CHECKPOINT: " + msg)
            if stats:
                print(stats)
            print(focus)
            print("")
        else:
            print("")
            print("ENCODING CHECKPOINT: " + msg)
            if stats:
                print(stats)
            print(focus)
            print("")
except Exception:
    pass

try:
    brain.save()
    brain.close()
except Exception:
    pass
' 2>/dev/null
