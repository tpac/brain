#!/bin/bash
# brain v5.2 — Post-response tracker: vocab gap detection + encoding heartbeat
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
            if severity == "urgent":
                print("")
                print("ENCODING ALERT: " + nudge_msg)
                print("Use brain.remember() to capture decisions, corrections, and learnings.")
                print("")
            else:
                print("")
                print("ENCODING NUDGE: " + nudge_msg)
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

# ── Encoding heartbeat (always runs) ──
try:
    brain.record_message()
    nudge = brain.get_encoding_heartbeat()
    if nudge:
        severity = nudge.get("severity", "gentle")
        msg = nudge.get("message", "")
        if severity == "urgent":
            print("")
            print("ENCODING ALERT: " + msg)
            print("Use brain.remember() to capture decisions, corrections, and learnings.")
            print("")
        else:
            print("")
            print("ENCODING NUDGE: " + msg)
            print("")
except Exception:
    pass

try:
    brain.save()
    brain.close()
except Exception:
    pass
' 2>/dev/null
