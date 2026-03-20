#!/bin/bash
# brain v5 — Post-response vocabulary gap detector
# Fires on user_message notifications (same event as pre-response-recall).
# Extracts candidate terms from the user's message and checks if they
# map to known vocabulary nodes. Unmapped terms are stored in the
# vocabulary_gap consciousness signal for surfacing at boot.
#
# Lightweight: regex extraction + SQLite lookups, no 3rd-party NLP.

# ── Resolve brain DB ──
source "$(dirname "$0")/resolve-brain-db.sh"

if [ -z "$BRAIN_DB_DIR" ] || [ ! -f "$BRAIN_DB_DIR/brain.db" ]; then
  exit 0
fi
export HOOK_INPUT=$(cat)

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

user_message = hook_input.get("message", "")
if not user_message or len(user_message) < 10:
    sys.exit(0)

# ── Extract candidate terms ──
# Strategy: find noun-phrase-like sequences near code/action words
# that could be operator shorthand for code concepts.

# 1. Quoted terms — highest signal ("the recall hook", "consciousness loop")
quoted = re.findall(r"[\"]([\w\s-]{3,30})[\"]", user_message)

# 2. "the X" patterns — "the recall hook", "the sidecar table", "the boot sequence"
the_patterns = re.findall(r"\bthe\s+([\w][\w\s-]{2,25}(?:hook|script|table|file|function|method|class|module|layer|loop|sequence|pipeline|system|engine|server|db|database|config|schema|signal|node|type|map|graph|cache|queue|log|test|spec))\b", user_message, re.IGNORECASE)

# 3. Compound terms near action verbs — "fix the warm-up problem", "update recall hook"
action_context = re.findall(r"\b(?:fix|update|change|modify|check|look at|review|debug|test|refactor|rewrite|add|remove|delete|move|rename|split|merge|clean)\s+(?:the\s+)?([\w]+(?:\s+[\w]+){0,3}?)(?:\s*(?:and|or|but|also|then|,|;|\.|!|\?)|\s*$)", user_message, re.IGNORECASE)
# Clean up trailing whitespace from captures
action_context = [t.strip() for t in action_context if len(t.strip()) >= 3]

# 4. Hyphenated compound terms — "warm-up", "pre-edit", "auto-heal"
hyphenated = re.findall(r"\b([\w]+-[\w]+(?:-[\w]+)?)\b", user_message)
# Filter out common non-terms
hyphenated = [h for h in hyphenated if len(h) > 4 and h.lower() not in (
    "re-run", "re-do", "non-null", "non-zero", "up-to-date", "e-g",
    "pre-existing", "co-authored-by",
)]

# Combine and deduplicate
candidates = set()
for term in quoted + the_patterns + action_context + hyphenated:
    term = term.strip().lower()
    # Skip too short, too long, or pure stopwords
    if len(term) < 3 or len(term) > 40:
        continue
    # Skip common phrases that are not vocabulary
    skip_words = {"the", "a", "an", "this", "that", "it", "them", "is", "are",
                  "was", "were", "be", "been", "do", "does", "did", "have", "has",
                  "can", "could", "will", "would", "should", "may", "might",
                  "yes", "no", "ok", "sure", "please", "thanks", "code", "file",
                  "thing", "stuff", "something", "everything", "nothing"}
    words = term.split()
    if all(w in skip_words for w in words):
        continue
    candidates.add(term)

if not candidates:
    sys.exit(0)

# ── Check against brain vocabulary ──
try:
    from servers.brain import Brain
    brain = Brain(db_path)
except Exception:
    sys.exit(0)

# Load all existing vocabulary terms for fast matching
try:
    vocab_rows = brain.conn.execute(
        "SELECT LOWER(title), LOWER(content) FROM nodes WHERE type = %s AND archived = 0" % repr("vocabulary")
    ).fetchall()
except Exception:
    vocab_rows = []

# Build a set of known terms (from titles and content)
known_terms = set()
for title, content in vocab_rows:
    # Extract the mapped term from the title (format: "term → maps_to" or just "term")
    if " → " in title:
        known_terms.add(title.split(" → ")[0].strip())
    elif " -> " in title:
        known_terms.add(title.split(" -> ")[0].strip())
    else:
        known_terms.add(title.strip())

# Also check node titles broadly — if a term matches an existing node title,
# it is not really unmapped (the user just knows what things are called)
try:
    all_titles = brain.conn.execute(
        "SELECT LOWER(title) FROM nodes WHERE archived = 0"
    ).fetchall()
    title_set = {r[0] for r in all_titles}
except Exception:
    title_set = set()

unmapped = []
for term in candidates:
    # Skip if it matches a known vocabulary term
    if term in known_terms:
        continue
    # Skip if it exactly matches an existing node title
    if term in title_set:
        continue
    # Skip if the term is a substring of a known vocab term or vice versa
    if any(term in kt or kt in term for kt in known_terms):
        continue
    # Skip if any word in the term matches a node title exactly
    # (means the user is using standard names, not shorthand)
    words = term.split()
    if len(words) == 1 and any(term in t for t in title_set):
        continue
    unmapped.append(term)

if unmapped:
    # Store unmapped terms in brain_meta for the vocabulary_gap signal to pick up
    try:
        existing = brain.get_config("vocabulary_gaps", "[]")
        try:
            gaps = json.loads(existing)
        except Exception:
            gaps = []

        # Add new gaps, keep max 20, deduplicate
        existing_terms = {g.get("term") if isinstance(g, dict) else g for g in gaps}
        for term in unmapped[:5]:
            if term not in existing_terms:
                gaps.append({"term": term, "message_preview": user_message[:80]})

        # Keep only most recent 20
        gaps = gaps[-20:]
        brain.set_config("vocabulary_gaps", json.dumps(gaps))
        brain.save()
    except Exception:
        pass

# ── Encoding heartbeat ──
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
    brain.save()
except Exception:
    pass

brain.close()
' 2>/dev/null
