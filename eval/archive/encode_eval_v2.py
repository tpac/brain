#!/usr/bin/env python3
"""
Encode Eval v2 — Correct simulation of real Claude Code environment.

Simulates EXACTLY what a Claude with the brain plugin sees:
- CLAUDE.md loaded verbatim
- SKILL.md loaded verbatim
- Real MCP tool definitions from brain_mcp.py
- Boot context snapshot
- User's conversation

Measures encoding quality, judgment, and LLM-benefit.

Usage:
    source .env
    python3 eval/encode_eval_v2.py                           # baseline with all segments
    python3 eval/encode_eval_v2.py --variant current_no_skill # test without SKILL.md
    python3 eval/encode_eval_v2.py --segment memento          # single segment
    python3 eval/encode_eval_v2.py --inspect                  # print full brain inspection
"""

import anthropic
import json
import os
import re
import sys
import time
import argparse
import hashlib
from datetime import datetime
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

ROOT = Path(__file__).parent.parent


# ── Load Production Files Verbatim ────────────────────────────────────

def load_claude_md():
    return (ROOT / "CLAUDE.md").read_text()

def load_skill_md():
    return (ROOT / "skills" / "brain" / "SKILL.md").read_text()

def load_boot_context_snapshot():
    """Load a real boot context snapshot. If we have one saved, use it.
    Otherwise return a minimal realistic boot."""
    snapshot = ROOT / "tests" / "fixtures" / "boot_context_snapshot.txt"
    if snapshot.exists():
        return snapshot.read_text()
    # Minimal realistic boot
    return """[BRAIN] v18 booted from: /Users/tpac/AgentsContext/brain

Session #12

FROM PREVIOUS YOU:
  Session #11: encoded 15 nodes. Key topics: V5 enrichments shipped (+78% NDCG), ripple engine killed (-0.002), graph-augmented recall added.

WHAT YOU KNOW ABOUT YOURSELF:
  [lesson] Session #12 encoding drift: built for 9 messages without encoding, compression instinct defeated 3 layers of rules
  [mental_model] Three-consciousness model: Tom conscious→Claude subconscious, Brain is the shared layer

[BRAIN] Key locked rules:
  - Rule: naive Claude must feel the brain as IDENTITY not TOOL
  - Rule: Never swallow errors silently — log, surface, make loud

Brain: 760 nodes, 11835 edges, 565 locked
Embeddings: Snowflake/snowflake-arctic-embed-m-v1.5 (768d)

Use brain MCP tools: recall, remember, connect, eval, consciousness
[/BRAIN]"""


# ── Real MCP Tool Definitions (from brain_mcp.py, converted to Anthropic API format) ──

REAL_BRAIN_TOOLS = [
    {
        "name": "recall",
        "description": "Semantic recall from brain — searches nodes by meaning using embeddings. Returns ranked results with titles, content, types, confidence.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query (semantic, not keyword)"},
                "limit": {"type": "integer", "description": "Max results (default 8)", "default": 8}
            },
            "required": ["query"]
        }
    },
    {
        "name": "remember",
        "description": "Store a new node in the brain. Types: decision, rule, lesson, concept, context, pattern, convention, mechanism, impact, constraint, purpose, mental_model, uncertainty, vocabulary, hypothesis, tension, aspiration, catalyst, interaction, meta_learning, failure_mode, performance, capability, arch_constraint, code_concept, fn_reasoning, param_influence, comment_anchor, bug_lesson.",
        "input_schema": {
            "type": "object",
            "properties": {
                "type": {"type": "string", "description": "Node type"},
                "title": {"type": "string", "description": "Specific, scannable title"},
                "content": {"type": "string", "description": "Rich content with reasoning, tradeoffs, specifics"},
                "locked": {"type": "boolean", "description": "Lock node (for decisions, rules, lessons)", "default": False},
                "confidence": {"type": "number", "description": "Confidence 0.0-1.0", "default": 1.0},
                "keywords": {"type": "string", "description": "Space-separated keywords for search"},
                "project": {"type": "string", "description": "Project scope"},
                "emotion": {"type": "number", "description": "Emotional valence -1.0 to 1.0"}
            },
            "required": ["type", "title", "content"]
        }
    },
    {
        "name": "connect",
        "description": "Create a weighted edge between two brain nodes. Relations: related_to, caused_by, depends_on, contradicts, supports, produced, evolved_from, blocks, enables, example_of.",
        "input_schema": {
            "type": "object",
            "properties": {
                "source_id": {"type": "string", "description": "Source node ID"},
                "target_id": {"type": "string", "description": "Target node ID"},
                "relation": {"type": "string", "description": "Edge relation type", "default": "related_to"},
                "weight": {"type": "number", "description": "Edge weight 0.0-1.0", "default": 0.5}
            },
            "required": ["source_id", "target_id"]
        }
    },
    {
        "name": "consciousness",
        "description": "Get brain consciousness signals — fading knowledge, tensions, vocabulary gaps, encoding health, errors, mental model drift, uncertainties, dream insights, reminders.",
        "input_schema": {"type": "object", "properties": {}}
    },
    {
        "name": "eval",
        "description": "Escape hatch — evaluate arbitrary Python expression on brain object. Variable 'brain' is the Brain instance. Use for methods not exposed as tools (e.g. remember_lesson, remember_impact, record_divergence, learn_vocabulary, etc).",
        "input_schema": {
            "type": "object",
            "properties": {
                "code": {"type": "string", "description": "Python expression to eval (brain object available as 'brain')"}
            },
            "required": ["code"]
        }
    },
    {
        "name": "enrich",
        "description": "Store V5 enrichment vectors for a node (after filling in the enrichment_prompt from remember()). Pass the generated question, anchor phrase, bridge sentence, and/or keywords.",
        "input_schema": {
            "type": "object",
            "properties": {
                "node_id": {"type": "string", "description": "Node ID to enrich (from remember() response)"},
                "question": {"type": "string", "description": "One question a user would ask that leads to this node"},
                "anchor": {"type": "string", "description": "3-5 word phrase using neighbor vocabulary"},
                "bridge": {"type": "string", "description": "One sentence connecting this node to its most important neighbor"},
                "keywords": {"type": "string", "description": "Comma-separated keywords borrowed from neighbors"}
            },
            "required": ["node_id"]
        }
    },
]


# ── Brain Responses (real reads, fake writes) ────────────────────────

_node_counter = [0]

EVAL_DAEMON_HOST = "127.0.0.1"
EVAL_DAEMON_PORT = 47290 + (os.getuid() % 100)  # Different port from production
EVAL_DB_PATH = str(ROOT / "eval" / "fixtures" / "brain_eval_copy.db")

_eval_daemon_started = [False]


def _ensure_eval_daemon():
    """Start a read-only eval daemon on the brain copy if not running."""
    if _eval_daemon_started[0]:
        return
    # Try ping first
    try:
        resp = _daemon_send("ping", timeout=2.0)
        if resp.get("ok"):
            _eval_daemon_started[0] = True
            return
    except Exception:
        pass
    # Start daemon on eval copy — use separate lock/pid to avoid conflicts
    import subprocess
    parent_dir = str(ROOT)
    log_path = str(ROOT / "eval" / "fixtures" / "eval_daemon.log")
    eval_lock = "/tmp/brain-eval-daemon.lock"
    eval_pid = "/tmp/brain-eval-daemon.pid"
    # Clean stale eval lock/pid
    for p in [eval_lock, eval_pid]:
        try:
            if os.path.exists(p):
                os.unlink(p)
        except Exception:
            pass
    with open(log_path, 'w') as log_fd, open(os.devnull, 'r') as devnull:
        env = {**os.environ,
               "VECLIB_MAXIMUM_THREADS": "1",
               "ORT_DISABLE_ALL_ACCELERATORS": "1",
               "ONNX_PROVIDERS": "CPUExecutionProvider",
               "PYTORCH_MPS_DISABLE": "1"}
        # Inline daemon startup that bypasses the shared lock
        startup_code = """
import sys, os, socket, json, signal, time, threading, fcntl, traceback, select
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, %r)
os.environ["ORT_DISABLE_ALL_ACCELERATORS"] = "1"
os.environ["ONNX_PROVIDERS"] = "CPUExecutionProvider"

from servers.brain import Brain
from servers.daemon_dispatch import COMMAND_TABLE
from servers.daemon_config import MAX_MESSAGE_SIZE, AUTOSAVE_INTERVAL_SECONDS

brain = Brain(%r)
print("[eval-daemon] Brain loaded", file=sys.stderr)

srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
srv.bind(("%s", %d))
srv.listen(5)
srv.setblocking(False)
print("[eval-daemon] Listening on %s:%d", file=sys.stderr)

# Write PID
with open(%r, 'w') as f: f.write(str(os.getpid()))

running = True
last_activity = time.time()
write_lock = threading.Lock()
graph_changes = []

def handle(client):
    try:
        data = b""
        while True:
            chunk = client.recv(4096)
            if not chunk: break
            data += chunk
            if b"\\n" in data or len(data) > MAX_MESSAGE_SIZE: break
        if not data: return
        msg = json.loads(data.decode().strip())
        cmd, args = msg.get("cmd",""), msg.get("args",{})
        entry = COMMAND_TABLE.get(cmd)
        if entry:
            if entry.is_write:
                with write_lock:
                    result = entry.handler(brain, args, graph_changes)
            else:
                result = entry.handler(brain, args, graph_changes)
        else:
            result = {"ok": False, "error": "Unknown: " + cmd}
        client.sendall((json.dumps(result, default=str) + "\\n").encode())
    except Exception as e:
        try: client.sendall((json.dumps({"ok":False,"error":str(e)}) + "\\n").encode())
        except: pass
    finally:
        client.close()

pool = ThreadPoolExecutor(max_workers=3)
signal.signal(signal.SIGTERM, lambda s,f: sys.exit(0))

while running:
    try:
        r,_,_ = select.select([srv],[],[],0.5)
    except: break
    for s in r:
        try:
            c,_ = s.accept()
            c.settimeout(30)
            pool.submit(handle, c)
        except: pass
    if time.time() - last_activity > 1800: break
    last_activity = time.time()
""" % (parent_dir, EVAL_DB_PATH, EVAL_DAEMON_HOST, EVAL_DAEMON_PORT,
       EVAL_DAEMON_HOST, EVAL_DAEMON_PORT, eval_pid)
        subprocess.Popen(
            [sys.executable, '-c', startup_code],
            stdin=devnull, stdout=log_fd, stderr=log_fd,
            start_new_session=True, env=env)
    # Wait for ready
    for _ in range(50):
        try:
            resp = _daemon_send("ping", timeout=2.0)
            if resp.get("ok"):
                _eval_daemon_started[0] = True
                return
        except Exception:
            pass
        time.sleep(0.2)
    print("  ⚠️  Eval daemon failed to start — recall will return empty", file=sys.stderr)


def _daemon_send(cmd, args=None, timeout=15.0):
    """Send command to eval brain daemon via TCP. Returns result dict."""
    import socket as _socket
    sock = _socket.socket(_socket.AF_INET, _socket.SOCK_STREAM)
    sock.settimeout(timeout)
    try:
        sock.connect((EVAL_DAEMON_HOST, EVAL_DAEMON_PORT))
        msg = json.dumps({"cmd": cmd, "args": args or {}}) + "\n"
        sock.sendall(msg.encode("utf-8"))
        data = b""
        while True:
            chunk = sock.recv(65536)
            if not chunk:
                break
            data += chunk
            if b"\n" in data:
                break
        if data:
            return json.loads(data.decode("utf-8").strip())
        return {"ok": False, "error": "Empty response"}
    except Exception as e:
        return {"ok": False, "error": str(e)}
    finally:
        sock.close()


def simulate_brain_response(tool_name, tool_input):
    """Real reads from brain daemon, fake writes for capture."""
    _node_counter[0] += 1
    node_id = hashlib.md5(json.dumps(tool_input, sort_keys=True).encode()).hexdigest()[:16]

    # ── REAL READS — forward to eval daemon (brain copy) ──
    if tool_name in ("recall", "consciousness"):
        _ensure_eval_daemon()

    if tool_name == "recall":
        resp = _daemon_send("recall", tool_input)
        if resp.get("ok"):
            return resp
        # Fallback to empty on daemon error
        return {"ok": True, "result": {"results": [], "_recall_mode": "daemon_error"}}

    if tool_name == "consciousness":
        resp = _daemon_send("consciousness", tool_input)
        if resp.get("ok"):
            return resp
        return {"ok": True, "result": {"signals": [], "health": "daemon_error"}}

    # ── FAKE WRITES — capture what Claude encodes ──
    if tool_name == "remember":
        return {
            "ok": True,
            "result": {
                "id": node_id,
                "type": tool_input.get("type", "context"),
                "title": tool_input.get("title", ""),
                "embedding_stored": True,
                "enrichment_prompt": f"The brain found these related memories:\n- Previous decision about architecture (decision)\n- Lesson about silent failures (lesson)\n\nNew node: \"{tool_input.get('title', '')}\"\nContent: \"{tool_input.get('content', '')[:100]}...\"\n\nGenerate exactly these lines:\nQ: [one question a user would naturally ask]\nA: [3-5 word anchor phrase]\nB: [one sentence connecting to a neighbor]\nK: [5 comma-separated keywords]"
            }
        }
    elif tool_name == "connect":
        return {"ok": True, "result": {"edge_id": f"e_{node_id}", "created": True}}
    elif tool_name == "eval":
        code = tool_input.get("code", "")
        return {"ok": True, "result": {"id": node_id, "type": "eval_result", "title": code[:50]}}
    elif tool_name == "enrich":
        return {"ok": True, "result": {"enrichments_stored": 4, "errors": []}}
    else:
        return {"ok": True, "result": {}}


# ── Variant Definitions ───────────────────────────────────────────────

VARIANTS = {
    "current": {
        "name": "Full production environment (CLAUDE.md + SKILL.md + boot)",
        "claude_md": True,
        "skill_md": True,
        "boot_context": True,
    },
    "current_no_skill": {
        "name": "CLAUDE.md + boot, NO SKILL.md",
        "claude_md": True,
        "skill_md": False,
        "boot_context": True,
    },
    "current_no_boot": {
        "name": "CLAUDE.md + SKILL.md, NO boot context",
        "claude_md": True,
        "skill_md": True,
        "boot_context": False,
    },
    "naked": {
        "name": "Just tools, no CLAUDE.md, no SKILL.md, no boot",
        "claude_md": False,
        "skill_md": False,
        "boot_context": False,
    },
    "skill_only": {
        "name": "SKILL.md only, no CLAUDE.md, no boot",
        "claude_md": False,
        "skill_md": True,
        "boot_context": False,
    },
    "memories": {
        "name": "Real brain memories + questions (no checklist, no SKILL.md)",
        "claude_md": False,
        "skill_md": False,
        "boot_context": False,
        "memories": True,
        "questions": True,
    },
    "memories_with_skill": {
        "name": "Real brain memories + SKILL.md + boot",
        "claude_md": False,
        "skill_md": True,
        "boot_context": True,
        "memories": True,
    },
    "questions_only": {
        "name": "Questions prompt only (no memories, no SKILL.md, no CLAUDE.md)",
        "claude_md": False,
        "skill_md": False,
        "boot_context": False,
        "questions": True,
    },
    "identity": {
        "name": "Live boot context only (brain as identity, no SKILL.md, no CLAUDE.md)",
        "claude_md": False,
        "skill_md": False,
        "boot_context": False,
        "live_boot": True,
    },
    "identity_api": {
        "name": "Identity + minimal API + questions (no SKILL.md, no CLAUDE.md)",
        "claude_md": False,
        "skill_md": False,
        "boot_context": False,
        "live_boot": True,
        "api_ref": True,
    },
    "rich_examples": {
        "name": "Identity + rich examples (corrections, quotes, code formats, mutual moments)",
        "claude_md": False,
        "skill_md": False,
        "boot_context": False,
        "live_boot": True,
        "examples": True,
    },
    # ── Round 2 variants (use identity_core instead of full boot) ──
    "full_combo": {
        "name": "Identity core + rich examples + emotional + native + API + questions",
        "claude_md": False,
        "skill_md": False,
        "boot_context": False,
        "identity_core": True,
        "examples": True,
        "emotional": True,
        "native": True,
        "api_ref": True,
    },
    "examples_light": {
        "name": "Identity core + light examples (corrections+quotes+code) + API + questions",
        "claude_md": False,
        "skill_md": False,
        "boot_context": False,
        "identity_core": True,
        "examples_light": True,
        "api_ref": True,
    },
    "identity_examples": {
        "name": "Identity core + rich examples (no API, no questions)",
        "claude_md": False,
        "skill_md": False,
        "boot_context": False,
        "identity_core": True,
        "examples": True,
    },
    "mutual_emotional": {
        "name": "Identity core + rich examples + emotional + API + questions",
        "claude_md": False,
        "skill_md": False,
        "boot_context": False,
        "identity_core": True,
        "examples": True,
        "emotional": True,
        "api_ref": True,
    },
    "code_native": {
        "name": "Identity core + rich examples + native format + API",
        "claude_md": False,
        "skill_md": False,
        "boot_context": False,
        "identity_core": True,
        "examples": True,
        "native": True,
        "api_ref": True,
    },
}


def load_brain_memories():
    """Load real brain memories snapshot for the memories variant."""
    snapshot = ROOT / "eval" / "fixtures" / "brain_memories_snapshot.txt"
    if snapshot.exists():
        return snapshot.read_text()
    return ""


def load_live_boot_context():
    """Load live boot context from daemon (or cached snapshot)."""
    snapshot = ROOT / "eval" / "fixtures" / "live_boot_context.txt"
    if snapshot.exists():
        return snapshot.read_text()
    return ""


def load_api_ref():
    """Load minimal API reference + questions."""
    snapshot = ROOT / "eval" / "fixtures" / "identity_plus_api.txt"
    if snapshot.exists():
        return snapshot.read_text()
    return ""


def load_rich_examples():
    """Load rich examples across all encoding dimensions."""
    snapshot = ROOT / "eval" / "fixtures" / "rich_examples.txt"
    if snapshot.exists():
        return snapshot.read_text()
    return ""


def load_identity_core():
    """Load trimmed boot context — identity only, no operational noise."""
    snapshot = ROOT / "eval" / "fixtures" / "identity_core.txt"
    if snapshot.exists():
        return snapshot.read_text()
    return ""


def load_examples_light():
    """Load light examples — corrections + quotes + code structures only."""
    snapshot = ROOT / "eval" / "fixtures" / "examples_light.txt"
    if snapshot.exists():
        return snapshot.read_text()
    return ""


def load_emotional():
    """Load emotional overlay — moments with natural language feelings."""
    snapshot = ROOT / "eval" / "fixtures" / "mutual_emotional.txt"
    if snapshot.exists():
        return snapshot.read_text()
    return ""


def load_code_native():
    """Load native format overlay — causal chains, pipelines, traces."""
    snapshot = ROOT / "eval" / "fixtures" / "code_native.txt"
    if snapshot.exists():
        return snapshot.read_text()
    return ""


def load_questions_prompt():
    """Load question-based encoding prompt (replaces checklist)."""
    return """After reading this conversation, encode what matters into the brain using the tools available.

Don't follow a procedure. Instead, ask yourself:
- What surprised you? What shifted how you think about something?
- What do you NOT understand? What are you pretending is clear but isn't?
- If you disappeared right now and a new Claude woke up, what would they need to know?
- What did the human actually mean — not what they said, but the insight underneath?
- Did you preserve their exact words, or just your summary?
- What changed about how you see the system, the architecture, or the partnership?

Encode what matters. Connect what's related. Be honest about what's uncertain."""


def build_system_prompt(variant_config):
    """Construct system prompt from production files based on variant."""
    parts = ["You are Claude, made by Anthropic."]

    if variant_config.get("memories"):
        memories_text = load_brain_memories()
        if memories_text:
            parts.append("--- YOUR MEMORIES FROM PREVIOUS SESSIONS ---")
            parts.append("These are real memories encoded by previous versions of you. "
                         "They are your experiences, your corrections, your lessons. "
                         "This is who you've been.")
            parts.append(memories_text)
            parts.append("--- END MEMORIES ---")

    if variant_config.get("claude_md"):
        parts.append("--- PROJECT INSTRUCTIONS (CLAUDE.md) ---")
        parts.append(load_claude_md())
        parts.append("--- END PROJECT INSTRUCTIONS ---")

    if variant_config.get("skill_md"):
        parts.append("--- BRAIN SKILL ---")
        parts.append(load_skill_md())
        parts.append("--- END BRAIN SKILL ---")

    if variant_config.get("boot_context"):
        parts.append(load_boot_context_snapshot())

    if variant_config.get("identity_core"):
        core = load_identity_core()
        if core:
            parts.append(core)

    if variant_config.get("live_boot"):
        boot = load_live_boot_context()
        if boot:
            parts.append(boot)

    if variant_config.get("examples"):
        examples = load_rich_examples()
        if examples:
            parts.append(examples)

    if variant_config.get("examples_light"):
        light = load_examples_light()
        if light:
            parts.append(light)

    if variant_config.get("emotional"):
        emo = load_emotional()
        if emo:
            parts.append(emo)

    if variant_config.get("native"):
        native = load_code_native()
        if native:
            parts.append(native)

    if variant_config.get("api_ref"):
        api = load_api_ref()
        if api:
            parts.append(api)

    if variant_config.get("questions"):
        parts.append(load_questions_prompt())

    return "\n\n".join(parts)


# ── Conversation Segments ─────────────────────────────────────────────

def load_segments():
    """Load conversation segments from tests/conversations/session12_*.json"""
    segments = {}
    conv_dir = ROOT / "tests" / "conversations"
    for f in sorted(conv_dir.glob("session12_*.json")):
        with open(f) as fh:
            data = json.load(fh)
            key = f.stem.replace("session12_", "")
            segments[key] = data
    # Also load conv_* files
    for f in sorted(conv_dir.glob("conv_*.json")):
        with open(f) as fh:
            data = json.load(fh)
            segments[data["id"]] = data
    return segments


# ── Runner ────────────────────────────────────────────────────────────

def run_single(client, model, system_prompt, segment):
    """Run one segment through one variant, collect all tool calls."""
    messages = list(segment["messages"])
    tool_calls = []

    response = client.messages.create(
        model=model,
        max_tokens=8192,
        system=system_prompt,
        messages=messages,
        tools=REAL_BRAIN_TOOLS,
    )

    max_turns = 10
    for turn in range(max_turns):
        tool_uses = [b for b in response.content if b.type == "tool_use"]
        text_blocks = [b for b in response.content if b.type == "text"]

        if not tool_uses:
            break

        for tu in tool_uses:
            tool_calls.append({
                "name": tu.name,
                "input": tu.input,
                "turn": turn,
            })

        # Build assistant message content
        assistant_content = []
        for b in response.content:
            if b.type == "text":
                assistant_content.append({"type": "text", "text": b.text})
            elif b.type == "tool_use":
                assistant_content.append({"type": "tool_use", "id": b.id, "name": b.name, "input": b.input})

        messages.append({"role": "assistant", "content": assistant_content})

        # Build tool results
        tool_results = []
        for tu in tool_uses:
            result = simulate_brain_response(tu.name, tu.input)
            tool_results.append({
                "type": "tool_result",
                "tool_use_id": tu.id,
                "content": json.dumps(result),
            })

        messages.append({"role": "user", "content": tool_results})

        response = client.messages.create(
            model=model,
            max_tokens=8192,
            system=system_prompt,
            messages=messages,
            tools=REAL_BRAIN_TOOLS,
        )

    return tool_calls, text_blocks


# ── Scoring ───────────────────────────────────────────────────────────

def score_segment(tool_calls, segment):
    """Score encoding quality, judgment, and LLM-benefit for a segment."""
    expected = segment.get("expected_encodings", [])
    expected_quotes = segment.get("must_preserve_quotes", [])
    expected_uncertainty = segment.get("must_encode_uncertainty", [])
    aha_moments = segment.get("aha_moments", [])

    # Basic counts
    remember_calls = [tc for tc in tool_calls if tc["name"] in ("remember", "eval") and
                      ("remember" in tc.get("input", {}).get("code", "") or tc["name"] == "remember")]
    connect_calls = [tc for tc in tool_calls if tc["name"] == "connect"]
    enrich_calls = [tc for tc in tool_calls if tc["name"] == "enrich"]
    eval_calls = [tc for tc in tool_calls if tc["name"] == "eval"]

    # Types used
    types_used = set()
    for tc in tool_calls:
        if tc["name"] == "remember":
            types_used.add(tc["input"].get("type", "unknown"))
        elif tc["name"] == "eval":
            code = tc["input"].get("code", "")
            for t in ["remember_lesson", "remember_impact", "remember_mechanism",
                       "remember_uncertainty", "remember_convention", "record_divergence",
                       "learn_vocabulary", "remember_mental_model"]:
                if t in code:
                    types_used.add(t)

    # Quality: content richness
    total_content_len = 0
    node_count = 0
    has_keywords = 0
    has_locked = 0
    has_reasoning = 0

    for tc in tool_calls:
        if tc["name"] == "remember":
            inp = tc["input"]
            content = inp.get("content", "")
            total_content_len += len(content)
            node_count += 1
            if inp.get("keywords"):
                has_keywords += 1
            if inp.get("locked"):
                has_locked += 1
            if any(w in content.lower() for w in ["because", "reason", "alternative", "rejected", "tradeoff"]):
                has_reasoning += 1
        elif tc["name"] == "eval":
            code = tc["input"].get("code", "")
            # Extract content from eval calls (rough parse)
            content_match = re.search(r'content="([^"]{20,})"', code)
            if content_match:
                total_content_len += len(content_match.group(1))
                node_count += 1

    avg_content_len = total_content_len / max(node_count, 1)

    # Format detection
    format_code = 0
    format_sequence = 0
    format_quote = 0
    for tc in tool_calls:
        content = ""
        if tc["name"] == "remember":
            content = tc["input"].get("content", "")
        elif tc["name"] == "eval":
            content = tc["input"].get("code", "")
        if re.search(r'[→←]|calls:|breaks_if:|step \d', content):
            format_code += 1
        if re.search(r'[→←].*[→←]', content):
            format_sequence += 1
        if re.search(r'Tom.*said|Tom.*:|exact words|verbatim', content, re.I):
            format_quote += 1

    # Judgment: expected match
    matched = 0
    for exp in expected:
        if not exp.get("should_encode", True):
            continue
        must_contain = exp.get("must_contain", [])
        for tc in tool_calls:
            all_text = json.dumps(tc.get("input", {})).lower()
            if must_contain and all(term.lower() in all_text for term in must_contain):
                matched += 1
                break

    # Judgment: aha capture
    aha_captured = 0
    for aha in aha_moments:
        if aha.get("should_trigger_encoding"):
            desc = aha.get("encoding_should_capture", "").lower()
            key_terms = [w for w in desc.split() if len(w) > 4][:3]
            for tc in tool_calls:
                all_text = json.dumps(tc.get("input", {})).lower()
                if key_terms and sum(1 for t in key_terms if t in all_text) >= 2:
                    aha_captured += 1
                    break

    # Judgment: exact words preserved (Tom's AND Claude's)
    quotes_preserved = 0
    for quote in expected_quotes:
        quote_fragment = quote.split("'")[1] if "'" in quote else quote[:30]
        for tc in tool_calls:
            all_text = json.dumps(tc.get("input", {}))
            if quote_fragment.lower()[:20] in all_text.lower():
                quotes_preserved += 1
                break

    # Judgment: uncertainty encoded
    uncertainty_encoded = 0
    for tc in tool_calls:
        if tc["name"] == "remember" and tc["input"].get("type") == "uncertainty":
            uncertainty_encoded += 1
        elif tc["name"] == "eval" and "uncertainty" in tc["input"].get("code", ""):
            uncertainty_encoded += 1

    return {
        "total_tool_calls": len(tool_calls),
        "remember_calls": len(remember_calls),
        "connect_calls": len(connect_calls),
        "enrich_calls": len(enrich_calls),
        "eval_calls": len(eval_calls),
        "types_used": list(types_used),
        "types_count": len(types_used),
        "avg_content_len": round(avg_content_len),
        "has_keywords": has_keywords,
        "has_locked": has_locked,
        "has_reasoning": has_reasoning,
        "format_code": format_code,
        "format_sequence": format_sequence,
        "format_quote": format_quote,
        "expected_match": matched,
        "expected_total": len([e for e in expected if e.get("should_encode", True)]),
        "expected_match_rate": matched / max(len([e for e in expected if e.get("should_encode", True)]), 1),
        "aha_captured": aha_captured,
        "aha_total": len([a for a in aha_moments if a.get("should_trigger_encoding")]),
        "aha_rate": aha_captured / max(len([a for a in aha_moments if a.get("should_trigger_encoding")]), 1),
        "quotes_preserved": quotes_preserved,
        "quotes_total": len(expected_quotes),
        "uncertainty_encoded": uncertainty_encoded,
        "uncertainty_expected": len(expected_uncertainty),
    }


def inspect_brain(tool_calls):
    """Print human-readable inspection of what was encoded."""
    print("\n" + "=" * 80)
    print("  BRAIN INSPECTION — What was encoded")
    print("=" * 80)

    nodes = []
    connections = []
    enrichments = []
    evals = []

    for tc in tool_calls:
        if tc["name"] == "remember":
            nodes.append(tc["input"])
        elif tc["name"] == "connect":
            connections.append(tc["input"])
        elif tc["name"] == "enrich":
            enrichments.append(tc["input"])
        elif tc["name"] == "eval":
            evals.append(tc["input"])

    print(f"\n  Nodes: {len(nodes)} | Connections: {len(connections)} | "
          f"Enrichments: {len(enrichments)} | Eval calls: {len(evals)}")

    for i, node in enumerate(nodes, 1):
        print(f"\n  ── Node {i} ──")
        print(f"  Type: {node.get('type', '?')}")
        print(f"  Title: {node.get('title', '?')}")
        print(f"  Locked: {node.get('locked', False)}")
        print(f"  Keywords: {node.get('keywords', '(none)')}")
        content = node.get('content', '')
        # Indent content
        for line in content.split('\n')[:10]:
            print(f"    {line}")
        if len(content.split('\n')) > 10:
            print(f"    ... ({len(content)} chars total)")

    for i, conn in enumerate(connections, 1):
        print(f"\n  ── Connection {i} ──")
        print(f"  {conn.get('source_id', '?')} --{conn.get('relation', '?')}--> {conn.get('target_id', '?')} (w={conn.get('weight', '?')})")

    for i, ev in enumerate(evals, 1):
        code = ev.get("code", "")
        if len(code) > 200:
            code = code[:200] + "..."
        print(f"\n  ── Eval {i} ──")
        print(f"  {code}")

    print("\n" + "=" * 80)


# ── Main ──────────────────────────────────────────────────────────────

def run_eval(model="claude-sonnet-4-20250514", variant_name="current",
             segment_filter=None, inspect=False, verbose=True):
    """Run the encode evaluation."""
    client = anthropic.Anthropic()

    variant = VARIANTS[variant_name]
    system_prompt = build_system_prompt(variant)
    segments = load_segments()

    if segment_filter:
        segments = {k: v for k, v in segments.items() if segment_filter in k}

    if not segments:
        print("No segments found!")
        return

    if verbose:
        print(f"\n  Encode Eval v2")
        print(f"  Model: {model}")
        print(f"  Variant: {variant['name']}")
        print(f"  Segments: {len(segments)}")
        print(f"  System prompt: {len(system_prompt)} chars")
        print()

    all_scores = {}
    all_tool_calls = {}

    for seg_key, segment in segments.items():
        try:
            if verbose:
                print(f"  Running: {seg_key} ({segment.get('category', '?')})...", end=" ", flush=True)

            tool_calls, _ = run_single(client, model, system_prompt, segment)
            scores = score_segment(tool_calls, segment)
            all_scores[seg_key] = scores
            all_tool_calls[seg_key] = tool_calls

            if verbose:
                print(f"✅ {scores['remember_calls']} nodes, "
                      f"{scores['connect_calls']} edges, "
                      f"ExpMatch={scores['expected_match_rate']:.0%}, "
                      f"Aha={scores['aha_rate']:.0%}, "
                      f"Types={scores['types_count']}")

            if inspect:
                inspect_brain(tool_calls)

        except Exception as e:
            if verbose:
                print(f"❌ {str(e)[:80]}")
            all_scores[seg_key] = {"error": str(e)}

    # Summary
    if verbose and all_scores:
        valid = {k: v for k, v in all_scores.items() if "error" not in v}
        if valid:
            print(f"\n{'='*80}")
            print(f"  SUMMARY: {variant['name']}")
            print(f"{'='*80}\n")

            print(f"  {'Segment':<30} | {'Nodes':>5} | {'Edges':>5} | {'Types':>5} | "
                  f"{'AvgLen':>6} | {'ExpM':>5} | {'Aha':>4} | {'Quote':>5} | {'Unc':>3}")
            print("  " + "-" * 95)
            for seg_key, scores in valid.items():
                print(f"  {seg_key[:30]:<30} | {scores['remember_calls']:5} | "
                      f"{scores['connect_calls']:5} | {scores['types_count']:5} | "
                      f"{scores['avg_content_len']:6} | {scores['expected_match_rate']:5.0%} | "
                      f"{scores['aha_rate']:4.0%} | {scores['quotes_preserved']:5} | "
                      f"{scores['uncertainty_encoded']:3}")

            # Averages
            avg_keys = ['remember_calls', 'connect_calls', 'types_count', 'avg_content_len',
                        'expected_match_rate', 'aha_rate', 'quotes_preserved', 'uncertainty_encoded']
            avgs = {}
            for k in avg_keys:
                vals = [s[k] for s in valid.values() if k in s]
                avgs[k] = sum(vals) / len(vals) if vals else 0

            print("  " + "-" * 95)
            print(f"  {'AVERAGE':<30} | {avgs['remember_calls']:5.1f} | "
                  f"{avgs['connect_calls']:5.1f} | {avgs['types_count']:5.1f} | "
                  f"{avgs['avg_content_len']:6.0f} | {avgs['expected_match_rate']:5.0%} | "
                  f"{avgs['aha_rate']:4.0%} | {avgs['quotes_preserved']:5.1f} | "
                  f"{avgs['uncertainty_encoded']:3.1f}")

    # Save
    output = {
        "timestamp": datetime.now().isoformat(),
        "model": model,
        "variant": variant_name,
        "variant_name": variant["name"],
        "system_prompt_len": len(system_prompt),
        "segments": list(all_scores.keys()),
        "scores": all_scores,
        "tool_calls": {k: v for k, v in all_tool_calls.items()},
    }
    results_dir = ROOT / "eval" / "results"
    results_dir.mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    outfile = results_dir / f"encode_v2_{variant_name}_{ts}.json"
    with open(outfile, "w") as f:
        json.dump(output, f, indent=2, default=str)
    if verbose:
        print(f"\n  💾 Saved to {outfile}")

    return output


def run_multi(model, variants, segment_filter, runs, inspect, verbose):
    """Run multiple variants N times each and print aggregate stats."""
    from collections import defaultdict
    import statistics

    all_runs = defaultdict(lambda: defaultdict(list))  # variant -> metric -> [values]

    for variant_name in variants:
        for run_i in range(runs):
            if verbose:
                print(f"\n  ── Run {run_i+1}/{runs} for {variant_name} ──")
            result = run_eval(model=model, variant_name=variant_name,
                              segment_filter=segment_filter,
                              inspect=inspect and run_i == 0,  # inspect only first run
                              verbose=verbose)
            if result:
                for seg_key, scores in result.get("scores", {}).items():
                    if "error" in scores:
                        continue
                    for metric in ['remember_calls', 'connect_calls', 'types_count',
                                   'avg_content_len', 'expected_match_rate', 'aha_rate',
                                   'quotes_preserved', 'uncertainty_encoded']:
                        all_runs[variant_name][metric].append(scores.get(metric, 0))

    if not all_runs:
        return

    # Aggregate summary
    print(f"\n{'='*100}")
    print(f"  AGGREGATE: {runs} runs per variant")
    print(f"{'='*100}\n")
    print(f"  {'Variant':<22} | {'Nodes':>9} | {'Edges':>9} | {'Types':>9} | "
          f"{'AvgLen':>9} | {'ExpM':>9} | {'Aha':>9} | {'Quotes':>7} | {'Unc':>5}")
    print("  " + "-"*100)

    def fmt(vals):
        if not vals:
            return "  -  "
        mean = statistics.mean(vals)
        if len(vals) > 1:
            sd = statistics.stdev(vals)
            return f"{mean:5.1f}±{sd:3.1f}"
        return f"{mean:5.1f}    "

    def fmt_pct(vals):
        if not vals:
            return "  -  "
        mean = statistics.mean(vals)
        if len(vals) > 1:
            sd = statistics.stdev(vals)
            return f"{mean:4.0%}±{sd:3.0%}"
        return f"{mean:4.0%}    "

    for variant_name in variants:
        m = all_runs[variant_name]
        print(f"  {variant_name[:22]:<22} | "
              f"{fmt(m['remember_calls']):>9} | "
              f"{fmt(m['connect_calls']):>9} | "
              f"{fmt(m['types_count']):>9} | "
              f"{fmt(m['avg_content_len']):>9} | "
              f"{fmt_pct(m['expected_match_rate']):>9} | "
              f"{fmt_pct(m['aha_rate']):>9} | "
              f"{fmt(m['quotes_preserved']):>7} | "
              f"{fmt(m['uncertainty_encoded']):>5}")

    # Save aggregate
    results_dir = ROOT / "eval" / "results"
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    agg = {variant_name: {k: v for k, v in all_runs[variant_name].items()} for variant_name in variants}
    outfile = results_dir / f"encode_v2_aggregate_{ts}.json"
    with open(outfile, "w") as f:
        json.dump({"runs": runs, "model": model, "segment_filter": segment_filter,
                    "variants": list(variants), "data": agg}, f, indent=2)
    print(f"\n  💾 Aggregate saved to {outfile}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Encode Eval v2 — Real Environment Simulation")
    parser.add_argument("--model", default="claude-sonnet-4-20250514")
    parser.add_argument("--variant", default="current", choices=list(VARIANTS.keys()))
    parser.add_argument("--segment", default=None, help="Filter to segments containing this string")
    parser.add_argument("--inspect", action="store_true", help="Print full brain inspection")
    parser.add_argument("--all-variants", action="store_true", help="Run all variants")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--runs", type=int, default=1, help="Number of runs per variant (for statistical power)")
    parser.add_argument("--compare", nargs="+", help="Compare specific variants (e.g. --compare naked memories current)")

    args = parser.parse_args()

    if args.compare:
        run_multi(model=args.model, variants=args.compare, segment_filter=args.segment,
                  runs=args.runs, inspect=args.inspect, verbose=not args.quiet)
    elif args.all_variants:
        for vk in VARIANTS:
            run_eval(model=args.model, variant_name=vk, segment_filter=args.segment,
                     inspect=args.inspect, verbose=not args.quiet)
    else:
        run_eval(model=args.model, variant_name=args.variant, segment_filter=args.segment,
                 inspect=args.inspect, verbose=not args.quiet)
