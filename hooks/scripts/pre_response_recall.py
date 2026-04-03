"""Pre-response recall — surfaces brain context before Claude responds.
Fires on UserPromptSubmit. Output: JSON {"additionalContext":"..."}.

Flow:
1. Command hook calls daemon → Layer 1 retrieval, writes candidates file
2. Reads candidates + recently recalled nodes + session context
3. Calls Haiku as Layer 2 judge → selects relevant nodes with reasoning
4. Formats structured output as additionalContext
"""
import sys, os, json, time

# Load .env from project root (API keys)
_project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_env_path = os.path.join(_project_root, ".env")
if os.path.exists(_env_path):
    with open(_env_path) as _f:
        for _line in _f:
            _line = _line.strip()
            if _line and not _line.startswith("#") and "=" in _line:
                _k, _v = _line.split("=", 1)
                _key, _val = _k.strip(), _v.strip()
                if _val:  # Only set if value is non-empty
                    os.environ[_key] = _val

sys.path.insert(0, os.path.dirname(__file__))
from hook_common import (get_hook_input, daemon_available, daemon_call_raw,
                         daemon_unavailable_error, brain_debug, log_hook_output)

APPROVE = json.dumps({"decision": "approve"})

hook_input = get_hook_input()
user_message = hook_input.get("prompt", "") or hook_input.get("message", "")

# Skip short, slash, or bang messages
if not user_message or len(user_message) < 5 or user_message.startswith("/") or user_message.startswith("!"):
    brain_debug("recall: skipped (short/slash/bang)")
    print(APPROVE)
    sys.exit(0)

t0 = time.time()
try:
    if not daemon_available():
        err = daemon_unavailable_error("recall")
        log_hook_output("recall", output_text="(daemon unavailable)", user_prompt=user_message)
        print(json.dumps({"additionalContext": err}))
        sys.exit(0)

    # Step 1: Get raw candidates from daemon
    resp = daemon_call_raw("hook_recall", {
        "prompt": hook_input.get("prompt", ""),
        "message": hook_input.get("message", ""),
    }, timeout=12.0)

    if not resp.get("ok"):
        err_msg = resp.get("error", "unknown error")
        log_hook_output("recall", output_text="(daemon error: %s)" % err_msg, user_prompt=user_message)
        print(json.dumps({"additionalContext":
            "[BRAIN]\n⚠️ RECALL FAILED: %s\nThe brain could not search for relevant memories. Operating without context.\n[/BRAIN]" % err_msg}))
        sys.exit(0)

    # Extract candidates from the file the daemon wrote
    result = resp.get("result", {})
    session_id = hook_input.get("session_id", "unknown")

    # Read candidates file written by daemon
    # The daemon returns its session_id in the response so we find the right file
    daemon_session = result.get("session_id", "")
    if not daemon_session:
        # Fallback: try Claude Code's session_id
        daemon_session = session_id
    candidates_path = "/tmp/brain-%s-recall-candidates.json" % daemon_session
    candidates_data = None
    if os.path.exists(candidates_path) and os.path.getsize(candidates_path) > 0:
        try:
            with open(candidates_path) as f:
                candidates_data = json.load(f)
        except Exception as e:
            print('[brain] ERROR recall_candidates_parse: %s' % e, file=sys.stderr)

    if not candidates_data or not candidates_data.get("candidates"):
        # No candidates — nothing to distill
        log_hook_output("recall", output_text="(no candidates)", user_prompt=user_message)
        print(APPROVE)
        sys.exit(0)

    candidates = candidates_data["candidates"]
    latency_fetch = (time.time() - t0) * 1000

    # Log immediately — dashboard sees the recall attempt even if judge fails/times out
    log_hook_output("recall", output_text="(judging %d candidates...)" % len(candidates),
                   user_prompt=user_message)

    # Step 2: Layer 2 — Haiku judges which candidates are relevant
    t1 = time.time()
    try:
        import anthropic
        client = anthropic.Anthropic()

        sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
            os.path.dirname(os.path.abspath(__file__)))), 'servers'))
        from pipeline_contract import build_judge_prompt, JUDGE

        # Gather recently recalled node IDs + titles from message_stream
        # Purpose: tells the judge what was already surfaced so it can deprioritize repeats
        recently_recalled = []
        try:
            import sqlite3
            _brain_db_dir = os.environ.get("BRAIN_DB_DIR", "")
            if _brain_db_dir:
                _logs_db = os.path.join(_brain_db_dir, "brain_logs.db")
                if os.path.exists(_logs_db):
                    _lconn = sqlite3.connect(_logs_db, timeout=2)
                    _lookback = JUDGE.get('recent_recalls_messages', 10)
                    _rows = _lconn.execute(
                        "SELECT recalled_node_ids FROM message_stream "
                        "WHERE recalled_node_ids IS NOT NULL AND role='user' "
                        "ORDER BY id DESC LIMIT ?", (_lookback,)).fetchall()
                    _seen_ids = set()
                    for _r in _rows:
                        for _nid in json.loads(_r[0]):
                            _seen_ids.add(_nid)
                    # Get titles from brain.db
                    if _seen_ids:
                        _brain_db = os.path.join(_brain_db_dir, "brain.db")
                        _bconn = sqlite3.connect(_brain_db, timeout=2)
                        for _nid in list(_seen_ids)[:20]:
                            _trow = _bconn.execute(
                                "SELECT title FROM nodes WHERE id = ?", (_nid,)).fetchone()
                            if _trow:
                                recently_recalled.append({"id": _nid, "title": _trow[0]})
                        _bconn.close()
                    _lconn.close()
        except Exception as _e:
            brain_debug("recall: recently_recalled fetch failed: %s" % _e)

        # Build the judge prompt
        # build_judge_prompt assembles: session context + conversation + recently recalled
        # + formatted candidates with metadata → single prompt for Haiku to judge
        judge_prompt, max_tokens = build_judge_prompt(
            candidates, user_message,
            session_context=candidates_data.get("session_context", ""),
            recent_messages=candidates_data.get("recent_messages", []),
            recently_recalled=recently_recalled)

        # Call Haiku — same API pattern as distiller, different prompt
        api_resp = client.messages.create(
            model="claude-haiku-4-5",
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": judge_prompt}]
        )
        raw_response = api_resp.content[0].text.strip()
        latency_judge = (time.time() - t1) * 1000

        # Parse JSON from Haiku response
        # Haiku sometimes wraps JSON in markdown — strip it
        _json_str = raw_response
        if _json_str.startswith("```"):
            _json_str = _json_str.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
        try:
            judgment = json.loads(_json_str)
        except json.JSONDecodeError:
            # Try to extract JSON from mixed text
            _start = _json_str.find("{")
            _end = _json_str.rfind("}") + 1
            if _start >= 0 and _end > _start:
                judgment = json.loads(_json_str[_start:_end])
            else:
                raise ValueError("No JSON found in judge response: %s" % raw_response[:200])

        selected = judgment.get("selected", [])
        selected_ids = {s.get("id", "")[:8] for s in selected}

        # Write judge-selected IDs for Hebbian strengthening in Stop hook.
        # Only judge-selected nodes get co_accessed edges — meaningful co-activation.
        try:
            _judge_path = "/tmp/brain-%s-judge-selected.json" % daemon_session
            with open(_judge_path, 'w') as _jf:
                json.dump({"selected_ids": list(selected_ids)}, _jf)
        except Exception:
            pass

        # Store prompt details for dashboard debugging
        _prompt_details = json.dumps({
            "judge_prompt_length": len(judge_prompt),
            "model": "claude-haiku-4-5",
            "candidates_count": len(candidates),
            "selected_count": len(selected),
            "selected_ids": list(selected_ids),
            "latency_fetch_ms": round(latency_fetch),
            "latency_judge_ms": round(latency_judge),
        })

        if not selected:
            # Judge found nothing relevant — silent to Claude AND dashboard
            log_hook_output("recall", output_text="",
                           user_prompt=user_message, metadata=_prompt_details)
            brain_debug("recall: judge selected 0/%d candidates in %dms" % (
                len(candidates), latency_judge))
            print(APPROVE)
        else:
            # Layer 3: Graph expansion from judge-selected seeds.
            # Traverse structural edges to pull in neighborhood context.
            graph_neighbors = []
            try:
                expand_resp = daemon_call_raw("graph_expand", {
                    "node_ids": list(selected_ids),
                    "depth": 1,
                    "limit_per_seed": 3,
                }, timeout=5.0)
                if expand_resp.get("ok"):
                    graph_neighbors = expand_resp.get("result", {}).get("neighbors", [])
                    brain_debug("recall: Layer 3 expanded %d neighbors from %d seeds" % (
                        len(graph_neighbors), len(selected_ids)))
            except Exception as _ge:
                brain_debug("recall: Layer 3 graph expand failed: %s" % _ge)

            # Format structured output with selected nodes + graph neighbors
            from pipeline_contract import format_judge_output
            context = format_judge_output(selected, candidates, graph_neighbors)
            log_hook_output("recall", output_text=context, user_prompt=user_message,
                           metadata=_prompt_details)
            brain_debug("recall: judge selected %d/%d candidates + %d neighbors in %dms (fetch: %dms)" % (
                len(selected), len(candidates), len(graph_neighbors), latency_judge, latency_fetch))
            print(json.dumps({"additionalContext": context}))

    except ImportError:
        # anthropic SDK not installed — fall back to raw candidates summary
        brain_debug("recall: anthropic SDK not available, returning raw summary")
        summary = "Brain found %d candidates but judge unavailable." % len(candidates)
        for c in candidates[:3]:
            summary += "\n- [%s] %s (id:%s)" % (c.get("type", "?"), c.get("title", "?")[:60], c.get("id", "")[:12])
        log_hook_output("recall", output_text=summary, user_prompt=user_message)
        print(json.dumps({"additionalContext": summary}))

    except Exception as e:
        # Judge failed — log error, return raw summary as fallback
        brain_debug("recall: judge failed: %s" % e)
        log_hook_output("recall", output_text="(judge error: %s)" % e, user_prompt=user_message)
        summary = "Brain recall found %d candidates (judge failed: %s)" % (len(candidates), str(e)[:50])
        for c in candidates[:3]:
            summary += "\n- [%s] %s (id:%s)" % (c.get("type", "?"), c.get("title", "?")[:60], c.get("id", "")[:12])
        print(json.dumps({"additionalContext": summary}))

except Exception as e:
    log_hook_output("recall", output_text="(exception) %s" % e, user_prompt=user_message)
    print(APPROVE)
