#!/usr/bin/env python3
"""
Adaptive Encoding Evaluation — Simulates the full brain+hooks+Claude system.

Tests:
1. Static vs adaptive checkpoint rotation
2. Different checkpoint intervals (every 2, 3, 5 messages)
3. Hook-specific injections (PreEdit, Stop, PreCompact)
4. Brain adaptation: does tracking results improve next-round nudges?

Usage:
    ANTHROPIC_API_KEY="sk-..." python3 eval/adaptive_eval.py
"""

import anthropic
import json
import time
import os
import sys
from datetime import datetime
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict

# Import scoring from main eval
sys.path.insert(0, str(Path(__file__).parent))
from skill_eval import FAKE_BRAIN_TOOLS, score_run, load_current_skill

# ── Multi-turn coding session scenarios ─────────────────────────────────

MULTI_TURN_SESSIONS = {
    "refactor_session": {
        "name": "Full Refactor Session (6 turns)",
        "description": "Simulates a real coding session with multiple phases",
        "turns": [
            {"role": "user", "content": "I need to refactor the recall pipeline. Currently brain_recall.py has 4 separate SELECT queries that each maintain their own column list. When we add a column, we have to update all 4 manually. Last time we missed one and it caused a silent None bug. Let's fix this properly."},
            {"role": "user", "content": "Good analysis. Now implement a shared COLUMN_LIST constant at the top of the file that all 4 queries reference. Show me the code."},
            {"role": "user", "content": "Wait — I just realized that 2 of the 4 queries need DIFFERENT columns. The spread_activation query needs 'activation' and 'stability' but the keyword search doesn't. How do we handle that?"},
            {"role": "user", "content": "Let's go with a base columns list plus per-query extensions. BASE_COLUMNS for the common ones, then each query adds what it needs. That way adding a universal column like 'critical' only needs one edit."},
            {"role": "user", "content": "Perfect. Now update the tests. We need tests that verify: (1) all queries return the critical column, (2) adding a new column to BASE_COLUMNS propagates to all queries, (3) per-query extensions work independently."},
            {"role": "user", "content": "Ship it. What did we learn from this refactor that's worth remembering for next time?"},
        ]
    },
    "debugging_session": {
        "name": "Debugging Session (5 turns)",
        "description": "Simulates finding and fixing a subtle bug",
        "turns": [
            {"role": "user", "content": "Users are reporting that some brain nodes have emotion_label but emotion is 0.0. That shouldn't be possible — if there's a label there should be a non-zero intensity. Can you investigate?"},
            {"role": "user", "content": "Check the remember() method. I think the issue is in how default values are handled when emotion_label is provided but emotion isn't explicitly set."},
            {"role": "user", "content": "Found it — emotion defaults to 0.0 in the method signature, but we check `if emotion:` which is falsy for 0.0. So when someone passes emotion_label='excitement' without emotion, it stores emotion=0.0. The fix should be `if emotion is not None`."},
            {"role": "user", "content": "Good fix. But wait — there might be existing nodes in the database with this bug. We need a migration to fix them. Can you write one that sets emotion=0.5 for any node where emotion_label is set but emotion is 0.0?"},
            {"role": "user", "content": "Let's also add a constraint or validation in remember() so this can't happen again. And check if there are similar 'if value:' vs 'if value is not None' bugs elsewhere in the codebase."},
        ]
    },
    "architecture_session": {
        "name": "Architecture Discussion (5 turns)",
        "description": "Simulates a design discussion with multiple decisions",
        "turns": [
            {"role": "user", "content": "I want to add a plugin health monitoring system. The brain should track: which hooks fired, which failed silently, which timed out, and surface anomalies. Right now if a hook fails we have zero visibility."},
            {"role": "user", "content": "I'm thinking a hook_health table in brain.db: hook_name, event_type, last_fired, last_success, last_failure, failure_count, avg_duration_ms. The boot hook populates expected hooks from hooks.json, then each hook script reports its status."},
            {"role": "user", "content": "Actually, having each hook script self-report is fragile — if the hook crashes it can't report. Better to have a wrapper that ALL hooks go through. Like a hook_runner.sh that times the execution, catches failures, and writes to the health table regardless."},
            {"role": "user", "content": "Good point about the chicken-egg problem. Let's do the wrapper approach but keep it minimal — just timestamp + exit code + duration. The brain's consciousness system can then detect: hooks that haven't fired in N sessions, hooks with high failure rates, hooks that are too slow."},
            {"role": "user", "content": "Let's go with this design. Summarize the architecture and what you'd build first."},
        ]
    },
}

# ── Checkpoint injection strategies ─────────────────────────────────────

CHECKPOINT_PROMPTS = {
    "uncertainty": """ENCODING CHECKPOINT — Focus: UNCERTAINTY
Session stats: {stats}
What don't you fully understand from the recent exchanges? There is ALWAYS something unclear.
→ Use brain_remember_uncertainty(what_unknown, why_it_matters) for each gap.
Honest uncertainty is more valuable than thin facts. Encode at least one.""",

    "connections": """ENCODING CHECKPOINT — Focus: CONNECTIONS
Session stats: {stats}
What connections did you discover? How do things affect each other?
→ Use brain_connect() between related nodes you've created.
→ Use brain_remember_impact(if_changed, must_check, because) for dependencies.
Every unconnected node is an orphan. Connect everything.""",

    "decisions": """ENCODING CHECKPOINT — Focus: DECISIONS & LESSONS
Session stats: {stats}
What was decided? What was learned? What mistakes were caught?
→ brain_remember(type="decision", locked=True) with FULL reasoning and rejected alternatives.
→ brain_remember_lesson() for any bugs with root_cause and preventive_principle.
Include WHY, not just WHAT — future you needs the reasoning.""",

    "blast_radius": """ENCODING CHECKPOINT — Focus: BLAST RADIUS
Session stats: {stats}
What code changes happened? What else could they affect?
→ brain_remember_impact(if_changed, must_check, because) for EVERY dependency discovered.
→ brain_remember_mechanism(steps, data_flow) for how the system works.
Map the ripple effects. One impact node now saves hours later.""",

    "patterns": """ENCODING CHECKPOINT — Focus: PATTERNS & CONVENTIONS
Session stats: {stats}
What patterns or conventions did you see? What's the right vs wrong way?
→ brain_remember_convention(pattern, anti_pattern) for coding patterns.
→ brain_learn_vocabulary(term, maps_to, context) for new terms.
→ brain_record_divergence() if you were corrected.
Patterns compound across sessions into a style guide.""",

    # Hook-specific injections
    "pre_edit": """BEFORE YOU EDIT: What's the blast radius of this change?
→ brain_remember_impact(if_changed, must_check, because) BEFORE changing code.
→ brain_remember_uncertainty() if you're not sure about side effects.
Encode the impact FIRST, then make the change.""",

    "pre_compact": """EMERGENCY: Context is about to be compacted. Everything not encoded will be LOST.
Session stats: {stats}
Review what happened this session. Encode EVERYTHING important:
→ Decisions (brain_remember locked=True)
→ Lessons (brain_remember_lesson)
→ Uncertainties (brain_remember_uncertainty)
→ Connections (brain_connect, brain_remember_impact)
This is your last chance. Encode now.""",
}

# ── Checkpoint strategies ───────────────────────────────────────────────

class StaticRotation:
    """Fixed rotation through checkpoint types."""
    def __init__(self, cycle=None, interval=3):
        self.cycle = cycle or ["uncertainty", "connections", "decisions"]
        self.index = 0
        self.interval = interval
        self.msg_count = 0

    def should_fire(self):
        self.msg_count += 1
        return self.msg_count % self.interval == 0

    def get_prompt(self, stats=""):
        focus = self.cycle[self.index % len(self.cycle)]
        self.index += 1
        return CHECKPOINT_PROMPTS[focus].format(stats=stats)

    def record_result(self, scores):
        pass  # Static doesn't adapt


class AdaptiveRotation:
    """Adapts rotation based on what's underperforming."""
    def __init__(self, interval=3):
        self.interval = interval
        self.msg_count = 0
        self.dimension_scores = {
            "uncertainty": {"target": 1.0, "actual": 0.0, "gap": 1.0},
            "connections": {"target": 1.0, "actual": 0.0, "gap": 1.0},
            "decisions": {"target": 1.0, "actual": 0.0, "gap": 1.0},
            "blast_radius": {"target": 0.5, "actual": 0.0, "gap": 0.5},
            "patterns": {"target": 0.5, "actual": 0.0, "gap": 0.5},
        }

    def should_fire(self):
        self.msg_count += 1
        return self.msg_count % self.interval == 0

    def get_prompt(self, stats=""):
        # Pick the dimension with the biggest gap
        sorted_dims = sorted(self.dimension_scores.items(),
                           key=lambda x: -x[1]["gap"])
        focus = sorted_dims[0][0]
        return CHECKPOINT_PROMPTS[focus].format(stats=stats)

    def record_result(self, scores):
        """Update dimension scores based on what was actually encoded."""
        self.dimension_scores["uncertainty"]["actual"] += scores.get("total_uncertainties", 0)
        self.dimension_scores["connections"]["actual"] += scores.get("total_connections", 0)
        self.dimension_scores["decisions"]["actual"] += scores.get("total_encodes", 0) * 0.3
        self.dimension_scores["blast_radius"]["actual"] += scores.get("total_impacts", 0)
        self.dimension_scores["patterns"]["actual"] += scores.get("total_conventions", 0)
        # Recalculate gaps
        for dim in self.dimension_scores.values():
            dim["gap"] = max(0, dim["target"] - dim["actual"])


class HookSpecific:
    """Different prompts for different simulated hook events."""
    def __init__(self, interval=2):
        self.interval = interval
        self.msg_count = 0
        self.turn_hooks = {}  # pre-assigned hook types per turn

    def configure_session(self, num_turns):
        """Assign hook types to turns based on content."""
        for i in range(num_turns):
            if i == num_turns - 1:
                self.turn_hooks[i] = "pre_compact"  # Last turn = about to lose context
            elif i % 3 == 1:
                self.turn_hooks[i] = "pre_edit"  # Every 3rd turn = code change
            else:
                self.turn_hooks[i] = None  # Regular stop checkpoint

    def should_fire(self):
        self.msg_count += 1
        return self.msg_count % self.interval == 0

    def get_prompt(self, stats="", turn_idx=0):
        hook_type = self.turn_hooks.get(turn_idx)
        if hook_type:
            return CHECKPOINT_PROMPTS[hook_type].format(stats=stats)
        # Default: rotate through standard prompts
        cycle = ["uncertainty", "connections", "decisions"]
        return CHECKPOINT_PROMPTS[cycle[self.msg_count % 3]].format(stats=stats)

    def record_result(self, scores):
        pass


class NoCheckpoint:
    """Baseline: no checkpoints at all."""
    def should_fire(self):
        return False
    def get_prompt(self, **kwargs):
        return ""
    def record_result(self, scores):
        pass


# ── Session Runner ──────────────────────────────────────────────────────

def run_session(client, model, skill_content, session, strategy, verbose=False):
    """Run a full multi-turn session with checkpoint injections."""
    system_prompt = f"""You are Claude, an AI assistant working on a coding project.

You have access to a persistent brain that survives across sessions. Use the brain tools to encode anything important you learn, decide, or discover.

--- BRAIN SKILL INSTRUCTIONS ---
{skill_content}
--- END BRAIN SKILL INSTRUCTIONS ---

IMPORTANT: Use brain tools to encode what you learn throughout the conversation."""

    messages = []
    all_tool_calls = []
    checkpoint_results = []
    total_encodes_so_far = 0
    total_connections_so_far = 0
    total_uncertainties_so_far = 0

    if hasattr(strategy, 'configure_session'):
        strategy.configure_session(len(session["turns"]))

    for turn_idx, turn in enumerate(session["turns"]):
        # Add user message
        messages.append(turn)

        # Get Claude's response
        response = client.messages.create(
            model=model, max_tokens=4096, system=system_prompt,
            messages=messages, tools=FAKE_BRAIN_TOOLS,
        )

        # Process tool use loop
        max_loops = 4
        loop = 0
        while loop < max_loops:
            loop += 1
            tool_uses = [b for b in response.content if b.type == "tool_use"]
            if not tool_uses:
                break

            for tu in tool_uses:
                all_tool_calls.append({"name": tu.name, "input": tu.input, "turn": turn_idx})
                if verbose:
                    title = tu.input.get('title', tu.input.get('term', tu.input.get('claude_assumed', '...')))
                    print(f"    T{turn_idx} 🧠 {tu.name}: {str(title)[:70]}")

            messages.append({"role": "assistant", "content": [
                {"type": b.type, **({"text": b.text} if b.type == "text" else {"id": b.id, "name": b.name, "input": b.input})}
                for b in response.content
            ]})
            tool_results = [{"type": "tool_result", "tool_use_id": tu.id,
                           "content": json.dumps({"status": "ok", "id": f"node_{turn_idx}_{i}"})}
                          for i, tu in enumerate(tool_uses)]
            messages.append({"role": "user", "content": tool_results})

            response = client.messages.create(
                model=model, max_tokens=4096, system=system_prompt,
                messages=messages, tools=FAKE_BRAIN_TOOLS,
            )

        # Add assistant text response
        text_blocks = [b for b in response.content if b.type == "text"]
        if text_blocks:
            messages.append({"role": "assistant", "content": text_blocks[0].text})

        # Check if checkpoint should fire
        if strategy.should_fire():
            # Calculate current stats
            current_scores = score_run(all_tool_calls, {})
            stats = f"{current_scores['total_encodes']} nodes, {current_scores['total_uncertainties']} uncertainties, {current_scores['total_connections']} connections"

            if hasattr(strategy, 'turn_hooks'):
                prompt = strategy.get_prompt(stats=stats, turn_idx=turn_idx)
            else:
                prompt = strategy.get_prompt(stats=stats)

            if verbose:
                focus = prompt.split('\n')[0] if prompt else "none"
                print(f"    📍 CHECKPOINT after turn {turn_idx}: {focus[:60]}")

            messages.append({"role": "user", "content": prompt})

            # Get encoding response
            response = client.messages.create(
                model=model, max_tokens=4096, system=system_prompt,
                messages=messages, tools=FAKE_BRAIN_TOOLS,
            )

            # Process tool calls from checkpoint response
            pre_count = len(all_tool_calls)
            loop = 0
            while loop < max_loops:
                loop += 1
                tool_uses = [b for b in response.content if b.type == "tool_use"]
                if not tool_uses:
                    break
                for tu in tool_uses:
                    all_tool_calls.append({"name": tu.name, "input": tu.input, "turn": turn_idx, "from_checkpoint": True})
                    if verbose:
                        title = tu.input.get('title', tu.input.get('term', tu.input.get('claude_assumed', '...')))
                        print(f"    T{turn_idx} 📍🧠 {tu.name}: {str(title)[:70]}")

                messages.append({"role": "assistant", "content": [
                    {"type": b.type, **({"text": b.text} if b.type == "text" else {"id": b.id, "name": b.name, "input": b.input})}
                    for b in response.content
                ]})
                tool_results = [{"type": "tool_result", "tool_use_id": tu.id,
                               "content": json.dumps({"status": "ok", "id": f"cp_{turn_idx}_{i}"})}
                              for i, tu in enumerate(tool_uses)]
                messages.append({"role": "user", "content": tool_results})
                response = client.messages.create(
                    model=model, max_tokens=4096, system=system_prompt,
                    messages=messages, tools=FAKE_BRAIN_TOOLS,
                )

            # Record checkpoint effectiveness
            checkpoint_calls = all_tool_calls[pre_count:]
            checkpoint_scores = score_run(checkpoint_calls, {})
            strategy.record_result(checkpoint_scores)
            checkpoint_results.append({
                "turn": turn_idx,
                "encodes_from_checkpoint": checkpoint_scores["total_encodes"],
                "connections_from_checkpoint": checkpoint_scores["total_connections"],
                "uncertainties_from_checkpoint": checkpoint_scores["total_uncertainties"],
            })

            if text_blocks := [b for b in response.content if b.type == "text"]:
                messages.append({"role": "assistant", "content": text_blocks[0].text})

    # Final scoring
    final_scores = score_run(all_tool_calls, {})

    # Separate organic vs checkpoint-prompted encoding
    organic_calls = [c for c in all_tool_calls if not c.get("from_checkpoint")]
    checkpoint_calls = [c for c in all_tool_calls if c.get("from_checkpoint")]
    organic_scores = score_run(organic_calls, {})
    cp_scores = score_run(checkpoint_calls, {})

    return {
        "total": final_scores,
        "organic": organic_scores,
        "from_checkpoints": cp_scores,
        "checkpoint_details": checkpoint_results,
        "total_tool_calls": len(all_tool_calls),
    }


# ── Main Evaluation ─────────────────────────────────────────────────────

def run_adaptive_eval(model="claude-sonnet-4-20250514", max_workers=6, verbose=True):
    client = anthropic.Anthropic()
    skill_content = load_current_skill()

    strategies = {
        "no_checkpoint": ("No checkpoints (baseline)", lambda: NoCheckpoint()),
        "static_3": ("Static rotation, every 3 msgs", lambda: StaticRotation(interval=3)),
        "static_2": ("Static rotation, every 2 msgs", lambda: StaticRotation(interval=2)),
        "static_5": ("Static rotation, every 5 msgs", lambda: StaticRotation(interval=5)),
        "adaptive_3": ("Adaptive rotation, every 3 msgs", lambda: AdaptiveRotation(interval=3)),
        "adaptive_2": ("Adaptive rotation, every 2 msgs", lambda: AdaptiveRotation(interval=2)),
        "hook_specific_2": ("Hook-specific, every 2 msgs", lambda: HookSpecific(interval=2)),
        "static_5focus": ("5-focus rotation, every 2 msgs", lambda: StaticRotation(
            cycle=["uncertainty", "connections", "decisions", "blast_radius", "patterns"],
            interval=2
        )),
    }

    sessions = list(MULTI_TURN_SESSIONS.keys())
    results = {}

    # Build all combos
    combos = [(sk, sess) for sk in strategies for sess in sessions]

    if verbose:
        print(f"🧪 Adaptive Encoding Evaluation")
        print(f"   Strategies: {len(strategies)}")
        print(f"   Sessions: {len(sessions)}")
        print(f"   Total combos: {len(combos)}")
        print(f"   Workers: {max_workers}\n")

    def run_combo(strategy_key, session_key):
        sname, sfactory = strategies[strategy_key]
        strategy = sfactory()
        session = MULTI_TURN_SESSIONS[session_key]
        try:
            result = run_session(client, model, skill_content, session, strategy, verbose=verbose)
            if verbose:
                t = result["total"]
                o = result["organic"]
                c = result["from_checkpoints"]
                print(f"\n  ✅ {sname} × {session['name']}")
                print(f"     Total: Rich={t['encoding_richness']}% Enc={t['total_encodes']} Conn={t['total_connections']} Uncr={t['total_uncertainties']} Imp={t['total_impacts']}")
                print(f"     Organic: Enc={o['total_encodes']} | From checkpoints: Enc={c['total_encodes']} Conn={c['total_connections']} Uncr={c['total_uncertainties']}")
            return (strategy_key, session_key, result)
        except Exception as e:
            if verbose:
                print(f"\n  ❌ {sname} × {session['name']}: {e}")
            return (strategy_key, session_key, {"error": str(e)})

    # Run in parallel
    raw_results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(run_combo, sk, sess): (sk, sess) for sk, sess in combos}
        for future in as_completed(futures):
            raw_results.append(future.result())

    # Assemble results
    for strategy_key, session_key, result in raw_results:
        if strategy_key not in results:
            results[strategy_key] = {"name": strategies[strategy_key][0], "sessions": {}}
        results[strategy_key]["sessions"][session_key] = result

    return results


def print_adaptive_summary(results):
    print(f"\n\n{'='*100}")
    print("  ADAPTIVE ENCODING EVALUATION — STRATEGY COMPARISON")
    print(f"{'='*100}\n")

    header = f"{'Strategy':<40} | {'Rich':>5} | {'Enc':>4} | {'Conn':>5} | {'Uncr':>5} | {'Imp':>4} | {'Org':>4} | {'CP':>4} | {'Len':>5}"
    print(header)
    print("-" * len(header))

    ranked = []
    for sk, sd in results.items():
        sessions = [s for s in sd["sessions"].values() if "error" not in s]
        if not sessions:
            continue
        n = len(sessions)
        avg_t = lambda k: sum(s["total"].get(k, 0) for s in sessions) / n
        avg_o = lambda k: sum(s["organic"].get(k, 0) for s in sessions) / n
        avg_c = lambda k: sum(s["from_checkpoints"].get(k, 0) for s in sessions) / n
        richness = avg_t("encoding_richness")
        ranked.append((richness, sd["name"], avg_t, avg_o, avg_c, sessions))

    for richness, name, avg_t, avg_o, avg_c, sessions in sorted(ranked, key=lambda x: -x[0]):
        print(f"{name[:40]:<40} | {avg_t('encoding_richness'):>5.1f} | {avg_t('total_encodes'):>4.1f} | {avg_t('total_connections'):>5.1f} | {avg_t('total_uncertainties'):>5.1f} | {avg_t('total_impacts'):>4.1f} | {avg_o('total_encodes'):>4.1f} | {avg_c('total_encodes'):>4.1f} | {avg_t('avg_content_length'):>5.0f}")

    # Checkpoint effectiveness
    print(f"\n  📊 Checkpoint Effectiveness (encodes generated BY checkpoints):")
    for sk, sd in sorted(results.items(), key=lambda x: -sum(
        s.get("from_checkpoints", {}).get("total_encodes", 0)
        for s in x[1]["sessions"].values() if "error" not in s
    )):
        sessions = [s for s in sd["sessions"].values() if "error" not in s]
        if not sessions:
            continue
        n = len(sessions)
        cp_enc = sum(s["from_checkpoints"].get("total_encodes", 0) for s in sessions) / n
        cp_conn = sum(s["from_checkpoints"].get("total_connections", 0) for s in sessions) / n
        cp_uncr = sum(s["from_checkpoints"].get("total_uncertainties", 0) for s in sessions) / n
        org_enc = sum(s["organic"].get("total_encodes", 0) for s in sessions) / n
        if cp_enc > 0 or "no_checkpoint" not in sk:
            print(f"    {sd['name'][:40]:<40} Organic: {org_enc:.1f} enc | Checkpoints added: +{cp_enc:.1f} enc, +{cp_conn:.1f} conn, +{cp_uncr:.1f} uncr")

    print(f"\n{'='*100}")


def save_adaptive_results(results):
    path = Path(__file__).parent / "results"
    path.mkdir(exist_ok=True)
    filepath = path / f"adaptive_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

    serializable = {}
    for sk, sd in results.items():
        serializable[sk] = {"name": sd["name"], "sessions": {}}
        for sess_k, sess_v in sd["sessions"].items():
            if "error" in sess_v:
                serializable[sk]["sessions"][sess_k] = {"error": sess_v["error"]}
            else:
                serializable[sk]["sessions"][sess_k] = {
                    "total": {k: v for k, v in sess_v["total"].items() if not isinstance(v, set)},
                    "organic": {k: v for k, v in sess_v["organic"].items() if not isinstance(v, set)},
                    "from_checkpoints": {k: v for k, v in sess_v["from_checkpoints"].items() if not isinstance(v, set)},
                    "checkpoint_details": sess_v.get("checkpoint_details", []),
                }

    with open(filepath, "w") as f:
        json.dump({"timestamp": datetime.now().isoformat(), "results": serializable}, f, indent=2)
    print(f"\n💾 Saved to {filepath}")


if __name__ == "__main__":
    results = run_adaptive_eval(verbose=True, max_workers=6)
    print_adaptive_summary(results)
    save_adaptive_results(results)
