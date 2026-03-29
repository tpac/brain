"""CapabilityTest — base class for all memory capability tests.

Each test:
1. Copies a fixture brain to temp dir
2. Verifies preconditions (required nodes exist, have expected content)
3. Runs the encoding agent on a conversation scenario
4. Captures ALL actions (remember, revise, connect, recall, find_node_by_title, divergence)
5. Verifies expected actions happened (must/should/must_not)
6. Optionally verifies recall after encoding
7. Returns a structured score

The encoding agent uses REAL brain calls — not fake tools.
This is the only way to test revision, connection, and recall interaction.
"""
import sys
import os
import json
import time
import shutil
import tempfile
from pathlib import Path
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field, asdict

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

import anthropic


# ── Action capture ──

@dataclass
class CapturedAction:
    """A single action the encoding agent took."""
    tool: str           # remember, revise, connect, recall, find_node_by_title, etc.
    args: Dict          # full arguments passed
    result: Any = None  # what the brain returned
    error: str = ""     # if it failed
    timestamp: float = 0


@dataclass
class CapabilityScore:
    """Score for a single scenario test."""
    capability: str
    scenario_id: str
    verdict: str = "UNKNOWN"    # PASS, FAIL, ERROR
    must_total: int = 0
    must_passed: int = 0
    should_total: int = 0
    should_passed: int = 0
    must_not_total: int = 0
    must_not_violated: int = 0
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    actions: List[Dict] = field(default_factory=list)
    elapsed_seconds: float = 0

    def to_dict(self):
        return asdict(self)


# ── Brain wrapper that captures all actions ──

class InstrumentedBrain:
    """Wraps a real Brain instance and captures every action for scoring."""

    def __init__(self, brain):
        self._brain = brain
        self.actions: List[CapturedAction] = []

    def _capture(self, tool: str, args: Dict, func, *a, **kw):
        action = CapturedAction(tool=tool, args=args, timestamp=time.time())
        try:
            result = func(*a, **kw)
            action.result = result
        except Exception as e:
            action.error = str(e)
            result = None
        self.actions.append(action)
        return result

    def remember(self, **kwargs):
        return self._capture("remember", kwargs, self._brain.remember, **kwargs)

    def revise(self, **kwargs):
        return self._capture("revise", kwargs, self._brain.revise, **kwargs)

    def connect(self, **kwargs):
        return self._capture("connect", kwargs, self._brain.connect, **kwargs)

    def recall(self, query, **kwargs):
        args = {"query": query, **kwargs}
        return self._capture("recall", args, self._brain.recall, query, **kwargs)

    def find_node_by_title(self, title_query, **kwargs):
        args = {"title_query": title_query, **kwargs}
        return self._capture("find_node_by_title", args,
                             self._brain.find_node_by_title, title_query, **kwargs)

    def get_node(self, node_id):
        return self._capture("get_node", {"node_id": node_id},
                             self._brain.get_node, node_id)

    def record_divergence(self, **kwargs):
        return self._capture("record_divergence", kwargs,
                             self._brain.record_divergence, **kwargs)

    def learn_vocabulary(self, **kwargs):
        return self._capture("learn_vocabulary", kwargs,
                             self._brain.learn_vocabulary, **kwargs)

    def remember_lesson(self, **kwargs):
        return self._capture("remember_lesson", kwargs,
                             self._brain.remember_lesson, **kwargs)

    def remember_mechanism(self, **kwargs):
        return self._capture("remember_mechanism", kwargs,
                             self._brain.remember_mechanism, **kwargs)

    def remember_convention(self, **kwargs):
        return self._capture("remember_convention", kwargs,
                             self._brain.remember_convention, **kwargs)

    def remember_impact(self, **kwargs):
        return self._capture("remember_impact", kwargs,
                             self._brain.remember_impact, **kwargs)

    def remember_mental_model(self, **kwargs):
        return self._capture("remember_mental_model", kwargs,
                             self._brain.remember_mental_model, **kwargs)

    def save(self):
        self._brain.save()

    def close(self):
        self._brain.close()

    @property
    def conn(self):
        return self._brain.conn


# ── MCP-style tools that route to InstrumentedBrain ──

def _build_capability_tools():
    """Build tool schemas from contract — single source of truth.
    Matches production MCP schemas so eval tests simulate real behavior.
    """
    from servers.contract import get_remember_fields

    # Build remember properties from contract
    remember_props = {"type": {"type": "string"}, "title": {"type": "string"}, "content": {"type": "string"}}
    for name, spec in get_remember_fields().items():
        if name in ("type", "title", "content", "id", "archived"):
            continue
        prop = {"type": {"str": "string", "float": "number", "bool": "boolean"}.get(spec.get("type", "str"), "string")}
        if spec.get("description"):
            prop["description"] = spec["description"]
        remember_props[name] = prop

    # Build revise properties from contract
    revise_props = {"node_id": {"type": "string"}, "reason": {"type": "string"}}
    for name, spec in get_remember_fields().items():
        if name in ("type", "title", "id", "archived"):
            continue
        prop = {"type": {"str": "string", "float": "number", "bool": "boolean"}.get(spec.get("type", "str"), "string")}
        desc = spec.get("description", "")
        if spec.get("append_on_revise"):
            desc = (desc + " " if desc else "") + "(appended on revise)"
        else:
            desc = (desc + " " if desc else "") + "(replaces existing value)"
        prop["description"] = desc.strip()
        revise_props[name] = prop

    return [
        {"name": "recall", "description": "Search brain for existing knowledge. Use types to filter by node type.",
         "input_schema": {"type": "object", "required": ["query"],
                          "properties": {"query": {"type": "string"},
                                         "limit": {"type": "integer", "default": 8},
                                         "types": {"type": "array", "items": {"type": "string"},
                                                   "description": "Filter by node types, e.g. ['correction', 'rule']"}}}},
        {"name": "find_node_by_title", "description": "Find an existing node by fuzzy title match.",
         "input_schema": {"type": "object", "required": ["title_query"],
                          "properties": {"title_query": {"type": "string"}, "threshold": {"type": "number", "default": 0.75}}}},
        {"name": "get_node", "description": "Get a node by its exact ID with full content and connections.",
         "input_schema": {"type": "object", "required": ["node_id"],
                          "properties": {"node_id": {"type": "string"}}}},
        {"name": "remember", "description": "Store a new memory node. Include situation for WHEN-relevance.",
         "input_schema": {"type": "object", "required": ["type", "title", "content"],
                          "properties": remember_props}},
        {"name": "revise", "description": "Update any field(s) on an existing node. Content is appended with revision history. All other fields are replaced.",
         "input_schema": {"type": "object", "required": ["node_id", "reason"],
                          "properties": revise_props}},
        {"name": "connect", "description": "Create a link between two nodes.",
         "input_schema": {"type": "object", "required": ["source_id", "target_id"],
                          "properties": {"source_id": {"type": "string"}, "target_id": {"type": "string"},
                                         "relation": {"type": "string", "default": "related_to"},
                                         "weight": {"type": "number", "default": 0.5}}}},
        {"name": "record_divergence", "description": "Record where the AI diverged from reality.",
         "input_schema": {"type": "object", "required": ["claude_assumed", "reality", "underlying_pattern"],
                          "properties": {"claude_assumed": {"type": "string"}, "reality": {"type": "string"},
                                         "underlying_pattern": {"type": "string"}}}},
        {"name": "learn_vocabulary", "description": "Map an operator term to its meaning.",
         "input_schema": {"type": "object", "required": ["term", "maps_to", "context"],
                          "properties": {"term": {"type": "string"}, "maps_to": {"type": "string"},
                                         "context": {"type": "string"}}}},
        {"name": "remember_lesson", "description": "Store a lesson learned from a mistake or discovery.",
         "input_schema": {"type": "object", "required": ["title", "what_happened", "root_cause", "fix", "preventive_principle"],
                          "properties": {"title": {"type": "string"}, "what_happened": {"type": "string"},
                                         "root_cause": {"type": "string"}, "fix": {"type": "string"},
                                         "preventive_principle": {"type": "string"}}}},
        {"name": "remember_mechanism", "description": "Store how something works — steps, data flow.",
         "input_schema": {"type": "object", "required": ["title", "content"],
                          "properties": {"title": {"type": "string"}, "content": {"type": "string"},
                                         "steps": {"type": "array", "items": {"type": "string"}}}}},
        {"name": "remember_mental_model", "description": "Store a mental model or framework for thinking about something.",
         "input_schema": {"type": "object", "required": ["title", "model_description", "applies_to"],
                          "properties": {"title": {"type": "string"}, "model_description": {"type": "string"},
                                         "applies_to": {"type": "string"},
                                         "confidence": {"type": "number", "default": 0.6}}}},
        {"name": "remember_impact", "description": "Record a dependency — if X changes, check Y.",
         "input_schema": {"type": "object", "required": ["title", "if_changed", "must_check", "because"],
                          "properties": {"title": {"type": "string"}, "if_changed": {"type": "string"},
                                         "must_check": {"type": "string"}, "because": {"type": "string"}}}},
        {"name": "remember_convention", "description": "Store a coding convention — pattern and anti-pattern.",
         "input_schema": {"type": "object", "required": ["title", "content", "pattern", "anti_pattern"],
                          "properties": {"title": {"type": "string"}, "content": {"type": "string"},
                                         "pattern": {"type": "string"}, "anti_pattern": {"type": "string"}}}},
    ]


CAPABILITY_TOOLS = _build_capability_tools()

def _load_encoding_system():
    """Load the encoding agent prompt from the canonical file."""
    prompt_path = os.path.join(ROOT, "hooks", "prompts", "encoding-agent.md")
    try:
        with open(prompt_path) as f:
            return f.read()
    except Exception:
        return "You are the encoding agent. Search before encoding. Revise stale nodes. Create only when nothing exists."

ENCODING_SYSTEM = _load_encoding_system()


def dispatch_tool(brain: InstrumentedBrain, tool_name: str, tool_input: Dict) -> str:
    """Route a tool call from the LLM to the InstrumentedBrain.
    Recall output uses pipeline_contract formatters for consistency with production.
    """
    try:
        if tool_name == "recall":
            kwargs = {"limit": tool_input.get("limit", 8)}
            if tool_input.get("types"):
                kwargs["types"] = tool_input["types"]
            result = brain.recall(tool_input["query"], **kwargs)
            nodes = result.get("results", []) or result.get("nodes", [])
            # Return structured JSON (agent needs IDs for connect) + formatted text
            from servers.brain_voice import BrainVoice
            summary = []
            for n in nodes[:8]:
                entry = {
                    "id": n.get("id", ""),
                    "type": n.get("type", ""),
                    "title": n.get("title", ""),
                    "content": (n.get("content", "") or "")[:500],
                    "confidence": n.get("confidence", 0),
                    "locked": n.get("locked", False),
                    "created_at": str(n.get("created_at", ""))[:10],
                    "revised_at": n.get("revised_at") or "never",
                }
                neighbors = n.get("_neighbors") or []
                if neighbors:
                    entry["neighbors"] = [
                        {"id": nb.get("id", ""), "title": nb.get("title", "")[:50],
                         "type": nb.get("type", ""), "relation": nb.get("relation", ""),
                         "confidence": nb.get("confidence", 0)}
                        for nb in neighbors[:3]
                    ]
                summary.append(entry)
            return json.dumps({"ok": True, "results": summary})

        elif tool_name == "find_node_by_title":
            result = brain.find_node_by_title(tool_input["title_query"],
                                               threshold=tool_input.get("threshold", 0.75))
            if result:
                return json.dumps({"ok": True, "found": True, "id": result.get("id", ""),
                                   "title": result.get("title", ""), "score": result.get("score", 0)})
            return json.dumps({"ok": True, "found": False})

        elif tool_name == "get_node":
            result = brain.get_node(tool_input["node_id"])
            if result:
                connections = result.get("connections") or result.get("_neighbors") or []
                return json.dumps({"ok": True, "node": {
                    "id": result.get("id", ""),
                    "type": result.get("type", ""),
                    "title": result.get("title", ""),
                    "content": result.get("content", ""),
                    "confidence": result.get("confidence", 0),
                    "locked": result.get("locked", False),
                    "created_at": str(result.get("created_at", ""))[:10],
                    "revised_at": result.get("revised_at") or "never",
                    "access_count": result.get("access_count", 0),
                    "connections": [
                        {"target_id": c.get("target_id", c.get("id", "")),
                         "title": c.get("title", "")[:60],
                         "type": c.get("type", ""),
                         "relation": c.get("relation", "")}
                        for c in connections[:8]
                    ],
                }})
            return json.dumps({"ok": True, "node": None})

        elif tool_name == "remember":
            result = brain.remember(**tool_input)
            return json.dumps({"ok": True, "id": result.get("id", "") if result else "error"})

        elif tool_name == "revise":
            result = brain.revise(**tool_input)
            verified = result.get("verified", False) if isinstance(result, dict) else False
            return json.dumps({"ok": True, "revised": True, "verified": verified})

        elif tool_name == "connect":
            result = brain.connect(**tool_input)
            return json.dumps({"ok": True, "connected": True})

        elif tool_name == "record_divergence":
            result = brain.record_divergence(**tool_input)
            return json.dumps({"ok": True, "recorded": True})

        elif tool_name == "learn_vocabulary":
            result = brain.learn_vocabulary(**tool_input)
            return json.dumps({"ok": True, "id": result.get("id", "") if result else "ok"})

        elif tool_name == "remember_lesson":
            result = brain.remember_lesson(**tool_input)
            return json.dumps({"ok": True, "id": result.get("id", "") if result else "ok"})

        elif tool_name == "remember_mechanism":
            result = brain.remember_mechanism(**tool_input)
            return json.dumps({"ok": True, "id": result.get("id", "") if result else "ok"})

        elif tool_name == "remember_mental_model":
            result = brain.remember_mental_model(**tool_input)
            return json.dumps({"ok": True, "id": result.get("id", "") if result else "ok"})

        elif tool_name == "remember_impact":
            result = brain.remember_impact(**tool_input)
            return json.dumps({"ok": True, "id": result.get("id", "") if result else "ok"})

        elif tool_name == "remember_convention":
            result = brain.remember_convention(**tool_input)
            return json.dumps({"ok": True, "id": result.get("id", "") if result else "ok"})

        else:
            return json.dumps({"ok": False, "error": "Unknown tool: %s" % tool_name})

    except Exception as e:
        return json.dumps({"ok": False, "error": str(e)})


# ── CapabilityTest base class ──

class CapabilityTest:
    """Base for all capability tests."""

    capability_name = "base"

    def __init__(self, fixture_path: str, model: str = "claude-sonnet-4-6"):
        self.fixture_path = fixture_path
        self.model = model
        self.client = anthropic.Anthropic()

    def load_scenario(self, path: str) -> Dict:
        """Load a scenario JSON file."""
        with open(path) as f:
            return json.load(f)

    def setup_brain(self, scenario: Dict) -> tuple:
        """Copy fixture brain to temp dir, return (InstrumentedBrain, work_dir)."""
        work_dir = tempfile.mkdtemp(prefix="brain_cap_")
        db_path = os.path.join(work_dir, "brain.db")
        shutil.copy2(self.fixture_path, db_path)

        # Also copy logs db if it exists
        logs_src = self.fixture_path.replace("brain.db", "brain_logs.db")
        if os.path.exists(logs_src):
            shutil.copy2(logs_src, os.path.join(work_dir, "brain_logs.db"))

        from servers.brain import Brain
        brain = Brain(db_path=db_path)
        instrumented = InstrumentedBrain(brain)

        # Resolve node references by title (fixture IDs change on rebuild)
        self._resolved_ids = {}
        raw_brain = instrumented._brain
        preconditions = scenario.get("brain_preconditions", {})
        for ref in preconditions.get("required_nodes", []):
            # ref can be a node ID or a title substring
            node = raw_brain.get_node(ref)
            if node:
                self._resolved_ids[ref] = ref
            else:
                # Try title match
                found = raw_brain.find_node_by_title(ref, threshold=0.7)
                if found:
                    self._resolved_ids[ref] = found.get("id", "")
                else:
                    raise ValueError("Precondition failed: node '%s' not found in fixture" % ref)

        # Replace refs in expected_actions and expected_recall
        self._resolve_scenario_refs(scenario)

        return instrumented, work_dir

    def _resolve_scenario_refs(self, scenario):
        """Replace title-based node references with actual IDs from fixture."""
        for action in scenario.get("expected_actions", []):
            for key in ("target", "target_node"):
                if key in action and action[key] in self._resolved_ids:
                    action[key] = self._resolved_ids[action[key]]

        for recall in scenario.get("expected_recall", []):
            if "must_surface" in recall and recall["must_surface"] in self._resolved_ids:
                recall["must_surface"] = self._resolved_ids[recall["must_surface"]]

    def run_encoding(self, brain: InstrumentedBrain, scenario: Dict) -> List[CapturedAction]:
        """Feed conversation to encoding agent, capture all actions."""
        exchanges = scenario["exchanges"]
        conv_text = "\n".join(
            "[%s]: %s" % (ex["role"].upper(), ex["content"][:1500])
            for ex in exchanges
        )

        # Include brain context if the scenario specifies nodes the agent should find
        context_hint = ""
        if scenario.get("brain_preconditions", {}).get("required_nodes"):
            context_hint = "\n\nNote: The brain already has existing knowledge. SEARCH before encoding."

        messages = [
            {"role": "user",
             "content": "Here is the conversation to analyze:\n\n%s%s\n\nEncode what's important. Search the brain first." % (conv_text, context_hint)}
        ]

        response = self.client.messages.create(
            model=self.model,
            max_tokens=4096,
            system=ENCODING_SYSTEM,
            messages=messages,
            tools=CAPABILITY_TOOLS,
        )

        # Tool use loop
        for _ in range(8):
            tool_uses = [b for b in response.content if b.type == "tool_use"]
            if not tool_uses:
                break

            tool_results = []
            for tu in tool_uses:
                result_text = dispatch_tool(brain, tu.name, tu.input)
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": tu.id,
                    "content": result_text,
                })

            messages.append({
                "role": "assistant",
                "content": [
                    {"type": b.type, **({"text": b.text} if b.type == "text" else
                                        {"id": b.id, "name": b.name, "input": b.input})}
                    for b in response.content
                ]
            })
            messages.append({"role": "user", "content": tool_results})

            response = self.client.messages.create(
                model=self.model,
                max_tokens=4096,
                system=ENCODING_SYSTEM,
                messages=messages,
                tools=CAPABILITY_TOOLS,
            )

        brain.save()
        return brain.actions

    def verify_actions(self, actions: List[CapturedAction],
                       expected: List[Dict]) -> CapabilityScore:
        """Check expected actions against captured actions.

        Expected action format:
        {
            "action": "revise" | "remember" | "connect" | "record_divergence" | "NOT_create" | ...,
            "priority": "must" | "should",
            # Plus action-specific match criteria
        }
        """
        score = CapabilityScore(
            capability=self.capability_name,
            scenario_id="",
            actions=[{"tool": a.tool, "args": a.args, "error": a.error} for a in actions],
        )

        for exp in expected:
            action_type = exp["action"]
            priority = exp.get("priority", "must")

            if action_type.startswith("NOT_"):
                # Negative assertion — this action should NOT have happened
                score.must_not_total += 1
                actual_type = action_type[4:]  # "NOT_create" → "create"
                # Map "create" to "remember" for matching
                if actual_type == "create":
                    actual_type = "remember"

                violated = False
                for a in actions:
                    if a.tool == actual_type and self._matches_negative(a, exp):
                        violated = True
                        break

                if violated:
                    score.must_not_violated += 1
                    score.errors.append("VIOLATED must_not: %s (found %s)" % (
                        exp.get("reason", action_type), actual_type))
            else:
                # Positive assertion — this action should have happened
                if priority == "must":
                    score.must_total += 1
                else:
                    score.should_total += 1

                found = False
                for a in actions:
                    if a.tool == action_type and self._matches_positive(a, exp):
                        found = True
                        break

                if found:
                    if priority == "must":
                        score.must_passed += 1
                    else:
                        score.should_passed += 1
                else:
                    if priority == "must":
                        score.errors.append("MISSING must: %s" % action_type)
                    else:
                        score.warnings.append("MISSING should: %s %s" % (
                            action_type, exp.get("reason", "")))

        # Verdict
        if score.must_not_violated > 0 or score.must_passed < score.must_total:
            score.verdict = "FAIL"
        else:
            score.verdict = "PASS"

        return score

    def _matches_positive(self, action: CapturedAction, expected: Dict) -> bool:
        """Check if a captured action matches a positive expectation."""
        # Check target node
        if "target" in expected or "target_node" in expected:
            target = expected.get("target") or expected.get("target_node")
            node_id = action.args.get("node_id", "")
            if target and not (node_id.startswith(target) or target.startswith(node_id)):
                return False

        # Check content contains required strings
        for key in ("should_contain", "must_contain"):
            required = expected.get(key, [])
            if isinstance(required, str):
                required = [required]
            if required:
                content = json.dumps(action.args).lower()
                if not all(r.lower() in content for r in required):
                    return False

        # Check content does NOT contain forbidden strings
        forbidden = expected.get("should_not_contain", [])
        if isinstance(forbidden, str):
            forbidden = [forbidden]
        if forbidden:
            content = json.dumps(action.args).lower()
            if any(f.lower() in content for f in forbidden):
                return False

        return True

    def _matches_negative(self, action: CapturedAction, expected: Dict) -> bool:
        """Check if a captured action matches a negative (must_not) expectation."""
        # For NOT_create: check if a new node was created with similar title
        title_like = expected.get("title_similar_to", "").lower()
        if title_like:
            created_title = (action.args.get("title", "") or "").lower()
            # Fuzzy: check if most words match
            words = [w for w in title_like.split() if len(w) > 2]
            if words:
                matches = sum(1 for w in words if w in created_title)
                return matches >= len(words) * 0.5
        return False

    def verify_recall(self, brain: InstrumentedBrain,
                      expected_recall: List[Dict]) -> List[Dict]:
        """After encoding, verify recall returns the right things."""
        results = []
        for er in expected_recall:
            query = er["query"]
            recall_result = brain.recall(query, limit=8)
            returned = recall_result.get("results", []) or recall_result.get("nodes", [])

            result = {"query": query, "returned": len(returned)}

            # Check must_surface
            if "must_surface" in er:
                target_id = er["must_surface"]
                found = any(
                    n.get("id", "").startswith(target_id) or target_id.startswith(n.get("id", ""))
                    for n in returned
                )
                result["must_surface"] = found

            # Check must_contain_updated
            if "must_contain_updated" in er:
                required = er["must_contain_updated"]
                if isinstance(required, str):
                    required = [required]
                # Check if any returned node contains all required strings
                found = False
                for n in returned:
                    text = ((n.get("title", "") or "") + " " + (n.get("content", "") or "")).lower()
                    if all(r.lower() in text for r in required):
                        found = True
                        break
                result["contains_updated"] = found

            results.append(result)

        return results

    def run_scenario(self, scenario_path: str, verbose: bool = False) -> CapabilityScore:
        """Run a single scenario end-to-end."""
        scenario = self.load_scenario(scenario_path)
        t0 = time.time()

        brain, work_dir = self.setup_brain(scenario)
        try:
            # Run encoding
            if verbose:
                print("  Encoding %s..." % scenario["id"])
            actions = self.run_encoding(brain, scenario)

            if verbose:
                for a in actions:
                    status = "OK" if not a.error else "ERR: %s" % a.error
                    args_summary = ""
                    if a.tool in ("remember", "revise", "remember_lesson", "remember_mechanism"):
                        args_summary = a.args.get("title", a.args.get("content", ""))[:50]
                    elif a.tool in ("recall", "find_node_by_title"):
                        args_summary = a.args.get("query", a.args.get("title_query", ""))[:50]
                    elif a.tool == "connect":
                        args_summary = "%s → %s" % (a.args.get("source_id", "?")[:8],
                                                      a.args.get("target_id", "?")[:8])
                    print("    [%s] %s %s" % (a.tool, args_summary, status))

            # Verify actions
            expected_actions = scenario.get("expected_actions", [])
            score = self.verify_actions(actions, expected_actions)
            score.scenario_id = scenario["id"]
            score.elapsed_seconds = time.time() - t0

            # Verify recall if specified
            expected_recall = scenario.get("expected_recall", [])
            if expected_recall:
                recall_results = self.verify_recall(brain, expected_recall)
                for rr in recall_results:
                    if rr.get("must_surface") is False:
                        score.must_total += 1
                        score.errors.append("RECALL MISS: '%s' not surfaced" % rr["query"])
                        score.verdict = "FAIL"
                    elif rr.get("must_surface") is True:
                        score.must_total += 1
                        score.must_passed += 1
                    if rr.get("contains_updated") is False:
                        score.warnings.append("RECALL STALE: '%s' missing updated content" % rr["query"])

            if verbose:
                print("  Result: %s (must: %d/%d, should: %d/%d, must_not_violated: %d)" % (
                    score.verdict, score.must_passed, score.must_total,
                    score.should_passed, score.should_total, score.must_not_violated))
                for e in score.errors:
                    print("    ERROR: %s" % e)
                for w in score.warnings:
                    print("    WARN: %s" % w)

            return score

        finally:
            brain.close()
            shutil.rmtree(work_dir, ignore_errors=True)
