#!/usr/bin/env python3
"""
⚠️  OBSOLETE — DO NOT USE (2026-03-21)

This module was an early prototype for replaying transcripts against a fresh
brain.  Several assumptions are outdated:

  • mark_recall_used() was never implemented on Brain, so recall-precision
    tracking (used_count / precision_score in recall_log) is dead code.
  • The encoding extraction heuristics predate the daemon consolidation
    (v5.3) and thin-client hook architecture.
  • Session lifecycle simulation is now covered by tests/test_system.py.

TODO:
  - Implement Brain.mark_recall_used() and wire it into post_response_track
    so recall_log.used_count actually gets populated.
  - Decide whether this replay framework is worth resurrecting or if
    test_system.py's lifecycle tests are sufficient.
  - If resurrecting: rewrite to use daemon_hooks.py functions directly
    instead of duplicating hook logic inline.

Original description (for reference):
    Relearning Comparison Engine — replays real conversation transcripts
    against a fresh brain to test whether the current brain logic can
    reproduce (and improve upon) what was learned.

Usage (disabled):
    python tests/relearning.py <transcript.jsonl> [--current-brain brain.db]
"""

import sys
import os
import json
import time
import re
import shutil
import traceback
from datetime import datetime
from typing import Dict, List, Any, Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from tests.transcript_parser import parse_transcript


# ═══════════════════════════════════════════════════════════════
# LLM ENCODER — Uses real Claude API for encoding decisions
# ═══════════════════════════════════════════════════════════════

class LLMEncoder:
    """
    Uses Claude to make encoding decisions, exactly as the production
    SKILL.md instructs Claude to do.

    Supports two backends:
      - 'api': Direct Anthropic API (needs ANTHROPIC_API_KEY)
      - 'cli': Claude CLI subprocess (works in Cowork/sandboxed environments)
    """

    # Noise patterns for pre-filtering turns before sending to LLM
    NOISE_PATTERNS = [
        re.compile(r'^Todos have been modified', re.I),
        re.compile(r'^Please proceed with the current tasks', re.I),
        re.compile(r'^Ensure that you continue to use the todo', re.I),
        re.compile(r'^File created successfully', re.I),
        re.compile(r'^The file /sessions/', re.I),
        re.compile(r'^total \d+\s*$'),
        re.compile(r'^-rw-', re.I),
        re.compile(r'^drwx', re.I),
        re.compile(r'^\s*$'),
        re.compile(r'^<system-reminder>', re.I),
        re.compile(r'^Skill .* is loading', re.I),
    ]

    def __init__(self, model: str = 'claude-haiku-4-5-20251001', batch_size: int = 15,
                 backend: str = 'auto', max_parallel: int = 5):
        """
        Args:
            model: Claude model to use. Haiku for speed/cost, Sonnet for quality.
            batch_size: How many turns to process per API call (default 50 for speed).
            backend: 'api' for Anthropic SDK, 'cli' for claude CLI, 'auto' to detect.
            max_parallel: Max concurrent API calls (API backend only, CLI is sequential).
        """
        self.model = model
        self.batch_size = batch_size
        self.max_parallel = max_parallel
        self.api_calls = 0
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.errors = []

        # Detect backend
        if backend == 'auto':
            backend = self._detect_backend()
        self.backend = backend

        if self.backend == 'api':
            import anthropic
            import httpx
            # Some environments (Cowork VM) use a proxy with self-signed certs.
            # Disable SSL verification to allow API calls through the proxy.
            http_client = httpx.Client(verify=False)
            self.client = anthropic.Anthropic(http_client=http_client)
        else:
            self.client = None

    def _detect_backend(self) -> str:
        """Auto-detect whether to use API or CLI."""
        if os.environ.get('ANTHROPIC_API_KEY'):
            return 'api'
        # Check for claude CLI
        import subprocess
        try:
            result = subprocess.run(['claude', '--version'], capture_output=True, timeout=5)
            if result.returncode == 0:
                return 'cli'
        except Exception:
            pass
        return 'api'  # Will fail later with a clear error

    @classmethod
    def is_meaningful_turn(cls, turn: Dict) -> bool:
        """Pre-filter: is this turn worth sending to the LLM?"""
        user_msg = turn.get('user_message', '').strip()
        if not user_msg or len(user_msg) < 10:
            return False
        for pat in cls.NOISE_PATTERNS:
            if pat.search(user_msg):
                return False
        return True

    def encode_batch(self, turns: List[Dict], session_context: Dict,
                     existing_node_titles: List[str] = None) -> List[Dict]:
        """
        Send a batch of turns to Claude for encoding decisions.
        Pre-filters noise turns before sending.
        """
        if not turns:
            return []

        prompt = self._build_prompt(turns, session_context, existing_node_titles)
        if not prompt:
            return []

        # Call Claude via the appropriate backend
        if self.backend == 'api':
            text = self._call_api(prompt)
        else:
            text = self._call_cli(prompt)

        if not text:
            return []

        return self._parse_response(text)

    def encode_batches_parallel(self, all_turns: List[Dict], session_context: Dict,
                                existing_node_titles: List[str] = None) -> List[Dict]:
        """
        Process multiple batches in parallel (API backend) or sequentially (CLI).
        Pre-filters noise turns, chunks into batches, sends concurrently.

        Args:
            all_turns: All turns to process (will be pre-filtered)
            session_context: Running context
            existing_node_titles: For connection matching

        Returns:
            All encoding dicts combined
        """
        # Pre-filter
        meaningful = [t for t in all_turns if self.is_meaningful_turn(t)]
        if not meaningful:
            return []

        # Chunk into batches
        batches = []
        for i in range(0, len(meaningful), self.batch_size):
            batch = meaningful[i:i + self.batch_size]
            batches.append(batch)

        if not batches:
            return []

        # Parallel execution (API backend with threading)
        if self.backend == 'api' and self.max_parallel > 1 and len(batches) > 1:
            return self._encode_parallel(batches, session_context, existing_node_titles)

        # Sequential fallback
        all_encodings = []
        for batch in batches:
            encodings = self.encode_batch(batch, session_context, existing_node_titles)
            all_encodings.extend(encodings)
        return all_encodings

    def _encode_parallel(self, batches: List[List[Dict]], session_context: Dict,
                         existing_node_titles: List[str] = None) -> List[Dict]:
        """Run batches in parallel using ThreadPoolExecutor."""
        from concurrent.futures import ThreadPoolExecutor, as_completed

        all_encodings = []

        with ThreadPoolExecutor(max_workers=self.max_parallel) as executor:
            futures = {}
            for i, batch in enumerate(batches):
                future = executor.submit(
                    self.encode_batch, batch, session_context, existing_node_titles
                )
                futures[future] = i

            for future in as_completed(futures):
                try:
                    encodings = future.result()
                    all_encodings.extend(encodings)
                except Exception as e:
                    self.errors.append(f'Parallel batch error: {e}')

        return all_encodings

    def _build_prompt(self, turns: List[Dict], session_context: Dict,
                      existing_node_titles: List[str] = None) -> Optional[str]:
        """Build the encoding prompt from turns."""
        turns_text = []
        for t in turns:
            user = t.get('user_message', '')[:600]
            assistant = t.get('assistant_response', '')[:600]
            files = t.get('files_edited', [])
            ts = t.get('timestamp', '')[:19]

            if user:
                turns_text.append(f"[{ts}] USER: {user}")
            if assistant:
                assistant = self._strip_system_noise(assistant)
                if assistant:
                    turns_text.append(f"  ASSISTANT: {assistant}")
            if files:
                turns_text.append(f"  FILES EDITED: {', '.join(files[:5])}")

        conversation = '\n'.join(turns_text)
        if len(conversation.strip()) < 50:
            return None

        titles_hint = ''
        if existing_node_titles:
            sample = existing_node_titles[:50]
            titles_hint = f"\nEXISTING BRAIN NODES (use these titles for connections when relevant):\n" + \
                          '\n'.join(f'  - {t}' for t in sample) + '\n'

        prompt = f"""You are the encoding engine for brain, a persistent brain for Claude.
Analyze this conversation excerpt and extract what should be remembered.

ENCODING RULES (from SKILL.md):
- Decisions, architecture changes → type:"decision", locked:true. EACH specific value as its own node.
- API gotchas, error patterns → type:"rule", locked:true. The specific failure AND the fix.
- User feedback, preferences → type:"rule", locked:true. The SPECIFIC thing said.
- Current state of work → type:"context", locked:false.
- User emotional reactions → type:"context", locked:false. WHAT they reacted to.
- New terms, names, jargon → type:"concept", locked:false.
- Work items, pending tasks → type:"task", locked:false.
- Files created or referenced → type:"file", locked:false.
- People, roles → type:"person", locked:false.
- Products, projects → type:"project", locked:false.
- Corrections (user says "actually", "no", "instead") → type:"decision", title starts with "Correction:", locked:true, emotion:0.7

TITLE RULES:
- Titles must be concise AND specific — scannable at a glance with key values.
- GOOD: "Auth: magic links only via Clerk, no passwords, free tier"
- GOOD: "GLO Brightness — Pricing tiers: Well($30) Bright($50) Shine($100)"
- BAD: "The file /sessions/gracious-lucid-johnson/..." (system paths)
- BAD: "Please proceed with the current tasks" (system noise)
- BAD: Raw conversation fragments without clear meaning

WHAT NOT TO ENCODE:
- System messages ("Please proceed with the current tasks if applicable")
- TodoList reminders, status messages
- File paths from the VM (/sessions/...)
- Claude's boilerplate responses
- Repetitive turn-by-turn status updates
- Content that's purely about brain's own internal operations (unless it's a decision about brain's design)

DECOMPOSE, don't summarize. Each specific value, name, number gets its OWN node.
If unsure whether it's valuable, ENCODE IT. Pruning handles cleanup.
If the conversation contains nothing worth remembering, return [].

Keywords should include: specific numbers, proper nouns, technical terms, user vocabulary.

Project context: {session_context.get('project', 'unknown')}
User: {session_context.get('user', 'unknown')}
{titles_hint}
CONVERSATION:
{conversation}

Return a JSON array. Each element:
{{"type":"decision|rule|concept|context|task|file|person|project","title":"concise scannable title with key value","content":"rich context - WHAT and WHY","keywords":"specific terms numbers names","locked":true|false,"emotion":0.0,"emotion_label":"neutral","connections":[{{"target_title":"existing node title","relation":"part_of|related|corrected_by|exemplifies|depends_on"}}]}}

Return ONLY a valid JSON array, no explanation. Empty array [] if nothing worth encoding."""
        return prompt

    def _call_api(self, prompt: str) -> Optional[str]:
        """Call Claude via Anthropic Python SDK."""
        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=4096,
                messages=[{'role': 'user', 'content': prompt}],
            )
            self.api_calls += 1
            self.total_input_tokens += response.usage.input_tokens
            self.total_output_tokens += response.usage.output_tokens
            return response.content[0].text.strip()
        except Exception as e:
            self.errors.append(f'API error: {e}')
            return None

    def _call_cli(self, prompt: str) -> Optional[str]:
        """Call Claude via claude CLI subprocess."""
        import subprocess
        try:
            result = subprocess.run(
                ['claude', '-p', prompt, '--model', self.model,
                 '--output-format', 'text', '--max-turns', '1'],
                capture_output=True, text=True, timeout=120
            )
            self.api_calls += 1
            if result.returncode == 0 and result.stdout.strip():
                return result.stdout.strip()
            else:
                err = result.stderr[:200] if result.stderr else 'empty response'
                self.errors.append(f'CLI error: {err}')
                return None
        except subprocess.TimeoutExpired:
            self.errors.append('CLI timeout (120s)')
            return None
        except Exception as e:
            self.errors.append(f'CLI error: {e}')
            return None

    def _parse_response(self, text: str) -> List[Dict]:
        """Parse and validate Claude's JSON response."""
        try:
            # Handle markdown code blocks
            if text.startswith('```'):
                text = re.sub(r'^```(?:json)?\s*', '', text)
                text = re.sub(r'\s*```$', '', text)

            encodings = json.loads(text)

            if not isinstance(encodings, list):
                return []

            valid = []
            for enc in encodings:
                if not isinstance(enc, dict):
                    continue
                if not enc.get('title') or not enc.get('type'):
                    continue
                title = enc['title']
                if any(noise in title.lower() for noise in [
                    'please proceed', '/sessions/', 'current tasks if applicable',
                    'todo list', 'system-reminder'
                ]):
                    continue
                valid.append(enc)

            return valid

        except json.JSONDecodeError as e:
            self.errors.append(f'JSON parse error: {e}')
            return []

    def _strip_system_noise(self, text: str) -> str:
        """Remove system injections from assistant responses."""
        noise_patterns = [
            r'Ensure that you continue to use the todo list.*?(?=\n|$)',
            r'Please proceed with the current tasks if applicable',
            r'<system-reminder>.*?</system-reminder>',
            r'Todos have been modified successfully\..*?(?=\n|$)',
        ]
        for pat in noise_patterns:
            text = re.sub(pat, '', text, flags=re.DOTALL | re.IGNORECASE)
        return text.strip()[:600]

    def get_stats(self) -> Dict[str, Any]:
        return {
            'api_calls': self.api_calls,
            'total_input_tokens': self.total_input_tokens,
            'total_output_tokens': self.total_output_tokens,
            'backend': self.backend,
            'errors': len(self.errors),
            'error_samples': self.errors[:5],
        }


# ═══════════════════════════════════════════════════════════════
# ENCODING EXTRACTOR — Rule-Based (follows SKILL.md encoding philosophy)
#
# This is the "what would Claude remember?" heuristic.
# Follows the encoding table from SKILL.md Step 2a.
# Can be upgraded to LLM-based by passing turns to a subagent.
# ═══════════════════════════════════════════════════════════════

# Patterns that indicate something worth remembering
DECISION_PATTERNS = [
    re.compile(r'\b(we(?:\'ll| will| should| decided| chose|\'re going)\s+(?:to\s+)?(?:use|go with|pick|choose|implement|build|adopt|switch to|keep|remove|replace|add))\b', re.I),
    re.compile(r'\b(decided|decision|let\'s go with|final(?:ly)?|confirmed|approved)\b', re.I),
    re.compile(r'\b(the plan is|the approach is|we\'re doing)\b', re.I),
]

RULE_PATTERNS = [
    re.compile(r'\b(always|never|must|rule:|must always|do not|don\'t ever)\b', re.I),
    re.compile(r'\b(important:|note:|remember:|convention:|standard:|principle:)\b', re.I),
]

CORRECTION_PATTERNS = [
    re.compile(r'\b(actually|no,? (let\'s|we should|change|instead)|wait,? |hold on|that\'s wrong|correction|not like that|change it to)\b', re.I),
    re.compile(r'\b(I said|I told you|I already|we already|go back|revert)\b', re.I),
]

CONCEPT_PATTERNS = [
    re.compile(r'\b(concept|architecture|pattern|framework|approach|strategy|model|system|engine|pipeline|workflow)\b', re.I),
    re.compile(r'\b(component|module|layer|service|adapter|handler|manager)\b', re.I),
]

PERSON_PATTERNS = [
    re.compile(r'\b(Tom|CEO|CTO|founder|user|client|partner|team)\b'),
]

PROJECT_PATTERNS = [
    re.compile(r'\b(Glo|EX\.CO|brain|project|platform|app|product|prototype|demo)\b', re.I),
]

TASK_PATTERNS = [
    re.compile(r'\b(TODO|need to|should|next step|implement|build|create|add|fix|update|refactor|migrate)\b', re.I),
    re.compile(r'\b(task|feature|bug|issue|ticket|work on|working on)\b', re.I),
]

CONTEXT_PATTERNS = [
    re.compile(r'\b(currently|right now|at this point|status|progress|state of|where we are)\b', re.I),
    re.compile(r'\b(session|conversation|discussed|talking about|working on)\b', re.I),
]

EMOTION_PATTERNS = {
    'excitement': re.compile(r'(!{2,}|love it|amazing|perfect|great|awesome|brilliant|exactly)', re.I),
    'frustration': re.compile(r'(again\??|I already|I said|why did you|still wrong|not what I|come on)', re.I),
    'emphasis': re.compile(r'(IMPORTANT|CRITICAL|KEY|MUST|NEVER|ALWAYS)', re.I),
    'satisfaction': re.compile(r'(good|nice|that\'s it|looks good|well done|thank)', re.I),
}


def extract_encoding_decisions(turn: Dict[str, Any], session_context: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Extract what should be remembered from a single conversation turn.

    Follows the SKILL.md encoding philosophy:
      - Decisions → decision nodes (locked)
      - Rules, constraints → rule nodes (locked)
      - Corrections → decision nodes with correction format (locked)
      - New concepts, terms → concept nodes
      - Emotional signals → emotion tags
      - File operations → file nodes
      - Work items → task nodes

    Args:
        turn: Conversation turn dict from transcript_parser
        session_context: Running context (project, recent topics, etc.)

    Returns:
        List of remember() call dicts ready to pass to brain.remember()
    """
    user_msg = turn.get('user_message', '')
    assistant_resp = turn.get('assistant_response', '')
    files = turn.get('files_edited', [])
    combined = user_msg + ' ' + assistant_resp

    encodings = []

    # Detect emotion
    emotion = 0.0
    emotion_label = 'neutral'
    for label, pattern in EMOTION_PATTERNS.items():
        if pattern.search(user_msg):
            emotion = 0.6
            emotion_label = label
            break

    # Check for decisions
    for pattern in DECISION_PATTERNS:
        if pattern.search(combined):
            # Extract the decision content
            title = _extract_title(combined, 'decision')
            if title and len(title) > 15:
                encodings.append({
                    'type': 'decision',
                    'title': title[:200],
                    'content': _make_content(user_msg, assistant_resp, 'decision'),
                    'keywords': _extract_keywords(combined),
                    'locked': True,
                    'emotion': max(emotion, 0.5),
                    'emotion_label': emotion_label if emotion > 0 else 'emphasis',
                    'project': session_context.get('project', ''),
                })
            break  # One decision per turn max

    # Check for corrections (higher priority — override decision)
    for pattern in CORRECTION_PATTERNS:
        if pattern.search(user_msg):
            title = _extract_correction_title(user_msg)
            if title and len(title) > 10:
                encodings.append({
                    'type': 'decision',
                    'title': f'Correction: {title[:180]}',
                    'content': _make_content(user_msg, assistant_resp, 'correction'),
                    'keywords': 'correction ' + _extract_keywords(user_msg),
                    'locked': True,
                    'emotion': max(emotion, 0.7),
                    'emotion_label': 'frustration' if 'frustration' in emotion_label else 'emphasis',
                    'project': session_context.get('project', ''),
                })
            break

    # Check for rules
    for pattern in RULE_PATTERNS:
        if pattern.search(combined):
            title = _extract_title(combined, 'rule')
            if title and len(title) > 10:
                # Don't duplicate if we already have a decision
                if not any(e['type'] == 'decision' for e in encodings):
                    encodings.append({
                        'type': 'rule',
                        'title': f'Rule: {title[:180]}',
                        'content': _make_content(user_msg, assistant_resp, 'rule'),
                        'keywords': _extract_keywords(combined),
                        'locked': True,
                        'emotion': max(emotion, 0.5),
                        'emotion_label': emotion_label if emotion > 0 else 'emphasis',
                        'project': session_context.get('project', ''),
                    })
                break

    # Concepts — architecture terms, technical patterns, new terminology
    if not any(e['type'] in ('decision', 'rule') for e in encodings):
        for pattern in CONCEPT_PATTERNS:
            if pattern.search(combined):
                title = _extract_title(combined, 'concept')
                if title and len(title) > 15:
                    encodings.append({
                        'type': 'concept',
                        'title': title[:200],
                        'content': _make_content(user_msg, assistant_resp, 'concept'),
                        'keywords': _extract_keywords(combined),
                        'locked': False,
                        'emotion': emotion,
                        'emotion_label': emotion_label,
                        'project': session_context.get('project', ''),
                    })
                break

    # Tasks — work items, TODOs, implementation plans
    for pattern in TASK_PATTERNS:
        if pattern.search(user_msg):
            title = _extract_title(user_msg, 'task')
            if title and len(title) > 15:
                if not any(e['type'] == 'task' for e in encodings):
                    encodings.append({
                        'type': 'task',
                        'title': title[:200],
                        'content': _make_content(user_msg, assistant_resp, 'task'),
                        'keywords': _extract_keywords(user_msg),
                        'locked': False,
                        'emotion': emotion,
                        'emotion_label': emotion_label,
                        'project': session_context.get('project', ''),
                    })
            break

    # Context — current state of work, session summaries, progress updates
    for pattern in CONTEXT_PATTERNS:
        if pattern.search(combined):
            title = _extract_title(combined, 'context')
            if title and len(title) > 15:
                if not any(e['type'] == 'context' for e in encodings):
                    encodings.append({
                        'type': 'context',
                        'title': title[:200],
                        'content': _make_content(user_msg, assistant_resp, 'context'),
                        'keywords': _extract_keywords(combined),
                        'locked': False,
                        'emotion': emotion,
                        'emotion_label': emotion_label,
                        'project': session_context.get('project', ''),
                    })
            break

    # Person references — people, roles, relationships
    for pattern in PERSON_PATTERNS:
        if pattern.search(user_msg):
            match = pattern.search(user_msg)
            person_name = match.group(1) if match else 'Unknown'
            title = f'{person_name}: {_extract_title(user_msg, "person")[:180]}'
            if len(title) > 10:
                if not any(e['type'] == 'person' for e in encodings):
                    encodings.append({
                        'type': 'person',
                        'title': title[:200],
                        'content': _make_content(user_msg, assistant_resp, 'person'),
                        'keywords': f'{person_name.lower()} ' + _extract_keywords(user_msg),
                        'locked': False,
                        'emotion': emotion,
                        'emotion_label': emotion_label,
                        'project': session_context.get('project', ''),
                    })
            break

    # Project references — products, platforms, repos
    for pattern in PROJECT_PATTERNS:
        if pattern.search(user_msg):
            match = pattern.search(user_msg)
            project_name = match.group(1) if match else ''
            if project_name:
                session_context['project'] = project_name
            title = _extract_title(user_msg, 'project')
            if title and len(title) > 15:
                if not any(e['type'] == 'project' for e in encodings):
                    encodings.append({
                        'type': 'project',
                        'title': title[:200],
                        'content': _make_content(user_msg, assistant_resp, 'project'),
                        'keywords': f'{project_name.lower()} ' + _extract_keywords(user_msg),
                        'locked': False,
                        'emotion': emotion,
                        'emotion_label': emotion_label,
                        'project': project_name or session_context.get('project', ''),
                    })
            break

    # File edits → file nodes (only for significant new files, cap to avoid bloat)
    seen_files = set()
    for f in files[:2]:
        if f and f not in seen_files and not any(e.get('title') == f for e in encodings):
            seen_files.add(f)
            encodings.append({
                'type': 'file',
                'title': f,
                'content': f'File edited during: {user_msg[:100]}',
                'keywords': f'file {f} ' + _extract_keywords(user_msg),
                'locked': False,
                'emotion': 0.0,
                'emotion_label': 'neutral',
                'project': session_context.get('project', ''),
            })

    return encodings


def _extract_title(text: str, node_type: str) -> str:
    """Extract a concise title from conversation text."""
    # Take the most informative sentence
    sentences = re.split(r'[.!?\n]', text)
    best = ''
    for s in sentences:
        s = s.strip()
        if len(s) > 15 and len(s) > len(best):
            # Prefer sentences with specific terms
            if any(c.isupper() for c in s[1:]) or any(c.isdigit() for c in s):
                best = s
                break
    if not best and sentences:
        best = max(sentences, key=len).strip()
    return best[:200]


def _extract_correction_title(user_msg: str) -> str:
    """Extract what was corrected from user message."""
    # Take the user's correction statement
    sentences = re.split(r'[.!?\n]', user_msg)
    for s in sentences:
        s = s.strip()
        if len(s) > 10:
            return s[:200]
    return user_msg[:200]


def _make_content(user_msg: str, assistant_resp: str, context: str) -> str:
    """Build rich content for a node."""
    parts = []
    if user_msg:
        parts.append(f'User: {user_msg[:500]}')
    if assistant_resp:
        parts.append(f'Context: {assistant_resp[:500]}')
    return '\n'.join(parts)


def _extract_keywords(text: str) -> str:
    """Extract meaningful keywords from text."""
    # Remove common words, keep specific terms
    words = re.findall(r'\b[a-zA-Z][a-zA-Z0-9_\-\.]{2,}\b', text)
    stop = {'the', 'and', 'for', 'that', 'this', 'with', 'from', 'have', 'are', 'was',
            'were', 'been', 'will', 'would', 'could', 'should', 'about', 'into', 'your',
            'their', 'what', 'when', 'where', 'which', 'there', 'also', 'just', 'more',
            'some', 'than', 'then', 'here', 'can', 'does', 'did', 'not', 'but', 'let',
            'its', 'you', 'has', 'had', 'how', 'all', 'each', 'she', 'her', 'him', 'his',
            'they', 'them', 'our', 'out', 'use', 'see', 'now', 'way', 'may', 'say', 'get',
            'make', 'like', 'very', 'after', 'before', 'other', 'being'}

    unique = []
    seen = set()
    for w in words:
        wl = w.lower()
        if wl not in stop and wl not in seen:
            seen.add(wl)
            unique.append(wl)

    return ' '.join(unique[:30])


# ═══════════════════════════════════════════════════════════════
# LLM-BASED ENCODING EXTRACTOR
# ═══════════════════════════════════════════════════════════════

def build_llm_encoding_prompt(turns_batch: List[Dict], session_context: Dict) -> str:
    """
    Build a prompt for an LLM to extract encoding decisions from a batch of turns.

    The LLM follows the SKILL.md encoding philosophy to decide what to remember.
    Returns a structured JSON prompt that can be passed to a subagent.
    """
    turns_text = []
    for t in turns_batch:
        user = t.get('user_message', '')[:300]
        assistant = t.get('assistant_response', '')[:300]
        files = t.get('files_edited', [])
        turns_text.append(f"[{t.get('timestamp', '')[:19]}] User: {user}")
        if assistant:
            turns_text.append(f"  Assistant: {assistant}")
        if files:
            turns_text.append(f"  Files edited: {', '.join(files[:5])}")

    conversation = '\n'.join(turns_text)

    prompt = f"""You are a brain encoding engine for brain. Analyze this conversation excerpt and extract what should be remembered.

Follow these encoding rules (from SKILL.md):
- Decisions, architecture changes → type:"decision", locked:true. EACH specific value as its own node.
- API gotchas, error patterns → type:"rule", locked:true. The specific failure AND the fix.
- User feedback, preferences → type:"rule", locked:true. The SPECIFIC thing said.
- Current state of work → type:"context", locked:false.
- User emotional reactions → type:"context", locked:false. WHAT they reacted to.
- New terms, names, jargon → type:"concept", locked:false.
- Work items → type:"task", locked:false.
- Files → type:"file", locked:false.
- Corrections → type:"decision", title starts with "Correction:", locked:true, emotion:0.7

DECOMPOSE, don't summarize. Each specific value, name, number gets its OWN node.
If unsure, ENCODE IT. Pruning handles cleanup. Missing knowledge is lost forever.

Keywords should include: specific numbers, proper nouns, technical terms, user vocabulary.

Project context: {session_context.get('project', 'unknown')}

CONVERSATION:
{conversation}

Return a JSON array of nodes to remember. Each node:
{{"type": "decision|rule|concept|context|task|file", "title": "concise scannable title with key value", "content": "rich context - WHAT and WHY", "keywords": "specific terms numbers names", "locked": true|false, "emotion": 0.0-1.0, "emotion_label": "emphasis|frustration|excitement|satisfaction|curiosity|neutral"}}

If nothing worth remembering, return [].
Return ONLY valid JSON array, no explanation."""

    return prompt


# ═══════════════════════════════════════════════════════════════
# REPLAY ENGINE
# ═══════════════════════════════════════════════════════════════

class ReplayEngine:
    """
    Replay conversations through a fresh brain, simulating the full hook lifecycle.

    ╔═══════════════════════════════════════════════════════════════════╗
    ║  LOCKED RULE: SIMULATION MUST MATCH PRODUCTION BRAIN BEHAVIOR   ║
    ║                                                                   ║
    ║  Every hook, method call, and lifecycle event in this simulation ║
    ║  MUST use the same Brain API methods as the production skill.    ║
    ║  No regex shortcuts, no workarounds, no simulation-only logic.  ║
    ║                                                                   ║
    ║  When brain.py adds a new method or changes a hook:             ║
    ║    1. Update ReplayEngine to call the same method               ║
    ║    2. Match the same call order and parameters                  ║
    ║    3. Verify with encoding_mode='llm' (production-quality)      ║
    ║                                                                   ║
    ║  The only acceptable difference: encoding decisions come from   ║
    ║  an LLM API call instead of live conversation context, because  ║
    ║  the simulation can't alter what was actually said.             ║
    ╚═══════════════════════════════════════════════════════════════════╝

    Two encoding modes:
      - 'rules': Fast regex-based extraction (for quick iteration ONLY)
      - 'llm': Real Claude API calls (production-identical encoding quality)
    """

    def __init__(self, brain, encoding_mode: str = 'rules',
                 llm_model: str = 'claude-haiku-4-5-20251001',
                 llm_batch_size: int = 8):
        """
        Args:
            brain: Fresh Brain instance to learn into
            encoding_mode: 'rules' for regex, 'llm' for real Claude API calls
            llm_model: Claude model for LLM encoding (haiku=fast/cheap, sonnet=quality)
            llm_batch_size: Turns per LLM API call (higher = fewer calls, more context)
        """
        self.brain = brain
        self.encoding_mode = encoding_mode
        self.session_context = {
            'project': 'Glo',
            'user': 'Tom',
            'session_count': 0,
        }
        self.replay_log = {
            'sessions_replayed': 0,
            'turns_processed': 0,
            'nodes_created': 0,
            'nodes_connected': 0,
            'recalls_performed': 0,
            'suggests_performed': 0,
            'dreams_run': 0,
            'consolidations_run': 0,
            'smart_prunes_run': 0,
            'handoff_notes_written': 0,
            'encoding_decisions': [],
            'recall_log': [],
            'errors': [],
            'timing': {},
            'llm_stats': {},
        }

        # LLM encoder (initialized lazily on first use)
        self.llm_encoder = None
        if encoding_mode == 'llm':
            self.llm_encoder = LLMEncoder(model=llm_model, batch_size=llm_batch_size)

        # Turn buffer for LLM batching
        self._turn_buffer = []
        self._llm_batch_size = llm_batch_size

    def replay_all(self, parsed_transcript: Dict[str, Any],
                   progress_callback=None) -> Dict[str, Any]:
        """
        Replay all sessions from a parsed transcript.

        Args:
            parsed_transcript: Output from transcript_parser.parse_transcript()
            progress_callback: Optional fn(session_idx, total_sessions, turn_idx, total_turns)

        Returns:
            Replay log with detailed results
        """
        sessions = parsed_transcript['sessions']
        total_sessions = len(sessions)

        t0 = time.time()

        for si, session in enumerate(sessions):
            self.session_context['session_count'] += 1
            session_id = f'replay_ses_{si}'

            # ── SessionStart hook ──
            self._boot_session(session_id)

            turns = session.get('turns', [])

            # ── Process all turns: recall/suggest per-turn, encoding in bulk ──
            for ti, turn in enumerate(turns):
                try:
                    self._process_turn(turn, session_id)
                except Exception as e:
                    self.replay_log['errors'].append({
                        'session': si,
                        'turn': ti,
                        'error': str(e),
                        'turn_preview': turn.get('user_message', '')[:100],
                    })

                self.replay_log['turns_processed'] += 1

                if progress_callback:
                    progress_callback(si, total_sessions, ti, len(turns))

            # ── LLM encoding: process entire session in one parallel batch ──
            if self.encoding_mode == 'llm' and self._turn_buffer:
                self._flush_llm_session(session_id)

            # ── End-of-session hooks ──
            self._write_handoff_note(session_id, session)
            self._end_session(session_id)
            self.replay_log['sessions_replayed'] += 1

        self.replay_log['timing']['total_seconds'] = time.time() - t0
        self.replay_log['timing']['avg_per_turn_ms'] = (
            (time.time() - t0) / max(self.replay_log['turns_processed'], 1) * 1000
        )

        # Capture LLM encoder stats
        if self.llm_encoder:
            self.replay_log['llm_stats'] = self.llm_encoder.get_stats()

        return self.replay_log

    def _boot_session(self, session_id: str):
        """Simulate SessionStart hook: reset → context_boot → health_check."""
        try:
            self.brain.reset_session_activity()
            self.brain.context_boot(
                user=self.session_context.get('user', 'Tom'),
                project=self.session_context.get('project', 'Glo'),
                task='replay simulation'
            )
            self.brain.health_check(session_id=session_id, auto_fix=True)
        except Exception as e:
            self.replay_log['errors'].append({
                'phase': 'boot',
                'session_id': session_id,
                'error': str(e),
            })

    def _process_turn(self, turn: Dict[str, Any], session_id: str):
        """
        Process a single conversation turn through the brain.
        Mirrors the real production hook lifecycle:
          1. Pre-response recall (what the brain surfaces for this query)
          2. Pre-edit suggest (what constraints apply before editing)
          3. Encoding (what to remember — regex or LLM)
          4. Remember + connect (store in brain with typed connections)
        """
        user_msg = turn.get('user_message', '')
        files = turn.get('files_edited', [])
        recall_result = None

        # ── 1. Pre-response recall (production: recall_with_embeddings) ──
        if user_msg and len(user_msg) > 10:
            try:
                recall_result = self.brain.recall(user_msg, limit=10, session_id=session_id)
                top_results = [
                    {'id': r['id'], 'title': r.get('title', '')[:60],
                     'score': r.get('effective_activation', 0)}
                    for r in recall_result.get('results', [])[:5]
                ]
                self.replay_log['recall_log'].append({
                    'turn_index': turn.get('index'),
                    'query': user_msg[:100],
                    'results_count': len(recall_result.get('results', [])),
                    'top_results': top_results,
                    'intent': recall_result.get('intent', 'unknown'),
                })
                self.replay_log['recalls_performed'] += 1

                # ── mark_recall_used: top results are assumed "used" ──
                # (strengthens precision signal in production)
                recall_log_id = recall_result.get('_recall_log_id')
                used_ids = [r['id'] for r in recall_result.get('results', [])[:3]]
                if recall_log_id and used_ids and hasattr(self.brain, 'mark_recall_used'):
                    try:
                        self.brain.mark_recall_used(recall_log_id, used_ids)
                    except Exception:
                        pass

            except Exception as e:
                self.replay_log['errors'].append({
                    'phase': 'recall',
                    'turn_index': turn.get('index'),
                    'error': str(e),
                })

        # ── 2. Pre-edit suggest (production: PreToolUse hook fires suggest) ──
        for f in files[:1]:
            try:
                self.brain.suggest(
                    context=user_msg[:200] if user_msg else f'editing {f}',
                    file=f,
                    action='edit'
                )
                self.replay_log['suggests_performed'] += 1
            except Exception:
                pass
            try:
                self.brain.pre_edit(file=f, tool_name='Edit')
            except Exception:
                pass

        # ── 3. Encoding: decide what to remember ──
        if self.encoding_mode == 'rules':
            encodings = extract_encoding_decisions(turn, self.session_context)
            self._store_encodings(encodings, turn, session_id)
        else:
            # LLM mode: buffer turns and flush in batches
            self._turn_buffer.append(turn)
            if len(self._turn_buffer) >= self._llm_batch_size:
                self._flush_llm_buffer(session_id)

    def _flush_llm_buffer(self, session_id: str):
        """Send buffered turns to the LLM encoder (single batch)."""
        if not self._turn_buffer or not self.llm_encoder:
            return

        turns_to_process = self._turn_buffer[:]
        self._turn_buffer.clear()

        # Get existing node titles for connection matching
        existing_titles = self._get_existing_titles()

        encodings = self.llm_encoder.encode_batch(
            turns_to_process,
            self.session_context,
            existing_node_titles=existing_titles
        )

        ref_turn = turns_to_process[0] if turns_to_process else {}
        self._store_encodings(encodings, ref_turn, session_id)

    def _flush_llm_session(self, session_id: str):
        """
        Process an entire session's buffered turns using parallel batch encoding.
        Pre-filters noise, chunks into batches, runs in parallel where possible.
        """
        if not self._turn_buffer or not self.llm_encoder:
            return

        turns_to_process = self._turn_buffer[:]
        self._turn_buffer.clear()

        existing_titles = self._get_existing_titles()

        # Use parallel batch processor
        encodings = self.llm_encoder.encode_batches_parallel(
            turns_to_process,
            self.session_context,
            existing_node_titles=existing_titles
        )

        ref_turn = turns_to_process[0] if turns_to_process else {}
        self._store_encodings(encodings, ref_turn, session_id)

    def _get_existing_titles(self) -> List[str]:
        """Get top node titles from brain for connection matching."""
        try:
            cursor = self.brain.conn.execute(
                'SELECT title FROM nodes WHERE title IS NOT NULL ORDER BY access_count DESC LIMIT 50'
            )
            return [row[0] for row in cursor.fetchall()]
        except Exception:
            return []

    def _store_encodings(self, encodings: List[Dict], ref_turn: Dict, session_id: str):
        """Store encoding decisions in the brain and create connections."""
        for enc in encodings:
            try:
                node_id = self.brain.remember(
                    type=enc.get('type', 'context'),
                    title=enc['title'],
                    content=enc.get('content', ''),
                    keywords=enc.get('keywords', ''),
                    locked=enc.get('locked', False),
                    emotion=float(enc.get('emotion', 0.0)),
                    emotion_label=enc.get('emotion_label', 'neutral'),
                    project=enc.get('project', self.session_context.get('project', '')),
                )
                self.replay_log['nodes_created'] += 1
                self.replay_log['encoding_decisions'].append({
                    'turn_index': ref_turn.get('index'),
                    'type': enc.get('type', 'context'),
                    'title': enc['title'][:80],
                    'locked': enc.get('locked', False),
                })

                # ── Create connections (production: Claude calls /connect) ──
                connections = enc.get('connections', [])
                for conn in connections[:3]:  # Cap at 3 connections per node
                    target_title = conn.get('target_title', '')
                    relation = conn.get('relation', 'related')
                    if not target_title:
                        continue
                    try:
                        # Fuzzy-match target title to existing node
                        target_id = self._resolve_connection_target(target_title)
                        if target_id and node_id:
                            valid_types = {
                                'part_of', 'related', 'corrected_by', 'exemplifies',
                                'depends_on', 'co_accessed', 'produced'
                            }
                            edge_type = relation if relation in valid_types else 'related'
                            # Use connect_typed for typed edges, plain connect for related
                            if hasattr(self.brain, 'connect_typed') and edge_type != 'related':
                                self.brain.connect_typed(
                                    source_id=node_id,
                                    target_id=target_id,
                                    relation=relation,
                                    weight=0.5,
                                    edge_type=edge_type,
                                )
                            else:
                                self.brain.connect(
                                    source_id=node_id,
                                    target_id=target_id,
                                    relation=relation,
                                    weight=0.5,
                                )
                            self.replay_log['nodes_connected'] += 1
                    except Exception:
                        pass

            except Exception as e:
                self.replay_log['errors'].append({
                    'phase': 'remember',
                    'turn_index': ref_turn.get('index'),
                    'error': str(e),
                    'encoding': enc.get('title', '')[:80],
                })

    def _resolve_connection_target(self, target_title: str) -> Optional[str]:
        """Find a node ID by fuzzy-matching a title string."""
        try:
            # Exact match first
            cursor = self.brain.conn.execute(
                'SELECT id FROM nodes WHERE title = ? LIMIT 1', (target_title,)
            )
            row = cursor.fetchone()
            if row:
                return row[0]

            # Fuzzy: LIKE with key words
            words = [w for w in target_title.split() if len(w) > 3][:3]
            if words:
                like_pattern = f'%{words[0]}%'
                cursor = self.brain.conn.execute(
                    'SELECT id FROM nodes WHERE title LIKE ? LIMIT 1', (like_pattern,)
                )
                row = cursor.fetchone()
                if row:
                    return row[0]
        except Exception:
            pass
        return None

    def _write_handoff_note(self, session_id: str, session: Dict):
        """
        Production behavior: Claude writes a session handoff note before ending.
        This context node helps the next session know what happened.
        """
        turns = session.get('turns', [])
        if not turns:
            return

        turn_count = len(turns)
        session_num = self.session_context['session_count']
        first_msg = turns[0].get('user_message', '')[:150] if turns else ''
        last_msg = turns[-1].get('user_message', '')[:150] if turns else ''

        try:
            self.brain.remember(
                type='context',
                title=f'Session Log — Replay #{session_num}',
                content=f'Session #{session_num} ({turn_count} turns). '
                        f'Started with: {first_msg}. '
                        f'Ended with: {last_msg}.',
                keywords=f'session log reset counter replay handoff {session_num}',
                locked=False,
                emotion=0.3,
                emotion_label='curiosity',
                project=self.session_context.get('project', ''),
            )
            self.replay_log['handoff_notes_written'] += 1
        except Exception:
            pass

    def _end_session(self, session_id: str):
        """
        Simulate end-of-session hooks.
        Production order: consolidate → smart-prune → dream → save.
        """
        try:
            self.brain.consolidate()
            self.replay_log['consolidations_run'] += 1
        except Exception as e:
            self.replay_log['errors'].append({
                'phase': 'consolidate', 'error': str(e)
            })

        # smart_prune — production runs this between consolidate and dream
        try:
            if hasattr(self.brain, 'smart_prune'):
                self.brain.smart_prune()
                self.replay_log['smart_prunes_run'] += 1
        except Exception as e:
            self.replay_log['errors'].append({
                'phase': 'smart_prune', 'error': str(e)
            })

        try:
            self.brain.dream(session_id=session_id)
            self.replay_log['dreams_run'] += 1
        except Exception as e:
            self.replay_log['errors'].append({
                'phase': 'dream', 'error': str(e)
            })

        try:
            self.brain.save()
        except Exception as e:
            self.replay_log['errors'].append({
                'phase': 'save', 'error': str(e)
            })


# ═══════════════════════════════════════════════════════════════
# BRAIN COMPARATOR
# ═══════════════════════════════════════════════════════════════

class BrainComparator:
    """
    Compare two brains (relearned vs current) across multiple dimensions.
    """

    def __init__(self, relearned_brain, current_brain):
        self.relearned = relearned_brain
        self.current = current_brain

    def compare(self) -> Dict[str, Any]:
        """Run full comparison between relearned and current brain."""
        return {
            'node_counts': self._compare_node_counts(),
            'locked_analysis': self._compare_locked_nodes(),
            'type_distribution': self._compare_type_distribution(),
            'edge_analysis': self._compare_edges(),
            'recall_comparison': self._compare_recall_quality(),
            'coverage': self._coverage_analysis(),
            'summary': self._generate_summary(),
        }

    def _compare_node_counts(self) -> Dict[str, int]:
        return {
            'relearned_total': self.relearned._get_node_count(),
            'current_total': self.current._get_node_count(),
            'relearned_locked': self.relearned._get_locked_count(),
            'current_locked': self.current._get_locked_count(),
            'relearned_edges': self.relearned._get_edge_count(),
            'current_edges': self.current._get_edge_count(),
        }

    def _compare_locked_nodes(self) -> Dict[str, Any]:
        """Compare locked node titles between brains."""
        relearned_locked = self._get_locked_titles(self.relearned)
        current_locked = self._get_locked_titles(self.current)

        relearned_set = set(relearned_locked.keys())
        current_set = set(current_locked.keys())

        # Fuzzy title matching for overlap
        overlap = 0
        only_relearned = []
        only_current = []

        for title in relearned_set:
            if self._fuzzy_match(title, current_set):
                overlap += 1
            else:
                only_relearned.append(title[:80])

        for title in current_set:
            if not self._fuzzy_match(title, relearned_set):
                only_current.append(title[:80])

        return {
            'relearned_count': len(relearned_set),
            'current_count': len(current_set),
            'approx_overlap': overlap,
            'only_in_relearned': only_relearned[:20],
            'only_in_current': only_current[:20],
        }

    def _compare_type_distribution(self) -> Dict[str, Dict[str, int]]:
        relearned = self._get_type_counts(self.relearned)
        current = self._get_type_counts(self.current)
        all_types = set(list(relearned.keys()) + list(current.keys()))

        return {
            t: {'relearned': relearned.get(t, 0), 'current': current.get(t, 0)}
            for t in sorted(all_types)
        }

    def _compare_edges(self) -> Dict[str, Any]:
        relearned_edges = self._get_edge_type_counts(self.relearned)
        current_edges = self._get_edge_type_counts(self.current)
        return {
            'relearned': relearned_edges,
            'current': current_edges,
        }

    def _compare_recall_quality(self) -> Dict[str, Any]:
        """Run the same golden queries against both brains."""
        golden_path = os.path.join(os.path.dirname(__file__), 'golden_dataset.json')
        if not os.path.exists(golden_path):
            return {'error': 'Golden dataset not found'}

        with open(golden_path) as f:
            test_cases = json.load(f)

        from tests.metrics import compute_all_metrics, aggregate_metrics

        relearned_metrics = []
        current_metrics = []

        for tc in test_cases:
            query = tc.get('query', '')
            expected = tc.get('expected_relevant', {})
            if not expected or not query:
                continue

            # Relearned brain
            try:
                r_result = self.relearned.recall(query, limit=20)
                r_ids = [r['id'] for r in r_result.get('results', [])]
                r_metrics = compute_all_metrics(r_ids, expected, [5, 10])
                relearned_metrics.append(r_metrics)
            except Exception:
                pass

            # Current brain
            try:
                c_result = self.current.recall(query, limit=20)
                c_ids = [r['id'] for r in c_result.get('results', [])]
                c_metrics = compute_all_metrics(c_ids, expected, [5, 10])
                current_metrics.append(c_metrics)
            except Exception:
                pass

        return {
            'relearned_aggregate': aggregate_metrics(relearned_metrics) if relearned_metrics else {},
            'current_aggregate': aggregate_metrics(current_metrics) if current_metrics else {},
            'note': 'Golden dataset was generated from current brain — relearned brain may not have the same node IDs',
        }

    def _coverage_analysis(self) -> Dict[str, Any]:
        """Check what fraction of current brain topics exist in relearned brain."""
        current_titles = self._get_all_titles(self.current)
        relearned_titles = self._get_all_titles(self.relearned)

        covered = 0
        missed = []
        for title in current_titles:
            if self._fuzzy_match(title, set(relearned_titles)):
                covered += 1
            else:
                missed.append(title[:80])

        coverage_pct = covered / max(len(current_titles), 1) * 100

        return {
            'current_nodes': len(current_titles),
            'relearned_nodes': len(relearned_titles),
            'covered': covered,
            'missed': len(missed),
            'coverage_pct': round(coverage_pct, 1),
            'sample_missed': missed[:15],
        }

    def _generate_summary(self) -> Dict[str, str]:
        r_count = self.relearned._get_node_count()
        c_count = self.current._get_node_count()
        r_locked = self.relearned._get_locked_count()
        c_locked = self.current._get_locked_count()

        if r_count > c_count * 1.1:
            quantity = 'Relearned brain captured MORE nodes than current brain'
        elif r_count < c_count * 0.9:
            quantity = 'Relearned brain captured FEWER nodes than current brain'
        else:
            quantity = 'Relearned brain has similar node count to current brain'

        return {
            'quantity': quantity,
            'relearned': f'{r_count} nodes ({r_locked} locked)',
            'current': f'{c_count} nodes ({c_locked} locked)',
        }

    # Helpers

    def _get_locked_titles(self, brain) -> Dict[str, str]:
        cursor = brain.conn.execute('SELECT id, title FROM nodes WHERE locked=1')
        return {row[1]: row[0] for row in cursor.fetchall() if row[1]}

    def _get_type_counts(self, brain) -> Dict[str, int]:
        cursor = brain.conn.execute('SELECT type, COUNT(*) FROM nodes GROUP BY type')
        return {row[0]: row[1] for row in cursor.fetchall()}

    def _get_edge_type_counts(self, brain) -> Dict[str, int]:
        cursor = brain.conn.execute('SELECT edge_type, COUNT(*) FROM edges GROUP BY edge_type')
        return {row[0]: row[1] for row in cursor.fetchall()}

    def _get_all_titles(self, brain) -> List[str]:
        cursor = brain.conn.execute('SELECT title FROM nodes WHERE title IS NOT NULL')
        return [row[0] for row in cursor.fetchall()]

    def _fuzzy_match(self, title: str, title_set: set) -> bool:
        """Check if title fuzzy-matches any title in the set."""
        if title in title_set:
            return True
        # Check if key words overlap (3+ word match)
        title_words = set(title.lower().split())
        for other in title_set:
            other_words = set(other.lower().split())
            overlap = title_words & other_words
            if len(overlap) >= 3:
                return True
        return False


# ═══════════════════════════════════════════════════════════════
# TERMINAL REPORTER
# ═══════════════════════════════════════════════════════════════

def print_replay_report(replay_log: Dict, comparison: Dict = None):
    """Print human-readable replay report."""
    print()
    print('=' * 70)
    print('  brain RELEARNING COMPARISON REPORT')
    print('=' * 70)
    print()

    print(f'  Sessions replayed: {replay_log["sessions_replayed"]}')
    print(f'  Turns processed:   {replay_log["turns_processed"]}')
    print(f'  Nodes created:     {replay_log["nodes_created"]}')
    print(f'  Recalls performed: {replay_log["recalls_performed"]}')
    print(f'  Dreams run:        {replay_log["dreams_run"]}')
    print(f'  Consolidations:    {replay_log["consolidations_run"]}')
    print(f'  Errors:            {len(replay_log["errors"])}')
    timing = replay_log.get('timing', {})
    print(f'  Total time:        {timing.get("total_seconds", 0):.1f}s')
    print(f'  Avg per turn:      {timing.get("avg_per_turn_ms", 0):.1f}ms')
    print()

    if replay_log['errors']:
        print('  ─── Errors (first 10) ───')
        for err in replay_log['errors'][:10]:
            phase = err.get('phase', 'unknown')
            error = err.get('error', '')[:80]
            print(f'    [{phase}] {error}')
        print()

    # Encoding breakdown
    enc_types = {}
    for e in replay_log.get('encoding_decisions', []):
        t = e.get('type', 'unknown')
        enc_types[t] = enc_types.get(t, 0) + 1

    if enc_types:
        print('  ─── Encoding Decisions by Type ───')
        for t, c in sorted(enc_types.items(), key=lambda x: -x[1]):
            print(f'    {t:>12s}: {c}')
        print()

    if comparison:
        print()
        print('  ─── Brain Comparison ───')
        counts = comparison.get('node_counts', {})
        print(f'    Relearned:  {counts.get("relearned_total", 0)} nodes ({counts.get("relearned_locked", 0)} locked, {counts.get("relearned_edges", 0)} edges)')
        print(f'    Current:    {counts.get("current_total", 0)} nodes ({counts.get("current_locked", 0)} locked, {counts.get("current_edges", 0)} edges)')

        print()
        types = comparison.get('type_distribution', {})
        if types:
            print('    Type Distribution:')
            print(f'      {"Type":>12s}  {"Relearned":>10s}  {"Current":>10s}')
            for t, d in types.items():
                print(f'      {t:>12s}  {d["relearned"]:>10d}  {d["current"]:>10d}')

        print()
        coverage = comparison.get('coverage', {})
        if coverage:
            print(f'    Coverage: {coverage.get("coverage_pct", 0)}% of current brain topics found in relearned')
            print(f'    Covered: {coverage.get("covered", 0)} / {coverage.get("current_nodes", 0)}')
            missed = coverage.get('sample_missed', [])
            if missed:
                print(f'    Sample missed ({len(missed)}):')
                for m in missed[:10]:
                    print(f'      - {m}')

        print()
        locked = comparison.get('locked_analysis', {})
        if locked:
            print(f'    Locked overlap: ~{locked.get("approx_overlap", 0)}')
            only_r = locked.get('only_in_relearned', [])
            only_c = locked.get('only_in_current', [])
            if only_r:
                print(f'    Only in relearned ({len(only_r)}):')
                for t in only_r[:5]:
                    print(f'      + {t}')
            if only_c:
                print(f'    Only in current ({len(only_c)}):')
                for t in only_c[:5]:
                    print(f'      - {t}')

        summary = comparison.get('summary', {})
        if summary:
            print()
            print(f'    {summary.get("quantity", "")}')

    print()
    print('=' * 70)
    print()


# ═══════════════════════════════════════════════════════════════
# MAIN ENTRY POINT
# ═══════════════════════════════════════════════════════════════

def run_relearning(transcript_path: str, current_brain_path: str = None,
                   output_brain_path: str = None,
                   encoding_mode: str = 'rules',
                   llm_model: str = 'claude-haiku-4-5-20251001',
                   llm_batch_size: int = 8) -> Dict[str, Any]:
    """
    Run the full relearning simulation.

    Args:
        transcript_path: Path to .jsonl transcript
        current_brain_path: Path to current brain.db for comparison
        output_brain_path: Where to save the relearned brain.db
        encoding_mode: 'rules' for fast regex, 'llm' for real Claude API calls
        llm_model: Claude model for LLM encoding
        llm_batch_size: Turns per API call

    Returns:
        Dict with replay_log, comparison, and paths
    """
    from servers.brain import Brain

    # Set default output path
    if not output_brain_path:
        output_brain_path = os.path.join(
            os.path.dirname(__file__), 'relearned_brain.db'
        )

    # Remove old relearned brain if exists
    if os.path.exists(output_brain_path):
        os.remove(output_brain_path)

    print(f'[relearning] Parsing transcript: {transcript_path}')
    parsed = parse_transcript(transcript_path)
    print(f'[relearning] Found {parsed["stats"]["total_turns"]} turns across {parsed["stats"]["total_sessions"]} sessions')
    print(f'[relearning] Date range: {parsed["stats"]["date_range"]["start"]} → {parsed["stats"]["date_range"]["end"]}')

    # Create fresh brain
    print(f'[relearning] Creating fresh brain: {output_brain_path}')
    fresh_brain = Brain(output_brain_path)

    # Replay
    engine = ReplayEngine(
        fresh_brain,
        encoding_mode=encoding_mode,
        llm_model=llm_model,
        llm_batch_size=llm_batch_size,
    )

    mode_label = encoding_mode
    if encoding_mode == 'llm':
        mode_label = f'llm ({llm_model}, batch={llm_batch_size})'
    print(f'[relearning] Starting replay (encoding: {mode_label})...')

    def progress(si, st, ti, tt):
        if encoding_mode == 'llm':
            # More frequent progress for slow LLM mode
            if ti % 20 == 0 and ti > 0:
                stats = engine.llm_encoder.get_stats() if engine.llm_encoder else {}
                api_calls = stats.get('api_calls', 0)
                print(f'  Session {si+1}/{st}, turn {ti}/{tt} '
                      f'({engine.replay_log["nodes_created"]} nodes, '
                      f'{api_calls} API calls)')
        else:
            if ti % 100 == 0 and ti > 0:
                print(f'  Session {si+1}/{st}, turn {ti}/{tt} '
                      f'({engine.replay_log["nodes_created"]} nodes so far)')

    replay_log = engine.replay_all(parsed, progress_callback=progress)

    print(f'[relearning] Replay complete: {replay_log["nodes_created"]} nodes created, '
          f'{replay_log["nodes_connected"]} connections made')

    if replay_log.get('llm_stats'):
        stats = replay_log['llm_stats']
        print(f'[relearning] LLM stats: {stats.get("api_calls", 0)} API calls, '
              f'{stats.get("total_input_tokens", 0)} input tokens, '
              f'{stats.get("total_output_tokens", 0)} output tokens')
        if stats.get('errors'):
            print(f'[relearning] LLM errors: {stats["errors"]}')

    # Save relearned brain
    fresh_brain.save()

    # Compare if current brain provided
    comparison = None
    if current_brain_path and os.path.exists(current_brain_path):
        print(f'[relearning] Comparing against current brain: {current_brain_path}')
        current_brain = Brain(current_brain_path)
        comparator = BrainComparator(fresh_brain, current_brain)
        comparison = comparator.compare()
        current_brain.close()

    fresh_brain.close()

    # Print report
    print_replay_report(replay_log, comparison)

    # Save results
    results_path = os.path.join(os.path.dirname(__file__), 'results', 'relearning_report.json')
    os.makedirs(os.path.dirname(results_path), exist_ok=True)

    report = {
        'replay_log': {k: v for k, v in replay_log.items()
                       if k not in ('recall_log', 'encoding_decisions')},
        'encoding_summary': {
            'total': len(replay_log.get('encoding_decisions', [])),
            'by_type': {},
        },
        'comparison': comparison,
        'paths': {
            'transcript': transcript_path,
            'relearned_brain': output_brain_path,
            'current_brain': current_brain_path,
        },
    }

    # Encoding type breakdown
    for e in replay_log.get('encoding_decisions', []):
        t = e.get('type', 'unknown')
        report['encoding_summary']['by_type'][t] = report['encoding_summary']['by_type'].get(t, 0) + 1

    with open(results_path, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    print(f'[relearning] Report saved: {results_path}')
    print(f'[relearning] Relearned brain saved: {output_brain_path}')

    return report


if __name__ == '__main__':
    args = sys.argv[1:]

    transcript = None
    current_brain = None
    output_brain = None
    mode = 'rules'
    model = 'claude-haiku-4-5-20251001'
    batch = 8

    i = 0
    while i < len(args):
        arg = args[i]
        if arg == '--llm':
            mode = 'llm'
        elif arg == '--model' and i + 1 < len(args):
            i += 1
            model = args[i]
        elif arg == '--batch' and i + 1 < len(args):
            i += 1
            batch = int(args[i])
        elif arg.endswith('.jsonl'):
            transcript = arg
        elif arg.endswith('.db') and 'relearned' not in arg:
            current_brain = arg
        elif arg.endswith('.db'):
            output_brain = arg
        i += 1

    if not transcript:
        print('Usage: python tests/relearning.py <transcript.jsonl> [current_brain.db] [output_brain.db]')
        print('       [--llm] [--model claude-haiku-4-5-20251001] [--batch 8]')
        sys.exit(1)

    run_relearning(transcript, current_brain, output_brain,
                   encoding_mode=mode, llm_model=model, llm_batch_size=batch)
