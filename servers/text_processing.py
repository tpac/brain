"""
brain — Text Processing Utilities

Shared module for text processing functions used across the brain:
- split_identifier(): Convert programming identifiers to search tokens
- split_sentences(): Sentence boundary detection (pySBD + fallback)
- is_domain_specific(): Check if a term is domain-specific vs common English
- filter_domain_terms(): Filter a list of candidates to domain-specific only

Each function is independently usable. No heavy dependencies required —
pySBD is optional (falls back to regex), common-word list is lazy-loaded.
"""

import os
import re
from typing import List, Optional, Set


# ── Identifier Splitting ─────────────────────────────────────────────

# Known code file extensions to strip
_EXT_RE = re.compile(
    r'\.(?:py|js|ts|tsx|jsx|sh|md|txt|json|yaml|yml|toml|cfg|db|sql|'
    r'html|css|scss|less|xml|csv|ini|env|lock|log|bak|tmp|gz|zip)$',
    re.IGNORECASE
)

# Version patterns to protect: v5.3.1, 2.0.1, V2, etc.
_VERSION_RE = re.compile(r'\bv?\d+(?:\.\d+)+\b', re.IGNORECASE)

# Core token pattern — order matters (earlier alternatives tried first):
#   1. [A-Z]+(?=[A-Z][a-z])  — acronym run before TitleCase: "HTML" in "HTMLParser"
#   2. [A-Z][a-z]+            — TitleCase word: "Parser", "Auth"
#   3. [A-Z]+                 — trailing/standalone acronym: "HTML" at end
#   4. [a-z]+                 — lowercase word: "parse", "get"
#   5. \d+                    — number run: "2" in "sha256"
_TOKEN_RE = re.compile(
    r'[A-Z]+(?=[A-Z][a-z])'   # acronym before Title word
    r'|[A-Z][a-z]+'            # Title word
    r'|[A-Z]+'                 # trailing/standalone acronym
    r'|[a-z]+'                 # lowercase word
    r'|\d+'                    # numbers
)


def split_identifier(name: str) -> List[str]:
    """Split a programming identifier into lowercase tokens for search.

    Handles camelCase, PascalCase, snake_case, kebab-case, file paths,
    file extensions, version numbers, and acronyms.

    Examples:
        >>> split_identifier('parseHTMLDoc')
        ['parse', 'html', 'doc']
        >>> split_identifier('getURLParser')
        ['get', 'url', 'parser']
        >>> split_identifier('pre_response_recall.py')
        ['pre', 'response', 'recall']
        >>> split_identifier('v2.3.1-beta')
        ['v2.3.1', 'beta']
        >>> split_identifier('servers/daemon_hooks.py')
        ['servers', 'daemon', 'hooks']
    """
    if not name or not name.strip():
        return []

    # Strip file extension
    name = _EXT_RE.sub('', name)

    # Protect version numbers by replacing with placeholders
    versions = {}
    def _protect_version(m):
        key = '\x00V%d\x00' % len(versions)
        versions[key] = m.group(0)
        return key
    name = _VERSION_RE.sub(_protect_version, name)

    # Split on separators: underscores, hyphens, dots, slashes
    parts = re.split(r'[_\-./\\]+', name)

    # Tokenize each part using the camelCase-aware pattern
    tokens = []
    for part in parts:
        part = part.strip()
        if not part:
            continue
        if part in versions:
            tokens.append(versions[part])
        else:
            found = _TOKEN_RE.findall(part)
            tokens.extend(found)

    # Lowercase and filter single-char alphabetic noise (keep digits)
    result = []
    for t in tokens:
        if t in versions.values():
            result.append(t.lower())
        elif t.isdigit():
            result.append(t)  # Always keep digits (e.g., "8" in "UTF8")
        elif len(t) > 1:
            result.append(t.lower())

    return result


# ── Common-Word Filter ────────────────────────────────────────────────

_COMMON_WORDS: Optional[Set[str]] = None

# Acronyms that are NOT domain-specific (common abbreviations)
_COMMON_ACRONYMS = frozenset({
    'I', 'OK', 'AM', 'PM', 'US', 'UK', 'EU', 'TV', 'ID', 'VS',
    'AD', 'BC', 'CEO', 'CFO', 'CTO', 'COO', 'VP', 'HR', 'PR',
    'FAQ', 'FYI', 'ASAP', 'ETA', 'DIY', 'RIP', 'AKA',
})


def load_common_words() -> Set[str]:
    """Load 10K common English words. Cached after first call.

    Source: Google Trillion Word Corpus via first20hours/google-10000-english
    License: MIT (Norvig's contributions) + LDC educational use
    """
    global _COMMON_WORDS
    if _COMMON_WORDS is not None:
        return _COMMON_WORDS

    word_file = os.path.join(os.path.dirname(__file__), '..', 'data', 'common_words_10k.txt')
    try:
        with open(word_file, 'r') as f:
            _COMMON_WORDS = frozenset(line.strip().lower() for line in f if line.strip())
    except FileNotFoundError:
        # Fallback: minimal set so the system still works without the file
        _COMMON_WORDS = frozenset()

    return _COMMON_WORDS


def is_domain_specific(term: str) -> bool:
    """Check if a term is domain-specific (not common English).

    Logic:
    - Acronyms (2-6 uppercase letters) are domain-specific unless in common set
    - Single words: domain-specific if NOT in 10K common words
    - Multi-word: domain-specific if head word (last word) is NOT common,
      OR if any non-head word is also not common

    Examples:
        >>> is_domain_specific('webhook')
        True
        >>> is_domain_specific('file')
        False
        >>> is_domain_specific('recall scorer')
        True
        >>> is_domain_specific('DAL')
        True
    """
    if not term or not term.strip():
        return False

    term = term.strip()
    common = load_common_words()

    # Empty word list = can't filter, pass everything through
    if not common:
        return True

    # Acronym check: all uppercase, 2-6 chars
    if term.isupper() and 2 <= len(term) <= 6:
        return term not in _COMMON_ACRONYMS

    # Capitalized proper noun: "Clerk", "Redis", "Valinor"
    # If the first letter is uppercase and it's not at sentence start,
    # treat as domain-specific (product name / entity)
    if term[0].isupper() and not term.isupper():
        return True

    words = term.lower().split()
    if not words:
        return False

    # Single word: just check the list
    if len(words) == 1:
        return words[0] not in common

    # Multi-word: check head word (last word — English is head-final)
    head = words[-1]
    if head not in common:
        return True

    # Head is common, but check if any modifier is domain-specific
    # "brain daemon" — "brain" is common, "daemon" is NOT → domain-specific
    for w in words[:-1]:
        if w not in common:
            return True

    # Both head and all modifiers are common individually, but the BIGRAM
    # might be domain-specific. "supply adapter", "hook chain", "precision loop"
    # are compound technical terms where common words combine into jargon.
    # Heuristic: if 2+ words and the full phrase is not a trivially common
    # collocation, lean towards domain-specific.
    if len(words) >= 2:
        # Trivially common collocations (adjective + noun, determiner + noun)
        _TRIVIAL_MODIFIERS = frozenset({
            'the', 'a', 'an', 'this', 'that', 'my', 'your', 'our', 'their',
            'his', 'her', 'its', 'some', 'any', 'each', 'every', 'all',
            'good', 'bad', 'new', 'old', 'big', 'small', 'great', 'last',
            'first', 'next', 'other', 'same', 'different', 'right', 'wrong',
            'best', 'most', 'more', 'very', 'real', 'main', 'full',
        })
        # If ALL modifiers are trivial, it's a common phrase
        if all(w in _TRIVIAL_MODIFIERS for w in words[:-1]):
            return False
        # Otherwise: two common words combining = likely domain compound
        return True

    return False


def filter_domain_terms(candidates: List[str]) -> List[str]:
    """Filter a list of candidate terms to only domain-specific ones.

    Preserves order. Removes duplicates (case-insensitive).

    Examples:
        >>> filter_domain_terms(['the file', 'recall scorer', 'webhook', 'DAL'])
        ['recall scorer', 'webhook', 'DAL']
    """
    seen = set()
    result = []
    for term in candidates:
        key = term.lower().strip()
        if key in seen or not key:
            continue
        seen.add(key)
        if is_domain_specific(term):
            result.append(term)
    return result
