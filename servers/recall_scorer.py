"""
brain — Recall Precision Scorer

Pure computation module for evaluating whether recalled brain nodes were
useful in a conversation. No database access, no Brain dependency.

Extracted from sim_combined_v3.py after 10 experiments validated the approach.
See docs/PRECISION-EVAL-RESULTS.md for the full evaluation history.

== THREE SIGNAL SOURCES ==

Layer 0: Expanded regex (30+ patterns) — intent markers
Layer 1: Arctic embeddings (already loaded by brain) — topic similarity
Layer 1b: BART-Large-MNLI zero-shot — stance detection
Layer 2: Cross-signal patterns — regex+BART combos, parroting, polite dismiss

== GRACEFUL DEGRADATION ==

If BART unavailable → regex + embeddings only.
If embedder also down → regex only.
match_method column records which layers were active.

== THE ONE CASE WE CAN'T HANDLE ==

Semantic contradiction (user advocates anti-pattern the node warns against)
requires LLM. Deferred to future Haiku integration (~$0.0001/call).
"""

import re
from typing import Any, Dict, List, Optional, Tuple

# ══════════════════════════════════════════════════════════════════════
# BART SINGLETON (mirrors embedder.py pattern)
# ══════════════════════════════════════════════════════════════════════

_bart_pipeline = None
_bart_loading = False

BART_LABELS = [
    "agreement", "disagreement", "redirect",
    "building on the idea", "topic change",
]


def load_bart():
    """Load BART-Large-MNLI for zero-shot classification. Safe to call multiple times."""
    global _bart_pipeline, _bart_loading
    if _bart_pipeline is not None or _bart_loading:
        return
    _bart_loading = True
    try:
        import os
        os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
        os.environ.setdefault('TRANSFORMERS_VERBOSITY', 'error')
        import warnings
        warnings.filterwarnings('ignore', category=FutureWarning)
        from transformers import pipeline
        _bart_pipeline = pipeline(
            "zero-shot-classification",
            model="facebook/bart-large-mnli",
        )
    except Exception:
        _bart_pipeline = None
    finally:
        _bart_loading = False


def is_bart_ready() -> bool:
    return _bart_pipeline is not None


# ══════════════════════════════════════════════════════════════════════
# EXPANDED REGEX PATTERNS
# ══════════════════════════════════════════════════════════════════════

# --- Affirmation: 30+ forms of agreement in English ---
AFFIRM_STRONG = [
    r"\b(perfect|exactly|yes|yeah|yep|absolutely|precisely)\b[,.:;!—–-]?\s*(and|so|now|how|that|this|the|we|it|i|but|should|let|can)",
    r"\bthat'?s?\s+(exactly|right|correct|perfect|it|true|fair|a\s+good|a\s+great|a\s+fair|the\s+right|solid|an?\s+excellent)",
    r"\b(good|great|excellent|solid|strong|fair|valid|important|interesting)\s+(point|call|idea|framing|insight|observation|catch|question|take|approach|analysis|thinking|reasoning)",
    r"\bmakes?\s+sense\b",
    r"\b(i\s+agree|i\s+concur|i\s+think\s+so\s+too|i\s+see\s+it\s+the\s+same)\b",
    r"\byou'?r?e?\s+(right|correct|absolutely right|spot\s+on)\b",
    r"\byou\s+(have|make|raise)\s+a\s+(good\s+)?(point|fair\s+point)\b",
    r"\b(maps?\s+to|aligns?\s+with|consistent\s+with|corresponds?\s+(to|with)|resonates?)\b",
    r"\b(that|this|it)\s+(maps?|aligns?|tracks?|checks?\s+out|adds?\s+up)\b",
]

AFFIRM_MODERATE = [
    r"^(right|sure|indeed|of\s+course|true|fair\s+enough|ok)\b[,.:;!—–-]",
    r"\bi\s+see\s+(what|where|how|your)\b",
    r"\bthat\s+(works|helps|clarifies)\b",
]

AFFIRM_BUILDING = [
    r"\b(and|so)\s+(since|because|given|if)\s+(we|that|this)\b",
    r"\bhow\s+(do|should|would|can|does)\s+we\b",
    r"\bshould\s+we\s+(also|add|consider|include|extend|apply)\b",
    r"\bsame\s+(pattern|approach|way|logic|principle|idea|reasoning)\b",
    r"\bcan\s+we\s+(extend|apply|use|build|take)\b",
    r"\bwhat\s+about\s+(also|adding|extending|applying)\b",
    r"\bbuilding\s+on\s+(that|this)\b",
]

# --- Redirect: user wants to change topic ---
REDIRECT_STRONG = [
    r"\bnot\s+what\s+(i\s+)?(asked|meant|want|need|was\s+asking)\b",
    r"\bwhy\s+(did|are|do)\s+you\s+(bring|mention|surface|connect|discuss)\b",
    r"\bnot\s+(related|relevant|what\s+i)\b",
    r"\birrelevant\b",
    r"\bcompletely\s+different\b",
    r"\bshouldn'?t\s+be\b.{0,30}\btogether\b",
    r"\bnothing\s+to\s+do\s+with\b",
    r"\bwhat\s+does\s+.{0,30}\bhave\s+to\s+do\s+with\b",
    r"\bdon'?t\s+(see|understand)\s+(the\s+)?(connection|relevance|relation)\b",
]

REDIRECT_MILD = [
    r"\bactually[,.:;—–-]?\s*(i|we|let|can|what)\b",
    r"\blet'?s?\s+(move|switch|focus|talk|get\s+back|go\s+back)\b",
    r"\bcan\s+we\s+(just|instead|focus|move|get\s+back|look\s+at|talk)\b",
    r"\bbut\s+what\s+(i|we)\s+actually\b",
    r"\bwhat\s+i\s+(actually\s+)?(need|want|meant|asked)\b",
    r"\binstead[,.]?\s+(can|let|how|what|i)\b",
    r"\b(back\s+to|getting\s+back\s+to|returning\s+to)\b",
    r"\bforget\s+(about\s+)?(that|this|it)\b",
]

# --- Complaint: user objects to recall itself ---
COMPLAINT = [
    r"\bwhy\s+(did|are)\s+you\b",
    r"\bdon'?t\s+need\b.{0,30}\b(repeated|again|back|reminded|told)\b",
    r"\bnot\s+(relevant|useful|helpful|needed|necessary)\b",
    r"\bi\s+(already\s+)?know\s+(that|this|about)\b",
    r"\bstop\s+(bringing|mentioning|repeating|surfacing)\b",
    r"\bwho\s+asked\b",
]

# --- Extension: user extends recalled content ---
EXTENSION = [
    r"\bshould\s+we\s+(also|add|consider|extend|apply)\b",
    r"\bsame\s+(pattern|approach|way|logic|principle)\b",
    r"\bsimilar\s+(detection|pattern|approach|mechanism|logic)\b",
    r"\bcan\s+we\s+(extend|apply|reuse|generalize)\b",
    r"\bwhat\s+about\s+(applying|using|extending)\b",
    r"\bfollowing\s+(the\s+same|that|this)\s+(pattern|approach|logic)\b",
]


def compute_regex_signals(followup: str) -> Dict[str, Any]:
    """Compute all regex-based signals from followup text."""
    fup_lower = followup.lower()
    s = {}

    s["affirm_strong"] = sum(1 for p in AFFIRM_STRONG if re.search(p, fup_lower))
    s["affirm_moderate"] = sum(1 for p in AFFIRM_MODERATE if re.search(p, fup_lower))
    s["affirm_building"] = sum(1 for p in AFFIRM_BUILDING if re.search(p, fup_lower))
    s["redirect_strong"] = sum(1 for p in REDIRECT_STRONG if re.search(p, fup_lower))
    s["redirect_mild"] = sum(1 for p in REDIRECT_MILD if re.search(p, fup_lower))
    s["complaint"] = sum(1 for p in COMPLAINT if re.search(p, fup_lower))
    s["extension"] = sum(1 for p in EXTENSION if re.search(p, fup_lower))

    first_word = fup_lower.split()[0] if fup_lower.strip() else ""
    first_clean = re.sub(r'[,.:;!?—–-]', '', first_word)
    s["fup_opens_positive"] = first_clean in (
        "yes", "yeah", "yep", "exactly", "perfect", "good", "right",
        "great", "sure", "absolutely", "correct", "true", "agreed",
    )
    s["fup_opens_negative"] = first_clean in (
        "no", "nope", "not", "wrong", "why", "actually", "stop", "don't", "dont",
    )
    s["fup_opens_transition"] = first_clean in (
        "but", "however", "although", "so", "and", "ok", "okay", "well", "hmm",
    )
    s["fup_has_question"] = "?" in followup

    return s


def compute_bart_signals(followup: str) -> Dict[str, float]:
    """BART zero-shot stance detection: whole-text + per-sentence, take strongest.

    Returns zero-value dict if BART not loaded (graceful degradation).
    """
    if not is_bart_ready():
        return _empty_bart()

    whole_scores = _bart_classify(followup)

    sents = [s.strip() for s in re.split(r'[.!?]+', followup) if len(s.strip()) > 3]
    max_sent_agree = 0.0
    max_sent_disagree = 0.0
    max_sent_redirect = 0.0
    max_sent_build = 0.0
    max_sent_topic_change = 0.0

    for sent in sents:
        ss = _bart_classify(sent)
        max_sent_agree = max(max_sent_agree, ss.get("agreement", 0))
        max_sent_disagree = max(max_sent_disagree, ss.get("disagreement", 0))
        max_sent_redirect = max(max_sent_redirect, ss.get("redirect", 0))
        max_sent_build = max(max_sent_build, ss.get("building on the idea", 0))
        max_sent_topic_change = max(max_sent_topic_change, ss.get("topic change", 0))

    agree = max(whole_scores.get("agreement", 0), max_sent_agree)
    build = max(whole_scores.get("building on the idea", 0), max_sent_build)
    disagree = max(whole_scores.get("disagreement", 0), max_sent_disagree)
    redirect = max(whole_scores.get("redirect", 0), max_sent_redirect)
    topic_change = max(whole_scores.get("topic change", 0), max_sent_topic_change)

    pos_signal = max(agree, build)
    neg_signal = max(disagree, redirect, topic_change)

    return {
        "bart_agree": agree,
        "bart_build": build,
        "bart_disagree": disagree,
        "bart_redirect": redirect,
        "bart_topic_change": topic_change,
        "bart_pos": pos_signal,
        "bart_neg": neg_signal,
        "bart_delta": pos_signal - neg_signal,
    }


def compute_embedding_signals(
    recalled_snippets: Dict[str, str],
    recalled_titles: Dict[str, str],
    response: str,
    followup: str,
) -> Dict[str, Any]:
    """Compute embedding-based topic similarity signals.

    Returns zero-value dict if embedder not ready (graceful degradation).
    """
    try:
        from servers import embedder
        if not embedder.is_ready():
            return _empty_emb()
    except Exception:
        return _empty_emb()

    recalled_text = " ".join(
        "%s %s" % (recalled_titles.get(nid, ""), snippet)
        for nid, snippet in recalled_snippets.items()
    )
    recalled_vec = embedder.embed(recalled_text)
    resp_vec = embedder.embed(response)
    fup_vec = embedder.embed(followup) if followup else None

    s = {}
    s["resp_words"] = len(response.split())
    recalled_len = sum(len(v) for v in recalled_snippets.values())
    s["depth_ratio"] = len(response) / max(recalled_len, 1)

    per_node_resp = {}
    per_node_fup = {}
    for nid, snippet in recalled_snippets.items():
        ntext = "%s: %s" % (recalled_titles.get(nid, ""), snippet)
        nvec = embedder.embed(ntext)
        per_node_resp[nid] = embedder.cosine_similarity(nvec, resp_vec) if nvec and resp_vec else 0.0
        if fup_vec:
            per_node_fup[nid] = embedder.cosine_similarity(nvec, fup_vec) if nvec else 0.0

    s["max_resp_sim"] = max(per_node_resp.values()) if per_node_resp else 0.0
    s["max_fup_sim"] = max(per_node_fup.values()) if per_node_fup else 0.0
    s["sim_delta"] = s["max_fup_sim"] - s["max_resp_sim"]

    resp_sents = [t.strip() for t in re.split(r'[.!?]+', response) if len(t.strip()) > 15]
    fup_sents = [t.strip() for t in re.split(r'[.!?]+', followup) if len(t.strip()) > 15]
    resp_on = sum(
        1 for t in resp_sents
        if embedder.cosine_similarity(recalled_vec, embedder.embed(t)) > 0.5
    )
    fup_on = sum(
        1 for t in fup_sents
        if embedder.cosine_similarity(recalled_vec, embedder.embed(t)) > 0.5
    )
    s["resp_on_topic_pct"] = resp_on / max(len(resp_sents), 1)
    s["fup_on_topic_pct"] = fup_on / max(len(fup_sents), 1)
    s["fup_sent_count"] = len(fup_sents)

    return s


def score_recall(
    regex: Dict[str, Any],
    bart_s: Dict[str, float],
    emb_s: Dict[str, Any],
) -> Tuple[float, float, List[Tuple[float, str, str]], int]:
    """Combined scoring using all available signal sources.

    Returns (evidence, confidence, reasons, n_signals).
    evidence: float in [-1, +1], positive = useful recall
    confidence: float in [0, 1], how sure we are
    reasons: list of (weight, reason_string, "+"/"-"/"?")
    n_signals: total signals detected
    """
    pos = []  # (weight, reason)
    neg = []
    unc = []

    # ── LAYER 0: Regex (intent markers) ──

    dr = emb_s.get("depth_ratio", 0)
    if dr > 0.3:
        pos.append((0.15, "L0: deep response (%.2f)" % dr))
    elif dr < 0.05:
        neg.append((0.25, "L0: near-empty response"))
    elif dr < 0.15:
        neg.append((0.08, "L0: shallow response"))

    if emb_s.get("resp_words", 0) < 5:
        neg.append((0.30, "L0: ultra-short response"))

    if regex["affirm_strong"] > 0:
        pos.append((0.25, "L0: strong affirm (%d)" % regex["affirm_strong"]))
    if regex["affirm_moderate"] > 0:
        pos.append((0.10, "L0: moderate affirm (%d)" % regex["affirm_moderate"]))
    if regex["affirm_building"] > 0:
        pos.append((0.15, "L0: building language (%d)" % regex["affirm_building"]))

    if regex["redirect_strong"] > 0:
        neg.append((0.30, "L0: strong redirect (%d)" % regex["redirect_strong"]))
    if regex["redirect_mild"] > 0 and regex["redirect_strong"] == 0:
        neg.append((0.12, "L0: mild redirect (%d)" % regex["redirect_mild"]))

    if regex["complaint"] > 0:
        neg.append((0.25, "L0: complaint (%d)" % regex["complaint"]))

    if regex["extension"] > 0:
        pos.append((0.20, "L0: extends content (%d)" % regex["extension"]))

    if regex["fup_opens_positive"]:
        pos.append((0.12, "L0: opens positive"))
    elif regex["fup_opens_negative"]:
        neg.append((0.12, "L0: opens negative"))
    elif regex["fup_opens_transition"]:
        unc.append((0.05, "L0: opens transition"))

    # ── LAYER 1: Embeddings (topic) ──

    if emb_s.get("max_resp_sim", 0) > 0.65:
        pos.append((0.08, "L1: high resp sim (%.3f)" % emb_s["max_resp_sim"]))
    elif emb_s.get("max_resp_sim", 0) < 0.40 and emb_s.get("max_resp_sim", 0) > 0:
        neg.append((0.10, "L1: low resp sim"))

    if emb_s.get("sim_delta", 0) < -0.20:
        neg.append((0.12, "L1: big fup sim drop"))
    elif emb_s.get("sim_delta", 0) > 0.05:
        pos.append((0.08, "L1: fup sim rise"))

    if emb_s.get("resp_on_topic_pct", 0) > 0.6:
        pos.append((0.06, "L1: resp mostly on-topic"))
    elif emb_s.get("resp_on_topic_pct", 0) == 0 and emb_s.get("resp_words", 0) > 10:
        neg.append((0.06, "L1: resp fully off-topic"))

    # ── LAYER 1b: BART (stance) ──

    bd = bart_s.get("bart_delta", 0)
    if bd > 0.3:
        pos.append((0.25, "L1b: BART strong positive (%+.2f)" % bd))
    elif bd > 0.1:
        pos.append((0.12, "L1b: BART moderate positive (%+.2f)" % bd))
    elif bd < -0.3:
        neg.append((0.25, "L1b: BART strong negative (%+.2f)" % bd))
    elif bd < -0.1:
        neg.append((0.12, "L1b: BART moderate negative (%+.2f)" % bd))
    else:
        unc.append((0.10, "L1b: BART uncertain (%+.2f)" % bd))

    if bart_s.get("bart_agree", 0) > 0.7:
        pos.append((0.15, "L1b: BART high agreement (%.0f%%)" % (bart_s["bart_agree"] * 100)))
    if bart_s.get("bart_disagree", 0) > 0.7:
        neg.append((0.15, "L1b: BART high disagreement (%.0f%%)" % (bart_s["bart_disagree"] * 100)))

    # ── LAYER 2: Cross-signal patterns ──

    if regex["affirm_strong"] > 0 and bart_s.get("bart_agree", 0) > 0.5:
        pos.append((0.15, "L2: regex+BART agree combo"))

    if (regex["redirect_strong"] > 0 or regex["complaint"] > 0) and bart_s.get("bart_disagree", 0) > 0.3:
        neg.append((0.15, "L2: regex+BART redirect combo"))

    if (emb_s.get("max_resp_sim", 0) > 0.50 and
            emb_s.get("max_fup_sim", 0) < 0.40 and
            emb_s.get("sim_delta", 0) < -0.10):
        neg.append((0.15, "L2: parroting pattern"))

    if regex["fup_opens_transition"] and (regex["redirect_strong"] > 0 or regex["redirect_mild"] > 0):
        neg.append((0.10, "L2: polite dismiss"))

    if emb_s.get("fup_sent_count", 0) > 3 and emb_s.get("fup_on_topic_pct", 0) < 0.3:
        neg.append((0.10, "L2: topic sprawl"))

    has_any_pos = (
        regex["affirm_strong"] > 0 or regex["affirm_moderate"] > 0 or
        regex["affirm_building"] > 0 or regex["extension"] > 0 or
        regex["fup_opens_positive"] or bart_s.get("bart_agree", 0) > 0.5
    )
    has_any_neg = (
        regex["redirect_strong"] > 0 or regex["redirect_mild"] > 0 or
        regex["complaint"] > 0 or regex["fup_opens_negative"] or
        bart_s.get("bart_disagree", 0) > 0.5
    )
    if not has_any_pos and not has_any_neg:
        neg.append((0.08, "L2: no engagement signals"))

    # ── Compute final ──

    total_pos = sum(w for w, _ in pos)
    total_neg = sum(w for w, _ in neg)
    total_unc = sum(w for w, _ in unc)
    total_weight = total_pos + total_neg + total_unc + 0.01

    evidence = (total_pos - total_neg) / total_weight
    n_signals = len(pos) + len(neg) + len(unc)
    signal_agreement = abs(total_pos - total_neg) / (total_pos + total_neg + 0.01)
    confidence = min(signal_agreement * min(n_signals / 3, 1.0), 1.0)

    all_reasons = (
        [(w, r, "+") for w, r in pos] +
        [(w, r, "-") for w, r in neg] +
        [(w, r, "?") for w, r in unc]
    )
    all_reasons.sort(key=lambda x: -x[0])

    return evidence, confidence, all_reasons, n_signals


def evidence_to_precision(evidence: float, confidence: float) -> Optional[float]:
    """Map evidence/confidence to a 0.0-1.0 precision score.

    Returns None if confidence is too low to make a determination.
    """
    if confidence < 0.15:
        return None
    # Map evidence [-1, +1] to precision [0, 1]
    return max(0.0, min(1.0, (evidence + 1.0) / 2.0))


def classify_followup_signal(evidence: float, confidence: float) -> str:
    """Classify the followup into a signal category.

    Returns one of: "positive", "negative", "neutral", "uncertain"
    """
    if confidence < 0.15:
        return "uncertain"
    if evidence > 0.1:
        return "positive"
    if evidence < -0.1:
        return "negative"
    return "neutral"


def determine_match_method(bart_available: bool, emb_available: bool) -> str:
    """Return a string describing which layers were active."""
    parts = ["regex"]
    if emb_available:
        parts.append("emb")
    if bart_available:
        parts.append("bart")
    return "+".join(parts)


# ── Helpers ──

def _bart_classify(text: str) -> Dict[str, float]:
    """Classify a single text with BART. Returns scores dict."""
    if not is_bart_ready():
        return {}
    r = _bart_pipeline(text, BART_LABELS, multi_label=True)
    return dict(zip(r['labels'], r['scores']))


def _empty_bart() -> Dict[str, float]:
    """Zero-value BART signals for degraded mode."""
    return {
        "bart_agree": 0.0, "bart_build": 0.0,
        "bart_disagree": 0.0, "bart_redirect": 0.0,
        "bart_topic_change": 0.0,
        "bart_pos": 0.0, "bart_neg": 0.0, "bart_delta": 0.0,
    }


def _empty_emb() -> Dict[str, Any]:
    """Zero-value embedding signals for degraded mode."""
    return {
        "resp_words": 0, "depth_ratio": 0.0,
        "max_resp_sim": 0.0, "max_fup_sim": 0.0, "sim_delta": 0.0,
        "resp_on_topic_pct": 0.0, "fup_on_topic_pct": 0.0,
        "fup_sent_count": 0,
    }
