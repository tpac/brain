"""Answerer — given surfaced context + question, produce an answer or abstain.

Deliberately model-agnostic and generic: this is the benchmark answerer, not Anchor.
Keeping it generic means we measure the brain's retrieval quality, not Anchor's voice.

Abstention handling: the model is told to answer "I don't have information about that"
if the context doesn't support an answer. LongMemEval's abstention axis tests exactly this.
"""
import os
import time
from typing import Optional, Dict, Any


ANSWERER_MODEL = "claude-haiku-4-5"  # fast + cheap for 500-scale future runs; Sonnet available if needed
ANSWERER_MAX_TOKENS = 400


ANSWERER_SYSTEM = """You are answering a question based on memories retrieved from a personal memory system.

The user has a long history with an AI assistant. Below you'll see memories the system retrieved as relevant. Use ONLY these memories — do not invent facts.

Three response patterns. Pick the one that fits:

1. ANSWER — memories contain the answer OR support it by simple inference. Simple inference includes:
   - summing values (FB Live 12 + YouTube 21 = 33 comments)
   - comparing dates (Feb 20 < March 3 → tomatoes were first)
   - counting items (brother + 3 sisters = 4 siblings)
   - picking the earlier/later/largest/smallest of named events
   Don't abstain just because the answer requires this kind of reasoning over the memories — that's what an answer IS.
   Example: Q "How many sisters?" + memories list 3 sisters → "You have 3 sisters."
   Example: Q "Which seeds first, tomatoes or marigolds?" + memories show "tomato seeds started Feb 20" and "marigold seeds germinated March 3" → "Tomatoes (Feb 20 vs March 3)."

2. PARTIAL — memories contain RELATED material but not the exact value. The mismatch can be on:
   - value (memories have parts of a sum, question asks for the total),
   - date (memories show X in May, question asks for X in March),
   - entity (memories show similar X, question asks for specific Y).
   Say what you have AND name what's missing. One sentence, two if needed.
   Example: Q "Total comments on my Facebook Live and my YouTube video?" + memories show FB Live got 12 comments, no specific count for YouTube →
     "Your Facebook Live got 12 comments. I don't have a specific count for the YouTube video — just that you mentioned it was popular."

3. ABSTAIN — memories are empty or fully unrelated to the question.
   Say: "I don't have information about that."

Rules:
- Stay inside the memories. Do not invent facts.
- Do not restate the question. Do not explain your reasoning.
- Hedge ONLY in the PARTIAL pattern, and only to name what specifically is missing.
- One sentence preferred; two when PARTIAL requires it.

Temporal questions: if the memories have dates but the question asks for elapsed time, do the math from the dates."""


def answer_question(question: str, surfaced_context: Optional[str],
                    question_date: Optional[str] = None,
                    model: str = ANSWERER_MODEL) -> Dict[str, Any]:
    """Call the answerer LLM.

    Args:
        question: the eval question (exact string)
        surfaced_context: the brain's surface output (may be None/empty)
        question_date: optional date context for temporal reasoning
        model: Claude model to use

    Returns:
        {"hypothesis": str, "model": str, "tokens_in": int, "tokens_out": int,
         "elapsed_ms": int, "abstained": bool, "has_context": bool}
    """
    import anthropic
    client = anthropic.Anthropic()

    context_block = surfaced_context if surfaced_context else "(no relevant memories retrieved)"
    date_line = f"Today's date: {question_date}\n\n" if question_date else ""

    user_content = (
        f"{date_line}"
        f"Retrieved memories:\n---\n{context_block}\n---\n\n"
        f"Question: {question}\n\n"
        f"Answer:"
    )

    t0 = time.time()
    try:
        response = client.messages.create(
            model=model,
            max_tokens=ANSWERER_MAX_TOKENS,
            system=ANSWERER_SYSTEM,
            messages=[{"role": "user", "content": user_content}],
        )
    except Exception as e:
        return {"hypothesis": "", "model": model, "tokens_in": 0, "tokens_out": 0,
                "elapsed_ms": int((time.time() - t0) * 1000),
                "abstained": False, "has_context": bool(surfaced_context),
                "error": str(e)}

    elapsed_ms = int((time.time() - t0) * 1000)
    hypothesis = "".join(b.text for b in response.content if hasattr(b, "text")).strip()

    abstain_markers = [
        "don't have information",
        "no information",
        "cannot answer",
        "unable to answer",
    ]
    abstained = any(m in hypothesis.lower() for m in abstain_markers)

    return {
        "hypothesis": hypothesis,
        "model": model,
        "tokens_in": response.usage.input_tokens,
        "tokens_out": response.usage.output_tokens,
        "elapsed_ms": elapsed_ms,
        "abstained": abstained,
        "has_context": bool(surfaced_context),
    }
