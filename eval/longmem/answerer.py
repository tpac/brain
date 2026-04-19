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

The user has a long history with an AI assistant. Below you'll see a set of memories that the memory system retrieved as relevant to the question. Use ONLY these memories to answer — do not invent facts.

Rules:
- If the memories contain the answer, state it directly and concisely.
- If the memories are RELATED but don't contain the specific answer the question asks for, say: "I don't have information about that."
- If the memories are empty or clearly unrelated to the question, say: "I don't have information about that."
- Do not hedge or qualify. Either answer from the memories, or abstain cleanly.
- Do not restate the question. Do not explain your reasoning. Answer in one sentence when possible.

Temporal questions: if the memories have dates but the question asks for elapsed time, do the math based on the dates provided."""


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
