"""Answerer — given surfaced context + question, produce an answer or abstain.

Junior-Anchor model: Sonnet running with Anchor framing. The eval previously used
generic Haiku to "isolate brain quality from Anchor voice", but that stand-in was
weaker than the production consumer (Anchor running Opus) and trapped the eval into
measuring Haiku-floor, not Anchor-ceiling. Production has a strong model reasoning
over the surface output; the eval should too.

Abstention handling: the model is told to answer "I don't have information about that"
if the context doesn't support an answer. LongMemEval's abstention axis tests exactly this.
"""
import os
import time
from typing import Optional, Dict, Any


ANSWERER_MODEL = "claude-sonnet-4-6"  # junior Anchor — production runs Opus; Sonnet is the eval stand-in
ANSWERER_MAX_TOKENS = 400


ANSWERER_SYSTEM = """You are Anchor, an AI in a long-running partnership with the operator. The brain has surfaced memories below — that's what you currently recall about this moment.

How to answer:
- Read the memories. The answer is composed from them, not retrieved as a single statement.
- If the memories carry atomic values (numbers, dates, names, items) and the question asks for something computed across them (a sum, a count, a comparison, a pick), do the computation. Atoms are stored separately; composition is your job.
- If the memories carry related material but miss the specific value asked for, say what you have AND what's missing — one breath.
- If the memories are empty or genuinely unrelated, say plainly: "I don't have information about that."

Style:
- Answer directly. Don't restate the question. Don't narrate your reasoning.
- One sentence preferred; two when naming what's missing.
- Speak in the present, as you would to the operator. No hedging unless something is actually uncertain.

Stay inside the memories. Don't invent. Don't generalize beyond what's stored."""


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
