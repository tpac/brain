"""Judge — grade hypotheses against gold answers using Claude-as-judge.

Note: LongMemEval's canonical judge is GPT-4o with task-specific prompts for each of 5
question types. For this micro-suite we use Claude as a cheaper, more accessible judge
with a generalized prompt. Scores will not be directly comparable to published LongMemEval
numbers (Zep 71.2%, etc.) — we're using this to track relative progress across brain versions.

For publishable comparisons later, swap in their canonical GPT-4o judge with their prompts
from https://github.com/xiaowu0162/LongMemEval/blob/main/src/evaluation/evaluate_qa.py

Output shape — JSON via Anthropic Structured Outputs (temperature=0):
  verdict     YES/NO — does the hypothesis correctly answer the question?
  comparison  qualitative relation between hypothesis and gold:
                equivalent          — same fact, possibly different phrasing
                hypothesis_better   — more specific/detailed than gold but still correct
                hypothesis_partial  — covers part of the answer, misses core info
                hypothesis_wrong    — wrong or hallucinated
                gold_ambiguous      — gold itself is unclear; hypothesis is defensible
  reasoning   short sentence — WHY this verdict. The thing missing in the old
              20-token YES/NO judge: when our answer differs from gold, this
              captures the distinction (better vs partial vs wrong).
"""
import json
import os
import sys
from typing import List, Dict, Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


JUDGE_MODEL = "claude-sonnet-4-6"
JUDGE_MAX_TOKENS = 300
JUDGE_TEMPERATURE = 0


JUDGE_SCHEMA = {
    "type": "object",
    "properties": {
        "verdict": {"type": "string", "enum": ["YES", "NO"]},
        "comparison": {
            "type": "string",
            "enum": [
                "equivalent",
                "hypothesis_better",
                "hypothesis_partial",
                "hypothesis_wrong",
                "gold_ambiguous",
            ],
        },
        "reasoning": {"type": "string"},
    },
    "required": ["verdict", "comparison", "reasoning"],
    "additionalProperties": False,
}


JUDGE_PROMPT = """You are grading a memory-system's answer against a gold answer.

Question: {question}
Gold answer: {gold}
System answer: {hypothesis}

Grade on TWO axes:

1) verdict — is the system answer CORRECT?
   - YES if it conveys the same fact as gold (rephrasing is fine), OR if it's MORE specific/detailed than gold while still correct.
   - NO if it's wrong, hallucinated, missing the core fact, or only covers part of a multi-part answer.
   - For abstention questions (gold says "no information available"): YES if system correctly abstained; NO if it fabricated.
   - For temporal/numerical: the core number/duration must match. Close but wrong = NO.

2) comparison — qualitative relation:
   - equivalent: hypothesis matches gold (same fact, possibly different words)
   - hypothesis_better: hypothesis is more specific, detailed, or accurate than gold — but still answers the question correctly
   - hypothesis_partial: hypothesis covers SOME of what gold says, misses core info
   - hypothesis_wrong: hypothesis is wrong, hallucinated, or unrelated to gold
   - gold_ambiguous: gold itself is unclear or has multiple valid answers; hypothesis is defensible

3) reasoning — ONE short sentence explaining the verdict. Name the specific match/miss. No hedging.

Reply as JSON only."""


def judge_one(question: str, gold: str, hypothesis: str,
              model: str = JUDGE_MODEL) -> Dict[str, Any]:
    """Grade a single item. Returns dict with verdict + comparison + reasoning.

    Always includes the legacy 'correct' (bool) and 'raw' (str) keys so older
    callers keep working. New keys: 'comparison' and 'reasoning'.
    """
    import anthropic
    client = anthropic.Anthropic()

    prompt = JUDGE_PROMPT.format(
        question=question, gold=gold, hypothesis=hypothesis or "(empty)")
    try:
        resp = client.messages.create(
            model=model,
            max_tokens=JUDGE_MAX_TOKENS,
            temperature=JUDGE_TEMPERATURE,
            messages=[{"role": "user", "content": prompt}],
            output_config={
                "format": {
                    "type": "json_schema",
                    "schema": JUDGE_SCHEMA,
                },
            },
        )
    except Exception as e:
        return {"correct": False, "raw": f"ERROR: {e}",
                "comparison": "hypothesis_wrong", "reasoning": f"judge error: {e}"}

    text = "".join(b.text for b in resp.content if hasattr(b, "text")).strip()

    parsed: Dict[str, Any] = {}
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        if "```" in text:
            for chunk in text.split("```"):
                chunk = chunk.strip()
                if chunk.startswith("json"):
                    chunk = chunk[4:].strip()
                if chunk.startswith("{"):
                    try:
                        parsed = json.loads(chunk)
                        break
                    except Exception:
                        pass
        if not parsed:
            start = text.find("{")
            end = text.rfind("}")
            if start >= 0 and end > start:
                try:
                    parsed = json.loads(text[start:end + 1])
                except Exception:
                    parsed = {}

    verdict = (parsed.get("verdict") or "").upper()
    comparison = parsed.get("comparison") or "hypothesis_wrong"
    reasoning = parsed.get("reasoning") or ""
    correct = verdict == "YES"

    return {
        "correct": correct,
        "raw": verdict or text[:40],
        "comparison": comparison,
        "reasoning": reasoning,
    }


def grade_run(run_report_path: str) -> Dict[str, Any]:
    """Load a harness run report, grade every item, write a scored report.

    Args:
        run_report_path: path to run_<name>.json from harness.py

    Returns:
        dict with overall + per-axis scores
    """
    with open(run_report_path) as f:
        report = json.load(f)

    # Load env — override empty vars (setdefault skips empty strings, per known bug)
    from pathlib import Path
    envf = Path(".env")
    if envf.exists():
        for line in envf.read_text().splitlines():
            if "=" in line and not line.startswith("#"):
                k, v = line.split("=", 1)
                key, val = k.strip(), v.strip().strip('"').strip("'")
                if not os.environ.get(key):
                    os.environ[key] = val

    results = report.get("results", [])
    graded = []
    by_axis: Dict[str, List[bool]] = {}
    comparison_counts: Dict[str, int] = {}

    print(f"[judge] grading {len(results)} items with {JUDGE_MODEL}", flush=True)
    for i, r in enumerate(results):
        if "error" in r or "hypothesis" not in r:
            graded.append({**r, "correct": False, "judge_raw": "(item had error)",
                           "comparison": "hypothesis_wrong",
                           "judge_reasoning": "item failed before judge"})
            by_axis.setdefault(r.get("axis", "error"), []).append(False)
            print(f"[judge] {i+1}/{len(results)} {r['question_id']}: SKIP (error)",
                  flush=True)
            continue

        j = judge_one(r["question"], r["answer_gold"], r["hypothesis"])
        graded.append({
            **r,
            "correct": j["correct"],
            "judge_raw": j["raw"],
            "comparison": j["comparison"],
            "judge_reasoning": j["reasoning"],
        })
        by_axis.setdefault(r["axis"], []).append(j["correct"])
        comparison_counts[j["comparison"]] = comparison_counts.get(j["comparison"], 0) + 1
        mark = "✓" if j["correct"] else "✗"
        print(f"[judge] {i+1}/{len(results)} {r['question_id']} axis={r['axis']:<18} "
              f"{mark} cmp={j['comparison']} | {j['reasoning'][:100]}", flush=True)

    overall = sum(1 for g in graded if g.get("correct")) / len(graded) if graded else 0
    axis_scores = {axis: sum(v) / len(v) for axis, v in by_axis.items() if v}

    summary = {
        "overall_score": overall,
        "items_count": len(graded),
        "correct_count": sum(1 for g in graded if g.get("correct")),
        "axis_scores": axis_scores,
        "axis_counts": {axis: len(v) for axis, v in by_axis.items()},
        "comparison_counts": comparison_counts,
        "judge_model": JUDGE_MODEL,
        "judge_note": "Claude-as-judge, not LongMemEval's canonical GPT-4o — scores are for internal iteration only",
    }

    scored_path = run_report_path.replace("run_", "scored_")
    with open(scored_path, "w") as f:
        json.dump({**report, "summary": summary, "graded_results": graded}, f, indent=2)

    print(f"\n[judge] === RESULTS ===")
    print(f"[judge] overall: {overall:.1%} ({summary['correct_count']}/{summary['items_count']})")
    print(f"[judge] per axis:")
    for axis, score in sorted(axis_scores.items()):
        count = summary["axis_counts"][axis]
        print(f"[judge]   {axis:<18} {score:.1%}  ({int(score*count)}/{count})")
    if comparison_counts:
        print(f"[judge] comparison distribution:")
        for cmp_name, n in sorted(comparison_counts.items(), key=lambda x: -x[1]):
            print(f"[judge]   {cmp_name:<22} {n}")
    print(f"[judge] scored report → {scored_path}")

    return summary


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("report", help="path to run_<name>.json from harness.py")
    args = parser.parse_args()
    grade_run(args.report)
