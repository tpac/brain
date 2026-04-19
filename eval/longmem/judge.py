"""Judge — grade hypotheses against gold answers using Claude-as-judge.

Note: LongMemEval's canonical judge is GPT-4o with task-specific prompts for each of 5
question types. For this micro-suite we use Claude as a cheaper, more accessible judge
with a generalized prompt. Scores will not be directly comparable to published LongMemEval
numbers (Zep 71.2%, etc.) — we're using this to track relative progress across brain versions.

For publishable comparisons later, swap in their canonical GPT-4o judge with their prompts
from https://github.com/xiaowu0162/LongMemEval/blob/main/src/evaluation/evaluate_qa.py
"""
import json
import os
import sys
from typing import List, Dict, Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


JUDGE_MODEL = "claude-sonnet-4-6"  # Sonnet for judgment quality
JUDGE_MAX_TOKENS = 20


JUDGE_PROMPT = """You are grading a memory-system's answer.

Question: {question}
Gold answer: {gold}
System answer: {hypothesis}

Decide: does the system answer correctly address the question given the gold answer as ground truth?

- For factual questions: the system answer must convey the same fact, even if phrased differently. Minor wording differences OK. Missing the core fact = NO.
- For temporal/numerical questions: the core number/duration must match. Close but wrong = NO.
- For abstention questions (when the gold answer indicates "no information available"): the system correctly abstaining = YES. Hallucinating an answer = NO.
- For multi-part questions: must cover all parts. Missing part of a multi-part answer = NO.

Reply with a single token: YES or NO. No explanation."""


def judge_one(question: str, gold: str, hypothesis: str, model: str = JUDGE_MODEL) -> Dict[str, Any]:
    """Grade a single item. Returns {"correct": bool, "raw": str}."""
    import anthropic
    client = anthropic.Anthropic()

    prompt = JUDGE_PROMPT.format(question=question, gold=gold, hypothesis=hypothesis or "(empty)")
    try:
        resp = client.messages.create(
            model=model,
            max_tokens=JUDGE_MAX_TOKENS,
            messages=[{"role": "user", "content": prompt}],
        )
    except Exception as e:
        return {"correct": False, "raw": f"ERROR: {e}"}

    text = "".join(b.text for b in resp.content if hasattr(b, "text")).strip().upper()
    correct = text.startswith("YES")
    return {"correct": correct, "raw": text}


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

    print(f"[judge] grading {len(results)} items with {JUDGE_MODEL}", flush=True)
    for i, r in enumerate(results):
        if "error" in r or "hypothesis" not in r:
            graded.append({**r, "correct": False, "judge_raw": "(item had error)"})
            by_axis.setdefault(r.get("axis", "error"), []).append(False)
            print(f"[judge] {i+1}/{len(results)} {r['question_id']}: SKIP (error)", flush=True)
            continue

        j = judge_one(r["question"], r["answer_gold"], r["hypothesis"])
        graded.append({**r, "correct": j["correct"], "judge_raw": j["raw"]})
        by_axis.setdefault(r["axis"], []).append(j["correct"])
        print(f"[judge] {i+1}/{len(results)} {r['question_id']} axis={r['axis']:<18} "
              f"{'✓' if j['correct'] else '✗'} | "
              f"hyp: {r['hypothesis'][:80]}",
              flush=True)

    overall = sum(1 for g in graded if g.get("correct")) / len(graded) if graded else 0
    axis_scores = {axis: sum(v) / len(v) for axis, v in by_axis.items() if v}

    summary = {
        "overall_score": overall,
        "items_count": len(graded),
        "correct_count": sum(1 for g in graded if g.get("correct")),
        "axis_scores": axis_scores,
        "axis_counts": {axis: len(v) for axis, v in by_axis.items()},
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
    print(f"[judge] scored report → {scored_path}")

    return summary


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("report", help="path to run_<name>.json from harness.py")
    args = parser.parse_args()
    grade_run(args.report)
