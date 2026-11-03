#!/usr/bin/env python3
"""Evaluator for DataMate (Zalo AI Challenge 2026).
Implements exact Penalized Accuracy (PA) scoring rule:
  Correct: +1
  Wrong:   -1
  Abstain:  0
  PA = (correct - wrong) / N
"""

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional, Tuple


# Matching rules live in src/datamate/scoring.py (shared with the answer voter)
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.datamate.scoring import match_answer, normalize_string  # noqa: E402


def evaluate(
    predictions_path: str,
    ground_truth_path: str,
    questions_path: Optional[str] = None,
    only_predicted: bool = False
) -> Dict[str, Any]:
    # Load ground truth
    ground_truths: Dict[str, Dict[str, Any]] = {}
    with open(ground_truth_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            ground_truths[item["id"]] = item

    # Load questions metadata if available
    q_meta: Dict[str, Dict[str, Any]] = {}
    if questions_path:
        with open(questions_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                q_meta[item["id"]] = item

    # Load predictions
    predictions: Dict[str, Dict[str, Any]] = {}
    with open(predictions_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            predictions[item["id"]] = item

    # Scoring a partial run (e.g. --limit 300): keep only the questions that were predicted
    if only_predicted:
        ground_truths = {qid: gt for qid, gt in ground_truths.items() if qid in predictions}

    total_n = len(ground_truths)
    correct = 0
    wrong = 0
    abstain = 0

    type_stats: Dict[str, Dict[str, int]] = {}
    biz_stats: Dict[str, Dict[str, int]] = {}

    for qid, gt in ground_truths.items():
        gt_type = gt.get("type", "unknown")
        biz_id = q_meta.get(qid, {}).get("business_id", "unknown")

        if gt_type not in type_stats:
            type_stats[gt_type] = {"correct": 0, "wrong": 0, "abstain": 0, "total": 0}
        type_stats[gt_type]["total"] += 1

        if biz_id not in biz_stats:
            biz_stats[biz_id] = {"correct": 0, "wrong": 0, "abstain": 0, "total": 0}
        biz_stats[biz_id]["total"] += 1

        if qid not in predictions:
            # Missing prediction treated as abstain per rulebook §9
            abstain += 1
            type_stats[gt_type]["abstain"] += 1
            biz_stats[biz_id]["abstain"] += 1
            continue

        pred = predictions[qid]
        p_type = pred.get("type")

        if p_type == "abstain":
            abstain += 1
            type_stats[gt_type]["abstain"] += 1
            biz_stats[biz_id]["abstain"] += 1
        elif match_answer(pred, gt):
            correct += 1
            type_stats[gt_type]["correct"] += 1
            biz_stats[biz_id]["correct"] += 1
        else:
            wrong += 1
            type_stats[gt_type]["wrong"] += 1
            biz_stats[biz_id]["wrong"] += 1

    pa = (correct - wrong) / total_n if total_n > 0 else 0.0

    results = {
        "total_questions": total_n,
        "correct": correct,
        "wrong": wrong,
        "abstain": abstain,
        "penalized_accuracy": round(pa, 4),
        "accuracy_on_answered": round(correct / (correct + wrong), 4) if (correct + wrong) > 0 else 0.0,
        "answer_rate": round((correct + wrong) / total_n, 4) if total_n > 0 else 0.0,
        "by_type": type_stats,
        "by_business": biz_stats if questions_path else {},
    }
    return results


def print_report(results: Dict[str, Any]):
    print("\n" + "=" * 60)
    print("           DATAMATE EVALUATION REPORT")
    print("=" * 60)
    print(f"Total Questions:     {results['total_questions']}")
    print(f"Correct (+1):        {results['correct']} ({results['correct'] / results['total_questions'] * 100:.2f}%)")
    print(f"Wrong   (-1):        {results['wrong']} ({results['wrong'] / results['total_questions'] * 100:.2f}%)")
    print(f"Abstain ( 0):        {results['abstain']} ({results['abstain'] / results['total_questions'] * 100:.2f}%)")
    print("-" * 60)
    print(f"Penalized Accuracy (PA):  {results['penalized_accuracy']:+.4f}")
    print(f"Accuracy on Answered:     {results['accuracy_on_answered'] * 100:.2f}%")
    print(f"Answer Rate:              {results['answer_rate'] * 100:.2f}%")
    print("-" * 60)
    print(f"{'Answer Type':<15} {'Total':<8} {'Correct':<9} {'Wrong':<8} {'Abstain':<9} {'PA':<8}")
    print("-" * 60)
    for t, s in sorted(results["by_type"].items()):
        t_pa = (s["correct"] - s["wrong"]) / s["total"] if s["total"] > 0 else 0.0
        print(f"{t:<15} {s['total']:<8} {s['correct']:<9} {s['wrong']:<8} {s['abstain']:<9} {t_pa:+.4f}")
    print("=" * 60 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Evaluate DataMate predictions")
    parser.add_argument("--predictions", "-p", required=True, help="Path to submission.jsonl")
    parser.add_argument("--ground-truth", "-g", required=True, help="Path to answers.jsonl")
    parser.add_argument("--questions", "-q", default=None, help="Path to questions.jsonl (optional)")
    parser.add_argument("--only-predicted", action="store_true",
                        help="Score only questions present in predictions (for runs with --limit/--business)")
    args = parser.parse_args()

    res = evaluate(args.predictions, args.ground_truth, args.questions, args.only_predicted)
    print_report(res)


if __name__ == "__main__":
    main()
