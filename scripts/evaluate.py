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
import math
import re
import unicodedata
from typing import Any, Dict, List, Optional, Tuple


def normalize_string(s: Any) -> str:
    """Normalize string per rulebook §9:
    1. Unicode NFC normalization
    2. Lowercase
    3. Non-alphanumeric chars -> space (preserves Vietnamese accented letters)
    4. Collapse spaces
    5. Strip
    """
    if s is None:
        return ""
    s = str(s)
    s = unicodedata.normalize("NFC", s)
    s = s.lower()
    # Replace anything that is not alphanumeric or whitespace with space
    # \w in Python regex includes Vietnamese letters
    s = re.sub(r"[^\w\s]", " ", s, flags=re.UNICODE)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def match_string(predicted: Any, reference: Any, aliases: Optional[List[Any]] = None) -> bool:
    if not isinstance(predicted, str):
        return False
    norm_pred = normalize_string(predicted)
    norm_ref = normalize_string(reference)
    if norm_pred == norm_ref:
        return True
    if aliases:
        for alias in aliases:
            if isinstance(alias, list):
                for a in alias:
                    if norm_pred == normalize_string(a):
                        return True
            else:
                if norm_pred == normalize_string(alias):
                    return True
    return False


def match_number(predicted: Any, reference: Any) -> bool:
    if isinstance(predicted, bool) or not isinstance(predicted, (int, float)):
        return False
    if isinstance(reference, bool) or not isinstance(reference, (int, float)):
        return False
    # Check exact integer equality if reference is integer
    if isinstance(reference, int) or (isinstance(reference, float) and reference.is_integer()):
        ref_int = int(round(reference))
        if isinstance(predicted, int) or (isinstance(predicted, float) and predicted.is_integer()):
            return int(round(predicted)) == ref_int
    # Float comparison with relative tolerance
    diff = abs(predicted - reference)
    tol = 1e-5 * max(1.0, abs(reference))
    return diff <= tol


def match_percent(predicted: Any, reference: Any) -> bool:
    if isinstance(predicted, bool) or not isinstance(predicted, (int, float)):
        return False
    if isinstance(reference, bool) or not isinstance(reference, (int, float)):
        return False
    # Percent is rounded to 2 decimal places per rulebook
    p_rounded = round(float(predicted), 2)
    r_rounded = round(float(reference), 2)
    return abs(p_rounded - r_rounded) < 1e-4 or abs(predicted - reference) <= 0.015


def match_date(predicted: Any, reference: Any) -> bool:
    if not isinstance(predicted, str):
        return False
    p = predicted.strip()
    r = str(reference).strip()
    return p == r


def match_boolean(predicted: Any, reference: Any) -> bool:
    if not isinstance(predicted, bool) or not isinstance(reference, bool):
        return False
    return predicted is reference


def match_element(pred_elem: Any, ref_elem: Any, elem_aliases: Optional[List[Any]] = None) -> bool:
    if isinstance(ref_elem, (int, float)) and not isinstance(ref_elem, bool):
        return match_number(pred_elem, ref_elem)
    elif isinstance(ref_elem, str):
        return match_string(pred_elem, ref_elem, elem_aliases)
    elif isinstance(ref_elem, bool):
        return match_boolean(pred_elem, ref_elem)
    return pred_elem == ref_elem


def match_list(predicted: Any, reference: Any, aliases: Optional[List[Any]] = None) -> bool:
    if not isinstance(predicted, list) or not isinstance(reference, list):
        return False
    if len(predicted) != len(reference):
        return False
    for i in range(len(reference)):
        ref_val = reference[i]
        pred_val = predicted[i]
        elem_alias = None
        if aliases and i < len(aliases):
            elem_alias = aliases[i]
            if not isinstance(elem_alias, list):
                elem_alias = [elem_alias]
        if not match_element(pred_val, ref_val, elem_alias):
            return False
    return True


def match_table(predicted: Any, reference: Any, aliases: Optional[List[Any]] = None) -> bool:
    if not isinstance(predicted, dict) or not isinstance(reference, dict):
        return False
    if "columns" not in predicted or "rows" not in predicted:
        return False
    if "columns" not in reference or "rows" not in reference:
        return False

    pred_cols = predicted["columns"]
    ref_cols = reference["columns"]
    pred_rows = predicted["rows"]
    ref_rows = reference["rows"]

    if len(pred_cols) != len(ref_cols):
        return False
    if len(pred_rows) != len(ref_rows):
        return False

    # Check rows set equality (order independent)
    matched_ref_indices = set()
    for pred_row in pred_rows:
        if not isinstance(pred_row, list) or len(pred_row) != len(ref_cols):
            return False
        found = False
        for idx, ref_row in enumerate(ref_rows):
            if idx in matched_ref_indices:
                continue
            # Compare elements
            elem_match = True
            for col_idx in range(len(ref_row)):
                ref_cell = ref_row[col_idx]
                pred_cell = pred_row[col_idx]
                row_alias = None
                if col_idx == 0 and aliases and idx < len(aliases):
                    row_alias = aliases[idx]
                    if not isinstance(row_alias, list):
                        row_alias = [row_alias]
                if not match_element(pred_cell, ref_cell, row_alias):
                    elem_match = False
                    break
            if elem_match:
                matched_ref_indices.add(idx)
                found = True
                break
        if not found:
            return False

    return len(matched_ref_indices) == len(ref_rows)


def match_answer(predicted_item: Dict[str, Any], ground_truth: Dict[str, Any]) -> bool:
    pred_type = predicted_item.get("type")
    gt_type = ground_truth.get("type")

    if pred_type != gt_type:
        return False

    pred_ans = predicted_item.get("answer")
    gt_ans = ground_truth.get("answer")
    aliases = ground_truth.get("alias", [])

    if gt_type == "number":
        return match_number(pred_ans, gt_ans)
    elif gt_type == "percent":
        return match_percent(pred_ans, gt_ans)
    elif gt_type == "string":
        return match_string(pred_ans, gt_ans, aliases)
    elif gt_type == "date":
        return match_date(pred_ans, gt_ans)
    elif gt_type == "boolean":
        return match_boolean(pred_ans, gt_ans)
    elif gt_type == "list":
        return match_list(pred_ans, gt_ans, aliases)
    elif gt_type == "table":
        return match_table(pred_ans, gt_ans, aliases)
    elif gt_type == "unanswerable":
        return True
    return False


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
