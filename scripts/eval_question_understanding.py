#!/usr/bin/env python3
"""Evaluate question understanding (src/datamate/question.py) on the training split.
Reports answer-type accuracy + confusion, accuracy by confidence bucket, date granularity, top-k extraction,
time-recognition coverage, concept firing on unanswerable questions, speed, and lists misclassified questions.
Usage: python scripts/eval_question_understanding.py [-d train] [--show 50]
"""

import argparse
import collections
import json
import os
import sys
import time

import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.datamate.question import analyze_question  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--data-dir", default="train")
    ap.add_argument("--show", type=int, default=60, help="max misclassified questions to print")
    args = ap.parse_args()

    questions = [json.loads(l) for l in open(os.path.join(args.data_dir, "questions.jsonl"), encoding="utf-8") if l.strip()]
    answers = {}
    ans_path = os.path.join(args.data_dir, "answers.jsonl")
    if os.path.exists(ans_path):
        answers = {json.loads(l)["id"]: json.loads(l) for l in open(ans_path, encoding="utf-8") if l.strip()}
    cal_path = os.path.join(args.data_dir, "calendar.csv")
    cal = pd.read_csv(cal_path) if os.path.exists(cal_path) else None

    conf_matrix = collections.Counter()
    per_type = collections.Counter()
    per_type_ok = collections.Counter()
    buckets = collections.defaultdict(lambda: [0, 0])
    wrong = []
    unans_pred = collections.Counter()
    gran_ok = gran_n = 0
    k_ok = k_n = k_missing = 0
    time_seen = time_conf = 0
    flagged_unans = flagged_ans = 0
    t0 = time.time()
    for q in questions:
        qa = analyze_question(q["question"], q["as_of_date"], cal)
        if qa.time.primary or qa.time.point_in_time or qa.time.matched:
            time_seen += 1
            time_conf += qa.time.confident
        gold = answers.get(q["id"])
        if not gold:
            continue
        gt = gold["type"]
        flag = qa.subjective or qa.external_knowledge or qa.forecast
        if gt == "unanswerable":
            unans_pred[qa.answer_type] += 1
            flagged_unans += flag
            continue
        flagged_ans += flag
        per_type[gt] += 1
        conf_matrix[(gt, qa.answer_type)] += 1
        ok = gt == qa.answer_type
        per_type_ok[gt] += ok
        b = "<0.7" if qa.answer_type_confidence < 0.7 else ("0.7-0.9" if qa.answer_type_confidence < 0.9 else ">=0.9")
        buckets[b][0] += 1
        buckets[b][1] += ok
        if not ok:
            wrong.append((gt, qa.answer_type, qa.answer_type_rule, q["question"]))
        if gt == "date" and qa.answer_type == "date":
            gran_n += 1
            gold_gran = "month" if len(str(gold["answer"])) == 7 else "day"
            gran_ok += gold_gran == qa.date_granularity
        if gt in ("list", "table") and qa.answer_type == gt:
            n_gold = len(gold["answer"]) if gt == "list" else len(gold["answer"]["rows"])
            all_str = gt == "table" or all(isinstance(x, str) for x in gold["answer"])
            if all_str:
                k_n += 1
                if qa.top_k is None:
                    k_missing += 1
                elif qa.top_k == n_gold or (qa.top_k > n_gold):
                    k_ok += 1
    elapsed = time.time() - t0

    n = sum(per_type.values())
    acc = sum(per_type_ok.values())
    print(f"Answer type accuracy (answerable): {acc}/{n} = {acc / max(n, 1):.4f}")
    for t in sorted(per_type, key=lambda t: -per_type[t]):
        print(f"  {t:9s} {per_type_ok[t]:5d}/{per_type[t]:<5d} {per_type_ok[t] / per_type[t]:.3f}")
    print("Confusion (gold -> predicted, errors only):")
    for (g, p), c in sorted(conf_matrix.items(), key=lambda x: -x[1]):
        if g != p:
            print(f"  {g:9s} -> {p:9s} {c}")
    print("Accuracy by confidence bucket:")
    for b in ("<0.7", "0.7-0.9", ">=0.9"):
        tot, ok = buckets[b]
        if tot:
            print(f"  {b:8s} {ok}/{tot} = {ok / tot:.3f}")
    print(f"Predicted type on unanswerable questions: {dict(unans_pred)}")
    print(f"Date granularity accuracy: {gran_ok}/{gran_n}")
    print(f"top_k on name list/table questions: correct {k_ok}/{k_n}, missing {k_missing} (gold shorter than k counts as correct)")
    print(f"Time expressions recognized: {time_seen}/{len(questions)}; confident {time_conf}/{time_seen}")
    print(f"Subjective/external/forecast flags: on unanswerable {flagged_unans}/{sum(unans_pred.values())}, on answerable {flagged_ans}/{n}")
    print(f"Speed: {1000 * elapsed / len(questions):.2f} ms/question")
    print(f"\nMisclassified ({len(wrong)}):")
    for g, p, rule, text in wrong[: args.show]:
        print(f"  [{g} -> {p} via {rule}] {text}")


if __name__ == "__main__":
    main()
