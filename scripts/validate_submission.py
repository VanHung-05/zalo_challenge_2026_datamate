#!/usr/bin/env python3
"""Validate submission.jsonl format before uploading (docs/evaluation.md, rulebook §9).
Checks every line parses, ids match questions.jsonl, no duplicates, and each answer has the
single JSON shape its type requires. Exit code 1 if any error is found.
"""

import argparse
import json
import re
import sys
from collections import Counter

VALID_TYPES = {"number", "percent", "string", "date", "boolean", "list", "table", "unanswerable", "abstain"}


def _is_number(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def check_answer(item: dict) -> str:
    """Return '' if the item is well-formed, else a description of the problem."""
    t = item.get("type")
    if t not in VALID_TYPES:
        return f"invalid type {t!r}"
    if t == "abstain":
        return ""
    if t == "unanswerable":
        reason = item.get("reason")
        return "" if reason is None or isinstance(reason, str) else "reason must be a string"
    if "answer" not in item:
        return "missing 'answer'"
    a = item["answer"]
    if t in ("number", "percent"):
        return "" if _is_number(a) else f"{t} must be a JSON number"
    if t == "string":
        return "" if isinstance(a, str) and a.strip() else "string must be a non-empty JSON string"
    if t == "date":
        return "" if isinstance(a, str) and re.fullmatch(r"\d{4}-\d{2}(-\d{2})?", a) else "date must be 'YYYY-MM-DD' or 'YYYY-MM'"
    if t == "boolean":
        return "" if isinstance(a, bool) else "boolean must be true/false"
    if t == "list":
        if not isinstance(a, list):
            return "list must be a JSON array"
        return "" if all(isinstance(x, str) or _is_number(x) for x in a) else "list elements must be strings or numbers"
    if t == "table":
        if not isinstance(a, dict) or not isinstance(a.get("columns"), list) or not isinstance(a.get("rows"), list):
            return "table must be {'columns': [...], 'rows': [[...], ...]}"
        n = len(a["columns"])
        for row in a["rows"]:
            if not isinstance(row, list) or len(row) != n:
                return "every table row must be a list with one cell per column"
            if any(not (isinstance(c, str) or _is_number(c)) for c in row):
                return "table cells must be strings or numbers"
        return ""
    return ""


def validate(predictions_path: str, questions_path: str) -> int:
    question_ids = []
    with open(questions_path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                question_ids.append(json.loads(line)["id"])
    expected = set(question_ids)

    errors = []
    seen = Counter()
    types = Counter()
    with open(predictions_path, encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as e:
                errors.append(f"line {n}: invalid JSON ({e})")
                continue
            qid = item.get("id")
            seen[qid] += 1
            if qid not in expected:
                errors.append(f"line {n}: id {qid!r} not in questions")
            problem = check_answer(item)
            if problem:
                errors.append(f"line {n} ({qid}): {problem}")
            types[item.get("type")] += 1

    dups = [q for q, c in seen.items() if c > 1]
    missing = expected - set(seen)
    for q in dups:
        errors.append(f"duplicate id {q!r} ({seen[q]} lines)")

    print(f"Questions: {len(expected)} | lines: {sum(seen.values())} | missing (scored as abstain): {len(missing)}")
    print("Types: " + ", ".join(f"{t}={c}" for t, c in types.most_common()))
    if errors:
        print(f"{len(errors)} error(s):")
        for e in errors[:50]:
            print("  " + e)
        return 1
    print("OK: submission format is valid")
    return 0


def main():
    parser = argparse.ArgumentParser(description="Validate submission.jsonl format")
    parser.add_argument("--predictions", "-p", required=True)
    parser.add_argument("--questions", "-q", required=True)
    args = parser.parse_args()
    sys.exit(validate(args.predictions, args.questions))


if __name__ == "__main__":
    main()
