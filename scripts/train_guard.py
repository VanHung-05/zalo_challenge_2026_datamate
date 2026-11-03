#!/usr/bin/env python3
"""Measure and fit the Answerability Guard on train (plan §4.6).

1. Rule-based Guard: precision / recall of "unanswerable" against answers.jsonl (answer: null).
2. Logistic regression over the Guard features with GroupKFold by business (no business is both
   in training and evaluation of a fold), threshold sweep on out-of-fold probabilities.
3. --save PATH writes the model fitted on all train businesses (JSON, read by guard.Guard).

The precision of the unanswerable label matters most: a wrong "unanswerable" costs −1.
"""

import argparse
import collections
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np  # noqa: E402

from src.datamate.business_card import build_business_card  # noqa: E402
from src.datamate.db import BusinessDB  # noqa: E402
from src.datamate.guard import GUARD_FEATURES, Guard  # noqa: E402
from src.datamate.loader import load_business_data  # noqa: E402
from src.datamate.question import analyze_question  # noqa: E402
from src.datamate.rules import load_calendar  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--data-dir", default="train")
    ap.add_argument("--save", default=None)
    ap.add_argument("--show", type=int, default=40)
    args = ap.parse_args()

    qs = [json.loads(l) for l in open(os.path.join(args.data_dir, "questions.jsonl"), encoding="utf-8")]
    gold = {g["id"]: g for g in (json.loads(l) for l in open(os.path.join(args.data_dir, "answers.jsonl"), encoding="utf-8"))}
    cal = load_calendar(args.data_dir)
    guard = Guard()
    cards = {}
    X, y, groups, rows = [], [], [], []
    for q in sorted(qs, key=lambda q: q["business_id"]):
        b = q["business_id"]
        if b not in cards:
            bd = load_business_data(os.path.join(args.data_dir, "data", b), keep_time=True)
            cards[b] = build_business_card(bd, BusinessDB(bd, cal))
        a = analyze_question(q["question"], q["as_of_date"], cal)
        r = guard.assess(a, cards[b])
        is_unans = gold[q["id"]]["type"] == "unanswerable"
        X.append([r.features[f] for f in GUARD_FEATURES])
        y.append(int(is_unans))
        groups.append(b)
        rows.append((q, r, is_unans))

    X, y = np.array(X), np.array(y)
    print(f"{len(y)} questions, {y.sum()} unanswerable")
    for thr in (0.5, 0.6, 0.7, 0.8, 0.9):
        pred = np.array([r.p_unanswerable >= thr for _, r, _ in rows])
        tp, fp = int((pred & (y == 1)).sum()), int((pred & (y == 0)).sum())
        print(f"rule-based  thr {thr:.1f}: predicted {tp + fp:4d}  TP {tp:3d}  FP {fp:3d}  precision {tp / max(1, tp + fp):.3f}  recall {tp / max(1, y.sum()):.3f}  net PA gain {(2 * tp - fp - tp) / len(y):+.4f}")

    sig = collections.Counter()
    for _, r, u in rows:
        for s in r.signals:
            sig[(s, u)] += 1
    print("\nsignal -> (#gold unanswerable, #gold answerable)")
    for s in sorted({s for s, _ in sig}):
        print(f"  {s:32s} {sig[(s, True)]:4d} {sig[(s, False)]:4d}")

    fps = [(q, r) for q, r, u in rows if r.verdict == "unanswerable" and not u]
    fns = [(q, r) for q, r, u in rows if u and r.verdict != "unanswerable"]
    print(f"\nFalse 'unanswerable' ({len(fps)}):")
    for q, r in fps[: args.show]:
        print(f"  {q['business_id']} | {q['question'][:90]} | {r.signals}")
    print(f"\nMissed unanswerable ({len(fns)}):")
    for q, r in fns[: args.show]:
        print(f"  {q['business_id']} | {q['question'][:90]} | {r.signals}")

    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import GroupKFold
    except ImportError:
        print("scikit-learn not installed: skipping the learned Guard")
        return
    oof = np.zeros(len(y))
    for tr, te in GroupKFold(n_splits=5).split(X, y, groups):
        m = LogisticRegression(C=1.0, class_weight=None, max_iter=1000).fit(X[tr], y[tr])
        oof[te] = m.predict_proba(X[te])[:, 1]
    print("\nlogistic (GroupKFold by business, out-of-fold):")
    for thr in (0.5, 0.6, 0.7, 0.8, 0.9):
        pred = oof >= thr
        tp, fp = int((pred & (y == 1)).sum()), int((pred & (y == 0)).sum())
        print(f"  thr {thr:.1f}: predicted {tp + fp:4d}  TP {tp:3d}  FP {fp:3d}  precision {tp / max(1, tp + fp):.3f}  recall {tp / max(1, y.sum()):.3f}")
    if args.save:
        m = LogisticRegression(C=1.0, max_iter=1000).fit(X, y)
        with open(args.save, "w", encoding="utf-8") as f:
            json.dump({"features": GUARD_FEATURES, "coef": m.coef_[0].tolist(), "intercept": float(m.intercept_[0]),
                       "trained_on": args.data_dir, "n": int(len(y))}, f, indent=1)
        print(f"saved {args.save}")


if __name__ == "__main__":
    main()
