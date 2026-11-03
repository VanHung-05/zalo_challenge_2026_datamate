#!/usr/bin/env python3
"""Fit the answer calibrator (plan §4.7) from the logs of a real run on train.

Input: the per-question logs of `runner.py` on train (logs/<id>.json, they hold decision.features
and the clusters) and train/answers.jsonl. Label = the top cluster's answer is correct.
Model: logistic regression on decision.FEATURE_ORDER, out-of-fold by business (GroupKFold), then a
sweep of tau on the out-of-fold probabilities to maximise PA. --save writes the JSON model read
by decision.Calibrator (no sklearn needed at inference).

  python scripts/train_calibrator.py --logs outputs/train_full/logs -g train/answers.jsonl -q train/questions.jsonl --save configs/calibrator.json
"""

import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np  # noqa: E402

from src.datamate.decision import FEATURE_ORDER, Calibrator  # noqa: E402
from src.datamate.scoring import match_answer  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", required=True)
    ap.add_argument("-g", "--gold", required=True)
    ap.add_argument("-q", "--questions", required=True)
    ap.add_argument("--save", default=None)
    args = ap.parse_args()

    gold = {g["id"]: g for g in (json.loads(l) for l in open(args.gold, encoding="utf-8"))}
    biz = {q["id"]: q["business_id"] for q in (json.loads(l) for l in open(args.questions, encoding="utf-8"))}
    X, y, groups = [], [], []
    for path in sorted(glob.glob(os.path.join(args.logs, "*.json"))):
        log = json.load(open(path, encoding="utf-8"))
        qid, dec = log.get("id"), log.get("decision")
        if qid not in gold or not dec or not dec.get("clusters"):
            continue
        top = dec["clusters"][0]["item"]
        X.append([float(dec["features"].get(f, 0.0)) for f in FEATURE_ORDER])
        y.append(int(match_answer(top, gold[qid])))
        groups.append(biz.get(qid, "?"))
    if len(set(y)) < 2 or len(set(groups)) < 3:
        print(f"not enough data to fit: {len(y)} questions with candidates, labels {set(y)}, {len(set(groups))} businesses")
        return
    X, y = np.array(X), np.array(y)
    print(f"{len(y)} questions with a candidate; top candidate correct for {y.mean():.3f}")

    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold
    oof = np.zeros(len(y))
    for tr, te in GroupKFold(n_splits=min(5, len(set(groups)))).split(X, y, groups):
        oof[te] = LogisticRegression(max_iter=2000).fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]

    # PA over these questions if we answer when p >= tau, abstain otherwise
    print(f"{'tau':>5s} {'answered':>9s} {'correct':>8s} {'wrong':>6s} {'PA':>8s}")
    best = (None, -1e9)
    for tau in np.arange(0.3, 0.96, 0.05):
        ans = oof >= tau
        c, w = int((ans & (y == 1)).sum()), int((ans & (y == 0)).sum())
        pa = (c - w) / len(y)
        print(f"{tau:5.2f} {int(ans.sum()):9d} {c:8d} {w:6d} {pa:+8.4f}")
        if pa > best[1]:
            best = (round(float(tau), 2), pa)
    print(f"best tau on out-of-fold probabilities: {best[0]} (PA {best[1]:+.4f}); "
          "prefer a tau a little above it so small shifts do not flip many decisions (plan §6.4)")

    if args.save:
        m = LogisticRegression(max_iter=2000).fit(X, y)
        Calibrator(m.coef_[0].tolist(), float(m.intercept_[0]),
                   {"n": int(len(y)), "best_tau_oof": best[0], "logs": args.logs}).save(args.save)
        print(f"saved {args.save}")


if __name__ == "__main__":
    main()
