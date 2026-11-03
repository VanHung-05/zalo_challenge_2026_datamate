#!/usr/bin/env python3
"""Measure the deterministic fast path (no LLM) against train answers.
For each intent: answered / correct / wrong / refused (semantic layer raised). Wrong answers are
printed with their evidence so the root cause (role mapping, state mapping, time, rule) can be fixed.

  python scripts/eval_fast_path.py -d train [--business biz_101] [--show-wrong 30]
"""

import argparse
import collections
import json
import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.datamate.business_card import build_business_card  # noqa: E402
from src.datamate.db import BusinessDB  # noqa: E402
from src.datamate.fast_path import try_fast_path  # noqa: E402
from src.datamate.formatter import format_submission_item  # noqa: E402
from src.datamate.loader import load_business_data  # noqa: E402
from src.datamate.question import analyze_question  # noqa: E402
from src.datamate.rules import get_rulebook, load_calendar  # noqa: E402
from src.datamate.scoring import score  # noqa: E402
from src.datamate.semantic import CapabilityMissing, OutOfCoverage, SemanticLayer, SemanticUncertain  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--data-dir", default="train")
    ap.add_argument("-b", "--business", default=None)
    ap.add_argument("--show-wrong", type=int, default=40)
    ap.add_argument("--out", default=None, help="write per-question results (jsonl)")
    ap.add_argument("--priors-out", default=None, help="write per-intent precision (configs/fast_path_priors.json)")
    args = ap.parse_args()

    qs = [json.loads(l) for l in open(os.path.join(args.data_dir, "questions.jsonl"), encoding="utf-8")]
    gold = {}
    ans_path = os.path.join(args.data_dir, "answers.jsonl")
    if os.path.exists(ans_path):
        gold = {g["id"]: g for g in (json.loads(l) for l in open(ans_path, encoding="utf-8"))}
    if args.business:
        qs = [q for q in qs if q["business_id"] == args.business]
    qs.sort(key=lambda q: (q["business_id"], q["id"]))
    cal = load_calendar(args.data_dir)
    rb = get_rulebook(args.data_dir)

    stats = collections.defaultdict(collections.Counter)
    wrong, rows = [], []
    cur_biz, sem, card = None, None, None
    t0 = time.time()
    for q in qs:
        if q["business_id"] != cur_biz:
            cur_biz = q["business_id"]
            bd = load_business_data(os.path.join(args.data_dir, "data", cur_biz), keep_time=True)
            db = BusinessDB(bd, cal)
            card = build_business_card(bd, db)
            sem = SemanticLayer(db, card, rb.value("shifts"))
        a = analyze_question(q["question"], q["as_of_date"], cal)
        g = gold.get(q["id"])
        res = {"id": q["id"], "business_id": cur_biz, "question": q["question"]}
        try:
            fa, trace = try_fast_path(a, card, sem)
        except (CapabilityMissing, OutOfCoverage) as e:
            intent = "raised_unanswerable"
            res.update(outcome=type(e).__name__, detail=str(e))
            stats[intent]["n"] += 1
            if g is not None:
                stats[intent]["gold_unanswerable" if g["type"] == "unanswerable" else "gold_answerable"] += 1
            rows.append(res)
            continue
        except SemanticUncertain as e:
            res.update(outcome="uncertain", detail=str(e))
            stats["uncertain"]["n"] += 1
            if g is not None:
                stats["uncertain"]["gold_unanswerable" if g["type"] == "unanswerable" else "gold_answerable"] += 1
            rows.append(res)
            continue
        if fa is None:
            stats["no_intent"]["n"] += 1
            res.update(outcome="no_intent")
            rows.append(res)
            continue
        item = format_submission_item(q["id"], fa.value, fa.answer_type, is_money=fa.money,
                                      date_granularity=a.date_granularity)
        res.update(outcome="answered", intent=fa.intent, item=item)
        st = stats[fa.intent]
        st["n"] += 1
        if g is not None:
            s = score(item, g)
            st[{1: "correct", -1: "wrong", 0: "abstain"}[s]] += 1
            res["score"] = s
            if s == -1:
                wrong.append((fa, item, g, q, trace))
        rows.append(res)

    print(f"\n{len(qs)} questions in {time.time() - t0:.0f}s")
    print(f"{'intent':32s} {'n':>5s} {'correct':>8s} {'wrong':>6s}   note")
    tot = collections.Counter()
    for k, st in sorted(stats.items(), key=lambda x: -x[1]["n"]):
        note = ""
        if k in ("raised_unanswerable", "uncertain"):
            note = f"gold unanswerable {st['gold_unanswerable']}, answerable {st['gold_answerable']}"
        print(f"{k:32s} {st['n']:5d} {st['correct']:8d} {st['wrong']:6d}   {note}")
        tot.update(st)
    answered = tot["correct"] + tot["wrong"]
    if gold:
        print(f"\nanswered {answered}: correct {tot['correct']}, wrong {tot['wrong']}  "
              f"(precision {tot['correct'] / max(1, answered):.3f}); PA contribution {(tot['correct'] - tot['wrong']) / len(qs):+.4f}")
    for fa, item, g, q, trace in wrong[: args.show_wrong]:
        print(f"\n[{fa.intent}] {q['business_id']} {q['as_of_date']} | {q['question']}")
        print(f"   got {json.dumps(item.get('answer'), ensure_ascii=False)[:150]}  gold {json.dumps(g.get('answer'), ensure_ascii=False)[:150]}")
        print(f"   evidence {json.dumps(fa.evidence, ensure_ascii=False, default=str)[:400]}")
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")
    if args.priors_out and gold:
        # Wilson lower bound (z = 1.28, ~90 % one-sided): an intent seen 5/5 times is not trusted like 300/300.
        # Measured on the same split the rules were written on, so still optimistic for new businesses.
        z = 1.28
        prec, support = {}, {}
        for k, st in stats.items():
            n = st["correct"] + st["wrong"]
            if not n:
                continue
            p = st["correct"] / n
            lb = (p + z * z / (2 * n) - z * ((p * (1 - p) + z * z / (4 * n)) / n) ** 0.5) / (1 + z * z / n)
            prec[k], support[k] = round(lb, 4), n
        with open(args.priors_out, "w", encoding="utf-8") as f:
            json.dump({"precision": prec, "support": support, "split": args.data_dir,
                       "note": "Wilson lower bound of fast-path precision per intent, measured on this split"}, f, indent=1)
        print(f"priors written to {args.priors_out}")


if __name__ == "__main__":
    main()
