"""Answerability Guard (plan §4.6).
Decides whether a question is unanswerable from the released data (rulebook §7), before and
independently of any generated program. Answering "unanswerable" wrongly costs −1 like any wrong
answer, so the Guard separates:
  - strong, deterministic evidence that the data cannot answer -> high p_unanswerable
  - doubt (could be missing data, could be our misunderstanding) -> flagged, never "unanswerable"

Signals:
  subjective / external knowledge / forecast questions (question.py flags)
  a concept the question needs that the Business Card's capability map says is absent
  the asked (or compared) period lies entirely outside the dates the data covers
A logistic model over the same signals can be fitted on train (scripts/train_guard.py); without
it, GUARD_RULE_P below are used. Those defaults are hand-set and must be checked with the script.
"""

import json
import math
import os
import re
from dataclasses import asdict, dataclass, field
from datetime import date
from typing import Any, Dict, List, Optional

from src.datamate.business_card import BusinessCard
from src.datamate.question import QuestionAnalysis

# Hand-set probabilities per signal (placeholders until scripts/train_guard.py measures them)
GUARD_RULE_P = {
    "subjective": 0.92,
    "external_knowledge": 0.92,
    "forecast": 0.85,
    "out_of_coverage": 0.9,
    "missing_concept": 0.85,
    "missing_concept_low_conf": 0.6,
}

# Concepts whose absence is decisive. Kept to those with clean evidence on train (scripts/train_guard.py:
# e.g. customer_gender 4/0, debt 4/0, customer_address 6/0) AND whose data is hard to miss. Absences of
# product groups, staff, channels, warranty... were mostly *our* detection failures on new formats
# (public test: repair / phone shops), so they only mark the question uncertain (-> abstain).
DECISIVE_CONCEPTS = {
    "customer_gender", "customer_address", "customer_age", "debt", "purchase", "supplier", "cost",
    "gross_profit", "net_profit", "expense", "expiry", "payment_method", "stock", "cancel_reason", "return_reason",
}
COVERAGE_README_TOLERANCE_DAYS = 7

GUARD_FEATURES = ["subjective", "external_knowledge", "forecast", "out_of_coverage", "n_missing_decisive",
                  "n_missing_other", "readme_none", "no_sales_model"]


@dataclass
class GuardResult:
    p_unanswerable: float
    verdict: str                         # "unanswerable" | "answerable" | "uncertain"
    reason: str = ""
    signals: List[str] = field(default_factory=list)
    features: Dict[str, float] = field(default_factory=dict)

    def to_log(self) -> Dict[str, Any]:
        return asdict(self)


def coverage_reliable(card: BusinessCard) -> bool:
    """The sales coverage must end within a few days of the README's stated last data day (when given)."""
    end, readme_end = card.coverage[1], card.readme_data_end
    if not end:
        return False
    if not readme_end:
        return True
    try:
        return abs((date.fromisoformat(end) - date.fromisoformat(readme_end)).days) <= COVERAGE_README_TOLERANCE_DAYS
    except ValueError:
        return False


def _out_of_coverage(p_start: date, p_end: date, cov: tuple) -> bool:
    cmin, cmax = cov
    return bool(cmin and cmax and (str(p_end) < cmin or str(p_start) > cmax))


class Guard:
    def __init__(self, weights_path: Optional[str] = None, threshold: float = 0.8):
        self.threshold = threshold
        self.coef: Optional[List[float]] = None
        self.intercept = 0.0
        if weights_path and os.path.exists(weights_path):
            with open(weights_path, "r", encoding="utf-8") as f:
                d = json.load(f)
            if d.get("features") == GUARD_FEATURES:
                self.coef, self.intercept = d["coef"], d["intercept"]

    @staticmethod
    def _concepts_outside_entities(a: QuestionAnalysis, card: BusinessCard) -> List[str]:
        """Concepts triggered by words of the question that are not part of a data name: the product
        "Tôn lạnh Hoa Sen" is not a stock question, "Bay thợ hồ" is not about staff."""
        from src.datamate.fast_path import find_entities
        from src.datamate.question import _concepts
        try:
            ents, _ = find_entities(a, card, None)
        except Exception:
            return list(a.concepts)
        if not ents:
            return list(a.concepts)
        masked = a.ascii
        for e in sorted(ents, key=lambda e: -len(e.phrase)):
            words = [re.escape(w) for w in e.phrase.split()]
            masked = re.sub(r"\b" + r"[^a-z0-9]+".join(words) + r"\b", " ", masked)
        return [c for c in _concepts(masked) if c in a.concepts]

    def assess(self, a: QuestionAnalysis, card: BusinessCard) -> GuardResult:
        signals: List[str] = []
        reasons: List[str] = []
        feats = {k: 0.0 for k in GUARD_FEATURES}

        for flag, why in (("subjective", "câu hỏi về ý kiến/thái độ, dữ liệu không ghi"),
                          ("external_knowledge", "cần kiến thức ngoài dữ liệu"),
                          ("forecast", "câu hỏi dự báo tương lai")):
            if getattr(a, flag):
                feats[flag] = 1.0
                signals.append(flag)
                reasons.append(why)

        periods = [p for p in [a.time.primary, a.time.compare] + list(a.time.periods) if p is not None]
        out = [p for p in periods if _out_of_coverage(p.start, p.end, card.coverage)]
        if out and not coverage_reliable(card):
            # Our coverage disagrees with the README's "có đến hết ..." date: probably part of the data
            # was not recognized (e.g. a yearly file not merged), so this is no evidence
            signals.append("out_of_coverage_unreliable")
            out = []
        if out:
            feats["out_of_coverage"] = 1.0
            signals.append("out_of_coverage")
            reasons.append(f"kỳ {out[0].start}..{out[0].end} nằm ngoài khoảng dữ liệu {card.coverage[0]}..{card.coverage[1]}")

        for concept in self._concepts_outside_entities(a, card):
            cap = card.capabilities.get(concept)
            if cap is None or cap.present:
                continue
            if concept in DECISIVE_CONCEPTS and cap.confidence >= 0.7:
                feats["n_missing_decisive"] += 1
                signals.append(f"missing:{concept}")
                reasons.append(f"dữ liệu không có thông tin {concept}")
            else:
                feats["n_missing_other"] += 1
                signals.append(f"missing_low_conf:{concept}")
        feats["readme_none"] = float(card.readme_quality == "none")
        feats["no_sales_model"] = float(not (card.sales.line_table or card.sales.header_table))

        if self.coef is not None:
            z = self.intercept + sum(w * feats[n] for w, n in zip(self.coef, GUARD_FEATURES))
            p = 1.0 / (1.0 + math.exp(-max(-50.0, min(50.0, z))))
        else:
            ps = []
            for s in signals:
                key = s.split(":", 1)[0]
                if key == "out_of_coverage_unreliable":
                    continue
                ps.append(GUARD_RULE_P["missing_concept" if key == "missing" else
                                       "missing_concept_low_conf" if key == "missing_low_conf" else key])
            # independent-evidence combination: 1 - prod(1 - p)
            p = 1.0 - math.prod(1.0 - x for x in ps) if ps else 0.0

        if p >= self.threshold:
            verdict = "unanswerable"
        elif signals:
            verdict = "uncertain"
        else:
            verdict = "answerable"
        return GuardResult(round(p, 4), verdict, "; ".join(reasons), signals, feats)
