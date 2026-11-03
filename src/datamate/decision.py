"""Aggregation & calibrated decision (plan §1.2, §4.7).
Candidates from the fast path, the LLM samples and the Answerability Guard are clustered with the
organizer's matching rule (scoring.same_answer), turned into features, and mapped to
p_v = P(best value is correct) and p_u = P(reference is unanswerable). The action with the highest
expected score wins: value 2·p_v − 1, unanswerable 2·p_u − 1, abstain 0, with a safety threshold τ
instead of 0.5 because estimated probabilities are optimistic and decisions must be reproducible.

Until a calibrator has been fitted on logged runs (scripts/train_calibrator.py) the probabilities
come from HEURISTIC_PRIORS below. Those numbers are hand-set placeholders, not measured.
"""

import json
import math
import os
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from src.datamate.scoring import same_answer

ANSWER_TYPES = ("number", "percent", "string", "date", "boolean", "list", "table")

# Uncalibrated starting points (see module docstring); replaced by Calibrator once trained
HEURISTIC_PRIORS = {
    "fast_path": 0.88,        # deterministic semantic-layer answer on a verified mapping
    "llm": 0.60,              # single LLM sample that executed and passed checks
    "per_extra_agreeing": 0.08,
    "guard_unanswerable": 0.85,
}

# Risk flags that force abstain whatever the score (plan §4.7 "veto cứng")
HARD_VETO_FLAGS = {
    "candidates_disagree", "entity_ambiguous", "sandbox_timeout", "rule_uncertain",
    "low_confidence_role", "type_mismatch",
}


@dataclass
class Candidate:
    item: Dict[str, Any]                 # submission-shaped {"type", "answer"} (or unanswerable)
    source: str                          # "fast_path" | "llm" | "guard"
    exec_ok: bool = True
    checks: Dict[str, bool] = field(default_factory=dict)   # sanity check name -> passed
    flags: List[str] = field(default_factory=list)          # risk flags raised while producing it
    meta: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_value(self) -> bool:
        return self.item.get("type") in ANSWER_TYPES

    @property
    def checks_ok(self) -> bool:
        return all(self.checks.values())


@dataclass
class Decision:
    item: Dict[str, Any]
    action: str                          # "value" | "unanswerable" | "abstain"
    p_value: float
    p_unanswerable: float
    reason: str
    features: Dict[str, float] = field(default_factory=dict)
    clusters: List[Dict[str, Any]] = field(default_factory=list)

    def to_log(self) -> Dict[str, Any]:
        return asdict(self)


def cluster_candidates(cands: Sequence[Candidate]) -> List[List[Candidate]]:
    """Group value candidates that the organizer would score as the same answer.
    Order is deterministic: by size, then by first appearance."""
    clusters: List[List[Candidate]] = []
    for c in cands:
        if not c.is_value or not c.exec_ok:
            continue
        for cl in clusters:
            if same_answer(cl[0].item, c.item):
                cl.append(c)
                break
        else:
            clusters.append([c])
    order = {id(cl): i for i, cl in enumerate(clusters)}
    clusters.sort(key=lambda cl: (-len(cl), order[id(cl)]))
    return clusters


def sanity_checks(item: Dict[str, Any], expected_type: Optional[str] = None, top_k: Optional[int] = None,
                  known_names: Optional[set] = None, is_count: bool = False) -> Dict[str, bool]:
    """Cheap plausibility checks on a formatted answer (plan §4.5 "Kiểm tra hợp lý tự động")."""
    from src.datamate.scoring import normalize_string
    t, a = item.get("type"), item.get("answer")
    checks: Dict[str, bool] = {}
    if expected_type:
        checks["type_matches_question"] = t == expected_type
    if t in ("number", "percent"):
        checks["finite"] = isinstance(a, (int, float)) and math.isfinite(a)
        if t == "percent" and checks["finite"]:
            checks["percent_magnitude"] = abs(a) <= 1000
            # 0 < |p| < 1 is legal (0.5 %) but is also what a forgotten ×100 looks like
            checks["percent_not_ratio_like"] = not (0 < abs(a) < 1)
        if t == "number" and is_count and checks["finite"]:
            checks["count_non_negative_int"] = a >= 0 and float(a).is_integer()
    if t == "list":
        checks["list_non_empty"] = isinstance(a, list) and len(a) > 0
        if top_k and isinstance(a, list):
            checks["list_len_le_k"] = len(a) <= top_k
        if known_names and isinstance(a, list) and all(isinstance(x, str) for x in a):
            checks["names_in_data"] = all(normalize_string(x) in known_names for x in a)
    if t == "string" and known_names:
        checks["name_in_data"] = normalize_string(a) in known_names
    if t == "table":
        checks["table_non_empty"] = isinstance(a, dict) and bool(a.get("rows"))
    return checks


def build_features(cands: Sequence[Candidate], clusters: List[List[Candidate]], guard_p_unans: float,
                   risk_flags: Sequence[str], answer_type_confidence: float) -> Dict[str, float]:
    n_value = sum(1 for c in cands if c.is_value and c.exec_ok)
    top = clusters[0] if clusters else []
    return {
        "n_candidates": float(len(cands)),
        "n_value": float(n_value),
        "n_clusters": float(len(clusters)),
        "top_size": float(len(top)),
        "agreement": len(top) / n_value if n_value else 0.0,
        "top_has_fast_path": float(any(c.source == "fast_path" for c in top)),
        "top_checks_ok": float(all(c.checks_ok for c in top)) if top else 0.0,
        "guard_p_unans": float(guard_p_unans),
        "n_risk_flags": float(len(set(risk_flags))),
        "answer_type_confidence": float(answer_type_confidence),
        "llm_said_unanswerable": float(sum(1 for c in cands if c.source == "llm" and c.item.get("type") == "unanswerable")),
    }


FEATURE_ORDER = [
    "n_candidates", "n_value", "n_clusters", "top_size", "agreement", "top_has_fast_path",
    "top_checks_ok", "guard_p_unans", "n_risk_flags", "answer_type_confidence", "llm_said_unanswerable",
]


class Calibrator:
    """Logistic model P(top value correct | features), fitted with group-CV by business
    (scripts/train_calibrator.py). Stored as plain JSON so inference needs no sklearn and is exact."""

    def __init__(self, coef: Optional[List[float]] = None, intercept: float = 0.0, meta: Optional[Dict[str, Any]] = None):
        self.coef = coef
        self.intercept = intercept
        self.meta = meta or {}

    @property
    def fitted(self) -> bool:
        return self.coef is not None

    def predict(self, features: Dict[str, float]) -> float:
        z = self.intercept + sum(w * features.get(n, 0.0) for w, n in zip(self.coef or [], FEATURE_ORDER))
        return 1.0 / (1.0 + math.exp(-max(-50.0, min(50.0, z))))

    @classmethod
    def load(cls, path: Optional[str]) -> "Calibrator":
        if path and os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                d = json.load(f)
            if d.get("features") != FEATURE_ORDER:
                print(f"[Calibrator Warning] {path} was fitted on other features; using heuristic priors")
                return cls()
            return cls(d["coef"], d["intercept"], d.get("meta"))
        return cls()

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"features": FEATURE_ORDER, "coef": self.coef, "intercept": self.intercept, "meta": self.meta}, f, indent=1)


def heuristic_p_value(top: List[Candidate]) -> float:
    if not top:
        return 0.0
    fast = [c for c in top if c.source == "fast_path"]
    if fast:
        # per-intent precision measured on train (scripts/eval_fast_path.py --priors-out), else the default
        p = fast[0].meta.get("prior", HEURISTIC_PRIORS["fast_path"])
    else:
        p = HEURISTIC_PRIORS["llm"]
    p += HEURISTIC_PRIORS["per_extra_agreeing"] * (len(top) - 1)
    if not all(c.checks_ok for c in top):
        p -= 0.3
    return max(0.0, min(0.97, p))


def decide(
    question_id: str,
    cands: Sequence[Candidate],
    guard_p_unans: float = 0.0,
    guard_reason: str = "",
    risk_flags: Sequence[str] = (),
    answer_type_confidence: float = 1.0,
    tau: float = 0.7,
    tau_unanswerable: float = 0.8,
    calibrator: Optional[Calibrator] = None,
) -> Decision:
    """Pick value / unanswerable / abstain (plan §1.2)."""
    clusters = cluster_candidates(cands)
    top = clusters[0] if clusters else []
    flags = set(risk_flags)
    for c in top:
        flags.update(c.flags)
    n_value = sum(len(cl) for cl in clusters)
    if len(clusters) > 1 and len(top) * 2 <= n_value:
        flags.add("candidates_disagree")
    if top and not all(c.checks.get("type_matches_question", True) for c in top):
        flags.add("type_mismatch")

    feats = build_features(cands, clusters, guard_p_unans, sorted(flags), answer_type_confidence)
    if calibrator is not None and calibrator.fitted:
        p_v = calibrator.predict(feats) if top else 0.0
    else:
        p_v = heuristic_p_value(top)
    p_u = guard_p_unans

    cl_log = [{"size": len(cl), "sources": [c.source for c in cl], "item": cl[0].item} for cl in clusters]
    veto = sorted(flags & HARD_VETO_FLAGS)

    def done(item: Dict[str, Any], action: str, reason: str) -> Decision:
        return Decision(dict(item, id=question_id), action, round(p_v, 4), round(p_u, 4), reason, feats, cl_log)

    if p_u >= tau_unanswerable and p_u >= p_v:
        return done({"type": "unanswerable", "reason": guard_reason or "Dữ liệu không đủ để trả lời"}, "unanswerable",
                    f"p_u={p_u:.2f} >= {tau_unanswerable}")
    if veto:
        return done({"type": "abstain"}, "abstain", "veto: " + ", ".join(veto))
    if top and p_v >= tau:
        item = {k: v for k, v in top[0].item.items() if k in ("type", "answer")}
        return done(item, "value", f"p_v={p_v:.2f} >= {tau}")
    return done({"type": "abstain"}, "abstain", f"p_v={p_v:.2f} < {tau}" if top else "no executable candidate")
