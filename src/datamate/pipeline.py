"""Per-question orchestration (plan §3):
question understanding (§4.3) -> Answerability Guard (§4.6) -> fast path on the semantic layer (§4.4)
-> LLM program synthesis with self-consistency (§4.5) -> calibrated decision (§4.7) -> formatter / log (§4.8).
"""

import json
import os
import re
import time
import traceback
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from src.datamate.business_card import BusinessCard, build_business_card
from src.datamate.db import BusinessDB
from src.datamate.decision import Calibrator, Candidate, decide, sanity_checks
from src.datamate.fast_path import find_entities, try_fast_path
from src.datamate.formatter import format_submission_item
from src.datamate.guard import Guard, coverage_reliable
from src.datamate.loader import BusinessData, load_business_data
from src.datamate.question import QuestionAnalysis, analyze_question
from src.datamate.rules import Rulebook
from src.datamate.semantic import CapabilityMissing, OutOfCoverage, SemanticLayer, SemanticUncertain


@dataclass
class BusinessContext:
    bdata: BusinessData
    db: BusinessDB
    card: BusinessCard
    sem: SemanticLayer
    known_names: set = field(default_factory=set)
    load_seconds: float = 0.0


def build_context(biz_path: str, calendar_df: Optional[pd.DataFrame], rulebook: Rulebook, config: Dict[str, Any],
                  cache_dir: Optional[str] = None) -> BusinessContext:
    t0 = time.time()
    bdata = load_business_data(biz_path, calendar_df=calendar_df, keep_time=True)
    db = BusinessDB(bdata, calendar_df, threads=config.get("agent", {}).get("sql_threads", 1))
    card = build_business_card(bdata, db)
    sem = SemanticLayer(db, card, rulebook.value("shifts"))
    if cache_dir:  # Business Card cached for inspection / logs (plan §4.1: built once per business)
        try:
            os.makedirs(cache_dir, exist_ok=True)
            card.save(os.path.join(cache_dir, f"{bdata.business_id}.card.json"))
        except Exception as e:
            print(f"[Warning] could not cache card: {e}")
    return BusinessContext(bdata, db, card, sem, card.known_names(("product", "customer", "staff", "supplier", "product_group", "customer_group")),
                           round(time.time() - t0, 2))


MONEY_RE = r"tien|doanh thu|dthu|\bgia\b|gia (ban|nhap|von|tri)|\bno\b|cong no|lai|loi nhuan|chi phi|von|thanh toan|voucher|phi ship|phi giao|\bdong\b|vnd"
NOT_MONEY_RE = r"\bso (don|ly|lon|chai|hop|goi|khach|ngay|lan|luot|mat hang|san pham)|bao nhieu (don|ly|lon|chai|hop|goi|khach|ngay|lan|luot|cai|vien|kg)|may (don|ly|lon|chai|hop|goi|khach|ngay|lan|luot|cai|vien|kg)"


def is_money_question(a: QuestionAnalysis) -> bool:
    """Money answers are integer VND, rounded half-up (rulebook §9); counts / quantities are not money."""
    if a.answer_type not in ("number", "list", "table"):
        return False
    return bool(re.search(MONEY_RE, a.ascii)) and not re.search(NOT_MONEY_RE, a.ascii)


# Question wording that depends on a rulebook definition: if this split's rulebook does not define it, the
# answer would rest on our assumption -> abstain (plan §4.2 "không tự đoán")
RULE_DEPENDENT = {
    r"khach sop": "big_spender_min_orders",
    r"trung thanh": "loyal_customer_metric",
    r"ua chuong|duoc yeu thich": "favourite_item_metric",
    r"gio cao diem": "peak_hour_bucket",
}


def risk_flags_for(a: QuestionAnalysis, ctx: BusinessContext, rulebook: Rulebook, used_fast_path: bool,
                   min_sales_confidence: float = 0.4) -> List[str]:
    flags: List[str] = []
    uncertain = set(rulebook.uncertain_params())
    for pattern, param in RULE_DEPENDENT.items():
        if re.search(pattern, a.ascii) and param in uncertain:
            flags.append("rule_uncertain")
    if not a.time.confident and a.time.matched:
        flags.append("time_uncertain")
    if used_fast_path and ctx.card.sales.confidence < min_sales_confidence:
        flags.append("low_confidence_role")
    return flags


def load_fast_path_priors(path: Optional[str]) -> Dict[str, float]:
    """Per-intent precision measured on train by scripts/eval_fast_path.py --priors-out."""
    if path and os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f).get("precision", {})
    return {}


def answer(
    q: Dict[str, Any],
    ctx: BusinessContext,
    rulebook: Rulebook,
    calendar_df: Optional[pd.DataFrame],
    config: Dict[str, Any],
    llm: Any = None,
    guard: Optional[Guard] = None,
    calibrator: Optional[Calibrator] = None,
    priors: Optional[Dict[str, float]] = None,
    deadline: Optional[float] = None,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Returns (submission item, log)."""
    dcfg = config.get("decision", {})
    acfg = config.get("agent", {})
    qid = q["id"]
    log: Dict[str, Any] = {}
    guard = guard or Guard(threshold=float(dcfg.get("tau_unanswerable", 0.8)))

    a = analyze_question(q["question"], q.get("as_of_date", ""), calendar_df)
    log["analysis"] = {
        "answer_type": a.answer_type, "answer_type_confidence": a.answer_type_confidence, "top_k": a.top_k,
        "time": {"primary": str(a.time.primary) if a.time.primary else None, "compare": str(a.time.compare) if a.time.compare else None,
                 "periods": [str(p) for p in a.time.periods], "point_in_time": str(a.time.point_in_time) if a.time.point_in_time else None,
                 "shift": a.time.shift, "confident": a.time.confident, "matched": a.time.matched},
        "thresholds": a.thresholds, "concepts": a.concepts, "overrides": a.overrides,
        "flags": {"subjective": a.subjective, "external_knowledge": a.external_knowledge, "forecast": a.forecast},
    }

    g = guard.assess(a, ctx.card)
    log["guard"] = g.to_log()
    p_unans, unans_reason = g.p_unanswerable, g.reason
    cands: List[Candidate] = []
    flags: List[str] = []
    money = is_money_question(a)

    # Fast path (deterministic). agent.fast_path: false -> answers come from the LLM only; the semantic
    # tables sl_* stay available to the LLM, and the Guard still covers out-of-coverage periods.
    fa = None
    if not acfg.get("fast_path", True):
        log["fast_path"] = {"disabled": True}
    else:
        try:
            fa, fp_trace = try_fast_path(a, ctx.card, ctx.sem)
            log["fast_path"] = fp_trace
            if fp_trace.get("ambiguous_entities"):
                flags.append("entity_ambiguous")
        except (CapabilityMissing, OutOfCoverage) as e:
            log["fast_path"] = {"raised": type(e).__name__, "detail": str(e)}
            # A period outside a coverage that agrees with the README is rulebook §7 evidence. A missing role
            # is often our own detection failure on an unfamiliar file format (public test), so it is weak.
            if isinstance(e, OutOfCoverage) and coverage_reliable(ctx.card):
                p_sem = float(dcfg.get("p_unanswerable_out_of_coverage", 0.85))
            else:
                p_sem = float(dcfg.get("p_unanswerable_capability", 0.5))
                flags.append("capability_unclear")
            if p_sem > p_unans:
                p_unans, unans_reason = p_sem, str(e)
        except SemanticUncertain as e:
            log["fast_path"] = {"raised": "SemanticUncertain", "detail": str(e)}
            flags.append("semantic_uncertain")
        except Exception:
            log["fast_path"] = {"raised": "error", "detail": traceback.format_exc()[-1500:]}
        if fa is not None:
            item = format_submission_item(qid, fa.value, fa.answer_type, is_money=fa.money, date_granularity=a.date_granularity)
            if item.get("type") != "abstain":
                checks = sanity_checks(item, a.answer_type, a.top_k, ctx.known_names if fa.answer_type in ("string", "list") else None)
                meta = {"intent": fa.intent, "evidence": fa.evidence}
                if priors and fa.intent in priors:
                    meta["prior"] = priors[fa.intent]
                cands.append(Candidate(item, "fast_path", True, checks, [], meta))
                log["fast_path"]["item"] = item

    # LLM program synthesis (skipped when the Guard is sure, or the fast path is enough, or time is up)
    run_llm = llm is not None and g.verdict != "unanswerable"
    if run_llm and cands and not acfg.get("llm_verify_fast_path", False):
        run_llm = False
    if run_llm and deadline is not None and time.time() > deadline:
        run_llm = False
        flags.append("question_timeout")
    if run_llm:
        ents, _amb = find_entities(a, ctx.card, ctx.sem) if ctx.sem.ok else ([], [])
        extra = "BUSINESS CARD:\n" + ctx.card.to_prompt()
        if ctx.sem.ok:
            extra += "\n\n" + ctx.sem.view_description()
        llm_trace: Dict[str, Any] = {}
        try:
            cands += generate_llm_candidates(qid, a, ctx, rulebook, extra, config, llm, ents, money, llm_trace)
        except Exception:
            llm_trace["error"] = traceback.format_exc()[-1500:]
        log["llm"] = llm_trace
        n_llm_unans = sum(1 for c in cands if c.source == "llm" and c.item.get("type") == "unanswerable")
        n_llm = sum(1 for c in cands if c.source == "llm")
        # The LLM saying "unanswerable" is weak evidence on its own (plan §4.6): only agreement across samples counts
        if n_llm and n_llm_unans == n_llm and n_llm >= 2:
            p_llm = float(dcfg.get("p_unanswerable_llm_unanimous", 0.6))
            if p_llm > p_unans:
                p_unans, unans_reason = p_llm, "các lời giải độc lập đều kết luận dữ liệu không đủ"

    flags += risk_flags_for(a, ctx, rulebook, used_fast_path=any(c.source == "fast_path" for c in cands),
                            min_sales_confidence=float(dcfg.get("min_sales_confidence", 0.4)))
    d = decide(qid, cands, p_unans, unans_reason, flags, a.answer_type_confidence,
               tau=float(dcfg.get("tau", 0.7)), tau_unanswerable=float(dcfg.get("tau_unanswerable", 0.8)),
               calibrator=calibrator)
    log["decision"] = {"action": d.action, "reason": d.reason, "p_value": d.p_value, "p_unanswerable": d.p_unanswerable,
                       "features": d.features, "clusters": d.clusters, "risk_flags": sorted(set(flags))}
    log["candidates"] = [{"source": c.source, "item": c.item, "exec_ok": c.exec_ok, "checks": c.checks, "flags": c.flags,
                          "meta": {k: v for k, v in c.meta.items() if k != "evidence"}} for c in cands]
    return d.item, log


def generate_llm_candidates(qid: str, a: QuestionAnalysis, ctx: BusinessContext, rulebook: Rulebook, extra_schema: str,
                            config: Dict[str, Any], llm: Any, ents: List[Any], money: bool, trace: Dict[str, Any]) -> List[Candidate]:
    from src.datamate.sql_agent import generate_sql_candidates
    return generate_sql_candidates(qid, a, ctx.db, ctx.bdata.readme, rulebook.raw_text, extra_schema, config, llm,
                                   entity_hits=ents, is_money=money, known_names=ctx.known_names or None, trace=trace)
