"""Deterministic fast paths (plan §3 [4c]): frequent question shapes answered by semantic-layer calls.
Each intent is a narrow pattern; anything outside it returns None and goes to the LLM agent.
A fast path never guesses: an ambiguous entity, an unmapped order state or a missing column
raises (see semantic.py) and the caller turns it into abstain / unanswerable.
"""

import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from src.datamate.business_card import BusinessCard, ascii_key
from src.datamate.question import QuestionAnalysis
from src.datamate.semantic import SemanticLayer, SemanticUncertain


@dataclass
class EntityHit:
    kind: str
    value: str          # spelling as in the data (used as filter value)
    phrase: str         # ascii phrase matched in the question
    column: str         # sl_lines column to filter on


@dataclass
class FastAnswer:
    intent: str
    value: Any
    answer_type: str
    money: bool = False
    evidence: Dict[str, Any] = field(default_factory=dict)


KIND_TO_COLUMN = {"product": "product_name", "product_group": "product_group", "customer_group": "customer_group",
                  "customer": "customer_name", "staff": "staff"}
MIN_ENTITY_CHARS = 4


def find_entities(a: QuestionAnalysis, card: BusinessCard, sem: SemanticLayer) -> Tuple[List[EntityHit], List[str]]:
    """Entity names of the data that occur as whole phrases in the (accent-free) question.
    Returns (hits, ambiguities). Longer matches win over their substrings."""
    text = " " + ascii_key(a.ascii) + " "
    cands: List[Tuple[int, int, EntityHit]] = []
    for e in card.entities:
        col = KIND_TO_COLUMN.get(e.kind)
        if not col:
            continue
        for name in e.names + ([e.code] if e.code else []):
            key = ascii_key(name)
            if len(key) < MIN_ENTITY_CHARS:
                continue
            pos = text.find(" " + key + " ")
            if pos >= 0:
                # File có mã thì theo mã (rulebook §8): filter on the code, else on the (normalized) name
                code_col = {"product": "product_code", "customer": "customer_code"}.get(e.kind)
                if e.code and code_col and code_col in card.sales.cols:
                    hit = EntityHit(e.kind, e.code, key, code_col)
                else:
                    hit = EntityHit(e.kind, name, key, col)
                cands.append((pos, pos + len(key), hit))
    # drop matches contained in a longer match
    hits: List[EntityHit] = []
    for s, t, h in sorted(cands, key=lambda x: -(x[1] - x[0])):
        if any(s >= s2 and t <= t2 and (s, t) != (s2, t2) for s2, t2, _ in [(c[0], c[1], c[2]) for c in cands if (c[1] - c[0]) > (t - s)]):
            continue
        if not any(h.phrase == x.phrase and h.kind == x.kind for x in hits):
            hits.append(h)
    ambiguous = []
    by_phrase: Dict[str, List[EntityHit]] = {}
    for h in hits:
        by_phrase.setdefault(h.phrase, []).append(h)
    for phrase, hs in by_phrase.items():
        if len({(h.kind, h.value) for h in hs}) > 1 and len({h.kind for h in hs}) == 1:
            ambiguous.append(phrase)
    return hits, ambiguous


def _has(a: QuestionAnalysis, pattern: str) -> bool:
    return re.search(pattern, a.ascii) is not None


REV = r"doanh thu|dthu|\bban (duoc|dc|ra)?.{0,20}\btien\b|tien (ban|hang)|thu ve|revenue|bao nhieu tien|nhieu tien|chi tieu|chi nhieu|tong tien"
EXCLUDE_TOTAL = (r"trung binh|binh quan|moi don|1 don|mot don|\bnhom\b|thanh toan|tien mat|chuyen khoan|vi dien tu|\bcod\b|\bkenh\b|"
                 r"zalo|dien thoai|tai quay|online|\bsan\b|khach|nhan vien|\btho\b|gia von|\blai\b|loi nhuan|\bnhap\b|cat hang|\bno\b|cong no|\bton\b|"
                 r"\bphi\b|voucher|\bvat\b|\bthue\b|ca sang|ca chieu|\bsize\b|\bly\b|\blon\b|\bchai\b|\bhop\b|\bgoi\b|\bkg\b|\bnguyen\b|\ble\b|tra hang|\bhuy\b|"
                 r"\bchenh\b|so voi|so vs|\btang\b|\bgiam\b|%|phan tram|ti le|ti trong|chiem|moi ngay|moi thang|lan luot|\btung\b|"
                 r"hoat chat|ke don|thuoc co|cac thuoc|\bloai\b|danh muc|mat hang|san pham|\bsp\b|\bmon\b|thu \d|chu nhat|cuoi tuan")


def intent_total_revenue(a, card, sem, ents) -> Optional[FastAnswer]:
    if a.answer_type != "number" or not a.time.primary or a.time.compare or a.time.periods or a.top_k or a.thresholds:
        return None
    if not _has(a, REV) or _has(a, EXCLUDE_TOTAL) or ents:
        return None
    # "mấy ngày / bao nhiêu đơn ... đạt X" counts things, it is not the period's revenue
    if _has(a, r"(may|bao nhieu|bn|nhieu|dem|so)\s+(ngay|don|bill|hoa don|lan|tuan|thang)\b|\bdon co\b|\bngay co\b"):
        return None
    r = sem.revenue(a.time.primary, a.as_of, shift=a.time.shift)
    return FastAnswer("total_revenue", r.value, "number", True, r.evidence)


def intent_order_count(a, card, sem, ents) -> Optional[FastAnswer]:
    if a.answer_type != "number" or not a.time.primary or a.time.compare or a.thresholds or ents:
        return None
    if not _has(a, r"\b(so|bao nhieu|may|dem|nhieu|bnhiu|bn)\s+(don|hoa don|bill)\b") or _has(a, r"khach|mat hang|san pham|\bsp\b|tra hang|trung binh|giao|dot"):
        return None
    if _has(a, r"huy"):
        r = sem.order_count(a.time.primary, a.as_of, states=("cancelled",))
        return FastAnswer("cancelled_count", r.value, "number", False, r.evidence)
    if _has(a, REV.replace("|bao nhieu tien", "")):
        return None
    r = sem.order_count(a.time.primary, a.as_of, shift=a.time.shift)
    return FastAnswer("order_count", r.value, "number", False, r.evidence)


def intent_cancel_rate(a, card, sem, ents) -> Optional[FastAnswer]:
    if a.answer_type != "percent" or not a.time.primary or a.time.compare or ents:
        return None
    if not _has(a, r"huy") or _has(a, r"khach|mat hang|san pham|nhom|ly do"):
        return None
    r = sem.cancel_rate(a.time.primary, a.as_of)
    return FastAnswer("cancel_rate", r.value, "percent", False, r.evidence)


def intent_aov(a, card, sem, ents) -> Optional[FastAnswer]:
    if a.answer_type != "number" or not a.time.primary or ents or a.time.compare:
        return None
    if not (_has(a, r"trung binh|binh quan") and _has(a, r"\bdon\b") and _has(a, REV + r"|gia tri")) or _has(a, r"khach|ngay thu|moi ngay|nhap"):
        return None
    r = sem.avg_order_value(a.time.primary, a.as_of)
    return FastAnswer("avg_order_value", r.value, "number", True, r.evidence)


def intent_growth(a, card, sem, ents) -> Optional[FastAnswer]:
    if a.answer_type != "percent" or not a.time.primary or not a.time.compare or ents:
        return None
    if not _has(a, REV) or _has(a, r"nhom|khach|ca sang|ca chieu|so (ly|lon|chai|hop)|gia|huy"):
        return None
    r = sem.period_compare("revenue", a.time.primary, a.time.compare, a.as_of)
    return FastAnswer("revenue_growth", r.value, "percent", False, r.evidence)


def intent_best_period(a, card, sem, ents) -> Optional[FastAnswer]:
    if a.answer_type != "date" or not a.time.primary or ents:
        return None
    if not _has(a, REV + r"|ban dat|dat nhat"):
        return None
    if _has(a, r"don nhat|nhieu don|it nhat|thap nhat|kem nhat|e nhat"):
        return None
    by = "month" if a.date_granularity == "month" else "day"
    r = sem.rank(by, "revenue", a.time.primary, a.as_of, k=1)
    return FastAnswer(f"best_{by}", r.value[0], "date", False, r.evidence)


PRODUCT_WORDS = r"mat hang|san pham|\bsp\b|\bhang\b|\bmon\b|thuoc|do uong"
# "ai mua hàng nhiều tiền nhất" asks for a customer ("ai" = who), although it contains "hàng"
CUSTOMER_WORDS = r"khach|\bai\b|nguoi mua"
# Comparisons of two customer segments ("khách đường X và khách đường Y, bên nào") are not rankings
SEGMENT_WORDS = r"\bduong\b|ben nao|nhom nao|\bva khach\b|so voi"


def intent_top_products(a, card, sem, ents) -> Optional[FastAnswer]:
    if a.answer_type not in ("list", "string") or not a.time.primary or ents or a.time.compare:
        return None
    if not _has(a, PRODUCT_WORDS) or _has(a, CUSTOMER_WORDS + r"|nhom|ton|nhap|het han|lai|loi|e\b|cham|it nhat|khong ban"):
        return None
    if _has(a, r"ban chay|ban nhieu|ban duoc nhieu|ban dc nhieu|hot|chay nhat"):
        metric = "revenue" if _has(a, r"doanh thu|dthu|nhieu tien|tien") else "quantity"
    elif _has(a, r"doanh thu|dthu|nhieu tien"):
        metric = "revenue"
    else:
        return None
    k = a.top_k or 1
    if a.answer_type == "string" and k != 1:
        return None
    r = sem.rank("product", metric, a.time.primary, a.as_of, k=k)
    val = r.value[0] if a.answer_type == "string" else r.value
    return FastAnswer(f"top_products_{metric}", val, a.answer_type, False, r.evidence)


def intent_top_customers(a, card, sem, ents) -> Optional[FastAnswer]:
    if a.answer_type not in ("list", "string") or not a.time.primary or ents or a.time.compare:
        return None
    if not _has(a, CUSTOMER_WORDS) or _has(a, r"nhom|\bno\b|cong no|huy|doi y|it nhat|trung thanh|sop|" + SEGMENT_WORDS):
        return None
    if _has(a, r"nhieu don|nhieu lan|so don|lan nhat|don nhat"):
        metric = "order_count"
    elif _has(a, r"nhieu tien|chi tieu|chi nhieu|tong tien|doanh thu|dthu|mua nhieu nhat|mua nhieu"):
        metric = "revenue"
    else:
        return None
    k = a.top_k or 1
    if a.answer_type == "string" and k != 1:
        return None
    r = sem.rank("customer", metric, a.time.primary, a.as_of, k=k)
    val = r.value[0] if a.answer_type == "string" else r.value
    return FastAnswer(f"top_customers_{metric}", val, a.answer_type, False, r.evidence)


def _threshold(a: QuestionAnalysis) -> Optional[Dict[str, Any]]:
    return a.thresholds[0] if len(a.thresholds) == 1 else None


def intent_count_with_threshold(a, card, sem, ents) -> Optional[FastAnswer]:
    th = _threshold(a)
    if a.answer_type not in ("number", "percent") or not a.time.primary or not th or ents or a.time.compare:
        return None
    op = th["op"]
    if _has(a, r"khach") and _has(a, r"\bdon\b"):
        r = sem.count_entities("customer", "order_count", op, th["value"], a.time.primary, a.as_of)
        if a.answer_type == "percent":
            total = sem.distinct_count("customer", a.time.primary, a.as_of).value
            if not total:
                raise SemanticUncertain("không có khách trong kỳ")
            return FastAnswer("customers_min_orders_share", r.value * 100.0 / total, "percent", False, r.evidence)
        return FastAnswer("customers_min_orders", r.value, "number", False, r.evidence)
    if a.answer_type != "number":
        return None
    if _has(a, r"\bdon\b") and _has(a, r"doanh thu|dthu|gia tri|tien") and th.get("unit") in ("d", "k", "tr", None) and th["value"] >= 1000:
        r = sem.count_entities("order", "revenue", op, th["value"], a.time.primary, a.as_of)
        return FastAnswer("orders_min_revenue", r.value, "number", False, r.evidence)
    if _has(a, r"\bngay\b") and _has(a, r"doanh thu|dthu|tien") and th["value"] >= 1000:
        r = sem.count_entities("day", "revenue", op, th["value"], a.time.primary, a.as_of)
        return FastAnswer("days_min_revenue", r.value, "number", False, r.evidence)
    return None


def intent_entity_revenue(a, card, sem, ents) -> Optional[FastAnswer]:
    """Revenue (or share of revenue) of one product / product group / customer group."""
    if a.answer_type not in ("number", "percent") or not a.time.primary or a.time.compare or a.top_k or a.thresholds:
        return None
    usable = [e for e in ents if e.kind in ("product", "product_group", "customer_group")]
    if len(usable) != 1 or len(ents) != 1:
        return None
    e = usable[0]
    if not _has(a, REV + r"|thu duoc"):
        return None
    if _has(a, r"trung binh|binh quan|so (ly|lon|chai|hop)|ban (duoc|dc) (may|bao nhieu|bn)|nhap|gia (ban|nhap)|don gia"):
        return None
    if e.kind == "product_group" and not _has(a, r"nhom|danh muc|loai"):
        return None
    if e.kind == "customer_group" and not _has(a, r"khach|nhom"):
        return None
    if a.answer_type == "percent":
        if not _has(a, r"chiem|ti trong|dong gop|phan tram|%"):
            return None
        r = sem.share_of_total({e.column: e.value}, a.time.primary, a.as_of)
        return FastAnswer(f"{e.kind}_revenue_share", r.value, "percent", False, r.evidence)
    r = sem.revenue(a.time.primary, a.as_of, filters={e.column: e.value})
    return FastAnswer(f"{e.kind}_revenue", r.value, "number", True, r.evidence)


def intent_monthly_series(a, card, sem, ents) -> Optional[FastAnswer]:
    """Revenue of several listed months: list of numbers, or boolean 'tăng liên tục'."""
    if len(a.time.periods) < 2 or ents or not _has(a, REV):
        return None
    vals = [sem.revenue(p, a.as_of).value for p in a.time.periods]
    if a.answer_type == "list" and not _has(a, r"tang|giam"):
        return FastAnswer("revenue_series", vals, "list", True, {"periods": [str(p) for p in a.time.periods]})
    if a.answer_type == "boolean" and _has(a, r"tang lien tuc|luon cao hon|tang deu"):
        return FastAnswer("revenue_increasing", all(x < y for x, y in zip(vals, vals[1:])), "boolean", False, {"values": vals})
    if a.answer_type == "boolean" and _has(a, r"giam lien tuc|luon thap hon|giam deu"):
        return FastAnswer("revenue_decreasing", all(x > y for x, y in zip(vals, vals[1:])), "boolean", False, {"values": vals})
    return None


def intent_order_count_with_state(a, card, sem, ents) -> Optional[FastAnswer]:
    """Đơn thành công / hoàn thành trong kỳ."""
    if a.answer_type != "number" or not a.time.primary or a.time.compare or ents or a.thresholds:
        return None
    if not _has(a, r"\b(so|bao nhieu|may|dem|bn|bnhiu)\s+(don|hoa don|bill)\b"):
        return None
    if _has(a, r"\bhuy\b|tra hang"):
        return None
    if _has(a, r"thanh cong|hoan thanh|da giao"):
        r = sem.order_count(a.time.primary, a.as_of, states=("completed",), shift=a.time.shift)
        return FastAnswer("completed_count", r.value, "number", False, r.evidence)
    return None


def intent_distinct_customer_count(a, card, sem, ents) -> Optional[FastAnswer]:
    """Số khách hàng mua trong kỳ."""
    if a.answer_type != "number" or not a.time.primary or a.time.compare or ents or a.thresholds:
        return None
    if not _has(a, r"\b(so|bao nhieu|may|dem|bn|bnhiu)\s+khach\b"):
        return None
    if _has(a, r"chi tieu|mua nhieu|don gia|\bnhom\b|\bno\b|cong no|tra gop|gop|qua han|quen|sop|moi don|trung binh"):
        return None
    r = sem.distinct_count("customer", a.time.primary, a.as_of)
    return FastAnswer("distinct_customers", r.value, "number", False, r.evidence)


INTENTS: List[Callable] = [
    intent_monthly_series, intent_cancel_rate, intent_growth, intent_aov, intent_count_with_threshold,
    intent_best_period, intent_top_customers, intent_top_products, intent_entity_revenue,
    intent_order_count_with_state, intent_distinct_customer_count,
    intent_order_count, intent_total_revenue,
]


def try_fast_path(a: QuestionAnalysis, card: BusinessCard, sem: Optional[SemanticLayer]) -> Tuple[Optional[FastAnswer], Dict[str, Any]]:
    """(answer or None, trace). Exceptions from the semantic layer propagate to the caller."""
    trace: Dict[str, Any] = {}
    if sem is None or not sem.ok:
        trace["skipped"] = "semantic layer unavailable"
        return None, trace
    ents, amb = find_entities(a, card, sem)
    trace["entities"] = [(e.kind, e.value) for e in ents]
    if amb:
        trace["ambiguous_entities"] = amb
        return None, trace
    for fn in INTENTS:
        ans = fn(a, card, sem, ents)
        if ans is not None:
            trace["intent"] = ans.intent
            return ans, trace
    return None, trace
