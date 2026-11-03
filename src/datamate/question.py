"""Question understanding for DataMate (plan §4.3).
analyze_question() turns a raw question + as_of_date into a QuestionAnalysis: normalized text, resolved time
(time_resolver), predicted answer type (+ date granularity, top k), numeric thresholds, overrides of rulebook
defaults, unanswerability signals (subjective / external knowledge / forecast) and the business concepts the
question needs (compared with the business capability map by the Answerability Guard).
Every decision is a deterministic rule on the accent-stripped text, so it is explainable in the logs.
"""

import re
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from src.datamate.text_norm import normalize_question
from src.datamate.time_resolver import Period, TimeSpec, resolve_time_spec

__all__ = ["Period", "TimeSpec", "QuestionAnalysis", "analyze_question", "resolve_time_spec", "predict_answer_type",
           "CONCEPT_PATTERNS"]


@dataclass
class QuestionAnalysis:
    raw: str
    text: str
    ascii: str
    as_of: date
    time: TimeSpec
    answer_type: str
    answer_type_confidence: float
    date_granularity: Optional[str]
    top_k: Optional[int]
    thresholds: List[Dict[str, Any]] = field(default_factory=list)
    overrides: Dict[str, bool] = field(default_factory=dict)
    subjective: bool = False
    external_knowledge: bool = False
    forecast: bool = False
    ambiguous: List[str] = field(default_factory=list)
    concepts: List[str] = field(default_factory=list)
    answer_type_rule: str = ""       # which rule decided the type (for logs / error analysis)


# ── Answer type ──────────────────────────────────────────────────────────────
# Entities a question can ask to name (singular -> string, k of them -> list)
_ENTITY = (r"(?:best seller|mat hang|mon|hang|san pham|sp|khach hang|khach|nhan vien|tho|nguoi|nha cung cap|ncc|thuoc|nhom(?: hang)?"
           r"|lo(?: hang)?|chi nhanh|ben|loai|hoat chat|hang hoa|do uong|nuoc|banh|size|hang xe|dich vu|viec|phieu)")
_COUNT_UNITS = r"(?:don|ngay|thang|tuan|nam|lan|ly|lon|chai|hop|goi|vien|vi|cai|kg|m3|bao|cay|thung|loc|phan|dia|bich|cuon|tuyp|binh)"

# (rule name, type, pattern, confidence). Checked in order; first match wins.
_TYPE_RULES: List[Tuple[str, str, str, float]] = [
    ("table_explicit", "table", r"\b(?:cho|lap|ke|xuat|lam)\s+(?:mot\s+|1\s+)?bang\b|dang bang|\bbang\s+(?:\d+|gom|liet ke)", 0.95),
    ("table_with_value", "table",
     r"(?:kem|va|cung)\s+(?:theo\s+)?(?:so|tong|muc)\s+(?:tien\s+)?(?:no|tien|luong|doanh thu|ban)[^?.]*?(?:cua\s+)?(?:tung|moi)?"
     r"|(?:moi|tung)\s+(?:nguoi|khach|mat hang|mon|ben)\s+(?:con\s+)?(?:no|mua|ban|chi)[^?.]*?bao nhieu"
     r"|va\s+so\s+no\s+cua\s+tung", 0.9),
    ("bool_trend", "boolean", r"(?:tang|giam)\s+(?:lien tuc|deu|lien tiep)|luon\s+(?:cao|thap|nhieu|it|tang|giam)\s+hon", 0.95),
    ("bool_yesno", "boolean",
     r"\bco\s+(?:phai|dung|dat|vuot|tang|giam|cao hon|thap hon|nhieu hon|it hon|ban|con|bi|het|du)\b[^?]*\b(?:khong|chua)\s*(?:\?|$|a\b|ha\b|nhi\b|vay\b|ta\b)"
     r"|(?:phai|dung)\s+khong\s*(?:\?|$)|\bco\s+\w+(?:\s+\w+){0,6}\s+khong\s*(?:\?|$|vay|ha|nhi|ta|a\b)", 0.85),
    ("date_month", "date", r"\b(?:thang|ky)\s+nao\b", 0.95),
    ("date_day", "date", r"\b(?:ngay|hom|bua)\s+(?:nao|may)\b|\bngay\s+bao nhieu\b|\bvao\s+ngay\s+nao|han (?:dung|su dung) (?:som|gan) nhat la (?:ngay|khi) nao", 0.93),
    ("list_several_numbers", "list",
     r"lan luot|\bmoi ben\b|theo (?:dung )?thu tu|ghi (?:hai|2|ba|3) so|\bmoi thang\b|chia theo"
     r"|\bnguyen\b[^?]*\ble\b|\bchan\b[^?]*\ble\b"
     r"|bao nhieu[^?,]*\b(?:va|,)\s+(?:da\s+)?(?:\w+\s+){0,5}(?:bao nhieu|duoc bao nhieu|may)"
     r"|(?:may|bao nhieu)\s+\w+\s+(?:chan|nguyen)", 0.85),
    ("list_expiry", "list", r"\b(?:hang|mat hang|san pham|sp|thuoc|lo(?: hang)?)\s+nao\s+(?:sap\s+)?(?:het han|con han|co han)", 0.85),
    ("list_topk", "list", r"\btop\s*\d+|(?<![\d.,/])(?:[2-9]|1\d|20)\s+" + _ENTITY + r"\b(?!\s+(?:tro len|tro xuong))", 0.9),
    ("list_enumerate", "list", r"liet ke|\bnhung\s+" + _ENTITY + r"\s+nao|\bcac\s+" + _ENTITY + r"\s+nao|\bnhung\s+(?:ai|gi)\b|danh sach", 0.85),
    ("percent", "percent", r"(?<![\d,.])\s*%|phan tram|\bti le\b|\bty le\b|ti trong|ty trong|\bchiem\b|tang truong", 0.95),
    # several numbers asked together: 'A và B mỗi loại bao nhiêu', 'A là bao nhiêu, B là bao nhiêu', 'doanh thu A và doanh thu B'
    ("list_two_values", "list",
     r"(?<!tinh )(?<!cong )moi loai(?! do| mon| hang)|tra loi (?:hai|2|ba|3) so|nhom tuoi no|\b0\s*[–-]\s*7\b"
     r"|(?:doanh thu|tien)[^?]*\b(?:va|voi)\s+(?:doanh thu|tien)\s+(?:cac|cua)"
     r"|\b(?:bao nhieu|nhieu|may)\b(?! nhat| hon)[^?]*(?:\bva\b|,)[^?]*\b(?:bao nhieu|nhieu|may)\b(?! nhat| hon)", 0.75),
    # 'có mấy mặt hàng ế, không ai mua món nào' is a count, not a name
    ("count_negated_which", "number", r"(?:co|tong cong|dem)\s+(?:tong cong\s+)?(?:bao nhieu|may)\s+" + _ENTITY + r"\b", 0.8),
    ("bool_co_phai", "boolean", r"\bco phai\b(?![^?]*\b(?:ai|nao)\b)", 0.8),
    ("string_who", "string",
     r"\b(?:la ai|ai la|la nguoi nao)\b|(?<!khong )(?<!chang )\bai\b|" + _ENTITY + r"\s+nao\b|\bla\s+(?:gi|j|chi)\b"
     r"|\bthu\s+may\b|ten\s+(?:gi|la gi)|ben nao", 0.85),
]

_NUMBER_HINT = re.compile(r"bao nhieu|\bmay\b|nhieu\b|tong|dem|so\b|trung binh|binh quan|chenh|hon hay kem|gia")


def predict_answer_type(a: str) -> Tuple[str, float, str]:
    """(answer_type, confidence, rule) from the accent-stripped normalized question."""
    for name, typ, pat, conf in _TYPE_RULES:
        m = re.search(pat, a)
        if not m:
            continue
        if typ == "list" and name == "list_topk":
            # 'top 1' / '1 khách' are a single name; 'N đơn trở lên' are thresholds, not k
            k = _top_k(a)
            if k is None or k == 1:
                continue
        if typ == "percent":
            # '% ... bên nào' style or '%' in an amount ('chiếm bao nhiêu')? percent wins
            return typ, conf, name
        if typ == "string" and re.search(r"bao nhieu|\bmay\b", a) and not re.search(r"\bai\b|" + _ENTITY + r"\s+nao", a[:m.start()] + a[m.start():]):
            continue
        if typ == "string" and re.search(r"(?:bao nhieu|\bmay)\s+" + _COUNT_UNITS + r"?\b", a) and re.search(_ENTITY + r"\s+nao", a) \
                and re.search(r"(?:co|duoc)\s+(?:bao nhieu|may)", a):
            # 'có bao nhiêu khách ... mặt hàng nào' : still a count
            return "number", 0.6, "number_count_with_which"
        return typ, conf, name
    conf = 0.9 if _NUMBER_HINT.search(a) else 0.55
    return "number", conf, "number_default"


_WORD_NUM = {"mot": 1, "hai": 2, "ba": 3, "bon": 4, "nam": 5, "sau": 6, "bay": 7, "tam": 8, "chin": 9, "muoi": 10}


def _top_k(a: str) -> Optional[int]:
    m = re.search(r"\btop\s*(\d+)", a)
    if m:
        return int(m.group(1))
    for m in re.finditer(r"(?<![\d.,/])(\d{1,2})\s+" + _ENTITY + r"\b", a):
        tail = a[m.end():m.end() + 14]
        if re.match(r"\s+(?:tro len|tro xuong|tro lai)|\s*x\b", tail):
            continue
        # pack sizes inside product names ('lốc 10 lọ x 10ml', 'hộp 3 vỉ')
        if re.search(r"(?:hop|loc|thung|vi|chai|goi|bich|tui|bao|loai)\s+$", a[max(0, m.start() - 6):m.start()]):
            continue
        # 'khách mua từ 5 đơn' is a threshold; '3 khách nào', '5 mặt hàng bán chạy' is k
        if re.search(r"(?:tu|it nhat|tren|duoi|qua|hon|>=|≥)\s*$", a[max(0, m.start() - 9):m.start()]):
            continue
        return int(m.group(1))
    m = re.search(r"\b(hai|ba|bon|nam)\s+(?:mat hang|mon|san pham|khach|nhan vien)\b", a)
    if m:
        return _WORD_NUM[m.group(1)]
    return None


# ── Thresholds ───────────────────────────────────────────────────────────────
_NUM = r"(\d{1,3}(?:[.,]\d{3})+|\d+(?:[.,]\d+)?)\s*(trieu|tr|k|nghin|ngan|d|dong|vnd)?"


def _parse_amount(num: str, unit: Optional[str]) -> float:
    s = num
    if re.fullmatch(r"\d{1,3}(?:[.,]\d{3})+", s):
        s = re.sub(r"[.,]", "", s)
    else:
        s = s.replace(",", ".")
    v = float(s)
    if unit in ("trieu", "tr"):
        v *= 1_000_000
    elif unit in ("k", "nghin", "ngan"):
        v *= 1000
    return v


_THRESHOLD_PATTERNS: List[Tuple[str, str]] = [
    (r"(?:tu|dat|>=|≥)\s*" + _NUM + r"\s*(?:\w+\s+){0,4}?tro len", ">="),   # "đạt 130.000 đồng trở lên" too
    (r"(?:it nhat|toi thieu|>=|≥|khong duoi)\s*" + _NUM, ">="),
    (r"(?:tu)\s*" + _NUM + r"\s*(?:\w+\s+){0,4}?tro xuong", "<="),
    (r"(?:toi da|khong qua|<=|≤)\s*" + _NUM, "<="),
    (r"(?:tren|qua|hon|lon hon|nhieu hon|cao hon|vuot|>)\s*" + _NUM, ">"),
    (r"(?:duoi|it hon|nho hon|thap hon|<)\s*" + _NUM, "<"),
]


def extract_thresholds(a: str) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    used: List[Tuple[int, int]] = []
    for pat, op in _THRESHOLD_PATTERNS:
        for m in re.finditer(pat, a):
            s, e = m.span(1)
            if any(s < ue and us < e for us, ue in used):
                continue
            # Dates ('từ 20/10/2025') and times are not thresholds
            if re.match(r"[/.-]\d", a[e:e + 2]) or re.search(r"\d[/]$", a[:s]):
                continue
            used.append((s, e))
            unit_after = a[m.end(2) if m.group(2) else e:][:12]
            out.append({"value": _parse_amount(m.group(1), m.group(2)), "op": op, "text": m.group(0).strip(),
                        "unit": (m.group(2) or (re.match(r"\s*(\w+)", unit_after).group(1) if re.match(r"\s*(\w+)", unit_after) else ""))})
    out.sort(key=lambda t: a.find(t["text"]))
    for t in out:
        if float(t["value"]).is_integer():
            t["value"] = int(t["value"])
    return out


# ── Overrides of rulebook defaults ───────────────────────────────────────────
_OVERRIDES: Dict[str, str] = {
    "include_cancelled": r"(?:ke ca|tinh ca|bao gom|gom ca|tinh luon|tinh)\s+(?:cac\s+|nhung\s+)?don\s+(?:bi\s+|da\s+)?huy",
    "include_returned": r"(?:ke ca|tinh ca|bao gom|gom ca|tinh luon)\s+(?:cac\s+|nhung\s+)?(?:don\s+)?(?:bi\s+)?tra(?:\s+hang)?",
    "by_revenue": r"(?:theo|xet theo|tinh theo|tinh bang)\s+(?:doanh thu|so tien|tien|gia tri)|nhieu tien nhat|chi nhieu nhat|chi tieu nhieu nhat",
    "by_quantity": r"(?:theo|xet theo|tinh theo)\s+so luong",
    "by_order_count": r"(?:theo|xet theo|tinh theo)\s+so (?:don|lan)|nhieu (?:don|lan)(?:\s+hang)?\s+nhat",
    "before_discount": r"(?:chua|khong)\s+tru\s+(?:giam gia|chiet khau|khuyen mai)|truoc\s+(?:giam gia|chiet khau|khi giam)",
    "after_order_discount": r"(?:da\s+)?tru\s+(?:ca\s+)?(?:voucher|giam gia (?:tren\s+)?(?:ca\s+|toan\s+)?don|khuyen mai (?:tren\s+)?(?:ca\s+|toan\s+)?don)",
    "include_vat": r"(?:gom|tinh ca|cong|ke ca|bao gom)\s+(?:ca\s+)?(?:vat|thue)",
    "include_shipping": r"(?:gom|tinh ca|cong|ke ca|bao gom)\s+(?:ca\s+)?(?:phi|tien)\s+(?:giao|ship|van chuyen)",
    "by_payment_received": r"tien (?:khach )?da (?:tra|thanh toan)|so tien thuc thu|thuc thu",
}

# ── Unanswerability signals ──────────────────────────────────────────────────
_SUBJECTIVE = (r"\bkhen\b|tan tam|le phep|quy men|lanh nghe|tay nghe|hai long|chu dao|than thien|nhiet tinh|vui ve"
               r"|tu te|de chiu|lich su voi|khach (?:hang )?kho tinh|phan nan|\bche\b|danh gia (?:cao|tot|kem)|yeu men|dang tin"
               r"|co tam|tot bung|dep trai|xinh|de thuong|cham soc khach (?:tot|nhat)|thai do")
_EXTERNAL = (r"lieu dung|nen uong|uong (?:moi lan|may vien|bao nhieu vien)|moi lan (?:uong|dung)|cong dung|tac dung phu|chong chi dinh"
             r"|cach dung|cach bao quan|co tot khong|co hai|gia thi truong|\bcho khac\b|doi thu|thoi tiet|nguoi lon nen|tre em nen"
             r"|chua benh|tri benh|benh gi")
_FORECAST = r"du bao|du doan|\bse\s+(?:ban|dat|co|la|tang|giam)\b|ngay mai|(?<!dau )\b(?:tuan|thang|nam) toi\b(?!\s+(?:het|nay|hom|bua))|\bnam sau\b|trong tuong lai|sap toi"
_FORECAST_NEXT_MONTH = r"\b(?:thang|tuan|quy)\s+sau\b"

# ── Concepts (capability vocabulary) ─────────────────────────────────────────
# Trigger patterns on the accent-stripped question. Precision matters more than recall: the Guard only treats a
# question as unanswerable when a triggered concept is missing from the business data.
CONCEPT_PATTERNS: Dict[str, str] = {
    "revenue": r"doanh thu|\bban (?:duoc|dc|ra)?[^?]{0,25}\btien\b|tien (?:ban|hang)|thu ve|revenue|ban dat|chi (?:tieu|nhieu)"
               r"|mua [^?]{0,25}nhieu tien|tong tien (?:mua|ban)|gia tri (?:trung binh|binh quan)? ?(?:mot|moi|1)? ?don|mang ve",
    "order_count": r"\b(?:so|bao nhieu|may|dem|nhieu)\s+(?:don|bill|hoa don|luot)|\bdon\s+(?:hang\s+)?nhat|nhieu don|\bbill\b",
    "quantity": "",  # set below
    "cancel": r"\bhuy\b",
    "return": r"tra hang|don tra|bi tra|hoan hang|tra lai hang",
    "customer": r"\bkhach\b",
    "customer_group": r"nhom\s+(?:khach\s+)?(?:than thiet|si|le|vip|thuong|thanh vien|moi)|khach\s+(?:si|vip|than thiet)\b|nhom khach",
    "customer_gender": r"khach\s+(?:hang\s+)?(?:nu|nam)\b(?!\s*\d)|gioi tinh|\bphai (?:nu|nam)\b",
    # 'duong'/'phuong' are also the names Dương / Phương: require a locative word before them
    # \b before the locative ("ngày mô quán" is not "ở quận"); no bare "quân 1" ("bình quân 1 đơn")
    "customer_address": r"\b(?:o|nha o|tai|tren|khu|khach|thuoc)\s+(?:duong|phuong|quan|huyen|xa)\s+(?!(?:ban|mua|nao|minh|ta)\b)\w|dia chi|khu vuc|tinh thanh|que quan",
    "customer_age": r"do tuoi|khach\s+(?:tren|duoi|tu)\s+\d+\s+tuoi|tuoi khach",
    "payment_method": r"tien mat|chuyen khoan|\bcod\b|vi dien tu|momo|zalopay|vnpay|quet the|\bthe\s+(?:tin dung|ngan hang)|hinh thuc thanh toan|phuong thuc thanh toan",
    # "hôm qua cửa hàng ..." must not read as the channel "qua cửa hàng": 'cua hang' is not a channel word
    "sales_channel": r"(?:qua|tai|tren)\s+(?:dien thoai|quay|zalo|facebook|fb|shopee|lazada|tiktok|san\b|online|app|website|web|grab|shopeefood)|kenh ban|\bban online|\bsan tmdt",
    "staff": r"nhan vien|(?<!ong )\btho\b|thu ngan|nguoi ban|\bnv\b|nguoi lam",
    "product_group": r"\bnhom\s+(?!khach)\w+|danh muc|loai hang|nganh hang",
    "unit_price": r"(?:don )?gia ban|gia (?:mot|1|moi)\s+\w+|voi (?:don )?gia|gia bao nhieu",
    "purchase": r"\bnhap\b|phieu nhap|tien nhap|gia nhap",
    "supplier": r"nha cung cap|\bncc\b",
    # not bare 'ton' (the product "Tôn lạnh") nor 'còn lại' ("các mặt hàng còn lại bán được ...")
    "stock": r"ton kho|trong kho|\bkho\b|het hang|hang ton\b|\bton (?:cuoi|dau|thuc|kho|lai)\b|\bchan\b[^?]*\ble\b|nguyen\b[^?]*\ble\b"
             r"|con (?:ton|bao nhieu|may)\b(?!\s+(?:khach|don|nguoi|tien|ngay|ky)\b)(?![^?]*\bban\b)",  # not "còn mấy khách nợ"
    "cost": r"gia von|tien von|\bvon\b",
    # 'còn lại', 'đem lại', 'thu lại', 'mang lại', 'quay lại', 'trả lại' are not 'lãi'
    "gross_profit": r"lai gop|loi nhuan|(?<!con )(?<!dem )(?<!thu )(?<!mang )(?<!quay )(?<!tra )(?<!gui )(?<!cong )\blai\b|\bloi\b(?! (?:san pham|hang))|\blo\s+(?:von|tien)",
    "net_profit": r"lai rong|loi nhuan rong|lai thuc",
    "expense": r"chi phi|tien dien|tien nuoc|mat bang|luong nhan vien|sua chua|so chi|\bchi ra\b|khoan chi",
    "debt": r"\bno\b|cong no|mua chiu|\bthieu\b|tuoi no|ghi no|tra no|thu no",
    # "hạn đúng hôm nay" (debt due date) is also "han dung" without accents
    "expiry": r"han dung\b(?!\s+(?:hom|bua|ngay|hen|ky))|het han|\bhsd\b|han su dung|\bdate\b|can date",
    "lot": r"\blo\s+(?:hang|nao|con|gan|som|nhap)|\bcac lo\b|\bso lo\b|theo lo\b",  # word boundary: "tính theo lon" is not "theo lô"
    "delivery": r"giao hang|dot giao|cho giao|phieu giao|van chuyen|dot hang|chua giao|da giao",
    "warranty": r"bao hanh",
    "cancel_reason": r"huy\s+(?:vi|do|ly do)\b|ly do huy",  # \b: "huỷ đơn" is not "huỷ do"
    # "đơn trả (bằng) ví điện tử" is a payment, not "trả hàng vì ..."
    "return_reason": r"tra\s+(?:hang\s+)?(?:vi|do)\b(?!\s+dien tu)|ly do tra",
    "price_change": r"tang gia|len gia|doi gia|dieu chinh gia|gia moi",
    "shift": r"\bca\s+(?:sang|chieu|toi|dem)|theo ca|bao cao ca",
    "order_time": r"gio cao diem|khung gio|gio nao|\bmay gio\b",
    "member_registration": r"dang ky|thanh vien moi|khach moi",
    "voucher": r"voucher|ma giam|khuyen mai",
    "points": r"\bdiem\b|tich diem",
    "shipping_fee": r"phi ship|phi giao|tien ship|phi van chuyen",
    "discount": r"giam gia|chiet khau",
    "active_ingredient": r"hoat chat",
    "prescription": r"ke don",
}
CONCEPT_PATTERNS["quantity"] = (r"ban (?:duoc|ra|dc)?\s*(?:tong cong\s+)?(?:bao nhieu|may)\s+(?!tien|don|%|phan tram)\w+"
                                r"|\bso\s+(?:ly|lon|chai|hop|goi|vien|vi|kg|bao|cay|thung|bich|tuyp|binh|dia|phan|cuc|cai|m3)\b"
                                r"|ban chay|ban nhieu nhat|hot nhat|so luong|ban (?:duoc|dc) nhieu nhat")


def _concepts(a: str) -> List[str]:
    found = [c for c, pat in CONCEPT_PATTERNS.items() if re.search(pat, a)]
    # Ranking customers 'nhiều tiền nhất' etc. needs revenue
    if "customer" in found and re.search(r"nhieu tien|chi nhieu|chi tieu", a) and "revenue" not in found:
        found.append("revenue")
    if "net_profit" in found and "gross_profit" in found:
        found.remove("gross_profit")
    return found


def analyze_question(question: str, as_of_date: str, calendar_df: Optional[pd.DataFrame] = None) -> QuestionAnalysis:
    text, a = normalize_question(question)
    a = re.sub(r"(\d)\s*/\s*(\d)", r"\1/\2", a)
    try:
        as_of = datetime.strptime(str(as_of_date)[:10], "%Y-%m-%d").date()
    except ValueError:
        as_of = date.today()
    spec = resolve_time_spec(question, as_of, calendar_df)

    answer_type, conf, rule = predict_answer_type(a)
    granularity = None
    if answer_type == "date":
        granularity = "month" if rule == "date_month" else "day"
    top_k = _top_k(a) if answer_type in ("list", "table", "string") else None
    if answer_type == "string" and top_k and top_k > 1:
        answer_type, conf, rule = "list", 0.75, "string_with_k"

    ambiguous: List[str] = []
    if not spec.confident:
        ambiguous.append("time:" + ",".join(m for m in spec.matched if m.startswith("?")))
    if conf < 0.7:
        ambiguous.append(f"answer_type:{answer_type}")

    forecast = bool(re.search(_FORECAST, a)) or (
        # "tháng sau (đều/luôn) cao hơn tháng trước" compares past months, it is not a forecast
        bool(re.search(_FORECAST_NEXT_MONTH, a)) and not re.search(r"(?:thang|tuan|quy) sau\s+(?:co\s+)?(?:deu\s+|luon\s+)*(?:cao|thap|nhieu|it|hon|tang|giam)", a))
    return QuestionAnalysis(
        raw=question, text=text, ascii=a, as_of=as_of, time=spec,
        answer_type=answer_type, answer_type_confidence=conf, date_granularity=granularity, top_k=top_k,
        thresholds=extract_thresholds(a),
        overrides={k: True for k, pat in _OVERRIDES.items() if re.search(pat, a)},
        subjective=bool(re.search(_SUBJECTIVE, a)),
        external_knowledge=bool(re.search(_EXTERNAL, a)),
        forecast=forecast,
        ambiguous=ambiguous,
        concepts=_concepts(a),
        answer_type_rule=rule,
    )
