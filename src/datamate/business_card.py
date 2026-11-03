"""Business Card (plan §4.1): a deterministic profile of one business, built once and cached.

- semantic role of each column (order id, order date, product, quantity, line amount, status, ...)
  scored from three independent signals: the README data dictionary, the column name, and the values;
- the sales model: which table holds order lines, which holds order headers, how they join;
- the date coverage of the transactions (needed by the Guard: a period fully outside it is unanswerable);
- the capability map: which business concepts the data can answer at all;
- the entity index: products / customers / staff / suppliers / groups with every spelling and code;
- data quality flags.
Everything is computed from the data; nothing depends on the business id.
"""

import json
import os
import re
import unicodedata
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from unidecode import unidecode

from src.datamate.db import BusinessDB, ColumnInfo, TableInfo
from src.datamate.loader import BusinessData, table_stem


def ascii_key(s: Any) -> str:
    """'Thành tiền (đ)' -> 'thanh tien d' (accent-free, lowercase, single spaces).
    Used for data values too, so it must stay the twin of semantic.norm_sql (no CamelCase split)."""
    s = unidecode(unicodedata.normalize("NFC", str(s))).lower()
    return re.sub(r"[^a-z0-9]+", " ", s).strip()


def header_key(s: Any) -> str:
    """Column/table header -> words: 'NgayLap' -> 'ngay lap', 'SLQuyDoi' -> 'sl quy doi', 'TenKH' -> 'ten kh'."""
    s = unidecode(unicodedata.normalize("NFC", str(s)))
    s = re.sub(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])", " ", s)
    return ascii_key(s)


# ── Role vocabulary ───────────────────────────────────────────────────────────
# name: regexes on ascii_key(column name); desc: regexes on ascii_key(README description).
# Patterns are anchored on word boundaries; earlier roles do not block later ones (a column can
# carry several candidate roles; the sales model picks a consistent assignment).
ROLE_PATTERNS: Dict[str, Dict[str, List[str]]] = {
    "order_id": {
        "name": [r"^(ma|so) ?(hoa don|hd|don|don hang|bill|gd|giao dich|chung tu|phieu|ct)$", r"^ma don\b", r"^(order|invoice|bill)( id| no| code)?$",
                 r"^so hd$", r"^ma hoa don\b", r"^ma don hang\b", r"^madon(hang)?$", r"^mahoadon$", r"^hoa don$"],
        "desc": [r"^ma don\b", r"^ma hoa don\b", r"^so hoa don\b"],
    },
    "order_date": {
        "name": [r"^(ngay|thoi gian|date|ngaythang|ngay thang|ngay gio|thoi diem)\b(?!.*(sinh|giao|tra|thu|het|han|nhap|dang ky|tao khach|cap nhat|ap dung))",
                 r"^ngay ?ban$", r"^ngay (dat|ban|mua|lap|tao|giao dich|hoa don)\b", r"^thoi gian (tao|lap|ban|dat)\b", r"^ngayban$"],
        "desc": [r"^ngay ban\b", r"^thoi diem dat don\b", r"^ngay dat\b", r"^ngay gio dat\b", r"^ngay lap\b"],
    },
    # Shifts and peak hours are "theo giờ tạo đơn" (rulebook §2/§3): the creation time-of-day column
    "order_time": {
        "name": [r"^gio( dat| ban| tao)?$", r"^time$", r"^(thoi gian tao|gio tao don|gio tao|thoi diem tao|ngay gio tao|tao luc|thoi gian lap|ngay gio lap)$"],
        "desc": [r"^gio dat don\b", r"^thoi diem dat don\b"],
    },
    "product_name": {
        "name": [r"^(ten )?(hang|hang hoa|hang ban|mat hang|san pham|sp|mon|thuoc|product|item)( ban)?$", r"^ten (hang|sp|san pham|mon|thuoc|hang hoa|mat hang)\b",
                 r"^noi dung$", r"^tenhang$", r"^ten$", r"^chi tiet hang$", r"^ten goi$", r"^ten hang hoa$"],
        "desc": [r"^ten hang\b", r"^ten mon\b", r"^ten$", r"^ten san pham\b"],
    },
    "product_code": {
        "name": [r"^(ma|id) ?(hang|sp|san pham|hh|hang hoa|sku|mat hang|mon|thuoc)( ban)?$", r"^sku$", r"^ma ?sku$", r"^(product|item) ?(id|code)$", r"^mahang$"],
        "desc": [r"^ma hang\b", r"^ma san pham\b", r"^ma mon\b", r"^ma thuoc\b"],
    },
    "quantity": {
        "name": [r"^(so luong|sl|sluong|soluong|qty|quantity|slg|s l)( ban| hang| xuat| san pham)?$", r"^so luong \(?dvt\)?$"],
        "desc": [r"^so luong$", r"^so luong ghi theo don vi\b", r"^so luong ban$"],
    },
    "quantity_base": {
        "name": [r"^(quy ra dv le|sl le|sl quy doi|so luong le|so luong quy doi|sl dv co ban|so luong co ban|sl thuc giao|so luong giao)$",
                 r"quy (ra|doi).*(le|co ban)", r"^so luong \(?dvt co ban\)?$", r"^sl (co ban|quy ra le|quy doi)$"],
        "desc": [r"so luong da doi ra don vi co ban", r"don vi co ban\)?$"],
    },
    "unit": {
        "name": [r"^(dvt|don vi|don vi tinh|dv tinh|unit|donvi)$", r"^don vi \(?tinh\)?$"],
        "desc": [r"^don vi tinh$"],
    },
    "unit_price": {
        "name": [r"^(don gia|gia|gia ban|gia ban le|gia le|gia san pham|gia don vi|gia 1 dv|price|dongia|unit price)( ban)?$"],
        "desc": [r"^don gia$"],
    },
    "line_amount": {
        "name": [r"^(thanh tien|t tien|tien|tien ban|tri gia|amount|trigia|thanhtien|thanh tien d|thanh tien sau giam|tong)$", r"^thanh tien\b(?!.*truoc)"],
        "desc": [r"^thanh tien sau giam\b", r"^thanh tien cua dong\b", r"^thanh tien\b(?!.*nhap)"],
    },
    "line_discount": {
        "name": [r"^(giam gia|chiet khau|ck|discount)( dong| sp| mat hang)?$"],
        "desc": [r"giam gia (tren|cua) (dong|tung dong)"],
    },
    "order_goods_amount": {
        "name": [r"^(tien hang|tong tien hang|tienhang)$"],
        "desc": [r"^tien hang cua don\b", r"^tong tien hang cua don\b"],
    },
    "order_total": {
        "name": [r"^(tong tien|tong bill|tong don|phai thu|tong sau giam gia|khach can tra|tongtien|tong cong|tong thanh toan)$"],
        "desc": [r"^tong tien don\b", r"^so khach phai tra cho ca don\b"],
    },
    "order_discount": {
        "name": [r"^(giam gia don|giam gia hoa don|voucher|khuyen mai|ma khuyen mai|giam gia)$"],
        "desc": [r"giam gia (tren )?ca don", r"^voucher"],
    },
    "shipping_fee": {
        "name": [r"^(phi giao|phi ship|phi van chuyen|phi giao hang|ship)$"],
        "desc": [r"^phi giao\b", r"^phi ship\b"],
    },
    "vat": {"name": [r"^(vat|thue|thue vat|tien thue)$"], "desc": [r"^thue\b", r"^vat\b"]},
    "status": {
        "name": [r"^(trang thai|ket qua|status|tinh trang|xu ly|trang thai don|trang thai xu ly|trang thai don hang|ghi chu tt|trangthai|tinh trang don)$"],
        "desc": [r"^trang thai don\b"],
    },
    "customer_code": {
        "name": [r"^(ma|so) ?(kh|khach|khach hang|khach mua|khach le si)$", r"^customer ?(id|code)$", r"^makh$"],
        "desc": [r"^ma khach\b"],
    },
    "customer_name": {
        "name": [r"^(ten )?(khach|khach hang|kh)( \(?ten\)?)?$", r"^(ten ?kh|ten khach|ten khach hang|doi tac|ben mua|nguoi mua|customer|tenkh)$", r"^khach le si$"],
        "desc": [r"^ten khach\b"],
    },
    "customer_group": {"name": [r"^(nhom khach|nhom khach hang|loai khach|loai khach hang|nhom kh|hang khach|hang thanh vien)$"], "desc": [r"^nhom khach\b"]},
    "customer_gender": {"name": [r"^(gioi tinh|gender|phai)$"], "desc": [r"^gioi tinh\b"]},
    "customer_address": {"name": [r"^(dia chi|addr|address|duong)\b"], "desc": [r"^dia chi\b"]},
    "customer_phone": {"name": [r"^(so dien thoai|dien thoai|sdt|phone|di dong)$"], "desc": [r"^so dien thoai\b"]},
    "member_registration": {"name": [r"^ngay dang ky\b"], "desc": [r"^ngay dang ky\b"]},
    "product_group": {
        "name": [r"^(nhom hang|nhom hang hoa|loai hang|danh muc|loai san pham|nhom sp|nhom|category|nhom mon)\b"],
        "desc": [r"^nhom hang\b"],
    },
    "payment_method": {
        "name": [r"^(hinh thuc thanh toan|hinh thuc tt|httt|pttt|phuong thuc|phuong thuc tt|phuong thuc thanh toan|tra bang|thanh toan bang)$"],
        "desc": [r"^hinh thuc thanh toan\b"],
    },
    "sales_channel": {"name": [r"^(kenh ban|kenh ban hang|ban qua|loai ban|kenh|nguon don|channel)$"], "desc": [r"^kenh ban\b"]},
    "staff": {
        "name": [r"^(nhan vien|nhan vien ban|nguoi ban|thu ngan|nv|nv ban|tho|ten tho|tho sua|ma nv|ma nhan vien|nguoi lap)$"],
        "desc": [r"nhan vien ban\b", r"^tho\b", r"^nguoi ban\b"],
    },
    "paid_amount": {"name": [r"^(da tra|khach tra|khach da tra|da thanh toan|thanh toan)$"], "desc": [r"^da thanh toan\b"]},
    "debt_remaining": {"name": [r"^(con no|con lai|du no|chua tra|no con lai|con thieu)$"], "desc": [r"^con no\b", r"khach con no\b"]},
    "payment_date": {"name": [r"^(ngay thu|thoi gian thu|thoi diem thu|ngay tra)$"], "desc": [r"^thoi diem thu\b"]},
    "due_date": {"name": [r"^(han thanh toan|han tra|han no)$"], "desc": [r"^han thanh toan\b"]},
    "cost_per_unit": {"name": [r"^gia von\b", r"^giavon"], "desc": [r"^gia von\b"]},
    "stock_qty": {"name": [r"^(ton cuoi|ton cuoi ky|ton thuc|ton kho|toncuoi|ton|so luong ton)$"], "desc": [r"^ton kho\b", r"^ton cuoi\b"]},
    "expiry_date": {"name": [r"^(han dung|han su dung|hsd|expiry|ngay het han)$"], "desc": [r"^han (su )?dung\b"]},
    "lot": {"name": [r"^(lo|ma lo|so lo|lo hang)$"], "desc": [r"^so lo\b"]},
    "supplier": {"name": [r"^(nha cung cap|ncc|ma ncc|ma nha cung cap|ten ncc)$"], "desc": [r"nha cung cap\b"]},
    "active_ingredient": {"name": [r"^hoat chat$"], "desc": [r"^hoat chat\b"]},
    "prescription_flag": {"name": [r"^(thuoc ke don|ke don|ban theo don|bantheodon)$"], "desc": [r"thuoc ke don\b"]},
    "delivery_date": {"name": [r"^(ngay giao|ngay giao hang|ngay cho)$"], "desc": [r"^ngay giao\b"]},
    "delivery_batch": {"name": [r"^(dot giao|lan giao|luot giao)$"], "desc": [r"^dot giao\b"]},
    "return_qty": {"name": [r"^(so luong tra|sl tra)\b"], "desc": [r"so luong khach tra lai\b"]},
    "price_effective_date": {"name": [r"^ngay ap dung$"], "desc": [r"^ngay bat dau ap dung gia\b"]},
    "base_unit": {"name": [r"^(dv goc|dv nho|don vi co ban|don vi goc|dv co ban)$"], "desc": [r"^don vi co ban$"]},
    "big_unit": {"name": [r"^(dv lon|dv nhap|don vi quy doi|don vi lon)$"], "desc": [r"^don vi lon\b"]},
    "conversion_factor": {"name": [r"^(he so|he so qd|ti le quy doi|ty le quy doi|he so quy doi|quy doi)$"], "desc": [r"^he so\b"]},
    "note": {"name": [r"^(ghi chu|ghi chu don|dien giai|note)$"], "desc": [r"^ghi chu\b"]},
    "reason": {"name": [r"\bly do\b", r"^(nguyen nhan|reason)\b"], "desc": [r"\bly do\b"]},
    "birth_date": {"name": [r"^(ngay sinh|nam sinh|sinh nhat|birthday|dob)$"], "desc": [r"^ngay sinh\b"]},
    "points": {"name": [r"\bdiem\b(?!.*danh)"], "desc": [r"\bdiem\b"]},
    "warranty": {"name": [r"bao hanh"], "desc": [r"bao hanh"]},
    "shift": {"name": [r"^(ca|ca ban|ca lam|ma ca)$"], "desc": [r"^ca\b"]},
}

# Table name hints (ascii_key of the original table name)
TABLE_KIND_PATTERNS = {
    "purchase": r"nhap",
    "stock": r"ton ?kho|bao cao ton|the kho",
    "customer": r"khach ?hang|^khach|customer",
    "product": r"san ?pham|hang ?hoa|danh ?muc|danh sach hang|thuoc|^mon\b|menu",
    "payment": r"thanh ?toan|phieu ?thu|so ?quy|cong ?no|thu ?tien",
    "conversion": r"quy ?doi|don ?vi ?tinh",
    "expense": r"chi ?phi",
    "price_history": r"lich ?su ?gia|bang ?gia",
    "lot": r"\blo\b|lo ?hang|theo ?lo",
    "delivery": r"giao ?hang|phieu ?giao",
    "debt": r"han ?no|cong ?no|so ?no",
    "staff": r"nhan ?vien|\btho\b",
}

STATUS_KEYWORDS = [  # (canonical, ascii regex) checked in order; \b matters: "van c-huy-en" is not "huỷ"
    ("returned", r"\btra hang\b|\bhoan tra\b|\btra lai\b|\breturn|\brefund|\bdoi tra\b|\bkhach tra\b|\bhoan hang\b|\btra ve\b"),
    # "Không giao được" = cancelled and "Đang xử lý" = in progress: measured on train answers
    # (scratch experiment over biz_107/116/124 and biz_109/120/125: cancelled 130 right / 12 wrong vs
    # returned 122 / 17 vs completed 86 / 53; "Đang xử lý" as cancelled is the only worse option)
    ("cancelled", r"\bhuy\b|\bcancel|\bbo don\b|\bvoid\b|\bkhong giao duoc\b|\bgiao that bai\b|\bthat bai\b|\bfailed\b"),
    ("delivering", r"\bdang giao\b|\bdang van chuyen\b|\bdang ship\b|\bshipping\b|\bcho giao\b|\bdelivering\b|\bin transit\b|\bdang gui\b"
                   r"|\bdang xu ly\b|\bcho xu ly\b|\bprocessing\b|\bpending\b"),
    ("completed", r"\bhoan thanh\b|\bxong\b|\bthanh cong\b|\bda giao\b|\bcompleted\b|\bdone\b|\bsuccess|\bda thanh toan\b|\bok\b|\bpaid\b|\bfinished\b|\bda lay\b"),
]
# Any other label stays unmapped (None) and is reported as an issue, so the semantic layer refuses
# status-dependent questions instead of guessing.


def canonical_status(value: Any) -> Optional[str]:
    """'Đã hủy' -> cancelled, 'Trả hàng' -> returned, 'Đang giao' -> delivering, 'xong' -> completed."""
    if value is None or (not isinstance(value, str) and pd.isna(value)):
        return None
    k = ascii_key(value)
    for canon, pattern in STATUS_KEYWORDS:
        if re.search(pattern, k):
            return canon
    return None


@dataclass
class RoleMatch:
    role: str
    table: str        # sql table name
    column: str       # sql column name
    score: float
    evidence: List[str] = field(default_factory=list)


@dataclass
class Capability:
    present: bool
    confidence: float          # how sure we are of `present` (0..1)
    evidence: str = ""


@dataclass
class EntityRecord:
    kind: str                  # product | customer | staff | supplier | product_group | customer_group
    code: Optional[str]
    names: List[str]           # every spelling found in the data
    table: str = ""


@dataclass
class SalesModel:
    line_table: Optional[str] = None
    header_table: Optional[str] = None
    join: Optional[Tuple[str, str]] = None            # (line col, header col) on the order id
    cols: Dict[str, Tuple[str, str]] = field(default_factory=dict)   # role -> (table, column)
    dims: Dict[str, Tuple[str, str]] = field(default_factory=dict)   # dimension table -> (key role in sales, key column in dim)
    conversion: Dict[str, Any] = field(default_factory=dict)        # unit-conversion table (see _attach_conversion)
    status_map: Dict[str, Optional[str]] = field(default_factory=dict)
    returns_by: List[str] = field(default_factory=list)   # how returns are recorded: status | negative_qty | return_qty
    grain: str = "line"                                # 'line' (product lines) or 'order' (one row per order)
    confidence: float = 0.0
    issues: List[str] = field(default_factory=list)


@dataclass
class BusinessCard:
    business_id: str
    readme_quality: str = "none"                       # none | notes | dictionary
    readme_data_end: Optional[str] = None              # "Dữ liệu có đến hết dd/mm/yyyy"
    roles: Dict[str, List[RoleMatch]] = field(default_factory=dict)
    table_kinds: Dict[str, List[str]] = field(default_factory=dict)   # sql table -> kinds
    sales: SalesModel = field(default_factory=SalesModel)
    coverage: Tuple[Optional[str], Optional[str]] = (None, None)      # order-date coverage of sales
    capabilities: Dict[str, Capability] = field(default_factory=dict)
    entities: List[EntityRecord] = field(default_factory=list)
    quality_flags: List[str] = field(default_factory=list)
    readme_conversions: List[Tuple[str, str, float, str]] = field(default_factory=list)  # (product, big unit, factor, base unit)

    def best(self, role: str, table: Optional[str] = None) -> Optional[RoleMatch]:
        for m in self.roles.get(role, []):
            if table is None or m.table == table:
                return m
        return None

    def has(self, concept: str) -> Optional[bool]:
        c = self.capabilities.get(concept)
        return None if c is None else c.present

    def known_names(self, kinds: Optional[Tuple[str, ...]] = None) -> set:
        from src.datamate.scoring import normalize_string
        out = set()
        for e in self.entities:
            if kinds is None or e.kind in kinds:
                out.update(normalize_string(n) for n in e.names)
                if e.code:
                    out.add(normalize_string(e.code))
        return out

    def to_prompt(self, max_roles: int = 40) -> str:
        """Compact text for the LLM prompt: roles, sales model, coverage, missing capabilities."""
        lines = [f"Khoảng ngày có dữ liệu bán hàng: {self.coverage[0]} .. {self.coverage[1]}"
                 + (f" (README: có đến hết {self.readme_data_end})" if self.readme_data_end else "")]
        s = self.sales
        if s.line_table or s.header_table:
            lines.append(f"Bảng dòng hàng: {s.line_table or '(không có)'}; bảng đơn: {s.header_table or '(không có)'}"
                         + (f"; JOIN {s.line_table}.{s.join[0]} = {s.header_table}.{s.join[1]}" if s.join else ""))
            if s.returns_by:
                lines.append("Trả hàng được ghi bằng: " + ", ".join(s.returns_by))
            if s.status_map:
                lines.append("Trạng thái đơn: " + "; ".join(f"{k!r} = {v or 'không rõ'}" for k, v in list(s.status_map.items())[:12]))
        shown = 0
        role_lines = []
        for role, matches in self.roles.items():
            if matches and shown < max_roles:
                m = matches[0]
                role_lines.append(f"{role}: {m.table}.{m.column}")
                shown += 1
        if role_lines:
            lines.append("Vai trò cột (tốt nhất): " + "; ".join(role_lines))
        missing = [k for k, c in self.capabilities.items() if not c.present and c.confidence >= 0.7]
        if missing:
            lines.append("Dữ liệu KHÔNG có: " + ", ".join(missing))
        return "\n".join(lines)

    def to_json(self) -> Dict[str, Any]:
        d = asdict(self)
        d["entities"] = {"n": len(self.entities), "by_kind": pd.Series([e.kind for e in self.entities]).value_counts().to_dict() if self.entities else {}}
        return d

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_json(), f, ensure_ascii=False, indent=1, default=str)


# ── README ────────────────────────────────────────────────────────────────────

README_COL_RE = re.compile(r"^- \*\*(.+?)\*\*:\s*(.+)$", re.M)
README_FILE_RE = re.compile(r"^## Cột trong `([^`]+)`", re.M)
README_END_RE = re.compile(r"có đến hết (?:ngày )?(\d{1,2})/(\d{1,2})/(\d{4})", re.I)


def parse_readme(readme: str) -> Tuple[Dict[Tuple[str, str], str], str, Optional[str]]:
    """({(file stem ascii, column ascii): description ascii}, quality, data end date)."""
    readme = unicodedata.normalize("NFC", readme or "")
    descs: Dict[Tuple[str, str], str] = {}
    # Split by "## Cột trong `file`" blocks
    positions = [(m.start(), table_stem(os.path.splitext(m.group(1))[0])) for m in README_FILE_RE.finditer(readme)]
    for i, (start, stem) in enumerate(positions):
        end = positions[i + 1][0] if i + 1 < len(positions) else len(readme)
        block = readme[start:end].split("\n## ", 1)[0]
        for m in README_COL_RE.finditer(block):
            descs[(header_key(stem), header_key(m.group(1)))] = ascii_key(m.group(2))
    m_end = README_END_RE.search(readme)
    data_end = f"{int(m_end.group(3)):04d}-{int(m_end.group(2)):02d}-{int(m_end.group(1)):02d}" if m_end else None
    quality = "dictionary" if descs else ("notes" if len(readme.strip()) > 80 else "none")
    return descs, quality, data_end


# ── Builder ───────────────────────────────────────────────────────────────────

def _value_evidence(role: str, col: ColumnInfo, db: BusinessDB, table: TableInfo) -> Tuple[float, str]:
    """Score adjustment from the column's values (types, distinct values)."""
    t = col.sql_type.upper()
    numeric = any(x in t for x in ("INT", "DOUBLE", "DECIMAL", "FLOAT", "BIGINT", "HUGEINT", "REAL"))
    temporal = "DATE" in t or "TIMESTAMP" in t
    if role in ("order_date", "payment_date", "due_date", "expiry_date", "delivery_date", "member_registration", "price_effective_date"):
        return (0.3, "temporal") if temporal else (-0.6, f"not a date ({t})")
    if role in ("quantity", "quantity_base", "line_amount", "unit_price", "order_total", "order_goods_amount",
                "paid_amount", "debt_remaining", "cost_per_unit", "stock_qty", "line_discount", "shipping_fee",
                "vat", "conversion_factor", "return_qty"):
        if numeric:
            return 0.2, "numeric"
        if col.note and "_so" in col.note:  # '1 ly' split into <col>_so
            return 0.0, "number+unit text"
        return -0.6, f"not numeric ({t})"
    if role == "status":
        vals = col.distinct_values or []
        canon = [canonical_status(v) for v in vals]
        if vals and sum(c is not None for c in canon) >= max(1, len(vals) * 0.6):
            return 0.4, "status-like values"
        return -0.5, "values do not look like order states"
    if role in ("product_name", "customer_name", "staff", "product_group", "customer_group", "supplier", "unit",
                "payment_method", "sales_channel", "customer_address", "customer_gender", "active_ingredient", "lot"):
        if t != "VARCHAR":
            return -0.5, f"not text ({t})"
        return 0.1, "text"
    if role in ("order_id", "product_code", "customer_code"):
        if temporal:
            return -0.8, "temporal"
        return 0.1, "id-like"
    return 0.0, ""


def detect_roles(db: BusinessDB, readme_descs: Dict[Tuple[str, str], str], table_stems: Dict[str, str]) -> Dict[str, List[RoleMatch]]:
    roles: Dict[str, List[RoleMatch]] = {r: [] for r in ROLE_PATTERNS}
    for table in db.tables.values():
        if table.hidden or table.original == "calendar.csv":
            continue
        stem = table_stems.get(table.sql_name, header_key(table.original))
        for col in table.columns:
            if table.n_rows and col.n_null == table.n_rows:
                continue
            name_key = header_key(col.original or col.sql_name)
            desc = readme_descs.get((stem, name_key))
            if desc is None:  # monthly slices / sheets share the dictionary of their file family
                desc = next((d for (s, c), d in readme_descs.items() if c == name_key and (stem.startswith(s) or s.startswith(stem))), None)
            for role, pats in ROLE_PATTERNS.items():
                score, ev = 0.0, []
                if desc and any(re.search(p, desc) for p in pats["desc"]):
                    score += 1.0
                    ev.append(f"README: {desc[:60]}")
                if any(re.search(p, name_key) for p in pats["name"]):
                    score += 0.6
                    ev.append(f"name: {name_key}")
                if score <= 0:
                    continue
                adj, why = _value_evidence(role, col, db, table)
                score += adj
                if why:
                    ev.append(why)
                if score > 0.3:
                    roles[role].append(RoleMatch(role, table.sql_name, col.sql_name, round(score, 3), ev))
        # Values alone can reveal an order-status column with an unusual name
        if not any(m.table == table.sql_name for m in roles["status"]):
            for col in table.columns:
                vals = col.distinct_values or []
                if col.sql_type == "VARCHAR" and 2 <= len(vals) <= 12:
                    canon = [canonical_status(v) for v in vals]
                    if sum(c is not None for c in canon) >= 0.6 * len(vals) and len({c for c in canon if c}) >= 2:
                        roles["status"].append(RoleMatch("status", table.sql_name, col.sql_name, 0.5, [f"status-like values {vals[:4]}"]))
    for r in roles:
        roles[r].sort(key=lambda m: (-m.score, m.table, m.column))
    return {r: v for r, v in roles.items() if v}


def _table_kinds(db: BusinessDB) -> Dict[str, List[str]]:
    out = {}
    for t in db.tables.values():
        if t.hidden or t.original == "calendar.csv":
            continue
        key = header_key(t.original)
        out[t.sql_name] = [k for k, p in TABLE_KIND_PATTERNS.items() if re.search(p, key)]
    return out


def _cols_of(roles: Dict[str, List[RoleMatch]], table: str) -> Dict[str, RoleMatch]:
    best: Dict[str, RoleMatch] = {}
    used_cols = set()
    # Strongest assignments first so one column gets one role inside a table
    for m in sorted((m for ms in roles.values() for m in ms if m.table == table), key=lambda m: -m.score):
        if m.role in best or m.column in used_cols:
            continue
        best[m.role] = m
        used_cols.add(m.column)
    return best


def build_sales_model(db: BusinessDB, roles: Dict[str, List[RoleMatch]], kinds: Dict[str, List[str]]) -> SalesModel:
    """Find the order-line table (product + quantity/amount) and the order-header table (order id + date)."""
    sm = SalesModel()
    excluded = {"purchase", "stock", "conversion", "price_history", "expense", "payment", "customer", "delivery", "debt"}
    candidates = []
    for tname, table in db.tables.items():
        if table.hidden or table.original == "calendar.csv" or set(kinds.get(tname, [])) & excluded:
            continue
        cols = _cols_of(roles, tname)
        line_score = (("product_name" in cols or "product_code" in cols) * 2 + ("quantity" in cols) + ("line_amount" in cols) * 2
                      + ("order_id" in cols) + ("order_date" in cols) * 0.5)
        header_score = (("order_id" in cols) * 2 + ("order_date" in cols) * 2 + ("status" in cols) + ("customer_code" in cols or "customer_name" in cols)
                        + ("order_total" in cols or "order_goods_amount" in cols))
        candidates.append((tname, cols, line_score, header_score, table.n_rows))

    lines = [c for c in candidates if ("product_name" in c[1] or "product_code" in c[1])
             and ("quantity" in c[1] or "quantity_base" in c[1] or "line_amount" in c[1])]
    lines.sort(key=lambda c: (-c[2], -c[4], c[0]))
    if lines:
        sm.line_table = lines[0][0]
        lcols = lines[0][1]
        for r, m in lcols.items():
            sm.cols[r] = (m.table, m.column)
        # Header table: has order id + date, joins on order id, is not the line table
        if "order_id" in lcols:
            heads = [c for c in candidates if c[0] != sm.line_table and "order_id" in c[1] and "order_date" in c[1]]
            heads.sort(key=lambda c: (-c[3], c[4], c[0]))
            if heads and _join_overlap(db, sm.line_table, lcols["order_id"].column, heads[0][0], heads[0][1]["order_id"].column) >= 0.5:
                sm.header_table = heads[0][0]
                sm.join = (lcols["order_id"].column, heads[0][1]["order_id"].column)
                for r, m in heads[0][1].items():
                    # Order-level attributes come from the header when the line table lacks them
                    if r not in sm.cols or r in ("order_date", "status", "customer_code", "customer_name", "payment_method", "sales_channel", "staff"):
                        sm.cols[r] = (m.table, m.column)
        sm.grain = "line"
    else:
        heads = [c for c in candidates if "order_id" in c[1] and "order_date" in c[1] and ("order_goods_amount" in c[1] or "order_total" in c[1])]
        heads.sort(key=lambda c: (-c[3], -c[4], c[0]))
        if heads:
            sm.header_table = heads[0][0]
            for r, m in heads[0][1].items():
                sm.cols[r] = (m.table, m.column)
            sm.grain = "order"

    if not sm.line_table and not sm.header_table:
        sm.issues.append("no sales table found")
        return sm

    # Dimension tables reached through a code (product catalogue, customer list), else through the
    # name: shops without codes link files by name ("sổ tay không có mã hàng, nối qua tên hàng")
    _attach_dimension(sm, db, roles, "product_code", PRODUCT_ATTRS)
    _attach_dimension(sm, db, roles, "customer_code", CUSTOMER_ATTRS)
    _attach_dimension_by_name(sm, db, roles, "product_name", PRODUCT_ATTRS)
    _attach_dimension_by_name(sm, db, roles, "customer_name", CUSTOMER_ATTRS)
    _attach_conversion(sm, db, roles, kinds)

    # Status values -> canonical states
    if "status" in sm.cols:
        t, c = sm.cols["status"]
        vals = [r[0] for r in db.conn.execute(f'SELECT DISTINCT "{c}" FROM "{t}" WHERE "{c}" IS NOT NULL ORDER BY 1 LIMIT 50').fetchall()]
        sm.status_map = {str(v): canonical_status(v) for v in vals}
        unknown = [v for v, k in sm.status_map.items() if k is None]
        if unknown:
            sm.issues.append(f"unrecognized status values: {unknown[:5]}")
        if "returned" in sm.status_map.values():
            sm.returns_by.append("status")
    if "quantity" in sm.cols and sm.line_table:
        t, c = sm.cols["quantity"]
        try:
            neg = db.conn.execute(f'SELECT COUNT(*) FROM "{t}" WHERE TRY_CAST("{c}" AS DOUBLE) < 0').fetchall()[0][0]
            if neg:
                sm.returns_by.append("negative_qty")
        except Exception:
            pass
    rq = _best_in(roles, "return_qty", {sm.line_table, sm.header_table})
    if rq:
        sm.cols["return_qty"] = (rq.table, rq.column)
        sm.returns_by.append("return_qty")

    needed = ["order_date"] + (["line_amount"] if sm.grain == "line" else ["order_goods_amount"])
    missing = [r for r in needed if r not in sm.cols]
    if sm.grain == "order" and "order_goods_amount" not in sm.cols and "order_total" in sm.cols:
        missing = [r for r in missing if r != "order_goods_amount"]
        sm.issues.append("revenue taken from order_total (may include shipping/discount)")
    if missing:
        sm.issues.append(f"missing roles: {missing}")
    used = [_best_score(roles, r, sm.cols[r]) for r in sm.cols]
    sm.confidence = round(min(1.0, (sum(used) / max(1, len(used))) / 1.6) * (0.5 if missing else 1.0), 3)
    return sm


PRODUCT_ATTRS = ["product_name", "product_group", "base_unit", "active_ingredient", "prescription_flag"]
CUSTOMER_ATTRS = ["customer_name", "customer_group", "customer_gender", "customer_address", "customer_phone", "member_registration"]


def _attach_dimension(sm: SalesModel, db: BusinessDB, roles: Dict[str, List[RoleMatch]], key_role: str, attrs: List[str]) -> None:
    """If the sales tables carry `key_role` (a code) but not some attribute, find a table with the
    same code and that attribute whose codes cover >= 80 % of the sales codes, and take it from there."""
    if key_role not in sm.cols:
        return
    kt, kc = sm.cols[key_role]
    sales_tables = {sm.line_table, sm.header_table}
    for key_match in roles.get(key_role, []):
        if key_match.table in sales_tables:
            continue
        tcols = _cols_of(roles, key_match.table)
        wanted = [a for a in attrs if a in tcols and a not in sm.cols]
        if not wanted:
            continue
        if _join_overlap(db, kt, kc, key_match.table, key_match.column) < 0.8:
            continue
        sm.dims[key_match.table] = (key_role, key_match.column)
        for a in wanted:
            sm.cols[a] = (tcols[a].table, tcols[a].column)


def _distinct_keys(db: BusinessDB, table: str, col: str, limit: int = 50000) -> set:
    rows = db.conn.execute(f'SELECT DISTINCT CAST("{col}" AS VARCHAR) FROM "{table}" WHERE "{col}" IS NOT NULL LIMIT {limit}').fetchall()
    return {ascii_key(r[0]) for r in rows if r[0] is not None}


def _attach_dimension_by_name(sm: SalesModel, db: BusinessDB, roles: Dict[str, List[RoleMatch]], name_role: str, attrs: List[str]) -> None:
    """Like _attach_dimension, joining on the accent/case/space-insensitive name (ascii_key)."""
    if name_role not in sm.cols:
        return
    st, sc = sm.cols[name_role]
    try:
        sales_keys = _distinct_keys(db, st, sc)
    except Exception:
        return
    if not sales_keys:
        return
    sales_tables = {sm.line_table, sm.header_table} | set(sm.dims)
    for name_match in roles.get(name_role, []):
        if name_match.table in sales_tables:
            continue
        tcols = _cols_of(roles, name_match.table)
        wanted = [a for a in attrs if a in tcols and a not in sm.cols]
        if not wanted:
            continue
        try:
            dim_keys = _distinct_keys(db, name_match.table, name_match.column)
        except Exception:
            continue
        if len(sales_keys & dim_keys) < 0.8 * len(sales_keys):
            continue
        sm.dims[name_match.table] = (name_role, name_match.column)
        for a in wanted:
            sm.cols[a] = (tcols[a].table, tcols[a].column)
        sales_tables.add(name_match.table)


README_CONV_RE = re.compile(r"^\s*-\s*(.+?):\s*1\s+(.+?)\s*=\s*(\d+(?:[.,]\d+)?)\s+(.+?)\s*$", re.M)


def parse_readme_conversions(readme: str) -> List[Tuple[str, str, float, str]]:
    """'- Coca Cola 330ml: 1 thùng = 24 lon' -> (product, big unit, factor, base unit)."""
    out = []
    for m in README_CONV_RE.finditer(unicodedata.normalize("NFC", readme or "")):
        out.append((m.group(1).strip(), m.group(2).strip(), float(m.group(3).replace(",", ".")), m.group(4).strip()))
    return out


def _attach_conversion(sm: SalesModel, db: BusinessDB, roles: Dict[str, List[RoleMatch]], kinds: Dict[str, List[str]]) -> None:
    """Locate a unit-conversion table (product, big unit, factor[, base unit]) for qty -> base units,
    used only when the sales lines carry no quantity already converted to base units."""
    if "quantity_base" in sm.cols or "quantity" not in sm.cols:
        return
    for m in roles.get("conversion_factor", []):
        tcols = _cols_of(roles, m.table)
        unit = tcols.get("big_unit") or tcols.get("unit")
        key = tcols.get("product_code") if "product_code" in sm.cols and "product_code" in tcols else tcols.get("product_name")
        if unit and key:
            sm.conversion = {"table": m.table, "key_role": key.role, "key": key.column, "unit": unit.column, "factor": m.column,
                             "base_unit": tcols["base_unit"].column if "base_unit" in tcols else None}
            return


def _best_in(roles: Dict[str, List[RoleMatch]], role: str, tables: set) -> Optional[RoleMatch]:
    for m in roles.get(role, []):
        if m.table in tables:
            return m
    return None


def _best_score(roles: Dict[str, List[RoleMatch]], role: str, tc: Tuple[str, str]) -> float:
    for m in roles.get(role, []):
        if (m.table, m.column) == tc:
            return m.score
    return 0.0


def _join_overlap(db: BusinessDB, t1: str, c1: str, t2: str, c2: str) -> float:
    """Share of distinct line order ids found in the header table."""
    try:
        n, hit = db.conn.execute(
            f'SELECT COUNT(DISTINCT a."{c1}"), COUNT(DISTINCT CASE WHEN b."{c2}" IS NOT NULL THEN a."{c1}" END) '
            f'FROM "{t1}" a LEFT JOIN (SELECT DISTINCT CAST("{c2}" AS VARCHAR) AS "{c2}" FROM "{t2}") b '
            f'ON CAST(a."{c1}" AS VARCHAR) = b."{c2}"').fetchall()[0]
        return hit / n if n else 0.0
    except Exception:
        return 0.0


def _coverage(db: BusinessDB, sm: SalesModel) -> Tuple[Optional[str], Optional[str]]:
    if "order_date" not in sm.cols:
        return None, None
    t, c = sm.cols["order_date"]
    try:
        mn, mx = db.conn.execute(f'SELECT MIN(CAST("{c}" AS DATE)), MAX(CAST("{c}" AS DATE)) FROM "{t}"').fetchall()[0]
        return (str(mn) if mn else None, str(mx) if mx else None)
    except Exception:
        return None, None


# Concept -> roles / table kinds that make it answerable. Used by the Guard (plan §4.6).
CONCEPT_REQUIREMENTS: Dict[str, Dict[str, List[str]]] = {
    "revenue": {"roles": ["line_amount", "order_goods_amount", "order_total"]},
    "order_count": {"roles": ["order_id", "line_amount", "order_goods_amount"]},
    "quantity": {"roles": ["quantity", "quantity_base"]},
    "cancel": {"roles": ["status"]},
    "return": {"roles": ["status", "return_qty"]},
    "customer": {"roles": ["customer_code", "customer_name"]},
    "customer_group": {"roles": ["customer_group"]},
    "customer_gender": {"roles": ["customer_gender"]},
    "customer_address": {"roles": ["customer_address"]},
    "payment_method": {"roles": ["payment_method"]},
    "sales_channel": {"roles": ["sales_channel"]},
    "staff": {"roles": ["staff"]},
    "product_group": {"roles": ["product_group"]},
    "unit_price": {"roles": ["unit_price", "line_amount"]},
    "purchase": {"kinds": ["purchase"]},
    "supplier": {"roles": ["supplier"], "kinds": ["purchase"]},
    # Without a stock table, stock is still imports − sales from the start of the data (rulebook §2 "đến hết ngày X")
    "stock": {"roles": ["stock_qty"], "kinds": ["stock", "purchase"]},
    "cost": {"roles": ["cost_per_unit"], "kinds": ["purchase"]},
    "gross_profit": {"roles": ["cost_per_unit"]},
    "net_profit": {"kinds": ["expense"]},
    "expense": {"kinds": ["expense"]},
    "debt": {"roles": ["debt_remaining", "paid_amount"], "kinds": ["payment", "debt"]},
    "expiry": {"roles": ["expiry_date"]},
    "lot": {"roles": ["lot"], "kinds": ["lot"]},
    "delivery": {"roles": ["delivery_date", "delivery_batch"], "kinds": ["delivery"]},
    # A price increase can be read from the unit price on the sales lines, not only from a price list
    "price_change": {"roles": ["price_effective_date", "unit_price"], "kinds": ["price_history"]},
    "shift": {"roles": ["order_time", "order_date"]},
    "member_registration": {"roles": ["member_registration"]},
    "active_ingredient": {"roles": ["active_ingredient"]},
    "prescription": {"roles": ["prescription_flag"]},
    "shipping_fee": {"roles": ["shipping_fee"]},
    "cancel_reason": {"roles": ["reason"]},
    "return_reason": {"roles": ["reason"]},
    "customer_age": {"roles": ["birth_date"]},
    "discount": {"roles": ["line_discount", "order_discount"]},
    "order_time": {"roles": ["order_time", "order_date"]},
    "points": {"roles": ["points"]},
    "voucher": {"roles": ["order_discount"]},
    "warranty": {"roles": ["warranty"]},
}


def build_capabilities(roles: Dict[str, List[RoleMatch]], kinds: Dict[str, List[str]], sm: SalesModel, db: BusinessDB) -> Dict[str, Capability]:
    all_kinds = {k for ks in kinds.values() for k in ks}
    caps: Dict[str, Capability] = {}
    for concept, req in CONCEPT_REQUIREMENTS.items():
        hits = [f"{m.table}.{m.column}" for r in req.get("roles", []) for m in roles.get(r, [])[:1] if m.score >= 0.6]
        khits = [k for k in req.get("kinds", []) if k in all_kinds]
        present = bool(hits or khits)
        # Absence is less certain when the README is poor: the column may exist under an unusual name
        caps[concept] = Capability(present, 0.9 if present else 0.75, ", ".join(hits + khits) or "no matching column/table")
    # A status column without any cancel/return value cannot answer cancel/return questions
    if sm.status_map:
        vals = set(sm.status_map.values())
        if "cancelled" not in vals and caps["cancel"].present:
            caps["cancel"] = Capability(False, 0.6, "status column has no cancelled value")
    # Shifts / peak hours need a time of day: in the creation-time column, else in the order timestamp
    has_time, where = False, ""
    for role in ("order_time", "order_date"):
        t_c = sm.cols.get(role)
        if not t_c:
            continue
        try:
            n = db.conn.execute(
                f'SELECT COUNT(*) FROM "{t_c[0]}" WHERE COALESCE(TRY_CAST("{t_c[1]}" AS TIME), '
                f'TRY_CAST(TRY_CAST("{t_c[1]}" AS TIMESTAMP) AS TIME)) <> TIME \'00:00:00\'').fetchall()[0][0]
        except Exception:
            n = 0
        if n > 0:
            has_time, where = True, f"{t_c[0]}.{t_c[1]}"
            break
    caps["shift"] = Capability(has_time, 0.85, f"time of day in {where}" if has_time else "order dates have no time of day")
    caps["order_time"] = caps["shift"]
    return caps


def build_entities(db: BusinessDB, roles: Dict[str, List[RoleMatch]], max_per_kind: int = 20000) -> List[EntityRecord]:
    """Group the spellings of each entity by its code (rulebook §8: 'File có mã thì theo mã')."""
    pairs = [("product", "product_code", "product_name"), ("customer", "customer_code", "customer_name")]
    singles = [("staff", "staff"), ("supplier", "supplier"), ("product_group", "product_group"), ("customer_group", "customer_group")]
    out: List[EntityRecord] = []
    for kind, code_role, name_role in pairs:
        by_code: Dict[str, set] = {}
        seen_names: set = set()
        # code + name in the same table
        for nm in roles.get(name_role, []):
            cm = next((m for m in roles.get(code_role, []) if m.table == nm.table), None)
            try:
                if cm:
                    rows = db.conn.execute(
                        f'SELECT DISTINCT CAST("{cm.column}" AS VARCHAR), CAST("{nm.column}" AS VARCHAR) FROM "{nm.table}" '
                        f'WHERE "{nm.column}" IS NOT NULL LIMIT {max_per_kind}').fetchall()
                    for code, name in rows:
                        if code:
                            by_code.setdefault(code, set()).add(name)
                        seen_names.add(name)
                else:
                    rows = db.conn.execute(f'SELECT DISTINCT CAST("{nm.column}" AS VARCHAR) FROM "{nm.table}" WHERE "{nm.column}" IS NOT NULL LIMIT {max_per_kind}').fetchall()
                    seen_names.update(r[0] for r in rows)
            except Exception:
                continue
        coded = {n for ns in by_code.values() for n in ns}
        for code in sorted(by_code):
            out.append(EntityRecord(kind, code, sorted(by_code[code])))
        for name in sorted(seen_names - coded):
            out.append(EntityRecord(kind, None, [name]))
    for kind, role in singles:
        for m in roles.get(role, [])[:2]:
            try:
                rows = db.conn.execute(f'SELECT DISTINCT CAST("{m.column}" AS VARCHAR) FROM "{m.table}" WHERE "{m.column}" IS NOT NULL LIMIT 2000').fetchall()
            except Exception:
                continue
            for (v,) in rows:
                out.append(EntityRecord(kind, None, [v], m.table))
    return out


def quality_flags(db: BusinessDB, sm: SalesModel, coverage: Tuple[Optional[str], Optional[str]], readme_end: Optional[str]) -> List[str]:
    flags = []
    if readme_end and coverage[1] and coverage[1] > readme_end:
        flags.append(f"sales dated after README end ({coverage[1]} > {readme_end})")
    if sm.line_table and "line_amount" in sm.cols:
        t, c = sm.cols["line_amount"]
        try:
            neg = db.conn.execute(f'SELECT COUNT(*) FROM "{t}" WHERE TRY_CAST("{c}" AS DOUBLE) < 0').fetchall()[0][0]
            if neg:
                flags.append(f"{neg} negative line amounts")
        except Exception:
            pass
    # qty × price ≈ amount (arithmetic check of the role assignment)
    if all(r in sm.cols for r in ("quantity", "unit_price", "line_amount")) and len({sm.cols[r][0] for r in ("quantity", "unit_price", "line_amount")}) == 1:
        t = sm.cols["line_amount"][0]
        q, p, a = (sm.cols[r][1] for r in ("quantity", "unit_price", "line_amount"))
        try:
            share = db.conn.execute(
                f'SELECT AVG(CASE WHEN ABS(TRY_CAST("{q}" AS DOUBLE) * TRY_CAST("{p}" AS DOUBLE) - TRY_CAST("{a}" AS DOUBLE)) <= 1 THEN 1.0 ELSE 0.0 END) '
                f'FROM "{t}" WHERE TRY_CAST("{a}" AS DOUBLE) IS NOT NULL').fetchall()[0][0]
            if share is not None and share < 0.5:
                flags.append(f"qty*price matches amount on only {share:.0%} of lines (discounts or wrong roles)")
        except Exception:
            pass
    sm_issue = [f"sales: {i}" for i in sm.issues]
    return flags + sm_issue


def build_business_card(bdata: BusinessData, db: BusinessDB) -> BusinessCard:
    descs, quality, data_end = parse_readme(bdata.readme)
    # table sql name -> file stem (ascii) used by the README dictionary
    stems = {}
    for original, sql in db.sql_name_of.items():
        stems[sql] = header_key(table_stem(original))
    roles = detect_roles(db, descs, stems)
    kinds = _table_kinds(db)
    sm = build_sales_model(db, roles, kinds)
    cov = _coverage(db, sm)
    card = BusinessCard(
        business_id=bdata.business_id, readme_quality=quality, readme_data_end=data_end, roles=roles,
        table_kinds=kinds, sales=sm, coverage=cov,
    )
    card.readme_conversions = parse_readme_conversions(bdata.readme)
    card.capabilities = build_capabilities(roles, kinds, sm, db)
    card.entities = build_entities(db, roles)
    card.quality_flags = quality_flags(db, sm, cov, data_end)
    return card
