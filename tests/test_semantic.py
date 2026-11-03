import pandas as pd
import pytest

from src.datamate.business_card import build_business_card, canonical_status, parse_readme, parse_readme_conversions
from src.datamate.db import BusinessDB
from src.datamate.loader import BusinessData, clean_dataframe, merge_split_tables
from src.datamate.semantic import CapabilityMissing, OutOfCoverage, SemanticLayer, SemanticUncertain
from src.datamate.time_resolver import Period
from datetime import date

README = """# Dữ liệu test
## Quy ước
- Dữ liệu có đến hết 31/03/2026.
- **Quy đổi đơn vị**: số lượng ghi theo đơn vị của dòng:
  - Coca Cola 330ml: 1 thùng = 24 lon
"""


def make(lines: pd.DataFrame, catalog: pd.DataFrame = None, readme: str = README):
    tables = {"BanHang": clean_dataframe(lines)}
    if catalog is not None:
        tables["DanhMuc"] = catalog
    bd = BusinessData(business_id="b", data_dir=".", readme=readme, tables=tables)
    db = BusinessDB(bd)
    card = build_business_card(bd, db)
    return db, card, SemanticLayer(db, card)


LINES = pd.DataFrame({
    "Mã đơn": ["O1", "O1", "O2", None, "O3", "O4", "O4", "O5", "O5"],
    "Ngày bán": ["01/03/2026", "01/03/2026", "02/03/2026", None, "02/03/2026", "03/03/2026", "03/03/2026", "03/03/2026", "03/03/2026"],
    "Trạng thái": ["Hoàn thành", "Hoàn thành", "Đang vận chuyển", None, "Đã hủy", "Hoàn thành", "Hoàn thành", "Không giao được", "Không giao được"],
    "Khách hàng": ["An", "An", "Tong Hoang Linh", None, "An", "Bình", "Bình", "An", "An"],
    "Tên hàng": ["Coca Cola 330ml", "Coca Cola 330ml", "Coca  cola 330ML", "Pepsi lon", "Pepsi lon", "Pepsi lon", "Pepsi lon", "Pepsi lon", "Pepsi lon"],
    "Đơn vị": ["thùng", "thùng", "lon", "lon", "lon", "lon", "lon", "lon", "lon"],
    "Số lượng": [1, 1, 2, 3, 1, 5, -1, 1, 1],
    "Thành tiền": [240000, 240000, 20000, 30000, 10000, 50000, -10000, 10000, 10000],
})
MARCH = Period(date(2026, 3, 1), date(2026, 3, 31))


def test_status_words_with_word_boundaries():
    assert canonical_status("Đang vận chuyển") == "delivering"   # "chuyển" contains "huy" but is not "huỷ"
    assert canonical_status("Đã hủy đơn") == "cancelled"
    assert canonical_status("Đã hoàn hàng") == "returned"
    assert canonical_status("Không giao được") == "cancelled"
    assert canonical_status("Đang xử lý") == "delivering"
    assert canonical_status("Thứ gì đó lạ") is None


def test_readme_parsing():
    _, quality, end = parse_readme(README)
    assert quality == "notes" and end == "2026-03-31"
    assert parse_readme_conversions(README) == [("Coca Cola 330ml", "thùng", 24.0, "lon")]


def test_revenue_applies_rulebook_defaults():
    db, card, sem = make(LINES)
    assert sem.ok, sem.problems
    # O1: the two identical lines are one export error (§8) -> 240 000. O2 (delivering) counts, its
    # second line inherits the order cells (§8) -> 50 000. O3 cancelled, O4 has a returned line
    # (partial return drops the whole order, §1), O5 'Không giao được' = cancelled.
    assert sem.revenue(MARCH).value == 290000
    assert sem.order_count(MARCH).value == 2
    assert sem.order_count(MARCH, states=("cancelled",)).value == 2
    assert round(sem.cancel_rate(MARCH).value, 2) == 40.0   # 2 cancelled of all 5 orders placed


def test_summary_filter_keeps_customer_named_tong():
    db, card, sem = make(LINES)
    names = [r[0] for r in db.conn.execute("SELECT DISTINCT customer_name FROM sl_orders").fetchall()]
    assert "Tong Hoang Linh" in names


def test_quantity_in_base_units_from_readme_and_name_variants():
    db, card, sem = make(LINES)
    # Coca: O1 1 thùng = 24 lon (dup line dropped) + O2 2 lon ("Coca  cola 330ML" is the same product)
    q = sem.quantity(MARCH, filters={"product_name": "Coca Cola 330ml"}).value
    assert q == 26
    top = sem.rank("product", "quantity", MARCH, k=1)
    assert top.value == ["Coca Cola 330ml"]


def test_guards_refuse_instead_of_guessing():
    db, card, sem = make(LINES)
    with pytest.raises(OutOfCoverage):
        sem.revenue(Period(date(2024, 1, 1), date(2024, 1, 31)))
    with pytest.raises(CapabilityMissing):
        sem.revenue(MARCH, shift="morning")       # dates without a time of day
    with pytest.raises(SemanticUncertain):
        sem.revenue(MARCH, filters={"product_group": "nước ngọt"})   # no group column at all


def test_unknown_state_is_refused():
    lines = LINES.copy()
    lines.loc[0:1, "Trạng thái"] = "Trạng thái lạ"
    db, card, sem = make(lines)
    with pytest.raises(SemanticUncertain):
        sem.revenue(MARCH)


def test_dimension_joined_by_name_when_no_code():
    catalog = pd.DataFrame({"Tên hàng": ["coca cola 330ml", "Pepsi lon"], "Nhóm hàng": ["nước ngọt", "nước ngọt"]})
    db, card, sem = make(LINES, catalog)
    assert "danh_muc" in card.sales.dims
    assert sem.revenue(MARCH, filters={"product_group": "Nước ngọt"}).value == 290000


def test_split_files_are_unioned_without_dropping_duplicates():
    t = {"SoBanHang_1": pd.DataFrame({"a": [1, 1]}), "SoBanHang_2026": pd.DataFrame({"a": [1]}), "Khac": pd.DataFrame({"a": [2]})}
    merged = merge_split_tables(t)
    assert merged == {"SoBanHang": ["SoBanHang_1", "SoBanHang_2026"]}
    assert len(t["SoBanHang"]) == 3
