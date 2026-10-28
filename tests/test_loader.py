import openpyxl
import pandas as pd

from src.datamate.loader import clean_dataframe, load_business_data, parse_vietnamese_date


def test_parse_date_keeps_time_only_when_asked():
    assert parse_vietnamese_date("2025-04-01 06:00:39") == "2025-04-01"
    assert parse_vietnamese_date("2025-04-01 06:00:39", keep_time=True) == "2025-04-01 06:00:39"
    assert parse_vietnamese_date("01/04/2025 14:05", keep_time=True) == "2025-04-01 14:05:00"
    assert parse_vietnamese_date("2025-04-01 00:00:00", keep_time=True) == "2025-04-01"


def test_parse_date_excel_serial():
    assert parse_vietnamese_date(45717) == "2025-03-01"
    assert parse_vietnamese_date("45748") == "2025-04-01"
    assert parse_vietnamese_date("45717.0") == "2025-03-01"
    # Not a plausible serial: number is kept as is
    assert parse_vietnamese_date(30) == 30


def test_clean_dataframe_drops_repeated_header_rows():
    df = pd.DataFrame({
        "Mã đơn": ["ORD1", "Mã đơn", "ORD2"],
        "Ngày đặt": ["01/01/2025", "Ngày đặt", "02/01/2025"],
        "Tổng tiền": ["10.000", "Tổng tiền", "20.000"],
    })
    out = clean_dataframe(df)
    assert out["Mã đơn"].tolist() == ["ORD1", "ORD2"]
    assert out["Tổng tiền"].tolist() == [10000, 20000]


def test_load_business_merges_same_schema_sheets(tmp_path):
    biz = tmp_path / "biz_x"
    biz.mkdir()
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "T01-2025"
    ws.append(["Mã HĐ", "Thời gian", "Tổng tiền"])
    ws.append(["ORD1", "05/01/2025 08:30", 1000])
    ws2 = wb.create_sheet("T02-2025")
    ws2.append(["Mã HĐ", "Thời gian", "Tổng tiền"])
    ws2.append(["ORD2", "03/02/2025 15:00", 2000])
    wb.save(biz / "HoaDon.xlsx")

    bd = load_business_data(str(biz), use_cache=False, keep_time=True)
    assert list(bd.tables) == ["HoaDon"]
    t = bd.tables["HoaDon"]
    assert t["sheet"].tolist() == ["T01-2025", "T02-2025"]
    assert t["Thời gian"].tolist() == ["2025-01-05 08:30:00", "2025-02-03 15:00:00"]
