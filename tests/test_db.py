import time

import pandas as pd
import pytest

from src.datamate.db import BusinessDB, to_sql_identifier
from src.datamate.loader import BusinessData


@pytest.fixture()
def db():
    orders = pd.DataFrame({
        "Mã đơn hàng": ["ORD1", "ORD2", "ORD3"],
        "ThoiGianLap": ["2025-04-01 06:00:39", "2025-04-01 14:10:00", "2025-04-02"],
        "Trạng thái": ["Hoàn thành", "Đã hủy", "Hoàn thành"],
        "Số lượng": ["1 ly", 2, "3 ly"],
        "Điện thoại": ["0909123456", "0912000111", None],
        "Tổng tiền": [10000, 20000, 30000],
    })
    bd = BusinessData(business_id="biz_t", data_dir=".", tables={"DonHang": orders})
    cal = pd.DataFrame({"date": ["2025-04-01"], "weekday": [2], "lunar_day": [4], "lunar_month": [3],
                        "lunar_year": [2025], "lunar_leap": [False], "holiday": [None]})
    database = BusinessDB(bd, cal)
    yield database
    database.close()


def test_identifiers():
    used = set()
    assert to_sql_identifier("Ngày đặt hàng", used) == "ngay_dat_hang"
    assert to_sql_identifier("ThoiGianLap", used) == "thoi_gian_lap"
    assert to_sql_identifier("Ngày đặt hàng", used) == "ngay_dat_hang_2"
    assert to_sql_identifier("2025", used) == "c_2025"
    assert to_sql_identifier("order", used) == "order_"


def test_column_types(db):
    cols = {c.sql_name: c for c in db.tables["don_hang"].columns}
    assert cols["thoi_gian_lap"].sql_type == "TIMESTAMP"
    assert cols["tong_tien"].sql_type == "BIGINT"
    assert cols["dien_thoai"].sql_type == "VARCHAR"  # leading zero kept
    assert "so_luong_so" in cols and "so_luong_don_vi" in cols
    assert "'Đã hủy'" in db.schema_text and "lich" in db.tables


def test_queries(db):
    df, err = db.execute("SELECT hour(thoi_gian_lap) AS h FROM don_hang WHERE ma_don_hang = 'ORD2'")
    assert err is None and df.iat[0, 0] == 14
    df, err = db.execute("SELECT SUM(so_luong_so) FROM don_hang")
    assert err is None and df.iat[0, 0] == 6
    df, err = db.execute("SELECT dien_thoai FROM don_hang WHERE ma_don_hang = 'ORD1'")
    assert df.iat[0, 0] == "0909123456"


@pytest.mark.parametrize("sql", [
    "DROP TABLE don_hang",
    "SELECT 1; SELECT 2",
    "SET threads = 4",
    "COPY don_hang TO 'out.csv'",
    "SELECT * FROM read_csv('x.csv')",
    "SELEC 1",
])
def test_rejects_non_select_and_file_access(db, sql):
    df, err = db.execute(sql)
    assert df is None and err


def test_timeout_interrupts_query(db):
    t = time.time()
    df, err = db.execute("SELECT COUNT(*) FROM range(1000000000) a, range(1000000) b WHERE a.range * b.range % 7 = 3", timeout=1)
    assert df is None and "quá" in err
    assert time.time() - t < 5
