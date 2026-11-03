import time

import pandas as pd

from src.datamate.question import analyze_question

CAL = pd.read_csv("train/calendar.csv")


def qa(q, d="2026-07-01"):
    return analyze_question(q, d, CAL)


def test_answer_types():
    assert qa("Tháng 4/2026 ngày nào cửa hàng bán được nhiều tiền nhất?").answer_type == "date"
    r = qa("tu thang 12/2025 toi thang 2/2026, thang nao cua hang ban duoc nhiu tien nhat vay")
    assert r.answer_type == "date" and r.date_granularity == "month"
    assert qa("dthu 3 thang 4/2026, 5/2026, 6/2026 co tang lien tuc khong").answer_type == "boolean"
    assert qa("Xu hướng doanh thu 3 tháng 6/2025 → 7/2025 → 8/2025: có phải tháng 7/2025 cao hơn tháng 6/2025 nhỉ?").answer_type == "boolean"
    r = qa("Cho mình 5 khách mua nhiều tiền nhất tháng 6/2025.")
    assert r.answer_type == "list" and r.top_k == 5
    assert qa("check gium dthu cac don ban qua dien thoai va doanh thu cac don ban tai quay thang nay lan luot la bao nhieu").answer_type == "list"
    assert qa("Cuối ngày hôm nay Bánh croissant còn mấy hộp chẵn, mấy cái lẻ?").answer_type == "list"
    r = qa("liệt kê 3 khách nợ nhìu nhất tính đến hết ngày 07/07/2025, kèm số tiền còn nợ")
    assert r.answer_type == "table" and r.top_k == 3
    assert qa("ti le huy don thang 4/2026 la bn %").answer_type == "percent"
    assert qa("Khách nào chi nhiều tiền nhất cho cửa hàng trong tháng 5/2026 nhỉ?").answer_type == "string"
    assert qa("Hôm nay cửa hàng bán được bao nhiêu tiền?").answer_type == "number"
    assert qa("Trong tháng này, số ly size L bán được là bao nhiêu, tính mọi loại đồ uống?").answer_type == "number"


def test_percent_sign_in_product_name_is_not_percent():
    assert qa("ton kho Otrivin 0,1% xit mui chai 10ml cuoi ngay hom nay la bao nhieu chai vay").answer_type == "number"


def test_negated_which_is_count():
    assert qa("Giai đoạn từ 05/07/2025 đến hết 11/07/2025, cửa hàng có mấy mặt hàng ế hẳn, không ai mua cái nào ạ?").answer_type == "number"


def test_pack_size_is_not_top_k():
    r = qa("het hom nay, Efticol 0,9% loc 10 lo x 10ml con lai may lo trong kho")
    assert r.answer_type == "number" and r.top_k is None


def test_thresholds():
    r = qa("tháng 9/2025 có mấy đơn có doanh thu từ 430.000 đồng trở lên")
    assert (r.thresholds[0]["value"], r.thresholds[0]["op"]) == (430000, ">=")
    r = qa("Bao nhiêu khách mua ít nhất 11 đơn trong tháng 10 năm 2025?")
    assert (r.thresholds[0]["value"], r.thresholds[0]["op"]) == (11, ">=")
    r = qa("dem so khach da mua ≥ 8 don trong thang 12/2025")
    assert (r.thresholds[0]["value"], r.thresholds[0]["op"]) == (8, ">=")
    r = qa("có bao nhiêu đơn trên 2tr")
    assert (r.thresholds[0]["value"], r.thresholds[0]["op"]) == (2000000, ">")
    assert qa("doanh thu từ ngày 20/10/2025 đến 22/10/2025").thresholds == []


def test_overrides():
    assert qa("doanh thu tháng 5 tính cả đơn huỷ").overrides.get("include_cancelled")
    assert qa("trong thang 3/2025, xet theo doanh thu, ngay nao ban dat nhat").overrides.get("by_revenue")
    assert qa("doanh thu chưa trừ giảm giá tháng 5").overrides.get("before_discount")
    assert qa("Hôm nay cửa hàng bán được bao nhiêu tiền?").overrides == {}


def test_unanswerability_signals():
    assert qa("Nhân viên Phùng Thu Trang làm việc có tận tâm với khách không?").subjective
    assert qa("check giùm khách hay khen nhân viên nào nhất").subjective
    assert qa("thuốc có hoạt chất nhôm phosphat người lớn nên uống mỗi lần mấy viên").external_knowledge
    assert qa("dự báo doanh thu tháng sau").forecast
    # look-alikes that are answerable
    assert not qa("Doanh thu từ đầu năm tới hết hôm nay là bao nhiêu?").forecast
    assert not qa("Từ tháng 2/2026 đến tháng 4/2026, doanh thu tháng sau có luôn cao hơn tháng trước không?").forecast
    assert not qa("tháng này cửa hàng đã chở giao bao nhiêu đợt hàng cho khách nhỉ").external_knowledge
    assert not qa("Cuối ngày 09/07/2026, tổng giá trị hàng còn trong kho tính theo giá vốn là bao nhiêu tiền?").subjective


def test_concepts():
    c = qa("cho hỏi trong tháng 12/2025, khách nữ chiếm bao nhiu % doanh thu").concepts
    assert "customer_gender" in c and "revenue" in c
    assert "customer_address" in qa("Có bao nhiêu khách hàng ở đường Trần Não?").concepts
    assert "stock" in qa("het hom nay, bia tiger 330ml con lai may lon trong kho ha").concepts
    assert "purchase" in qa("Cho hỏi tổng tiền nhập hàng tháng này là bao nhiêu?").concepts
    assert "gross_profit" in qa("Cho hỏi tháng 3/2025 cửa hàng lãi gộp bao nhiêu tiền?").concepts
    assert "debt" in qa("Đến hết ngày 09/07/2026, số tiền khách còn thiếu là bao nhiêu?").concepts
    assert "sales_channel" in qa("Tháng này, doanh thu các đơn bán qua điện thoại chiếm bao nhiêu phần trăm tổng doanh thu?").concepts
    assert "payment_method" in qa("dthu các đơn trả bằng COD và doanh thu các đơn trả bằng tiền mặt tháng này").concepts
    assert "cancel_reason" in qa("thang 3/2026 co bnhiu don bi huy vi giao tre").concepts
    assert "return_reason" in qa("thang 3/2026, tra hang vi loi va tra vi mua nham, ben nao nhieu don hon").concepts
    assert "staff" not in qa("sữa đặc ông thọ bán được mấy hộp").concepts


def test_speed():
    t0 = time.time()
    for _ in range(200):
        qa("doanh thu từ đầu tháng này đến hết hôm nay hơn hay kém cùng kỳ tháng trước bao nhiêu %")
    assert (time.time() - t0) / 200 < 0.02
