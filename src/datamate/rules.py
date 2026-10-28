"""Rulebook loader and extractor for DataMate.
Dynamically reads rulebook.md from the active data split to avoid hard-coding.
"""

import os
from typing import Dict, Optional


class Rulebook:
    def __init__(self, rulebook_path: Optional[str] = None):
        self.raw_text = ""
        self.sections: Dict[str, str] = {}
        if rulebook_path and os.path.exists(rulebook_path):
            try:
                with open(rulebook_path, "r", encoding="utf-8") as f:
                    self.raw_text = f.read()
                self._parse_sections()
            except Exception as e:
                print(f"[Rules Warning] Failed to read {rulebook_path}: {e}")

    def _parse_sections(self):
        current_section = "intro"
        current_lines = []
        for line in self.raw_text.splitlines():
            if line.startswith("## "):
                if current_lines:
                    self.sections[current_section] = "\n".join(current_lines).strip()
                current_section = line[3:].strip()
                current_lines = [line]
            else:
                current_lines.append(line)
        if current_lines:
            self.sections[current_section] = "\n".join(current_lines).strip()

    def get_core_rules_prompt(self) -> str:
        """Returns standard concise rules to guide code generation."""
        return """- DOANH THU: Tổng thành tiền các dòng bán, KHÔNG trừ voucher toàn đơn, KHÔNG cộng VAT, KHÔNG cộng phí ship.
- SỐ ĐƠN / SỐ ĐƠN HÀNG: Đếm số lượng mã đơn khác nhau (unique): `df['Mã hóa đơn'].nunique()` hoặc `df['Mã đơn hàng'].nunique()`, KHÔNG dùng len() hay shape[0] vì 1 đơn có nhiều dòng sản phẩm.
- LỌC ĐƠN HUỶ / TRẢ: Dùng str.contains để loại bỏ mọi biến thể: `df = df[~df['cột_trạng_thái'].astype(str).str.lower().str.contains('hủy|huỷ|cancel|trả|hoàn')]` (ví dụ: 'Đã hủy', 'Hủy đơn'). Nếu không có cột trạng thái, tìm ở cột Ghi chú.
- GIÁ MỘT ĐƠN VỊ (ví dụ: 'giá bao nhiêu một gói/chai/lon/cái/thùng'): Lọc đúng dòng có đơn vị đó (vd: `df[df['Đơn vị (tính)'].str.lower() == 'gói']`) hoặc tính đơn giá quy đổi lẻ: `df['Thanh tien'] / df['Quy ra ĐV lẻ']`.
- BÁN CHẠY: Xếp theo tổng SỐ LƯỢNG quy về ĐƠN VỊ CƠ BẢN (dùng bảng quy đổi nếu có).
- KHÁCH SỘP: Giá trị trung bình mỗi đơn cao nhất, điều kiện mua ÍT NHẤT 3 ĐƠN.
- KHÁCH TRUNG THÀNH: Số NGÀY có mua hàng nhiều nhất.
- MẶT HÀNG ƯA CHUỘNG: Xuất hiện trong NHIỀU ĐƠN HÀNG nhất.
- BẰNG NHAU (TIE-BREAK): Xếp theo MÃ tăng dần (nếu không có mã thì theo TÊN tăng dần).
- GIỜ CAO ĐIỂM: Khung 1 giờ tròn (08:00, 09:00...) có nhiều đơn nhất.
- TIỀN: Là số nguyên VND, làm tròn về đồng (kể cả trung bình tiền).
- PHẦN TRĂM: Là số %, làm tròn 2 chữ số thập phân (ví dụ: 12.34)."""


_RULEBOOK_CACHE: Dict[str, Rulebook] = {}


def find_rulebook_path(start_dir: str, max_up: int = 2) -> Optional[str]:
    """Locate rulebook.md of the split: <split>/rulebook.md.
    Accepts the split dir itself or a business dir (<split>/data/<business_id>), searching upwards.
    """
    cur = os.path.abspath(start_dir)
    for _ in range(max_up + 1):
        candidate = os.path.join(cur, "rulebook.md")
        if os.path.exists(candidate):
            return candidate
        cur = os.path.dirname(cur)
    return None


def get_rulebook(data_dir: str) -> Rulebook:
    rb_path = find_rulebook_path(data_dir)
    key = rb_path or ""
    if key not in _RULEBOOK_CACHE:
        if rb_path is None:
            print(f"[Rules Warning] rulebook.md not found from {data_dir}")
        _RULEBOOK_CACHE[key] = Rulebook(rb_path)
    return _RULEBOOK_CACHE[key]
