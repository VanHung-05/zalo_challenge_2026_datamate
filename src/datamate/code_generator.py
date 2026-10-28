"""Code generator for DataMate.
Builds tailored prompts containing business schema, rules, and question context,
and extracts Python code from LLM responses.
"""

import re
from typing import Any, Dict, List, Optional
import pandas as pd

from src.datamate.loader import BusinessData
from src.datamate.profiler import SchemaCard


def extract_python_code(text: str) -> str:
    """Extract Python code block from LLM response, stripping markdown fences robustly."""
    if not text:
        return ""

    # 1. Look for closed code block ```python\n ... \n``` or ```\n ... \n```
    match = re.search(r"```(?:python)?\s*\n(.*?)\n```", text, re.DOTALL | re.IGNORECASE)
    if match:
        code = match.group(1).strip()
    else:
        # 2. Look for unclosed code block starting with ```python or ```
        match_unclosed = re.search(r"```(?:python)?\s*\n(.*)", text, re.DOTALL | re.IGNORECASE)
        if match_unclosed:
            code = match_unclosed.group(1).strip()
        else:
            code = text.strip()

    # Clean any stray markdown fences or language tags on individual lines
    cleaned_lines = []
    for line in code.splitlines():
        stripped = line.strip()
        if stripped.startswith("```") or stripped.endswith("```"):
            continue
        cleaned_lines.append(line)

    return "\n".join(cleaned_lines).strip()


def build_schema_summary(business_data: BusinessData, max_tables: int = 6) -> str:
    """Generate compact schema description of business tables."""
    lines = []
    # If unified BanHang exists, prioritize it
    tables_to_show = []
    if "BanHang" in business_data.tables:
        tables_to_show.append("BanHang")
    
    for t_name in business_data.tables:
        if t_name not in tables_to_show and len(tables_to_show) < max_tables:
            tables_to_show.append(t_name)

    for t_name in tables_to_show:
        df = business_data.tables[t_name]
        lines.append(f"- Bảng `data['{t_name}']` ({len(df):,} dòng):")
        col_desc = []
        for col in df.columns:
            # Get 2 sample values
            samples = df[col].dropna().head(2).tolist()
            sample_str = f" (vd: {samples})" if samples else ""
            col_desc.append(f"  * `{col}` [{df[col].dtype}]{sample_str}")
        lines.extend(col_desc)

    return "\n".join(lines)


def build_prompt(
    question: str,
    as_of_date: str,
    business_data: BusinessData,
    rules_summary: str,
    time_range: Optional[Any] = None,
    entity_hints: Optional[str] = None,
    error_feedback: Optional[str] = None,
    previous_code: Optional[str] = None
) -> str:
    """Construct full prompt for code generation."""
    schema_text = build_schema_summary(business_data)
    readme_text = business_data.readme.strip()
    if len(readme_text) > 800:
        readme_text = readme_text[:800] + "\n..."

    time_section = f"as_of_date = \"{as_of_date}\""
    if time_range is not None:
        time_section += f"""
start_date = "{time_range.start_date}"
end_date = "{time_range.end_date}"
* CHÚ Ý: Câu hỏi yêu cầu dữ liệu trong khoảng '{time_range.description}' (từ {time_range.start_date} đến {time_range.end_date}). Hãy lọc đúng khoảng này."""

    entity_section = f"\n{entity_hints}\n" if entity_hints else ""

    prompt = f"""Bạn là chuyên gia phân tích dữ liệu kinh doanh (Data Analyst). Hãy viết code Python (pandas) để trả lời chính xác câu hỏi dựa trên dữ liệu được cung cấp.

### QUY TẮC CỐT LÕI (RULEBOOK):
{rules_summary}

### DỮ LIỆU CỦA CỬA HÀNG:
Dict `data` chứa các bảng pandas DataFrame:
{schema_text}

Thông tin bổ sung (README):
{readme_text}
{entity_section}
### THỜI GIAN:
{time_section}

### CÂU HỎI:
"{question}"

### YÊU CẦU BẮT BUỘC:
1. Gán kết quả cuối cùng vào biến `result`.
2. CHỈ sử dụng biến có sẵn: `data` (dict các DataFrame), `pd`, `np`, `datetime`, `date`, `timedelta`, `as_of_date`.
3. KHÔNG import thêm thư viện nào khác.
4. Lọc ngày: Chú ý so sánh ngày dưới dạng chuỗi 'YYYY-MM-DD' hoặc datetime.
5. Loại bỏ đơn bị huỷ / hoàn tiền khi tính doanh thu hoặc số lượng bán.
6. Nếu kết quả là tiền, để số nguyên VND (hoặc float tròn). Nếu là %, để số phần trăm (vd: 12.34).
7. Nếu câu hỏi không thể trả lời từ dữ liệu (thiếu cột, thiếu thông tin), gán: result = "UNANSWERABLE"
8. Viết code trong khối ```python ... ```.
"""

    if error_feedback and previous_code:
        prompt += f"""
### LƯU Ý SỬA LỖI:
Đoạn code trước đã gặp lỗi khi chạy:
```python
{previous_code}
```
Thông báo lỗi:
{error_feedback}
Hãy phân tích nguyên nhân lỗi và viết lại đoạn code chính xác.
"""

    return prompt
