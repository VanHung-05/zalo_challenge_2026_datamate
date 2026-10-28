"""Text-to-SQL agent for DataMate (SQL mode, Phase 1 baseline).
Prompt (rulebook + README + schema + question + as-of date) -> LLM writes TYPE + one DuckDB SELECT
-> execute read-only with timeout -> self-debug on error (<= max_retries) -> format the answer.
"""

import re
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.datamate.agent import infer_expected_type
from src.datamate.db import BusinessDB
from src.datamate.formatter import format_submission_item
from src.datamate.llm import LLMWrapper
from src.datamate.time_resolver import resolve_time


VALID_TYPES = ["number", "percent", "string", "date", "boolean", "list", "table", "unanswerable"]
SCALAR_TYPES = {"number", "percent", "string", "date", "boolean"}
WEEKDAYS_VI = ["Thứ Hai", "Thứ Ba", "Thứ Tư", "Thứ Năm", "Thứ Sáu", "Thứ Bảy", "Chủ nhật"]
MAX_README_CHARS = 6000
MAX_LOGGED_RESPONSE_CHARS = 4000

INSTRUCTIONS = """Bạn là chuyên gia phân tích dữ liệu cho cửa hàng nhỏ ở Việt Nam. Nhiệm vụ: viết MỘT câu truy vấn SQL (DuckDB) trên dữ liệu của cửa hàng để trả lời câu hỏi. Mọi con số phải do SQL tính ra.

## RULEBOOK (luật mặc định; câu hỏi không nói khác thì BẮT BUỘC theo luật này)
{rulebook}
"""

BUSINESS_SECTION = """
## DỮ LIỆU CỦA CỬA HÀNG
### README của cửa hàng
{readme}

### SCHEMA (DuckDB, chỉ đọc; tên bảng/cột đã chuyển sang không dấu, tên gốc ghi trong chú thích)
{schema}
"""

QUESTION_SECTION = """
## CÂU HỎI
Hôm nay (as_of_date) là {as_of_date} ({weekday}).
Câu hỏi: "{question}"
{hints}
## CÁCH TRẢ LỜI
Chỉ trả lời đúng định dạng sau, không giải thích:
TYPE: <một trong: number, percent, string, date, boolean, list, table, unanswerable>
```sql
SELECT ...
```

Quy tắc SQL quan trọng theo nghiệp vụ (BẮT BUỘC TUÂN THỦ):
1. Đúng một câu SELECT (được dùng WITH), cú pháp DuckDB, chỉ dùng đúng tên bảng/cột có trong SCHEMA.
2. DOANH THU, SỐ ĐƠN, SỐ LƯỢNG BÁN, XẾP HẠNG: BẮT BUỘC loại bỏ đơn huỷ và trả hàng:
   - Nếu có cột trạng thái/ghi chú đơn: KHÔNG lấy đơn có trạng thái chứa 'hủy', 'huy', 'trả', 'tra', 'cancel', 'return'.
     Ví dụ: `(trang_thai IS NULL OR (LOWER(trang_thai) NOT LIKE '%h_y%' AND LOWER(trang_thai) NOT LIKE '%tr_%'))`
   - Đơn đang giao / đang vận chuyển VẪN TÍNH vào doanh thu!
3. ĐƠN CÓ DOANH THU/GIÁ TRỊ TỪ X TRỞ LÊN:
   - Bảng bán hàng chi tiết là từng dòng mặt hàng. Muốn tính số đơn có tổng giá trị >= X:
     `WITH don AS (SELECT ma_don_hang, SUM(thanh_tien) AS val FROM ban_hang WHERE <lọc ngày, trạng thái> GROUP BY ma_don_hang HAVING SUM(thanh_tien) >= X) SELECT COUNT(*) FROM don`
4. GIÁ NHẬP BÌNH QUÂN một mặt hàng trong kỳ:
   - Bình quân GIA QUYỀN THEO SỐ LƯỢNG NHẬP: `SUM(so_luong * don_gia) * 1.0 / SUM(so_luong)`.
   - TUYỆT ĐỐI KHÔNG dùng AVG(don_gia) và KHÔNG lấy tổng tiền cả phiếu nhập chia số lượng một mặt hàng!
5. TỈ LỆ HUỶ ĐƠN TRONG KỲ:
   - Tỉ lệ = `ROUND(COUNT(DISTINCT CASE WHEN LOWER(trang_thai) LIKE '%h_y%' THEN ma_hoa_don END) * 100.0 / COUNT(DISTINCT ma_hoa_don), 2)`
   - Mẫu số phải là MỌI ĐƠN trong kỳ (cả hoàn thành, huỷ, trả, đang giao).
6. JOIN BẢNG CHI TIẾT & HOÁ ĐƠN / SẢN PHẨM:
   - Nếu bảng chi tiết bán/nhập hàng không có cột ngày/thời gian thì PHẢI JOIN với bảng hoá đơn/phiếu nhập (qua `ma_hoa_don` hoặc `ma_nhap_hang`) để lọc ngày.
   - Nếu cần thuộc tính hàng (hoạt chất, thuốc kê đơn...) thì JOIN với bảng danh mục/thuốc (qua `ma_hang`).
   - Tham khảo các gợi ý JOIN ở cuối phần SCHEMA!
7. TÌM KIẾM CHUỖI TIẾNG VIỆT: Dùng `LOWER(cot) LIKE '%từ_khóa%'` hoặc `ILIKE '%...%'`, tránh so sánh bằng tuyệt đối `=`.
8. LỌC THỜI GIAN:
   - Dùng `CAST(cot_thoi_gian AS DATE) BETWEEN DATE 'start_date' AND DATE 'end_date'`.
   - "Tháng này" (as_of_date: YYYY-MM-DD): tính từ ngày đầu tháng đến hết hôm nay: `BETWEEN DATE 'YYYY-MM-01' AND DATE '{as_of_date}'`.
   - "Hôm nay": đúng ngày `{as_of_date}`.
   - "Hôm qua": đúng ngày `{as_of_date} - 1 ngày`.
9. HÌNH DẠNG KẾT QUẢ THEO TYPE:
   - number, percent, string, date, boolean: đúng 1 dòng, 1 cột.
   - Tiền VND: `ROUND(x)` (số nguyên). percent: `ROUND(x, 2)` (số phần trăm, ví dụ 12.34). Số trung bình khác: `ROUND(x, 2)`.
   - date: chuỗi `strftime(x, '%Y-%m-%d')` hoặc `strftime(x, '%Y-%m')`. boolean: TRUE/FALSE.
   - string hoặc list: tên như trong dữ liệu, không kèm số đo.
10. Nếu dữ liệu không có thông tin cần để trả lời thì trả lời `TYPE: unanswerable` và không viết SQL.
"""

REPAIR_SECTION = """
## LẦN TRƯỚC BỊ LỖI, HÃY SỬA
TYPE lần trước: {prev_type}
```sql
{prev_sql}
```
Vấn đề: {feedback}
Hãy viết lại đúng định dạng (TYPE + khối ```sql```).
"""


def build_sql_prompt(
    question: str,
    as_of_date: str,
    rulebook_text: str,
    readme: str,
    schema_text: str,
    hints: str = "",
    previous_type: Optional[str] = None,
    previous_sql: Optional[str] = None,
    feedback: Optional[str] = None,
) -> str:
    """Static parts first (rulebook, then business) so the LLM server can reuse the prompt-prefix cache."""
    try:
        weekday = WEEKDAYS_VI[datetime.strptime(as_of_date[:10], "%Y-%m-%d").weekday()]
    except ValueError:
        weekday = "không rõ thứ"
    readme = (readme or "").strip() or "(không có README)"
    if len(readme) > MAX_README_CHARS:
        readme = readme[:MAX_README_CHARS] + "\n..."

    prompt = INSTRUCTIONS.format(rulebook=(rulebook_text or "(không có rulebook)").strip())
    prompt += BUSINESS_SECTION.format(readme=readme, schema=schema_text)
    prompt += QUESTION_SECTION.format(
        as_of_date=as_of_date, weekday=weekday, question=question, hints=f"{hints}\n" if hints else ""
    )
    if feedback:
        prompt += REPAIR_SECTION.format(prev_type=previous_type or "(không có)", prev_sql=previous_sql or "(không có)", feedback=feedback)
    return prompt


def parse_llm_response(text: str) -> Tuple[Optional[str], Optional[str]]:
    """Extract (answer_type, sql) from the LLM response. Either may be None."""
    if not text:
        return None, None
    answer_type = None
    m_type = re.search(r"TYPE\s*[:=]\s*`?([A-Za-z_]+)", text, re.IGNORECASE)
    if m_type and m_type.group(1).lower() in VALID_TYPES:
        answer_type = m_type.group(1).lower()

    sql = None
    m_block = re.search(r"```(?:sql|duckdb)?\s*\n(.*?)(?:```|$)", text, re.DOTALL | re.IGNORECASE)
    if m_block:
        sql = m_block.group(1)
    else:
        m_bare = re.search(r"^\s*((?:WITH|SELECT)\b.*)", text, re.DOTALL | re.IGNORECASE | re.MULTILINE)
        if m_bare:
            sql = m_bare.group(1)
    if sql is not None:
        sql = sql.strip().rstrip(";").strip() or None
    return answer_type, sql


def _to_python(v: Any) -> Any:
    if isinstance(v, np.generic):
        return v.item()
    if isinstance(v, pd.Timestamp):
        return v.to_pydatetime()
    return v


def _is_missing(v: Any) -> bool:
    try:
        return v is None or bool(pd.isna(v))
    except (TypeError, ValueError):
        return False


def sql_result_to_raw(df: pd.DataFrame, answer_type: str) -> Tuple[Any, Optional[str]]:
    """Turn the result DataFrame into the raw value expected by format_submission_item.
    Returns (raw, None) or (None, feedback for the LLM).
    """
    n_rows, n_cols = df.shape
    if n_rows == 0:
        return None, "Kết quả rỗng (0 dòng); kiểm tra lại điều kiện lọc (ngày, trạng thái, tên)."

    if answer_type in SCALAR_TYPES:
        if n_rows != 1:
            return None, f"TYPE {answer_type} cần đúng 1 dòng, kết quả có {n_rows} dòng."
        value = _to_python(df.iat[0, 0])
        if _is_missing(value):
            return None, "Kết quả là NULL; kiểm tra lại điều kiện lọc và tên cột."
        return value, None

    if answer_type == "list":
        if n_cols == 1:
            values = [_to_python(v) for v in df.iloc[:, 0].tolist()]
        elif n_rows == 1:
            values = [_to_python(v) for v in df.iloc[0].tolist()]
        else:
            values = [_to_python(v) for v in df.iloc[:, 0].tolist()]
        if any(_is_missing(v) for v in values):
            return None, "Danh sách có phần tử NULL."
        return values, None

    if answer_type == "table":
        return df.map(_to_python) if hasattr(df, "map") else df.applymap(_to_python), None

    return None, f"TYPE không hợp lệ: {answer_type}"


def _preview(df: Optional[pd.DataFrame], max_rows: int = 5) -> Optional[Dict[str, Any]]:
    if df is None:
        return None
    return {"columns": [str(c) for c in df.columns], "n_rows": len(df), "head": df.head(max_rows).astype(str).values.tolist()}


def diagnose_duckdb_error(error_msg: str, db: BusinessDB, sql: str) -> str:
    """Analyze DuckDB error message and provide concrete, actionable hints for the LLM."""
    if not error_msg:
        return ""

    # 1. Catalog Error: Table not found (with "Did you mean ...")
    m_did_you_mean = re.search(r'Table with name (\S+) does not exist!\s*Did you mean "([^"]+)"\?', error_msg, re.IGNORECASE)
    if m_did_you_mean:
        wrong_t, right_t = m_did_you_mean.group(1), m_did_you_mean.group(2)
        return f"Bảng '{wrong_t}' không tồn tại. DuckDB gợi ý bảng đúng là '{right_t}'. Hãy thay '{wrong_t}' bằng '{right_t}'."

    # 1b. Table does not exist (general)
    m_table_not_exist = re.search(r'Table with name (\S+) does not exist', error_msg, re.IGNORECASE)
    if m_table_not_exist:
        tbl = m_table_not_exist.group(1)
        valid_tables = [t.sql_name for t in db.tables.values() if not t.hidden]
        return f"Bảng '{tbl}' không tồn tại. Chỉ được dùng các bảng có trong SCHEMA: {', '.join(valid_tables)}."

    # 2. Binder Error: Referenced table not found
    m_ref_table = re.search(r'Referenced table "([^"]+)" not found', error_msg, re.IGNORECASE)
    if m_ref_table:
        tbl = m_ref_table.group(1)
        return (
            f"Lỗi cú pháp: Bạn đang dùng cột từ bảng '{tbl}' nhưng trong mệnh đề FROM chưa JOIN bảng này! "
            f"Hãy thêm 'JOIN {tbl} ON ...' vào mệnh đề FROM."
        )

    # 3. Binder Error: Table does not have a column named ...
    m_no_col = re.search(r'Table "([^"]+)" does not have a column named "([^"]+)"', error_msg, re.IGNORECASE)
    if m_no_col:
        tbl, col = m_no_col.group(1), m_no_col.group(2)
        table_obj = db.tables.get(tbl)
        if table_obj:
            cols = [c.sql_name for c in table_obj.columns]
            return f"Bảng '{tbl}' không có cột '{col}'. Các cột hợp lệ của bảng '{tbl}' gồm: {', '.join(cols[:15])}."
        return f"Bảng '{tbl}' không có cột '{col}'. Hãy tra cứu lại SCHEMA để xem cột đó thuộc bảng nào và JOIN thêm bảng đó."

    # 4. Referenced column not found in FROM clause
    m_col_not_found = re.search(r'Referenced column "([^"]+)" not found in FROM clause', error_msg, re.IGNORECASE)
    if m_col_not_found:
        col = m_col_not_found.group(1)
        candidate_tables = []
        for t in db.tables.values():
            if not t.hidden and any(c.sql_name == col for c in t.columns):
                candidate_tables.append(t.sql_name)
        if candidate_tables:
            return (
                f"Cột '{col}' không có trong các bảng ở FROM! Cột này nằm ở bảng: {', '.join(candidate_tables)}. "
                f"Hãy JOIN bảng đó qua khoá chung."
            )
        return f"Cột '{col}' không tồn tại trong các bảng ở FROM. Hãy tra cứu lại tên cột chính xác trong SCHEMA."

    # 5. Kết quả là NULL
    if "Kết quả là NULL" in error_msg:
        return (
            "Kết quả trả về là NULL. Hãy kiểm tra lại: "
            "1) Điều kiện lọc chuỗi tên sản phẩm/khách hàng: hãy dùng LOWER(cot) LIKE '%từ_khóa%' hoặc ILIKE. "
            "2) Cột tính toán có giá trị NULL: dùng COALESCE(..., 0). "
            "3) Tỉ lệ huỷ đơn: nếu tính % hãy đảm bảo dùng COUNT(DISTINCT ma_don) cho cả tử số và mẫu số."
        )

    # 6. Kết quả rỗng (0 dòng)
    if "0 dòng" in error_msg or "rỗng" in error_msg:
        return (
            "Kết quả bị rỗng (0 dòng). Hãy kiểm tra lại điều kiện WHERE: "
            "1) Lọc ngày: kiểm tra lại ngày có đúng khoảng không. "
            "2) Lọc tên: dùng LIKE '%từ_khóa%' thay vì dấu bằng tuyệt đối '='. "
            "3) Lọc trạng thái: kiểm tra các giá trị trạng thái thực tế có trong SCHEMA."
        )

    return error_msg


def answer_question_sql(
    question_id: str,
    question: str,
    as_of_date: str,
    db: BusinessDB,
    readme: str,
    rulebook_text: str,
    config: Dict[str, Any],
    llm: LLMWrapper,
    hints: str = "",
    calendar_df: Optional[pd.DataFrame] = None,
    trace: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Generate -> execute -> self-debug -> format. Every attempt is appended to trace['attempts']."""
    agent_cfg = config.get("agent", {})
    max_retries = int(agent_cfg.get("max_retries", 2))
    timeout = float(agent_cfg.get("sandbox_timeout", 30))
    on_unanswerable = agent_cfg.get("on_llm_unanswerable", "unanswerable")
    trace = trace if trace is not None else {}
    attempts: List[Dict[str, Any]] = trace.setdefault("attempts", [])

    # Automatic time resolution hint
    if not hints and as_of_date:
        tr = resolve_time(question, as_of_date, calendar_df=calendar_df)
        if tr:
            hints = (
                f"[GỢI Ý THỜI GIAN]: Câu hỏi đề cập khoảng thời gian từ '{tr.start_date}' đến '{tr.end_date}' (gồm cả 2 ngày). "
                f"Hãy lọc ngày theo điều kiện: CAST(<cot_thoi_gian> AS DATE) BETWEEN DATE '{tr.start_date}' AND DATE '{tr.end_date}'."
            )

    prev_type = prev_sql = feedback = None
    for attempt in range(max_retries + 1):
        prompt = build_sql_prompt(
            question, as_of_date, rulebook_text, readme, db.schema_text, hints,
            previous_type=prev_type, previous_sql=prev_sql, feedback=feedback,
        )
        record: Dict[str, Any] = {"attempt": attempt, "prompt_chars": len(prompt)}
        attempts.append(record)

        t0 = time.time()
        responses = llm.generate(prompt)
        record["llm_seconds"] = round(time.time() - t0, 2)
        response = responses[0] if responses else ""
        record["llm_response"] = response[:MAX_LOGGED_RESPONSE_CHARS]

        answer_type, sql = parse_llm_response(response)
        record["parsed_type"], record["sql"] = answer_type, sql

        if answer_type == "unanswerable":
            record["outcome"] = "llm_unanswerable"
            if on_unanswerable == "abstain":
                return {"id": question_id, "type": "abstain"}
            return format_submission_item(question_id, "UNANSWERABLE", "unanswerable")

        if answer_type is None:
            answer_type = infer_expected_type(question)
            record["type_source"] = "heuristic"

        if not sql:
            feedback = "Không tìm thấy câu SQL trong khối ```sql```."
            prev_type, prev_sql = answer_type, None
            record["error"] = feedback
            continue

        t0 = time.time()
        df, err = db.execute(sql, timeout=timeout)
        record["sql_seconds"] = round(time.time() - t0, 3)
        record["result"] = _preview(df)
        if err is None:
            raw, err = sql_result_to_raw(df, answer_type)
            if err is None:
                item = format_submission_item(question_id, raw, answer_type)
                if item.get("type") == "abstain":
                    err = f"Không định dạng được kết quả thành TYPE {answer_type}: {str(raw)[:200]}"
                else:
                    record["outcome"] = "answered"
                    return item
            if df is not None:
                err += f" Kết quả nhận được: {_preview(df, 3)}"

        record["error"] = err
        actionable_feedback = diagnose_duckdb_error(err, db, sql)
        feedback, prev_type, prev_sql = actionable_feedback, answer_type, sql

    trace["outcome"] = "abstain_after_retries"
    return {"id": question_id, "type": "abstain"}
