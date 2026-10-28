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

Quy tắc:
1. Đúng một câu SELECT (được dùng WITH), cú pháp DuckDB, dùng đúng tên bảng/cột trong SCHEMA.
2. Hình dạng kết quả theo TYPE:
   - number, percent, string, date, boolean: đúng 1 dòng, 1 cột.
   - list: 1 cột, mỗi phần tử một dòng, ORDER BY đúng thứ tự câu hỏi cần (mặc định: số đo giảm dần, bằng nhau thì mã tăng dần). Nếu đáp án là vài số theo thứ tự (vd [số thùng, số lẻ]) thì trả 1 dòng nhiều cột theo đúng thứ tự đó.
   - table: cột đầu là tên đối tượng, các cột sau là số đo theo đúng thứ tự câu hỏi nêu.
3. Tiền là số nguyên VND: ROUND(x) (kể cả trung bình tiền). percent là số phần trăm (12.34, không phải 0.1234): ROUND(x, 2). Số trung bình khác: ROUND(x, 2).
4. date: trả chuỗi strftime(x, '%Y-%m-%d'); câu hỏi theo tháng: strftime(x, '%Y-%m'). boolean: trả TRUE/FALSE.
5. string và phần tử list là tên: trả nguyên giá trị trong dữ liệu (đủ dấu), không kèm số đo.
6. Cột TIMESTAMP có giờ: lọc theo ngày dùng CAST(cot AS DATE); lấy giờ dùng hour(cot). Ngày viết dạng DATE 'YYYY-MM-DD'.
7. Bản ghi có ngày sau hôm nay coi như chưa xảy ra (trừ trạng thái huỷ/trả của đơn, xem rulebook).
8. Lọc trạng thái/nhóm bằng đúng các giá trị liệt kê trong SCHEMA.
9. Nếu dữ liệu không có thông tin cần để trả lời (thiếu cột, thiếu loại dữ liệu, kỳ hỏi nằm hoàn toàn ngoài khoảng ngày có dữ liệu) thì trả lời TYPE: unanswerable và không viết SQL.
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
    trace: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Generate -> execute -> self-debug -> format. Every attempt is appended to trace['attempts']."""
    agent_cfg = config.get("agent", {})
    max_retries = int(agent_cfg.get("max_retries", 2))
    timeout = float(agent_cfg.get("sandbox_timeout", 30))
    on_unanswerable = agent_cfg.get("on_llm_unanswerable", "unanswerable")
    trace = trace if trace is not None else {}
    attempts: List[Dict[str, Any]] = trace.setdefault("attempts", [])

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
        feedback, prev_type, prev_sql = err, answer_type, sql

    trace["outcome"] = "abstain_after_retries"
    return {"id": question_id, "type": "abstain"}
