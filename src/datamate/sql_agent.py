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


VALID_TYPES = ["number", "percent", "string", "date", "boolean", "list", "table", "unanswerable", "explore"]
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
TYPE: <một trong: number, percent, string, date, boolean, list, table, unanswerable, explore>
```sql
SELECT ...
```

Quy tắc (BẮT BUỘC):
1. Đúng một câu SELECT (được dùng WITH), cú pháp DuckDB, chỉ dùng đúng tên bảng/cột/view có trong SCHEMA.
2. Nghiệp vụ theo RULEBOOK ở trên. Nếu SCHEMA có view sl_orders / sl_lines thì ƯU TIÊN dùng chúng: các view này đã áp
   luật đơn huỷ, đơn trả hàng (kể cả trả một phần), ô trống kế thừa dòng đầu đơn, dòng trùng của cùng đơn.
3. Khoảng thời gian: dùng ĐÚNG các ngày trong [GỢI Ý] nếu có (đã tính sẵn theo rulebook), không tự tính lại.
4. Tên trong dữ liệu có thể viết khác nhau (hoa thường, dấu, khoảng trắng): lọc bằng `ILIKE '%...%'` hoặc theo mã.
5. HÌNH DẠNG KẾT QUẢ: number, percent, string, date, boolean: đúng 1 dòng, 1 cột. list: 1 cột theo đúng thứ tự cần trả.
   table: cột đầu là tên đối tượng, các cột sau là số đo theo thứ tự câu hỏi. Không tự làm tròn: hệ thống làm tròn.
   percent là số phần trăm (12.34, không phải 0.1234). date: strftime(x, '%Y-%m-%d') hoặc strftime(x, '%Y-%m').
   string / list tên: lấy nguyên giá trị trong dữ liệu.
6. Muốn xem dữ liệu trước (giá trị khác nhau của một cột, kiểm tra tên) thì trả lời `TYPE: explore` kèm một SELECT
   (LIMIT ≤ 30); hệ thống gửi lại kết quả để bạn viết câu trả lời thật.
7. Nếu dữ liệu không có thông tin cần để trả lời thì trả lời `TYPE: unanswerable` và không viết SQL.
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


# ── Program synthesis with self-consistency (plan §4.5) ───────────────────────

EXPLORE_SECTION = """
## KẾT QUẢ KHÁM PHÁ (lần trước bạn chọn TYPE: explore)
```sql
{sql}
```
Kết quả: {result}
Giờ hãy viết câu trả lời thật (TYPE khác explore).
"""


def build_hints(a: Any, entity_hits: Optional[List[Any]] = None) -> str:
    """Pre-computed facts for the prompt. Times are only given when the resolver is confident:
    a wrong hint does more harm than none (progress report 28/10)."""
    lines: List[str] = []
    t = a.time
    if t.confident:
        if t.primary:
            lines.append(f"- Kỳ hỏi: từ {t.primary.start} đến {t.primary.end} (gồm cả 2 ngày){f' — {t.primary.label}' if t.primary.label else ''}")
        if t.compare:
            lines.append(f"- Kỳ so sánh: từ {t.compare.start} đến {t.compare.end}; phần trăm thay đổi = (kỳ hỏi − kỳ so) / kỳ so × 100")
        for i, p in enumerate(t.periods, 1):
            lines.append(f"- Kỳ {i}: từ {p.start} đến {p.end}")
        if t.point_in_time:
            lines.append(f"- Tình trạng tính đến hết ngày {t.point_in_time} (mọi phát sinh từ đầu dữ liệu đến hết ngày đó)")
        if t.shift:
            lines.append(f"- Chỉ lấy đơn ca {'sáng' if t.shift == 'morning' else 'chiều'} (theo giờ tạo đơn)")
    lines.append(f"- Kiểu đáp án dự kiến: {a.answer_type}" + (f" ({'yyyy-mm' if a.date_granularity == 'month' else 'yyyy-mm-dd'})" if a.answer_type == "date" else ""))
    if a.top_k:
        lines.append(f"- Số phần tử cần trả: {a.top_k} (ít hơn thì trả hết); bằng nhau thì theo mã tăng dần")
    for th in a.thresholds:
        lines.append(f"- Ngưỡng: {th['op']} {th['value']} ({th.get('text', '')})")
    for e in entity_hits or []:
        lines.append(f"- '{e.phrase}' trong câu hỏi là {e.kind} '{e.value}' (lọc theo cột {e.column} của sl_lines)")
    return "[GỢI Ý]\n" + "\n".join(lines) if lines else ""


def run_sql_sample(
    question: str, as_of_date: str, db: BusinessDB, readme: str, rulebook_text: str, schema_text: str, hints: str,
    llm: Any, max_rounds: int, timeout: float, temperature: float, seed: Optional[int], default_type: str,
    max_explore: int = 2,
) -> Dict[str, Any]:
    """One independent solution: generate -> (explore) -> execute -> repair. Returns a record with
    'outcome' in answered | unanswerable | failed, plus 'raw', 'answer_type', 'sql', 'rounds'."""
    rounds: List[Dict[str, Any]] = []
    prev_type = prev_sql = feedback = None
    explore_note = ""
    n_explore = n_answer_rounds = 0
    for r in range(max_rounds + max_explore):
        # explore rounds have their own budget: a failing answer may only use max_rounds attempts
        if n_answer_rounds >= max_rounds:
            break
        prompt = build_sql_prompt(question, as_of_date, rulebook_text, readme, schema_text, hints,
                                  previous_type=prev_type, previous_sql=prev_sql, feedback=feedback) + explore_note
        rec: Dict[str, Any] = {"round": r, "prompt_chars": len(prompt)}
        rounds.append(rec)
        t0 = time.time()
        out = llm.generate(prompt, temperature=temperature, seed=seed) if temperature or seed is not None else llm.generate(prompt)
        rec["llm_seconds"] = round(time.time() - t0, 2)
        response = out[0] if out else ""
        rec["llm_response"] = response[:MAX_LOGGED_RESPONSE_CHARS]
        answer_type, sql = parse_llm_response(response)
        rec["parsed_type"], rec["sql"] = answer_type, sql
        if answer_type == "unanswerable":
            return {"outcome": "unanswerable", "rounds": rounds}
        if answer_type == "explore" and sql and n_explore < max_explore:
            n_explore += 1
            df, err = db.execute(sql, timeout=timeout, max_rows=30)
            explore_note = EXPLORE_SECTION.format(sql=sql, result=err or _preview(df, 30))
            rec["explore"] = err or _preview(df, 5)
            continue
        n_answer_rounds += 1
        if answer_type in (None, "explore"):
            answer_type = default_type
            rec["type_source"] = "question_analysis"
        if not sql:
            feedback, prev_type, prev_sql = "Không tìm thấy câu SQL trong khối ```sql```.", answer_type, None
            rec["error"] = feedback
            continue
        t0 = time.time()
        df, err = db.execute(sql, timeout=timeout)
        rec["sql_seconds"] = round(time.time() - t0, 3)
        rec["result"] = _preview(df)
        if err is None:
            raw, err = sql_result_to_raw(df, answer_type)
            if err is None:
                return {"outcome": "answered", "raw": raw, "answer_type": answer_type, "sql": sql, "rounds": rounds}
            err += f" Kết quả nhận được: {_preview(df, 3)}"
        rec["error"] = err
        if "quá" in str(err) and "giây" in str(err):
            rec["timeout"] = True
        feedback, prev_type, prev_sql = diagnose_duckdb_error(err, db, sql), answer_type, sql
    return {"outcome": "failed", "rounds": rounds}


def generate_sql_candidates(
    question_id: str, a: Any, db: BusinessDB, readme: str, rulebook_text: str, extra_schema: str,
    config: Dict[str, Any], llm: Any, entity_hits: Optional[List[Any]] = None, is_money: bool = False,
    known_names: Optional[set] = None, trace: Optional[Dict[str, Any]] = None,
) -> List[Any]:
    """N independent samples (plan §4.5 self-consistency). Sample 0 is greedy; the others use
    temperature agent.sample_temperature with seed base+i, so a rerun reproduces them."""
    from src.datamate.decision import Candidate, sanity_checks

    agent_cfg = config.get("agent", {})
    n = max(1, int(agent_cfg.get("n_samples", 1)))
    max_rounds = int(agent_cfg.get("max_retries", 2)) + 1
    timeout = float(agent_cfg.get("sandbox_timeout", 30))
    temp = float(agent_cfg.get("sample_temperature", 0.7))
    base_seed = int(config.get("llm", {}).get("seed", 42))
    hints = build_hints(a, entity_hits)
    schema_text = (extra_schema + "\n\n" if extra_schema else "") + db.schema_text
    trace = trace if trace is not None else {}
    samples = trace.setdefault("llm_samples", [])
    cands: List[Any] = []
    for i in range(n):
        rec = run_sql_sample(a.raw, a.as_of.isoformat(), db, readme, rulebook_text, schema_text, hints, llm, max_rounds,
                             timeout, temperature=0.0 if i == 0 else temp, seed=None if i == 0 else base_seed + i,
                             default_type=a.answer_type)
        samples.append(rec)
        flags = ["sandbox_timeout"] if any(r.get("timeout") for r in rec["rounds"]) else []
        if rec["outcome"] == "unanswerable":
            cands.append(Candidate({"type": "unanswerable"}, "llm", True, {}, flags, {"sample": i}))
        elif rec["outcome"] == "answered":
            item = format_submission_item(question_id, rec["raw"], rec["answer_type"], is_money=is_money,
                                          date_granularity=a.date_granularity)
            if item.get("type") == "abstain":
                cands.append(Candidate(item, "llm", False, {}, flags + ["format_failed"], {"sample": i, "sql": rec["sql"]}))
                continue
            checks = sanity_checks(item, a.answer_type, a.top_k, known_names,
                                   is_count=a.answer_type == "number" and not is_money and "quantity" not in a.concepts)
            cands.append(Candidate(item, "llm", True, checks, flags, {"sample": i, "sql": rec["sql"]}))
        else:
            cands.append(Candidate({"type": "abstain"}, "llm", False, {}, flags, {"sample": i}))
    return cands
