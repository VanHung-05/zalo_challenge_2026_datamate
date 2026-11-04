"""Code agent (plan §4.5, "hướng 2"): the LLM writes pandas code over the business's DataFrames.

Per independent sample:
  1. the LLM sees the catalog of loaded tables (original column names, types, example values), the
     Business Card, the rulebook-applied frames sl_lines / sl_orders and pre-computed hints;
  2. it may run up to `max_explore` exploration programs (STEP: explore) whose printed output is sent back;
  3. it writes the final program, which assigns `result` (TYPE: ...);
  4. errors, empty / NaN / wrongly shaped results are sent back with the traceback, up to max_retries times.
Programs run in pyexec.PyExecutor (separate process, real timeout, fresh copies of the data).
"""

import re
import time
from typing import Any, Dict, List, Optional, Tuple

from src.datamate.db import BusinessDB
from src.datamate.formatter import format_submission_item
from src.datamate.pyexec import PyExecutor

VALID_TYPES = ["number", "percent", "string", "date", "boolean", "list", "table", "unanswerable"]
SCALAR_TYPES = {"number", "percent", "string", "date", "boolean"}
MAX_README_CHARS = 6000
MAX_FEEDBACK_CHARS = 3500
MAX_LOGGED = 4000

INSTRUCTIONS = """Bạn là chuyên gia phân tích dữ liệu cho cửa hàng nhỏ ở Việt Nam. Nhiệm vụ: viết code Python (pandas) trên dữ liệu ĐÃ NẠP SẴN của cửa hàng để trả lời câu hỏi. Mọi con số phải do code tính ra từ dữ liệu, không tự nhẩm, không đoán.

## RULEBOOK (luật mặc định; câu hỏi không nói khác thì BẮT BUỘC theo luật này)
{rulebook}
"""

DATA_SECTION = """
## DỮ LIỆU CỦA CỬA HÀNG
### README
{readme}

### BUSINESS CARD (hệ thống tự phân tích; dùng để định hướng, kiểm tra lại bằng code nếu cần)
{card}

### BIẾN CÓ SẴN TRONG CODE
- `tables`: dict tên bảng -> DataFrame, đã làm sạch: gộp các file/sheet tách theo tháng, ngày dạng chuỗi 'YYYY-MM-DD' hoặc 'YYYY-MM-DD HH:MM:SS', số đã chuyển sang số, ô trống của các dòng sau trong cùng một đơn đã được điền theo dòng đầu đơn.
{semantic}
- `pd`, `np`, `datetime`, `date`, `timedelta`, `re`, `math`. Không đọc/ghi file, không import thư viện khác.

### CÁC BẢNG TRONG `tables`
{catalog}
"""

SEMANTIC_DESC = """- `sl_lines` (DataFrame, mỗi dòng một dòng hàng, ĐÃ áp rulebook: gộp dòng trùng của cùng đơn, ô trống kế thừa dòng đầu đơn): order_id, order_ts (Timestamp), order_date (date), product_code, product_name, product_group, unit, qty, qty_base (số lượng quy về đơn vị cơ bản), unit_price, line_amount (thành tiền dòng), cost_per_unit, state, customer_code, customer_name, customer_group, payment_method, sales_channel, staff.
- `sl_orders` (DataFrame, mỗi dòng một đơn): order_id, order_ts, order_date, state, customer_code, customer_name, customer_group, payment_method, sales_channel, staff, amount (tổng thành tiền các dòng), n_lines.
- `state` ∈ 'completed', 'delivering', 'cancelled', 'returned' (đơn có trả hàng, kể cả trả một phần), 'unknown'. Doanh thu = tổng line_amount của các đơn state ∈ {'completed', 'delivering'}.
- Cột nào của sl_lines toàn None nghĩa là dữ liệu không có thông tin đó (hoặc hệ thống chưa nhận ra): khi đó dùng trực tiếp `tables`."""

QUESTION_SECTION = """
## CÂU HỎI
Hôm nay (as_of_date) là {as_of_date} ({weekday}). Bản ghi có ngày sau hôm nay coi như chưa xảy ra.
Câu hỏi: "{question}"
{hints}
## CÁCH TRẢ LỜI (chỉ đúng định dạng, không giải thích dài)
Bước khám phá (tuỳ chọn, tối đa {max_explore} lần) — khi chưa chắc tên bảng/cột/giá trị (tên hàng, trạng thái, nhóm...):
STEP: explore
```python
# chỉ print() những gì cần xem, ngắn gọn (vd: giá trị khác nhau của cột, vài dòng đã lọc)
```
Bước trả lời:
TYPE: <một trong: number, percent, string, date, boolean, list, table, unanswerable>
```python
# ... tính toán ...
result = ...
```
Quy tắc:
1. Xác định bảng và cột cần dùng từ danh mục ở trên; tên cột phải đúng y như danh mục (có dấu, có khoảng trắng).
2. Ưu tiên `sl_lines` / `sl_orders` cho doanh thu, số đơn, số lượng, xếp hạng (luật đơn huỷ/trả đã áp). Cần thông tin không có trong đó (nhập hàng, tồn kho, công nợ, giá vốn, lô...) thì dùng `tables`.
3. Lọc ngày bằng khoảng trong [GỢI Ý] nếu có, gồm cả hai đầu. Cột ngày trong `tables` là chuỗi: dùng pd.to_datetime(...) rồi so sánh.
4. Tên trong dữ liệu có thể viết khác nhau (hoa thường, dấu, khoảng trắng): so khớp mềm (str.lower, str.contains) và kiểm tra bằng bước explore.
5. `result`: number/percent: một số (percent là số phần trăm, 12.34 chứ không phải 0.1234); string: tên đúng như trong dữ liệu; date: 'YYYY-MM-DD' hoặc 'YYYY-MM'; boolean: True/False; list: list Python theo đúng thứ tự cần trả (top k: giảm dần, bằng nhau thì theo mã tăng dần); table: DataFrame, cột đầu là tên đối tượng, các cột sau là số đo theo thứ tự câu hỏi. Không tự làm tròn, hệ thống sẽ làm tròn.
6. Nếu dữ liệu không có thông tin cần thiết thì trả lời `TYPE: unanswerable` (không cần code).
"""

HISTORY_EXPLORE = """
## KẾT QUẢ KHÁM PHÁ LẦN {i}
```python
{code}
```
Output:
```
{output}
```
"""

HISTORY_ERROR = """
## LẦN TRƯỚC BỊ LỖI, HÃY SỬA (TYPE: {type})
```python
{code}
```
Vấn đề: {feedback}
Viết lại đúng định dạng (TYPE + khối ```python```), gán `result`.
"""

WEEKDAYS_VI = ["Thứ Hai", "Thứ Ba", "Thứ Tư", "Thứ Năm", "Thứ Sáu", "Thứ Bảy", "Chủ nhật"]


def _short(v: Any, n: int = 40) -> str:
    s = str(v)
    return s if len(s) <= n else s[: n - 3] + "..."


def build_catalog(db: BusinessDB) -> Tuple[str, Dict[str, str]]:
    """Catalog of `tables` with ORIGINAL table / column names (the pandas keys), from the DuckDB profile.
    Returns (text, {table original name: sql name})."""
    lines: List[str] = []
    names: Dict[str, str] = {}
    for t in db.tables.values():
        if t.hidden or t.original == "calendar.csv":
            continue
        names[t.original] = t.sql_name
        lines.append(f"tables[{t.original!r}]  -- {t.n_rows:,} dòng".replace(",", "."))
        for c in t.columns:
            if not c.original:  # columns derived by db.py (x_so / x_don_vi) do not exist in pandas
                continue
            if t.n_rows and c.n_null == t.n_rows:
                continue
            desc = [c.sql_type.lower()]
            if c.distinct_values:
                desc.append("giá trị: " + ", ".join(repr(_short(v)) for v in c.distinct_values))
            elif c.samples:
                desc.append(f"{c.n_distinct} giá trị, vd: " + ", ".join(repr(_short(v)) for v in c.samples))
            elif c.min_val is not None:
                desc.append(f"từ {_short(c.min_val)} đến {_short(c.max_val)}")
            if c.n_null:
                desc.append(f"{c.n_null} trống")
            if c.note:
                desc.append("dạng 'số + đơn vị' (vd '2 hộp')")
            lines.append(f"  {c.original!r}: " + "; ".join(desc))
        lines.append("")
    if db.calendar_table:
        lines.append("Lịch dương–âm: biến `calendar` (DataFrame: date, weekday 1=Thứ Hai..7=Chủ nhật, lunar_day, lunar_month, lunar_year, lunar_leap, holiday)")
    return "\n".join(lines).strip(), names


def build_prompt(question: str, as_of: str, rulebook: str, readme: str, card_text: str, catalog: str, has_semantic: bool,
                 hints: str, history: str, max_explore: int) -> str:
    """Static parts first (rulebook, business data) so the llama.cpp / vLLM prefix cache is reused."""
    try:
        from datetime import datetime
        weekday = WEEKDAYS_VI[datetime.strptime(as_of[:10], "%Y-%m-%d").weekday()]
    except ValueError:
        weekday = "không rõ thứ"
    readme = (readme or "").strip() or "(không có README)"
    if len(readme) > MAX_README_CHARS:
        readme = readme[:MAX_README_CHARS] + "\n..."
    p = INSTRUCTIONS.format(rulebook=(rulebook or "(không có rulebook)").strip())
    p += DATA_SECTION.format(readme=readme, card=card_text or "(không có)", catalog=catalog,
                             semantic=SEMANTIC_DESC if has_semantic else "- (không có sl_lines / sl_orders cho cửa hàng này)")
    p += QUESTION_SECTION.format(as_of_date=as_of, weekday=weekday, question=question, hints=f"{hints}\n" if hints else "",
                                 max_explore=max_explore)
    return p + history


def parse_response(text: str) -> Tuple[Optional[str], Optional[str]]:
    """(kind, code): kind is 'explore', an answer type, or None."""
    if not text:
        return None, None
    kind = None
    if re.search(r"STEP\s*[:=]\s*explore", text, re.I):
        kind = "explore"
    m = re.search(r"TYPE\s*[:=]\s*`?([A-Za-z_]+)", text, re.I)
    if m and m.group(1).lower() in VALID_TYPES and kind is None:
        kind = m.group(1).lower()
    code = None
    blocks = re.findall(r"```(?:python|py)?\s*\n(.*?)(?:```|$)", text, re.S | re.I)
    if blocks:
        code = blocks[-1].strip() or None
    return kind, code


def _is_missing(v: Any) -> bool:
    return v is None or (isinstance(v, float) and v != v)


def result_to_raw(res: Any, answer_type: str) -> Tuple[Any, Optional[str]]:
    """Plain result (pyexec.to_plain) -> value for format_submission_item, or feedback for the LLM."""
    if isinstance(res, dict) and res.get("__table__"):
        cols, rows = res["columns"], res["rows"]
        if not rows:
            return None, "result là bảng rỗng (0 dòng): kiểm tra lại điều kiện lọc (ngày, trạng thái, tên)."
        if answer_type == "table":
            return {"columns": cols, "rows": rows}, None
        if answer_type == "list":
            return ([r[0] for r in rows] if len(cols) == 1 or len(rows) > 1 else rows[0]), None
        if len(rows) == 1 and len(cols) == 1:
            res = rows[0][0]
        else:
            return None, f"TYPE {answer_type} cần một giá trị, nhưng result là bảng {res['n_rows']} dòng × {len(cols)} cột."
    if answer_type in SCALAR_TYPES:
        if isinstance(res, list):
            if len(res) != 1:
                return None, f"TYPE {answer_type} cần một giá trị, nhưng result là list {len(res)} phần tử."
            res = res[0]
        if _is_missing(res):
            return None, "result là None/NaN: chưa gán result, hoặc phép lọc không ra dòng nào."
        return res, None
    if answer_type == "list":
        if not isinstance(res, list):
            res = [res]
        if not res:
            return None, "result là list rỗng: kiểm tra lại điều kiện lọc."
        if any(_is_missing(x) for x in res):
            return None, "list có phần tử None/NaN."
        return res, None
    if answer_type == "table":
        return None, "TYPE table cần result là DataFrame."
    return None, f"TYPE không hợp lệ: {answer_type}"


def run_code_sample(question: str, as_of: str, executor: PyExecutor, static_prompt_parts: Dict[str, Any], hints: str,
                    llm: Any, max_retries: int, max_explore: int, timeout: float, temperature: float,
                    seed: Optional[int], default_type: str) -> Dict[str, Any]:
    """One independent solution. Returns {'outcome': answered|unanswerable|failed, 'raw', 'answer_type', 'code', 'rounds'}."""
    rounds: List[Dict[str, Any]] = []
    explore_hist = ""
    error_note = ""
    n_explore = n_answer = 0
    while n_answer <= max_retries:
        prompt = build_prompt(question, as_of, hints=hints, history=explore_hist + error_note, max_explore=max_explore,
                              **static_prompt_parts)
        rec: Dict[str, Any] = {"round": len(rounds), "prompt_chars": len(prompt)}
        rounds.append(rec)
        t0 = time.time()
        out = llm.generate(prompt, temperature=temperature, seed=seed)
        rec["llm_seconds"] = round(time.time() - t0, 2)
        text = out[0] if out else ""
        rec["llm_response"] = text[:MAX_LOGGED]
        kind, code = parse_response(text)
        rec["kind"], rec["code"] = kind, code
        if kind == "unanswerable":
            return {"outcome": "unanswerable", "rounds": rounds}
        if kind == "explore":
            if code and n_explore < max_explore:
                n_explore += 1
                r = executor.run(code, timeout=timeout)
                output = r["stdout"] if r["ok"] else f"{r['stdout']}\nLỖI: {r['error']}\n{r.get('traceback', '')}"
                rec["explore_output"] = output[:MAX_LOGGED]
                explore_hist += HISTORY_EXPLORE.format(i=n_explore, code=code, output=output[:MAX_FEEDBACK_CHARS] or "(không in gì)")
                continue
            # explore budget spent: ask for the answer
            error_note = HISTORY_ERROR.format(type="explore", code=code or "", feedback="Đã hết lượt khám phá: hãy trả lời (TYPE + code gán result).")
            n_answer += 1
            continue
        n_answer += 1
        answer_type = kind or default_type
        if not code:
            error_note = HISTORY_ERROR.format(type=answer_type, code="", feedback="Không tìm thấy khối ```python```.")
            continue
        t0 = time.time()
        r = executor.run(code, timeout=timeout)
        rec["exec_seconds"] = round(time.time() - t0, 3)
        rec["stdout"] = r.get("stdout", "")[:1000]
        if not r["ok"]:
            rec["error"] = r["error"]
            if r.get("timeout"):
                rec["timeout"] = True
            fb = f"{r['error']}\n{r.get('traceback', '')}"
        else:
            raw, fb = result_to_raw(r["result"], answer_type)
            rec["result"] = str(r["result"])[:500]
            if fb is None:
                return {"outcome": "answered", "raw": raw, "answer_type": answer_type, "code": code, "rounds": rounds,
                        "n_explore": n_explore, "n_answer_rounds": n_answer}
            rec["error"] = fb
        error_note = HISTORY_ERROR.format(type=answer_type, code=code, feedback=fb[:MAX_FEEDBACK_CHARS])
    return {"outcome": "failed", "rounds": rounds}


def generate_code_candidates(question_id: str, a: Any, executor: PyExecutor, static_prompt_parts: Dict[str, Any],
                             config: Dict[str, Any], llm: Any, entity_hits: Optional[List[Any]] = None,
                             is_money: bool = False, known_names: Optional[set] = None,
                             trace: Optional[Dict[str, Any]] = None) -> List[Any]:
    """N independent samples, like sql_agent.generate_sql_candidates (sample 0 greedy, then seed base+i)."""
    from src.datamate.decision import Candidate, sanity_checks
    from src.datamate.sql_agent import build_hints

    acfg = config.get("agent", {})
    n = max(1, int(acfg.get("n_samples", 1)))
    max_retries = int(acfg.get("max_retries", 3))
    max_explore = int(acfg.get("max_explore", 2))
    timeout = float(acfg.get("sandbox_timeout", 30))
    temp = float(acfg.get("sample_temperature", 0.7))
    base_seed = int(config.get("llm", {}).get("seed", 42))
    hints = build_hints(a, entity_hits).replace("của sl_lines", "của sl_lines (pandas)")
    trace = trace if trace is not None else {}
    samples = trace.setdefault("llm_samples", [])
    cands: List[Any] = []
    for i in range(n):
        rec = run_code_sample(a.raw, a.as_of.isoformat(), executor, static_prompt_parts, hints, llm, max_retries, max_explore,
                              timeout, temperature=0.0 if i == 0 else temp, seed=base_seed + i, default_type=a.answer_type)
        samples.append(rec)
        flags = ["sandbox_timeout"] if any(r.get("timeout") for r in rec["rounds"]) else []
        meta = {"sample": i, "code": rec.get("code"), "n_explore": rec.get("n_explore", 0), "n_answer_rounds": rec.get("n_answer_rounds", 0)}
        if rec["outcome"] == "unanswerable":
            cands.append(Candidate({"type": "unanswerable"}, "llm", True, {}, flags, meta))
        elif rec["outcome"] == "answered":
            item = format_submission_item(question_id, rec["raw"], rec["answer_type"], is_money=is_money,
                                          date_granularity=a.date_granularity)
            if item.get("type") == "abstain":
                cands.append(Candidate(item, "llm", False, {}, flags + ["format_failed"], meta))
                continue
            checks = sanity_checks(item, a.answer_type, a.top_k, known_names,
                                   is_count=a.answer_type == "number" and not is_money and "quantity" not in a.concepts)
            # A revenue / amount of exactly 0 is far more often a wrong filter than a real zero
            if item.get("type") in ("number", "percent") and item.get("answer") == 0:
                checks["non_zero"] = False
            cands.append(Candidate(item, "llm", True, checks, flags, meta))
        else:
            cands.append(Candidate({"type": "abstain"}, "llm", False, {}, flags, meta))
    return cands
