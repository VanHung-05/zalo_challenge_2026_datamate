import json

import pandas as pd

from src.datamate.db import BusinessDB
from src.datamate.loader import BusinessData
from src.datamate.runner import run_pipeline
from src.datamate.sql_agent import answer_question_sql, parse_llm_response, sql_result_to_raw


class FakeLLM:
    """Returns the scripted responses in order and records prompts."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.prompts = []

    def generate(self, prompt, **kwargs):
        self.prompts.append(prompt)
        return [self.responses.pop(0) if self.responses else ""]


CONFIG = {"agent": {"mode": "sql", "max_retries": 2, "sandbox_timeout": 10}}


def make_db():
    sales = pd.DataFrame({
        "Mã đơn": ["O1", "O2", "O3"],
        "Ngày": ["2026-03-02", "2026-03-03", "2026-03-03"],
        "Tên hàng": ["Trà đào", "Bạc xỉu", "Trà đào"],
        "Thành tiền": [30000, 25000, 30000],
    })
    return BusinessDB(BusinessData(business_id="b", data_dir=".", tables={"BanHang": sales}))


def test_parse_llm_response():
    t, sql = parse_llm_response("TYPE: number\n```sql\nSELECT 1;\n```")
    assert t == "number" and sql == "SELECT 1"
    t, sql = parse_llm_response("TYPE: list\n```sql\nSELECT a FROM t")  # unclosed fence (stop token)
    assert t == "list" and sql == "SELECT a FROM t"
    assert parse_llm_response("TYPE: unanswerable")[0] == "unanswerable"


def test_sql_result_to_raw_shapes():
    assert sql_result_to_raw(pd.DataFrame({"x": [5]}), "number") == (5, None)
    assert sql_result_to_raw(pd.DataFrame({"x": [1, 2]}), "number")[1]
    assert sql_result_to_raw(pd.DataFrame({"x": [None]}), "number")[1]
    assert sql_result_to_raw(pd.DataFrame({"n": ["A", "B"]}), "list") == (["A", "B"], None)
    assert sql_result_to_raw(pd.DataFrame({"a": [3], "b": [10]}), "list") == ([3, 10], None)


def test_self_debug_then_answer():
    db = make_db()
    llm = FakeLLM([
        "TYPE: number\n```sql\nSELECT SUM(tien) FROM ban_hang\n```",  # wrong column -> binder error
        "TYPE: number\n```sql\nSELECT SUM(thanh_tien) FROM ban_hang WHERE ngay = DATE '2026-03-03'\n```",
    ])
    trace = {}
    item = answer_question_sql("q1", "hôm nay bán được bao nhiêu tiền", "2026-03-03", db, "", "RULES", CONFIG, llm, trace=trace)
    assert item == {"id": "q1", "type": "number", "answer": 55000}
    assert len(trace["attempts"]) == 2 and "tien" in trace["attempts"][0]["error"]
    assert "LẦN TRƯỚC BỊ LỖI" in llm.prompts[1] and "RULES" in llm.prompts[0]


def test_table_and_list_results_are_not_abstained():
    db = make_db()
    llm = FakeLLM(["TYPE: table\n```sql\nSELECT ten_hang, SUM(thanh_tien) AS doanh_thu FROM ban_hang GROUP BY 1 ORDER BY 1\n```"])
    item = answer_question_sql("q2", "doanh thu theo mặt hàng", "2026-03-03", db, "", "", CONFIG, llm)
    assert item["type"] == "table"
    assert item["answer"]["rows"] == [["Bạc xỉu", 25000], ["Trà đào", 60000]]


def test_gives_up_with_abstain():
    db = make_db()
    llm = FakeLLM(["TYPE: number\n```sql\nSELECT nope FROM ban_hang\n```"] * 3)
    item = answer_question_sql("q3", "?", "2026-03-03", db, "", "", CONFIG, llm)
    assert item == {"id": "q3", "type": "abstain"}
    assert len(llm.prompts) == 3


def test_run_pipeline_end_to_end(tmp_path):
    split = tmp_path / "split"
    biz = split / "data" / "biz_1"
    biz.mkdir(parents=True)
    (biz / "README.md").write_text("Quán trà", encoding="utf-8")
    pd.DataFrame({"ma_don": ["O1", "O2"], "ngay_ban": ["2026-03-02", "2026-03-03"], "tong_tien": [1000, 2000]}).to_csv(
        biz / "ban_hang.csv", index=False)
    (split / "rulebook.md").write_text("# Rulebook test\nDoanh thu = tổng tiền", encoding="utf-8")
    questions = [
        {"id": "q_a", "business_id": "biz_1", "question": "tổng doanh thu", "as_of_date": "2026-03-03"},
        {"id": "q_b", "business_id": "biz_missing", "question": "x", "as_of_date": "2026-03-03"},
    ]
    (split / "questions.jsonl").write_text("\n".join(json.dumps(q, ensure_ascii=False) for q in questions), encoding="utf-8")
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text("agent:\n  mode: sql\n  max_retries: 1\n", encoding="utf-8")

    llm = FakeLLM(["TYPE: number\n```sql\nSELECT SUM(tong_tien) FROM ban_hang\n```"])
    out = tmp_path / "out" / "submission.jsonl"
    run_pipeline(str(cfg), str(split), str(out), llm=llm)

    rows = [json.loads(l) for l in out.read_text(encoding="utf-8").splitlines()]
    assert {"id": "q_a", "type": "number", "answer": 3000} in rows
    assert {"id": "q_b", "type": "abstain"} in rows
    assert "Rulebook test" in llm.prompts[0]
    log = json.loads((tmp_path / "out" / "logs" / "q_a.json").read_text(encoding="utf-8"))
    assert log["final"]["answer"] == 3000 and log["attempts"][0]["sql"] and "elapsed_seconds" in log
