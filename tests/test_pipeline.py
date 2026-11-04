import json

import pandas as pd

from src.datamate.runner import run_pipeline


class FakeLLM:
    def __init__(self, responses):
        self.responses = list(responses)
        self.prompts = []

    def generate(self, prompt, **kwargs):
        self.prompts.append(prompt)
        return [self.responses.pop(0) if self.responses else ""]


RULEBOOK = """# Rulebook test
## 1. Đơn hàng và doanh thu
| Luật | Mặc định |
|---|---|
| Đơn tính vào doanh thu | đơn **không huỷ** và **không trả hàng**. Đơn có trả hàng, **kể cả chỉ trả một phần**, là đơn trả hàng: bỏ **cả đơn**. Đơn đang giao vẫn tính |
"""


def make_split(tmp_path, questions):
    split = tmp_path / "split"
    biz = split / "data" / "biz_1"
    biz.mkdir(parents=True)
    (biz / "README.md").write_text("Quán trà", encoding="utf-8")
    # Order O2 is written as a group: its second line leaves the order cells empty (rulebook §8)
    pd.DataFrame({
        "Mã đơn": ["O1", "O2", None, "O3", "O4"],
        "Ngày bán": ["02/03/2026", "03/03/2026", None, "03/03/2026", "03/03/2026"],
        "Trạng thái": ["Hoàn thành", "Hoàn thành", None, "Đã hủy", "Trả hàng"],
        "Tên hàng": ["Trà đào", "Bạc xỉu", "Trà đào", "Trà đào", "Bạc xỉu"],
        "Số lượng": [1, 1, 2, 1, 1],
        "Thành tiền": [30000, 25000, 60000, 30000, 25000],
    }).to_csv(biz / "ban_hang.csv", index=False)
    (split / "rulebook.md").write_text(RULEBOOK, encoding="utf-8")
    (split / "questions.jsonl").write_text("\n".join(json.dumps(q, ensure_ascii=False) for q in questions), encoding="utf-8")
    return split


def test_v2_fast_path_guard_and_logs(tmp_path):
    questions = [
        {"id": "q_rev", "business_id": "biz_1", "question": "Hôm nay cửa hàng bán được bao nhiêu tiền?", "as_of_date": "2026-03-03"},
        {"id": "q_subj", "business_id": "biz_1", "question": "Nhân viên nào được khách khen nhiều nhất?", "as_of_date": "2026-03-03"},
        {"id": "q_out", "business_id": "biz_1", "question": "Doanh thu tháng 1/2025 là bao nhiêu?", "as_of_date": "2026-03-03"},
    ]
    split = make_split(tmp_path, questions)
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text("agent:\n  mode: sql\n  pipeline: v2\n  no_llm: true\ndecision:\n  tau: 0.7\n", encoding="utf-8")
    out = tmp_path / "out" / "submission.jsonl"
    run_pipeline(str(cfg), str(split), str(out))

    rows = {r["id"]: r for r in (json.loads(l) for l in out.read_text(encoding="utf-8").splitlines())}
    # 03/03: O2 (25 000 + 60 000 on its filled-down line); O3 cancelled and O4 returned are excluded
    assert rows["q_rev"] == {"id": "q_rev", "type": "number", "answer": 85000}
    assert rows["q_subj"]["type"] == "unanswerable"
    assert rows["q_out"]["type"] == "unanswerable"  # period entirely before the data (rulebook §7)

    log = json.loads((tmp_path / "out" / "logs" / "q_rev.json").read_text(encoding="utf-8"))
    for key in ("analysis", "guard", "fast_path", "decision", "candidates", "final", "elapsed_seconds"):
        assert key in log
    assert log["fast_path"]["intent"] == "total_revenue"


def test_v2_llm_samples_must_agree(tmp_path):
    questions = [{"id": "q1", "business_id": "biz_1", "question": "Trà đào bán ra bao nhiêu ly từ trước đến nay?", "as_of_date": "2026-03-03"}]
    split = make_split(tmp_path, questions)
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text("agent:\n  mode: sql\n  pipeline: v2\n  n_samples: 3\n  max_retries: 0\ndecision:\n  tau: 0.7\n", encoding="utf-8")
    sql = "TYPE: number\n```sql\nSELECT {v}\n```"
    out = tmp_path / "out" / "submission.jsonl"
    # No majority (3 / 4 / 5) -> abstain
    run_pipeline(str(cfg), str(split), str(out), llm=FakeLLM([sql.format(v=3), sql.format(v=4), sql.format(v=5)]))
    assert json.loads(out.read_text(encoding="utf-8"))["type"] == "abstain"
    # Three agreeing samples clear tau with the default priors (0.60 + 2 x 0.08)
    run_pipeline(str(cfg), str(split), str(out), llm=FakeLLM([sql.format(v=3)] * 3))
    assert json.loads(out.read_text(encoding="utf-8")) == {"id": "q1", "type": "number", "answer": 3}
    # Two agreeing samples alone stay below tau (0.68): one more independent agreement is required
    cfg.write_text("agent:\n  mode: sql\n  pipeline: v2\n  n_samples: 2\n  max_retries: 0\ndecision:\n  tau: 0.7\n", encoding="utf-8")
    run_pipeline(str(cfg), str(split), str(out), llm=FakeLLM([sql.format(v=3)] * 2))
    assert json.loads(out.read_text(encoding="utf-8"))["type"] == "abstain"


def test_fast_path_disabled_uses_llm_only(tmp_path):
    questions = [{"id": "q_rev", "business_id": "biz_1", "question": "Hôm nay cửa hàng bán được bao nhiêu tiền?", "as_of_date": "2026-03-03"}]
    split = make_split(tmp_path, questions)
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text("agent:\n  mode: sql\n  fast_path: false\n  n_samples: 3\n  max_retries: 0\ndecision:\n  tau: 0.7\n", encoding="utf-8")
    out = tmp_path / "out" / "submission.jsonl"
    llm = FakeLLM(["TYPE: number\n```sql\nSELECT SUM(line_amount) FROM sl_lines WHERE state IN ('completed','delivering') AND order_date = DATE '2026-03-03'\n```"] * 3)
    run_pipeline(str(cfg), str(split), str(out), llm=llm)
    assert json.loads(out.read_text(encoding="utf-8")) == {"id": "q_rev", "type": "number", "answer": 85000}
    assert "sl_lines" in llm.prompts[0]   # the rulebook-applied tables are still offered to the LLM
    log = json.loads((tmp_path / "out" / "logs" / "q_rev.json").read_text(encoding="utf-8"))
    assert log["fast_path"] == {"disabled": True} and all(c["source"] == "llm" for c in log["candidates"])


def test_resume_skips_answered(tmp_path):
    questions = [{"id": "q_rev", "business_id": "biz_1", "question": "Hôm nay cửa hàng bán được bao nhiêu tiền?", "as_of_date": "2026-03-03"}]
    split = make_split(tmp_path, questions)
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text("agent:\n  mode: sql\n  no_llm: true\n", encoding="utf-8")
    out = tmp_path / "out" / "submission.jsonl"
    out.parent.mkdir(parents=True)
    out.write_text('{"id": "q_rev", "type": "abstain"}\n', encoding="utf-8")
    run_pipeline(str(cfg), str(split), str(out), resume=True)
    assert out.read_text(encoding="utf-8").splitlines() == ['{"id": "q_rev", "type": "abstain"}']
