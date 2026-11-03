import json

from src.datamate.writer import SubmissionWriter


def _lines(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def test_streaming_and_logs(tmp_path):
    out = tmp_path / "submission.jsonl"
    logs = tmp_path / "logs"
    with SubmissionWriter(str(out), str(logs)) as w:
        w.write({"id": "q1", "type": "number", "answer": 5}, {"final": {"type": "number"}, "elapsed_seconds": 0.1})
        # visible before close (flushed per line)
        assert _lines(out) == [{"id": "q1", "type": "number", "answer": 5}]
        w.write({"id": "q1", "type": "abstain"})  # duplicate id ignored
    assert len(_lines(out)) == 1
    log = json.loads((logs / "q1.json").read_text(encoding="utf-8"))
    assert log["id"] == "q1" and log["elapsed_seconds"] == 0.1


def test_resume_keeps_valid_lines_and_drops_truncated(tmp_path):
    out = tmp_path / "submission.jsonl"
    out.write_text('{"id": "q1", "type": "abstain"}\n{"id": "q2", "type": "num', encoding="utf-8")
    with SubmissionWriter(str(out), resume=True) as w:
        assert w.is_done("q1") and not w.is_done("q2")
        w.write({"id": "q2", "type": "abstain"})
    assert [x["id"] for x in _lines(out)] == ["q1", "q2"]


def test_no_resume_truncates(tmp_path):
    out = tmp_path / "submission.jsonl"
    out.write_text('{"id": "q1", "type": "abstain"}\n', encoding="utf-8")
    with SubmissionWriter(str(out)) as w:
        assert not w.is_done("q1")
    assert _lines(out) == []
