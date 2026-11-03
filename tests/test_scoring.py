import json
import os

from src.datamate.scoring import match_answer, normalize_string, same_answer, score


def test_normalize_string_keeps_diacritics():
    assert normalize_string("  Sting  Dâu-330ml ") == "sting dâu 330ml"
    assert normalize_string("khach_hang") == "khach hang"
    assert normalize_string("Sting dầu 330ml") != normalize_string("Sting dâu 330ml")


def test_number_exact_except_float_noise():
    ref = {"type": "number", "answer": 89526400}
    assert match_answer({"type": "number", "answer": 89526400.0}, ref)
    assert not match_answer({"type": "number", "answer": 89526401}, ref)
    assert match_answer({"type": "number", "answer": 0.1 + 0.2}, {"type": "number", "answer": 0.3})
    assert not match_answer({"type": "number", "answer": "89526400"}, ref)
    assert not match_answer({"type": "number", "answer": True}, {"type": "number", "answer": 1})


def test_percent_rounded_to_two_decimals():
    ref = {"type": "percent", "answer": -12.45}
    assert match_answer({"type": "percent", "answer": -12.449}, ref)
    assert not match_answer({"type": "percent", "answer": -12.46}, ref)
    assert not match_answer({"type": "percent", "answer": -0.1245}, ref)
    assert not match_answer({"type": "percent", "answer": 12.45}, ref)


def test_type_mismatch_and_unanswerable():
    assert not match_answer({"type": "string", "answer": "5"}, {"type": "number", "answer": 5})
    assert match_answer({"type": "unanswerable"}, {"type": "unanswerable", "answer": None})
    assert score({"type": "number", "answer": 1}, {"type": "unanswerable", "answer": None}) == -1
    assert score({"type": "abstain"}, {"type": "number", "answer": 1}) == 0
    assert score(None, {"type": "number", "answer": 1}) == 0


def test_string_and_list_aliases():
    ref = {"type": "list", "answer": ["Cao Long Minh", "Ông Quang Hoàng"], "alias": [["KH0112"], []]}
    assert match_answer({"type": "list", "answer": ["kh0112", "ông quang hoàng"]}, ref)
    assert not match_answer({"type": "list", "answer": ["Ông Quang Hoàng", "Cao Long Minh"]}, ref)
    s = {"type": "string", "answer": "Vũ Châu Mai", "alias": ["KH0174"]}
    assert match_answer({"type": "string", "answer": "KH0174"}, s)


def test_table_columns_by_name_rows_as_set():
    ref = {"type": "table", "answer": {"columns": ["khach_hang", "so_tien_no"], "rows": [["An", 5], ["Bình", 3]]},
           "alias": [["KH01"], ["KH02"]]}
    reordered = {"type": "table", "answer": {"columns": ["so_tien_no", "khach_hang"], "rows": [[3, "Bình"], [5, "KH01"]]}}
    assert match_answer(reordered, ref)
    by_position = {"type": "table", "answer": {"columns": ["ten", "no"], "rows": [["Bình", 3], ["An", 5]]}}
    assert match_answer(by_position, ref)
    wrong = {"type": "table", "answer": {"columns": ["ten", "no"], "rows": [["Bình", 5], ["An", 3]]}}
    assert not match_answer(wrong, ref)


def test_same_answer_symmetric():
    a = {"type": "string", "answer": "Cà phê sữa (M)"}
    b = {"type": "string", "answer": "cà phê sữa m"}
    assert same_answer(a, b) and same_answer(b, a)


def test_gold_answers_score_perfectly():
    path = os.path.join(os.path.dirname(__file__), "..", "train", "answers.jsonl")
    if not os.path.exists(path):
        return
    with open(path, encoding="utf-8") as f:
        gold = [json.loads(line) for line in f if line.strip()]
    assert all(score(g, g) == 1 for g in gold)
