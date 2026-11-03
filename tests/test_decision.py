from src.datamate.decision import Calibrator, Candidate, FEATURE_ORDER, cluster_candidates, decide, sanity_checks


def num(v, source="llm", **kw):
    return Candidate({"type": "number", "answer": v}, source, **kw)


def test_clusters_use_official_matching():
    a = Candidate({"type": "string", "answer": "Cà phê sữa (M)"}, "llm")
    b = Candidate({"type": "string", "answer": "cà phê sữa m"}, "llm")
    c = Candidate({"type": "string", "answer": "Bạc xỉu"}, "llm")
    clusters = cluster_candidates([c, a, b])
    assert [len(cl) for cl in clusters] == [2, 1]


def test_agreeing_fast_path_answers():
    d = decide("q", [num(100, "fast_path"), num(100.0)])
    assert d.action == "value" and d.item == {"id": "q", "type": "number", "answer": 100}


def test_disagreement_is_vetoed():
    d = decide("q", [num(100), num(101)])
    assert d.action == "abstain" and "candidates_disagree" in d.reason


def test_single_llm_answer_below_tau_abstains():
    d = decide("q", [num(5)], tau=0.7)
    assert d.action == "abstain"


def test_guard_unanswerable_wins_when_confident():
    d = decide("q", [], guard_p_unans=0.9, guard_reason="Dữ liệu không có giá vốn")
    assert d.item == {"id": "q", "type": "unanswerable", "reason": "Dữ liệu không có giá vốn"}
    assert decide("q", [], guard_p_unans=0.6).action == "abstain"


def test_risk_flags_force_abstain():
    d = decide("q", [num(1, "fast_path")], risk_flags=["entity_ambiguous"])
    assert d.action == "abstain"


def test_failed_checks_lower_confidence():
    c = num(1, "fast_path", checks={"type_matches_question": False})
    assert decide("q", [c]).action == "abstain"


def test_sanity_checks():
    assert sanity_checks({"type": "percent", "answer": 0.12})["percent_not_ratio_like"] is False
    assert sanity_checks({"type": "number", "answer": -2}, is_count=True)["count_non_negative_int"] is False
    ck = sanity_checks({"type": "list", "answer": ["A", "B", "C", "D"]}, "list", top_k=3, known_names={"a", "b", "c", "d"})
    assert ck["list_len_le_k"] is False and ck["names_in_data"] is True


def test_calibrator_roundtrip(tmp_path):
    cal = Calibrator([0.0] * len(FEATURE_ORDER), 2.0)
    path = tmp_path / "cal.json"
    cal.save(str(path))
    loaded = Calibrator.load(str(path))
    assert loaded.fitted and abs(loaded.predict({}) - 0.8808) < 1e-3
    assert not Calibrator.load(str(tmp_path / "missing.json")).fitted
