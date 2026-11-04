import pandas as pd
import pytest

from src.datamate.pyexec import PyExecutor, check_code


@pytest.fixture(scope="module")
def ex(tmp_path_factory):
    tables = {"BanHang": pd.DataFrame({"ten": ["A", "B"], "tien": [10, 20]})}
    e = PyExecutor(tables, extra={"sl_lines": tables["BanHang"]}, cache_dir=str(tmp_path_factory.mktemp("c")), name="t")
    yield e
    e.close()


def test_result_and_stdout(ex):
    out = ex.run("df = tables['BanHang']\nprint(df.shape)\nresult = df['tien'].sum()")
    assert out["ok"] and out["result"] == 30 and "(2, 2)" in out["stdout"]
    out = ex.run("result = tables['BanHang']")
    assert out["result"]["columns"] == ["ten", "tien"] and out["result"]["rows"] == [["A", 10], ["B", 20]]


def test_mutation_does_not_leak_between_runs(ex):
    ex.run("tables['BanHang'].drop(index=0, inplace=True)\ntables['BanHang']['tien'] = 0\nsl_lines['tien'] = 0")
    assert ex.run("result = int(tables['BanHang']['tien'].sum() + sl_lines['tien'].sum())")["result"] == 60


def test_errors_are_reported_with_traceback(ex):
    out = ex.run("x = tables['KhongCo']")
    assert not out["ok"] and "KeyError" in out["error"] and "<llm_code>" in out["traceback"]


def test_static_checks():
    assert check_code("import os") and check_code("open('x')") and check_code("pd.read_csv('x')")
    assert check_code("df.to_csv('x')") and check_code("().__class__")
    assert check_code("import pandas as pd\nfrom collections import Counter\nresult = 1") is None


def test_timeout_kills_and_recovers(ex):
    out = ex.run("while True:\n    pass", timeout=2)
    assert not out["ok"] and out.get("timeout")
    assert ex.run("result = 1")["result"] == 1
