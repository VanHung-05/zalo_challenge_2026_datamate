"""Sandboxed execution of LLM-written pandas code (plan §4.5, code-agent mode).

One worker process per business keeps the business's DataFrames in memory. Each run:
  - gets fresh copies of the DataFrames (Copy-on-Write: cheap, and nothing a program does —
    inplace drops, assignments — can leak into the next question; the old thread sandbox leaked),
  - is checked statically first (no os / sys / subprocess / sockets / open / eval / exec / imports
    outside a whitelist),
  - has a real wall-clock timeout: a stuck worker is killed and restarted (the old thread-based
    timeout returned late and left the code running),
  - returns stdout (truncated), the value of `result` converted to plain Python, or the traceback.
"""

import ast
import io
import multiprocessing as mp
import os
import pickle
import sys
import traceback
from typing import Any, Dict, Optional, Tuple

ALLOWED_IMPORTS = {"pandas", "numpy", "math", "datetime", "re", "collections", "itertools", "statistics",
                   "decimal", "unicodedata", "functools", "operator", "calendar", "string"}
FORBIDDEN_NAMES = {"eval", "exec", "compile", "__import__", "open", "input", "globals", "locals", "vars",
                   "getattr", "setattr", "delattr", "breakpoint", "exit", "quit", "help", "memoryview"}
FORBIDDEN_ATTRS = {"to_csv", "to_excel", "to_parquet", "to_pickle", "to_sql", "to_json", "to_hdf", "to_feather",
                   "read_csv", "read_excel", "read_parquet", "read_pickle", "read_sql", "read_json", "system",
                   "popen", "remove", "unlink", "rmtree", "__subclasses__", "__globals__", "__builtins__", "__class__"}
MAX_STDOUT = 6000


def check_code(code: str) -> Optional[str]:
    """None if the code passes the static checks, else the reason."""
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return f"SyntaxError: {e}"
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                if a.name.split(".")[0] not in ALLOWED_IMPORTS:
                    return f"import không được phép: {a.name} (chỉ dùng {', '.join(sorted(ALLOWED_IMPORTS))})"
        elif isinstance(node, ast.ImportFrom):
            if (node.module or "").split(".")[0] not in ALLOWED_IMPORTS:
                return f"import không được phép: {node.module}"
        elif isinstance(node, ast.Name) and node.id in FORBIDDEN_NAMES:
            return f"không được dùng {node.id}"
        elif isinstance(node, ast.Attribute) and (node.attr in FORBIDDEN_ATTRS or node.attr.startswith("__")):
            return f"không được dùng .{node.attr} (không đọc/ghi file, dữ liệu đã nạp sẵn trong `tables`)"
    return None


def to_plain(v: Any, max_rows: int = 2000) -> Any:
    """result -> JSON-able Python (DataFrame -> {'columns','rows'}, Series/ndarray -> list, numpy scalars)."""
    import numpy as np
    import pandas as pd
    if isinstance(v, pd.DataFrame):
        df = v.head(max_rows)
        return {"__table__": True, "columns": [str(c) for c in df.columns],
                "rows": [[to_plain(x) for x in r] for r in df.itertuples(index=False, name=None)], "n_rows": len(v)}
    if isinstance(v, pd.Series):
        return [to_plain(x) for x in v.head(max_rows).tolist()]
    if isinstance(v, (list, tuple)):
        return [to_plain(x) for x in list(v)[:max_rows]]
    if isinstance(v, np.ndarray):
        return [to_plain(x) for x in v.tolist()[:max_rows]]
    if isinstance(v, np.generic):
        return v.item()
    if isinstance(v, pd.Timestamp):
        return v.strftime("%Y-%m-%d %H:%M:%S") if (v.hour or v.minute or v.second) else v.strftime("%Y-%m-%d")
    if v is pd.NaT:
        return None
    try:
        import datetime as _dt
        if isinstance(v, (_dt.date, _dt.datetime)):
            return v.isoformat()
    except Exception:
        pass
    if isinstance(v, float) and v != v:
        return None
    if isinstance(v, (str, int, float, bool)) or v is None:
        return v
    return str(v)


def _worker(data_path: str, conn) -> None:
    import builtins
    import datetime
    import math
    import re
    import numpy as np
    import pandas as pd
    if int(pd.__version__.split(".")[0]) < 3:
        pd.options.mode.copy_on_write = True   # default from pandas 3; makes the per-run copies safe
    with open(data_path, "rb") as f:
        data = pickle.load(f)
    real_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name.split(".")[0] not in ALLOWED_IMPORTS:
            raise ImportError(f"import {name} không được phép")
        return real_import(name, *args, **kwargs)

    safe_builtins = {k: getattr(builtins, k) for k in dir(builtins) if k not in FORBIDDEN_NAMES and not k.startswith("_")}
    safe_builtins["__import__"] = guarded_import
    conn.send({"ready": True})
    while True:
        msg = conn.recv()
        if msg is None:
            return
        code = msg["code"]
        env = {
            "__builtins__": safe_builtins, "pd": pd, "np": np, "math": math, "re": re, "datetime": datetime.datetime,
            "date": datetime.date, "timedelta": datetime.timedelta,
            "tables": {k: v.copy(deep=False) for k, v in data["tables"].items()},
            "result": None,
        }
        for k, v in data.get("extra", {}).items():
            env[k] = v.copy(deep=False) if hasattr(v, "copy") else v
        env.update(msg.get("variables", {}))
        buf = io.StringIO()
        old = sys.stdout
        sys.stdout = buf
        try:
            exec(compile(code, "<llm_code>", "exec"), env)
            out = {"ok": True, "result": to_plain(env.get("result")), "result_type": type(env.get("result")).__name__}
        except Exception as e:
            tb = traceback.format_exc()
            # keep the frames of the generated code only
            lines = [l for l in tb.splitlines() if "<llm_code>" in l or not l.startswith("  File")]
            out = {"ok": False, "error": f"{type(e).__name__}: {e}", "traceback": "\n".join(lines[-12:])}
        finally:
            sys.stdout = old
        s = buf.getvalue()
        out["stdout"] = s if len(s) <= MAX_STDOUT else s[:MAX_STDOUT] + f"\n... (cắt bớt, tổng {len(s)} ký tự)"
        conn.send(out)


class PyExecutor:
    """Holds one worker process for one business. Use .run(code) -> dict."""

    def __init__(self, tables: Dict[str, Any], extra: Optional[Dict[str, Any]] = None, cache_dir: Optional[str] = None,
                 name: str = "biz"):
        cache_dir = cache_dir or os.path.join(os.environ.get("TMPDIR", "/tmp"), "datamate_cache")
        os.makedirs(cache_dir, exist_ok=True)
        self.data_path = os.path.join(cache_dir, f"{name}.frames.pkl")
        with open(self.data_path, "wb") as f:
            pickle.dump({"tables": tables, "extra": extra or {}}, f, protocol=pickle.HIGHEST_PROTOCOL)
        self._ctx = mp.get_context("spawn")
        self.proc = None
        self.conn = None

    def _start(self) -> None:
        parent, child = self._ctx.Pipe()
        self.proc = self._ctx.Process(target=_worker, args=(self.data_path, child), daemon=True)
        self.proc.start()
        self.conn = parent
        # loading the business's frames is not charged to the first program's timeout
        if not parent.poll(300) or not parent.recv().get("ready"):
            raise RuntimeError("code worker did not start")

    def run(self, code: str, timeout: float = 30.0, variables: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        problem = check_code(code)
        if problem:
            return {"ok": False, "error": problem, "traceback": "", "stdout": ""}
        if self.proc is None or not self.proc.is_alive():
            self._start()
        self.conn.send({"code": code, "variables": variables or {}})
        if self.conn.poll(timeout):
            try:
                return self.conn.recv()
            except EOFError:
                pass
        self.restart()
        return {"ok": False, "error": f"Code chạy quá {timeout:.0f} giây (bị dừng); hãy viết cách tính gọn hơn, lọc sớm",
                "traceback": "", "stdout": "", "timeout": True}

    def restart(self) -> None:
        if self.proc is not None:
            self.proc.kill()
            self.proc.join(5)
        self.proc = None

    def close(self) -> None:
        if self.proc is not None and self.proc.is_alive():
            try:
                self.conn.send(None)
                self.proc.join(5)
            except Exception:
                pass
            if self.proc.is_alive():
                self.proc.kill()
        self.proc = None
