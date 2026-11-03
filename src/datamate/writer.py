"""Streaming submission writer and per-question logs (plan §4.8).
Every answer is written and flushed (fsync) as soon as it is decided, so a crash or the 6-hour
cut-off keeps everything already answered (Rule 8: only written lines are scored). With
resume=True a restarted run keeps the valid lines of the previous run and skips their ids.
"""

import json
import os
import time
from typing import Any, Dict, Optional, Set


class SubmissionWriter:
    def __init__(self, output_path: str, logs_dir: Optional[str] = None, resume: bool = False):
        self.output_path = output_path
        self.logs_dir = logs_dir
        out_dir = os.path.dirname(os.path.abspath(output_path))
        os.makedirs(out_dir, exist_ok=True)
        if logs_dir:
            os.makedirs(logs_dir, exist_ok=True)

        self.done: Set[str] = set()
        if resume and os.path.exists(output_path):
            kept = []
            with open(output_path, "r", encoding="utf-8") as f:
                for line in f:
                    # A run killed mid-write can leave a truncated last line: drop it
                    try:
                        item = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if isinstance(item, dict) and isinstance(item.get("id"), str) and item["id"] not in self.done:
                        self.done.add(item["id"])
                        kept.append(json.dumps(item, ensure_ascii=False))
            with open(output_path, "w", encoding="utf-8") as f:
                f.write("".join(k + "\n" for k in kept))
            self._f = open(output_path, "a", encoding="utf-8")
        else:
            self._f = open(output_path, "w", encoding="utf-8")

    def is_done(self, qid: str) -> bool:
        return qid in self.done

    def write(self, item: Dict[str, Any], log: Optional[Dict[str, Any]] = None) -> None:
        qid = item["id"]
        if qid in self.done:
            return
        self._f.write(json.dumps(item, ensure_ascii=False) + "\n")
        self._f.flush()
        try:
            os.fsync(self._f.fileno())
        except OSError:
            pass
        self.done.add(qid)
        if log is not None:
            self.write_log(qid, log)

    def write_log(self, qid: str, log: Dict[str, Any]) -> None:
        """/result/logs/<id>.json: at least id, final answer and elapsed time (data_overview.md)."""
        if not self.logs_dir:
            return
        log.setdefault("id", qid)
        log.setdefault("logged_at", time.strftime("%Y-%m-%dT%H:%M:%S"))
        tmp = os.path.join(self.logs_dir, f".{qid}.json.tmp")
        try:
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(log, f, ensure_ascii=False, indent=1, default=str)
            os.replace(tmp, os.path.join(self.logs_dir, f"{qid}.json"))
        except Exception as e:
            print(f"[Warning] Failed to write log for {qid}: {e}")

    def close(self) -> None:
        if not self._f.closed:
            self._f.close()

    def __enter__(self) -> "SubmissionWriter":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()
