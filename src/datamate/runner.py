"""Pipeline runner for DataMate.
Orchestrates loading data, answering questions, and writing submission.jsonl.
"""

import argparse
import json
import os
import sys
import time
import traceback
from typing import Any, Dict, List, Optional
import yaml
import pandas as pd

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.datamate.loader import BusinessData, load_business_data
from src.datamate.formatter import format_submission_item
from src.datamate.agent import answer_question
from src.datamate.llm import LLMWrapper
from src.datamate.rules import get_rulebook


def load_config(config_path: str) -> Dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_calendar(split_dir: str) -> Optional[pd.DataFrame]:
    cal_path = os.path.join(split_dir, "calendar.csv")
    if os.path.exists(cal_path):
        try:
            return pd.read_csv(cal_path)
        except Exception:
            pass
    return None


def write_question_log(logs_dir: Optional[str], qid: str, log: Dict[str, Any]) -> None:
    """Per-question log (/result/logs/<id>.json): id, final answer, elapsed time and execution trace (Rule 8)."""
    if not logs_dir:
        return
    try:
        with open(os.path.join(logs_dir, f"{qid}.json"), "w", encoding="utf-8") as f:
            json.dump(log, f, ensure_ascii=False, indent=1, default=str)
    except Exception as e:
        print(f"[Warning] Failed to write log for {qid}: {e}")


def abstain_reason(log: Dict[str, Any]) -> str:
    """One-line cause of an abstain from the question log, so the console shows why without opening logs."""
    if log.get("exception"):
        return "exception: " + log["exception"].strip().splitlines()[-1]
    if log.get("load_error"):
        return "load_error: " + log["load_error"]
    attempts = log.get("attempts") or []
    if not attempts:
        return "abstain (no LLM attempt; LLM not initialized?)"
    last = attempts[-1]
    if not last.get("llm_response"):
        return f"abstain: empty LLM response x{len(attempts)} (check max_tokens / <think>)"
    return f"abstain after {len(attempts)} attempts: {last.get('outcome') or last.get('error') or 'unknown'}"


def run_pipeline(
    config_path: str,
    data_dir: str,
    output_path: str,
    limit: Optional[int] = None,
    business_filter: Optional[str] = None,
    dry_run: bool = False,
    agent_fn: Optional[Any] = None,
    llm: Optional[Any] = None,
    logs_dir: Optional[str] = None
) -> None:
    start_time = time.time()
    config = load_config(config_path)
    runner_cfg = config.get("runner", {})
    mode = config.get("agent", {}).get("mode", "pandas")

    # Initialize LLM if running live
    if llm is None and not dry_run and agent_fn is None:
        try:
            llm = LLMWrapper(config)
        except Exception as e:
            print(f"[Warning] Failed to initialize LLM: {e}. Falling back to abstain.")
            dry_run = True

    questions_path = os.path.join(data_dir, "questions.jsonl")
    if not os.path.exists(questions_path):
        raise FileNotFoundError(f"Questions file not found: {questions_path}")

    # Load calendar
    calendar_df = load_calendar(data_dir)

    # Read questions
    questions = []
    with open(questions_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                q_item = json.loads(line)
                if business_filter and q_item.get("business_id") != business_filter:
                    continue
                questions.append(q_item)

    if limit is not None:
        questions = questions[:limit]

    # Answer the questions of one business together: data stays hot and the LLM server can reuse
    # the cached prompt prefix (rulebook + README + schema). Line order does not affect scoring (matched by id).
    if runner_cfg.get("group_by_business", True):
        questions.sort(key=lambda q: q.get("business_id", ""))

    print(f"Loaded {len(questions)} questions from {questions_path} (mode={mode})")
    print(f"Output will be saved to: {output_path}")

    # Ensure output directory exists
    out_dir = os.path.dirname(os.path.abspath(output_path))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    if logs_dir is None:
        logs_dir = runner_cfg.get("logs_dir") or os.path.join(out_dir, "logs")
    os.makedirs(logs_dir, exist_ok=True)
    print(f"Per-question logs: {logs_dir}")

    # The split's own rulebook (it may differ between train / public / private)
    rulebook = get_rulebook(data_dir)
    if not rulebook.raw_text:
        print(f"[Warning] No rulebook.md found in {data_dir}")

    # Cache loaded businesses
    business_cache: Dict[str, BusinessData] = {}
    db_cache: Dict[str, Any] = {}
    biz_base_dir = os.path.join(data_dir, "data")

    # Open output file for streaming writes
    processed_count = 0
    with open(output_path, "w", encoding="utf-8") as out_f:
        for q in questions:
            q_start = time.time()
            qid = q["id"]
            biz_id = q["business_id"]
            question_text = q["question"]
            as_of_date = q.get("as_of_date", "")

            log: Dict[str, Any] = {
                "id": qid, "business_id": biz_id, "question": question_text,
                "as_of_date": as_of_date, "mode": mode,
            }

            if dry_run:
                submission_row = {"id": qid, "type": "abstain"}
            else:
                # Ensure business data is loaded
                if biz_id not in business_cache:
                    biz_path = os.path.join(biz_base_dir, biz_id)
                    bdata = None
                    if os.path.exists(biz_path):
                        try:
                            t_load = time.time()
                            bdata = load_business_data(biz_path, calendar_df=calendar_df, keep_time=(mode == "sql"))
                            if mode == "sql":
                                from src.datamate.db import BusinessDB
                                db_cache[biz_id] = BusinessDB(
                                    bdata, calendar_df, threads=config.get("agent", {}).get("sql_threads", 1)
                                )
                            print(f"[Load] {biz_id}: {len(bdata.tables)} tables in {time.time() - t_load:.1f}s", flush=True)
                        except Exception:
                            log["load_error"] = traceback.format_exc()
                            bdata = None
                    business_cache[biz_id] = bdata
                else:
                    bdata = business_cache[biz_id]

                if bdata is None:
                    submission_row = {"id": qid, "type": "abstain"}
                    log.setdefault("load_error", "business data not found or failed to load")
                else:
                    try:
                        if mode == "sql" and agent_fn is None:
                            from src.datamate.sql_agent import answer_question_sql
                            submission_row = answer_question_sql(
                                question_id=qid,
                                question=question_text,
                                as_of_date=as_of_date,
                                db=db_cache[biz_id],
                                readme=bdata.readme,
                                rulebook_text=rulebook.raw_text,
                                config=config,
                                llm=llm,
                                calendar_df=calendar_df,
                                trace=log
                            )
                        elif agent_fn is not None:
                            submission_row = agent_fn(
                                question_id=qid,
                                business_id=biz_id,
                                question=question_text,
                                as_of_date=as_of_date,
                                business_data=bdata,
                                config=config
                            )
                        else:
                            submission_row = answer_question(
                                question_id=qid,
                                business_id=biz_id,
                                question=question_text,
                                as_of_date=as_of_date,
                                business_data=bdata,
                                config=config,
                                llm=llm
                            )
                    except Exception:
                        submission_row = {"id": qid, "type": "abstain"}
                        log["exception"] = traceback.format_exc()

            out_f.write(json.dumps(submission_row, ensure_ascii=False) + "\n")
            out_f.flush()
            processed_count += 1
            q_dur = time.time() - q_start
            log["final"] = submission_row
            log["elapsed_seconds"] = round(q_dur, 3)
            write_question_log(logs_dir, qid, log)

            # Real-time per-question logging
            p_type = submission_row.get("type", "unknown")
            ans_val = submission_row.get("answer") if p_type != "abstain" else abstain_reason(log)
            ans_str = str(ans_val)
            if len(ans_str) > 120:
                ans_str = ans_str[:117] + "..."

            q_short = question_text if len(question_text) <= 60 else question_text[:57] + "..."
            print(f"[{processed_count}/{len(questions)}] [{biz_id}] ({q_dur:.1f}s) \"{q_short}\"", flush=True)
            print(f"       -> [{p_type}]: {ans_str}", flush=True)

    total_elapsed = time.time() - start_time
    print(f"\n==========================================")
    print(f"Completed {processed_count} questions in {total_elapsed:.2f}s ({processed_count / total_elapsed:.2f} q/s)")
    print(f"Results saved to: {output_path}")
    print(f"==========================================")


def main():
    parser = argparse.ArgumentParser(description="Run DataMate agent pipeline")
    parser.add_argument("--config", "-c", default="configs/dev.yaml", help="Path to config yaml")
    parser.add_argument("--data-dir", "-d", default="train", help="Path to split root (e.g. train, public_test, /data)")
    parser.add_argument("--output", "-o", default="submission.jsonl", help="Path to write submission.jsonl")
    parser.add_argument("--limit", "-l", type=int, default=None, help="Limit number of questions")
    parser.add_argument("--business", "-b", type=str, default=None, help="Filter by business_id")
    parser.add_argument("--dry-run", action="store_true", help="Output abstain for all questions")
    parser.add_argument("--logs-dir", type=str, default=None, help="Per-question log dir (default: <output dir>/logs)")
    args = parser.parse_args()

    run_pipeline(
        config_path=args.config,
        data_dir=args.data_dir,
        output_path=args.output,
        limit=args.limit,
        business_filter=args.business,
        dry_run=args.dry_run,
        logs_dir=args.logs_dir
    )


if __name__ == "__main__":
    main()
