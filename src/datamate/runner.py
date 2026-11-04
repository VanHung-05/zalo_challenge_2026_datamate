"""Pipeline runner for DataMate.
Orchestrates loading data, answering questions, and writing submission.jsonl.

agent.mode = sql with agent.pipeline = v2 (default) runs src/datamate/pipeline.py: Business Card +
semantic layer + fast path + Guard + LLM self-consistency + calibrated decision. pipeline = v1 keeps
the single-shot text-to-SQL baseline; mode = pandas the older code-generation agent.
Answers are streamed (SubmissionWriter: one fsync'd line per question, resumable) and every
question gets /result/logs/<id>.json. A global time budget turns the remaining questions into
abstain before the 6-hour limit (Rule 7).
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
from src.datamate.rules import get_rulebook, load_calendar as _load_validated_calendar
from src.datamate.writer import SubmissionWriter


def load_config(config_path: str) -> Dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_calendar(split_dir: str) -> Optional[pd.DataFrame]:
    return _load_validated_calendar(split_dir)


def write_question_log(logs_dir: Optional[str], qid: str, log: Dict[str, Any]) -> None:
    """Per-question log (/result/logs/<id>.json): id, final answer, elapsed time and execution trace (Rule 8)."""
    if not logs_dir:
        return
    try:
        with open(os.path.join(logs_dir, f"{qid}.json"), "w", encoding="utf-8") as f:
            json.dump(log, f, ensure_ascii=False, indent=1, default=str)
    except Exception as e:
        print(f"[Warning] Failed to write log for {qid}: {e}")


def _resolve(path: Optional[str]) -> Optional[str]:
    """Config paths are relative to the repository root (the image's WORKDIR)."""
    if not path:
        return None
    if os.path.isabs(path):
        return path
    return os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")), path)


def run_pipeline(
    config_path: str,
    data_dir: str,
    output_path: str,
    limit: Optional[int] = None,
    business_filter: Optional[str] = None,
    dry_run: bool = False,
    agent_fn: Optional[Any] = None,
    llm: Optional[Any] = None,
    logs_dir: Optional[str] = None,
    resume: bool = False,
) -> None:
    start_time = time.time()
    config = load_config(config_path)
    runner_cfg = config.get("runner", {})
    agent_cfg = config.get("agent", {})
    mode = agent_cfg.get("mode", "pandas")
    use_v2 = mode == "sql" and agent_cfg.get("pipeline", "v2") == "v2" and agent_fn is None
    time_budget = float(runner_cfg.get("time_budget_seconds", 5 * 3600))
    question_budget = float(runner_cfg.get("question_timeout_seconds", 150))

    # Initialize LLM if running live (v2 can run without one: fast path + Guard only)
    if llm is None and not dry_run and agent_fn is None and not agent_cfg.get("no_llm", False):
        try:
            llm = LLMWrapper(config)
        except Exception as e:
            print(f"[Warning] Failed to initialize LLM: {e}. " + ("Continuing without LLM." if use_v2 else "Falling back to abstain."))
            if not use_v2:
                dry_run = True

    questions_path = os.path.join(data_dir, "questions.jsonl")
    if not os.path.exists(questions_path):
        raise FileNotFoundError(f"Questions file not found: {questions_path}")

    calendar_df = load_calendar(data_dir)

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

    if use_v2:
        # agent.mode only selects the v2 pipeline; what the LLM writes is agent.synthesis
        run_desc = (f"pipeline=v2, LLM writes {agent_cfg.get('synthesis', 'sql')}, "
                    f"fast_path={'on' if agent_cfg.get('fast_path', True) else 'off'}, "
                    f"n_samples={agent_cfg.get('n_samples', 1)}, calibrator={config.get('decision', {}).get('calibrator') or 'none'}")
    else:
        run_desc = f"mode={mode}, pipeline=v1"
    print(f"Loaded {len(questions)} questions from {questions_path} ({run_desc})")
    print(f"Output will be saved to: {output_path}")

    out_dir = os.path.dirname(os.path.abspath(output_path))
    if logs_dir is None:
        logs_dir = runner_cfg.get("logs_dir") or os.path.join(out_dir, "logs")
    print(f"Per-question logs: {logs_dir}")

    # The split's own rulebook (it may differ between train / public / private)
    rulebook = get_rulebook(data_dir)
    if not rulebook.raw_text:
        print(f"[Warning] No rulebook.md found in {data_dir}")
    elif rulebook.uncertain_params():
        print(f"[Rules] parameters not found in this rulebook (questions depending on them abstain): {rulebook.uncertain_params()}")

    cache_dir = runner_cfg.get("cache_dir") or os.path.join(os.environ.get("TMPDIR", "/tmp"), "datamate_cache")
    guard = calibrator = priors = None
    if use_v2:
        from src.datamate.decision import Calibrator
        from src.datamate.guard import Guard
        from src.datamate.pipeline import load_fast_path_priors
        dcfg = config.get("decision", {})
        guard = Guard(_resolve(dcfg.get("guard_weights")), threshold=float(dcfg.get("tau_unanswerable", 0.8)))
        calibrator = Calibrator.load(_resolve(dcfg.get("calibrator")))
        priors = load_fast_path_priors(_resolve(dcfg.get("fast_path_priors")))

    business_cache: Dict[str, Any] = {}
    db_cache: Dict[str, Any] = {}
    biz_base_dir = os.path.join(data_dir, "data")

    processed_count = 0
    with SubmissionWriter(output_path, logs_dir, resume=resume) as writer:
        if resume and writer.done:
            print(f"Resuming: {len(writer.done)} answers kept from the previous run")
        for q in questions:
            q_start = time.time()
            qid = q["id"]
            if writer.is_done(qid):
                continue
            biz_id = q["business_id"]
            question_text = q["question"]
            as_of_date = q.get("as_of_date", "")

            log: Dict[str, Any] = {
                "id": qid, "business_id": biz_id, "question": question_text,
                "as_of_date": as_of_date, "mode": mode,
            }

            if dry_run:
                submission_row = {"id": qid, "type": "abstain"}
            elif time.time() - start_time > time_budget:
                # Global watchdog: whatever is left is abstain (scored 0), never a crash past the limit
                submission_row = {"id": qid, "type": "abstain"}
                log["skipped"] = f"time budget {time_budget:.0f}s exhausted"
            else:
                if biz_id not in business_cache:
                    biz_path = os.path.join(biz_base_dir, biz_id)
                    ctx = None
                    if os.path.exists(biz_path):
                        try:
                            t_load = time.time()
                            if use_v2:
                                from src.datamate.pipeline import build_context
                                ctx = build_context(biz_path, calendar_df, rulebook, config, cache_dir)
                            else:
                                ctx = load_business_data(biz_path, calendar_df=calendar_df, keep_time=(mode == "sql"))
                                if mode == "sql":
                                    from src.datamate.db import BusinessDB
                                    db_cache[biz_id] = BusinessDB(ctx, calendar_df, threads=agent_cfg.get("sql_threads", 1))
                            print(f"[Load] {biz_id} in {time.time() - t_load:.1f}s", flush=True)
                        except Exception:
                            log["load_error"] = traceback.format_exc()
                            ctx = None
                    # one code worker at a time: the previous business's process is stopped (it restarts if needed)
                    for other in business_cache.values():
                        if hasattr(other, "close"):
                            other.close()
                    business_cache[biz_id] = ctx
                ctx = business_cache[biz_id]

                if ctx is None:
                    submission_row = {"id": qid, "type": "abstain"}
                    log.setdefault("load_error", "business data not found or failed to load")
                else:
                    try:
                        if use_v2:
                            from src.datamate.pipeline import answer as answer_v2
                            submission_row, trace = answer_v2(q, ctx, rulebook, calendar_df, config, llm, guard, calibrator,
                                                              priors, deadline=q_start + question_budget)
                            log.update(trace)
                        elif mode == "sql" and agent_fn is None:
                            from src.datamate.sql_agent import answer_question_sql
                            submission_row = answer_question_sql(
                                question_id=qid, question=question_text, as_of_date=as_of_date, db=db_cache[biz_id],
                                readme=ctx.readme, rulebook_text=rulebook.raw_text, config=config, llm=llm,
                                calendar_df=calendar_df, trace=log)
                        elif agent_fn is not None:
                            submission_row = agent_fn(question_id=qid, business_id=biz_id, question=question_text,
                                                      as_of_date=as_of_date, business_data=ctx, config=config)
                        else:
                            submission_row = answer_question(question_id=qid, business_id=biz_id, question=question_text,
                                                             as_of_date=as_of_date, business_data=ctx, config=config, llm=llm)
                    except Exception:
                        submission_row = {"id": qid, "type": "abstain"}
                        log["exception"] = traceback.format_exc()

            q_dur = time.time() - q_start
            log["final"] = submission_row
            log["elapsed_seconds"] = round(q_dur, 3)
            writer.write(submission_row, log)
            processed_count += 1

            # Real-time per-question logging
            p_type = submission_row.get("type", "unknown")
            ans_val = submission_row.get("answer") if p_type != "abstain" else submission_row.get("reason", "abstain")
            ans_str = str(ans_val)
            if len(ans_str) > 50:
                ans_str = ans_str[:47] + "..."
            q_short = question_text if len(question_text) <= 60 else question_text[:57] + "..."
            print(f"[{processed_count}/{len(questions)}] [{biz_id}] ({q_dur:.1f}s) \"{q_short}\"", flush=True)
            print(f"       -> [{p_type}]: {ans_str}", flush=True)

    for ctx in business_cache.values():
        if hasattr(ctx, "close"):
            ctx.close()
    total_elapsed = time.time() - start_time
    print(f"\n==========================================")
    print(f"Completed {processed_count} questions in {total_elapsed:.2f}s ({processed_count / max(total_elapsed, 1e-9):.2f} q/s)")
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
    parser.add_argument("--resume", action="store_true", help="Keep answers already in --output and skip those ids")
    parser.add_argument("--no-llm", action="store_true", help="v2 pipeline without LLM (fast path + Guard only)")
    args = parser.parse_args()

    if args.no_llm:
        cfg = load_config(args.config)
        cfg.setdefault("agent", {})["no_llm"] = True
        tmp = os.path.join(os.path.dirname(os.path.abspath(args.output)) or ".", ".config_no_llm.yaml")
        os.makedirs(os.path.dirname(tmp), exist_ok=True)
        with open(tmp, "w", encoding="utf-8") as f:
            yaml.safe_dump(cfg, f, allow_unicode=True)
        args.config = tmp

    run_pipeline(
        config_path=args.config,
        data_dir=args.data_dir,
        output_path=args.output,
        limit=args.limit,
        business_filter=args.business,
        dry_run=args.dry_run,
        logs_dir=args.logs_dir,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()
