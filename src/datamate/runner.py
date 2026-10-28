"""Pipeline runner for DataMate.
Orchestrates loading data, answering questions, and writing submission.jsonl.
"""

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional
import yaml
import pandas as pd

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.datamate.loader import BusinessData, load_business_data
from src.datamate.formatter import format_submission_item
from src.datamate.agent import answer_question
from src.datamate.llm import LLMWrapper


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


def run_pipeline(
    config_path: str,
    data_dir: str,
    output_path: str,
    limit: Optional[int] = None,
    business_filter: Optional[str] = None,
    dry_run: bool = False,
    agent_fn: Optional[Any] = None
) -> None:
    start_time = time.time()
    config = load_config(config_path)

    # Initialize LLM if running live
    llm = None
    if not dry_run and agent_fn is None:
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

    print(f"Loaded {len(questions)} questions from {questions_path}")
    print(f"Output will be saved to: {output_path}")

    # Ensure output directory exists
    out_dir = os.path.dirname(os.path.abspath(output_path))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    # Cache loaded businesses
    business_cache: Dict[str, BusinessData] = {}
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

            if dry_run:
                submission_row = {"id": qid, "type": "abstain"}
            else:
                # Ensure business data is loaded
                if biz_id not in business_cache:
                    biz_path = os.path.join(biz_base_dir, biz_id)
                    if os.path.exists(biz_path):
                        bdata = load_business_data(biz_path, calendar_df=calendar_df)
                        business_cache[biz_id] = bdata
                    else:
                        bdata = None
                else:
                    bdata = business_cache[biz_id]

                if bdata is None:
                    submission_row = {"id": qid, "type": "abstain"}
                else:
                    try:
                        if agent_fn is not None:
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
                    except Exception as e:
                        submission_row = {"id": qid, "type": "abstain"}

            out_f.write(json.dumps(submission_row, ensure_ascii=False) + "\n")
            out_f.flush()
            processed_count += 1
            q_dur = time.time() - q_start

            # Real-time per-question logging
            p_type = submission_row.get("type", "unknown")
            ans_val = submission_row.get("answer") if p_type != "abstain" else submission_row.get("reason", "abstain")
            ans_str = str(ans_val)
            if len(ans_str) > 50:
                ans_str = ans_str[:47] + "..."

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
    args = parser.parse_args()

    run_pipeline(
        config_path=args.config,
        data_dir=args.data_dir,
        output_path=args.output,
        limit=args.limit,
        business_filter=args.business,
        dry_run=args.dry_run
    )


if __name__ == "__main__":
    main()
