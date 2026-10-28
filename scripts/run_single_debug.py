#!/usr/bin/env python3
"""Debug script to run a single question with detailed execution trace."""

import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.datamate.loader import load_business_data
from src.datamate.code_generator import build_prompt, extract_python_code
from src.datamate.rules import get_rulebook
from src.datamate.sandbox import execute_code
from src.datamate.llm import LLMWrapper
from src.datamate.agent import answer_question, infer_expected_type
from src.datamate.runner import load_config, load_calendar


def main():
    q_id = sys.argv[1] if len(sys.argv) > 1 else "q_0cb7c9b93048"  # "Cho tôi biết số đơn hôm qua." biz_101
    split_dir = "train"
    config = load_config("configs/dev.yaml")

    # Find question
    target_q = None
    with open(f"{split_dir}/questions.jsonl") as f:
        for line in f:
            item = json.loads(line)
            if item["id"] == q_id:
                target_q = item
                break

    if not target_q:
        print(f"Question {q_id} not found!")
        return

    print("=" * 60)
    print(f"Question ID: {target_q['id']}")
    print(f"Business:    {target_q['business_id']}")
    print(f"Question:    {target_q['question']}")
    print(f"As of Date:  {target_q['as_of_date']}")
    print(f"Inferred:    {infer_expected_type(target_q['question'])}")

    # Find ground truth
    with open(f"{split_dir}/answers.jsonl") as f:
        for line in f:
            item = json.loads(line)
            if item["id"] == q_id:
                print(f"Ground Truth: {item}")
                break

    print("=" * 60)

    # Load business
    calendar_df = load_calendar(split_dir)
    biz_dir = f"{split_dir}/data/{target_q['business_id']}"
    bdata = load_business_data(biz_dir, calendar_df=calendar_df)
    print(f"Loaded tables: {list(bdata.tables.keys())}")

    from src.datamate.time_resolver import resolve_time
    time_range = resolve_time(target_q["question"], target_q["as_of_date"], calendar_df=calendar_df)
    if time_range:
        print(f"Resolved Time Range: [{time_range.start_date} -> {time_range.end_date}] ({time_range.description})")

    # Build prompt
    rb = get_rulebook(split_dir)
    prompt = build_prompt(
        question=target_q["question"],
        as_of_date=target_q["as_of_date"],
        business_data=bdata,
        rules_summary=rb.get_core_rules_prompt(),
        time_range=time_range
    )
    print("\n--- GENERATED PROMPT (snippet) ---")
    print(prompt[:600] + "\n...\n" + prompt[-300:])

    # Generate
    llm = LLMWrapper(config)
    print("\nCalling LLM...")
    resps = llm.generate(prompt)
    raw_resp = resps[0] if resps else ""
    print("\n--- LLM RAW RESPONSE ---")
    print(raw_resp)

    code = extract_python_code(raw_resp)
    print("\n--- EXTRACTED CODE ---")
    print(code)

    # Sandbox
    print("\nExecuting code in sandbox...")
    context = {
        "as_of_date": target_q["as_of_date"],
        "calendar": bdata.calendar,
        "unit_conversion": bdata.unit_conversion
    }
    result, err = execute_code(code, bdata.tables, context=context, timeout=30)
    print(f"\nResult: {result}")
    print(f"Error:  {err}")

    # Final formatted answer
    final_sub = answer_question(
        question_id=target_q["id"],
        business_id=target_q["business_id"],
        question=target_q["question"],
        as_of_date=target_q["as_of_date"],
        business_data=bdata,
        config=config,
        llm=llm
    )
    print("\n--- FINAL PREDICTION ---")
    print(final_sub)


if __name__ == "__main__":
    main()
