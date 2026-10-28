"""Main answering agent for DataMate.
Orchestrates prompt generation, LLM inference, sandbox execution, self-repair, and answer formatting.
"""

import re
from typing import Any, Dict, Optional, Tuple
import pandas as pd

from src.datamate.code_generator import build_prompt, extract_python_code
from src.datamate.formatter import format_submission_item
from src.datamate.llm import LLMWrapper
from src.datamate.loader import BusinessData
from src.datamate.rules import Rulebook, get_rulebook
from src.datamate.sandbox import execute_code
from src.datamate.time_resolver import resolve_time


def infer_expected_type(question: str) -> str:
    """Infer expected response type based on question phrasing."""
    q_low = question.lower()

    if any(k in q_low for k in ["%", "phần trăm", "tỉ lệ", "tỷ lệ", "chiếm bao nhiêu"]):
        return "percent"
    if any(k in q_low for k in ["ngày nào", "hạn dùng", "hạn sử dụng", "ngày mấy"]):
        return "date"
    if any(k in q_low for k in ["có phải", "có đúng không", "phải không", "đúng không", "có không", "liệu có"]):
        return "boolean"
    if any(k in q_low for k in ["top ", "danh sách", "những sản phẩm", "những khách hàng", "liệt kê", "các mặt hàng", "các sản phẩm"]):
        return "list"
    if any(k in q_low for k in ["lập bảng", "dạng bảng", "cho biết bảng"]):
        return "table"
    if any(k in q_low for k in ["bao nhiêu", "bnhiu", "bnhieu", "mấy", "tổng tiền", "doanh thu", "chi phí", "tiền", "số lượng", "đếm"]):
        return "number"
    if any(k in q_low for k in ["tên gì", "tên là gì", "ai là", "sản phẩm nào", "mặt hàng nào", "khách hàng nào", "nhà cung cấp nào"]):
        return "string"

    return "number"


def answer_question(
    question_id: str,
    business_id: str,
    question: str,
    as_of_date: str,
    business_data: BusinessData,
    config: Dict[str, Any],
    llm: Optional[LLMWrapper] = None
) -> Dict[str, Any]:
    """Execute the generate -> sandbox -> repair -> format pipeline for a question."""
    agent_cfg = config.get("agent", {})
    max_retries = agent_cfg.get("max_retries", 2)
    timeout = agent_cfg.get("sandbox_timeout", 30)

    # Initialize LLM if not provided
    if llm is None:
        llm = LLMWrapper(config)

    # Load rulebook
    rulebook = get_rulebook(business_data.data_dir)
    rules_summary = rulebook.get_core_rules_prompt()

    expected_type = infer_expected_type(question)
    time_range = resolve_time(question, as_of_date, calendar_df=business_data.calendar)

    # Entity linking
    if business_data.value_index is None:
        from src.datamate.profiler import build_value_index
        business_data.value_index = build_value_index(business_data.tables)

    from src.datamate.entity_linker import link_entities, format_entity_matches_for_prompt
    entity_matches = link_entities(question, business_data.value_index)
    entity_hints = format_entity_matches_for_prompt(entity_matches)

    previous_code = None
    error_feedback = None
    final_result = None
    success = False

    context = {
        "as_of_date": as_of_date,
        "calendar": business_data.calendar,
        "unit_conversion": business_data.unit_conversion,
        "start_date": time_range.start_date if time_range else "",
        "end_date": time_range.end_date if time_range else ""
    }

    for attempt in range(max_retries + 1):
        prompt = build_prompt(
            question=question,
            as_of_date=as_of_date,
            business_data=business_data,
            rules_summary=rules_summary,
            time_range=time_range,
            entity_hints=entity_hints,
            error_feedback=error_feedback,
            previous_code=previous_code
        )

        try:
            llm_responses = llm.generate(prompt)
            if not llm_responses or not llm_responses[0]:
                error_feedback = "LLM returned empty response"
                continue

            raw_code = extract_python_code(llm_responses[0])
            previous_code = raw_code

            # Execute code in sandbox
            result, err = execute_code(
                code=raw_code,
                data=business_data.tables,
                context=context,
                timeout=timeout
            )

            if err is None:
                final_result = result
                success = True
                break
            else:
                error_feedback = err
        except Exception as e:
            error_feedback = str(e)

    if not success or final_result is None:
        # If code failed after retries, abstain to avoid -1 penalty
        return {"id": question_id, "type": "abstain"}

    if isinstance(final_result, str) and final_result == "UNANSWERABLE":
        return format_submission_item(question_id, "UNANSWERABLE", "unanswerable")

    # Format answer according to inferred type
    return format_submission_item(question_id, final_result, expected_type)
