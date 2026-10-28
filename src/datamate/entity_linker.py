"""Entity linker for DataMate.
Performs fuzzy entity matching between question tokens/phrases and database values using RapidFuzz.
"""

import re
import unicodedata
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from rapidfuzz import fuzz, process

from src.datamate.profiler import ValueIndex


@dataclass
class EntityMatch:
    query_phrase: str
    matched_value: str
    table: str
    column: str
    score: float


STOPWORDS = {
    "cho", "tôi", "biết", "hỏi", "xem", "giúp", "bán", "được", "bao", "nhiêu",
    "tiền", "nào", "mặt", "hàng", "sản", "phẩm", "khách", "hàng", "đơn", "hôm",
    "qua", "nay", "tháng", "năm", "quý", "tuần", "với", "trong", "của", "tổng",
    "doanh", "thu", "số", "lượng", "giá", "cửa", "hàng", "là", "gì", "nhất"
}


def extract_candidate_phrases(question: str) -> List[str]:
    """Extract multi-word candidate phrases from question that could be entity names."""
    # Remove punctuation
    clean_q = re.sub(r"[^\w\s]", " ", question)
    tokens = clean_q.split()
    candidates = []

    # 1. Quoted text e.g. "Sting dâu"
    quotes = re.findall(r"[\"']([^\"']+)[\"']", question)
    for q in quotes:
        if q.strip():
            candidates.append(q.strip())

    # 2. Capitalized consecutive words (e.g. "Kẹo Alpenliebe 100g", "Bột ngọt Ajinomoto")
    cap_phrases = re.findall(r"\b[A-ZÀ-Ỹ][\w\d]*(?:\s+[A-ZÀ-Ỹ0-9][\w\d]*)*\b", question)
    for cp in cap_phrases:
        if len(cp.split()) >= 1 and cp.lower() not in STOPWORDS:
            candidates.append(cp.strip())

    # 3. N-grams (2 to 4 tokens) that don't consist purely of stopwords
    for n in range(2, 5):
        for i in range(len(tokens) - n + 1):
            phrase = " ".join(tokens[i : i + n])
            # Check that not all tokens are stopwords
            non_stop = [t for t in tokens[i : i + n] if t.lower() not in STOPWORDS]
            if len(non_stop) >= 1:
                candidates.append(phrase)

    # Dedup preserving order
    seen = set()
    deduped = []
    for c in candidates:
        c_norm = c.strip().lower()
        if c_norm not in seen and len(c.strip()) >= 3:
            seen.add(c_norm)
            deduped.append(c.strip())

    return deduped


def link_entities(
    question: str,
    value_index: ValueIndex,
    top_k: int = 4,
    score_cutoff: float = 75.0
) -> List[EntityMatch]:
    """Find the best database values matching entities mentioned in question."""
    if not value_index or not value_index.entries:
        return []

    candidates = extract_candidate_phrases(question)
    matches: List[EntityMatch] = []
    seen_matches = set()

    for phrase in candidates:
        results = value_index.search(phrase, top_k=2, score_cutoff=score_cutoff)
        for res in results:
            key = (res["matched_value"], res["table"], res["column"])
            if key not in seen_matches:
                seen_matches.add(key)
                matches.append(EntityMatch(
                    query_phrase=phrase,
                    matched_value=res["matched_value"],
                    table=res["table"],
                    column=res["column"],
                    score=res["score"]
                ))

    # Sort by score descending
    matches.sort(key=lambda x: x.score, reverse=True)
    return matches[:top_k]


def format_entity_matches_for_prompt(matches: List[EntityMatch]) -> str:
    """Format entity matches as concise hints for the LLM prompt."""
    if not matches:
        return ""
    lines = ["### THỰC THỂ KHỚP VỚI DỮ LIỆU (ENTITY LINKING):"]
    for m in matches:
        lines.append(f"- Cụm từ '{m.query_phrase}' khớp với giá trị: '{m.matched_value}' ở cột `data['{m.table}']['{m.column}']` (độ tin cậy: {m.score*100:.0f}%)")
    return "\n".join(lines)
