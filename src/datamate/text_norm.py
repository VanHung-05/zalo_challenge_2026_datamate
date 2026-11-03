"""Text normalization for DataMate.
- btc_normalize: the organizers' string-matching normalization (evaluation.md), shared by scorer and voter.
- normalize_question: NFC + lowercase + Telex-typo repair + teencode/abbreviation expansion, plus an
  accent-stripped copy that all question-understanding regexes run on (questions are often typed without accents).
"""

import os
import re
import unicodedata
from functools import lru_cache
from typing import Any, Dict, FrozenSet, List, Optional, Tuple

from unidecode import unidecode


def nfc(s: Any) -> str:
    if s is None:
        return ""
    return unicodedata.normalize("NFC", str(s))


def strip_accents(s: Any) -> str:
    """'Đường Hai Bà Trưng' -> 'Duong Hai Ba Trung'. Digits and punctuation are kept."""
    out = []
    for ch in nfc(s):
        if ch in "đĐ":
            out.append("d" if ch == "đ" else "D")
            continue
        base = unicodedata.normalize("NFD", ch)
        stripped = "".join(c for c in base if unicodedata.category(c) != "Mn")
        # Non-Latin symbols (≥, –, …) keep their own form so patterns can still see them
        out.append(stripped if stripped.isascii() else ch)
    return "".join(out)


def btc_normalize(s: Any) -> str:
    """Official normalization: NFC, lowercase, non letter/digit/whitespace -> space, collapse spaces. Diacritics kept."""
    text = nfc(s).lower()
    text = "".join(ch if (ch.isalnum() or ch.isspace()) else " " for ch in text)
    return re.sub(r"\s+", " ", text).strip()


# ── Telex repair ──────────────────────────────────────────────────────────────
# Typing with the Telex IME switched off leaves the key strokes in the text: 'tieenf' = 'tiền', 'nguwowif' = 'người'.

_TONE_KEYS = {"s": "́", "f": "̀", "r": "̉", "x": "̃", "j": "̣"}
_VOWELS = set("aăâeêioôơuưy")
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
# Vocabulary sources: training questions and the rulebook only (never test questions, Rule 6)
_VOCAB_FILES = ["train/questions.jsonl", "train/rulebook.md"]
_TOKEN_RE = re.compile(r"[^\W\d_]+", re.UNICODE)


def _telex_shape(word: str) -> Optional[Tuple[str, Optional[str]]]:
    """Apply Telex vowel/consonant keys; return (base word without tone, tone mark or None)."""
    w = word
    tone = None
    # The tone key may be typed at the end ('tieenf') or right after the vowel ('nguwowif', 'laij')
    for i in range(len(w) - 1, 0, -1):
        if w[i] in _TONE_KEYS and any(c in "aeiouyw" for c in w[:i]):
            rest = w[:i] + w[i + 1:]
            # Only treat it as tone key if nothing after it is a vowel modifier sequence left unprocessed
            tone = _TONE_KEYS[w[i]]
            w = rest
            break
    w = w.replace("dd", "đ")
    for src, dst in (("uwow", "ươ"), ("uow", "ươ"), ("aa", "â"), ("aw", "ă"), ("ee", "ê"), ("oo", "ô"),
                     ("ow", "ơ"), ("uw", "ư")):
        w = w.replace(src, dst)
    w = w.replace("w", "ư") if w.startswith("w") else w
    if "w" in w:
        return None
    return w, tone


def _with_tone(base: str, tone: str) -> List[str]:
    """All candidates that put the tone mark on one of the vowels of base."""
    out = []
    for i, ch in enumerate(base):
        if ch in _VOWELS:
            out.append(nfc(base[: i + 1] + tone + base[i + 1:]))
    return out


@lru_cache(maxsize=1)
def _vocab() -> FrozenSet[str]:
    words = set()
    for rel in _VOCAB_FILES:
        path = os.path.join(_REPO_ROOT, rel)
        if not os.path.exists(path):
            continue
        with open(path, "r", encoding="utf-8") as f:
            for tok in _TOKEN_RE.findall(nfc(f.read()).lower()):
                if tok != strip_accents(tok):
                    words.add(tok)
    return frozenset(words)


@lru_cache(maxsize=20000)
def repair_telex(token: str) -> str:
    """'tieenf' -> 'tiền', 'thangs' -> 'tháng'; unchanged unless the decoded word is a known Vietnamese word."""
    vocab = _vocab()
    if not token or token in vocab or not token.isalpha():
        return token
    # Old/new tone placement ('hoà' / 'hòa') are both accepted through the candidates
    shaped = _telex_shape(token)
    if shaped is None:
        return token
    base, tone = shaped
    if tone is None:
        return base if base != token and base in vocab else token
    for cand in _with_tone(base, tone):
        if cand in vocab:
            return cand
    return token


# ── Teencode / abbreviations ──────────────────────────────────────────────────
# token (lowercase, after Telex repair) -> replacement. Only standalone tokens are replaced, so codes such as
# 'SP0042' or product words are untouched. Ambiguous short tokens are handled by _contextual() below.
_TEENCODE: Dict[str, str] = {
    "bnhiu": "bao nhiêu", "bnhieu": "bao nhiêu", "bnhiêu": "bao nhiêu", "bnh": "bao nhiêu", "bn": "bao nhiêu",
    "dthu": "doanh thu", "đthu": "doanh thu",
    "dc": "được", "đc": "được", "đươc": "được", "dk": "được",
    "ko": "không", "hok": "không", "hông": "không", "khg": "không",
    "trc": "trước", "j": "gì", "z": "vậy", "vs": "với",
    "sp": "sản phẩm", "nv": "nhân viên", "hnay": "hôm nay", "hqua": "hôm qua",
    "mik": "mình", "mk": "mình", "lun": "luôn", "ntn": "như thế nào", "nhìu": "nhiều",
}
# 'nhiu' means 'nhiều' before a comparative / superlative, otherwise 'nhiêu' (bao nhiêu, có nhiêu)
_MANY_NEXT = {"nhất", "nhat", "hơn", "hon", "tiền", "tien", "đơn", "don", "lần", "lan"}
_TIME_NEXT = {"trước", "truoc", "trc", "này", "nay", "sau", "rồi", "roi"}


def _contextual(tok: str, prev: str, nxt: str) -> str:
    if tok == "nhiu":
        return "nhiều" if nxt in _MANY_NEXT else "nhiêu"
    if tok == "th" and (nxt in _TIME_NEXT or nxt.isdigit()):
        return "tháng"
    if tok == "r" and prev and not prev.isdigit():
        return "rồi"
    if tok == "v" and not nxt:
        return "vậy"
    if tok == "k" and not prev.isdigit():
        return "không"
    if tok == "e" and not prev:
        return "em"
    return _TEENCODE.get(tok, tok)


def normalize_question(q: str) -> Tuple[str, str]:
    """Return (text, ascii): text = NFC, lowercase, Telex-repaired, teencode expanded (original accents kept,
    accent-less words stay accent-less); ascii = accent-stripped text, used for regex matching."""
    text = nfc(q).lower().replace(" ", " ")
    parts = re.split(r"(\w+)", text)
    word_pos = [i for i, p in enumerate(parts) if p and re.match(r"\w", p)]
    repaired = {i: (repair_telex(parts[i]) if parts[i].isalpha() else parts[i]) for i in word_pos}
    for n, i in enumerate(word_pos):
        prev = repaired[word_pos[n - 1]] if n > 0 else ""
        nxt = repaired[word_pos[n + 1]] if n + 1 < len(word_pos) else ""
        parts[i] = _contextual(repaired[i], prev, nxt)
    text = re.sub(r"[ 	]+", " ", "".join(parts)).strip()
    return text, strip_accents(text)
