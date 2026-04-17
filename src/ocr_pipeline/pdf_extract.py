from __future__ import annotations

import re


def has_sufficient_text(text: str, min_chars: int = 80, min_words: int = 15) -> bool:
    cleaned = text.strip()
    if len(cleaned) < min_chars:
        return False
    words = re.findall(r"\S+", cleaned)
    return len(words) >= min_words
