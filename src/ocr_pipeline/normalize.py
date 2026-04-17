from __future__ import annotations

import re
import unicodedata


def normalize_text(text: str) -> str:
    out = text.replace("\r\n", "\n").replace("\r", "\n")
    out = unicodedata.normalize("NFC", out)
    out = re.sub(r"[ \t]+", " ", out)
    out = re.sub(r"\n{3,}", "\n\n", out)
    return out.strip()


def strip_accents_lower(text: str) -> str:
    normalized = unicodedata.normalize("NFD", text)
    no_marks = "".join(ch for ch in normalized if unicodedata.category(ch) != "Mn")
    return no_marks.lower()
