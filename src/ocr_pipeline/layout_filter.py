from __future__ import annotations

from dataclasses import dataclass

from .normalize import strip_accents_lower
from .types import TextBlock

TITLE_KEYWORDS = [
    "quyet dinh",
    "nghi quyet",
    "thong bao",
    "to trinh",
    "cong van",
    "chi thi",
]

SIGNATURE_KEYWORDS = [
    "nguoi ky",
    "noi nhan",
    "ky ten",
    "kt.",
    "tl.",
    "tuq.",
    "chu tich",
    "pho chu tich",
]


@dataclass
class LayoutFilterConfig:
    title_zone_ratio: float = 0.35
    signature_zone_top_ratio: float = 0.55
    signature_zone_left_ratio: float = 0.45


@dataclass
class LayoutFilterResult:
    kept_blocks: list[TextBlock]
    removed_blocks: list[TextBlock]
    title_candidates: list[str]


def _is_title_candidate(block: TextBlock, page_width: float, page_height: float, cfg: LayoutFilterConfig) -> bool:
    x0, y0, x1, y1 = block.bbox
    if y1 > page_height * cfg.title_zone_ratio:
        return False
    raw = block.text.strip()
    if len(raw) < 6 or len(raw) > 180:
        return False
    folded = strip_accents_lower(raw)
    if any(key in folded for key in TITLE_KEYWORDS):
        return True
    letters = [ch for ch in raw if ch.isalpha()]
    if not letters:
        return False
    upper_ratio = sum(1 for ch in letters if ch.isupper()) / max(len(letters), 1)
    return upper_ratio >= 0.65 and (x1 - x0) >= page_width * 0.25


def _is_signature_or_stamp(block: TextBlock, page_width: float, page_height: float, cfg: LayoutFilterConfig) -> bool:
    x0, y0, x1, y1 = block.bbox
    folded = strip_accents_lower(block.text)
    in_signature_zone = y0 >= page_height * cfg.signature_zone_top_ratio and x0 >= page_width * cfg.signature_zone_left_ratio
    has_signature_keyword = any(key in folded for key in SIGNATURE_KEYWORDS)
    likely_stamp_noise = (
        in_signature_zone
        and len(block.text.strip()) <= 40
        and block.confidence < 0.85
    )
    return (in_signature_zone and has_signature_keyword) or likely_stamp_noise


def filter_layout_blocks(
    blocks: list[TextBlock],
    page_width: float,
    page_height: float,
    cfg: LayoutFilterConfig | None = None,
) -> LayoutFilterResult:
    config = cfg or LayoutFilterConfig()
    kept: list[TextBlock] = []
    removed: list[TextBlock] = []
    titles: list[str] = []

    for block in blocks:
        if _is_signature_or_stamp(block, page_width, page_height, config):
            removed.append(block)
            continue
        kept.append(block)
        if _is_title_candidate(block, page_width, page_height, config):
            titles.append(block.text.strip())

    # De-duplicate while preserving order.
    dedup_titles: list[str] = []
    seen: set[str] = set()
    for title in titles:
        key = strip_accents_lower(title)
        if key in seen:
            continue
        seen.add(key)
        dedup_titles.append(title)

    return LayoutFilterResult(kept_blocks=kept, removed_blocks=removed, title_candidates=dedup_titles)
