from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class TextBlock:
    text: str
    bbox: tuple[float, float, float, float]
    confidence: float
    source: str  # digital_text | ocr
    page_index: int


@dataclass
class PageResult:
    page_index: int
    source: str  # digital_text | ocr
    text: str
    title_candidates: list[str] = field(default_factory=list)
    removed_blocks: list[TextBlock] = field(default_factory=list)
    kept_blocks: list[TextBlock] = field(default_factory=list)


@dataclass
class DocumentResult:
    input_path: str
    full_text: str
    title_candidates: list[str]
    pages: list[PageResult]
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        return payload
