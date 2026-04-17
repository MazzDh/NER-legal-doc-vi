from ocr_pipeline.layout_filter import filter_layout_blocks
from ocr_pipeline.types import TextBlock


def test_layout_filter_keeps_title_and_removes_signature_noise():
    blocks = [
        TextBlock(
            text="QUYET DINH VE VIEC PHE DUYET DE AN",
            bbox=(40.0, 60.0, 500.0, 95.0),
            confidence=0.99,
            source="ocr",
            page_index=0,
        ),
        TextBlock(
            text="Nguoi ky: Nguyen Van A",
            bbox=(420.0, 700.0, 820.0, 740.0),
            confidence=0.84,
            source="ocr",
            page_index=0,
        ),
        TextBlock(
            text="Noi dung van ban hanh chinh...",
            bbox=(40.0, 200.0, 820.0, 240.0),
            confidence=0.98,
            source="ocr",
            page_index=0,
        ),
    ]

    result = filter_layout_blocks(blocks=blocks, page_width=900, page_height=1000)

    assert result.title_candidates == ["QUYET DINH VE VIEC PHE DUYET DE AN"]
    assert len(result.removed_blocks) == 1
    assert "Nguoi ky" in result.removed_blocks[0].text
    assert len(result.kept_blocks) == 2
