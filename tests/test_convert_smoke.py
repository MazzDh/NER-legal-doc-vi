import shutil
import uuid
from pathlib import Path

import pandas as pd

from ner_pipeline.convert import ConverterConfig, convert_excel_to_bio_jsonl


def test_convert_smoke(monkeypatch):
    tmp_root = Path("E:/NER/tmp_smoke") / f"ner_test_{uuid.uuid4().hex[:8]}"
    tmp_root.mkdir(parents=True, exist_ok=True)
    src = tmp_root / "virtual.xlsx"
    out = tmp_root / "out"

    df = pd.DataFrame(
        [
            {
                "id": "1",
                "document_number": "115/NQ-HĐBCQG",
                "title": "Nghị quyết 115/NQ-HĐBCQG năm 2026",
                "url": "https://example.com",
                "legal_type": "Nghị quyết",
                "legal_sectors": "Bầu cử",
                "issuing_authority": "Hội đồng bầu cử quốc gia",
                "issuance_date": "29/01/2026",
                "signers": "Trần Thanh Mẫn:2140",
                "content": "Số: 115/NQ-HĐBCQG\nNghị quyết 115/NQ-HĐBCQG năm 2026\nHỘI ĐỒNG BẦU CỬ QUỐC GIA\nHà Nội, ngày 29 tháng 01 năm 2026\nNgười ký: Trần Thanh Mẫn",
            }
        ]
    )
    monkeypatch.setattr(pd, "read_excel", lambda _: df)

    try:
        outputs = convert_excel_to_bio_jsonl(ConverterConfig(xlsx_path=src, output_dir=out))
        assert outputs.train_path.exists()
        assert outputs.label_map_path.exists()
        assert outputs.summary_path.exists()
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)
