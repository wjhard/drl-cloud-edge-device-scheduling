"""Audit generated technical-report structure directly from DOCX OOXML."""

from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET


ROOT = Path(__file__).resolve().parents[1]
WORD_NAMESPACE = {
    "w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
}


def inspect_docx(path: Path) -> dict[str, int | str]:
    with zipfile.ZipFile(path) as archive:
        corrupt_member = archive.testzip()
        document = ET.fromstring(archive.read("word/document.xml"))

    tables = document.findall(".//w:tbl", WORD_NAMESPACE)
    data_tables = 0
    code_block_tables = 0
    for table in tables:
        rows = table.findall("./w:tr", WORD_NAMESPACE)
        column_count = max(
            (len(row.findall("./w:tc", WORD_NAMESPACE)) for row in rows),
            default=0,
        )
        if len(rows) == 1 and column_count == 1:
            code_block_tables += 1
        else:
            data_tables += 1

    drawings = document.findall(".//w:drawing", WORD_NAMESPACE)
    page_breaks = [
        node
        for node in document.findall(".//w:br", WORD_NAMESPACE)
        if node.get(f"{{{WORD_NAMESPACE['w']}}}type") == "page"
    ]
    return {
        "report": path.resolve().relative_to(ROOT).as_posix(),
        "docx_zip_corrupt_member": corrupt_member or "none",
        "word_table_object_count": len(tables),
        "data_table_count": data_tables,
        "code_block_table_count": code_block_tables,
        "drawing_object_count": len(drawings),
        "explicit_page_break_count": len(page_breaks),
        "size_bytes": path.stat().st_size,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", default="docs/技术报告.docx")
    parser.add_argument("--json-output")
    args = parser.parse_args()

    report = (ROOT / args.report).resolve()
    result = inspect_docx(report)
    output = json.dumps(result, ensure_ascii=False, indent=2)
    print(output)
    if args.json_output:
        target = (ROOT / args.json_output).resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(output + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
