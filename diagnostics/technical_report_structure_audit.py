"""Audit generated technical-report structure directly from DOCX OOXML."""

from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET


ROOT = Path(__file__).resolve().parents[1]
WORD_NAMESPACE = {
    "w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main",
    "m": "http://schemas.openxmlformats.org/officeDocument/2006/math",
}


def border_value(parent, path: str) -> str | None:
    node = parent.find(path, WORD_NAMESPACE)
    if node is None:
        return None
    return node.get(f"{{{WORD_NAMESPACE['w']}}}val")


def is_three_line_table(table) -> bool:
    borders = table.find("./w:tblPr/w:tblBorders", WORD_NAMESPACE)
    if borders is None:
        return False
    if border_value(borders, "./w:top") != "single":
        return False
    if border_value(borders, "./w:bottom") != "single":
        return False
    for edge in ("left", "right", "insideH", "insideV"):
        if border_value(borders, f"./w:{edge}") not in (None, "nil", "none"):
            return False
    header_cells = table.findall("./w:tr[1]/w:tc", WORD_NAMESPACE)
    return bool(header_cells) and all(
        border_value(cell, "./w:tcPr/w:tcBorders/w:bottom") == "single"
        for cell in header_cells
    )


def inspect_docx(path: Path) -> dict[str, int | str]:
    with zipfile.ZipFile(path) as archive:
        corrupt_member = archive.testzip()
        document = ET.fromstring(archive.read("word/document.xml"))

    tables = document.findall(".//w:tbl", WORD_NAMESPACE)
    data_tables = 0
    code_block_tables = 0
    equation_layout_tables = 0
    three_line_tables = 0
    for table in tables:
        rows = table.findall("./w:tr", WORD_NAMESPACE)
        column_count = max(
            (len(row.findall("./w:tc", WORD_NAMESPACE)) for row in rows),
            default=0,
        )
        if table.find(".//m:oMath", WORD_NAMESPACE) is not None:
            equation_layout_tables += 1
        elif len(rows) == 1 and column_count == 1:
            code_block_tables += 1
        else:
            data_tables += 1
            if is_three_line_table(table):
                three_line_tables += 1

    drawings = document.findall(".//w:drawing", WORD_NAMESPACE)
    equations = document.findall(".//m:oMath", WORD_NAMESPACE)
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
        "three_line_table_count": three_line_tables,
        "non_three_line_data_table_count": data_tables - three_line_tables,
        "code_block_table_count": code_block_tables,
        "equation_layout_table_count": equation_layout_tables,
        "equation_object_count": len(equations),
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
