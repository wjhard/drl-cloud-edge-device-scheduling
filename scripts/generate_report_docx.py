"""Generate the technical report DOCX through the native Microsoft Word COM API.

This script deliberately does not use python-docx. Microsoft Word creates,
paginates, updates, saves, and exports the document itself through pywin32.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import xml.etree.ElementTree as ET
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pythoncom
import win32com.client


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "docs" / "技术报告.md"
DEFAULT_OUTPUT = ROOT / "docs" / "技术报告.docx"
DEFAULT_PDF = ROOT / "docs" / "技术报告.pdf"
DEFAULT_SCREENSHOTS = ROOT / "docs" / "screenshots"

# Word constants are kept local so the script does not depend on makepy output.
WD_ALIGN_PARAGRAPH_LEFT = 0
WD_ALIGN_PARAGRAPH_CENTER = 1
WD_ALIGN_PARAGRAPH_RIGHT = 2
WD_ALIGN_PARAGRAPH_JUSTIFY = 3
WD_ALIGN_VERTICAL_TOP = 0
WD_ALIGN_VERTICAL_CENTER = 1
WD_AUTOFIT_WINDOW = 2
WD_BREAK_SECTION_NEXT_PAGE = 2
WD_COLLAPSE_END = 0
WD_COLOR_AUTOMATIC = -16777216
WD_EXPORT_FORMAT_PDF = 17
WD_FORMAT_DOCX = 12
WD_LINE_SPACE_1PT5 = 1
WD_PAGE_NUMBER_STYLE_ARABIC = 0
WD_PAGE_NUMBER_STYLE_LOWERCASE_ROMAN = 2
WD_PAPER_A4 = 7
WD_PREFERRED_WIDTH_POINTS = 3
WD_ROW_HEIGHT_AUTO = 0
WD_STATISTIC_PAGES = 2
WD_STYLE_HEADING_1 = -2
WD_STYLE_HEADING_2 = -3
WD_STYLE_HEADING_3 = -4
WD_STYLE_NORMAL = -1
WD_TEXTURE_NONE = 0


def rgb(red: int, green: int, blue: int) -> int:
    return red + green * 256 + blue * 65536


@dataclass
class InlineSpan:
    start: int
    end: int
    kind: str
    value: str | None = None


@dataclass
class BuildState:
    markdown_tables: list[object]
    heading_pages: dict[str, int]


def clean_markdown_text(text: str) -> tuple[str, list[InlineSpan]]:
    """Strip common inline Markdown while retaining formatting span offsets."""
    output: list[str] = []
    spans: list[InlineSpan] = []
    index = 0
    patterns = [
        ("link", re.compile(r"\[([^\]]+)\]\((https?://[^)]+)\)")),
        ("bold", re.compile(r"\*\*([^*]+)\*\*")),
        ("code", re.compile(r"`([^`]+)`")),
        ("italic", re.compile(r"(?<!\*)\*([^*]+)\*(?!\*)")),
    ]

    while index < len(text):
        candidates = []
        for kind, pattern in patterns:
            match = pattern.search(text, index)
            if match:
                candidates.append((match.start(), kind, match))
        if not candidates:
            output.append(text[index:])
            break
        _, kind, match = min(candidates, key=lambda item: item[0])
        output.append(text[index : match.start()])
        start = sum(len(part) for part in output)
        label = match.group(1)
        output.append(label)
        end = start + len(label)
        value = match.group(2) if kind == "link" else None
        spans.append(InlineSpan(start, end, kind, value))
        index = match.end()

    plain = "".join(output)
    plain = plain.replace("\\_", "_").replace("\\*", "*")
    return plain, spans


def split_table_row(line: str) -> list[str]:
    line = line.strip()
    if line.startswith("|"):
        line = line[1:]
    if line.endswith("|"):
        line = line[:-1]
    cells = re.split(r"(?<!\\)\|", line)
    return [cell.strip().replace("\\|", "|") for cell in cells]


def is_table_separator(cells: Iterable[str]) -> bool:
    cells = list(cells)
    return bool(cells) and all(re.fullmatch(r":?-{3,}:?", cell.strip()) for cell in cells)


def end_range(doc):
    return doc.Range(doc.Content.End - 1, doc.Content.End - 1)


def set_font(font, east_asia: str, ascii_font: str, size: float, bold: bool | None = None):
    font.NameFarEast = east_asia
    font.NameAscii = ascii_font
    font.NameOther = ascii_font
    font.Name = ascii_font
    font.Size = size
    if bold is not None:
        font.Bold = -1 if bold else 0


def apply_inline_spans(doc, paragraph_start: int, spans: list[InlineSpan]) -> None:
    for span in spans:
        if span.end <= span.start:
            continue
        target = doc.Range(paragraph_start + span.start, paragraph_start + span.end)
        if span.kind == "bold":
            target.Font.Bold = -1
        elif span.kind == "italic":
            target.Font.Italic = -1
        elif span.kind == "code":
            set_font(target.Font, "等线", "Consolas", 9.5)
            target.Shading.Texture = WD_TEXTURE_NONE
            target.Shading.BackgroundPatternColor = rgb(242, 242, 242)
        elif span.kind == "link" and span.value:
            doc.Hyperlinks.Add(Anchor=target, Address=span.value, TextToDisplay=target.Text)


def add_paragraph(
    doc,
    markdown_text: str,
    *,
    style_id: int = WD_STYLE_NORMAL,
    alignment: int = WD_ALIGN_PARAGRAPH_JUSTIFY,
    first_line_indent: float = 24.0,
    left_indent: float = 0.0,
    space_before: float = 0.0,
    space_after: float = 6.0,
    line_spacing_rule: int = WD_LINE_SPACE_1PT5,
    east_asia_font: str = "宋体",
    ascii_font: str = "Consolas",
    size: float = 12.0,
    bold: bool | None = None,
    keep_with_next: bool = False,
):
    plain, spans = clean_markdown_text(markdown_text)
    start = doc.Content.End - 1
    insertion = doc.Range(start, start)
    insertion.Text = plain + "\r"
    paragraph = doc.Range(start, start + len(plain) + 1).Paragraphs.Item(1)
    try:
        paragraph.Range.Style = doc.Styles.Item(style_id)
    except Exception:
        pass
    paragraph.Alignment = alignment
    paragraph.Format.FirstLineIndent = first_line_indent
    paragraph.Format.LeftIndent = left_indent
    paragraph.Format.RightIndent = 0
    paragraph.Format.SpaceBefore = space_before
    paragraph.Format.SpaceAfter = space_after
    paragraph.Format.LineSpacingRule = line_spacing_rule
    paragraph.Format.KeepWithNext = -1 if keep_with_next else 0
    set_font(paragraph.Range.Font, east_asia_font, ascii_font, size, bold)
    apply_inline_spans(doc, start, spans)
    return paragraph


def add_heading(doc, text: str, level: int):
    settings = {
        1: (WD_STYLE_HEADING_1, 16.0, 14.0, 8.0),
        2: (WD_STYLE_HEADING_2, 14.0, 12.0, 6.0),
        3: (WD_STYLE_HEADING_3, 12.0, 9.0, 4.0),
    }
    style_id, size, before, after = settings[level]
    paragraph = add_paragraph(
        doc,
        text,
        style_id=style_id,
        alignment=WD_ALIGN_PARAGRAPH_LEFT,
        first_line_indent=0,
        space_before=before,
        space_after=after,
        line_spacing_rule=0,
        east_asia_font="黑体",
        ascii_font="Arial",
        size=size,
        bold=True,
        keep_with_next=True,
    )
    if level == 1:
        paragraph.Format.PageBreakBefore = -1
    return paragraph


def add_list_item(doc, text: str, level: int, ordered: bool, marker: str):
    prefix = f"{marker} " if ordered else "• "
    return add_paragraph(
        doc,
        prefix + text,
        alignment=WD_ALIGN_PARAGRAPH_LEFT,
        first_line_indent=-14,
        left_indent=24 + level * 18,
        space_after=3,
    )


def add_equation(doc, lines: list[str]):
    expression = " ".join(line.strip() for line in lines if line.strip())
    expression = expression.rstrip("。").replace("\\qquad", "    ")
    expression = expression.replace("\\left", "").replace("\\right", "")
    expression = re.sub(r"\\frac\{([^{}]+)\}\{([^{}]+)\}", r"(\1)/(\2)", expression)
    replacements = {
        "\\max": "max",
        "\\min": "min",
        "\\sum": "Σ",
        "\\in": "∈",
        "\\times": "×",
        "\\theta": "θ",
        "\\pi": "π",
        "\\alpha": "α",
        "\\beta": "β",
        "\\Delta": "Δ",
        "\\widehat": "",
        "\\_": "_",
    }
    for source, target in replacements.items():
        expression = expression.replace(source, target)
    paragraph = add_paragraph(
        doc,
        expression,
        alignment=WD_ALIGN_PARAGRAPH_CENTER,
        first_line_indent=0,
        space_before=6,
        space_after=6,
        line_spacing_rule=0,
        east_asia_font="宋体",
        ascii_font="Cambria Math",
        size=11.5,
    )
    return paragraph


def style_table(table, header: bool = True) -> None:
    table.Borders.Enable = 1
    table.AllowAutoFit = True
    table.AutoFitBehavior(WD_AUTOFIT_WINDOW)
    table.Rows.Alignment = WD_ALIGN_PARAGRAPH_CENTER
    table.Rows.AllowBreakAcrossPages = False
    table.TopPadding = 3
    table.BottomPadding = 3
    table.LeftPadding = 4
    table.RightPadding = 4
    table.Range.ParagraphFormat.SpaceAfter = 0
    table.Range.ParagraphFormat.LineSpacingRule = 0
    set_font(table.Range.Font, "宋体", "Consolas", 9.5)
    if header:
        header_range = table.Rows.Item(1).Range
        table.Rows.Item(1).HeadingFormat = -1
        header_range.Bold = -1
        header_range.Shading.Texture = WD_TEXTURE_NONE
        header_range.Shading.ForegroundPatternColor = WD_COLOR_AUTOMATIC
        header_range.Shading.BackgroundPatternColor = rgb(221, 235, 247)
        header_range.Font.Color = rgb(31, 78, 121)


def set_table_column_widths(table, rows: list[list[str]]) -> None:
    column_count = max(len(row) for row in rows)
    usable_width = 440.0
    minimum_width = 48.0 if column_count <= 6 else 36.0
    weights = []
    for column_index in range(column_count):
        widths = []
        for row in rows:
            raw = row[column_index] if column_index < len(row) else ""
            plain, _ = clean_markdown_text(raw)
            widths.append(sum(2 if ord(char) > 127 else 1 for char in plain))
        weights.append(min(45.0, max(8.0, float(max(widths, default=8)))))
    extra_width = max(0.0, usable_width - minimum_width * column_count)
    weight_total = sum(weights) or 1.0
    table.AllowAutoFit = False
    for column_index, weight in enumerate(weights, 1):
        width = minimum_width + extra_width * weight / weight_total
        column = table.Columns.Item(column_index)
        column.PreferredWidthType = WD_PREFERRED_WIDTH_POINTS
        column.PreferredWidth = width


def add_markdown_table(doc, rows: list[list[str]]):
    if not rows:
        return None
    column_count = max(len(row) for row in rows)
    insertion = end_range(doc)
    table = doc.Tables.Add(insertion, len(rows), column_count)
    for row_index, row in enumerate(rows, 1):
        for column_index in range(1, column_count + 1):
            raw = row[column_index - 1] if column_index <= len(row) else ""
            plain, spans = clean_markdown_text(raw)
            cell_range = table.Cell(row_index, column_index).Range
            cell_range.Text = plain
            cell_start = cell_range.Start
            apply_inline_spans(doc, cell_start, spans)
    style_table(table, header=True)
    set_table_column_widths(table, rows)
    end = table.Range.End
    doc.Range(end, end).InsertAfter("\r")
    return table


def add_code_block(doc, code: str, language: str) -> None:
    insertion = end_range(doc)
    table = doc.Tables.Add(insertion, 1, 1)
    cell = table.Cell(1, 1)
    label = f"[{language}]\r" if language and language.lower() != "text" else ""
    cell.Range.Text = label + code.rstrip() + "\r"
    table.Borders.Enable = 1
    table.Rows.AllowBreakAcrossPages = False
    table.TopPadding = 6
    table.BottomPadding = 6
    table.LeftPadding = 8
    table.RightPadding = 8
    table.Range.Shading.Texture = WD_TEXTURE_NONE
    table.Range.Shading.ForegroundPatternColor = WD_COLOR_AUTOMATIC
    table.Range.Shading.BackgroundPatternColor = rgb(242, 242, 242)
    set_font(table.Range.Font, "等线", "Consolas", 9.0)
    table.Range.ParagraphFormat.SpaceAfter = 0
    table.Range.ParagraphFormat.LineSpacingRule = 0
    if label:
        label_range = doc.Range(cell.Range.Start, cell.Range.Start + len(label.rstrip()))
        label_range.Font.Bold = -1
        label_range.Font.Color = rgb(89, 89, 89)
    end = table.Range.End
    doc.Range(end, end).InsertAfter("\r")


def configure_styles(doc) -> None:
    normal = doc.Styles.Item(WD_STYLE_NORMAL)
    set_font(normal.Font, "宋体", "Consolas", 12.0)
    normal.ParagraphFormat.Alignment = WD_ALIGN_PARAGRAPH_JUSTIFY
    normal.ParagraphFormat.FirstLineIndent = 24
    normal.ParagraphFormat.SpaceAfter = 6
    normal.ParagraphFormat.LineSpacingRule = WD_LINE_SPACE_1PT5

    for style_id, size in ((WD_STYLE_HEADING_1, 16), (WD_STYLE_HEADING_2, 14), (WD_STYLE_HEADING_3, 12)):
        style = doc.Styles.Item(style_id)
        set_font(style.Font, "黑体", "Arial", size, True)
        style.Font.Color = rgb(31, 78, 121)
        style.ParagraphFormat.KeepWithNext = -1
        style.ParagraphFormat.FirstLineIndent = 0


def configure_page_setup(doc) -> None:
    for section in doc.Sections:
        setup = section.PageSetup
        setup.PaperSize = WD_PAPER_A4
        setup.TopMargin = 72
        setup.BottomMargin = 72
        setup.LeftMargin = 79.2
        setup.RightMargin = 72
        setup.HeaderDistance = 36
        setup.FooterDistance = 36


def add_cover(doc) -> None:
    doc.Sections.Item(1).PageSetup.VerticalAlignment = WD_ALIGN_VERTICAL_CENTER
    add_paragraph(
        doc,
        "第三届中国研究生操作系统开源创新大赛",
        alignment=WD_ALIGN_PARAGRAPH_CENTER,
        first_line_indent=0,
        space_after=22,
        line_spacing_rule=0,
        east_asia_font="黑体",
        ascii_font="Arial",
        size=18,
        bold=True,
    )
    add_paragraph(
        doc,
        "第 16 题：云—边—端异构计算资源调度",
        alignment=WD_ALIGN_PARAGRAPH_CENTER,
        first_line_indent=0,
        space_after=42,
        line_spacing_rule=0,
        east_asia_font="黑体",
        ascii_font="Arial",
        size=14,
        bold=True,
    )
    add_paragraph(
        doc,
        "基于深度强化学习的\n云—边—端异构计算资源管理调度方法".replace("\n", "\v"),
        alignment=WD_ALIGN_PARAGRAPH_CENTER,
        first_line_indent=0,
        space_after=30,
        line_spacing_rule=0,
        east_asia_font="方正小标宋简体",
        ascii_font="Arial",
        size=24,
        bold=True,
    )
    add_paragraph(
        doc,
        "技 术 报 告",
        alignment=WD_ALIGN_PARAGRAPH_CENTER,
        first_line_indent=0,
        space_after=54,
        line_spacing_rule=0,
        east_asia_font="黑体",
        ascii_font="Arial",
        size=20,
        bold=True,
    )
    add_paragraph(
        doc,
        "参赛队伍：[匿名]",
        alignment=WD_ALIGN_PARAGRAPH_CENTER,
        first_line_indent=0,
        space_after=12,
        line_spacing_rule=0,
        east_asia_font="宋体",
        ascii_font="Consolas",
        size=12,
    )
    add_paragraph(
        doc,
        "二〇二六年七月",
        alignment=WD_ALIGN_PARAGRAPH_CENTER,
        first_line_indent=0,
        line_spacing_rule=0,
        east_asia_font="宋体",
        ascii_font="Consolas",
        size=12,
    )


def insert_section_break(doc) -> None:
    end_range(doc).InsertBreak(WD_BREAK_SECTION_NEXT_PAGE)


def add_toc(doc):
    doc.Sections.Item(2).PageSetup.VerticalAlignment = WD_ALIGN_VERTICAL_TOP
    title = add_paragraph(
        doc,
        "目 录",
        alignment=WD_ALIGN_PARAGRAPH_CENTER,
        first_line_indent=0,
        space_before=12,
        space_after=18,
        line_spacing_rule=0,
        east_asia_font="黑体",
        ascii_font="Arial",
        size=18,
        bold=True,
    )
    title.Format.KeepWithNext = -1
    toc_range = end_range(doc)
    toc = doc.TablesOfContents.Add(
        Range=toc_range,
        UseHeadingStyles=True,
        UpperHeadingLevel=1,
        LowerHeadingLevel=3,
        UseFields=False,
        TableID="",
        RightAlignPageNumbers=True,
        IncludePageNumbers=True,
        AddedStyles="",
        UseHyperlinks=True,
        HidePageNumbersInWeb=True,
        UseOutlineLevels=True,
    )
    end_range(doc).InsertAfter("\r")
    insert_section_break(doc)
    return toc


def configure_page_numbers(doc) -> None:
    if doc.Sections.Count < 3:
        return
    first_footer = doc.Sections.Item(1).Footers.Item(1)
    first_footer.LinkToPrevious = False
    first_footer.Range.Text = ""

    toc_footer = doc.Sections.Item(2).Footers.Item(1)
    toc_footer.LinkToPrevious = False
    toc_footer.Range.ParagraphFormat.Alignment = WD_ALIGN_PARAGRAPH_CENTER
    toc_numbers = toc_footer.PageNumbers
    toc_numbers.RestartNumberingAtSection = True
    toc_numbers.StartingNumber = 1
    toc_numbers.NumberStyle = WD_PAGE_NUMBER_STYLE_LOWERCASE_ROMAN
    toc_numbers.Add(WD_ALIGN_PARAGRAPH_CENTER, True)

    body_footer = doc.Sections.Item(3).Footers.Item(1)
    body_footer.LinkToPrevious = False
    body_footer.Range.ParagraphFormat.Alignment = WD_ALIGN_PARAGRAPH_CENTER
    body_numbers = body_footer.PageNumbers
    body_numbers.RestartNumberingAtSection = True
    body_numbers.StartingNumber = 1
    body_numbers.NumberStyle = WD_PAGE_NUMBER_STYLE_ARABIC
    body_numbers.Add(WD_ALIGN_PARAGRAPH_CENTER, True)


def parse_markdown_into_word(doc, markdown: str) -> BuildState:
    lines = markdown.splitlines()
    start_index = next((i for i, line in enumerate(lines) if line.strip() == "## 摘要"), 0)
    lines = lines[start_index:]
    state = BuildState(markdown_tables=[], heading_pages={})
    index = 0
    paragraph_buffer: list[str] = []

    def flush_paragraph() -> None:
        if paragraph_buffer:
            text = " ".join(part.strip() for part in paragraph_buffer if part.strip())
            paragraph_buffer.clear()
            if text:
                add_paragraph(doc, text)

    while index < len(lines):
        line = lines[index]
        stripped = line.strip()

        if not stripped:
            flush_paragraph()
            index += 1
            continue

        if stripped.startswith("```"):
            flush_paragraph()
            language = stripped[3:].strip()
            code_lines = []
            index += 1
            while index < len(lines) and not lines[index].strip().startswith("```"):
                code_lines.append(lines[index])
                index += 1
            add_code_block(doc, "\n".join(code_lines), language)
            index += 1
            continue

        if stripped == r"\[":
            flush_paragraph()
            equation_lines = []
            index += 1
            while index < len(lines) and lines[index].strip() != r"\]":
                equation_lines.append(lines[index])
                index += 1
            add_equation(doc, equation_lines)
            index += 1
            continue

        if stripped.startswith("|") and index + 1 < len(lines):
            header = split_table_row(stripped)
            separator = split_table_row(lines[index + 1])
            if is_table_separator(separator):
                flush_paragraph()
                table_rows = [header]
                index += 2
                while index < len(lines) and lines[index].strip().startswith("|"):
                    table_rows.append(split_table_row(lines[index]))
                    index += 1
                table = add_markdown_table(doc, table_rows)
                if table is not None:
                    state.markdown_tables.append(table)
                continue

        heading_match = re.match(r"^(#{1,3})\s+(.+)$", stripped)
        if heading_match:
            flush_paragraph()
            level = len(heading_match.group(1))
            heading_text, _ = clean_markdown_text(heading_match.group(2))
            if heading_text == "摘要":
                level = 1
            add_heading(doc, heading_text, level)
            index += 1
            continue

        list_match = re.match(r"^(\s*)([-+*]|\d+\.)\s+(.+)$", line)
        if list_match:
            flush_paragraph()
            indentation = len(list_match.group(1).replace("\t", "    ")) // 4
            marker = list_match.group(2)
            add_list_item(doc, list_match.group(3), indentation, marker.endswith("."), marker)
            index += 1
            continue

        if stripped.startswith(">"):
            flush_paragraph()
            paragraph = add_paragraph(
                doc,
                stripped.lstrip("> "),
                alignment=WD_ALIGN_PARAGRAPH_LEFT,
                first_line_indent=0,
                left_indent=24,
                space_after=4,
            )
            paragraph.Range.Font.Italic = -1
            paragraph.Range.Font.Color = rgb(89, 89, 89)
            index += 1
            continue

        if stripped == "---":
            flush_paragraph()
            index += 1
            continue

        paragraph_buffer.append(stripped)
        index += 1

    flush_paragraph()
    return state


def find_heading_page(doc, prefix: str) -> int | None:
    for paragraph in doc.Paragraphs:
        text = paragraph.Range.Text.strip().replace("\r", "")
        if text.startswith(prefix) and int(paragraph.OutlineLevel) == 1:
            return int(paragraph.Range.Information(3))  # wdActiveEndPageNumber
    return None


def update_and_paginate(doc, toc) -> int:
    doc.Fields.Update()
    toc.Update()
    doc.Repaginate()
    return int(doc.ComputeStatistics(WD_STATISTIC_PAGES))


def render_pdf_screenshots(pdf_path: Path, screenshots_dir: Path, page_map: dict[str, int]) -> list[Path]:
    import fitz

    screenshots_dir.mkdir(parents=True, exist_ok=True)
    for old_file in screenshots_dir.glob("*.png"):
        old_file.unlink()
    pdf = fitz.open(pdf_path)
    outputs = []
    for label, page_number in page_map.items():
        safe_page = max(1, min(page_number, pdf.page_count))
        page = pdf.load_page(safe_page - 1)
        pixmap = page.get_pixmap(matrix=fitz.Matrix(1.8, 1.8), alpha=False)
        output = screenshots_dir / f"{label}_第{safe_page}页.png"
        pixmap.save(output)
        outputs.append(output)
    pdf.close()
    return outputs


def find_official_template() -> Path | None:
    exact_patterns = ("*初赛作品提交模板*.doc", "*初赛作品提交模板*.docx", "*初赛作品提交模板*.dotx")
    roots = [ROOT, ROOT.parent, Path.home() / "Desktop", Path.home() / "Downloads"]
    candidates = []
    for root in roots:
        if not root.exists():
            continue
        for pattern in exact_patterns:
            candidates.extend(path for path in root.rglob(pattern) if path.is_file())
    return sorted(set(candidates))[0] if candidates else None


def set_anonymous_document_metadata(doc) -> None:
    """Remove workstation identity from Office core properties."""
    for property_name in ("Author", "Last Save By"):
        try:
            doc.BuiltInDocumentProperties.Item(property_name).Value = "Anonymous"
        except Exception:
            # Some Word/template combinations expose Last Save By as read-only.
            pass


def sanitize_docx_core_properties(docx_path: Path) -> None:
    """Replace author fields directly in the OOXML package after Word closes it."""
    namespaces = {
        "cp": "http://schemas.openxmlformats.org/package/2006/metadata/core-properties",
        "dc": "http://purl.org/dc/elements/1.1/",
        "dcterms": "http://purl.org/dc/terms/",
        "dcmitype": "http://purl.org/dc/dcmitype/",
        "xsi": "http://www.w3.org/2001/XMLSchema-instance",
    }
    for prefix, uri in namespaces.items():
        ET.register_namespace(prefix, uri)
    core_properties_path = "docProps/core.xml"
    creator_tag = "{http://purl.org/dc/elements/1.1/}creator"
    last_modified_by_tag = (
        "{http://schemas.openxmlformats.org/package/2006/metadata/core-properties}lastModifiedBy"
    )
    temporary_path = docx_path.with_name(f".{docx_path.name}.anonymous.tmp")

    try:
        with zipfile.ZipFile(docx_path, "r") as source, zipfile.ZipFile(temporary_path, "w") as target:
            for item in source.infolist():
                data = source.read(item.filename)
                if item.filename == core_properties_path:
                    root = ET.fromstring(data)
                    for tag in (creator_tag, last_modified_by_tag):
                        element = root.find(tag)
                        if element is not None:
                            element.text = "Anonymous"
                    data = ET.tostring(root, encoding="utf-8", xml_declaration=True)
                target.writestr(item, data)
        temporary_path.replace(docx_path)
    finally:
        temporary_path.unlink(missing_ok=True)


def generate(input_path: Path, output_path: Path, template_path: Path | None, visible: bool) -> dict:
    pdf_path = output_path.with_suffix(".pdf")
    screenshots_dir = output_path.parent / "screenshots"
    markdown = input_path.read_text(encoding="utf-8")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    pythoncom.CoInitialize()
    word = None
    doc = None
    original_user_name = None
    original_user_initials = None
    try:
        word = win32com.client.DispatchEx("Word.Application")
        word.Visible = visible
        word.DisplayAlerts = 0
        original_user_name = word.UserName
        original_user_initials = word.UserInitials
        word.UserName = "Anonymous"
        word.UserInitials = "ANON"
        if template_path:
            doc = word.Documents.Open(str(template_path.resolve()))
            doc.Content.Delete()
        else:
            doc = word.Documents.Add()
        set_anonymous_document_metadata(doc)

        configure_styles(doc)
        configure_page_setup(doc)
        add_cover(doc)
        insert_section_break(doc)
        toc = add_toc(doc)
        state = parse_markdown_into_word(doc, markdown)
        configure_page_setup(doc)
        configure_page_numbers(doc)

        set_anonymous_document_metadata(doc)
        doc.SaveAs2(str(output_path.resolve()), FileFormat=WD_FORMAT_DOCX)
        page_count = update_and_paginate(doc, toc)
        set_anonymous_document_metadata(doc)
        doc.Save()

        first_table_page = (
            int(doc.Range(state.markdown_tables[0].Range.Start, state.markdown_tables[0].Range.Start).Information(3))
            if state.markdown_tables
            else 3
        )
        appendix_page = find_heading_page(doc, "附录 A") or find_heading_page(doc, "附录") or page_count
        page_map = {
            "01_封面": 1,
            "02_目录": 2,
            "03_正文表格": first_table_page,
            "04_附录": appendix_page,
        }

        doc.ExportAsFixedFormat(str(pdf_path.resolve()), WD_EXPORT_FORMAT_PDF)
        # Word is intentionally still open while screenshots are generated.
        screenshots = render_pdf_screenshots(pdf_path, screenshots_dir, page_map)
        time.sleep(1.0)

        doc.Close(True)
        doc = None
        if original_user_name is not None:
            word.UserName = original_user_name
        if original_user_initials is not None:
            word.UserInitials = original_user_initials
        word.Quit()
        word = None
        sanitize_docx_core_properties(output_path)

        result = {
            "input": str(input_path.resolve()),
            "template": str(template_path.resolve()) if template_path else None,
            "output_docx": str(output_path.resolve()),
            "output_pdf": str(pdf_path.resolve()),
            "page_count": page_count,
            "table_count": len(state.markdown_tables),
            "page_map": page_map,
            "screenshots": [str(path.resolve()) for path in screenshots],
            "docx_size_bytes": output_path.stat().st_size,
            "pdf_size_bytes": pdf_path.stat().st_size,
        }
        metadata_path = screenshots_dir / "generation_summary.json"
        metadata_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return result
    except Exception:
        if doc is not None:
            try:
                doc.Close(False)
            except Exception:
                pass
        if word is not None:
            try:
                word.Quit()
            except Exception:
                pass
        raise
    finally:
        if doc is not None:
            try:
                doc.Close(True)
            except Exception:
                pass
        if word is not None:
            try:
                if original_user_name is not None:
                    word.UserName = original_user_name
                if original_user_initials is not None:
                    word.UserInitials = original_user_initials
            except Exception:
                pass
            try:
                word.Quit()
            except Exception:
                pass
        doc = None
        word = None
        pythoncom.CoUninitialize()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--template", type=Path)
    parser.add_argument("--hidden", action="store_true", help="Keep Word hidden during generation.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    template = args.template
    if template is None:
        template = find_official_template()
    if template is not None and not template.exists():
        raise FileNotFoundError(f"Template not found: {template}")
    print(f"pywin32 Word COM input: {args.input.resolve()}")
    print(f"official template: {template.resolve() if template else 'not found; using general academic layout'}")
    generate(args.input, args.output, template, visible=not args.hidden)
    return 0


if __name__ == "__main__":
    sys.exit(main())
