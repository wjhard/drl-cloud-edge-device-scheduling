"""Build docs/技术报告.docx with native Microsoft Word COM automation."""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import pythoncom
import win32com.client


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "docs" / "技术报告.md"
DEFAULT_OUTPUT = ROOT / "docs" / "技术报告.docx"
DEFAULT_TEMPLATE = ROOT / "docs" / "references" / "操作系统开源创新大赛项目说明书_官方模板.docx"

WD_ALIGN_LEFT = 0
WD_ALIGN_CENTER = 1
WD_ALIGN_RIGHT = 2
WD_ALIGN_JUSTIFY = 3
WD_BREAK_PAGE = 7
WD_BREAK_SECTION_NEXT_PAGE = 2
WD_COLLAPSE_END = 0
WD_FORMAT_DOCX = 16
WD_EXPORT_PDF = 17
WD_ORIENT_PORTRAIT = 0
WD_PAPER_A4 = 7
WD_LINE_SPACE_1PT5 = 1
WD_STATISTIC_PAGES = 2
WD_STYLE_NORMAL = -1
WD_STYLE_HEADING_1 = -2
WD_STYLE_HEADING_2 = -3
WD_STYLE_HEADING_3 = -4
WD_TEXTURE_NONE = 0
WD_COLOR_AUTOMATIC = -16777216
WD_ROW_HEIGHT_AUTO = 0
WD_AUTOFIT_WINDOW = 2
WD_PAGE_NUMBER_ARABIC = 0
WD_PAGE_NUMBER_LOWERCASE_ROMAN = 2
WD_BORDER_TOP = -1
WD_BORDER_LEFT = -2
WD_BORDER_BOTTOM = -3
WD_BORDER_RIGHT = -4
WD_BORDER_HORIZONTAL = -5
WD_BORDER_VERTICAL = -6
WD_LINE_STYLE_NONE = 0
WD_LINE_STYLE_SINGLE = 1
WD_LINE_WIDTH_050PT = 4
WD_LINE_WIDTH_150PT = 12
WD_TAB_ALIGN_CENTER = 1
WD_TAB_ALIGN_RIGHT = 2
WD_TAB_LEADER_SPACES = 0

TABLE_CAPTIONS = [
    "默认异构资源参数",
    "赛题环节与实现证据对应关系",
    "项目目录与模块职责",
    "观测字段及归一化策略",
    "关键技术演进与性能变化",
    "未采纳方案及消融结论",
    "最终搜索阶段的不变量与保障机制",
    "最终方案五次重复评测结果",
    "计算量对齐对比结果",
    "MILP 最优解距离对比",
    "结构化泛化评测结果",
    "复现配置与用途",
    "关键结论与原始证据索引",
    "关键代码索引",
]


def rgb(red: int, green: int, blue: int) -> int:
    return red + green * 256 + blue * 65536


def end_range(doc):
    return doc.Range(doc.Content.End - 1, doc.Content.End - 1)


def set_font(font, east_asia: str, ascii_name: str, size: float, bold: bool = False) -> None:
    font.Name = ascii_name
    font.NameAscii = ascii_name
    font.NameFarEast = east_asia
    font.NameOther = ascii_name
    font.Size = size
    font.Bold = -1 if bold else 0


def add_paragraph(
    doc,
    text: str = "",
    *,
    style=None,
    alignment: int = WD_ALIGN_JUSTIFY,
    first_line: float = 24,
    space_before: float = 0,
    space_after: float = 6,
    keep_with_next: bool = False,
    east_asia_font: str = "宋体",
    ascii_font: str = "Times New Roman",
    size: float = 12,
    bold: bool = False,
):
    rng = end_range(doc)
    start = int(rng.Start)
    rng.InsertAfter(text + "\r")
    paragraph = doc.Range(start, start + max(1, len(text))).Paragraphs.Item(1)
    paragraph.Range.Style = WD_STYLE_NORMAL if style is None else style
    fmt = paragraph.Format
    fmt.Alignment = alignment
    fmt.FirstLineIndent = first_line
    fmt.SpaceBefore = space_before
    fmt.SpaceAfter = space_after
    fmt.LineSpacingRule = WD_LINE_SPACE_1PT5
    fmt.KeepWithNext = -1 if keep_with_next else 0
    try:
        fmt.WordWrap = -1
    except Exception:
        pass
    set_font(paragraph.Range.Font, east_asia_font, ascii_font, size, bold)
    trailing = doc.Paragraphs.Item(doc.Paragraphs.Count)
    if int(trailing.Range.Start) > int(paragraph.Range.Start):
        trailing.Range.Style = WD_STYLE_NORMAL
        trailing.Format.OutlineLevel = 10
        trailing.Range.ListFormat.RemoveNumbers()
    return paragraph


def add_heading(doc, text: str, level: int) -> None:
    styles = {1: WD_STYLE_HEADING_1, 2: WD_STYLE_HEADING_2, 3: WD_STYLE_HEADING_3}
    paragraph = add_paragraph(
        doc,
        text,
        style=styles[level],
        alignment=WD_ALIGN_LEFT,
        first_line=0,
        space_before=10 if level > 1 else 0,
        space_after=8,
        keep_with_next=True,
        east_asia_font="黑体",
        ascii_font="Arial",
        size={1: 16, 2: 14, 3: 12}[level],
        bold=True,
    )
    paragraph.Range.ListFormat.RemoveNumbers()
    paragraph.Format.OutlineLevel = level


def clean_inline(text: str) -> str:
    text = re.sub(r"!\[([^]]*)\]\([^)]+\)", r"\1", text)
    text = re.sub(r"\[([^]]+)\]\(([^)]+)\)", r"\1（\2）", text)
    text = text.replace("**", "").replace("__", "")
    text = text.replace("`", "")
    return text


def split_table_row(line: str) -> list[str]:
    cells = [clean_inline(cell.strip()) for cell in line.strip().strip("|").split("|")]
    return ["−" + cell[1:] if re.fullmatch(r"-\d+(?:\.\d+)?", cell) else cell for cell in cells]


def is_separator(cells: list[str]) -> bool:
    return bool(cells) and all(re.fullmatch(r":?-{3,}:?", cell.replace(" ", "")) for cell in cells)


def style_table(table, header: bool = True) -> None:
    table.Range.Style = WD_STYLE_NORMAL
    table.Range.Font.NameFarEast = "宋体"
    table.Range.Font.Name = "Times New Roman"
    table.Range.Font.Size = 9.5
    table.Range.ParagraphFormat.FirstLineIndent = 0
    table.Range.ParagraphFormat.SpaceAfter = 2
    table.Range.ParagraphFormat.LineSpacingRule = 0
    table.Range.ParagraphFormat.Alignment = WD_ALIGN_CENTER
    try:
        table.Range.ParagraphFormat.WordWrap = -1
    except Exception:
        pass
    table.Rows.AllowBreakAcrossPages = False
    table.AutoFitBehavior(WD_AUTOFIT_WINDOW)
    for border_type in (
        WD_BORDER_TOP,
        WD_BORDER_LEFT,
        WD_BORDER_BOTTOM,
        WD_BORDER_RIGHT,
        WD_BORDER_HORIZONTAL,
        WD_BORDER_VERTICAL,
    ):
        table.Borders.Item(border_type).LineStyle = WD_LINE_STYLE_NONE
    for border_type in (WD_BORDER_TOP, WD_BORDER_BOTTOM):
        border = table.Borders.Item(border_type)
        border.LineStyle = WD_LINE_STYLE_SINGLE
        border.LineWidth = WD_LINE_WIDTH_150PT
        border.Color = WD_COLOR_AUTOMATIC
    if header:
        row = table.Rows.Item(1)
        row.Range.Bold = -1
        row.Range.Font.NameFarEast = "黑体"
        row.Range.Shading.BackgroundPatternColor = WD_COLOR_AUTOMATIC
        border = row.Borders.Item(WD_BORDER_BOTTOM)
        border.LineStyle = WD_LINE_STYLE_SINGLE
        border.LineWidth = WD_LINE_WIDTH_050PT
        border.Color = WD_COLOR_AUTOMATIC
        row.HeadingFormat = True
    for index in range(2 if header else 1, table.Rows.Count + 1):
        table.Rows.Item(index).Range.Shading.BackgroundPatternColor = WD_COLOR_AUTOMATIC


def add_table(doc, rows: list[list[str]], table_number: int) -> None:
    caption = TABLE_CAPTIONS[table_number - 1]
    add_paragraph(
        doc,
        f"表 {table_number}  {caption}",
        alignment=WD_ALIGN_CENTER,
        first_line=0,
        space_before=6,
        space_after=4,
        keep_with_next=True,
        east_asia_font="宋体",
        ascii_font="Times New Roman",
        size=10.5,
    )
    column_count = max(len(row) for row in rows)
    table = doc.Tables.Add(end_range(doc), len(rows), column_count)
    for row_index, row in enumerate(rows, 1):
        for column_index in range(1, column_count + 1):
            table.Cell(row_index, column_index).Range.Text = row[column_index - 1] if column_index <= len(row) else ""
    style_table(table)
    header = rows[0]
    if column_count >= 7 and any("Best-of-64" in cell for cell in header):
        # Keep dense repeated-evaluation values on one line without widening the page.
        table.Range.Font.Size = 8.5
    width_map = {
        "阶段": [68, 96, 58, 96, 122],
        "赛题环节": [62, 112, 136, 130],
        "不变量": [92, 130, 218],
        "场景组": [88, 82, 90, 78, 102],
        "重复": [43, 68, 66, 78, 60, 55, 70],
    }
    widths = width_map.get(header[0])
    if widths and len(widths) == column_count:
        table.AllowAutoFit = False
        for index, width in enumerate(widths, 1):
            table.Columns.Item(index).Width = width
    doc.Range(table.Range.End, table.Range.End).InsertAfter("\r")


def add_code_block(doc, code: str, language: str) -> None:
    table = doc.Tables.Add(end_range(doc), 1, 1)
    prefix = f"[{language}]\r" if language else ""
    table.Cell(1, 1).Range.Text = prefix + code.rstrip() + "\r"
    table.Range.Style = WD_STYLE_NORMAL
    table.Borders.Enable = 1
    table.TopPadding = 6
    table.BottomPadding = 6
    table.LeftPadding = 8
    table.RightPadding = 8
    table.Range.Shading.Texture = WD_TEXTURE_NONE
    table.Range.Shading.BackgroundPatternColor = rgb(242, 242, 242)
    set_font(table.Range.Font, "等线", "Consolas", 9)
    table.Range.ParagraphFormat.FirstLineIndent = 0
    table.Range.ParagraphFormat.SpaceAfter = 0
    doc.Range(table.Range.End, table.Range.End).InsertAfter("\r")


def add_equation(doc, expression: str, equation_number: int) -> None:
    # A borderless layout table keeps the equation number outside the OMath
    # range. This avoids Word extending a built-up equation into the following
    # paragraph for expressions that contain operators such as min/max.
    table = doc.Tables.Add(end_range(doc), 1, 3)
    table.AllowAutoFit = False
    for index, width in enumerate((55, 338, 55), 1):
        table.Columns.Item(index).Width = width
    table.Borders.Enable = 0
    for border_type in (
        WD_BORDER_TOP,
        WD_BORDER_LEFT,
        WD_BORDER_BOTTOM,
        WD_BORDER_RIGHT,
        WD_BORDER_HORIZONTAL,
        WD_BORDER_VERTICAL,
    ):
        table.Borders.Item(border_type).LineStyle = WD_LINE_STYLE_NONE
    table.TopPadding = 2
    table.BottomPadding = 2
    table.LeftPadding = 0
    table.RightPadding = 0
    table.Rows.AllowBreakAcrossPages = False
    table.Range.Style = WD_STYLE_NORMAL
    table.Range.ParagraphFormat.FirstLineIndent = 0
    table.Range.ParagraphFormat.SpaceBefore = 4
    table.Range.ParagraphFormat.SpaceAfter = 4
    table.Range.ParagraphFormat.LineSpacingRule = WD_LINE_SPACE_1PT5
    set_font(table.Range.Font, "宋体", "Cambria Math", 11.5, False)

    formula_cell = table.Cell(1, 2)
    formula_cell.Range.Text = expression
    formula_cell.Range.ParagraphFormat.Alignment = WD_ALIGN_CENTER
    equation_range = doc.Range(formula_cell.Range.Start, formula_cell.Range.End - 1)
    equation_range.Font.Name = "Cambria Math"
    equation_range.Font.NameAscii = "Cambria Math"
    equation_range.Font.NameOther = "Cambria Math"
    equation_range.Font.Size = 11.5
    math_range = equation_range.OMaths.Add(equation_range)
    math_range.OMaths.Item(1).BuildUp()

    number_cell = table.Cell(1, 3)
    number_cell.Range.Text = f"（{equation_number}）"
    number_cell.Range.ParagraphFormat.Alignment = WD_ALIGN_RIGHT
    set_font(number_cell.Range.Font, "宋体", "Times New Roman", 11.5, False)
    doc.Range(table.Range.End, table.Range.End).InsertAfter("\r")


def add_figure(doc, alt_text: str, relative_path: str, figure_number: int) -> None:
    path = ROOT / relative_path
    if not path.is_file():
        raise FileNotFoundError(f"figure not found: {relative_path}")
    paragraph = add_paragraph(doc, "", alignment=WD_ALIGN_CENTER, first_line=0, space_before=6, space_after=2)
    shape = doc.InlineShapes.AddPicture(str(path.resolve()), False, True, paragraph.Range)
    max_width = 440.0
    if shape.Width > max_width:
        ratio = max_width / float(shape.Width)
        shape.Width = max_width
        shape.Height = float(shape.Height) * ratio
    caption_text = re.sub(r"^图\s*\d+\s*", "", alt_text).strip()
    add_paragraph(
        doc,
        f"图 {figure_number}  {caption_text}",
        alignment=WD_ALIGN_CENTER,
        first_line=0,
        space_after=8,
        east_asia_font="宋体",
        ascii_font="Times New Roman",
        size=10.5,
    )


def add_list_item(doc, text: str, numbered: bool) -> None:
    paragraph = add_paragraph(doc, clean_inline(text), alignment=WD_ALIGN_JUSTIFY, first_line=0, space_after=2)
    paragraph.Format.LeftIndent = 24
    paragraph.Format.FirstLineIndent = -18
    if numbered:
        paragraph.Range.ListFormat.ApplyNumberDefault()
    else:
        paragraph.Range.ListFormat.ApplyBulletDefault()


def configure_styles(doc) -> None:
    normal = doc.Styles.Item(WD_STYLE_NORMAL)
    set_font(normal.Font, "宋体", "Times New Roman", 12)
    normal.ParagraphFormat.Alignment = WD_ALIGN_JUSTIFY
    normal.ParagraphFormat.FirstLineIndent = 24
    normal.ParagraphFormat.SpaceAfter = 6
    normal.ParagraphFormat.LineSpacingRule = WD_LINE_SPACE_1PT5
    normal.ParagraphFormat.OutlineLevel = 10
    try:
        normal.ParagraphFormat.WordWrap = -1
    except Exception:
        pass
    for style_id, size, color in (
        (WD_STYLE_HEADING_1, 16, WD_COLOR_AUTOMATIC),
        (WD_STYLE_HEADING_2, 14, WD_COLOR_AUTOMATIC),
        (WD_STYLE_HEADING_3, 12, WD_COLOR_AUTOMATIC),
    ):
        style = doc.Styles.Item(style_id)
        set_font(style.Font, "黑体", "Arial", size, True)
        style.Font.Color = color
        style.ParagraphFormat.FirstLineIndent = 0
        style.ParagraphFormat.KeepWithNext = -1
        style.ParagraphFormat.PageBreakBefore = -1 if style_id == WD_STYLE_HEADING_1 else 0
        style.ParagraphFormat.OutlineLevel = {-2: 1, -3: 2, -4: 3}[style_id]


def configure_page_setup(doc) -> None:
    for section in doc.Sections:
        setup = section.PageSetup
        setup.PaperSize = WD_PAPER_A4
        setup.Orientation = WD_ORIENT_PORTRAIT
        setup.TopMargin = 72
        setup.BottomMargin = 68
        setup.LeftMargin = 79.2
        setup.RightMargin = 68
        setup.HeaderDistance = 32
        setup.FooterDistance = 32


def add_cover(doc) -> None:
    doc.Sections.Item(1).PageSetup.VerticalAlignment = 1
    add_paragraph(doc, "第三届中国研究生操作系统开源创新大赛", alignment=WD_ALIGN_CENTER, first_line=0, space_after=18, east_asia_font="黑体", ascii_font="Arial", size=18, bold=True)
    add_paragraph(doc, "暨开放原子大赛操作系统专项赛", alignment=WD_ALIGN_CENTER, first_line=0, space_after=32, east_asia_font="黑体", ascii_font="Arial", size=15, bold=True)
    add_paragraph(doc, "第 16 题：云—边—端异构计算资源调度", alignment=WD_ALIGN_CENTER, first_line=0, space_after=48, east_asia_font="黑体", ascii_font="Arial", size=14, bold=True)
    add_paragraph(doc, "基于深度强化学习的\v云—边—端异构计算资源管理调度方法", alignment=WD_ALIGN_CENTER, first_line=0, space_after=36, east_asia_font="方正小标宋简体", ascii_font="Arial", size=24, bold=True)
    add_paragraph(doc, "项 目 技 术 报 告", alignment=WD_ALIGN_CENTER, first_line=0, space_after=52, east_asia_font="黑体", ascii_font="Arial", size=20, bold=True)
    add_paragraph(doc, "参赛队伍：[匿名]", alignment=WD_ALIGN_CENTER, first_line=0, space_after=10, size=12)
    add_paragraph(doc, "二〇二六年七月", alignment=WD_ALIGN_CENTER, first_line=0, size=12)


def add_toc(doc):
    end_range(doc).InsertBreak(WD_BREAK_SECTION_NEXT_PAGE)
    doc.Sections.Item(2).PageSetup.VerticalAlignment = 0
    add_paragraph(doc, "目  录", alignment=WD_ALIGN_CENTER, first_line=0, space_before=10, space_after=18, east_asia_font="黑体", ascii_font="Arial", size=18, bold=True)
    toc = doc.TablesOfContents.Add(end_range(doc), True, 1, 3, False, "", True, True, "", True, True, False)
    end_range(doc).InsertAfter("\r")
    end_range(doc).InsertBreak(WD_BREAK_SECTION_NEXT_PAGE)
    return toc


def configure_headers_and_pages(doc) -> None:
    for index in range(1, doc.Sections.Count + 1):
        section = doc.Sections.Item(index)
        section.Headers.Item(1).LinkToPrevious = False
        section.Footers.Item(1).LinkToPrevious = False
        if index == 1:
            section.Headers.Item(1).Range.Text = ""
            section.Footers.Item(1).Range.Text = ""
            continue
        header = section.Headers.Item(1).Range
        header.Text = "基于深度强化学习的云—边—端异构计算资源管理调度方法"
        header.ParagraphFormat.Alignment = WD_ALIGN_CENTER
        set_font(header.Font, "宋体", "Times New Roman", 9)
        footer = section.Footers.Item(1)
        footer.Range.ParagraphFormat.Alignment = WD_ALIGN_CENTER
        numbers = footer.PageNumbers
        numbers.RestartNumberingAtSection = True
        numbers.StartingNumber = 1
        numbers.NumberStyle = WD_PAGE_NUMBER_LOWERCASE_ROMAN if index == 2 else WD_PAGE_NUMBER_ARABIC
        numbers.Add(WD_ALIGN_CENTER, True)


def parse_markdown(doc, markdown: str) -> dict:
    lines = markdown.splitlines()
    start = next(i for i, line in enumerate(lines) if line.strip() == "## 摘要")
    lines = lines[start:]
    index = 0
    paragraph_buffer: list[str] = []
    table_count = 0
    code_block_count = 0
    equation_count = 0
    figure_count = 0

    def flush() -> None:
        if not paragraph_buffer:
            return
        value = " ".join(item.strip() for item in paragraph_buffer if item.strip())
        paragraph_buffer.clear()
        if value:
            add_paragraph(doc, clean_inline(value))

    while index < len(lines):
        line = lines[index]
        stripped = line.strip()
        if not stripped:
            flush()
            index += 1
            continue
        if stripped.startswith("```"):
            flush()
            language = stripped[3:].strip()
            code: list[str] = []
            index += 1
            while index < len(lines) and not lines[index].strip().startswith("```"):
                code.append(lines[index])
                index += 1
            if language.lower() == "math":
                equation_count += 1
                add_equation(doc, " ".join(item.strip() for item in code), equation_count)
            else:
                add_code_block(doc, "\n".join(code), language)
                code_block_count += 1
            index += 1
            continue
        image_match = re.fullmatch(r"!\[([^]]+)\]\(([^)]+)\)", stripped)
        if image_match:
            flush()
            figure_count += 1
            add_figure(doc, image_match.group(1), image_match.group(2), figure_count)
            index += 1
            continue
        if stripped.startswith("|") and index + 1 < len(lines):
            separator = split_table_row(lines[index + 1])
            if is_separator(separator):
                flush()
                rows = [split_table_row(stripped)]
                index += 2
                while index < len(lines) and lines[index].strip().startswith("|"):
                    rows.append(split_table_row(lines[index]))
                    index += 1
                table_count += 1
                add_table(doc, rows, table_count)
                continue
        heading = re.match(r"^(#{1,3})\s+(.+)$", stripped)
        if heading:
            flush()
            raw_level = len(heading.group(1))
            text = clean_inline(heading.group(2))
            level = 1 if text == "摘要" else raw_level
            add_heading(doc, text, min(level, 3))
            index += 1
            continue
        list_match = re.match(r"^\s*(\d+\.|[-+*])\s+(.+)$", line)
        if list_match:
            flush()
            add_list_item(doc, list_match.group(2), list_match.group(1).endswith("."))
            index += 1
            continue
        if stripped.startswith(">"):
            flush()
            paragraph = add_paragraph(doc, clean_inline(stripped.lstrip("> ")), alignment=WD_ALIGN_JUSTIFY, first_line=0, space_after=5, east_asia_font="楷体", ascii_font="Times New Roman", size=10.5)
            paragraph.Format.LeftIndent = 24
            paragraph.Range.Shading.BackgroundPatternColor = rgb(245, 247, 250)
            index += 1
            continue
        if stripped == "---":
            flush()
            index += 1
            continue
        paragraph_buffer.append(stripped)
        index += 1
    flush()
    return {
        "data_table_count": table_count,
        "code_block_table_count": code_block_count,
        "equation_count": equation_count,
        "figure_count": figure_count,
    }


def render_key_pages(pdf_path: Path, output_dir: Path) -> list[Path]:
    import fitz

    output_dir.mkdir(parents=True, exist_ok=True)
    for old in output_dir.glob("*.png"):
        old.unlink()
    document = fitz.open(pdf_path)
    targets = {
        "01_封面": 0,
        "02_目录": 1,
    }
    searches = {
        "03_公式与模型": "第一章 赛题问题",
        "04_三线表": "默认异构资源",
        "05_奖励公式": "2.5 相对 HEFT 奖励",
        "06_残差公式": "3.5 残差式排序策略",
        "07_技术演进": "第三章 技术演进",
        "08_实验结果": "第六章 结果与证据",
    }
    for label, needle in searches.items():
        matching_pages = [i for i, page in enumerate(document) if needle in page.get_text()]
        if matching_pages:
            targets[label] = matching_pages[-1]
    outputs: list[Path] = []
    for label, page_number in targets.items():
        page = document.load_page(max(0, min(page_number, document.page_count - 1)))
        pix = page.get_pixmap(matrix=fitz.Matrix(1.65, 1.65), alpha=False)
        output = output_dir / f"{label}_第{page_number + 1}页.png"
        pix.save(output)
        outputs.append(output)
    document.close()
    return outputs


def build(input_path: Path, output_path: Path, template_path: Path, visible: bool) -> dict:
    markdown = input_path.read_text(encoding="utf-8")
    pdf_path = output_path.with_suffix(".pdf")
    screenshots_dir = output_path.parent / "screenshots"
    pythoncom.CoInitialize()
    word = None
    doc = None
    original_name = None
    original_initials = None
    try:
        word = win32com.client.DispatchEx("Word.Application")
        word.Visible = visible
        word.DisplayAlerts = 0
        original_name = word.UserName
        original_initials = word.UserInitials
        word.UserName = "Anonymous"
        word.UserInitials = "ANON"
        doc = word.Documents.Open(str(template_path.resolve()))
        doc.RemovePersonalInformation = True
        doc.Content.Delete()
        configure_styles(doc)
        configure_page_setup(doc)
        add_cover(doc)
        toc = add_toc(doc)
        stats = parse_markdown(doc, markdown)
        configure_page_setup(doc)
        configure_headers_and_pages(doc)
        for property_name in ("Author", "Last author"):
            try:
                doc.BuiltInDocumentProperties(property_name).Value = "Anonymous"
            except Exception:
                pass
        output_path.parent.mkdir(parents=True, exist_ok=True)
        doc.SaveAs2(str(output_path.resolve()), WD_FORMAT_DOCX)
        doc.Fields.Update()
        toc.Update()
        doc.Repaginate()
        page_count = int(doc.ComputeStatistics(WD_STATISTIC_PAGES))
        for property_name in ("Author", "Last author"):
            try:
                doc.BuiltInDocumentProperties(property_name).Value = "Anonymous"
            except Exception:
                pass
        doc.Save()
        doc.ExportAsFixedFormat(str(pdf_path.resolve()), WD_EXPORT_PDF)
        time.sleep(0.8)
        doc.Close(True)
        doc = None
        word.UserName = original_name
        word.UserInitials = original_initials
        word.Quit()
        word = None
        screenshots = render_key_pages(pdf_path, screenshots_dir)
        data_table_count = stats["data_table_count"]
        code_block_table_count = stats["code_block_table_count"]
        result = {
            "input": input_path.resolve().relative_to(ROOT).as_posix(),
            "template": template_path.resolve().relative_to(ROOT).as_posix(),
            "output_docx": output_path.resolve().relative_to(ROOT).as_posix(),
            "output_pdf": pdf_path.resolve().relative_to(ROOT).as_posix(),
            "page_count": page_count,
            "data_table_count": data_table_count,
            "code_block_table_count": code_block_table_count,
            "equation_layout_table_count": stats["equation_count"],
            "word_table_object_count": data_table_count + code_block_table_count + stats["equation_count"],
            "equation_count": stats["equation_count"],
            "figure_count": stats["figure_count"],
            "docx_size_bytes": output_path.stat().st_size,
            "pdf_size_bytes": pdf_path.stat().st_size,
            "screenshots": [path.resolve().relative_to(ROOT).as_posix() for path in screenshots],
        }
        summary_path = screenshots_dir / "generation_summary.json"
        summary_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return result
    finally:
        if doc is not None:
            try:
                doc.Close(False)
            except Exception:
                pass
        if word is not None:
            try:
                if original_name is not None:
                    word.UserName = original_name
                if original_initials is not None:
                    word.UserInitials = original_initials
            except Exception:
                pass
            try:
                word.Quit()
            except Exception:
                pass
        doc = None
        word = None
        pythoncom.CoUninitialize()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--template", type=Path, default=DEFAULT_TEMPLATE)
    parser.add_argument("--hidden", action="store_true")
    args = parser.parse_args()
    if not args.template.is_file():
        raise FileNotFoundError(f"official template not found: {args.template}")
    build(args.input, args.output, args.template, visible=not args.hidden)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
