# -*- coding: utf-8 -*-
"""把论文 markdown 转成 .docx。

格式约定（对齐目标期刊模板）：
- 标题用真实样式（Title / Heading 1 / 2 / 3），便于生成目录；字体黑体。
- 正文宋体 / Times New Roman，五号（10.5pt）。
- 表格三线表（顶线底线 1.5pt、表头下线 0.75pt、无竖线），表头跨页重复。
- [n] 引用上标；表题/图题用 "> " 前缀居中。

用法：python scripts/md_to_docx.py <输入.md> <输出.docx>
"""
import re
import sys
import docx
from docx.shared import Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml.ns import qn
from docx.oxml import OxmlElement


def set_run_font(run, sz_pt, bold, cjk='宋体'):
    run.font.name = 'Times New Roman'
    run.font.size = Pt(sz_pt)
    run.bold = bold
    run.font.color.rgb = RGBColor(0, 0, 0)
    rPr = run._element.get_or_add_rPr()
    rFonts = rPr.find(qn('w:rFonts'))
    if rFonts is None:
        rFonts = OxmlElement('w:rFonts')
        rPr.insert(0, rFonts)
    rFonts.set(qn('w:ascii'), 'Times New Roman')
    rFonts.set(qn('w:hAnsi'), 'Times New Roman')
    rFonts.set(qn('w:eastAsia'), cjk)


def add_para(doc, text, sz_pt, bold, before=0, after=4, center=False, superscript_cite=True):
    """普通正文段（Normal 样式）。"""
    p = doc.add_paragraph()
    pf = p.paragraph_format
    pf.space_before = Pt(before)
    pf.space_after = Pt(after)
    if center:
        pf.alignment = WD_ALIGN_PARAGRAPH.CENTER
    for part in re.split(r'(\*\*.*?\*\*|\[\d+\])', text):
        if not part:
            continue
        if part.startswith('**') and part.endswith('**') and len(part) > 4:
            run = p.add_run(part[2:-2])
            set_run_font(run, sz_pt, True)
        elif superscript_cite and re.fullmatch(r'\[\d+\]', part):
            run = p.add_run(part)
            set_run_font(run, sz_pt, bold)
            run.font.superscript = True
        else:
            run = p.add_run(part)
            set_run_font(run, sz_pt, bold)
    return p


def add_heading(doc, text, level, sz_pt, before=0, after=6, center=False):
    """标题段：套用真实样式（Title / Heading 1~3），字体黑体。"""
    style = {0: 'Title', 1: 'Heading 1', 2: 'Heading 2', 3: 'Heading 3'}[level]
    p = doc.add_paragraph(style=style)
    pf = p.paragraph_format
    pf.space_before = Pt(before)
    pf.space_after = Pt(after)
    if center:
        pf.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p.add_run(text)
    set_run_font(run, sz_pt, False, cjk='黑体')
    return p


def _set_cell_border_bottom(cell, sz):
    """给单元格设下边框（三线表的表头下线）。"""
    tcPr = cell._tc.get_or_add_tcPr()
    tcBorders = tcPr.find(qn('w:tcBorders'))
    if tcBorders is None:
        tcBorders = OxmlElement('w:tcBorders')
        tcPr.append(tcBorders)
    bottom = OxmlElement('w:bottom')
    bottom.set(qn('w:val'), 'single')
    bottom.set(qn('w:sz'), str(sz))
    bottom.set(qn('w:space'), '0')
    bottom.set(qn('w:color'), 'auto')
    tcBorders.append(bottom)


def _set_repeat_header(row):
    """表头行跨页重复。"""
    trPr = row._tr.get_or_add_trPr()
    if trPr.find(qn('w:tblHeader')) is None:
        tblHeader = OxmlElement('w:tblHeader')
        tblHeader.set(qn('w:val'), 'true')
        trPr.append(tblHeader)


def add_table(doc, rows):
    n_rows = len(rows)
    n_cols = len(rows[0])
    tbl = doc.add_table(rows=n_rows, cols=n_cols)
    tbl.alignment = WD_TABLE_ALIGNMENT.CENTER

    # 三线表：表格级只保留顶线/底线（粗 1.5pt），其余无
    tblPr = tbl._tbl.tblPr
    borders = OxmlElement('w:tblBorders')
    for edge, val, sz in [('top', 'single', '12'), ('bottom', 'single', '12'),
                          ('left', 'none', '0'), ('right', 'none', '0'),
                          ('insideH', 'none', '0'), ('insideV', 'none', '0')]:
        el = OxmlElement('w:' + edge)
        el.set(qn('w:val'), val)
        el.set(qn('w:sz'), sz)
        el.set(qn('w:space'), '0')
        el.set(qn('w:color'), 'auto')
        borders.append(el)
    tblPr.append(borders)

    for i, row in enumerate(rows):
        for j, val in enumerate(row):
            if j < n_cols:
                cell = tbl.rows[i].cells[j]
                cell.text = val
                if i == 0:
                    _set_cell_border_bottom(cell, 6)
                for p in cell.paragraphs:
                    p.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER
                    for run in p.runs:
                        set_run_font(run, 10.5, False)
    # 跨页重复表头
    _set_repeat_header(tbl.rows[0])
    return tbl


def parse_table_row(line):
    cells = [c.strip() for c in line.strip().strip('|').split('|')]
    return cells


def convert(md_path, docx_path):
    with open(md_path, encoding='utf-8') as f:
        lines = f.read().splitlines()

    doc = docx.Document()
    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        if not stripped:
            i += 1
            continue

        # 表格
        if stripped.startswith('|'):
            rows = []
            while i < len(lines) and lines[i].strip().startswith('|'):
                rows.append(parse_table_row(lines[i]))
                i += 1
            data_rows = [r for r in rows if not all(re.fullmatch(r'[-: ]+', c or '') for c in r)]
            add_table(doc, data_rows)
            continue

        # 标题 / 小节（真实标题样式）
        if stripped.startswith('# '):
            add_heading(doc, stripped[2:], 0, 16, before=0, after=6, center=True)
            # 英文标题：紧随其后的整行加粗，居中
            j = i + 1
            while j < len(lines) and not lines[j].strip():
                j += 1
            if j < len(lines):
                nt = lines[j].strip()
                if nt.startswith('**') and nt.endswith('**') and nt.count('**') == 2 and len(nt) > 4:
                    add_para(doc, nt[2:-2], 14, True, before=2, after=8, center=True)
                    i = j
        elif stripped.startswith('#### '):
            add_heading(doc, stripped[5:], 3, 10.5, before=6, after=3)
        elif stripped.startswith('## '):
            add_heading(doc, stripped[3:], 1, 14, before=12, after=6)
        elif stripped.startswith('### '):
            add_heading(doc, stripped[4:], 2, 12, before=8, after=4)
        # 表题/图题（居中）
        elif stripped.startswith('> '):
            add_para(doc, stripped[2:], 10.5, False, before=4, after=1, center=True)
        # 列表
        elif stripped.startswith('- '):
            add_para(doc, '• ' + stripped[2:], 10.5, False)
        elif re.match(r'^\d+\.\s', stripped):
            add_para(doc, stripped, 10.5, False)
        else:
            is_ref = bool(re.match(r'^\[\d+\]', stripped))
            add_para(doc, stripped, 10.5, False, superscript_cite=not is_ref)
        i += 1

    doc.save(docx_path)


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print('用法: python scripts/md_to_docx.py <输入.md> <输出.docx>')
        sys.exit(1)
    convert(sys.argv[1], sys.argv[2])
    print('saved:', sys.argv[2])
