# -*- coding: utf-8 -*-
"""把论文 markdown 转成 .docx（字体宋体/Times New Roman，标题分级加粗，表格 Table Grid）。

用法：python scripts/md_to_docx.py <输入.md> <输出.docx>
"""
import re
import sys
import docx
from docx.shared import Pt
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.oxml import OxmlElement


def set_run_font(run, sz_pt, bold):
    run.font.name = 'Times New Roman'
    run.font.size = Pt(sz_pt)
    run.bold = bold
    rPr = run._element.get_or_add_rPr()
    rFonts = rPr.find(qn('w:rFonts'))
    if rFonts is None:
        rFonts = OxmlElement('w:rFonts')
        rPr.insert(0, rFonts)
    rFonts.set(qn('w:ascii'), 'Times New Roman')
    rFonts.set(qn('w:hAnsi'), 'Times New Roman')
    rFonts.set(qn('w:eastAsia'), '宋体')


def add_para(doc, text, sz_pt, bold, before=0, after=4, center=False, superscript_cite=True):
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


def parse_table_row(line):
    cells = [c.strip() for c in line.strip().strip('|').split('|')]
    return cells


def add_table(doc, rows):
    n_rows = len(rows)
    n_cols = len(rows[0])
    tbl = doc.add_table(rows=n_rows, cols=n_cols)
    tbl.style = 'Table Grid'
    for i, row in enumerate(rows):
        for j, val in enumerate(row):
            if j < n_cols:
                tbl.rows[i].cells[j].text = val
    return tbl


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
            # 去掉分隔行（---）
            data_rows = [r for r in rows if not all(re.fullmatch(r'[-: ]+', c or '') for c in r)]
            add_table(doc, data_rows)
            continue

        # 标题 / 小节
        if stripped.startswith('# '):
            add_para(doc, stripped[2:], 16, True, before=0, after=6, center=True)
        elif stripped.startswith('## '):
            add_para(doc, stripped[3:], 14, True, before=12, after=6)
        elif stripped.startswith('### '):
            add_para(doc, stripped[4:], 12.5, True, before=8, after=4)
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
