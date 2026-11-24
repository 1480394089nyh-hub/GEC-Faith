# -*- coding: utf-8 -*-
"""生成「反馈再生成」标注表（xlsx），供标注员回标新改句的忠实性。

- 读 refine 输出 `results/refine/重改结果_全量119.csv`。
- 用 multi 的 5 类真实标签补「上次真实标签」上下文。
- 新改句 == 上次改句 的行（模型没改）自动沿用上次 5 类标签；其余行留空给标注员标。
- 产出 `results/refine/标注表_反馈再生成_全量119.xlsx`，重跑即更新。
"""
import csv
import os

from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Alignment

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
REFINE = os.path.join(ROOT, "results", "refine", "重改结果_全量119.csv")
MULTI = os.path.join(ROOT, "results", "discriminator", "multi", "test_predictions.csv")
OUT = os.path.join(ROOT, "results", "refine", "标注表_反馈再生成_全量119.xlsx")


def load(p):
    with open(p, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def main():
    rows = load(REFINE)
    multi_true = {r["序号"]: r["真实标签"] for r in load(MULTI)}

    wb = Workbook()
    ws = wb.active
    ws.title = "反馈再生成标注"
    cols = ["序号", "句子序号", "模型", "诊断失败类型", "原句", "上次改句",
            "上次真实标签", "新改句", "是否变化", "新改句标注"]
    ws.append(cols)

    hdr_fill = PatternFill("solid", fgColor="D9E1F2")
    for c in ws[1]:
        c.font = Font(bold=True)
        c.fill = hdr_fill
        c.alignment = Alignment(horizontal="center", vertical="center")

    changed_fill = PatternFill("solid", fgColor="FFF2CC")  # 需要你标注的行
    n_changed = 0
    for r in rows:
        changed = r["新改句"].strip() != r["上次改句"].strip()
        label = "" if changed else multi_true.get(r["序号"], "")  # 未变则沿用上次标签
        if changed:
            n_changed += 1
        ws.append([
            int(r["序号"]), int(r["句子序号"]), r["模型"], r["诊断失败类型"],
            r["原句"], r["上次改句"], multi_true.get(r["序号"], ""),
            r["新改句"], "是" if changed else "否", label,
        ])
        if changed:
            row_idx = ws.max_row
            for c in ws[row_idx]:
                c.fill = changed_fill

    widths = {"A": 7, "B": 8, "C": 10, "D": 12, "E": 42, "F": 42,
              "G": 12, "H": 42, "I": 9, "J": 13}
    for col, w in widths.items():
        ws.column_dimensions[col].width = w
    for row in ws.iter_rows(min_row=2, min_col=5, max_col=8):
        for c in row:
            c.alignment = Alignment(wrap_text=True, vertical="top")
    for row in ws.iter_rows(min_row=2, min_col=9, max_col=10):
        for c in row:
            c.alignment = Alignment(horizontal="center", vertical="top")
    ws.freeze_panes = "A2"
    ws.auto_filter.ref = f"A1:J{ws.max_row}"

    wb.save(OUT)
    print(f"OK -> {OUT}")
    print(f"共 {len(rows)} 行；需要标注（新改句 != 上次改句）：{n_changed} 行")


if __name__ == "__main__":
    main()
