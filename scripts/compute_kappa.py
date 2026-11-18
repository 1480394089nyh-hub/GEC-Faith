# -*- coding: utf-8 -*-
"""Compute inter-annotator agreement kappa (Cohen's kappa) between two annotators.

Usage: python compute_kappa.py 标注表_我.csv 标注表_同学.csv
Assumes: both CSVs have the columns "序号" and "你的标注" (same header as the annotation table).

Output: printed to console + saved to results/annotation/kappa_报告.md (for the paper).
"""
import csv
import datetime
import os
import sys
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(ROOT, "results", "annotation", "kappa_报告.md")


def read_csv(path):
    raw = open(path, "rb").read()
    for enc in ("utf-8-sig", "gb18030", "gbk", "utf-8"):
        try:
            return list(csv.reader(raw.decode(enc).splitlines()))
        except UnicodeDecodeError:
            continue
    raise RuntimeError(f"cannot decode {path}")


def main():
    a_path, b_path = sys.argv[1], sys.argv[2]
    a_rows = read_csv(a_path)
    b_rows = read_csv(b_path)
    # header: 序号, 原句, 模型, 大模型改句, 你的标注
    header = a_rows[0]
    assert header[0] == "序号" and header[4] == "你的标注", f"表头不符: {header}"

    # key by 序号, store (原句, 改句, 标签)
    def to_dict(rows):
        d = {}
        for r in rows[1:]:
            if len(r) < 5:
                continue
            d[r[0]] = (r[1], r[3], r[4].strip())
        return d

    A = to_dict(a_rows)
    B = to_dict(b_rows)
    ids = sorted(set(A) & set(B), key=int)

    empty = [i for i in ids if not A[i][2] or not B[i][2]]
    ids = [i for i in ids if A[i][2] and B[i][2]]

    n = len(ids)
    pair = Counter((A[i][2], B[i][2]) for i in ids)
    labels = sorted(set(A[i][2] for i in ids) | set(B[i][2] for i in ids))

    po = sum(v for (x, y), v in pair.items() if x == y) / n
    a_cnt = Counter(A[i][2] for i in ids)
    b_cnt = Counter(B[i][2] for i in ids)
    pe = sum((a_cnt[k] / n) * (b_cnt[k] / n) for k in labels)
    kappa = (po - pe) / (1 - pe) if (1 - pe) != 0 else 0.0

    rep = []
    rep.append("# 双标一致性 κ 报告")
    rep.append("")
    rep.append(f"- 生成日期：{datetime.date.today().isoformat()}")
    rep.append(f"- 标注员 A：`{os.path.basename(a_path)}`")
    rep.append(f"- 标注员 B：`{os.path.basename(b_path)}`")
    rep.append(f"- 有效条目数：{n}")
    if empty:
        rep.append(f"- 跳过空标签：{len(empty)} 条，序号 {empty}")
    rep.append(f"- 标签类别：{labels}")
    rep.append("")
    rep.append("## 混淆矩阵（行 = 标注员A，列 = 标注员B）")
    rep.append("")
    rep.append("| " + " | ".join([""] + labels) + " |")
    rep.append("|---|" + "---|" * len(labels))
    for x in labels:
        row = [str(pair.get((x, y), 0)) for y in labels]
        rep.append(f"| {x} | " + " | ".join(row) + " |")
    rep.append("")
    rep.append(f"- 观察一致率 Po = {po:.4f}")
    rep.append(f"- 期望一致率 Pe = {pe:.4f}")
    rep.append(f"- **Cohen's κ = {kappa:.4f}**")
    if kappa >= 0.8:
        verdict = "几乎完全一致"
    elif kappa >= 0.6:
        verdict = "较好（substantial）"
    elif kappa >= 0.4:
        verdict = "中等（moderate）"
    else:
        verdict = "较差（fair/slight）"
    rep.append(f"- 解读：κ = {kappa:.4f} → {verdict}")

    disagree = [i for i in ids if A[i][2] != B[i][2]]
    if disagree:
        rep.append("")
        rep.append(f"## 分歧条目（共 {len(disagree)} 条）")
        rep.append("")
        for i in disagree:
            src, hyp_a, lab_a = A[i]
            _, hyp_b, lab_b = B[i]
            rep.append(f"- [{i}] `{lab_a}` vs `{lab_b}` — 原句：{src[:40]}")
            rep.append(f"  - 改句：{hyp_a[:40]}")

    text = "\n".join(rep)
    print(text)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as fh:
        fh.write(text)
    print(f"\n[report saved] {OUT}")


if __name__ == "__main__":
    main()
