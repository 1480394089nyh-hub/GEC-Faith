# -*- coding: utf-8 -*-
"""Discrimination verification: prove the failure-mode labels capture information F0.5 cannot.

This corresponds to leg 4(c) "unique value" of the proposal.

Core argument
-------------
F0.5 only counts "whether the edit spans are right", not "whether the corrected
meaning is right". So two systems with nearly equal F0.5 can differ hugely in their
failure-mode (5-class) distributions -- that is F0.5's blind spot, and exactly the
incremental information this dataset (failure-mode labels) provides.

Approach
--------
For each annotated instance, compute two orthogonal signals:
  (1) the F0.5 "edit verdict" (perfect / missed / extra / missed+extra / no-error-untouched)
  (2) the human failure-mode label (5 classes)
Then look at two views:
  A. system-level: how far apart are the failure-mode distributions of F0.5-close systems (total variation distance).
  B. instance-level cross-tab: which failure labels are mixed inside each F0.5 verdict bucket (showing F0.5 can't tell them apart).

Outputs (all saved, for the paper):
  results/annotation/区分度_报告.md     full report (method + all numbers)
  results/annotation/区分度_逐条明细.csv per-instance F0.5 details, any subset re-checkable

Usage
-----
  python scripts/verify_discrimination.py [annotated_csv]
Default: results/annotation/双标_300_我.csv (my-side final labels).
"""
import csv
import datetime
import os
import sys
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src", "eval"))

import m2_scorer

RAW = os.path.join(ROOT, "data", "raw", "nlpcc2018", "extracted", "Data", "test")
FULL_TSV = os.path.join(ROOT, "results", "full", "collect_full.tsv")
DEFAULT_ANNOTATED = os.path.join(ROOT, "results", "annotation", "双标_300_我.csv")
REPORT = os.path.join(ROOT, "results", "annotation", "区分度_报告.md")
DETAIL = os.path.join(ROOT, "results", "annotation", "区分度_逐条明细.csv")

MODELS = ["deepseek", "qwen", "glm"]
UNFAITHFUL = {"语义漂移", "幻觉", "过度纠正"}


def read_csv(path):
    raw = open(path, "rb").read()
    for enc in ("utf-8-sig", "gb18030", "gbk", "utf-8"):
        try:
            return list(csv.reader(raw.decode(enc).splitlines()))
        except UnicodeDecodeError:
            continue
    raise RuntimeError(f"cannot decode {path}")


def verdict(matched, n_gold, n_sys):
    """Translate the per-sentence F0.5 counts into an edit verdict."""
    if n_gold == 0 and n_sys == 0:
        return "无错未动"
    if matched == n_gold == n_sys:
        return "满分"
    parts = []
    if matched < n_gold:
        parts.append("漏改")
    if n_sys > matched:
        parts.append("多改")
    return "+".join(parts) or "满分"


def tvd(a, b, labels):
    """Total variation distance: 0.5 * sum|p-q| over labels between distributions a and b."""
    na, nb = sum(a.values()), sum(b.values())
    return 0.5 * sum(abs(a.get(l, 0) / na - b.get(l, 0) / nb) for l in labels)


def corpus_f05(srcs, golds, hyps):
    """Compute overall (P, R, F0.5) for a set of (src, gold, hyp)."""
    p, r, f, _ = m2_scorer.score_corpus(zip(srcs, golds, hyps))
    return p, r, f


def main():
    annotated_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_ANNOTATED

    srcs = m2_scorer.read_lines(os.path.join(RAW, "source.txt"))
    golds = m2_scorer.read_lines(os.path.join(RAW, "gold", "gold.0.txt"))
    assert len(srcs) == len(golds), f"source and gold line counts differ: {len(srcs)} vs {len(golds)}"

    rows = read_csv(annotated_path)
    header, data = rows[0], rows[1:]
    assert header[:4] == ["序号", "原句", "模型", "大模型改句"], f"表头不符: {header}"
    label_col = header.index("你的标注")

    recs = []
    for r in data:
        if len(r) <= label_col or not r[label_col].strip():
            continue
        no = int(r[0])
        src_idx = (no - 1) // 3
        model = MODELS[(no - 1) % 3]
        recs.append((no, srcs[src_idx], golds[src_idx], r[3], model, r[label_col].strip()))

    detail = []
    for no, src, gold, hyp, model, label in recs:
        matched, n_gold, n_sys = m2_scorer.score_one(src, gold, hyp)
        precision = matched / n_sys if n_sys else (1.0 if n_gold == 0 else 0.0)
        recall = matched / n_gold if n_gold else 1.0
        f = m2_scorer.f_beta(precision, recall, beta=0.5)
        detail.append(dict(no=no, model=model, label=label, v=verdict(matched, n_gold, n_sys),
                           f=f, matched=matched, n_gold=n_gold, n_sys=n_sys,
                           src=src, hyp=hyp))

    n = len(detail)
    labels_all = sorted({d["label"] for d in detail})
    rep = []  # report content, printed + saved

    rep.append("# 区分度验证报告（F0.5 vs 失败模式标签）")
    rep.append("")
    rep.append(f"- 生成日期：{datetime.date.today().isoformat()}")
    rep.append(f"- 标注来源：`{os.path.basename(annotated_path)}`（我侧最终标签）")
    rep.append(f"- 实例数：{n} 条（{n // 3} 句 × 3 模型）")
    rep.append("")
    rep.append("## 方法与参数（写进论文 Methods）")
    rep.append("- 数据集：NLPCC-2018 测试集 `source.txt`（2000 错句），参考答案用 `gold.0.txt`（单参考，第 1 份）。")
    rep.append("- 评分器：字符级 MaxMatch F0.5（`src/eval/m2_scorer.py`），β=0.5（偏重精确率）。")
    rep.append("- 匹配规则：模型改动与 gold 改动只要 **span 相同即算命中**，不要求替换文字完全一致（GEC 惯例）。")
    rep.append("- 序号映射：`错句序号 = (序号-1)//3 + 1`；`模型 = (序号-1)%3` → deepseek/qwen/glm。")
    rep.append("- 标注抽样：双标子集 100 句（排除试标错句 1-60，seed=42），与 `prepare_annotation.py` 一致。")
    rep.append("- 统计量：失败模式分布的总变异距离 TVD = 0.5 · Σ|p_i − q_i|。")
    rep.append("")

    # ---- 1. system-level ----
    by_model = defaultdict(list)
    for d in detail:
        by_model[d["model"]].append(d)

    rep.append("## 一、系统级：各模型 F0.5 与失败模式分布（同一批实例上算）")
    rep.append("")
    rep.append("| 模型 | F0.5 | P | R | " + " | ".join(labels_all) + " |")
    rep.append("|---|---|---|---|" + "---|" * len(labels_all))
    f05 = {}
    dist = {}
    for m in MODELS:
        ds = by_model[m]
        if not ds:
            continue
        tp = sum(d["matched"] for d in ds)
        fn = sum(d["n_gold"] - d["matched"] for d in ds)
        fp = sum(d["n_sys"] - d["matched"] for d in ds)
        p = tp / (tp + fp) if (tp + fp) else 0.0
        r = tp / (tp + fn) if (tp + fn) else 0.0
        f = m2_scorer.f_beta(p, r, beta=0.5)
        f05[m] = f
        dist[m] = Counter(d["label"] for d in ds)
        cells = [str(dist[m].get(l, 0)) for l in labels_all]
        rep.append(f"| {m} | {f:.4f} | {p:.4f} | {r:.4f} | " + " | ".join(cells) + " |")
    rep.append("")
    rep.append(f"注：F0.5 为该 100 句（300 实例）子集上算，与全量 2000 句的对照见第四节。")
    rep.append("")

    # ---- 2. discrimination ----
    rep.append("## 二、区分度：F0.5 相近的系统，失败模式分布差多远（核心证据）")
    rep.append("")
    rep.append("| 系统对 | \\|ΔF0.5\\| | 失败模式 TVD |")
    rep.append("|---|---|---|")
    pairs = [(MODELS[i], MODELS[j]) for i in range(3) for j in range(i + 1, 3)]
    for a, b in pairs:
        rep.append(f"| {a} ↔ {b} | {abs(f05[a] - f05[b]):.4f} | {tvd(dist[a], dist[b], labels_all):.4f} |")
    rep.append("")
    rep.append("解读：ΔF0.5 越小、TVD 越大，越说明「F0.5 认为两个系统差不多，但失败模式其实差很多」。")
    rep.append("")

    # ---- 3. cross-tab ----
    rep.append("## 三、F0.5 编辑判定 × 失败模式标签 交叉表")
    rep.append("")
    rep.append("| 编辑判定 | " + " | ".join(labels_all) + " | 合计 |")
    rep.append("|---|---|" + "---|" * len(labels_all))
    vs = ["满分", "无错未动", "漏改", "多改", "漏改+多改"]
    for v in vs:
        ds = [d for d in detail if d["v"] == v]
        c = Counter(d["label"] for d in ds)
        cells = [str(c.get(l, 0)) for l in labels_all]
        rep.append(f"| {v} | " + " | ".join(cells) + f" | {len(ds)} |")
    assert sum(len([d for d in detail if d['v'] == v]) for v in vs) == n, "判定覆盖不完整"
    rep.append("")
    rep.append("解读：同一个 F0.5 判定桶里混着多种失败标签 → F0.5 只能判「改没改对」，分不清「哪种失败」。")
    rep.append("")

    # ---- 4. full-corpus F0.5 reference ----
    rep.append("## 四、参考：全量 2000 句 F0.5（稳定性对照）")
    rep.append("")
    if os.path.exists(FULL_TSV):
        flines = open(FULL_TSV, encoding="utf-8-sig").read().splitlines()
        fdata = [l.split("\t") for l in flines[1:]]
        rep.append("| 模型 | 全量 F0.5 | P | R |")
        rep.append("|---|---|---|---|")
        for mi, m in enumerate(MODELS):
            hyps = [d[mi + 1] for d in fdata if len(d) > mi + 1]
            p, r, f = corpus_f05(srcs, golds, hyps)
            rep.append(f"| {m} | {f:.4f} | {p:.4f} | {r:.4f} |")
        rep.append("")
        rep.append("说明：用于确认「F0.5 孪生系统」不是 100 句子集的抽样噪声，而是全量上也成立。")
    else:
        rep.append("（未找到 `results/full/collect_full.tsv`，跳过。）")
    rep.append("")

    # ---- 5. blind spot ----
    blind = [d for d in detail if d["v"] == "满分" and d["label"] in UNFAITHFUL]
    rep.append(f"## 五、F0.5 盲区：F0.5 判「满分」但实为保真失败（语义漂移/幻觉/过度纠正）")
    rep.append("")
    rep.append(f"- 当前盲区实例：**{len(blind)} 条**。")
    rep.append("- 说明：样本小（100 句）时可能为 0；判别力主要靠第二、三节的系统级与桶级证据。随主标量增大重跑本脚本再看这格。")
    rep.append("")

    text = "\n".join(rep)
    print(text)
    with open(REPORT, "w", encoding="utf-8") as fh:
        fh.write(text)
    print(f"\n[report saved] {REPORT}")

    # ---- per-instance detail CSV ----
    with open(DETAIL, "w", encoding="utf-8-sig", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["序号", "模型", "标签", "原句", "大模型改句", "编辑判定",
                    "命中", "gold改动数", "系统改动数", "单句F0.5"])
        for d in detail:
            w.writerow([d["no"], d["model"], d["label"], d["src"], d["hyp"], d["v"],
                        d["matched"], d["n_gold"], d["n_sys"], f"{d['f']:.4f}"])
    print(f"[detail saved] {DETAIL}（{n} 条）")


if __name__ == "__main__":
    main()
