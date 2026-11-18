# -*- coding: utf-8 -*-
"""Quality-gate filtering experiment: use the trained faithfulness discriminator
as a post-processing gate over LLM corrections.

Reads results/discriminator/{bin,multi}/test_predictions.csv (600 held-out test
examples), simulates keeping only predictions labeled "faithful / faithful-correct"
and dropping the rest, then reports faithfulness before/after, gate precision /
recall, and per-class catch rate. Outputs a report.

Re-run to regenerate: results/discriminator/质量闸门_报告.md
"""
import csv
import os
import sys

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # D:/zh-gec-llm

GOOD = {"bin": "忠实", "multi": "忠实正确"}
LABELS5 = ["忠实正确", "漏改", "语义漂移", "过度纠正", "幻觉"]


def load(path):
    with open(path, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def gate_metrics(rows, good):
    """rows: list[dict] with 真实标签 / 预测标签 columns. good: the class to keep."""
    N = len(rows)
    true = [r["真实标签"] for r in rows]
    pred = [r["预测标签"] for r in rows]

    good_true = sum(1 for t in true if t == good)          # truly good count
    good_pred = sum(1 for p in pred if p == good)          # kept count
    tp = sum(1 for t, p in zip(true, pred) if t == good and p == good)    # correctly kept
    fp = sum(1 for t, p in zip(true, pred) if t != good and p == good)    # wrongly kept (bad passed)
    fn = sum(1 for t, p in zip(true, pred) if t == good and p != good)    # wrongly blocked (good dropped)
    tn = sum(1 for t, p in zip(true, pred) if t != good and p != good)    # correctly blocked

    return {
        "N": N,
        "good_true": good_true,
        "good_pred": good_pred,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "base_rate": good_true / N,
        "gate_precision": tp / good_pred if good_pred else 0.0,
        "recall": tp / good_true if good_true else 0.0,
    }


def per_class_catch(rows):
    """Per-class catch rate: fraction of each true class that the gate blocked."""
    out = []
    for lbl in LABELS5:
        true_cnt = sum(1 for r in rows if r["真实标签"] == lbl)
        caught = sum(1 for r in rows
                     if r["真实标签"] == lbl and r["预测标签"] != "忠实正确")
        out.append((lbl, true_cnt, caught, caught / true_cnt if true_cnt else 0.0))
    return out


def fmt_pct(x):
    return f"{x * 100:.2f}%"


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    report = []
    report.append("# 判别器「质量闸门」过滤实验报告\n")
    report.append("> 用训好的忠实度判别器做大模型改句的**后处理闸门**：判为「忠实 / 忠实正确」放行，其余 4 类失败拦下。\n")
    report.append("> 数据：test 集 600 条（模型从没见过的数据），来源 `results/discriminator/{bin,multi}/test_predictions.csv`。\n")
    report.append("> 重跑即更新：`scripts/eval_quality_gate.py`。\n")
    report.append("## 方法与参数（写 Methods 用）\n")
    report.append("- 判别器：`hfl/chinese-roberta-wwm-ext`，句对输入 `[CLS]原句[SEP]改句[SEP]`，max_len=256，epochs=5，batch=32，lr=2e-5，seed=42。\n")
    report.append("- 闸门策略：只放行预测为「忠实」（二分类）/「忠实正确」（五分类）的改句，其余失败类拦下（可退回大模型重改或交人工）。\n")
    report.append("- 指标：过滤前后忠实率、闸门 precision / recall、逐类失败捕获率。\n")

    metrics = {}
    for task in ("bin", "multi"):
        good = GOOD[task]
        rows = load(os.path.join(BASE, "results", "discriminator", task, "test_predictions.csv"))
        m = gate_metrics(rows, good)
        metrics[task] = m

        title = "二分类 bin（忠实 / 不忠实）" if task == "bin" else "五分类 multi（5 类失败）"
        report.append(f"\n## {title}\n")
        report.append(f"- 测试样本：{m['N']} 条\n")
        report.append(f"- **不过滤（大模型原始输出）忠实率：{fmt_pct(m['base_rate'])}**（{m['good_true']}/{m['N']}）\n")
        report.append(f"- **过滤后忠实率：{fmt_pct(m['gate_precision'])}**（{m['tp']}/{m['good_pred']}，即闸门 precision）\n")
        report.append(f"- 合格句保留率（recall）：{fmt_pct(m['recall'])}（{m['tp']}/{m['good_true']}）\n")
        report.append(f"- 提升幅度：**{fmt_pct(m['gate_precision'] - m['base_rate'])}**（忠实率绝对提升）\n")
        report.append("\n| 闸门结果 | 真实合格 | 真实不合格 |\n|---|---|---|\n")
        report.append(f"| 放行（预测合格） | {m['tp']} ✅ | {m['fp']} ❌（误放行） |\n")
        report.append(f"| 拦下（预测不合格） | {m['fn']} ⚠️（误拦） | {m['tn']} ✅ |\n")

        if task == "multi":
            report.append("\n### 逐类失败捕获率（闸门拦下了多少）\n")
            report.append("| 失败类 | 真实数 | 被拦下 | 捕获率 |\n|---|---|---|---|\n")
            for lbl, cnt, caught, rate in per_class_catch(rows):
                report.append(f"| {lbl} | {cnt} | {caught} | {fmt_pct(rate)} |\n")

    b = metrics["bin"]
    m = metrics["multi"]
    report.append("\n## 结论（写 Results 用）\n")
    report.append(f"- **二分类闸门更优**：忠实率 {fmt_pct(b['base_rate'])} → {fmt_pct(b['gate_precision'])}（+{fmt_pct(b['gate_precision'] - b['base_rate'])}），合格句保留率 {fmt_pct(b['recall'])}。\n")
    report.append(f"- 五分类闸门忠实率 {fmt_pct(m['gate_precision'])}（+{fmt_pct(m['gate_precision'] - m['base_rate'])}，保留率 {fmt_pct(m['recall'])}），略低于二分类——五分类更难，但换来逐类失败诊断（见上表）。\n")
    report.append("- 意义：证明判别器可作后处理质量闸门，**不训练大模型本身即可提升其输出忠实度**；语义漂移/幻觉等稀有失败捕获率仍有限，是 limitation。\n")

    out_path = os.path.join(BASE, "results", "discriminator", "质量闸门_报告.md")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report))
    print("\n".join(report))
    print(f"\n[saved] {out_path}")


if __name__ == "__main__":
    main()
