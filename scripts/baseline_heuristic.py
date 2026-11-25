# -*- coding: utf-8 -*-
"""规则基线：不训练，只看 (原句, 改句) 的字符编辑距离，判「忠实 / 不忠实」。

对比对象：训练好的二分类判别器（不忠实 F1 = 0.5110）。
三个启发式（阈值在 val 上调，test 上报）：
  H1 原样未改：src == hyp -> 不忠实
  H2 改太少：  改动率 < t -> 不忠实（漏改）
  H3 双向：    改动率 < t_low 或 > t_high -> 不忠实（漏改 或 漂移/幻觉/过度纠正）
"""
import json
import numpy as np


def lev(a, b):
    """字符级编辑距离（Levenshtein）。"""
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def load(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def features(rows):
    src = [r["src"] for r in rows]
    hyp = [r["hyp"] for r in rows]
    y = np.array([r["label_bin_id"] for r in rows])
    dist = np.array([lev(s, h) for s, h in zip(src, hyp)])
    ratio = dist / np.maximum(np.array([len(s) for s in src]), 1)
    return y, dist, ratio


def metrics(y, pred):
    tp = ((y == 1) & (pred == 1)).sum()
    fn = ((y == 1) & (pred == 0)).sum()
    fp = ((y == 0) & (pred == 1)).sum()
    tn = ((y == 0) & (pred == 0)).sum()
    rec = tp / (tp + fn) if tp + fn else 0
    prec = tp / (tp + fp) if tp + fp else 0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0
    return rec, prec, f1, (tp, fp, fn, tn)


def main():
    val = load("results/discriminator/val.jsonl")
    test = load("results/discriminator/test.jsonl")
    val_y, val_dist, val_ratio = features(val)
    test_y, test_dist, test_ratio = features(test)

    print("=== H1 原样未改（src == hyp → 不忠实）===")
    val_pred1 = (val_dist == 0).astype(int)
    test_pred1 = (test_dist == 0).astype(int)
    r = metrics(test_y, test_pred1)
    print(f"  val: {metrics(val_y, val_pred1)[:3]}")
    print(f"  test: recall={r[0]:.4f} prec={r[1]:.4f} F1={r[2]:.4f}  CM={r[3]}")

    print("\n=== H2 改太少（改动率 < t → 不忠实），val 扫阈值 ===")
    best = None
    for t in np.arange(0.0, 0.55, 0.05):
        pred = (val_ratio < t).astype(int)
        rec, prec, f1, _ = metrics(val_y, pred)
        if best is None or f1 > best[1]:
            best = (t, f1, rec, prec)
        print(f"  t<{t:.2f}: val recall={rec:.4f} prec={prec:.4f} F1={f1:.4f}")
    t_best = best[0]
    test_pred2 = (test_ratio < t_best).astype(int)
    r = metrics(test_y, test_pred2)
    print(f"  最优 t<{t_best:.2f} (val F1={best[1]:.4f}) -> test recall={r[0]:.4f} prec={r[1]:.4f} F1={r[2]:.4f} CM={r[3]}")

    print("\n=== H3 双向（改动率 < t_low 或 > t_high → 不忠实）===")
    best3 = None
    for tl in np.arange(0.0, 0.35, 0.05):
        for th in np.arange(0.4, 1.6, 0.2):
            pred = ((val_ratio < tl) | (val_ratio > th)).astype(int)
            rec, prec, f1, _ = metrics(val_y, pred)
            if best3 is None or f1 > best3[1]:
                best3 = ((tl, th), f1, rec, prec)
    tl_b, th_b = best3[0]
    test_pred3 = ((test_ratio < tl_b) | (test_ratio > th_b)).astype(int)
    r = metrics(test_y, test_pred3)
    print(f"  最优 <{tl_b:.2f} 或 >{th_b:.2f} (val F1={best3[1]:.4f}) -> test recall={r[0]:.4f} prec={r[1]:.4f} F1={r[2]:.4f} CM={r[3]}")

    print("\n=== 对照：训练判别器（一致口径）===")
    print("  不忠实 recall=0.4309 prec=0.6279 F1=0.5110  |  全判忠实基线 F1=0")

    # 落盘报告
    out = ["# 规则基线 vs 训练判别器（bin）\n",
           "> 规则只用 (原句, 改句) 的字符编辑距离，不用训练、不用 gold 参考，是公平的简单基线。\n",
           "> 阈值在 val 上调，test 上报。\n\n",
           "| 方法 | recall(拦截率) | precision | F1 | 混淆矩阵 TP/FP/FN/TN |",
           "|---|---|---|---|---|"]
    def row(name, y, pred):
        rec, prec, f1, cm = metrics(y, pred)
        out.append(f"| {name} | {rec:.4f} | {prec:.4f} | {f1:.4f} | {cm} |")
    row("H1 原样未改", test_y, test_pred1)
    row(f"H2 改动率<{t_best:.2f}", test_y, test_pred2)
    row(f"H3 双向<{tl_b:.2f}或>{th_b:.2f}", test_y, test_pred3)
    out.append("| 训练判别器 @0.5 | 0.4309 | 0.6279 | 0.5110 | (81,48,107,364) |")
    out.append("| 全判忠实基线 | 0.0000 | — | 0.0000 | (0,0,188,412) |")
    open("results/discriminator/bin/基线对比_报告.md", "w", encoding="utf-8").write("\n".join(out))
    print("\n报告已写 results/discriminator/bin/基线对比_报告.md")


if __name__ == "__main__":
    main()
