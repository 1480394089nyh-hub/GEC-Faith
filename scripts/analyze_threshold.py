# -*- coding: utf-8 -*-
"""加载本地 bin 判别器，重算 val/test 的「不忠实」预测概率，做阈值扫描。

目的：原 CSV 只存了硬标签（默认 0.5 阈值），看不到调阈值的收益。
本脚本输出：
  1) val/test 逐条概率  -> results/discriminator/bin/probs_{val,test}.csv
  2) 阈值扫描表（在 val 上调，套用到 test）-> results/discriminator/bin/阈值扫描_报告.md
"""
import json, csv, os
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_DIR = "results/discriminator/bin/best_model"
MAX_LEN = 256
BATCH = 32


def load_model():
    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
    model.eval()
    return tok, model


def load_jsonl(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def predict(tok, model, rows):
    """返回 shape (n,) 的 P(不忠实)（label_bin_id==1）。"""
    srcs = [r["src"] for r in rows]
    hyps = [r["hyp"] for r in rows]
    probs = []
    with torch.no_grad():
        for i in range(0, len(rows), BATCH):
            s = srcs[i:i + BATCH]
            h = hyps[i:i + BATCH]
            enc = tok(s, h, padding=True, truncation=True,
                      max_length=MAX_LEN, return_tensors="pt")
            logits = model(**enc).logits  # (b, 2)
            p = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()
            probs.extend(p.tolist())
    return np.array(probs)


def save_probs(path, rows, probs):
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["id", "模型", "真实标签", "真实id", "P(不忠实)"])
        for r, p in zip(rows, probs):
            w.writerow([r["id"], r["model"], r["label_bin"], r["label_bin_id"], f"{p:.6f}"])


def metrics_at(y_true, probs, th):
    pred = (probs >= th).astype(int)  # 1 = 不忠实
    tp = int(((y_true == 1) & (pred == 1)).sum())
    fn = int(((y_true == 1) & (pred == 0)).sum())
    fp = int(((y_true == 0) & (pred == 1)).sum())
    tn = int(((y_true == 0) & (pred == 0)).sum())
    rec = tp / (tp + fn) if tp + fn else 0.0       # 拦截率(不忠实 recall)
    prec = tp / (tp + fp) if tp + fp else 0.0       # 不忠实 precision
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    faithful_pass = tn / (tn + fn) if tn + fn else 0.0   # 过滤后忠实率
    retain_good = tn / (tn + fp) if tn + fp else 0.0     # 合格句保留率
    return rec, prec, f1, faithful_pass, retain_good, (tp, fp, fn, tn)


def main():
    tok, model = load_model()
    print("模型已加载")

    val_rows = load_jsonl("results/discriminator/val.jsonl")
    test_rows = load_jsonl("results/discriminator/test.jsonl")
    val_y = np.array([r["label_bin_id"] for r in val_rows])
    test_y = np.array([r["label_bin_id"] for r in test_rows])

    print(f"推理 val {len(val_rows)} 条 ...")
    val_p = predict(tok, model, val_rows)
    print(f"推理 test {len(test_rows)} 条 ...")
    test_p = predict(tok, model, test_rows)

    save_probs("results/discriminator/bin/probs_val.csv", val_rows, val_p)
    save_probs("results/discriminator/bin/probs_test.csv", test_rows, test_p)
    print("概率已落盘 probs_{val,test}.csv")

    # 在 val 上扫阈值
    print("\n=== 阈值扫描（val 600 条）===")
    print("th | rec(拦截率) | prec | F1 | 过滤后忠实率 | 合格句保留率")
    results = []
    for th in np.arange(0.05, 1.0, 0.05):
        rec, prec, f1, fp_rate, retain, _ = metrics_at(val_y, val_p, th)
        results.append((th, rec, prec, f1, fp_rate, retain))
        print(f"{th:.2f} | {rec:.4f} | {prec:.4f} | {f1:.4f} | {fp_rate:.4f} | {retain:.4f}")

    # 挑几个操作点，在 test 上套用
    def report(th, name):
        rec, prec, f1, fp_rate, retain, cm = metrics_at(test_y, test_p, th)
        print(f"\n[{name}] 阈值={th:.2f}  (test 600 条)")
        print(f"  不忠实: recall(拦截率)={rec:.4f}  precision={prec:.4f}  F1={f1:.4f}")
        print(f"  过滤后忠实率={fp_rate:.4f}  合格句保留率={retain:.4f}")
        print(f"  混淆矩阵 TP/FP/FN/TN = {cm}")

    # 默认 0.5、最大 F1 点、以及几个召回更高的点
    report(0.50, "默认 0.5")
    # 在 val 上找 F1 最大、以及 recall 达到 0.6/0.7 的最小阈值
    best_f1_th = max(results, key=lambda x: x[3])[0]
    report(best_f1_th, "val 上 F1 最大")
    for target_rec in (0.6, 0.7, 0.8):
        cand = [r for r in results if r[1] >= target_rec]
        if cand:
            th = min(cand, key=lambda x: x[0])[0]
            report(th, f"recall≥{target_rec} 的最小阈值")


if __name__ == "__main__":
    main()
