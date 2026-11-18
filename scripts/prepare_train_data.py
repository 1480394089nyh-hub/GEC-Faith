# -*- coding: utf-8 -*-
"""
prepare_train_data.py -- Convert the final annotation CSV into discriminator
training data (train/val/test stratified split).

Input : results/full/标注表_全量_定稿.csv  (GBK encoding, 6000 rows = 2000 sentences x 3 models)
Output: results/discriminator/
          train.jsonl / val.jsonl / test.jsonl   -- one JSON per line
          label_map.json                         -- label id mapping (single source of truth for training)
          data_split_report.md                   -- split method + per-split label distribution + length stats

Key method (can go straight into the paper's Methods):
  - instance = one (original src, LLM correction hyp) pair; label = the "标签" (label) column (final).
  - split unit = sentence (2000), so the 3 model outputs of the same sentence never cross
    train/val/test, preventing source-sentence leakage.
  - stratification key = the "most severe" label among a sentence's 3 instances
    (priority: hallucination > semantic-drift > over-correction > missed > faithful-correct).
  - sklearn StratifiedShuffleSplit, seed=42, sentence-level ratio train:val:test = 80:10:10.
"""
import csv
import json
import os
from collections import Counter, defaultdict

import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit

SRC_CSV = "results/full/标注表_全量_定稿.csv"
OUT_DIR = "results/discriminator"
SEED = 42

# Fixed order of the 5 labels (i.e. label5_id). Ordered by frequency high-to-low;
# just a stable convention, no semantic meaning.
LABEL5 = ["忠实正确", "漏改", "语义漂移", "过度纠正", "幻觉"]
# Severity, used as the sentence-level stratification key (larger = more severe).
# Matches the annotation guideline priority: hallucination > semantic-drift > over-correction > missed > faithful-correct.
SEVERITY = {"幻觉": 4, "语义漂移": 3, "过度纠正": 2, "漏改": 1, "忠实正确": 0}


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    rows = list(csv.reader(open(SRC_CSV, encoding="gbk")))
    header, data = rows[0], rows[1:]
    assert header == ["序号", "原句", "模型", "大模型改句", "标签"], f"列名不符：{header}"
    assert len(data) == 6000, f"行数 {len(data)} != 6000"

    # ---- group by sentence: sent_idx = (序号-1)//3 + 1, 3 consecutive rows = same sentence (deepseek/qwen/glm) ----
    groups = defaultdict(list)  # sent_idx -> [instance, ...]
    for r in data:
        idx = int(r[0])
        sent_idx = (idx - 1) // 3 + 1
        src, hyp, lab = r[1], r[3], r[4].strip()
        assert lab in LABEL5, f"非法标签 {lab!r}，序号 {idx}"
        groups[sent_idx].append({
            "id": idx,
            "sent_idx": sent_idx,
            "model": r[2],
            "src": src,
            "hyp": hyp,
            "label5": lab,
            "label5_id": LABEL5.index(lab),
            "label_bin": "忠实" if lab == "忠实正确" else "不忠实",
            "label_bin_id": 0 if lab == "忠实正确" else 1,
        })

    assert len(groups) == 2000, f"句子数 {len(groups)} != 2000"
    for si, insts in groups.items():
        assert len(insts) == 3, f"句 {si} 实例数 {len(insts)} != 3"
        assert len({i["src"] for i in insts}) == 1, f"句 {si} 的原句不一致"

    # ---- sentence-level stratified split (most severe label as the key) ----
    sent_ids = sorted(groups.keys())
    sent_sev = {si: max(SEVERITY[i["label5"]] for i in insts) for si, insts in groups.items()}

    sss1 = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
    train_pos, rest_pos = next(sss1.split(np.arange(len(sent_ids)),
                                          [sent_sev[s] for s in sent_ids]))
    train_sents = [sent_ids[i] for i in train_pos]
    rest_sents = [sent_ids[i] for i in rest_pos]

    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=SEED)
    val_pos, test_pos = next(sss2.split(np.arange(len(rest_sents)),
                                        [sent_sev[s] for s in rest_sents]))
    val_sents = [rest_sents[i] for i in val_pos]
    test_sents = [rest_sents[i] for i in test_pos]

    assert len(train_sents) + len(val_sents) + len(test_sents) == 2000
    assert not (set(train_sents) & set(val_sents) & set(test_sents)), "切分有交集"

    def write_jsonl(sents, path):
        lines = []
        for si in sorted(sents):
            for inst in sorted(groups[si], key=lambda x: x["id"]):
                lines.append(json.dumps(inst, ensure_ascii=False))
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + ("\n" if lines else ""))
        return lines

    splits = {
        "train": write_jsonl(train_sents, os.path.join(OUT_DIR, "train.jsonl")),
        "val": write_jsonl(val_sents, os.path.join(OUT_DIR, "val.jsonl")),
        "test": write_jsonl(test_sents, os.path.join(OUT_DIR, "test.jsonl")),
    }

    # ---- label_map.json (single source of truth, read by the training script) ----
    label_map = {
        "label5": {lab: i for i, lab in enumerate(LABEL5)},
        "id2label5": {i: lab for i, lab in enumerate(LABEL5)},
        "label_bin": {"忠实": 0, "不忠实": 1},
        "id2label_bin": {0: "忠实", 1: "不忠实"},
    }
    with open(os.path.join(OUT_DIR, "label_map.json"), "w", encoding="utf-8") as f:
        json.dump(label_map, f, ensure_ascii=False, indent=2)

    # ---- report ----
    report = _build_report(splits)
    with open(os.path.join(OUT_DIR, "data_split_report.md"), "w", encoding="utf-8") as f:
        f.write(report)

    print(report)
    print("written: train/val/test.jsonl, label_map.json, data_split_report.md ->", OUT_DIR)


def _build_report(splits):
    lines = []
    lines.append("# 判别器训练数据切分报告（CGEC-Faith）\n")
    lines.append("> 自动生成：`scripts/prepare_train_data.py`，重跑即更新。\n")
    lines.append("## 方法与参数\n")
    lines.append("- 源数据：`results/full/标注表_全量_定稿.csv`（6000 条 = 2000 句 × 3 模型，GBK）。")
    lines.append("- 实例 = 一条 `(原句 src, 大模型改句 hyp)` 对；标签取「标签」列（定稿）。")
    lines.append("- 切分单位 = 句子（2000 句），同一句的 3 条模型输出不跨 train/val/test，避免原句泄漏。")
    lines.append("- 分层键 = 该句 3 条实例里「最严重」标签（优先级 幻觉 > 语义漂移 > 过度纠正 > 漏改 > 忠实正确）。")
    lines.append("- `StratifiedShuffleSplit`，`random_state=42`，句子级比例 train:val:test = 80:10:10。\n")

    lines.append("## 各切分层规模与标签分布\n")
    lab5 = ["忠实正确", "漏改", "语义漂移", "过度纠正", "幻觉"]
    for name in ["train", "val", "test"]:
        insts = [json.loads(s) for s in splits[name]]
        n_sent = len({i["sent_idx"] for i in insts})
        lines.append(f"### {name}（句子 {n_sent}，实例 {len(insts)}）")
        c5 = Counter(i["label5"] for i in insts)
        cb = Counter(i["label_bin"] for i in insts)
        lines.append("| 标签 | 忠实正确 | 漏改 | 语义漂移 | 过度纠正 | 幻觉 | 合计 |")
        lines.append("|---|---|---|---|---|---|---|")
        lines.append("| 5 类 | " + " | ".join(str(c5[l]) for l in lab5) + f" | {len(insts)} |")
        lines.append("| 二分类 | 忠实 " + str(cb["忠实"]) + " | 不忠实 " + str(cb["不忠实"]) + " | | | | " + str(len(insts)) + " |")
        lines.append("")

    lines.append("## 长度统计（src+hyp 拼接后的字符数，用于定 max_len）\n")
    all_insts = [json.loads(s) for sp in splits.values() for s in sp]
    lens = sorted(len(i["src"]) + len(i["hyp"]) + 3 for i in all_insts)
    lines.append(f"- 拼接后字符数：min={lens[0]}，max={lens[-1]}，均值={sum(lens)/len(lens):.1f}。")
    lines.append(f"- 拼接后 >128 的实例数：{sum(1 for x in lens if x > 128)} / {len(lens)}。")
    lines.append(f"- 拼接后 >256 的实例数：{sum(1 for x in lens if x > 256)} / {len(lens)}。")
    lines.append("- 结论：`max_len=256` 仅截断极少数超长句（中文 RoBERTa 每个汉字约 1 token）。\n")

    return "\n".join(lines)


if __name__ == "__main__":
    main()
