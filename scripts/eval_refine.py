# -*- coding: utf-8 -*-
"""反馈再生成（refine）实验评估脚本。

读入「反馈再生成」全量 119 条的回标结果，计算：
  1. 忠实率对比（重改前 vs 重改后）
  2. 修复率 / 损伤率 / 净效果
  3. 5 类混淆矩阵
  4. 按「是否变化」「模型」「诊断失败类型」分层

输出（重跑即更新）：
  results/refine/反馈再生成_报告.md
  results/refine/反馈再生成_逐条明细.csv
"""
import csv
import openpyxl
from collections import Counter, OrderedDict

SRC_XLSX = "results/refine/标注表_反馈再生成_全量119.xlsx"
OUT_MD = "results/refine/反馈再生成_报告.md"
OUT_CSV = "results/refine/反馈再生成_逐条明细.csv"

# 5 类体系（与数据集定稿一致）
CANON = ["忠实正确", "漏改", "语义漂移", "过度纠正", "幻觉"]
# 笔误归一（用户在标注时的同义/错别字）
FIX = {"过塑纠正": "过度纠正"}
FAITHFUL = "忠实正确"          # 二分类里的「忠实」


def norm(label):
    if label is None:
        return ""
    s = str(label).strip()
    return FIX.get(s, s)


def is_faithful(label):
    return label == FAITHFUL


def main():
    wb = openpyxl.load_workbook(SRC_XLSX)
    ws = wb.active

    rows = []
    fixed_typo = []
    for r in range(2, ws.max_row + 1):
        idx = ws.cell(row=r, column=1).value
        sent_idx = ws.cell(row=r, column=2).value
        model = ws.cell(row=r, column=3).value
        diag = ws.cell(row=r, column=4).value
        src = ws.cell(row=r, column=5).value
        prev_out = ws.cell(row=r, column=6).value
        prev_label_raw = ws.cell(row=r, column=7).value
        new_out = ws.cell(row=r, column=8).value
        changed = ws.cell(row=r, column=9).value
        new_label_raw = ws.cell(row=r, column=10).value

        prev_label = norm(prev_label_raw)
        new_label = norm(new_label_raw)

        # 归一笔误（写回 xlsx 保持源数据干净）
        if new_label != new_label_raw and new_label_raw is not None \
                and str(new_label_raw).strip() == "过塑纠正":
            ws.cell(row=r, column=10).value = "过度纠正"
            fixed_typo.append(idx)

        rows.append(dict(
            idx=idx, sent_idx=sent_idx, model=model, diag=diag, src=src,
            prev_out=prev_out, prev_label=prev_label, new_out=new_out,
            changed=str(changed).strip() if changed is not None else "",
            new_label=new_label,
        ))

    if fixed_typo:
        wb.save(SRC_XLSX)
        print(f"[归一] 已把 {len(fixed_typo)} 条「过塑纠正」写回为「过度纠正」：序号 {fixed_typo}")

    n = len(rows)
    assert n == 119, f"预期 119 条，实际 {n} 条"

    # ---- 基本统计 ----
    prev_c = Counter(r["prev_label"] for r in rows)
    new_c = Counter(r["new_label"] for r in rows)
    chg_c = Counter(r["changed"] for r in rows)

    # ---- 5 类混淆矩阵：行=上次真实标签，列=新改句标注 ----
    cm = OrderedDict((pl, OrderedDict((nl, 0) for nl in CANON)) for pl in CANON)
    for r in rows:
        if r["prev_label"] in CANON and r["new_label"] in CANON:
            cm[r["prev_label"]][r["new_label"]] += 1

    # ---- 二分类（忠实 / 不忠实）----
    def faithful_prev(r):
        return is_faithful(r["prev_label"])

    def faithful_new(r):
        return is_faithful(r["new_label"])

    n_prev_faith = sum(1 for r in rows if faithful_prev(r))
    n_prev_unfaith = n - n_prev_faith
    n_new_faith = sum(1 for r in rows if faithful_new(r))
    n_new_unfaith = n - n_new_faith

    # 修复 / 损伤 / 保持 / 未修复
    repaired = [r for r in rows if (not faithful_prev(r)) and faithful_new(r)]
    damaged = [r for r in rows if faithful_prev(r) and (not faithful_new(r))]
    kept_faith = [r for r in rows if faithful_prev(r) and faithful_new(r)]
    still_unfaith = [r for r in rows if (not faithful_prev(r)) and (not faithful_new(r))]

    repair_rate = len(repaired) / n_prev_unfaith if n_prev_unfaith else 0
    damage_rate = len(damaged) / n_prev_faith if n_prev_faith else 0
    net = len(repaired) - len(damaged)
    faith_rate_before = n_prev_faith / n
    faith_rate_after = n_new_faith / n

    # ---- 按「是否变化」分层 ----
    def split_by(pred):
        grp = OrderedDict()
        for r in rows:
            k = pred(r)
            grp.setdefault(k, []).append(r)
        return grp

    def subgroup_metrics(sub):
        m = len(sub)
        pf = sum(1 for r in sub if faithful_prev(r))
        nf = sum(1 for r in sub if faithful_new(r))
        rep = sum(1 for r in sub if (not faithful_prev(r)) and faithful_new(r))
        pu = m - pf
        dmg = sum(1 for r in sub if faithful_prev(r) and (not faithful_new(r)))
        return dict(n=m, prev_faith=pf, new_faith=nf,
                    repair=rep, repair_rate=(rep / pu if pu else None),
                    damage=dmg, damage_rate=(dmg / pf if pf else None),
                    net=rep - dmg)

    # 一致性检查：未变（否）的句子，新改句标注应等于上次真实标签
    unchanged_rows = [r for r in rows if r["changed"] == "否"]
    inconsistent = [r for r in unchanged_rows if r["new_label"] != r["prev_label"]]

    # ---- 写报告 ----
    L = []
    w = L.append
    w("# 反馈再生成（refine）实验评估报告")
    w("")
    w("> 本报告由 `scripts/eval_refine.py` 自动生成，重跑即更新。")
    w("")
    w("## 一、方法与口径")
    w("")
    w("- **样本**：全量 119 条 = 二分类判别器（bin）判为「不忠实」而被质量闸门拦下的全部改句。")
    w("- **标签体系**：与数据集定稿一致的 5 类（忠实正确 / 漏改 / 语义漂移 / 过度纠正 / 幻觉）。")
    w("  二分类口径：忠实 = 忠实正确；不忠实 = 其余 4 类。")
    w("- **对比对象**：`上次真实标签`（原输出的定稿标签，作为重改前基线） vs `新改句标注`（重改输出的人工回标）。")
    w("- **指标定义**：")
    w("  - 修复率 = 重改前「不忠实」中、重改后变「忠实正确」的比例（真失败被修好的比例）。")
    w("  - 损伤率 = 重改前「忠实正确」（即 bin 的误报）中、重改后变「不忠实」的比例（误报被改坏的比例）。")
    w("  - 净效果 = 修复成功条数 − 损伤条数；忠实率 = 忠实正确条数 / 119。")
    w("- **数据归一**：标注中 1 条「过塑纠正」按笔误归一为「过度纠正」（序号已写回 xlsx）。")
    w("")
    w("## 二、总体结果（119 条）")
    w("")
    w("### 忠实率对比（核心「提升」数字）")
    w("")
    w("| 阶段 | 忠实正确 | 不忠实 | 忠实率 |")
    w("|---|---|---|---|")
    w(f"| 重改前（原输出定稿标签） | {n_prev_faith} | {n_prev_unfaith} | {faith_rate_before:.2%} |")
    w(f"| 重改后（新改句人工回标） | {n_new_faith} | {n_new_unfaith} | {faith_rate_after:.2%} |")
    w("")
    w(f"**忠实率 {faith_rate_before:.2%} → {faith_rate_after:.2%}，提升 +{faith_rate_after - faith_rate_before:.2%}（净增 {net:+d} 条忠实输出）。**")
    w("")
    w("### 修复 / 损伤 / 净效果")
    w("")
    w(f"- 修复成功（真不忠实 → 忠实正确）：**{len(repaired)} 条**，修复率 = {len(repaired)}/{n_prev_unfaith} = **{repair_rate:.2%}**。")
    w(f"- 损伤（误报的忠实 → 不忠实）：**{len(damaged)} 条**，损伤率 = {len(damaged)}/{n_prev_faith} = **{damage_rate:.2%}**。")
    w(f"- 保持忠实：{len(kept_faith)} 条；仍未修复（不忠实→不忠实）：{len(still_unfaith)} 条。")
    w(f"- **净效果 = 修复 {len(repaired)} − 损伤 {len(damaged)} = {net:+d} 条**。")
    w("")
    w("### 5 类混淆矩阵（行 = 重改前标签，列 = 重改后标签）")
    w("")
    header = "| 上次 \\ 新改 | " + " | ".join(CANON) + " | 合计 |"
    sep = "|---" * (len(CANON) + 2) + "|"
    w(header)
    w(sep)
    for pl in CANON:
        cells = [str(cm[pl][nl]) for nl in CANON]
        total = prev_c.get(pl, 0)
        w(f"| {pl} | " + " | ".join(cells) + f" | {total} |")
    w("| **合计** | " + " | ".join(str(new_c.get(nl, 0)) for nl in CANON) + f" | {n} |")
    w("")
    w("## 三、按「是否变化」分层")
    w("")
    w("`是否变化` = 新改句是否与上次改句不同（否 = 模型自认为没错、原样返回）。")
    w("")
    w("| 是否变化 | 条数 | 重改前忠实 | 重改后忠实 | 修复 | 损伤 | 净效果 |")
    w("|---|---|---|---|---|---|---|")
    for k in ["是", "否"]:
        m = subgroup_metrics(split_by(lambda r: r["changed"]).get(k, []))
        if m["n"] == 0:
            continue
        repair_s = f"{m['repair']}" + (f"（{m['repair_rate']:.0%}）" if m["repair_rate"] is not None else "")
        damage_s = f"{m['damage']}" + (f"（{m['damage_rate']:.0%}）" if m["damage_rate"] is not None else "")
        w(f"| {k} | {m['n']} | {m['prev_faith']} | {m['new_faith']} | {repair_s} | {damage_s} | {m['net']:+d} |")
    w("")
    if inconsistent:
        note = "，不一致序号：" + "、".join(str(r['idx']) for r in inconsistent)
    else:
        note = ""
    w(f"- 一致性检查：{len(unchanged_rows)} 条「否」中，有 {len(inconsistent)} 条的新改句标注 ≠ 上次真实标签"
      f"（理论上未变句应一致）{note}。")
    w("")
    w("## 四、按模型分层")
    w("")
    w("| 模型 | 条数 | 重改前忠实 | 重改后忠实 | 修复 | 损伤 | 净效果 |")
    w("|---|---|---|---|---|---|---|")
    for model in ["deepseek", "qwen", "glm"]:
        m = subgroup_metrics(split_by(lambda r: r["model"]).get(model, []))
        if m["n"] == 0:
            continue
        repair_s = f"{m['repair']}" + (f"（{m['repair_rate']:.0%}）" if m["repair_rate"] is not None else "")
        damage_s = f"{m['damage']}" + (f"（{m['damage_rate']:.0%}）" if m["damage_rate"] is not None else "")
        w(f"| {model} | {m['n']} | {m['prev_faith']} | {m['new_faith']} | {repair_s} | {damage_s} | {m['net']:+d} |")
    w("")
    w("## 五、按「诊断失败类型」分层（multi 判定的失败类型）")
    w("")
    w("| 诊断类型 | 条数 | 重改前忠实 | 重改后忠实 | 修复 | 损伤 | 净效果 |")
    w("|---|---|---|---|---|---|---|")
    for diag in ["忠实正确", "漏改", "语义漂移", "过度纠正"]:
        m = subgroup_metrics(split_by(lambda r: r["diag"]).get(diag, []))
        if m["n"] == 0:
            continue
        repair_s = f"{m['repair']}" + (f"（{m['repair_rate']:.0%}）" if m["repair_rate"] is not None else "")
        damage_s = f"{m['damage']}" + (f"（{m['damage_rate']:.0%}）" if m["damage_rate"] is not None else "")
        w(f"| {diag} | {m['n']} | {m['prev_faith']} | {m['new_faith']} | {repair_s} | {damage_s} | {m['net']:+d} |")
    w("")
    w("## 六、结论（可直接引用于论文第 6 节）")
    w("")
    w(f"- 反馈再生成把闸门拦下的 119 条改句的忠实率从 **{faith_rate_before:.2%}** 提升到 **{faith_rate_after:.2%}**"
      f"（+{faith_rate_after - faith_rate_before:.2%}，净增 {net:+d} 条），证明「被动过滤」可升级为「主动修复」。")
    w(f"- 修复率 {repair_rate:.2%}：约 1/5 的真失败被修好，主要来源是「漏改」（{cm['漏改']['忠实正确']} 条）。")
    w(f"- 损伤率 {damage_rate:.2%}：仅 {len(damaged)} 条误报被改坏，代价远小于收益（净效果 +{net}）。")
    w(f"- 未变句 {len(unchanged_rows)}/{n}（{len(unchanged_rows)/n:.0%}）原样返回：模型自认为没错就不改，"
      "这是修复率受限的主因——模型「不承认」的失败无法靠提示词纠正，是 limitation。")
    w("")

    with open(OUT_MD, "w", encoding="utf-8") as f:
        f.write("\n".join(L))

    # ---- 写逐条明细 CSV ----
    with open(OUT_CSV, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["序号", "句子序号", "模型", "诊断失败类型", "上次真实标签",
                         "新改句标注", "是否变化", "结果分类", "原句", "上次改句", "新改句"])
        for r in rows:
            if faithful_prev(r) and faithful_new(r):
                cat = "保持忠实"
            elif faithful_prev(r) and (not faithful_new(r)):
                cat = "损伤"
            elif (not faithful_prev(r)) and faithful_new(r):
                cat = "修复成功"
            else:
                cat = "未修复"
            writer.writerow([r["idx"], r["sent_idx"], r["model"], r["diag"],
                             r["prev_label"], r["new_label"], r["changed"], cat,
                             r["src"], r["prev_out"], r["new_out"]])

    print(f"报告已写：{OUT_MD}")
    print(f"明细已写：{OUT_CSV}")
    print(f"核心：忠实率 {faith_rate_before:.2%} -> {faith_rate_after:.2%} "
          f"(+{faith_rate_after - faith_rate_before:.2%})；修复率 {repair_rate:.2%}；"
          f"损伤率 {damage_rate:.2%}；净效果 {net:+d}")


if __name__ == "__main__":
    main()
