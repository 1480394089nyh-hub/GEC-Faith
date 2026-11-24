# -*- coding: utf-8 -*-
"""反馈再生成实验：用判别器挑出「不忠实」改句，带失败类型喂回同一模型重改。

流程（对应 证据链与实验方案.md 的「反馈再生成」）：
  原句 → 大模型改句 → 判别器(bin)判「不忠实」 → multi 给具体失败类型
       → 反馈提示喂回同一模型 → 出新改句 → 人工标注新改句的忠实性

用法：
  python src/refine.py [--limit N] [--out 结果.csv] [--workers 6]

产出：一个 CSV，含「原句/上次改句/诊断失败类型/新改句 + 空的新改句标注列」，
     供标注员回标；随后由 scripts/eval_refine.py 算修复率（前后对比）。
"""
import argparse
import csv
import os
import sys
import time
import concurrent.futures

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

import config
import client
import prompts

BIN_PRED = os.path.join(ROOT, "results", "discriminator", "bin", "test_predictions.csv")
MULTI_PRED = os.path.join(ROOT, "results", "discriminator", "multi", "test_predictions.csv")

# 失败类型 → 反馈提示里的一句话说明
FAIL_HINT = {
    "漏改": "还存在语法错误没有改正（漏改）",
    "语义漂移": "改变了原句的意思（语义漂移）",
    "过度纠正": "把原本正确的部分改错了（过度纠正）",
    "幻觉": "凭空添加了原文没有的内容（幻觉）",
}

SYSTEM = "你是一名专业的中文语法纠错助手。"


def feedback_messages(src, hyp, fail_type):
    if fail_type in FAIL_HINT:
        problem = f"这次修改存在「{fail_type}」的问题：{FAIL_HINT[fail_type]}。"
    else:
        problem = "这次修改可能不够忠实，可能存在改错、改偏或遗漏的问题。"
    user = (
        f"原句：{src}\n"
        f"你上一次的修改是：{hyp}\n"
        f"{problem}\n"
        "请重新改正原句中的语法错误，要求：只改正真正的语法错误、保持原句意思不变、"
        "不要遗漏错误、不要添加原文没有的内容。\n"
        "只输出改正后的完整句子本身，不要任何解释、说明、前缀或引号。"
    )
    return [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": user},
    ]


def load_csv(path):
    with open(path, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="只处理前 N 条（0=全部 119 条）")
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "refine", "重改结果.csv"))
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()

    bin_rows = load_csv(BIN_PRED)
    multi_rows = load_csv(MULTI_PRED)
    multi_type = {r["序号"]: r["预测标签"] for r in multi_rows}

    # bin 判「不忠实」的实例（= 闸门拦下的 119 条）
    selected = [r for r in bin_rows if r["预测标签"] == "不忠实"]
    if args.limit:
        selected = selected[: args.limit]

    # 逐模型统计
    from collections import Counter
    by_model = Counter(r["模型"] for r in selected)
    by_type = Counter(multi_type.get(r["序号"], "(bin/multi不一致)") for r in selected)
    print(f"选中 {len(selected)} 条（bin 判不忠实）：按模型 {dict(by_model)}；按诊断类型 {dict(by_type)}")

    active = {name: p for name, p in config.PROVIDERS.items() if p["api_key"]}
    if not active:
        print("没有可用的模型 key，请先填 config.py")
        sys.exit(1)

    def process(r):
        model = r["模型"]
        p = active.get(model)
        if not p:
            return r, f"模型 {model} 无 key"
        fail_type = multi_type.get(r["序号"], "")
        msgs = feedback_messages(r["原句"], r["改句"], fail_type)
        for attempt in range(3):
            try:
                new_hyp = prompts._clean(client.chat(p["base_url"], p["api_key"], p["model"], msgs))
                return r, new_hyp
            except Exception as e:
                if attempt == 2:
                    return r, f"API失败: {type(e).__name__}"
                time.sleep(2)
        return r, "重试耗尽"

    results = {}
    t0 = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(process, r): r for r in selected}
        done = 0
        for fut in concurrent.futures.as_completed(futs):
            r, new_hyp = fut.result()
            results[r["序号"]] = (r, new_hyp)
            done += 1
            if done % 20 == 0 or done == len(selected):
                print(f"  {done}/{len(selected)} ({time.time() - t0:.0f}s)", flush=True)

    out_dir = os.path.dirname(args.out)
    os.makedirs(out_dir, exist_ok=True)
    cols = ["序号", "句子序号", "模型", "诊断失败类型", "原句", "上次改句", "上次真实标签", "新改句", "新改句标注", "备注"]
    with open(args.out, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for r in selected:
            new_hyp = results[r["序号"]][1]
            if new_hyp.startswith("API失败") or new_hyp == "重试耗尽":
                remark, new_hyp = new_hyp, ""
            else:
                remark = ""
            w.writerow([
                r["序号"], r["句子序号"], r["模型"],
                multi_type.get(r["序号"], ""), r["原句"], r["改句"],
                r["真实标签"], new_hyp, "", remark,
            ])
    print(f"已保存: {args.out}")


if __name__ == "__main__":
    main()
