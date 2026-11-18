# -*- coding: utf-8 -*-
"""Run one correction experiment: read source -> call LLM to correct -> save -> score.

Usage:
  python src/run_experiment.py --provider deepseek --strategy direct [--limit 20]
"""
import argparse
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "eval"))

import config
import client
import prompts
import m2_scorer

RAW = os.path.join(ROOT, "data", "raw", "nlpcc2018", "extracted", "Data", "test")
RESULTS = os.path.join(ROOT, "results")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", required=True, choices=list(config.PROVIDERS))
    ap.add_argument("--strategy", required=True, choices=list(prompts.STRATEGIES))
    ap.add_argument("--limit", type=int, default=0, help="run only the first N sentences (0 = all 2000)")
    args = ap.parse_args()

    p = config.PROVIDERS[args.provider]
    if not p["api_key"]:
        print(f"Please fill the api_key for {args.provider} in config.py first.")
        sys.exit(1)

    def call(messages):
        return client.chat(p["base_url"], p["api_key"], p["model"], messages)

    srcs = m2_scorer.read_lines(os.path.join(RAW, "source.txt"))
    if args.limit:
        srcs = srcs[: args.limit]

    correct = prompts.STRATEGIES[args.strategy]
    out_dir = os.path.join(RESULTS, args.provider, args.strategy)
    os.makedirs(out_dir, exist_ok=True)

    hyps = []
    t0 = time.time()
    for i, s in enumerate(srcs):
        for attempt in range(3):
            try:
                hyps.append(correct(call, s))
                break
            except Exception as e:
                if attempt == 2:
                    print(f"[{i}] failed, keeping original: {e}")
                    hyps.append(s)
                else:
                    time.sleep(3)
        if (i + 1) % 20 == 0:
            print(f"  processed {i + 1}/{len(srcs)} ({time.time() - t0:.0f}s)", flush=True)

    hyp_path = os.path.join(out_dir, "hyp.txt")
    with open(hyp_path, "w", encoding="utf-8") as f:
        f.write("\n".join(hyps) + "\n")

    # Score against gold.0 as reference (can switch to gold.01.txt / gold.1.txt)
    golds = m2_scorer.read_lines(os.path.join(RAW, "gold", "gold.0.txt"))
    if args.limit:
        golds = golds[: args.limit]
    p_, r_, f_, _ = m2_scorer.score_corpus(zip(srcs, golds, hyps))
    print(f"\n[{args.provider}/{args.strategy}] P={p_:.4f} R={r_:.4f} F0.5={f_:.4f}")
    print("(reference: inter-annotator agreement F0.5 ~= 0.9405)")
    print(f"results saved: {hyp_path}")


if __name__ == "__main__":
    main()
