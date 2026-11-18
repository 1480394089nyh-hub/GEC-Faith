# -*- coding: utf-8 -*-
"""Batch collection: for a set of error sentences, use every model with a
configured API key to generate candidate corrections, and dump a TSV for annotation.

Usage:
  python src/collect.py [--limit N] [--source src.txt] [--out collect.tsv] [--workers 6]
"""
import argparse
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

DEFAULT_SRC = os.path.join(ROOT, "data", "raw", "nlpcc2018",
                           "extracted", "Data", "test", "source.txt")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="collect only the first N sentences (0 = all)")
    ap.add_argument("--source", default=DEFAULT_SRC)
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "collect.tsv"))
    ap.add_argument("--workers", type=int, default=6, help="number of worker threads")
    args = ap.parse_args()

    active = [(name, p) for name, p in config.PROVIDERS.items() if p["api_key"]]
    if not active:
        print("No model with a configured api_key; please fill config.py first.")
        sys.exit(1)
    print(f"Models used: {', '.join(n for n, _ in active)} | {args.workers} threads")

    srcs = open(args.source, encoding="utf-8-sig").read().splitlines()
    if args.limit:
        srcs = srcs[: args.limit]
    print(f"To collect: {len(srcs)} sentences x {len(active)} models")

    cols = {}  # provider -> list of corrections
    for name, p in active:
        def call(messages):
            return client.chat(p["base_url"], p["api_key"], p["model"], messages)

        def process(item):
            i, s = item
            for attempt in range(3):
                try:
                    return i, prompts.direct(call, s)
                except Exception:
                    if attempt == 2:
                        print(f"  [{name}] sentence {i} failed, keeping original", flush=True)
                        return i, s
                    time.sleep(2)
            return i, s

        t0 = time.time()
        results = {}
        items = list(enumerate(srcs))
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = [ex.submit(process, it) for it in items]
            done = 0
            for fut in concurrent.futures.as_completed(futs):
                i, hyp = fut.result()
                results[i] = hyp
                done += 1
                if done % 200 == 0:
                    print(f"  [{name}] {done}/{len(srcs)} ({time.time() - t0:.0f}s)", flush=True)
        hyps = [results[i] for i in range(len(srcs))]
        unchanged = sum(1 for i, s in enumerate(srcs) if hyps[i] == s)
        cols[name] = hyps
        print(f"  [{name}] done {len(hyps)} sentences in {time.time() - t0:.0f}s, {unchanged} unchanged")

    header = ["source"] + [n for n, _ in active]
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write("\t".join(header) + "\n")
        for i, s in enumerate(srcs):
            row = [s] + [cols[n][i] for n, _ in active]
            f.write("\t".join(row) + "\n")
    print(f"Saved: {args.out} ({len(srcs)} rows)")


if __name__ == "__main__":
    main()
