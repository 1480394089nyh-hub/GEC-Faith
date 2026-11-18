# -*- coding: utf-8 -*-
"""Character-level MaxMatch F0.5 scorer (Chinese grammatical error correction, GEC).

Background
----------
GEC turns an "error sentence" into a "correct sentence". We evaluate a model by
how well its edits match the gold edits. Following Chinese GEC convention, this
script compares at the **character** level (no word segmentation).

Scoring
-------
For each sentence, compute:
  original -> gold   edit set (gold edits)
  original -> model  edit set (system edits)
then match by "edit span identity":
  - precision P = matched edits / system edits
  - recall R    = matched edits / gold edits
  - F0.5        = a precision-weighted score ("wrong edits" are worse than
                  "missed edits" in GEC).

Note: two edits match if their spans are equal, even if the replacement text
differs. This is the GEC scoring convention, to avoid penalizing equivalent
alternative edits.

Usage
-----
  python src/eval/m2_scorer.py --self-test                 # run built-in self test
  python src/eval/m2_scorer.py src.txt gold.txt hyp.txt    # three files, line-aligned

Note: this is a simplified MaxMatch for quick experiments. For final paper
numbers, verify with chERRANT (the MuCGEC repo).
"""

import difflib
import sys


def get_edits(src, tgt):
    """Return the minimal character edits to turn src into tgt, as a list of (start, end, repl).

    start/end is the [start, end) span to replace in src; repl is the replacement text.
    repl == "" means deletion; start == end means pure insertion.
    """
    sm = difflib.SequenceMatcher(a=src, b=tgt, autojunk=False)
    edits = []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            continue
        edits.append((i1, i2, tgt[j1:j2]))
    return edits


def f_beta(precision, recall, beta=0.5):
    """F_beta score. With beta=0.5 precision is weighted higher."""
    if precision + recall == 0:
        return 0.0
    beta2 = beta * beta
    return (1.0 + beta2) * precision * recall / (beta2 * precision + recall)


def score_one(src, gold, hyp):
    """Score one sentence; returns (matched, n_gold, n_sys) counts for aggregation."""
    gold_edits = get_edits(src, gold)
    sys_edits = get_edits(src, hyp)

    gold_spans = [(s, e) for (s, e, _) in gold_edits]
    sys_spans = [(s, e) for (s, e, _) in sys_edits]

    # Match each system edit to an unused gold edit with the same span.
    used = [False] * len(gold_spans)
    matched = 0
    for span in sys_spans:
        for i, gspan in enumerate(gold_spans):
            if not used[i] and gspan == span:
                used[i] = True
                matched += 1
                break

    return matched, len(gold_spans), len(sys_spans)


def score_corpus(rows):
    """rows is [(src, gold, hyp), ...]; returns overall (P, R, F0.5, counts)."""
    tp = fp = fn = 0
    for src, gold, hyp in rows:
        matched, n_gold, n_sys = score_one(src, gold, hyp)
        tp += matched
        fn += n_gold - matched
        fp += n_sys - matched
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f = f_beta(precision, recall, beta=0.5)
    return precision, recall, f, (tp, fp, fn)


def self_test():
    # Example: two errors in the source, model fixes only one.
    src = "我不有这本书，而且也没看国"
    gold = "我没有这本书，而且也没看过"
    hyp = "我没有这本书，而且也没看国"
    p, r, f, _ = score_corpus([(src, gold, hyp)])
    print("self-test 1 (2 errors, 1 corrected):")
    print(f"  P={p:.4f}  R={r:.4f}  F0.5={f:.4f}   (expected P=1.0, R=0.5, F0.5~=0.8333)")

    # Example: all errors corrected.
    hyp2 = "我没有这本书，而且也没看过"
    p2, r2, f2, _ = score_corpus([(src, gold, hyp2)])
    print("self-test 2 (all corrected):")
    print(f"  P={p2:.4f}  R={r2:.4f}  F0.5={f2:.4f}   (expected 1.0/1.0/1.0)")

    # Example: a correct sentence edited wrongly -> precision drops.
    src3 = "今天天气很好"
    gold3 = "今天天气很好"
    hyp3 = "今天天气很豪"
    p3, r3, f3, _ = score_corpus([(src3, gold3, hyp3)])
    print("self-test 3 (correct sentence, wrong edit):")
    print(f"  P={p3:.4f}  R={r3:.4f}  F0.5={f3:.4f}   (expected P=0, R=0, F0.5=0)")

    ok = (
        abs(f - 0.8333) < 0.01
        and abs(f2 - 1.0) < 1e-9
        and f3 == 0.0
    )
    print("self-test result:", "PASS" if ok else "FAIL")
    return ok


def read_lines(path):
    with open(path, encoding="utf-8-sig") as fh:
        return [ln.rstrip("\r\n") for ln in fh]


def main(argv):
    if len(argv) == 2 and argv[1] == "--self-test":
        sys.exit(0 if self_test() else 1)
    if len(argv) != 4:
        print("usage: python src/eval/m2_scorer.py src.txt gold.txt hyp.txt")
        print("       python src/eval/m2_scorer.py --self-test")
        sys.exit(2)
    srcs = read_lines(argv[1])
    golds = read_lines(argv[2])
    hyps = read_lines(argv[3])
    if not (len(srcs) == len(golds) == len(hyps)):
        print("error: the three files have different line counts", file=sys.stderr)
        sys.exit(2)
    p, r, f, (tp, fp, fn) = score_corpus(zip(srcs, golds, hyps))
    print(f"sentences: {len(srcs)}")
    print(f"P    = {p:.4f}")
    print(f"R    = {r:.4f}")
    print(f"F0.5 = {f:.4f}")
    print(f"(matched {tp}, extra edits {fp}, missed {fn})")


if __name__ == "__main__":
    main(sys.argv)
