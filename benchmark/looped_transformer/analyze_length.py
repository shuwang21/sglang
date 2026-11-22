# Copyright 2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License").

"""Is the loop's accuracy gain explained by longer reasoning?

Reads two eval_gptoss.py result directories (baseline, loop) and compares
them question by question, for every seed they share:

- accuracy and completion tokens, and the per-question length ratio
  loop / baseline;
- the length ratio inside each outcome group (loop-only correct,
  baseline-only correct, both correct, both wrong);
- the accuracy change in tertiles of the length ratio, and the correlation
  between the change in correctness and the log length ratio;
- accuracy per arm within pooled length quintiles, and the loop accuracy
  re-weighted to the baseline's length distribution.

With two or more seeds it also compares seed 0 with seed 1 inside each arm.
That is pure sampling noise and is the reference for the numbers above.

If the gain came from thinking longer, the loop-only-correct questions would
show a larger length ratio than the rest, and the gain would sit in the
"longer" tertile. Caveat for the quintile table: length is itself an outcome
of the loop, so comparing at equal length mixes in a composition effect
(the loop moves easier questions into longer bins). It can argue against a
pure length explanation, not estimate the effect size.

    python benchmark/looped_transformer/analyze_length.py BASELINE_DIR LOOP_DIR [--seeds 0 1]
"""

import argparse
import json
import math
import statistics as st
from pathlib import Path


def load(directory):
    rows = {}
    with (Path(directory) / "samples.jsonl").open() as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                if not row.get("error") and row.get("completion_tokens"):
                    rows[(row["id"], row["seed"])] = row
    return rows


def pearson(x, y):
    mx, my = st.fmean(x), st.fmean(y)
    sx = math.sqrt(sum((a - mx) ** 2 for a in x))
    sy = math.sqrt(sum((b - my) ** 2 for b in y))
    if not sx or not sy:
        return float("nan")
    return sum((a - mx) * (b - my) for a, b in zip(x, y)) / (sx * sy)


def compare(label, A, sa, B, sb, names=("baseline", "loop")):
    """B relative to A, on the questions both answered."""

    qids = sorted({q for q, s in A if s == sa} & {q for q, s in B if s == sb})
    if not qids:
        print(f"\n=== {label}: no common questions")
        return None
    ta = [A[(q, sa)]["completion_tokens"] for q in qids]
    tb = [B[(q, sb)]["completion_tokens"] for q in qids]
    ca = [int(A[(q, sa)]["correct"]) for q in qids]
    cb = [int(B[(q, sb)]["correct"]) for q in qids]
    lr = [math.log(y / x) for x, y in zip(ta, tb)]
    dc = [y - x for x, y in zip(ca, cb)]
    n = len(qids)
    a_name, b_name = names
    print(f"\n=== {label} (n={n}; A={a_name}, B={b_name})")
    print(
        f"accuracy {st.fmean(ca):.4f} -> {st.fmean(cb):.4f} "
        f"({st.fmean(dc) * 100:+.2f} pp); mean tokens {st.fmean(ta):.0f} -> "
        f"{st.fmean(tb):.0f} ({(st.fmean(tb) / st.fmean(ta) - 1) * 100:+.1f}%); "
        f"median per-question ratio B/A {math.exp(st.median(lr)):.3f}"
    )
    groups = {
        "B-only correct": [i for i in range(n) if cb[i] and not ca[i]],
        "A-only correct": [i for i in range(n) if ca[i] and not cb[i]],
        "both correct": [i for i in range(n) if ca[i] and cb[i]],
        "both wrong": [i for i in range(n) if not ca[i] and not cb[i]],
    }
    print("  group              n   mean tok A   mean tok B   median ratio B/A")
    for name, idx in groups.items():
        if idx:
            print(
                f"  {name:15s} {len(idx):5d}   {st.fmean(ta[i] for i in idx):9.0f}"
                f"   {st.fmean(tb[i] for i in idx):9.0f}"
                f"   {math.exp(st.median(lr[i] for i in idx)):9.3f}"
            )
    order = sorted(range(n), key=lambda i: lr[i])
    print(
        "  B/A length tertile   ratio range       n   delta acc (pp)   B-only / A-only"
    )
    for t, tag in enumerate(("shorter", "similar", "longer")):
        idx = order[t * n // 3 : (t + 1) * n // 3]
        if not idx:
            continue
        b_only = sum(1 for i in idx if cb[i] and not ca[i])
        a_only = sum(1 for i in idx if ca[i] and not cb[i])
        print(
            f"  {tag:8s}   {math.exp(lr[idx[0]]):6.2f} .. {math.exp(lr[idx[-1]]):6.2f}"
            f"   {len(idx):5d}   {st.fmean(dc[i] for i in idx) * 100:+8.2f}"
            f"         {b_only} / {a_only}"
        )
    print(f"  corr(delta correct, log length ratio) = {pearson(dc, lr):+.3f}")
    return dict(ta=ta, tb=tb, ca=ca, cb=cb)


def length_matched(data):
    pooled = sorted(data["ta"] + data["tb"])
    edges = [pooled[len(pooled) * k // 5] for k in range(1, 5)]

    def bin_of(tokens):
        return sum(tokens >= edge for edge in edges)

    print(f"  accuracy within pooled length quintiles (token edges {edges}):")
    print("  quintile    A n   A acc     B n   B acc   B-A (pp)")
    reweighted = 0.0
    for k in range(5):
        a = [c for t, c in zip(data["ta"], data["ca"]) if bin_of(t) == k]
        b = [c for t, c in zip(data["tb"], data["cb"]) if bin_of(t) == k]
        if a and b:
            print(
                f"  Q{k + 1}       {len(a):5d}  {st.fmean(a):.4f}  {len(b):5d}  "
                f"{st.fmean(b):.4f}  {(st.fmean(b) - st.fmean(a)) * 100:+7.2f}"
            )
            reweighted += len(a) / len(data["ta"]) * st.fmean(b)
    print(
        f"  B re-weighted to A's length distribution: {reweighted:.4f} "
        f"(A {st.fmean(data['ca']):.4f}, raw B {st.fmean(data['cb']):.4f}); "
        "biased by composition, see the module docstring"
    )


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("baseline")
    parser.add_argument("loop")
    parser.add_argument(
        "--seeds", type=int, nargs="+", help="default: all common seeds"
    )
    args = parser.parse_args(argv)

    base, loop = load(args.baseline), load(args.loop)
    seeds = args.seeds or sorted({s for _, s in base} & {s for _, s in loop})
    for seed in seeds:
        data = compare(f"seed {seed}: baseline vs loop", base, seed, loop, seed)
        if data:
            length_matched(data)
    if len(seeds) >= 2:
        s0, s1 = seeds[:2]
        compare(
            f"NULL: baseline seed {s0} vs seed {s1}",
            base,
            s0,
            base,
            s1,
            names=(f"baseline s{s0}", f"baseline s{s1}"),
        )
        compare(
            f"NULL: loop seed {s0} vs seed {s1}",
            loop,
            s0,
            loop,
            s1,
            names=(f"loop s{s0}", f"loop s{s1}"),
        )


if __name__ == "__main__":
    main()
