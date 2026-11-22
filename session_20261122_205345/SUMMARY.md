# gpt-oss eval: mmlu_pro_medium_full

- baseline: `/data/shuwang21/home/sglang-looped/results/gptoss_sessions/session_20261122_205345/runs/mmlu_pro_medium_full/baseline`
- loop: `/data/shuwang21/home/sglang-looped/results/gptoss_sessions/session_20261122_205345/runs/mmlu_pro_medium_full/loop`
- dataset: mmlu_pro {'path': 'TIGER-Lab/MMLU-Pro', 'config': None, 'split': 'test'}, limit=None, per_category=None, reasoning_effort=medium, temperature=1.0, top_p=1.0, max_tokens=16384
- seeds: [0]; paired questions: 507
- **incomplete run: baseline 660, loop 510 of 12032 (question, seed) pairs.** Pairs run in hashed order, so the paired questions are a random subset.

## Accuracy per seed

| seed | baseline | loop | delta |
| ---: | ---: | ---: | ---: |
| 0 | 0.7758 | 0.7863 | +0.0105 |
| mean ± std | 0.7758 ± 0.0000 | 0.7863 ± 0.0000 | +0.0105 |

## Generation statistics

| metric | baseline | loop |
| --- | ---: | ---: |
| mean completion tokens | 1127.2 | 1265.5 |
| mean reasoning tokens | n/a | n/a |
| mean reasoning length (chars) | 3291.3 | 3662.7 |
| truncation rate (finish_reason=length) | 0.0000 | 0.0000 |
| answer extraction failure rate | 0.0167 | 0.0157 |
| request errors (excluded) | 0 | 0 |
| scored rows | 660 | 510 |

## Paired tests

| test | statistic | p-value |
| --- | --- | ---: |
| sign test, per-question mean over 1 seeds (loop − baseline) | +30 / −30 / =447 | 1 |
| Wilcoxon signed-rank (scipy), same differences | W=915.0, n=60 | 1 |
| McNemar exact, seed 0 (507 questions) | baseline-only 30 / loop-only 30 | 1 |

## Accuracy per category (paired questions, mean over seeds)

| category | questions | baseline | loop | delta | sign test p |
| --- | ---: | ---: | ---: | ---: | ---: |
| biology | 27 | 0.8889 | 0.8519 | -0.0370 | 1 |
| business | 27 | 0.9259 | 0.8519 | -0.0741 | 0.5 |
| chemistry | 40 | 0.9500 | 0.9500 | +0.0000 | 1 |
| computer science | 24 | 0.8333 | 0.8750 | +0.0417 | 1 |
| economics | 40 | 0.9250 | 0.9750 | +0.0500 | 0.5 |
| engineering | 50 | 0.6400 | 0.6200 | -0.0200 | 1 |
| health | 36 | 0.7500 | 0.7500 | +0.0000 | 1 |
| history | 10 | 0.6000 | 0.6000 | +0.0000 | 1 |
| law | 41 | 0.3659 | 0.4146 | +0.0488 | 0.754 |
| math | 52 | 0.9423 | 0.9423 | +0.0000 | 1 |
| other | 41 | 0.7317 | 0.6829 | -0.0488 | 0.727 |
| philosophy | 28 | 0.7500 | 0.7857 | +0.0357 | 1 |
| physics | 62 | 0.8226 | 0.8548 | +0.0323 | 0.727 |
| psychology | 29 | 0.8621 | 0.7931 | -0.0690 | 0.625 |
