# gpt-oss eval: mmlu_pro_medium_full

- baseline: `/data/shuwang21/home/sglang-looped/results/gptoss_sessions/session_20261122_205345/runs/mmlu_pro_medium_full/baseline`
- loop: `/data/shuwang21/home/sglang-looped/results/gptoss_sessions/session_20261122_205345/runs/mmlu_pro_medium_full/loop`
- dataset: mmlu_pro {'path': 'TIGER-Lab/MMLU-Pro', 'config': None, 'split': 'test'}, limit=None, per_category=None, reasoning_effort=medium, temperature=1.0, top_p=1.0, max_tokens=16384
- seeds: [0]; paired questions: 1095
- **incomplete run: baseline 1485, loop 1095 of 12032 (question, seed) pairs.** Pairs run in hashed order, so the paired questions are a random subset.

## Accuracy per seed

| seed | baseline | loop | delta |
| ---: | ---: | ---: | ---: |
| 0 | 0.7603 | 0.7772 | +0.0169 |
| mean ± std | 0.7603 ± 0.0000 | 0.7772 ± 0.0000 | +0.0169 |

## Generation statistics

| metric | baseline | loop |
| --- | ---: | ---: |
| mean completion tokens | 1188.2 | 1402.1 |
| mean reasoning tokens | n/a | n/a |
| mean reasoning length (chars) | 3533.8 | 4177.2 |
| truncation rate (finish_reason=length) | 0.0007 | 0.0037 |
| answer extraction failure rate | 0.0141 | 0.0210 |
| request errors (excluded) | 0 | 0 |
| scored rows | 1485 | 1095 |

## Paired tests

| test | statistic | p-value |
| --- | --- | ---: |
| sign test, per-question mean over 1 seeds (loop − baseline) | +74 / −65 / =956 | 0.4976 |
| Wilcoxon signed-rank (scipy), same differences | W=4550.0, n=139 | 0.4452 |
| McNemar exact, seed 0 (1095 questions) | baseline-only 65 / loop-only 74 | 0.4976 |

## Accuracy per category (paired questions, mean over seeds)

| category | questions | baseline | loop | delta | sign test p |
| --- | ---: | ---: | ---: | ---: | ---: |
| biology | 64 | 0.9375 | 0.9219 | -0.0156 | 1 |
| business | 64 | 0.8906 | 0.9062 | +0.0156 | 1 |
| chemistry | 93 | 0.8710 | 0.8925 | +0.0215 | 0.754 |
| computer science | 46 | 0.8261 | 0.8478 | +0.0217 | 1 |
| economics | 71 | 0.9014 | 0.9014 | +0.0000 | 1 |
| engineering | 102 | 0.5490 | 0.5784 | +0.0294 | 0.664 |
| health | 83 | 0.7952 | 0.8072 | +0.0120 | 1 |
| history | 25 | 0.6400 | 0.7200 | +0.0800 | 0.625 |
| law | 84 | 0.3929 | 0.4405 | +0.0476 | 0.481 |
| math | 124 | 0.9113 | 0.9113 | +0.0000 | 1 |
| other | 88 | 0.7273 | 0.6818 | -0.0455 | 0.523 |
| philosophy | 56 | 0.6429 | 0.6964 | +0.0536 | 0.453 |
| physics | 125 | 0.8080 | 0.7920 | -0.0160 | 0.791 |
| psychology | 70 | 0.8143 | 0.8000 | -0.0143 | 1 |
