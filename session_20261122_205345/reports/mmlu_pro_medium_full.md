# gpt-oss eval: mmlu_pro_medium_full

- baseline: `/data/shuwang21/home/sglang-looped/results/gptoss_sessions/session_20261122_205345/runs/mmlu_pro_medium_full/baseline`
- loop: `/data/shuwang21/home/sglang-looped/results/gptoss_sessions/session_20261122_205345/runs/mmlu_pro_medium_full/loop`
- dataset: mmlu_pro {'path': 'TIGER-Lab/MMLU-Pro', 'config': None, 'split': 'test'}, limit=None, per_category=None, reasoning_effort=medium, temperature=1.0, top_p=1.0, max_tokens=16384
- seeds: [0]; paired questions: 3350
- **incomplete run: baseline 4522, loop 3350 of 12032 (question, seed) pairs.** Pairs run in hashed order, so the paired questions are a random subset.

## Accuracy per seed

| seed | baseline | loop | delta |
| ---: | ---: | ---: | ---: |
| 0 | 0.7397 | 0.7612 | +0.0215 |
| mean ± std | 0.7397 ± 0.0000 | 0.7612 ± 0.0000 | +0.0215 |

## Generation statistics

| metric | baseline | loop |
| --- | ---: | ---: |
| mean completion tokens | 1210.1 | 1464.2 |
| mean reasoning tokens | n/a | n/a |
| mean reasoning length (chars) | 3584.0 | 4420.3 |
| truncation rate (finish_reason=length) | 0.0002 | 0.0051 |
| answer extraction failure rate | 0.0166 | 0.0203 |
| request errors (excluded) | 0 | 0 |
| scored rows | 4522 | 3350 |

## Paired tests

| test | statistic | p-value |
| --- | --- | ---: |
| sign test, per-question mean over 1 seeds (loop − baseline) | +226 / −197 / =2927 | 0.1733 |
| Wilcoxon signed-rank (scipy), same differences | W=41764.0, n=423 | 0.1585 |
| McNemar exact, seed 0 (3350 questions) | baseline-only 197 / loop-only 226 | 0.1733 |

## Accuracy per category (paired questions, mean over seeds)

| category | questions | baseline | loop | delta | sign test p |
| --- | ---: | ---: | ---: | ---: | ---: |
| biology | 201 | 0.8756 | 0.8607 | -0.0149 | 0.664 |
| business | 206 | 0.8301 | 0.8689 | +0.0388 | 0.0963 |
| chemistry | 318 | 0.8302 | 0.8616 | +0.0314 | 0.0755 |
| computer science | 113 | 0.8319 | 0.8496 | +0.0177 | 0.727 |
| economics | 244 | 0.8115 | 0.8320 | +0.0205 | 0.383 |
| engineering | 273 | 0.6337 | 0.6374 | +0.0037 | 1 |
| health | 220 | 0.7773 | 0.7545 | -0.0227 | 0.383 |
| history | 103 | 0.6214 | 0.6990 | +0.0777 | 0.0574 |
| law | 309 | 0.4337 | 0.4660 | +0.0324 | 0.282 |
| math | 380 | 0.9184 | 0.8921 | -0.0263 | 0.11 |
| other | 280 | 0.6750 | 0.6750 | +0.0000 | 1 |
| philosophy | 135 | 0.6370 | 0.6741 | +0.0370 | 0.267 |
| physics | 350 | 0.8171 | 0.8029 | -0.0143 | 0.551 |
| psychology | 218 | 0.7615 | 0.7752 | +0.0138 | 0.678 |
