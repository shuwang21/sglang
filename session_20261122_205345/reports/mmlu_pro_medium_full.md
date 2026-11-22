# gpt-oss eval: mmlu_pro_medium_full

- baseline: `/data/shuwang21/home/sglang-looped/results/gptoss_sessions/session_20261122_205345/runs/mmlu_pro_medium_full/baseline`
- loop: `/data/shuwang21/home/sglang-looped/results/gptoss_sessions/session_20261122_205345/runs/mmlu_pro_medium_full/loop`
- dataset: mmlu_pro {'path': 'TIGER-Lab/MMLU-Pro', 'config': None, 'split': 'test'}, limit=None, per_category=None, reasoning_effort=medium, temperature=1.0, top_p=1.0, max_tokens=16384
- seeds: [0]; paired questions: 6066
- **incomplete run: baseline 8087, loop 6066 of 12032 (question, seed) pairs.** Pairs run in hashed order, so the paired questions are a random subset.

## Accuracy per seed

| seed | baseline | loop | delta |
| ---: | ---: | ---: | ---: |
| 0 | 0.7364 | 0.7517 | +0.0154 |
| mean ± std | 0.7364 ± 0.0000 | 0.7517 ± 0.0000 | +0.0154 |

## Generation statistics

| metric | baseline | loop |
| --- | ---: | ---: |
| mean completion tokens | 1236.5 | 1494.6 |
| mean reasoning tokens | n/a | n/a |
| mean reasoning length (chars) | 3668.8 | 4500.2 |
| truncation rate (finish_reason=length) | 0.0009 | 0.0058 |
| answer extraction failure rate | 0.0182 | 0.0216 |
| request errors (excluded) | 0 | 0 |
| scored rows | 8087 | 6066 |

## Paired tests

| test | statistic | p-value |
| --- | --- | ---: |
| sign test, per-question mean over 1 seeds (loop − baseline) | +427 / −364 / =5275 | 0.02743 |
| Wilcoxon signed-rank (scipy), same differences | W=144144.0, n=791 | 0.02509 |
| McNemar exact, seed 0 (6066 questions) | baseline-only 364 / loop-only 427 | 0.02743 |

## Accuracy per category (paired questions, mean over seeds)

| category | questions | baseline | loop | delta | sign test p |
| --- | ---: | ---: | ---: | ---: | ---: |
| biology | 357 | 0.8627 | 0.8627 | +0.0000 | 1 |
| business | 385 | 0.8104 | 0.8468 | +0.0364 | 0.0385 |
| chemistry | 571 | 0.8109 | 0.8389 | +0.0280 | 0.044 |
| computer science | 214 | 0.8271 | 0.8364 | +0.0093 | 0.791 |
| economics | 439 | 0.7995 | 0.8246 | +0.0251 | 0.135 |
| engineering | 492 | 0.6098 | 0.6301 | +0.0203 | 0.348 |
| health | 409 | 0.7335 | 0.7311 | -0.0024 | 1 |
| history | 198 | 0.6212 | 0.6818 | +0.0606 | 0.0652 |
| law | 568 | 0.4489 | 0.4489 | +0.0000 | 1 |
| math | 676 | 0.9246 | 0.9053 | -0.0192 | 0.0984 |
| other | 478 | 0.6590 | 0.6653 | +0.0063 | 0.834 |
| philosophy | 236 | 0.6059 | 0.6356 | +0.0297 | 0.281 |
| physics | 651 | 0.8203 | 0.8233 | +0.0031 | 0.911 |
| psychology | 392 | 0.7423 | 0.7423 | +0.0000 | 1 |
