# gpt-oss eval: mmlu_pro_medium_full

- baseline: `/data/shuwang21/home/sglang-looped/results/gptoss_sessions/session_20261122_205345/runs/mmlu_pro_medium_full/baseline`
- loop: `/data/shuwang21/home/sglang-looped/results/gptoss_sessions/session_20261122_205345/runs/mmlu_pro_medium_full/loop`
- dataset: mmlu_pro {'path': 'TIGER-Lab/MMLU-Pro', 'config': None, 'split': 'test'}, limit=None, per_category=None, reasoning_effort=medium, temperature=1.0, top_p=1.0, max_tokens=16384
- seeds: [0]; paired questions: 3880
- **incomplete run: baseline 5280, loop 3880 of 12032 (question, seed) pairs.** Pairs run in hashed order, so the paired questions are a random subset.

## Accuracy per seed

| seed | baseline | loop | delta |
| ---: | ---: | ---: | ---: |
| 0 | 0.7428 | 0.7562 | +0.0134 |
| mean ± std | 0.7428 ± 0.0000 | 0.7562 ± 0.0000 | +0.0134 |

## Generation statistics

| metric | baseline | loop |
| --- | ---: | ---: |
| mean completion tokens | 1205.1 | 1478.8 |
| mean reasoning tokens | n/a | n/a |
| mean reasoning length (chars) | 3574.3 | 4450.3 |
| truncation rate (finish_reason=length) | 0.0004 | 0.0052 |
| answer extraction failure rate | 0.0159 | 0.0201 |
| request errors (excluded) | 0 | 0 |
| scored rows | 5280 | 3880 |

## Paired tests

| test | statistic | p-value |
| --- | --- | ---: |
| sign test, per-question mean over 1 seeds (loop − baseline) | +265 / −227 / =3388 | 0.0952 |
| Wilcoxon signed-rank (scipy), same differences | W=55955.5, n=492 | 0.08668 |
| McNemar exact, seed 0 (3880 questions) | baseline-only 227 / loop-only 265 | 0.0952 |

## Accuracy per category (paired questions, mean over seeds)

| category | questions | baseline | loop | delta | sign test p |
| --- | ---: | ---: | ---: | ---: | ---: |
| biology | 233 | 0.8712 | 0.8541 | -0.0172 | 0.541 |
| business | 230 | 0.8087 | 0.8522 | +0.0435 | 0.0414 |
| chemistry | 359 | 0.8273 | 0.8607 | +0.0334 | 0.0501 |
| computer science | 134 | 0.8209 | 0.8358 | +0.0149 | 0.727 |
| economics | 286 | 0.8042 | 0.8287 | +0.0245 | 0.248 |
| engineering | 323 | 0.6378 | 0.6471 | +0.0093 | 0.804 |
| health | 257 | 0.7626 | 0.7432 | -0.0195 | 0.442 |
| history | 130 | 0.6385 | 0.6769 | +0.0385 | 0.332 |
| law | 350 | 0.4229 | 0.4543 | +0.0314 | 0.248 |
| math | 441 | 0.9252 | 0.8980 | -0.0272 | 0.0576 |
| other | 321 | 0.6667 | 0.6636 | -0.0031 | 1 |
| philosophy | 154 | 0.6039 | 0.6558 | +0.0519 | 0.0768 |
| physics | 410 | 0.8171 | 0.8122 | -0.0049 | 0.892 |
| psychology | 252 | 0.7421 | 0.7579 | +0.0159 | 0.557 |
