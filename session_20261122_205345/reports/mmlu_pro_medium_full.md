# gpt-oss eval: mmlu_pro_medium_full

- baseline: `/data/shuwang21/home/sglang-looped/results/gptoss_sessions/session_20261122_205345/runs/mmlu_pro_medium_full/baseline`
- loop: `/data/shuwang21/home/sglang-looped/results/gptoss_sessions/session_20261122_205345/runs/mmlu_pro_medium_full/loop`
- dataset: mmlu_pro {'path': 'TIGER-Lab/MMLU-Pro', 'config': None, 'split': 'test'}, limit=None, per_category=None, reasoning_effort=medium, temperature=1.0, top_p=1.0, max_tokens=16384
- seeds: [0]; paired questions: 5015
- **incomplete run: baseline 6614, loop 5015 of 12032 (question, seed) pairs.** Pairs run in hashed order, so the paired questions are a random subset.

## Accuracy per seed

| seed | baseline | loop | delta |
| ---: | ---: | ---: | ---: |
| 0 | 0.7409 | 0.7551 | +0.0143 |
| mean ± std | 0.7409 ± 0.0000 | 0.7551 ± 0.0000 | +0.0143 |

## Generation statistics

| metric | baseline | loop |
| --- | ---: | ---: |
| mean completion tokens | 1237.5 | 1475.7 |
| mean reasoning tokens | n/a | n/a |
| mean reasoning length (chars) | 3669.8 | 4461.5 |
| truncation rate (finish_reason=length) | 0.0006 | 0.0054 |
| answer extraction failure rate | 0.0175 | 0.0211 |
| request errors (excluded) | 0 | 0 |
| scored rows | 6614 | 5015 |

## Paired tests

| test | statistic | p-value |
| --- | --- | ---: |
| sign test, per-question mean over 1 seeds (loop − baseline) | +351 / −294 / =4370 | 0.02738 |
| Wilcoxon signed-rank (scipy), same differences | W=94962.0, n=645 | 0.02481 |
| McNemar exact, seed 0 (5015 questions) | baseline-only 294 / loop-only 351 | 0.02738 |

## Accuracy per category (paired questions, mean over seeds)

| category | questions | baseline | loop | delta | sign test p |
| --- | ---: | ---: | ---: | ---: | ---: |
| biology | 296 | 0.8615 | 0.8547 | -0.0068 | 0.86 |
| business | 308 | 0.8182 | 0.8506 | +0.0325 | 0.11 |
| chemistry | 454 | 0.8216 | 0.8546 | +0.0330 | 0.0315 |
| computer science | 177 | 0.8136 | 0.8192 | +0.0056 | 1 |
| economics | 368 | 0.7935 | 0.8261 | +0.0326 | 0.073 |
| engineering | 403 | 0.6228 | 0.6427 | +0.0199 | 0.428 |
| health | 335 | 0.7493 | 0.7403 | -0.0090 | 0.743 |
| history | 167 | 0.6108 | 0.6766 | +0.0659 | 0.0522 |
| law | 459 | 0.4357 | 0.4510 | +0.0153 | 0.538 |
| math | 570 | 0.9281 | 0.9088 | -0.0193 | 0.117 |
| other | 414 | 0.6715 | 0.6763 | +0.0048 | 0.91 |
| philosophy | 200 | 0.6200 | 0.6550 | +0.0350 | 0.21 |
| physics | 536 | 0.8116 | 0.8078 | -0.0037 | 0.905 |
| psychology | 328 | 0.7439 | 0.7500 | +0.0061 | 0.871 |
