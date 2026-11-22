# gpt-oss eval: mmlu_pro_medium_full

- baseline: `/data/shuwang21/home/sglang-looped/results/gptoss_sessions/session_20261122_205345/runs/mmlu_pro_medium_full/baseline`
- loop: `/data/shuwang21/home/sglang-looped/results/gptoss_sessions/session_20261122_205345/runs/mmlu_pro_medium_full/loop`
- dataset: mmlu_pro {'path': 'TIGER-Lab/MMLU-Pro', 'config': None, 'split': 'test'}, limit=None, per_category=None, reasoning_effort=medium, temperature=1.0, top_p=1.0, max_tokens=16384
- seeds: [0]; paired questions: 5567
- **incomplete run: baseline 7393, loop 5567 of 12032 (question, seed) pairs.** Pairs run in hashed order, so the paired questions are a random subset.

## Accuracy per seed

| seed | baseline | loop | delta |
| ---: | ---: | ---: | ---: |
| 0 | 0.7393 | 0.7548 | +0.0155 |
| mean ± std | 0.7393 ± 0.0000 | 0.7548 ± 0.0000 | +0.0155 |

## Generation statistics

| metric | baseline | loop |
| --- | ---: | ---: |
| mean completion tokens | 1233.0 | 1475.6 |
| mean reasoning tokens | n/a | n/a |
| mean reasoning length (chars) | 3657.6 | 4451.5 |
| truncation rate (finish_reason=length) | 0.0008 | 0.0054 |
| answer extraction failure rate | 0.0179 | 0.0212 |
| request errors (excluded) | 0 | 0 |
| scored rows | 7393 | 5567 |

## Paired tests

| test | statistic | p-value |
| --- | --- | ---: |
| sign test, per-question mean over 1 seeds (loop − baseline) | +392 / −332 / =4843 | 0.02826 |
| Wilcoxon signed-rank (scipy), same differences | W=120350.0, n=724 | 0.02576 |
| McNemar exact, seed 0 (5567 questions) | baseline-only 332 / loop-only 392 | 0.02826 |

## Accuracy per category (paired questions, mean over seeds)

| category | questions | baseline | loop | delta | sign test p |
| --- | ---: | ---: | ---: | ---: | ---: |
| biology | 334 | 0.8623 | 0.8593 | -0.0030 | 1 |
| business | 342 | 0.8129 | 0.8421 | +0.0292 | 0.132 |
| chemistry | 517 | 0.8182 | 0.8511 | +0.0329 | 0.0241 |
| computer science | 196 | 0.8214 | 0.8316 | +0.0102 | 0.791 |
| economics | 405 | 0.8000 | 0.8247 | +0.0247 | 0.164 |
| engineering | 441 | 0.6145 | 0.6417 | +0.0272 | 0.23 |
| health | 376 | 0.7394 | 0.7367 | -0.0027 | 1 |
| history | 182 | 0.6154 | 0.6758 | +0.0604 | 0.0708 |
| law | 513 | 0.4425 | 0.4444 | +0.0019 | 1 |
| math | 628 | 0.9299 | 0.9092 | -0.0207 | 0.0789 |
| other | 447 | 0.6689 | 0.6756 | +0.0067 | 0.828 |
| philosophy | 220 | 0.6136 | 0.6545 | +0.0409 | 0.122 |
| physics | 597 | 0.8157 | 0.8141 | -0.0017 | 1 |
| psychology | 369 | 0.7453 | 0.7480 | +0.0027 | 1 |
