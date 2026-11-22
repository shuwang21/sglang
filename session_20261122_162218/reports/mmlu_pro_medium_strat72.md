# gpt-oss eval: mmlu_pro_medium_strat72

- baseline: `/data/shuwang21/home/sglang-looped/results/gptoss_sessions/session_20261122_162218/runs/mmlu_pro_medium_strat72/baseline`
- loop: `/data/shuwang21/home/sglang-looped/results/gptoss_sessions/session_20261122_162218/runs/mmlu_pro_medium_strat72/loop`
- dataset: mmlu_pro {'path': 'TIGER-Lab/MMLU-Pro', 'config': None, 'split': 'test'}, limit=None, per_category=72, reasoning_effort=medium, temperature=1.0, top_p=1.0, max_tokens=16384
- seeds: [0]; paired questions: 1008

## Accuracy per seed

| seed | baseline | loop | delta |
| ---: | ---: | ---: | ---: |
| 0 | 0.7560 | 0.7688 | +0.0129 |
| mean ± std | 0.7560 ± 0.0000 | 0.7688 ± 0.0000 | +0.0129 |

## Generation statistics

| metric | baseline | loop |
| --- | ---: | ---: |
| mean completion tokens | 1158.0 | 1338.3 |
| mean reasoning tokens | n/a | n/a |
| mean reasoning length (chars) | 3527.0 | 4155.4 |
| truncation rate (finish_reason=length) | 0.0010 | 0.0020 |
| answer extraction failure rate | 0.0129 | 0.0139 |
| request errors (excluded) | 0 | 0 |
| scored rows | 1008 | 1008 |

## Paired tests

| test | statistic | p-value |
| --- | --- | ---: |
| sign test, per-question mean over 1 seeds (loop − baseline) | +71 / −58 / =879 | 0.2907 |
| Wilcoxon signed-rank (scipy), same differences | W=3770.0, n=129 | 0.2524 |
| McNemar exact, seed 0 (1008 questions) | baseline-only 58 / loop-only 71 | 0.2907 |

## Accuracy per category (paired questions, mean over seeds)

| category | questions | baseline | loop | delta | sign test p |
| --- | ---: | ---: | ---: | ---: | ---: |
| biology | 72 | 0.9028 | 0.9167 | +0.0139 | 1 |
| business | 72 | 0.8194 | 0.8333 | +0.0139 | 1 |
| chemistry | 72 | 0.8056 | 0.8194 | +0.0139 | 1 |
| computer science | 72 | 0.8194 | 0.7917 | -0.0278 | 0.727 |
| economics | 72 | 0.8750 | 0.9167 | +0.0417 | 0.549 |
| engineering | 72 | 0.5417 | 0.5694 | +0.0278 | 0.804 |
| health | 72 | 0.7917 | 0.8611 | +0.0694 | 0.18 |
| history | 72 | 0.6667 | 0.7083 | +0.0417 | 0.453 |
| law | 72 | 0.4583 | 0.4722 | +0.0139 | 1 |
| math | 72 | 0.9028 | 0.8750 | -0.0278 | 0.625 |
| other | 72 | 0.7083 | 0.7222 | +0.0139 | 1 |
| philosophy | 72 | 0.6806 | 0.6667 | -0.0139 | 1 |
| physics | 72 | 0.8194 | 0.8333 | +0.0139 | 1 |
| psychology | 72 | 0.7917 | 0.7778 | -0.0139 | 1 |
