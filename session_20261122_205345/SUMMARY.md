# gpt-oss eval: mmlu_pro_medium_full

- baseline: `/data/shuwang21/home/sglang-looped/results/gptoss_sessions/session_20261122_205345/runs/mmlu_pro_medium_full/baseline`
- loop: `/data/shuwang21/home/sglang-looped/results/gptoss_sessions/session_20261122_205345/runs/mmlu_pro_medium_full/loop`
- dataset: mmlu_pro {'path': 'TIGER-Lab/MMLU-Pro', 'config': None, 'split': 'test'}, limit=None, per_category=None, reasoning_effort=medium, temperature=1.0, top_p=1.0, max_tokens=16384
- seeds: [0]; paired questions: 2264
- **incomplete run: baseline 3054, loop 2264 of 12032 (question, seed) pairs.** Pairs run in hashed order, so the paired questions are a random subset.

## Accuracy per seed

| seed | baseline | loop | delta |
| ---: | ---: | ---: | ---: |
| 0 | 0.7521 | 0.7655 | +0.0133 |
| mean ± std | 0.7521 ± 0.0000 | 0.7655 ± 0.0000 | +0.0133 |

## Generation statistics

| metric | baseline | loop |
| --- | ---: | ---: |
| mean completion tokens | 1186.5 | 1433.6 |
| mean reasoning tokens | n/a | n/a |
| mean reasoning length (chars) | 3526.9 | 4350.4 |
| truncation rate (finish_reason=length) | 0.0003 | 0.0049 |
| answer extraction failure rate | 0.0167 | 0.0225 |
| request errors (excluded) | 0 | 0 |
| scored rows | 3054 | 2264 |

## Paired tests

| test | statistic | p-value |
| --- | --- | ---: |
| sign test, per-question mean over 1 seeds (loop − baseline) | +157 / −143 / =1964 | 0.453 |
| Wilcoxon signed-rank (scipy), same differences | W=21521.5, n=300 | 0.4189 |
| McNemar exact, seed 0 (2264 questions) | baseline-only 143 / loop-only 157 | 0.453 |

## Accuracy per category (paired questions, mean over seeds)

| category | questions | baseline | loop | delta | sign test p |
| --- | ---: | ---: | ---: | ---: | ---: |
| biology | 143 | 0.9091 | 0.8811 | -0.0280 | 0.454 |
| business | 135 | 0.8296 | 0.8815 | +0.0519 | 0.0923 |
| chemistry | 201 | 0.8706 | 0.8955 | +0.0249 | 0.332 |
| computer science | 76 | 0.8421 | 0.8684 | +0.0263 | 0.727 |
| economics | 160 | 0.8125 | 0.8438 | +0.0312 | 0.267 |
| engineering | 191 | 0.6230 | 0.6230 | +0.0000 | 1 |
| health | 148 | 0.7905 | 0.7770 | -0.0135 | 0.804 |
| history | 59 | 0.6271 | 0.7119 | +0.0847 | 0.18 |
| law | 203 | 0.4236 | 0.4631 | +0.0394 | 0.332 |
| math | 251 | 0.9084 | 0.8924 | -0.0159 | 0.503 |
| other | 196 | 0.6939 | 0.6735 | -0.0204 | 0.644 |
| philosophy | 105 | 0.6286 | 0.6762 | +0.0476 | 0.227 |
| physics | 242 | 0.8306 | 0.7851 | -0.0455 | 0.0614 |
| psychology | 154 | 0.7662 | 0.7792 | +0.0130 | 0.804 |
