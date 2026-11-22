# gpt-oss eval: mmlu_pro_medium_full

- baseline: `/data/shuwang21/home/sglang-looped/results/gptoss_sessions/session_20261122_205345/runs/mmlu_pro_medium_full/baseline`
- loop: `/data/shuwang21/home/sglang-looped/results/gptoss_sessions/session_20261122_205345/runs/mmlu_pro_medium_full/loop`
- dataset: mmlu_pro {'path': 'TIGER-Lab/MMLU-Pro', 'config': None, 'split': 'test'}, limit=None, per_category=None, reasoning_effort=medium, temperature=1.0, top_p=1.0, max_tokens=16384
- seeds: [0]; paired questions: 7736
- **incomplete run: baseline 10461, loop 7736 of 12032 (question, seed) pairs.** Pairs run in hashed order, so the paired questions are a random subset.

## Accuracy per seed

| seed | baseline | loop | delta |
| ---: | ---: | ---: | ---: |
| 0 | 0.7349 | 0.7464 | +0.0115 |
| mean ± std | 0.7349 ± 0.0000 | 0.7464 ± 0.0000 | +0.0115 |

## Generation statistics

| metric | baseline | loop |
| --- | ---: | ---: |
| mean completion tokens | 1229.1 | 1506.1 |
| mean reasoning tokens | n/a | n/a |
| mean reasoning length (chars) | 3650.6 | 4542.5 |
| truncation rate (finish_reason=length) | 0.0011 | 0.0063 |
| answer extraction failure rate | 0.0180 | 0.0225 |
| request errors (excluded) | 0 | 0 |
| scored rows | 10461 | 7736 |

## Paired tests

| test | statistic | p-value |
| --- | --- | ---: |
| sign test, per-question mean over 1 seeds (loop − baseline) | +545 / −485 / =6706 | 0.06596 |
| Wilcoxon signed-rank (scipy), same differences | W=250017.5, n=1030 | 0.06155 |
| McNemar exact, seed 0 (7736 questions) | baseline-only 485 / loop-only 545 | 0.06596 |

## Accuracy per category (paired questions, mean over seeds)

| category | questions | baseline | loop | delta | sign test p |
| --- | ---: | ---: | ---: | ---: | ---: |
| biology | 453 | 0.8565 | 0.8609 | +0.0044 | 0.88 |
| business | 487 | 0.8131 | 0.8357 | +0.0226 | 0.161 |
| chemistry | 710 | 0.8197 | 0.8380 | +0.0183 | 0.148 |
| computer science | 266 | 0.8158 | 0.8346 | +0.0188 | 0.332 |
| economics | 564 | 0.7943 | 0.8121 | +0.0177 | 0.26 |
| engineering | 632 | 0.6028 | 0.6218 | +0.0190 | 0.315 |
| health | 522 | 0.7203 | 0.7261 | +0.0057 | 0.798 |
| history | 242 | 0.5950 | 0.6446 | +0.0496 | 0.0884 |
| law | 727 | 0.4402 | 0.4443 | +0.0041 | 0.88 |
| math | 873 | 0.9210 | 0.9061 | -0.0149 | 0.154 |
| other | 615 | 0.6650 | 0.6650 | +0.0000 | 1 |
| philosophy | 309 | 0.6181 | 0.6375 | +0.0194 | 0.43 |
| physics | 838 | 0.8258 | 0.8186 | -0.0072 | 0.627 |
| psychology | 498 | 0.7349 | 0.7390 | +0.0040 | 0.896 |
