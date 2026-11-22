# gpt-oss eval: mmlu_pro_medium_full

- baseline: `/data/shuwang21/home/sglang-looped/results/gptoss_sessions/session_20261122_205345/runs/mmlu_pro_medium_full/baseline`
- loop: `/data/shuwang21/home/sglang-looped/results/gptoss_sessions/session_20261122_205345/runs/mmlu_pro_medium_full/loop`
- dataset: mmlu_pro {'path': 'TIGER-Lab/MMLU-Pro', 'config': None, 'split': 'test'}, limit=None, per_category=None, reasoning_effort=medium, temperature=1.0, top_p=1.0, max_tokens=16384
- seeds: [0]; paired questions: 7142
- **incomplete run: baseline 9607, loop 7142 of 12032 (question, seed) pairs.** Pairs run in hashed order, so the paired questions are a random subset.

## Accuracy per seed

| seed | baseline | loop | delta |
| ---: | ---: | ---: | ---: |
| 0 | 0.7361 | 0.7480 | +0.0118 |
| mean ± std | 0.7361 ± 0.0000 | 0.7480 ± 0.0000 | +0.0118 |

## Generation statistics

| metric | baseline | loop |
| --- | ---: | ---: |
| mean completion tokens | 1231.1 | 1506.4 |
| mean reasoning tokens | n/a | n/a |
| mean reasoning length (chars) | 3653.3 | 4542.8 |
| truncation rate (finish_reason=length) | 0.0009 | 0.0062 |
| answer extraction failure rate | 0.0177 | 0.0224 |
| request errors (excluded) | 0 | 0 |
| scored rows | 9607 | 7142 |

## Paired tests

| test | statistic | p-value |
| --- | --- | ---: |
| sign test, per-question mean over 1 seeds (loop − baseline) | +497 / −437 / =6208 | 0.05348 |
| Wilcoxon signed-rank (scipy), same differences | W=204297.5, n=934 | 0.04962 |
| McNemar exact, seed 0 (7142 questions) | baseline-only 437 / loop-only 497 | 0.05348 |

## Accuracy per category (paired questions, mean over seeds)

| category | questions | baseline | loop | delta | sign test p |
| --- | ---: | ---: | ---: | ---: | ---: |
| biology | 414 | 0.8575 | 0.8647 | +0.0072 | 0.755 |
| business | 457 | 0.8118 | 0.8359 | +0.0241 | 0.144 |
| chemistry | 656 | 0.8110 | 0.8338 | +0.0229 | 0.0817 |
| computer science | 245 | 0.8163 | 0.8327 | +0.0163 | 0.454 |
| economics | 521 | 0.7965 | 0.8119 | +0.0154 | 0.358 |
| engineering | 585 | 0.6068 | 0.6239 | +0.0171 | 0.391 |
| health | 487 | 0.7331 | 0.7331 | +0.0000 | 1 |
| history | 225 | 0.5956 | 0.6578 | +0.0622 | 0.0385 |
| law | 665 | 0.4436 | 0.4496 | +0.0060 | 0.81 |
| math | 812 | 0.9249 | 0.9089 | -0.0160 | 0.124 |
| other | 559 | 0.6637 | 0.6601 | -0.0036 | 0.92 |
| philosophy | 282 | 0.6241 | 0.6418 | +0.0177 | 0.511 |
| physics | 770 | 0.8195 | 0.8195 | +0.0000 | 1 |
| psychology | 464 | 0.7306 | 0.7328 | +0.0022 | 1 |
