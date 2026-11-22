# gpt-oss eval: mmlu_pro_medium_l500

- baseline: `/data/shuwang21/home/sglang-looped/results/gptoss_sessions/session_20261122_062555/runs/mmlu_pro_medium_l500/baseline`
- loop: `/data/shuwang21/home/sglang-looped/results/gptoss_sessions/session_20261122_062555/runs/mmlu_pro_medium_l500/loop`
- dataset: mmlu_pro {'path': 'TIGER-Lab/MMLU-Pro', 'config': None, 'split': 'test'}, limit=500, reasoning_effort=medium, temperature=1.0, top_p=1.0, max_tokens=16384
- seeds: [0, 1]; paired questions: 500

## Accuracy per seed

| seed | baseline | loop | delta |
| ---: | ---: | ---: | ---: |
| 0 | 0.8260 | 0.8480 | +0.0220 |
| 1 | 0.8200 | 0.8160 | -0.0040 |
| mean ± std | 0.8230 ± 0.0042 | 0.8320 ± 0.0226 | +0.0090 |

## Generation statistics

| metric | baseline | loop |
| --- | ---: | ---: |
| mean completion tokens | 1323.5 | 1629.9 |
| mean reasoning tokens | n/a | n/a |
| mean reasoning length (chars) | 3624.5 | 4572.8 |
| truncation rate (finish_reason=length) | 0.0000 | 0.0040 |
| answer extraction failure rate | 0.0120 | 0.0200 |
| request errors (excluded) | 0 | 0 |
| scored rows | 1000 | 1000 |

## Paired tests

| test | statistic | p-value |
| --- | --- | ---: |
| sign test, per-question mean over 2 seeds (loop − baseline) | +48 / −40 / =412 | 0.4557 |
| Wilcoxon signed-rank (scipy), same differences | W=1772.0, n=88 | 0.3893 |
| McNemar exact, seed 0 (500 questions) | baseline-only 22 / loop-only 33 | 0.177 |
