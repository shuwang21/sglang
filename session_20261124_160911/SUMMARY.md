# gpt-oss eval: aime24_low_s5

- baseline: `/data/shuwang21/home/sglang-looped/results/gptoss_sessions/session_20261124_160911/runs/aime24_low_s5/baseline`
- loop: `/data/shuwang21/home/sglang-looped/results/gptoss_sessions/session_20261124_160911/runs/aime24_low_s5/loop`
- dataset: aime24 {'path': 'HuggingFaceH4/aime_2024', 'config': None, 'split': 'train'}, limit=None, per_category=None, reasoning_effort=low, temperature=1.0, top_p=1.0, max_tokens=32768
- seeds: [0, 1, 2, 3, 4]; paired questions: 30

## Accuracy per seed

| seed | baseline | loop | delta |
| ---: | ---: | ---: | ---: |
| 0 | 0.4333 | 0.4667 | +0.0333 |
| 1 | 0.4000 | 0.4333 | +0.0333 |
| 2 | 0.3333 | 0.4333 | +0.1000 |
| 3 | 0.3667 | 0.3667 | +0.0000 |
| 4 | 0.4000 | 0.5667 | +0.1667 |
| mean ± std | 0.3867 ± 0.0380 | 0.4533 ± 0.0730 | +0.0667 |

## Generation statistics

| metric | baseline | loop |
| --- | ---: | ---: |
| mean completion tokens | 2040.6 | 2495.1 |
| mean reasoning tokens | n/a | n/a |
| mean reasoning length (chars) | 4047.5 | 5011.7 |
| truncation rate (finish_reason=length) | 0.0000 | 0.0000 |
| answer extraction failure rate | 0.1600 | 0.1400 |
| request errors (excluded) | 0 | 0 |
| scored rows | 150 | 150 |

## Paired tests

| test | statistic | p-value |
| --- | --- | ---: |
| sign test, per-question mean over 5 seeds (loop − baseline) | +11 / −6 / =13 | 0.3323 |
| Wilcoxon signed-rank (scipy), same differences | W=45.5, n=17 | 0.1387 |
| McNemar exact, seed 0 (30 questions) | baseline-only 4 / loop-only 5 | 1 |
