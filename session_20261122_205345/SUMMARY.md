# gpt-oss eval: mmlu_pro_medium_full

- baseline: `/data/shuwang21/home/sglang-looped/results/gptoss_sessions/session_20261122_205345/runs/mmlu_pro_medium_full/baseline`
- loop: `/data/shuwang21/home/sglang-looped/results/gptoss_sessions/session_20261122_205345/runs/mmlu_pro_medium_full/loop`
- dataset: mmlu_pro {'path': 'TIGER-Lab/MMLU-Pro', 'config': None, 'split': 'test'}, limit=None, per_category=None, reasoning_effort=medium, temperature=1.0, top_p=1.0, max_tokens=16384
- seeds: [0]; paired questions: 4440
- **incomplete run: baseline 5923, loop 4440 of 12032 (question, seed) pairs.** Pairs run in hashed order, so the paired questions are a random subset.

## Accuracy per seed

| seed | baseline | loop | delta |
| ---: | ---: | ---: | ---: |
| 0 | 0.7412 | 0.7523 | +0.0111 |
| mean ± std | 0.7412 ± 0.0000 | 0.7523 ± 0.0000 | +0.0111 |

## Generation statistics

| metric | baseline | loop |
| --- | ---: | ---: |
| mean completion tokens | 1225.4 | 1485.8 |
| mean reasoning tokens | n/a | n/a |
| mean reasoning length (chars) | 3629.8 | 4481.9 |
| truncation rate (finish_reason=length) | 0.0005 | 0.0054 |
| answer extraction failure rate | 0.0171 | 0.0212 |
| request errors (excluded) | 0 | 0 |
| scored rows | 5923 | 4440 |

## Paired tests

| test | statistic | p-value |
| --- | --- | ---: |
| sign test, per-question mean over 1 seeds (loop − baseline) | +311 / −259 / =3870 | 0.03258 |
| Wilcoxon signed-rank (scipy), same differences | W=73944.5, n=570 | 0.0294 |
| McNemar exact, seed 0 (4440 questions) | baseline-only 259 / loop-only 311 | 0.03258 |

## Accuracy per category (paired questions, mean over seeds)

| category | questions | baseline | loop | delta | sign test p |
| --- | ---: | ---: | ---: | ---: | ---: |
| biology | 262 | 0.8550 | 0.8550 | +0.0000 | 1 |
| business | 274 | 0.7993 | 0.8467 | +0.0474 | 0.0192 |
| chemistry | 404 | 0.8317 | 0.8639 | +0.0322 | 0.047 |
| computer science | 155 | 0.8065 | 0.8129 | +0.0065 | 1 |
| economics | 336 | 0.7857 | 0.8214 | +0.0357 | 0.0576 |
| engineering | 364 | 0.6346 | 0.6374 | +0.0027 | 1 |
| health | 292 | 0.7534 | 0.7432 | -0.0103 | 0.736 |
| history | 149 | 0.6107 | 0.6644 | +0.0537 | 0.134 |
| law | 400 | 0.4250 | 0.4475 | +0.0225 | 0.374 |
| math | 502 | 0.9223 | 0.8984 | -0.0239 | 0.073 |
| other | 375 | 0.6693 | 0.6667 | -0.0027 | 1 |
| philosophy | 173 | 0.6012 | 0.6416 | +0.0405 | 0.167 |
| physics | 462 | 0.8030 | 0.8074 | +0.0043 | 0.899 |
| psychology | 292 | 0.7500 | 0.7568 | +0.0068 | 0.86 |
