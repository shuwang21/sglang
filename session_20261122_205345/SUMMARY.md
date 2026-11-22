# gpt-oss eval: mmlu_pro_medium_full

- baseline: `/data/shuwang21/home/sglang-looped/results/gptoss_sessions/session_20261122_205345/runs/mmlu_pro_medium_full/baseline`
- loop: `/data/shuwang21/home/sglang-looped/results/gptoss_sessions/session_20261122_205345/runs/mmlu_pro_medium_full/loop`
- dataset: mmlu_pro {'path': 'TIGER-Lab/MMLU-Pro', 'config': None, 'split': 'test'}, limit=None, per_category=None, reasoning_effort=medium, temperature=1.0, top_p=1.0, max_tokens=16384
- seeds: [0]; paired questions: 8381
- **incomplete run: baseline 11326, loop 8381 of 12032 (question, seed) pairs.** Pairs run in hashed order, so the paired questions are a random subset.

## Accuracy per seed

| seed | baseline | loop | delta |
| ---: | ---: | ---: | ---: |
| 0 | 0.7354 | 0.7425 | +0.0071 |
| mean ± std | 0.7354 ± 0.0000 | 0.7425 ± 0.0000 | +0.0071 |

## Generation statistics

| metric | baseline | loop |
| --- | ---: | ---: |
| mean completion tokens | 1228.6 | 1510.1 |
| mean reasoning tokens | n/a | n/a |
| mean reasoning length (chars) | 3641.0 | 4564.4 |
| truncation rate (finish_reason=length) | 0.0010 | 0.0063 |
| answer extraction failure rate | 0.0181 | 0.0231 |
| request errors (excluded) | 0 | 0 |
| scored rows | 11326 | 8381 |

## Paired tests

| test | statistic | p-value |
| --- | --- | ---: |
| sign test, per-question mean over 1 seeds (loop − baseline) | +595 / −536 / =7250 | 0.08455 |
| Wilcoxon signed-rank (scipy), same differences | W=303376.0, n=1131 | 0.07937 |
| McNemar exact, seed 0 (8381 questions) | baseline-only 536 / loop-only 595 | 0.08455 |

## Accuracy per category (paired questions, mean over seeds)

| category | questions | baseline | loop | delta | sign test p |
| --- | ---: | ---: | ---: | ---: | ---: |
| biology | 489 | 0.8548 | 0.8569 | +0.0020 | 1 |
| business | 528 | 0.8049 | 0.8277 | +0.0227 | 0.141 |
| chemistry | 773 | 0.8150 | 0.8396 | +0.0246 | 0.0422 |
| computer science | 289 | 0.8166 | 0.8408 | +0.0242 | 0.167 |
| economics | 610 | 0.7918 | 0.8082 | +0.0164 | 0.289 |
| engineering | 684 | 0.6009 | 0.6257 | +0.0249 | 0.165 |
| health | 563 | 0.7158 | 0.7140 | -0.0018 | 1 |
| history | 258 | 0.5969 | 0.6473 | +0.0504 | 0.066 |
| law | 793 | 0.4388 | 0.4376 | -0.0013 | 1 |
| math | 939 | 0.9201 | 0.9031 | -0.0170 | 0.0846 |
| other | 663 | 0.6712 | 0.6682 | -0.0030 | 0.927 |
| philosophy | 341 | 0.6276 | 0.6305 | +0.0029 | 1 |
| physics | 901 | 0.8158 | 0.8091 | -0.0067 | 0.645 |
| psychology | 550 | 0.7236 | 0.7327 | +0.0091 | 0.62 |
