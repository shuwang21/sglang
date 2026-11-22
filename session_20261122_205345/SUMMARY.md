# gpt-oss eval: mmlu_pro_medium_full

- baseline: `/data/shuwang21/home/sglang-looped/results/gptoss_sessions/session_20261122_205345/runs/mmlu_pro_medium_full/baseline`
- loop: `/data/shuwang21/home/sglang-looped/results/gptoss_sessions/session_20261122_205345/runs/mmlu_pro_medium_full/loop`
- dataset: mmlu_pro {'path': 'TIGER-Lab/MMLU-Pro', 'config': None, 'split': 'test'}, limit=None, per_category=None, reasoning_effort=medium, temperature=1.0, top_p=1.0, max_tokens=16384
- seeds: [0]; paired questions: 1702
- **incomplete run: baseline 2313, loop 1702 of 12032 (question, seed) pairs.** Pairs run in hashed order, so the paired questions are a random subset.

## Accuracy per seed

| seed | baseline | loop | delta |
| ---: | ---: | ---: | ---: |
| 0 | 0.7609 | 0.7691 | +0.0082 |
| mean ± std | 0.7609 ± 0.0000 | 0.7691 ± 0.0000 | +0.0082 |

## Generation statistics

| metric | baseline | loop |
| --- | ---: | ---: |
| mean completion tokens | 1175.2 | 1449.3 |
| mean reasoning tokens | n/a | n/a |
| mean reasoning length (chars) | 3490.6 | 4390.1 |
| truncation rate (finish_reason=length) | 0.0004 | 0.0053 |
| answer extraction failure rate | 0.0160 | 0.0217 |
| request errors (excluded) | 0 | 0 |
| scored rows | 2313 | 1702 |

## Paired tests

| test | statistic | p-value |
| --- | --- | ---: |
| sign test, per-question mean over 1 seeds (loop − baseline) | +112 / −100 / =1490 | 0.45 |
| Wilcoxon signed-rank (scipy), same differences | W=10650.0, n=212 | 0.4098 |
| McNemar exact, seed 0 (1702 questions) | baseline-only 100 / loop-only 112 | 0.45 |

## Accuracy per category (paired questions, mean over seeds)

| category | questions | baseline | loop | delta | sign test p |
| --- | ---: | ---: | ---: | ---: | ---: |
| biology | 105 | 0.8952 | 0.8857 | -0.0095 | 1 |
| business | 99 | 0.8687 | 0.8990 | +0.0303 | 0.508 |
| chemistry | 148 | 0.8851 | 0.8986 | +0.0135 | 0.791 |
| computer science | 59 | 0.8475 | 0.8814 | +0.0339 | 0.688 |
| economics | 118 | 0.8644 | 0.8814 | +0.0169 | 0.727 |
| engineering | 149 | 0.5839 | 0.5973 | +0.0134 | 0.864 |
| health | 117 | 0.7692 | 0.7863 | +0.0171 | 0.774 |
| history | 40 | 0.6500 | 0.7250 | +0.0750 | 0.375 |
| law | 154 | 0.4286 | 0.4545 | +0.0260 | 0.618 |
| math | 186 | 0.9194 | 0.9086 | -0.0108 | 0.754 |
| other | 144 | 0.6944 | 0.6667 | -0.0278 | 0.572 |
| philosophy | 82 | 0.6341 | 0.6829 | +0.0488 | 0.289 |
| physics | 188 | 0.8245 | 0.7926 | -0.0319 | 0.263 |
| psychology | 113 | 0.7699 | 0.7788 | +0.0088 | 1 |
