# gpt-oss eval: mmlu_pro_medium_full

- baseline: `/data/shuwang21/home/sglang-looped/results/gptoss_sessions/session_20261122_205345/runs/mmlu_pro_medium_full/baseline`
- loop: `/data/shuwang21/home/sglang-looped/results/gptoss_sessions/session_20261122_205345/runs/mmlu_pro_medium_full/loop`
- dataset: mmlu_pro {'path': 'TIGER-Lab/MMLU-Pro', 'config': None, 'split': 'test'}, limit=None, per_category=None, reasoning_effort=medium, temperature=1.0, top_p=1.0, max_tokens=16384
- seeds: [0]; paired questions: 2823
- **incomplete run: baseline 3751, loop 2823 of 12032 (question, seed) pairs.** Pairs run in hashed order, so the paired questions are a random subset.

## Accuracy per seed

| seed | baseline | loop | delta |
| ---: | ---: | ---: | ---: |
| 0 | 0.7505 | 0.7609 | +0.0104 |
| mean ± std | 0.7505 ± 0.0000 | 0.7609 ± 0.0000 | +0.0104 |

## Generation statistics

| metric | baseline | loop |
| --- | ---: | ---: |
| mean completion tokens | 1215.8 | 1462.0 |
| mean reasoning tokens | n/a | n/a |
| mean reasoning length (chars) | 3598.4 | 4425.8 |
| truncation rate (finish_reason=length) | 0.0003 | 0.0053 |
| answer extraction failure rate | 0.0152 | 0.0213 |
| request errors (excluded) | 0 | 0 |
| scored rows | 3751 | 2823 |

## Paired tests

| test | statistic | p-value |
| --- | --- | ---: |
| sign test, per-question mean over 1 seeds (loop − baseline) | +194 / −173 / =2456 | 0.2965 |
| Wilcoxon signed-rank (scipy), same differences | W=31832.0, n=367 | 0.273 |
| McNemar exact, seed 0 (2823 questions) | baseline-only 173 / loop-only 194 | 0.2965 |

## Accuracy per category (paired questions, mean over seeds)

| category | questions | baseline | loop | delta | sign test p |
| --- | ---: | ---: | ---: | ---: | ---: |
| biology | 171 | 0.8830 | 0.8655 | -0.0175 | 0.629 |
| business | 169 | 0.8107 | 0.8580 | +0.0473 | 0.0768 |
| chemistry | 259 | 0.8764 | 0.8919 | +0.0154 | 0.481 |
| computer science | 93 | 0.8602 | 0.8817 | +0.0215 | 0.727 |
| economics | 200 | 0.8200 | 0.8500 | +0.0300 | 0.21 |
| engineering | 236 | 0.6229 | 0.6229 | +0.0000 | 1 |
| health | 185 | 0.7784 | 0.7568 | -0.0216 | 0.481 |
| history | 82 | 0.6220 | 0.7195 | +0.0976 | 0.0386 |
| law | 264 | 0.4318 | 0.4583 | +0.0265 | 0.443 |
| math | 314 | 0.9076 | 0.8885 | -0.0191 | 0.327 |
| other | 243 | 0.6831 | 0.6749 | -0.0082 | 0.888 |
| philosophy | 114 | 0.6316 | 0.6754 | +0.0439 | 0.267 |
| physics | 299 | 0.8127 | 0.7893 | -0.0234 | 0.349 |
| psychology | 194 | 0.7526 | 0.7680 | +0.0155 | 0.664 |
