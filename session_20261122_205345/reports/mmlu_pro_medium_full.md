# gpt-oss eval: mmlu_pro_medium_full

- baseline: `/data/shuwang21/home/sglang-looped/results/gptoss_sessions/session_20261122_205345/runs/mmlu_pro_medium_full/baseline`
- loop: `/data/shuwang21/home/sglang-looped/results/gptoss_sessions/session_20261122_205345/runs/mmlu_pro_medium_full/loop`
- dataset: mmlu_pro {'path': 'TIGER-Lab/MMLU-Pro', 'config': None, 'split': 'test'}, limit=None, per_category=None, reasoning_effort=medium, temperature=1.0, top_p=1.0, max_tokens=16384
- seeds: [0]; paired questions: 6570
- **incomplete run: baseline 8808, loop 6570 of 12032 (question, seed) pairs.** Pairs run in hashed order, so the paired questions are a random subset.

## Accuracy per seed

| seed | baseline | loop | delta |
| ---: | ---: | ---: | ---: |
| 0 | 0.7356 | 0.7499 | +0.0143 |
| mean ± std | 0.7356 ± 0.0000 | 0.7499 ± 0.0000 | +0.0143 |

## Generation statistics

| metric | baseline | loop |
| --- | ---: | ---: |
| mean completion tokens | 1238.9 | 1504.4 |
| mean reasoning tokens | n/a | n/a |
| mean reasoning length (chars) | 3672.0 | 4531.9 |
| truncation rate (finish_reason=length) | 0.0010 | 0.0058 |
| answer extraction failure rate | 0.0184 | 0.0218 |
| request errors (excluded) | 0 | 0 |
| scored rows | 8808 | 6570 |

## Paired tests

| test | statistic | p-value |
| --- | --- | ---: |
| sign test, per-question mean over 1 seeds (loop − baseline) | +460 / −398 / =5712 | 0.03723 |
| Wilcoxon signed-rank (scipy), same differences | W=170941.0, n=858 | 0.03429 |
| McNemar exact, seed 0 (6570 questions) | baseline-only 398 / loop-only 460 | 0.03723 |

## Accuracy per category (paired questions, mean over seeds)

| category | questions | baseline | loop | delta | sign test p |
| --- | ---: | ---: | ---: | ---: | ---: |
| biology | 383 | 0.8538 | 0.8590 | +0.0052 | 0.871 |
| business | 419 | 0.8138 | 0.8449 | +0.0310 | 0.0596 |
| chemistry | 606 | 0.8102 | 0.8350 | +0.0248 | 0.0674 |
| computer science | 231 | 0.8182 | 0.8355 | +0.0173 | 0.454 |
| economics | 482 | 0.7967 | 0.8133 | +0.0166 | 0.332 |
| engineering | 535 | 0.5963 | 0.6187 | +0.0224 | 0.271 |
| health | 441 | 0.7370 | 0.7302 | -0.0068 | 0.775 |
| history | 210 | 0.6190 | 0.6857 | +0.0667 | 0.0336 |
| law | 615 | 0.4455 | 0.4504 | +0.0049 | 0.866 |
| math | 733 | 0.9222 | 0.9072 | -0.0150 | 0.185 |
| other | 519 | 0.6667 | 0.6647 | -0.0019 | 1 |
| philosophy | 257 | 0.6109 | 0.6381 | +0.0272 | 0.311 |
| physics | 713 | 0.8275 | 0.8275 | +0.0000 | 1 |
| psychology | 426 | 0.7418 | 0.7394 | -0.0023 | 1 |
