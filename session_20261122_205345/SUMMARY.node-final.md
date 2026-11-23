<!-- Copied by hand from the node's SUMMARY.md on 2026-11-22 (ssh/rsync to the
node failed with "Permission denied (publickey)"). The samples.jsonl files next
to it are from the last successful fetch at 15:32 (baseline 11882, loop 8761
rows) and do NOT match these numbers. -->

# gpt-oss eval: mmlu_pro_medium_full

- baseline: `/data/shuwang21/home/sglang-looped/results/gptoss_sessions/session_20261122_205345/runs/mmlu_pro_medium_full/baseline`
- loop: `/data/shuwang21/home/sglang-looped/results/gptoss_sessions/session_20261122_205345/runs/mmlu_pro_medium_full/loop`
- dataset: mmlu_pro {'path': 'TIGER-Lab/MMLU-Pro', 'config': None, 'split': 'test'}, limit=None, per_category=None, reasoning_effort=medium, temperature=1.0, top_p=1.0, max_tokens=16384
- seeds: [0]; paired questions: 11336
- **incomplete run: baseline 12032, loop 11336 of 12032 (question, seed) pairs.** Pairs run in hashed order, so the paired questions are a random subset.

## Accuracy per seed

| seed | baseline | loop | delta |
| ---: | ---: | ---: | ---: |
| 0 | 0.7349 | 0.7430 | +0.0082 |
| mean ± std | 0.7349 ± 0.0000 | 0.7430 ± 0.0000 | +0.0082 |

## Generation statistics

| metric | baseline | loop |
| --- | ---: | ---: |
| mean completion tokens | 1239.9 | 1502.6 |
| mean reasoning tokens | n/a | n/a |
| mean reasoning length (chars) | 3678.0 | 4525.5 |
| truncation rate (finish_reason=length) | 0.0011 | 0.0063 |
| answer extraction failure rate | 0.0181 | 0.0227 |
| request errors (excluded) | 0 | 0 |
| scored rows | 12032 | 11336 |

## Paired tests

| test | statistic | p-value |
| --- | --- | ---: |
| sign test, per-question mean over 1 seeds (loop − baseline) | +801 / −715 / =9820 | 0.029 |
| Wilcoxon signed-rank (scipy), same differences | W=542327.5, n=1516 | 0.02719 |
| McNemar exact, seed 0 (11336 questions) | baseline-only 715 / loop-only 801 | 0.029 |

## Accuracy per category (paired questions, mean over seeds)

| category | questions | baseline | loop | delta | sign test p |
| --- | ---: | ---: | ---: | ---: | ---: |
| biology | 674 | 0.8516 | 0.8561 | +0.0045 | 0.801 |
| business | 740 | 0.8000 | 0.8230 | +0.0230 | 0.0712 |
| chemistry | 1062 | 0.8202 | 0.8409 | +0.0207 | 0.0467 |
| computer science | 386 | 0.8264 | 0.8523 | +0.0259 | 0.0987 |
| economics | 802 | 0.7868 | 0.7980 | +0.0112 | 0.397 |
| engineering | 904 | 0.6173 | 0.6283 | +0.0111 | 0.502 |
| health | 761 | 0.7188 | 0.7148 | -0.0039 | 0.839 |
| history | 357 | 0.5826 | 0.6162 | +0.0336 | 0.162 |
| law | 1037 | 0.4339 | 0.4436 | +0.0096 | 0.568 |
| math | 1278 | 0.9053 | 0.8944 | -0.0110 | 0.198 |
| other | 878 | 0.6595 | 0.6549 | -0.0046 | 0.807 |
| philosophy | 473 | 0.6195 | 0.6342 | +0.0148 | 0.47 |
| physics | 1227 | 0.8272 | 0.8264 | -0.0008 | 1 |
| psychology | 757 | 0.7173 | 0.7279 | +0.0106 | 0.451 |
