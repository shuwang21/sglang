# Copyright 2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License").

"""Generative eval for gpt-oss against a running SGLang server: multiple
choice (MMLU-Pro, GPQA) and integer-answer math (AIME 2024 / 2025).

Two subcommands:

``run``: query ``/v1/chat/completions`` with ``reasoning_effort`` for every
(question, seed) pair and append one JSON line per pair to
``<output>/samples.jsonl``. Re-running with the same ``--output`` resumes:
pairs already answered are skipped, and pairs that errored are retried.
Pairs run in a fixed hashed order, so a run that is cut off has answered a
uniform random subset (MMLU-Pro itself is sorted by category). Several
``--base-url`` servers (e.g. one per GPU) share one run.

``summarize``: compare a baseline and a loop result directory. It reports
per-seed accuracy (with mean and std), generation statistics, a paired
per-question test (sign test + Wilcoxon on the 5-seed mean accuracy) and an
exact McNemar test on one seed, written as a markdown report.

Only the Python standard library is needed, plus ``datasets`` for ``run``.
``scipy`` is used for Wilcoxon when it is installed.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import os
import queue
import random
import re
import statistics
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

LETTERS = "ABCDEFGHIJ"

# name -> (HF dataset, config, split). Every field can be overridden on the
# command line; e.g. GPQA diamond is --dataset gpqa --hf-config gpqa_diamond.
DATASETS = {
    "mmlu_pro": ("TIGER-Lab/MMLU-Pro", None, "test"),
    "gpqa": ("Idavidrein/gpqa", "gpqa_main", "train"),
    # 30 problems each, integer answers 0..999; not gated.
    "aime24": ("HuggingFaceH4/aime_2024", None, "train"),
    "aime25": ("MathArena/aime_2025", None, "train"),
    # aime24 + aime25 (60 problems), with the year as the category.
    "aime": None,
}
AIME_PARTS = ("aime24", "aime25")

PROMPT = (
    "Answer the following multiple choice question. The last line of your "
    "response should be of the following format: 'Answer: $LETTER' (without "
    "quotes) where LETTER is one of {letters}. Think step by step before "
    "answering.\n\n{question}\n\n{options}"
)

# Integer-answer problems (AIME). Same "last line" convention as PROMPT;
# \\boxed{N} is accepted as well.
MATH_PROMPT = (
    "Solve the following math problem. The answer is an integer from 0 to "
    "999. The last line of your response should be of the following format: "
    "'Answer: $NUMBER' (without quotes). Think step by step before "
    "answering.\n\n{question}"
)

# After "Answer:", skip markdown / LaTeX wrappers the model adds around the
# value: "**Answer:** $B$", "Answer: \\(25\\)", "Answer: \\(\\boxed{25}\\)",
# "Answer: \\(\\displaystyle 107\\)", "\\boxed{\\text{Answer: }5}".
_WRAP = (
    r"(?:[\*\(\[\s\$\\\}]"
    r"|\\?(?:text|textbf|mathrm|mathbf|boxed|displaystyle)\b\{?)*"
)
_ANSWER_RE = re.compile(
    r"(?i)answer\s*(?:is)?\s*[:：]\s*" + _WRAP + r"([A-J])(?![A-Za-z])"
)
_BOXED_RE = re.compile(r"\\boxed\{\s*(?:\\text\{)?\s*([A-J])\s*\}")
_INT_ANSWER_RE = re.compile(
    r"(?i)answer\s*(?:is)?\s*[:：]\s*" + _WRAP + r"(\d[\d,]*)(?![\d.]*\d)"
)
_INT_BOXED_RE = re.compile(r"\\boxed\{\s*(?:\\text\{)?\s*(\d[\d,]*)\s*\}?\s*\}")


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------


def _gpqa_options(row, record_id):
    """GPQA lists the correct answer first; shuffle it deterministically per
    question (independent of the sampling seed, so every seed and both the
    baseline and loop runs see the same option order)."""

    options = [
        row["Correct Answer"],
        row["Incorrect Answer 1"],
        row["Incorrect Answer 2"],
        row["Incorrect Answer 3"],
    ]
    options = [str(item).strip() for item in options]
    order = list(range(4))
    rng = random.Random(question_hash(record_id))
    rng.shuffle(order)
    shuffled = [options[i] for i in order]
    return shuffled, LETTERS[order.index(0)]


def load_questions(args):
    from datasets import load_dataset

    if args.dataset == "aime":
        if args.data_file or args.hf_dataset or args.hf_config or args.split:
            raise SystemExit(
                "--dataset aime combines aime24 and aime25; override those instead"
            )
        questions, sources = [], {}
        for part in AIME_PARTS:
            sub_args = argparse.Namespace(**vars(args))
            sub_args.dataset, sub_args.limit, sub_args.per_category = part, None, None
            part_questions, sources[part] = load_questions(sub_args)
            questions += part_questions
        return _sample(args, questions), sources
    default_path, default_config, default_split = DATASETS[args.dataset]
    if args.data_file:
        # Local copy (e.g. gated GPQA prepared elsewhere): same columns as the
        # HF dataset; format from the extension (.csv / .json / .jsonl / .parquet).
        suffix = Path(args.data_file).suffix.lstrip(".").lower()
        builder = {"jsonl": "json", "csv": "csv", "json": "json", "parquet": "parquet"}
        if suffix not in builder:
            raise SystemExit(f"--data-file: unsupported extension .{suffix}")
        dataset = load_dataset(
            builder[suffix], data_files=args.data_file, split="train"
        )
        source = dict(data_file=str(Path(args.data_file).resolve()))
        return _to_questions(args, dataset), source
    path = args.hf_dataset or default_path
    config = args.hf_config if args.hf_config is not None else default_config
    split = args.split or default_split
    dataset = (
        load_dataset(path, config, split=split)
        if config
        else load_dataset(path, split=split)
    )
    return _to_questions(args, dataset), dict(path=path, config=config, split=split)


def _to_questions(args, dataset):
    questions = []
    for index, row in enumerate(dataset):
        if args.dataset in AIME_PARTS:
            # HuggingFaceH4/aime_2024 (id, problem, answer), MathArena/aime_2025
            # (problem_idx, problem, answer); capitalised columns work too
            # (e.g. Maxwell-Jia/AIME_2024).
            pick = lambda *keys: next(  # noqa: E731
                (row[k] for k in keys if row.get(k) is not None), None
            )
            key = pick("id", "ID", "problem_idx")
            qid = f"{args.dataset}/{key if key is not None else index}"
            question = str(pick("problem", "Problem")).strip()
            options = None
            gold = str(int(str(pick("answer", "Answer")).strip()))
            category = args.dataset
        elif args.dataset == "mmlu_pro":
            options = [str(item) for item in row["options"]]
            qid = f"mmlu_pro/{row.get('question_id', index)}"
            gold = row["answer"].strip()
            question = row["question"].strip()
            category = row.get("category")
        else:
            qid = f"gpqa/{row.get('Record ID') or index}"
            options, gold = _gpqa_options(row, qid)
            question = row["Question"].strip()
            category = row.get("High-level domain")
        questions.append(
            dict(
                id=qid,
                question=question,
                options=options,
                gold=gold,
                category=category,
            )
        )
    return _sample(args, questions)


def _sample(args, questions):
    if getattr(args, "per_category", None) is not None:
        questions = stratified_sample(questions, args.per_category)
    if args.limit is not None:
        questions = questions[: args.limit]
    return questions


def question_hash(qid):
    """Stable pseudo-random key for a question id (same on every machine)."""

    return int(hashlib.sha256(qid.encode()).hexdigest(), 16)


def stratified_sample(questions, per_category):
    """At most ``per_category`` questions from every category, picked by
    ``question_hash`` (a fixed pseudo-random subset, the same for baseline
    and loop). The result keeps the dataset order."""

    by_category = {}
    for question in questions:
        by_category.setdefault(question.get("category"), []).append(question)
    keep = set()
    for group in by_category.values():
        group = sorted(group, key=lambda q: question_hash(q["id"]))
        keep.update(q["id"] for q in group[:per_category])
    return [q for q in questions if q["id"] in keep]


def run_order(questions, seeds, done):
    """Pairs still to run, seed by seed, each seed in ``question_hash``
    order. MMLU-Pro is sorted by category, so running it in dataset order
    would make a cut-off run cover only the first few categories; in hash
    order, whatever has finished is a uniform random subset."""

    ordered = sorted(questions, key=lambda q: question_hash(q["id"]))
    return [
        (question, seed)
        for seed in seeds
        for question in ordered
        if (question["id"], seed) not in done
    ]


def prompt_template(dataset):
    return MATH_PROMPT if dataset.startswith("aime") else PROMPT


def prompt_for(question):
    if question["options"] is None:
        return MATH_PROMPT.format(question=question["question"])
    return build_prompt(question["question"], question["options"])


def build_prompt(question, options):
    letters = LETTERS[: len(options)]
    body = "\n".join(f"{letter}) {text}" for letter, text in zip(letters, options))
    return PROMPT.format(letters=letters, question=question, options=body)


def extract_answer(text, num_options):
    """Last ``Answer: X`` in the text (X within the valid letters), else the
    last ``\\boxed{X}``. Returns None when nothing usable is found."""

    if not text:
        return None
    valid = LETTERS[:num_options]
    for pattern in (_ANSWER_RE, _BOXED_RE):
        matches = [m.upper() for m in pattern.findall(text) if m.upper() in valid]
        if matches:
            return matches[-1]
    return None


def extract_integer(text):
    """Last ``Answer: N`` in the text, else the last ``\\boxed{N}``, as a
    canonical integer string ("033" -> "33", "1,000" -> "1000")."""

    if not text:
        return None
    for pattern in (_INT_ANSWER_RE, _INT_BOXED_RE):
        matches = pattern.findall(text)
        if matches:
            return str(int(matches[-1].replace(",", "")))
    return None


def extractor_for(question):
    """Answer parser for one question: integer for math, letter otherwise."""

    if question["options"] is None:
        return extract_integer
    num_options = len(question["options"])
    return lambda text: extract_answer(text, num_options)


# ---------------------------------------------------------------------------
# Requests
# ---------------------------------------------------------------------------


def _post(url, payload, timeout):
    data = json.dumps(payload).encode()
    request = Request(url, data=data, headers={"Content-Type": "application/json"})
    with urlopen(request, timeout=timeout) as response:
        return json.loads(response.read())


def _get(url, timeout=30):
    with urlopen(url, timeout=timeout) as response:
        return json.loads(response.read())


def resolve_model(base_url, model):
    if model:
        return model
    listing = _get(base_url.rstrip("/") + "/v1/models")
    return listing["data"][0]["id"]


def build_payload(args, model, prompt, seed):
    return {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_tokens": args.max_tokens,
        "seed": seed,
        "reasoning_effort": args.reasoning_effort,
    }


def parse_response(response, extract):
    """``extract(text)`` -> answer or None (see ``extractor_for``)."""

    choice = response["choices"][0]
    message = choice.get("message") or {}
    content = message.get("content") or ""
    reasoning = message.get("reasoning_content") or message.get("reasoning")
    usage = response.get("usage") or {}
    details = usage.get("completion_tokens_details") or {}
    predicted = extract(content)
    # Diagnostic only: an answer that appears in the reasoning but not in the
    # final content is still scored as an extraction failure.
    in_reasoning = extract(reasoning) if predicted is None and reasoning else None
    return dict(
        predicted=predicted,
        answer_in_reasoning_only=in_reasoning,
        finish_reason=choice.get("finish_reason"),
        completion_tokens=usage.get("completion_tokens"),
        reasoning_tokens=details.get("reasoning_tokens"),
        reasoning_chars=len(reasoning) if reasoning is not None else None,
        content=content,
        reasoning=reasoning,
    )


class ServerSlots:
    """``concurrency`` request slots per server URL. A request takes a free
    slot from whichever server has one, so a faster server gets more work."""

    def __init__(self, urls, concurrency):
        self.size = len(urls) * concurrency
        self._free = queue.Queue()
        for _ in range(concurrency):
            for url in urls:
                self._free.put(url)

    @contextlib.contextmanager
    def take(self):
        url = self._free.get()
        try:
            yield url
        finally:
            self._free.put(url)


def run_pair(args, slots, model, question, seed):
    prompt = prompt_for(question)
    payload = build_payload(args, model, prompt, seed)
    error = None
    for attempt in range(args.max_retries + 1):
        try:
            with slots.take() as base_url:
                started = time.monotonic()
                response = _post(
                    base_url.rstrip("/") + "/v1/chat/completions",
                    payload,
                    args.request_timeout,
                )
            parsed = parse_response(response, extractor_for(question))
            break
        except (
            HTTPError,
            URLError,
            TimeoutError,
            OSError,
            KeyError,
            ValueError,
        ) as exc:
            error = f"{type(exc).__name__}: {exc}"
            if isinstance(exc, HTTPError):
                error += " " + exc.read().decode(errors="replace")[:500]
            if attempt < args.max_retries:
                time.sleep(min(60, 2**attempt))
    else:
        return dict(
            id=question["id"],
            seed=seed,
            gold=question["gold"],
            predicted=None,
            correct=False,
            extraction_failed=None,
            error=error,
        )
    return dict(
        id=question["id"],
        seed=seed,
        category=question.get("category"),
        gold=question["gold"],
        predicted=parsed["predicted"],
        correct=parsed["predicted"] == question["gold"],
        extraction_failed=parsed["predicted"] is None,
        answer_in_reasoning_only=parsed["answer_in_reasoning_only"],
        finish_reason=parsed["finish_reason"],
        completion_tokens=parsed["completion_tokens"],
        reasoning_tokens=parsed["reasoning_tokens"],
        reasoning_chars=parsed["reasoning_chars"],
        latency_s=round(time.monotonic() - started, 3),
        error=None,
        output=parsed["content"],
        reasoning=parsed["reasoning"],
    )


def rescore(args):
    """Re-extract answers from the saved outputs after a parser fix; no
    server needed. The first rescore keeps the original file as
    samples.jsonl.orig. Rows store no option count, so MMLU-Pro accepts
    A-J and GPQA A-D."""

    for directory in args.dirs:
        path = Path(directory) / "samples.jsonl"
        dataset = json.loads((Path(directory) / "run_config.json").read_text())[
            "dataset"
        ]
        if dataset.startswith("aime"):
            extract = extract_integer
        else:
            num_options = 4 if dataset == "gpqa" else len(LETTERS)
            extract = lambda text, n=num_options: extract_answer(text, n)  # noqa: E731
        backup = path.with_name("samples.jsonl.orig")
        if not backup.exists():
            backup.write_bytes(path.read_bytes())
        rows, changed = [], 0
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("error") is None:
                predicted = extract(row.get("output"))
                reasoning = row.get("reasoning")
                update = dict(
                    predicted=predicted,
                    correct=predicted == row["gold"],
                    extraction_failed=predicted is None,
                    answer_in_reasoning_only=(
                        extract(reasoning) if predicted is None and reasoning else None
                    ),
                )
                changed += any(row.get(k) != v for k, v in update.items())
                row.update(update)
            rows.append(row)
        tmp = path.with_name("samples.jsonl.tmp")
        tmp.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
        tmp.replace(path)
        print(f"{directory}: {len(rows)} rows, {changed} re-scored")


def load_samples(directory):
    """Latest row per (id, seed) from ``samples.jsonl``."""

    path = Path(directory) / "samples.jsonl"
    rows = {}
    if path.exists():
        with path.open() as handle:
            for line in handle:
                line = line.strip()
                if line:
                    row = json.loads(line)
                    rows[(row["id"], row["seed"])] = row
    return rows


def run(args):
    questions, source = load_questions(args)
    if args.prepare_only:
        # Download/cache the dataset and stop; no server needed.
        print(f"{args.dataset} {source}: {len(questions)} questions ready")
        return
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    models = {url: resolve_model(url, args.model) for url in args.base_url}
    if len(set(models.values())) != 1:
        raise SystemExit(f"servers report different models: {models}")
    model = models[args.base_url[0]]
    config = dict(
        dataset=args.dataset,
        source=source,
        limit=args.limit,
        per_category=args.per_category,
        num_questions=len(questions),
        seeds=args.seeds,
        model=model,
        base_url=args.base_url,
        reasoning_effort=args.reasoning_effort,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        prompt_template=prompt_template(args.dataset),
    )
    config_path = out / "run_config.json"
    if config_path.exists():
        previous = json.loads(config_path.read_text())
        keys = [k for k in config if k not in ("base_url",)]
        # Directories written before --per-category existed lack the key.
        previous.setdefault("per_category", None)
        changed = [k for k in keys if previous.get(k) != config[k]]
        if changed:
            raise SystemExit(
                f"{config_path} was written with different settings ({changed}); "
                "use a new --output"
            )
    config_path.write_text(json.dumps(config, indent=2) + "\n")

    done = {key for key, row in load_samples(out).items() if row.get("error") is None}
    todo = run_order(questions, args.seeds, done)
    slots = ServerSlots(args.base_url, args.concurrency)
    print(
        f"{len(questions)} questions x {len(args.seeds)} seeds; "
        f"{len(done)} done, {len(todo)} to run; "
        f"{len(args.base_url)} server(s) x concurrency {args.concurrency}",
        flush=True,
    )
    lock = threading.Lock()
    finished = 0
    tokens = 0
    step = max(1, min(50, len(todo) // 20))
    started = time.monotonic()
    handle = (out / "samples.jsonl").open("a")
    pool = ThreadPoolExecutor(slots.size)
    try:
        futures = [
            pool.submit(run_pair, args, slots, model, question, seed)
            for question, seed in todo
        ]
        for future in as_completed(futures):
            row = future.result()
            with lock:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                handle.flush()
                finished += 1
                tokens += row.get("completion_tokens") or 0
                if finished % step == 0 or finished == len(todo):
                    elapsed = time.monotonic() - started
                    eta = elapsed / finished * (len(todo) - finished)
                    print(
                        f"{finished}/{len(todo)}  elapsed {elapsed / 60:.1f} min"
                        f"  eta {eta / 60:.1f} min"
                        f"  {finished / elapsed * 60:.1f} req/min"
                        f"  {tokens / elapsed:.0f} out tok/s",
                        flush=True,
                    )
    except KeyboardInterrupt:
        # Do not wait for in-flight requests (minutes each at medium effort).
        # Rows written so far are kept; re-running the same command resumes.
        handle.flush()
        print(f"\ninterrupted after {finished}/{len(todo)}; rows kept", flush=True)
        os._exit(130)
    pool.shutdown()
    handle.close()
    errors = sum(1 for row in load_samples(out).values() if row.get("error"))
    if errors:
        print(f"{errors} pairs still errored; re-run the same command to retry")


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------


def binom_two_sided(k, n):
    """Exact two-sided binomial test p-value for k successes in n, p=0.5."""

    if n == 0:
        return 1.0
    k = min(k, n - k)
    tail = sum(math.comb(n, i) for i in range(k + 1))
    return min(1.0, 2 * tail / 2**n)


def sign_test(diffs):
    pos = sum(1 for d in diffs if d > 0)
    neg = sum(1 for d in diffs if d < 0)
    return dict(
        positive=pos,
        negative=neg,
        ties=len(diffs) - pos - neg,
        p=binom_two_sided(pos, pos + neg),
    )


def wilcoxon(diffs):
    """Two-sided Wilcoxon signed-rank test (zeros dropped)."""

    nonzero = [d for d in diffs if d != 0]
    n = len(nonzero)
    if n == 0:
        return dict(n=0, statistic=None, p=1.0, method="no non-zero differences")
    try:
        from scipy.stats import wilcoxon as scipy_wilcoxon

        result = scipy_wilcoxon(nonzero)
        return dict(
            n=n,
            statistic=float(result.statistic),
            p=float(result.pvalue),
            method="scipy",
        )
    except ImportError:
        pass
    # Normal approximation with tie correction.
    order = sorted(range(n), key=lambda i: abs(nonzero[i]))
    ranks = [0.0] * n
    tie_term = 0.0
    i = 0
    while i < n:
        j = i
        while j + 1 < n and abs(nonzero[order[j + 1]]) == abs(nonzero[order[i]]):
            j += 1
        rank = (i + j) / 2 + 1
        for t in range(i, j + 1):
            ranks[order[t]] = rank
        size = j - i + 1
        tie_term += size**3 - size
        i = j + 1
    w_plus = sum(r for r, d in zip(ranks, nonzero) if d > 0)
    mean = n * (n + 1) / 4
    var = n * (n + 1) * (2 * n + 1) / 24 - tie_term / 48
    if var <= 0:
        return dict(n=n, statistic=w_plus, p=1.0, method="normal approx")
    z = (w_plus - mean) / math.sqrt(var)
    p = math.erfc(abs(z) / math.sqrt(2))
    return dict(n=n, statistic=w_plus, z=z, p=p, method="normal approx")


def mcnemar(base, loop):
    """Exact McNemar: base right / loop wrong (b) vs base wrong / loop right (c)."""

    b = sum(1 for x, y in zip(base, loop) if x and not y)
    c = sum(1 for x, y in zip(base, loop) if not x and y)
    return dict(base_only=b, loop_only=c, p=binom_two_sided(b, b + c))


def _mean(values):
    values = [v for v in values if v is not None]
    return statistics.fmean(values) if values else None


def describe(rows, seeds):
    """Per-seed accuracy and generation statistics for one run."""

    valid = [r for r in rows if not r.get("error")]
    per_seed = {}
    for seed in seeds:
        seed_rows = [r for r in valid if r["seed"] == seed]
        if seed_rows:
            per_seed[seed] = statistics.fmean(r["correct"] for r in seed_rows)
    accs = list(per_seed.values())
    return dict(
        per_seed=per_seed,
        mean=statistics.fmean(accs) if accs else None,
        std=statistics.stdev(accs) if len(accs) > 1 else 0.0,
        n_rows=len(valid),
        errors=len(rows) - len(valid),
        completion_tokens=_mean(r.get("completion_tokens") for r in valid),
        reasoning_chars=_mean(r.get("reasoning_chars") for r in valid),
        reasoning_tokens=_mean(r.get("reasoning_tokens") for r in valid),
        truncation_rate=_mean(r.get("finish_reason") == "length" for r in valid),
        extraction_failure_rate=_mean(bool(r.get("extraction_failed")) for r in valid),
    )


def category_table(base_rows, loop_rows, paired, seeds):
    """Per-category mean accuracy over ``seeds`` on the paired questions,
    sorted by category name."""

    groups = {}
    for qid in paired:
        category = base_rows[(qid, seeds[0])].get("category") or "unknown"
        groups.setdefault(category, []).append(qid)

    def acc(rows, qid):
        return statistics.fmean(rows[(qid, s)]["correct"] for s in seeds)

    table = {}
    for category in sorted(groups):
        qids = groups[category]
        b = [acc(base_rows, q) for q in qids]
        l = [acc(loop_rows, q) for q in qids]
        table[category] = dict(
            n=len(qids),
            baseline=statistics.fmean(b),
            loop=statistics.fmean(l),
            delta=statistics.fmean(l) - statistics.fmean(b),
            sign_p=sign_test([y - x for x, y in zip(b, l)])["p"],
        )
    return table


def _fmt(value, spec=".4f"):
    return "n/a" if value is None else format(value, spec)


def summarize(args):
    base_rows = load_samples(args.baseline)
    loop_rows = load_samples(args.loop)
    if not base_rows or not loop_rows:
        raise SystemExit("both directories need a non-empty samples.jsonl")
    seeds = sorted({seed for _, seed in base_rows} & {seed for _, seed in loop_rows})
    base = describe(list(base_rows.values()), seeds)
    loop = describe(list(loop_rows.values()), seeds)

    # Paired per-question comparison on questions answered (no error) for
    # every common seed in both runs.
    def ok(rows, qid):
        return all((qid, s) in rows and not rows[(qid, s)].get("error") for s in seeds)

    qids = sorted({qid for qid, _ in base_rows} & {qid for qid, _ in loop_rows})
    paired = [q for q in qids if ok(base_rows, q) and ok(loop_rows, q)]
    diffs = [
        statistics.fmean(loop_rows[(q, s)]["correct"] for s in seeds)
        - statistics.fmean(base_rows[(q, s)]["correct"] for s in seeds)
        for q in paired
    ]
    sign = sign_test(diffs)
    wil = wilcoxon(diffs)
    mc_seed = args.mcnemar_seed if args.mcnemar_seed in seeds else seeds[0]
    mc_q = [
        q
        for q in qids
        if (q, mc_seed) in base_rows
        and (q, mc_seed) in loop_rows
        and not base_rows[(q, mc_seed)].get("error")
        and not loop_rows[(q, mc_seed)].get("error")
    ]
    mc = mcnemar(
        [base_rows[(q, mc_seed)]["correct"] for q in mc_q],
        [loop_rows[(q, mc_seed)]["correct"] for q in mc_q],
    )

    def config_of(directory):
        path = Path(directory) / "run_config.json"
        return json.loads(path.read_text()) if path.exists() else {}

    base_cfg, loop_cfg = config_of(args.baseline), config_of(args.loop)
    mismatched = [
        key
        for key in (
            "dataset",
            "source",
            "limit",
            "per_category",
            "reasoning_effort",
            "temperature",
            "top_p",
            "max_tokens",
            "prompt_template",
        )
        if base_cfg.get(key) != loop_cfg.get(key)
    ]

    lines = [f"# gpt-oss eval: {args.title or 'baseline vs loop'}", ""]
    lines += [
        f"- baseline: `{args.baseline}`",
        f"- loop: `{args.loop}`",
        f"- dataset: {base_cfg.get('dataset')} {base_cfg.get('source')}, "
        f"limit={base_cfg.get('limit')}, per_category={base_cfg.get('per_category')}, "
        f"reasoning_effort="
        f"{base_cfg.get('reasoning_effort')}, temperature={base_cfg.get('temperature')}, "
        f"top_p={base_cfg.get('top_p')}, max_tokens={base_cfg.get('max_tokens')}",
        f"- seeds: {seeds}; paired questions: {len(paired)}",
    ]
    expected = (base_cfg.get("num_questions") or 0) * len(seeds)
    if expected and min(base["n_rows"], loop["n_rows"]) < expected:
        lines.append(
            f"- **incomplete run: baseline {base['n_rows']}, loop {loop['n_rows']} "
            f"of {expected} (question, seed) pairs.** Pairs run in hashed order, "
            "so the paired questions are a random subset."
        )
    if mismatched:
        lines.append(f"- **WARNING: run settings differ: {mismatched}**")
    lines += ["", "## Accuracy per seed", ""]
    lines += ["| seed | baseline | loop | delta |", "| ---: | ---: | ---: | ---: |"]
    for seed in seeds:
        b, l = base["per_seed"].get(seed), loop["per_seed"].get(seed)
        delta = None if b is None or l is None else l - b
        lines.append(f"| {seed} | {_fmt(b)} | {_fmt(l)} | {_fmt(delta, '+.4f')} |")
    lines.append(
        f"| mean ± std | {_fmt(base['mean'])} ± {_fmt(base['std'])} "
        f"| {_fmt(loop['mean'])} ± {_fmt(loop['std'])} "
        f"| {_fmt(None if base['mean'] is None or loop['mean'] is None else loop['mean'] - base['mean'], '+.4f')} |"
    )
    lines += ["", "## Generation statistics", ""]
    lines += ["| metric | baseline | loop |", "| --- | ---: | ---: |"]
    for key, label, spec in [
        ("completion_tokens", "mean completion tokens", ".1f"),
        ("reasoning_tokens", "mean reasoning tokens", ".1f"),
        ("reasoning_chars", "mean reasoning length (chars)", ".1f"),
        ("truncation_rate", "truncation rate (finish_reason=length)", ".4f"),
        ("extraction_failure_rate", "answer extraction failure rate", ".4f"),
        ("errors", "request errors (excluded)", "d"),
        ("n_rows", "scored rows", "d"),
    ]:
        lines.append(f"| {label} | {_fmt(base[key], spec)} | {_fmt(loop[key], spec)} |")
    lines += ["", "## Paired tests", ""]
    lines += ["| test | statistic | p-value |", "| --- | --- | ---: |"]
    lines.append(
        f"| sign test, per-question mean over {len(seeds)} seeds (loop − baseline) "
        f"| +{sign['positive']} / −{sign['negative']} / ={sign['ties']} "
        f"| {sign['p']:.4g} |"
    )
    stat = "n/a" if wil["statistic"] is None else f"{wil['statistic']:.1f}"
    lines.append(
        f"| Wilcoxon signed-rank ({wil['method']}), same differences "
        f"| W={stat}, n={wil['n']} | {wil['p']:.4g} |"
    )
    lines.append(
        f"| McNemar exact, seed {mc_seed} ({len(mc_q)} questions) "
        f"| baseline-only {mc['base_only']} / loop-only {mc['loop_only']} "
        f"| {mc['p']:.4g} |"
    )
    per_category = category_table(base_rows, loop_rows, paired, seeds)
    if len(per_category) > 1:
        lines += [
            "",
            "## Accuracy per category (paired questions, mean over seeds)",
            "",
        ]
        lines += [
            "| category | questions | baseline | loop | delta | sign test p |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
        for name, cat in per_category.items():
            lines.append(
                f"| {name} | {cat['n']} | {cat['baseline']:.4f} | {cat['loop']:.4f} "
                f"| {cat['delta']:+.4f} | {cat['sign_p']:.3g} |"
            )
    report = "\n".join(lines) + "\n"
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(report)
    summary = dict(
        baseline=base,
        loop=loop,
        sign_test=sign,
        wilcoxon=wil,
        mcnemar=dict(mc, seed=mc_seed, n=len(mc_q)),
        paired_questions=len(paired),
        per_category=per_category,
        mismatched_settings=mismatched,
    )
    Path(args.output).with_suffix(".json").write_text(
        json.dumps(summary, indent=2, default=str) + "\n"
    )
    print(report)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="action", required=True)

    run_p = sub.add_parser("run", help="query a running server")
    run_p.add_argument(
        "--base-url",
        nargs="+",
        default=["http://127.0.0.1:30000"],
        help="one or more servers running the same model and flags (e.g. one "
        "per GPU); requests go to whichever has a free slot",
    )
    run_p.add_argument(
        "--model", help="model name for the request (default: /v1/models)"
    )
    run_p.add_argument("--dataset", choices=sorted(DATASETS), required=True)
    run_p.add_argument("--hf-dataset", help="override the HF dataset repo")
    run_p.add_argument("--hf-config", help="override the HF config, e.g. gpqa_diamond")
    run_p.add_argument("--split", help="override the split")
    run_p.add_argument(
        "--data-file",
        help="local .csv/.json/.jsonl/.parquet with the HF dataset's columns, "
        "e.g. GPQA copied to a node where no HF login is allowed",
    )
    run_p.add_argument(
        "--per-category",
        type=int,
        help="at most N questions per category, a fixed pseudo-random subset "
        "(MMLU-Pro is sorted by category, so --limit alone picks one category)",
    )
    run_p.add_argument(
        "--limit", type=int, help="first N questions only (after --per-category)"
    )
    run_p.add_argument(
        "--reasoning-effort", choices=["low", "medium", "high"], default="low"
    )
    run_p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    run_p.add_argument("--temperature", type=float, default=1.0)
    run_p.add_argument("--top-p", type=float, default=1.0)
    run_p.add_argument("--max-tokens", type=int, default=16384)
    run_p.add_argument(
        "--concurrency", type=int, default=32, help="in-flight requests per server"
    )
    run_p.add_argument("--max-retries", type=int, default=3)
    run_p.add_argument("--request-timeout", type=float, default=3600)
    run_p.add_argument(
        "--output", help="result directory (required unless --prepare-only)"
    )
    run_p.add_argument(
        "--prepare-only",
        action="store_true",
        help="only download/load the dataset (fail fast on gated data), then exit",
    )

    sum_p = sub.add_parser("summarize", help="compare baseline and loop results")
    sum_p.add_argument("baseline")
    sum_p.add_argument("loop")
    sum_p.add_argument("--output", required=True, help="markdown report path")
    sum_p.add_argument("--mcnemar-seed", type=int, default=0)
    sum_p.add_argument("--title")

    rescore_p = sub.add_parser(
        "rescore", help="re-extract answers from saved outputs (after a parser fix)"
    )
    rescore_p.add_argument("dirs", nargs="+", help="result directories")

    args = parser.parse_args(argv)
    if args.action == "run" and not args.prepare_only and not args.output:
        parser.error("run needs --output")
    {"run": run, "summarize": summarize, "rescore": rescore}[args.action](args)


if __name__ == "__main__":
    sys.exit(main())
