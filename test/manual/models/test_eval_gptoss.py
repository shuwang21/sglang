"""CPU tests for benchmark/looped_transformer/eval_gptoss.py (no server).

PYTHONPATH=python python -m pytest -q test/manual/models/test_eval_gptoss.py
"""

import importlib.util
import json
import threading
import time
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import pytest

_PATH = (
    Path(__file__).resolve().parents[3]
    / "benchmark"
    / "looped_transformer"
    / "eval_gptoss.py"
)
_spec = importlib.util.spec_from_file_location("eval_gptoss", _PATH)
ev = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ev)

CATEGORIES = ["business", "law", "math", "physics"]


def mmlu_rows(per_category=20):
    """MMLU-Pro-like rows, sorted by category like the real test split."""

    rows = []
    for c in CATEGORIES:
        for i in range(per_category):
            qid = len(rows)
            rows.append(
                dict(
                    question_id=qid,
                    question=f"q{qid}",
                    options=[f"o{j}" for j in range(10)],
                    answer="ABCDEFGHIJ"[qid % 10],
                    category=c,
                )
            )
    return rows


def questions(**kwargs):
    args = SimpleNamespace(dataset="mmlu_pro", limit=None, per_category=None)
    for key, value in kwargs.items():
        setattr(args, key, value)
    return ev._to_questions(args, mmlu_rows())


def test_limit_alone_takes_the_first_category():
    # The bug this replaces: the first N of MMLU-Pro is a single category.
    picked = questions(limit=10)
    assert {q["category"] for q in picked} == {"business"}


def test_per_category_is_balanced_stable_and_in_dataset_order():
    picked = questions(per_category=5)
    assert Counter(q["category"] for q in picked) == {c: 5 for c in CATEGORIES}
    assert [q["id"] for q in picked] == [q["id"] for q in questions(per_category=5)]
    ids = [int(q["id"].split("/")[1]) for q in picked]
    assert ids == sorted(ids)
    # Not simply the first 5 of each category.
    assert [q["id"] for q in picked[:5]] != [f"mmlu_pro/{i}" for i in range(5)]


def test_per_category_larger_than_category_keeps_all():
    assert len(questions(per_category=1000)) == len(mmlu_rows())


def test_per_category_then_limit():
    picked = questions(per_category=5, limit=7)
    assert len(picked) == 7
    assert picked == questions(per_category=5)[:7]


def test_run_order_is_seed_major_hashed_and_skips_done():
    qs = questions()
    done = {(qs[0]["id"], 0)}
    todo = ev.run_order(qs, [0, 1], done)
    assert len(todo) == 2 * len(qs) - 1
    assert [s for _, s in todo] == [0] * (len(qs) - 1) + [1] * len(qs)
    seed1 = [q["id"] for q, s in todo if s == 1]
    assert seed1 == sorted(seed1, key=ev.question_hash)
    # A prefix (a cut-off run) already spans every category.
    prefix = {q["category"] for q, _ in todo[:12]}
    assert prefix == set(CATEGORIES)


def test_server_slots_bound_per_server_and_spread_load():
    urls = ["http://a", "http://b"]
    slots = ev.ServerSlots(urls, concurrency=3)
    assert slots.size == 6
    lock = threading.Lock()
    active = Counter()
    peak = Counter()
    used = Counter()

    def work():
        with slots.take() as url:
            with lock:
                active[url] += 1
                used[url] += 1
                peak[url] = max(peak[url], active[url])
            time.sleep(0.01)
            with lock:
                active[url] -= 1

    threads = [threading.Thread(target=work) for _ in range(60)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert set(used) == set(urls)
    assert all(peak[u] <= 3 for u in urls)
    assert sum(used.values()) == 60


def _write_run(directory, rows, config):
    directory.mkdir(parents=True)
    (directory / "run_config.json").write_text(json.dumps(config))
    with (directory / "samples.jsonl").open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def test_summarize_per_category_and_incomplete(tmp_path):
    qs = questions(per_category=4)
    config = dict(dataset="mmlu_pro", num_questions=len(qs), per_category=4)

    def rows(correct_fn, n):
        return [
            dict(
                id=q["id"],
                seed=0,
                category=q["category"],
                gold="A",
                predicted="A",
                correct=correct_fn(q),
                extraction_failed=False,
                finish_reason="stop",
                completion_tokens=10,
                error=None,
            )
            for q in qs[:n]
        ]

    _write_run(tmp_path / "b", rows(lambda q: q["category"] != "law", 16), config)
    _write_run(tmp_path / "l", rows(lambda q: True, 12), config)
    out = tmp_path / "r.md"
    ev.main(
        ["summarize", str(tmp_path / "b"), str(tmp_path / "l"), "--output", str(out)]
    )
    report = out.read_text()
    assert "incomplete run: baseline 16, loop 12 of 16" in report
    summary = json.loads(out.with_suffix(".json").read_text())
    assert summary["paired_questions"] == 12
    table = summary["per_category"]
    assert table["law"]["baseline"] == 0.0 and table["law"]["loop"] == 1.0
    assert table["business"]["delta"] == 0.0
    assert "physics" not in table  # its questions were not run by the loop arm


def test_resume_accepts_config_written_before_per_category(tmp_path, monkeypatch):
    out = tmp_path / "run"
    out.mkdir()
    qs = questions(limit=3)
    old = dict(
        dataset="mmlu_pro",
        source={},
        limit=3,
        num_questions=3,
        seeds=[0],
        model="m",
        base_url="http://x",
        reasoning_effort="low",
        temperature=1.0,
        top_p=1.0,
        max_tokens=16384,
        prompt_template=ev.PROMPT,
    )
    (out / "run_config.json").write_text(json.dumps(old))
    monkeypatch.setattr(ev, "load_questions", lambda args: (qs, {}))
    monkeypatch.setattr(ev, "resolve_model", lambda url, model: "m")
    posted = []
    monkeypatch.setattr(
        ev,
        "_post",
        lambda url, payload, timeout: posted.append(url)
        or {
            "choices": [{"message": {"content": "Answer: A"}, "finish_reason": "stop"}],
            "usage": {"completion_tokens": 5},
        },
    )
    ev.main(
        [
            "run",
            "--dataset",
            "mmlu_pro",
            "--limit",
            "3",
            "--seeds",
            "0",
            "--reasoning-effort",
            "low",
            "--base-url",
            "http://x",
            "http://y",
            "--output",
            str(out),
        ]
    )
    assert len(ev.load_samples(out)) == 3
    assert set(posted) <= {
        "http://x/v1/chat/completions",
        "http://y/v1/chat/completions",
    }

    # A real settings change is still refused.
    with pytest.raises(SystemExit):
        ev.main(
            [
                "run",
                "--dataset",
                "mmlu_pro",
                "--limit",
                "3",
                "--seeds",
                "0",
                "--reasoning-effort",
                "low",
                "--per-category",
                "2",
                "--output",
                str(out),
            ]
        )


# ---------------------------------------------------------------------------
# AIME 2024 (integer answers)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text, expected",
    [
        ("so x = 5.\nAnswer: 204", "204"),
        ("Answer: $\\boxed{033}$", "33"),
        ("**Answer:** 1,000", "1000"),
        ("the result is \\boxed{ 70 }", "70"),
        ("\\boxed{\\text{25}}", "25"),
        ("Answer: 12\nwait, recheck.\nAnswer: 13", "13"),
        ("Thus xy is \\(25\\).\n\nAnswer: \\(25\\)", "25"),
        ("**Answer:** \\(2730\\)", "2730"),
        ("Answer: \\(116\\).", "116"),
        ("Answer: \\(\\boxed{33}\\)", "33"),
        ("Answer: \\[ 204 \\]", "204"),
        ("Answer: \\(\\displaystyle 107\\)", "107"),
        ("\\[\n\\boxed{\\text{Answer: }721}\n\\]", "721"),
        ("\\[\n\\text{Answer: }116\n\\]", "116"),
        ("Answer: $\\displaystyle 116$", "116"),
        ("Answer: 3.5", None),
        ("no final answer", None),
        ("", None),
    ],
)
def test_extract_integer(text, expected):
    assert ev.extract_integer(text) == expected


def test_aime_rows_to_questions_both_column_styles():
    args = SimpleNamespace(dataset="aime24", limit=None, per_category=None)
    h4 = [dict(id=60, problem=" Find x. ", answer="033", solution="...")]
    capital = [dict(ID="2024-I-1", Problem="P", Answer=204)]
    (a,) = ev._to_questions(args, h4)
    (b,) = ev._to_questions(args, capital)
    assert a == dict(
        id="aime24/60",
        question="Find x.",
        options=None,
        gold="33",
        category="aime24",
    )
    assert b["id"] == "aime24/2024-I-1" and b["gold"] == "204"
    prompt = ev.prompt_for(a)
    assert prompt.endswith("Find x.") and "'Answer: $NUMBER'" in prompt
    assert ev.prompt_template("aime24") == ev.MATH_PROMPT
    assert ev.prompt_template("mmlu_pro") == ev.PROMPT  # unchanged: old runs resume


def test_aime_run_and_summarize_end_to_end(tmp_path, monkeypatch):
    qs = [
        dict(
            id=f"aime24/{i}", question=f"p{i}", options=None, gold=str(i), category=None
        )
        for i in range(3)
    ]
    monkeypatch.setattr(ev, "load_questions", lambda args: (qs, {}))
    monkeypatch.setattr(ev, "resolve_model", lambda url, model: "m")

    def server(right):
        def post(url, payload, timeout):
            assert payload["max_tokens"] == 32768
            assert payload["reasoning_effort"] == "low"
            problem = payload["messages"][0]["content"].rsplit("\n", 1)[-1]
            i = int(problem[1:])
            answer = i if right(i, payload["seed"]) else 999
            return {
                "choices": [
                    {
                        "message": {
                            "content": f"Answer: {answer}",
                            "reasoning_content": "thinking",
                        },
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"completion_tokens": 7},
            }

        return post

    common = "--dataset aime24 --reasoning-effort low --max-tokens 32768 --seeds 0 1 --concurrency 6".split()
    monkeypatch.setattr(ev, "_post", server(lambda i, seed: i == 0))
    ev.main(["run", *common, "--output", str(tmp_path / "b")])
    monkeypatch.setattr(ev, "_post", server(lambda i, seed: i < 2))
    ev.main(["run", *common, "--output", str(tmp_path / "l")])

    rows = [
        json.loads(l)
        for l in (tmp_path / "l" / "samples.jsonl").read_text().splitlines()
    ]
    assert len(rows) == 6 and all(r["error"] is None for r in rows)
    config = json.loads((tmp_path / "l" / "run_config.json").read_text())
    assert config["prompt_template"] == ev.MATH_PROMPT

    out = tmp_path / "r.md"
    ev.main(
        ["summarize", str(tmp_path / "b"), str(tmp_path / "l"), "--output", str(out)]
    )
    summary = json.loads(out.with_suffix(".json").read_text())
    assert summary["paired_questions"] == 3
    assert summary["baseline"]["mean"] == pytest.approx(1 / 3)
    assert summary["loop"]["mean"] == pytest.approx(2 / 3)
    assert (
        summary["sign_test"]["positive"] == 1 and summary["sign_test"]["negative"] == 0
    )
    assert summary["per_category"] == {"unknown": summary["per_category"]["unknown"]}
    assert "Accuracy per category" not in out.read_text()


@pytest.mark.parametrize(
    "text, expected",
    [
        ("Answer: B", "B"),
        ("Answer: $B$", "B"),
        ("**Answer:** \\(C\\)", "C"),
        ("Answer: \\(\\text{D}\\)", "D"),
        ("Answer: (E)", "E"),
        ("\\boxed{F}", "F"),
        ("\\boxed{\\text{Answer: }G}", "G"),
        ("Answer: \\(\\displaystyle H\\)", "H"),
        ("Answer: \\frac{1}{2}", None),
    ],
)
def test_extract_letter_with_latex_wrappers(text, expected):
    assert ev.extract_answer(text, 10) == expected


def test_rescore_fixes_saved_rows_and_keeps_original(tmp_path):
    run = tmp_path / "run"
    rows = [
        dict(
            id="aime24/1",
            seed=0,
            gold="25",
            predicted=None,
            correct=False,
            extraction_failed=True,
            output="Answer: \\(25\\)",
            reasoning="r",
            error=None,
        ),
        dict(
            id="aime24/2",
            seed=0,
            gold="7",
            predicted="7",
            correct=True,
            extraction_failed=False,
            output="Answer: 7",
            reasoning="r",
            error=None,
        ),
        dict(
            id="aime24/3",
            seed=0,
            gold="9",
            predicted=None,
            correct=False,
            extraction_failed=None,
            error="URLError: down",
        ),
    ]
    _write_run(run, rows, dict(dataset="aime24"))
    original = (run / "samples.jsonl").read_text()
    ev.main(["rescore", str(run)])
    fixed = [json.loads(l) for l in (run / "samples.jsonl").read_text().splitlines()]
    assert fixed[0]["predicted"] == "25" and fixed[0]["correct"] is True
    assert fixed[0]["extraction_failed"] is False
    assert fixed[1]["correct"] is True
    assert fixed[2]["error"] and fixed[2]["predicted"] is None  # errors untouched
    assert (run / "samples.jsonl.orig").read_text() == original
    ev.main(["rescore", str(run)])  # idempotent; the backup stays the original
    assert (run / "samples.jsonl.orig").read_text() == original


def test_aime25_rows_and_combined_aime(monkeypatch):
    import sys
    import types

    tables = {
        "HuggingFaceH4/aime_2024": [
            dict(id=i, problem=f"a{i}", answer=str(i)) for i in range(30)
        ],
        "MathArena/aime_2025": [
            dict(
                problem_idx=str(i + 1),
                problem=f"b{i}",
                answer=str(100 + i),
                problem_type="[]",
            )
            for i in range(30)
        ],
    }
    loaded = []

    def load_dataset(path, *config, split):
        loaded.append((path, config, split))
        return tables[path]

    monkeypatch.setitem(
        sys.modules, "datasets", types.SimpleNamespace(load_dataset=load_dataset)
    )

    def args(**kw):
        base = dict(
            dataset="aime",
            data_file=None,
            hf_dataset=None,
            hf_config=None,
            split=None,
            limit=None,
            per_category=None,
        )
        base.update(kw)
        return SimpleNamespace(**base)

    qs, source = ev.load_questions(args(dataset="aime25"))
    assert len(qs) == 30 and qs[0]["id"] == "aime25/1" and qs[0]["gold"] == "100"
    assert source == dict(path="MathArena/aime_2025", config=None, split="train")

    loaded.clear()
    qs, source = ev.load_questions(args())
    assert [p for p, _, _ in loaded] == [
        "HuggingFaceH4/aime_2024",
        "MathArena/aime_2025",
    ]
    assert len(qs) == 60 and len({q["id"] for q in qs}) == 60
    assert Counter(q["category"] for q in qs) == {"aime24": 30, "aime25": 30}
    assert set(source) == {"aime24", "aime25"}
    assert ev.prompt_for(qs[-1]).endswith("b29")
    assert ev.prompt_template("aime") == ev.MATH_PROMPT

    # Sampling applies to the union, so --per-category keeps both years.
    qs, _ = ev.load_questions(args(per_category=5))
    assert Counter(q["category"] for q in qs) == {"aime24": 5, "aime25": 5}
    with pytest.raises(SystemExit):
        ev.load_questions(args(hf_dataset="x/y"))
