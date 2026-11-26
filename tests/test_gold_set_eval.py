"""gold set 评测脚本的指标计算与 gold set 自身校验。

不构建索引、不下载模型：只覆盖纯函数逻辑和 fixtures/gold_set.json 的完整性，
检索质量的实跑结果由 `uv run python scripts/eval_gold_set.py` 给出。
"""

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts" / "eval_gold_set.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("eval_gold_set", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


eval_gold_set = _load_script()


def test_hit_rate_at_k_counts_expected_file_within_first_k():
    files = ["a.txt", "b.txt", "c.txt"]
    assert eval_gold_set.hit_rate_at_k(files, ["a.txt"], 1) == 1.0
    assert eval_gold_set.hit_rate_at_k(files, ["c.txt"], 1) == 0.0
    assert eval_gold_set.hit_rate_at_k(files, ["c.txt"], 3) == 1.0
    assert eval_gold_set.hit_rate_at_k(files, ["missing.txt"], 3) == 0.0


def test_reciprocal_rank_uses_first_expected_hit():
    files = ["x.txt", "a.txt", "b.txt"]
    assert eval_gold_set.reciprocal_rank(files, ["b.txt"]) == 1 / 3
    assert eval_gold_set.reciprocal_rank(files, ["a.txt", "b.txt"]) == 1 / 2
    assert eval_gold_set.reciprocal_rank(files, ["missing.txt"]) == 0.0
    assert eval_gold_set.reciprocal_rank([], ["a.txt"]) == 0.0


def test_hit_rate_rejects_non_positive_k():
    with pytest.raises(ValueError):
        eval_gold_set.hit_rate_at_k(["a.txt"], ["a.txt"], 0)


def test_evaluate_mode_aggregates_hit_rate_and_mrr():
    questions = [
        {"id": "q1", "question": "one", "expected_files": ["a.txt"]},
        {"id": "q2", "question": "two", "expected_files": ["b.txt"]},
    ]
    rankings = {
        "q1": ["x.txt", "a.txt"],
        "q2": ["x.txt", "y.txt"],
    }
    result = eval_gold_set.evaluate_mode(rankings, questions, [1, 2])
    assert result["num_queries"] == 2
    assert result["hit_rate_at_k"] == {"1": 0.0, "2": 0.5}
    assert result["mrr"] == 0.25
    assert result["per_query"][0]["first_hit_rank"] == 2
    assert result["per_query"][1]["first_hit_rank"] is None


def test_evaluate_mode_requires_result_for_every_question():
    questions = [{"id": "q1", "question": "one", "expected_files": ["a.txt"]}]
    with pytest.raises(KeyError):
        eval_gold_set.evaluate_mode({}, questions, [1])


def test_gold_set_fixture_is_well_formed_and_matches_corpus():
    questions = eval_gold_set.load_gold_set(ROOT / "tests" / "fixtures" / "gold_set.json")
    assert len(questions) >= 5

    corpus_files = {
        path.name for path in (ROOT / "tests" / "fixtures" / "corpus").iterdir() if path.is_file()
    }
    for item in questions:
        assert item["id"].strip()
        assert item["question"].strip()
        assert item["expected_files"]
        assert set(item["expected_files"]) <= corpus_files, item["id"]


def test_load_gold_set_rejects_empty_or_duplicate_entries(tmp_path):
    empty = tmp_path / "empty.json"
    empty.write_text('{"questions": []}', encoding="utf-8")
    with pytest.raises(ValueError):
        eval_gold_set.load_gold_set(empty)

    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text(
        '{"questions": ['
        '{"id": "q1", "question": "one", "expected_files": ["a.txt"]}, '
        '{"id": "q1", "question": "two", "expected_files": ["b.txt"]}]}',
        encoding="utf-8",
    )
    with pytest.raises(ValueError):
        eval_gold_set.load_gold_set(duplicate)
