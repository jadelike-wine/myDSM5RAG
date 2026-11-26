"""Gold set 检索评测脚本（第一步：hit-rate@k / MRR）。

只测量检索链路本身：用确定性的词袋 embedding 替身（tests/lexical_embedding.py）
和真实的 BM25 关键词索引，在 tests/fixtures/corpus 上构建索引，按 gold set 逐条
查询，统计每个模式在 top-k 内的命中率与首个命中的倒数排名。

刻意不做的事：不引入 rerank、不调用 LLM、不新增依赖、不改动服务默认配置语义。
因此这里直接取 HybridRetriever 的排序结果，不套 SimilarityPostprocessor 阈值，
也不做 LongContextReorder（它们属于后续步骤）。

用法：
    uv run python scripts/eval_gold_set.py
    uv run python scripts/eval_gold_set.py --modes KEYWORD_ONLY,HYBRID --ks 1,3,5
    uv run python scripts/eval_gold_set.py --json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Sequence

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CORPUS = ROOT / "tests" / "fixtures" / "corpus"
DEFAULT_GOLD_SET = ROOT / "tests" / "fixtures" / "gold_set.json"
DEFAULT_MODES = ("KEYWORD_ONLY", "HYBRID")


# ==================== 纯函数：gold set 与指标 ====================


def load_gold_set(path: Path) -> List[Dict[str, Any]]:
    """读取并校验 gold set。schema 错误直接抛异常，避免静默算出无意义的指标。"""
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not isinstance(data.get("questions"), list):
        raise ValueError(f"gold set 格式错误：{path} 顶层需要 questions 列表")

    questions: List[Dict[str, Any]] = []
    seen_ids: set[str] = set()
    for index, item in enumerate(data["questions"]):
        if not isinstance(item, dict):
            raise ValueError(f"gold set 第 {index} 项不是对象")
        qid = item.get("id")
        question = item.get("question")
        expected = item.get("expected_files")
        if not isinstance(qid, str) or not qid.strip():
            raise ValueError(f"gold set 第 {index} 项缺少有效的 id")
        if qid in seen_ids:
            raise ValueError(f"gold set id 重复：{qid}")
        if not isinstance(question, str) or not question.strip():
            raise ValueError(f"gold set 第 {index} 项缺少有效的 question")
        if (
            not isinstance(expected, list)
            or not expected
            or not all(isinstance(name, str) and name.strip() for name in expected)
        ):
            raise ValueError(f"gold set 第 {index} 项缺少有效的 expected_files")
        seen_ids.add(qid)
        questions.append(
            {
                "id": qid,
                "question": question,
                "expected_files": sorted({name.strip() for name in expected}),
            }
        )

    if not questions:
        raise ValueError(f"gold set 为空：{path}")
    return questions


def hit_rate_at_k(ranked_files: Sequence[str], expected_files: Sequence[str], k: int) -> float:
    """单个查询的 hit@k：前 k 个文件里出现任一期望文件即为 1，否则为 0。"""
    if k <= 0:
        raise ValueError("k 必须为正整数")
    expected = set(expected_files)
    return 1.0 if any(name in expected for name in ranked_files[:k]) else 0.0


def reciprocal_rank(ranked_files: Sequence[str], expected_files: Sequence[str]) -> float:
    """单个查询的倒数排名：首个期望文件的名次取倒数，未命中为 0。"""
    expected = set(expected_files)
    for rank, name in enumerate(ranked_files, start=1):
        if name in expected:
            return 1.0 / rank
    return 0.0


def evaluate_mode(
    rankings: Dict[str, List[str]],
    questions: List[Dict[str, Any]],
    ks: Sequence[int],
) -> Dict[str, Any]:
    """把一个模式的逐查询排序汇总成 hit-rate@k 与 MRR。"""
    missing = [q["id"] for q in questions if q["id"] not in rankings]
    if missing:
        raise KeyError(f"缺少查询结果：{', '.join(missing)}")

    per_query: List[Dict[str, Any]] = []
    for item in questions:
        files = rankings[item["id"]]
        expected = item["expected_files"]
        ranks = [rank for rank, name in enumerate(files, start=1) if name in set(expected)]
        first_hit = ranks[0] if ranks else None
        per_query.append(
            {
                "id": item["id"],
                "hit_at_k": {str(k): bool(hit_rate_at_k(files, expected, k)) for k in ks},
                "first_hit_rank": first_hit,
                "reciprocal_rank": round(reciprocal_rank(files, expected), 4),
            }
        )

    count = len(questions)
    return {
        "num_queries": count,
        "hit_rate_at_k": {
            str(k): round(sum(q["hit_at_k"][str(k)] for q in per_query) / count, 4) for k in ks
        },
        "mrr": round(sum(q["reciprocal_rank"] for q in per_query) / count, 4),
        "per_query": per_query,
    }


def evaluate_all(
    mode_rankings: Dict[str, Dict[str, List[str]]],
    questions: List[Dict[str, Any]],
    ks: Sequence[int],
) -> Dict[str, Any]:
    return {
        mode: evaluate_mode(rankings, questions, ks) for mode, rankings in mode_rankings.items()
    }


def parse_csv(value: str, cast=str) -> List[Any]:
    return [cast(part.strip()) for part in value.split(",") if part.strip()]


# ==================== 检索与报告 ====================


async def retrieve_rankings(
    system: Any,
    questions: List[Dict[str, Any]],
    modes: Sequence[str],
    top_k: int,
    vector_weight: float,
) -> Dict[str, Dict[str, List[str]]]:
    """按 (模式, 查询) 取检索器输出的 file_name 排序列表。"""
    from llama_index.core import QueryBundle

    mode_rankings: Dict[str, Dict[str, List[str]]] = {}
    for mode in modes:
        per_query: Dict[str, List[str]] = {}
        for item in questions:
            retriever = system._build_hybrid_retriever(mode, top_k, vector_weight)
            nodes = retriever.retrieve(QueryBundle(item["question"]))
            per_query[item["id"]] = [str(node.node.metadata.get("file_name", "")) for node in nodes]
        mode_rankings[mode] = per_query
    return mode_rankings


def build_report(
    corpus: Path,
    gold_set: Path,
    modes: Sequence[str],
    ks: Sequence[int],
    top_k: int,
    index_stats: Dict[str, Any],
    metrics: Dict[str, Any],
) -> Dict[str, Any]:
    return {
        "corpus": str(corpus),
        "gold_set": str(gold_set),
        "top_k": top_k,
        "modes": list(modes),
        "ks": list(ks),
        "index_stats": index_stats,
        "metrics": metrics,
    }


def print_report(report: Dict[str, Any], verbose: bool) -> None:
    print(f"语料: {report['corpus']}")
    print(f"gold set: {report['gold_set']}  检索 top_k={report['top_k']}")
    for mode, result in report["metrics"].items():
        hits = "  ".join(f"hit@{k}={value:.3f}" for k, value in result["hit_rate_at_k"].items())
        print(f"{mode}: n={result['num_queries']}  {hits}  MRR={result['mrr']:.3f}")
        if verbose:
            for query in result["per_query"]:
                rank = query["first_hit_rank"]
                print(
                    f"  {query['id']:24s} first_hit_rank={rank} rr={query['reciprocal_rank']:.4f}"
                )


async def run(args: argparse.Namespace) -> Dict[str, Any]:
    corpus: Path = args.corpus.resolve()
    gold_path: Path = args.gold_set.resolve()
    if not corpus.is_dir():
        raise FileNotFoundError(f"语料目录不存在：{corpus}")
    if not gold_path.is_file():
        raise FileNotFoundError(f"gold set 不存在：{gold_path}")

    questions = load_gold_set(gold_path)
    corpus_files = {path.name for path in corpus.iterdir() if path.is_file()}
    expected = {name for item in questions for name in item["expected_files"]}
    unknown = sorted(expected - corpus_files)
    if unknown:
        raise ValueError(f"gold set 期望的文件不在语料目录中：{', '.join(unknown)}")

    modes = tuple(args.modes)
    ks = sorted(set(args.ks))
    top_k = args.top_k or max(ks)

    tmp = Path(tempfile.mkdtemp(prefix="dsm5-gold-eval-"))
    os.environ["DOCUMENTS_PATH"] = str(corpus)
    os.environ["PERSIST_DIR"] = str(tmp / "chroma")
    os.environ["MODELS_DIR"] = str(tmp / "models")
    os.environ["DEEPSEEK_API_KEY"] = ""
    os.environ["LOG_LEVEL"] = "WARNING"

    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(ROOT / "src"))

    from tests.lexical_embedding import install_fake_huggingface_embedding

    install_fake_huggingface_embedding(embed_dim=256)

    from dsm5_rag.config import RagConfig
    from dsm5_rag.system import DeepSeekRAGSystem

    config = RagConfig()
    system = DeepSeekRAGSystem(config, auto_build_index=False)
    system._init_storage()
    built = await system.build_index(force_rebuild=True)
    if built.get("status") not in ("built", "loaded"):
        raise RuntimeError(f"索引构建失败：{built}")

    mode_rankings = await retrieve_rankings(
        system, questions, modes, top_k, config.default_vector_weight
    )
    metrics = evaluate_all(mode_rankings, questions, ks)
    return build_report(corpus, gold_path, modes, ks, top_k, built, metrics)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="gold set 检索评测（hit-rate@k / MRR）")
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS, help="语料目录")
    parser.add_argument("--gold-set", type=Path, default=DEFAULT_GOLD_SET, help="gold set JSON")
    parser.add_argument(
        "--modes",
        type=str,
        default=",".join(DEFAULT_MODES),
        help="逗号分隔的检索模式，默认 KEYWORD_ONLY,HYBRID",
    )
    parser.add_argument("--ks", type=str, default="1,3,5", help="逗号分隔的 k，默认 1,3,5")
    parser.add_argument("--top-k", type=int, default=None, help="检索条数，默认取最大 k")
    parser.add_argument("--json", action="store_true", help="只输出 JSON 结果")
    parser.add_argument("--verbose", action="store_true", help="逐查询打印命中名次")
    args = parser.parse_args(argv)

    args.modes = parse_csv(args.modes)
    args.ks = sorted(set(parse_csv(args.ks, int)))
    if not args.modes:
        parser.error("--modes 不能为空")
    if not args.ks or any(k <= 0 for k in args.ks):
        parser.error("--ks 必须都是正整数")
    if args.top_k is not None and args.top_k < max(args.ks):
        parser.error("--top-k 不能小于最大的 k")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        report = asyncio.run(run(args))
    except Exception as exc:
        print(f"评测失败：{type(exc).__name__}: {exc}", file=sys.stderr)
        return 2

    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print_report(report, args.verbose)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
