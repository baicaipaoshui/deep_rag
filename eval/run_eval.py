#!/usr/bin/env python3
"""Retrieval eval: Recall@1/3/5 + MRR.  python -m eval.run_eval"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mcp_server.core.index_manager import IndexManager
from mcp_server.core.search_engine import SearchEngine


def load_qa(path: str) -> list[dict]:
    return [json.loads(l) for l in Path(path).read_text("utf-8").splitlines() if l.strip()]


def recall_at_k(results, expected_file, k):
    return int(any(r.get("file_name") == expected_file for r in results[:k]))


def reciprocal_rank(results, expected_file):
    for i, r in enumerate(results, 1):
        if r.get("file_name") == expected_file:
            return 1.0 / i
    return 0.0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--qa", default="eval/sample_qa.jsonl")
    p.add_argument("--k",  type=int, default=5)
    args = p.parse_args()

    engine = SearchEngine(IndexManager())
    items  = load_qa(args.qa)
    r1 = r3 = r5 = mrr = 0.0
    for item in items:
        res = engine.search(item["question"], search_type="hybrid", max_results=args.k)
        ef  = item["expected_file"]
        r1  += recall_at_k(res, ef, 1)
        r3  += recall_at_k(res, ef, 3)
        r5  += recall_at_k(res, ef, 5)
        mrr += reciprocal_rank(res, ef)
    n = len(items)
    print(json.dumps({"n": n, "Recall@1": round(r1/n, 4),
                      "Recall@3": round(r3/n, 4), "Recall@5": round(r5/n, 4),
                      "MRR": round(mrr/n, 4)}, indent=2))


if __name__ == "__main__":
    main()
