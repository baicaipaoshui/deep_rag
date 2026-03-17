from __future__ import annotations

from evaluation import run_eval


def main() -> None:
    assert run_eval.Dataset is not None, "datasets unavailable"
    assert run_eval.ragas_evaluate is not None, "ragas unavailable"
    assert run_eval.context_precision is not None, "ragas.context_precision unavailable"
    assert run_eval.context_recall is not None, "ragas.context_recall unavailable"
    assert run_eval.faithfulness is not None, "ragas.faithfulness unavailable"
    assert run_eval.answer_relevancy is not None, "ragas.answer_relevancy unavailable"
    print("ragas precheck ok")


if __name__ == "__main__":
    main()
