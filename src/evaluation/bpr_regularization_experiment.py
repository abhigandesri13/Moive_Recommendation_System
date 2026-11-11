from __future__ import annotations

import csv
import time
from pathlib import Path

from src.data.loaders import MovieLensLoader
from src.data.preprocessing.rating_preprocessor import preprocess_ratings
from src.data.preprocessing.splitter import temporal_split
from src.evaluation.bpr_validation import evaluate_bpr_on_validation
from src.recommenders.collaborative.bpr import BPRRecommender


RESULTS_DIR = Path("experiments") / "results"
RESULTS_FILE = RESULTS_DIR / "bpr_regularization_experiments.csv"


def run_regularization_experiments() -> None:
    """Compare BPR regularization values using validation data."""

    ratings = preprocess_ratings(
        MovieLensLoader("data").load_ratings()
    )

    train, validation, _ = temporal_split(ratings)

    regularization_values = [0.0001, 0.001, 0.01]

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    rows = []

    for regularization in regularization_values:

        print()
        print("=" * 60)
        print(
            f"BPR REGULARIZATION EXPERIMENT: "
            f"{regularization}"
        )
        print("=" * 60)

        model = BPRRecommender(
            factors=32,
            learning_rate=0.005,
            regularization=regularization,
            epochs=10,
            random_seed=42,
        )

        start_time = time.perf_counter()

        model.fit(train)

        training_time = time.perf_counter() - start_time

        metrics = evaluate_bpr_on_validation(
            model=model,
            validation=validation,
            k=10,
        )

        row = {
            "factors": 32,
            "epochs": 10,
            "learning_rate": 0.005,
            "regularization": regularization,
            "precision@10": metrics["precision@10"],
            "recall@10": metrics["recall@10"],
            "ndcg@10": metrics["ndcg@10"],
            "users_evaluated": metrics["users_evaluated"],
            "training_time_seconds": training_time,
        }

        rows.append(row)

        print()
        print("RESULT")
        print("-" * 40)

        for key, value in row.items():
            if isinstance(value, float):
                print(f"{key}: {value:.6f}")
            else:
                print(f"{key}: {value}")

    fieldnames = [
        "factors",
        "epochs",
        "learning_rate",
        "regularization",
        "precision@10",
        "recall@10",
        "ndcg@10",
        "users_evaluated",
        "training_time_seconds",
    ]

    with RESULTS_FILE.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(rows)

    print()
    print("=" * 60)
    print("BPR REGULARIZATION EXPERIMENT COMPLETE")
    print("=" * 60)
    print(f"Results saved to: {RESULTS_FILE}")


if __name__ == "__main__":
    run_regularization_experiments()