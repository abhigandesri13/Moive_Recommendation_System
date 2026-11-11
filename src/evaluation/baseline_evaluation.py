import pandas as pd

from src.evaluation.ranking_metrics import (
    aggregate_user_metrics,
    catalog_coverage,
    evaluate_user,
)
from src.recommenders.baseline.collaborative_baseline import (
    CollaborativeBaseline,
)


def evaluate_baseline(
    model: CollaborativeBaseline,
    train: pd.DataFrame,
    test: pd.DataFrame,
    k: int = 10,
) -> dict[str, float]:
    """
    Evaluate the collaborative-filtering baseline across all users.

    The model must already be fitted on the training data.
    Test positives are defined using the is_positive column.
    """

    if k <= 0:
        raise ValueError("k must be greater than zero.")

    required_columns = {"userId", "movieId", "is_positive"}

    missing_columns = required_columns - set(test.columns)

    if missing_columns:
        raise ValueError(
            f"test: missing required columns: {sorted(missing_columns)}"
        )

    user_results: list[dict[str, float]] = []
    all_recommended_items: set[int] = set()

    users_evaluated = 0

    for user_id in sorted(test["userId"].unique()):

        # Relevant items = positive interactions in the user's test period.
        relevant_items = set(
            test[
                (test["userId"] == user_id)
                & (test["is_positive"])
            ]["movieId"]
            .astype(int)
            .tolist()
        )

        if not relevant_items:
            continue

        recommendations = model.recommend(
            user_id=int(user_id),
            n=k,
        )

        recommended_items = (
            recommendations["movieId"]
            .astype(int)
            .tolist()
        )

        if not recommended_items:
            continue

        all_recommended_items.update(recommended_items)

        metrics = evaluate_user(
            recommended_items=recommended_items,
            relevant_items=relevant_items,
            k=k,
        )

        user_results.append(metrics)
        users_evaluated += 1

    if not user_results:
        raise RuntimeError(
            "No users could be evaluated."
        )

    metrics = aggregate_user_metrics(user_results)

    # Global catalog coverage.
    catalog_size = train["movieId"].nunique()

    metrics["catalog_coverage"] = catalog_coverage(
        all_recommended_items,
        catalog_size,
    )

    metrics["users_evaluated"] = float(users_evaluated)
    metrics["catalog_size"] = float(catalog_size)

    return metrics


def run_baseline_evaluation() -> dict[str, float]:
    """Load data, train the baseline, and evaluate it."""

    from src.data.loaders import MovieLensLoader
    from src.data.preprocessing.rating_preprocessor import (
        preprocess_ratings,
    )
    from src.data.preprocessing.splitter import temporal_split

    ratings = preprocess_ratings(
        MovieLensLoader("data").load_ratings()
    )

    train, _, test = temporal_split(ratings)

    model = CollaborativeBaseline()

    model.fit(train)

    return evaluate_baseline(
        model=model,
        train=train,
        test=test,
        k=10,
    )


if __name__ == "__main__":
    results = run_baseline_evaluation()

    print("BASELINE EVALUATION COMPLETE")
    print()

    for metric, value in results.items():
        print(f"{metric}: {value:.6f}")