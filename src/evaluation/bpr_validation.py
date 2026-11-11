from src.data.loaders import MovieLensLoader
from src.data.preprocessing.rating_preprocessor import preprocess_ratings
from src.data.preprocessing.splitter import temporal_split
from src.evaluation.ranking_metrics import (
    aggregate_user_metrics,
    evaluate_user,
)
from src.recommenders.collaborative.bpr import BPRRecommender


def evaluate_bpr_on_validation(
    model: BPRRecommender,
    validation,
    k: int = 10,
) -> dict[str, float]:
    """Evaluate a trained BPR model using positive validation interactions."""

    if k <= 0:
        raise ValueError("k must be greater than zero.")

    user_results = []
    users_evaluated = 0

    for user_id in sorted(validation["userId"].unique()):

        relevant_items = set(
            validation[
                (validation["userId"] == user_id)
                & (validation["is_positive"])
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

        if recommendations.empty:
            continue

        recommended_items = (
            recommendations["movieId"]
            .astype(int)
            .tolist()
        )

        metrics = evaluate_user(
            recommended_items=recommended_items,
            relevant_items=relevant_items,
            k=k,
        )

        user_results.append(metrics)
        users_evaluated += 1

    if not user_results:
        raise RuntimeError(
            "No users could be evaluated on validation data."
        )

    metrics = aggregate_user_metrics(user_results)

    metrics["users_evaluated"] = float(users_evaluated)

    return metrics


def run_bpr_validation(
    factors: int = 64,
    learning_rate: float = 0.01,
    regularization: float = 0.001,
    epochs: int = 5,
    k: int = 10,
) -> dict[str, float]:
    """Train BPR on training data and evaluate on validation data."""

    ratings = preprocess_ratings(
        MovieLensLoader("data").load_ratings()
    )

    train, validation, _ = temporal_split(ratings)

    model = BPRRecommender(
        factors=factors,
        learning_rate=learning_rate,
        regularization=regularization,
        epochs=epochs,
        random_seed=42,
    )

    print("TRAINING BPR")
    print("-" * 40)

    model.fit(train)

    print()
    print("VALIDATING BPR")
    print("-" * 40)

    return evaluate_bpr_on_validation(
        model=model,
        validation=validation,
        k=k,
    )


if __name__ == "__main__":
    results = run_bpr_validation(
        factors=32,
        learning_rate=0.005,
        regularization=0.001,
        epochs=10,
        k=10,
    )

    print()
    print("BPR VALIDATION COMPLETE")
    print()

    for metric, value in results.items():
        print(f"{metric}: {value:.6f}")