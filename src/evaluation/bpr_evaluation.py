from src.data.loaders import MovieLensLoader
from src.data.preprocessing.rating_preprocessor import preprocess_ratings
from src.data.preprocessing.splitter import temporal_split
from src.recommenders.collaborative.bpr import BPRRecommender
from src.evaluation.ranking_metrics import (
    aggregate_user_metrics,
    catalog_coverage,
    evaluate_user,
)


def evaluate_bpr(
    model: BPRRecommender,
    test,
    k: int = 10,
) -> dict[str, float]:
    """Evaluate a trained BPR model on positive test interactions."""

    if k <= 0:
        raise ValueError("k must be greater than zero.")

    user_results = []
    all_recommended_items = set()
    users_evaluated = 0

    for user_id in sorted(test["userId"].unique()):

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

        if recommendations.empty:
            continue

        recommended_items = (
            recommendations["movieId"]
            .astype(int)
            .tolist()
        )

        all_recommended_items.update(recommended_items)

        user_metrics = evaluate_user(
            recommended_items=recommended_items,
            relevant_items=relevant_items,
            k=k,
        )

        user_results.append(user_metrics)
        users_evaluated += 1

    if not user_results:
        raise RuntimeError("No users could be evaluated.")

    metrics = aggregate_user_metrics(user_results)

    metrics["catalog_coverage"] = catalog_coverage(
        all_recommended_items,
        model.item_to_index.__len__(),
    )

    metrics["users_evaluated"] = float(users_evaluated)
    metrics["catalog_size"] = float(
        model.item_to_index.__len__()
    )

    return metrics


def run_bpr_evaluation() -> dict[str, float]:
    """Train BPR and evaluate it on the temporal test split."""

    ratings = preprocess_ratings(
        MovieLensLoader("data").load_ratings()
    )

    train, _, test = temporal_split(ratings)

    model = BPRRecommender(
        factors=64,
        learning_rate=0.01,
        regularization=0.001,
        epochs=5,
        random_seed=42,
    )

    print("TRAINING BPR")
    print("-" * 40)

    model.fit(train)

    print()
    print("EVALUATING BPR")
    print("-" * 40)

    return evaluate_bpr(
        model=model,
        test=test,
        k=10,
    )


if __name__ == "__main__":
    results = run_bpr_evaluation()

    print()
    print("BPR EVALUATION COMPLETE")
    print()

    for metric, value in results.items():
        print(f"{metric}: {value:.6f}")