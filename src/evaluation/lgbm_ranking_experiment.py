from __future__ import annotations

import csv
import time
from pathlib import Path

import pandas as pd

from src.data.loaders import MovieLensLoader
from src.data.preprocessing.rating_preprocessor import preprocess_ratings
from src.data.preprocessing.splitter import temporal_split

from src.recommenders.collaborative.bpr import BPRRecommender
from src.recommenders.semantic.content_recommender import (
    ContentRecommender,
)
from src.recommenders.semantic.embedding_recommender import (
    EmbeddingRecommender,
)
from src.recommenders.popularity.popularity_recommender import (
    PopularityRecommender,
)
from src.recommenders.hybrid.hybrid_recommender import (
    HybridRecommender,
)

from src.ranking.feature_builder import (
    RankingFeatureBuilder,
)
from src.ranking.label_builder import (
    RankingLabelBuilder,
)
from src.ranking.lgbm_ranker import (
    LightGBMRanker,
)

RESULTS_DIR = Path("experiments") / "results"
RESULTS_FILE = (
    RESULTS_DIR
    / "lgbm_ranking_experiments.csv"
)

CANDIDATES_PER_SOURCE = 100
RANKER_FEATURE_COLUMNS = (
    LightGBMRanker.DEFAULT_FEATURE_COLUMNS
)


def build_models(
    train: pd.DataFrame,
    movies: pd.DataFrame,
    tags: pd.DataFrame,
) -> tuple[
    BPRRecommender,
    ContentRecommender,
    EmbeddingRecommender,
    PopularityRecommender,
]:
    """
    Train/build all candidate-generation models.
    """

    print()
    print("=" * 60)
    print("BUILDING CANDIDATE GENERATORS")
    print("=" * 60)

    print("\nTraining BPR...")

    bpr = BPRRecommender(
        factors=32,
        learning_rate=0.005,
        regularization=0.001,
        epochs=10,
        random_seed=42,
    )

    bpr.fit(train)

    print("\nBuilding TF-IDF content model...")

    content = ContentRecommender()
    content.fit(
        movies,
        tags,
    )

    print("\nBuilding semantic embedding model...")

    embedding = EmbeddingRecommender()
    embedding.fit(
        movies,
        tags,
    )

    print("\nBuilding popularity model...")

    popularity = PopularityRecommender(
        minimum_rating_count=10,
    )

    popularity.fit(
        movies,
        train,
    )

    return (
        bpr,
        content,
        embedding,
        popularity,
    )


def build_ranking_dataset(
    users: list[int],
    hybrid: HybridRecommender,
    feature_builder: RankingFeatureBuilder,
    label_builder: RankingLabelBuilder,
    train: pd.DataFrame,
    validation: pd.DataFrame,
    movies: pd.DataFrame,
) -> pd.DataFrame:
    """
    Generate candidate rows, features, and validation labels
    for multiple users.
    """

    datasets: list[pd.DataFrame] = []

    total_users = len(users)

    for position, user_id in enumerate(
        users,
        start=1,
    ):
        print(
            f"\rBuilding ranking data: "
            f"{position}/{total_users}",
            end="",
            flush=True,
        )

        candidates = hybrid.generate_candidates(
            user_id=user_id,
            candidates_per_source=(
                CANDIDATES_PER_SOURCE
            ),
        )

        if candidates.empty:
            continue

        features = feature_builder.build(
            candidates=candidates,
            ratings=train,
            movies=movies,
        )

        labeled = label_builder.build(
            candidates=features,
            labels_ratings=validation,
        )

        if not labeled.empty:
            datasets.append(labeled)

    print()

    if not datasets:
        raise RuntimeError(
            "No ranking data was generated."
        )

    return pd.concat(
        datasets,
        ignore_index=True,
    )


def evaluate_ranker(
    ranker: LightGBMRanker,
    ranking_data: pd.DataFrame,
) -> dict[str, float]:
    """
    Evaluate ranking quality on the generated validation dataset.

    This measures how high validation-positive movies are ranked
    within each user's candidate group.
    """

    required_columns = {
        "userId",
        "movieId",
        "label",
    }

    missing = (
        required_columns
        - set(ranking_data.columns)
    )

    if missing:
        raise ValueError(
            "ranking_data: missing required columns: "
            f"{sorted(missing)}"
        )

    scored = ranking_data.copy()

    scored["ranking_score"] = (
        ranker.predict(scored)
    )

    precisions: list[float] = []
    recalls: list[float] = []
    ndcgs: list[float] = []

    users_evaluated = 0

    k = 10

    for user_id, group in scored.groupby(
        "userId",
        sort=False,
    ):
        positives = int(
            group["label"].sum()
        )

        if positives == 0:
            continue

        ranked = group.sort_values(
            "ranking_score",
            ascending=False,
        )

        top_k = ranked.head(k)

        hits = int(
            top_k["label"].sum()
        )

        precision = hits / k

        recall = (
            hits / positives
            if positives > 0
            else 0.0
        )

        dcg = 0.0

        for rank, label in enumerate(
            top_k["label"].tolist(),
            start=1,
        ):
            if label:
                dcg += 1.0 / (
                    __import__("math").log2(
                        rank + 1
                    )
                )

        ideal_hits = min(
            positives,
            k,
        )

        idcg = sum(
            1.0 / __import__("math").log2(
                rank + 1
            )
            for rank in range(
                1,
                ideal_hits + 1,
            )
        )

        ndcg = (
            dcg / idcg
            if idcg > 0
            else 0.0
        )

        precisions.append(
            precision
        )
        recalls.append(
            recall
        )
        ndcgs.append(
            ndcg
        )

        users_evaluated += 1

    if users_evaluated == 0:
        return {
            "precision@10": 0.0,
            "recall@10": 0.0,
            "ndcg@10": 0.0,
            "users_evaluated": 0,
        }

    return {
        "precision@10": sum(precisions)
        / users_evaluated,
        "recall@10": sum(recalls)
        / users_evaluated,
        "ndcg@10": sum(ndcgs)
        / users_evaluated,
        "users_evaluated": users_evaluated,
    }


def run_experiment() -> None:
    """
    Run the first multi-user LightGBM ranking experiment.
    """

    start_total = time.perf_counter()

    loader = MovieLensLoader("data")

    ratings = preprocess_ratings(
        loader.load_ratings()
    )

    movies = loader.load_movies()
    tags = loader.load_tags()

    train, validation, _ = temporal_split(
        ratings
    )

    (
        bpr,
        content,
        embedding,
        popularity,
    ) = build_models(
        train=train,
        movies=movies,
        tags=tags,
    )

    source_weights = {
        "bpr": 0.45,
        "content": 0.20,
        "semantic": 0.25,
        "popularity": 0.10,
    }

    hybrid = HybridRecommender(
        bpr_model=bpr,
        content_model=content,
        embedding_model=embedding,
        popularity_model=popularity,
        ratings=train,
        source_weights=source_weights,
    )

    feature_builder = RankingFeatureBuilder(
        positive_threshold=4.0,
    )

    label_builder = RankingLabelBuilder(
        positive_threshold=4.0,
    )

    users = sorted(
        train["userId"].unique()
    )

    ranking_data = build_ranking_dataset(
        users=users,
        hybrid=hybrid,
        feature_builder=feature_builder,
        label_builder=label_builder,
        train=train,
        validation=validation,
        movies=movies,
    )

    print()
    print("=" * 60)
    print("RANKING DATASET SUMMARY")
    print("=" * 60)

    print(
        f"Rows: {len(ranking_data)}"
    )

    print(
        f"Users: "
        f"{ranking_data['userId'].nunique()}"
    )

    print(
        f"Positive labels: "
        f"{int(ranking_data['label'].sum())}"
    )

    print(
        f"Negative labels: "
        f"{int((ranking_data['label'] == 0).sum())}"
    )

    groups = (
        ranking_data
        .groupby(
            "userId",
            sort=False,
        )
        .size()
        .tolist()
    )

    print()
    print("=" * 60)
    print("TRAINING LIGHTGBM RANKER")
    print("=" * 60)

    ranker = LightGBMRanker(
        n_estimators=200,
        learning_rate=0.05,
        num_leaves=31,
        random_seed=42,
    )

    ranker.fit(
        features=ranking_data,
        groups=groups,
    )

    metrics = evaluate_ranker(
        ranker=ranker,
        ranking_data=ranking_data,
    )

    print()
    print("=" * 60)
    print("LIGHTGBM RANKING RESULT")
    print("=" * 60)

    for key, value in metrics.items():
        if isinstance(value, float):
            print(
                f"{key}: {value:.6f}"
            )
        else:
            print(
                f"{key}: {value}"
            )

    print()
    print("=" * 60)
    print("FEATURE IMPORTANCE")
    print("=" * 60)

    print(
        ranker
        .feature_importance()
        .to_string(index=False)
    )

    elapsed = (
        time.perf_counter()
        - start_total
    )

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    result_row = {
        "n_estimators": 200,
        "learning_rate": 0.05,
        "num_leaves": 31,
        "candidates_per_source": (
            CANDIDATES_PER_SOURCE
        ),
        "users": metrics[
            "users_evaluated"
        ],
        "rows": len(ranking_data),
        "positive_labels": int(
            ranking_data["label"].sum()
        ),
        "precision@10": metrics[
            "precision@10"
        ],
        "recall@10": metrics[
            "recall@10"
        ],
        "ndcg@10": metrics[
            "ndcg@10"
        ],
        "total_time_seconds": elapsed,
    }

    with RESULTS_FILE.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(
                result_row.keys()
            ),
        )

        writer.writeheader()
        writer.writerow(
            result_row
        )

    print()
    print(
        f"Results saved to: "
        f"{RESULTS_FILE}"
    )

    print(
        f"Total experiment time: "
        f"{elapsed:.2f} seconds"
    )


if __name__ == "__main__":
    run_experiment()