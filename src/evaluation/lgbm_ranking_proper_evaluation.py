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
    / "lgbm_ranking_proper_evaluation.csv"
)

CANDIDATES_PER_SOURCE = 100


def build_candidate_models(
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
    Build all candidate generators using only the supplied
    training ratings.
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


def build_hybrid(
    train: pd.DataFrame,
    movies: pd.DataFrame,
    tags: pd.DataFrame,
) -> HybridRecommender:
    """
    Build the complete hybrid candidate generator.
    """

    (
        bpr,
        content,
        embedding,
        popularity,
    ) = build_candidate_models(
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

    return HybridRecommender(
        bpr_model=bpr,
        content_model=content,
        embedding_model=embedding,
        popularity_model=popularity,
        ratings=train,
        source_weights=source_weights,
    )


def build_ranking_dataset(
    users: list[int],
    hybrid: HybridRecommender,
    feature_builder: RankingFeatureBuilder,
    label_builder: RankingLabelBuilder,
    feature_ratings: pd.DataFrame,
    label_ratings: pd.DataFrame,
    movies: pd.DataFrame,
) -> pd.DataFrame:
    """
    Generate candidates from one rating period, construct features
    using that same available history, and label candidates using
    a later holdout period.
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
            candidates_per_source=CANDIDATES_PER_SOURCE,
        )

        if candidates.empty:
            continue

        features = feature_builder.build(
            candidates=candidates,
            ratings=feature_ratings,
            movies=movies,
        )

        labeled = label_builder.build(
            candidates=features,
            labels_ratings=label_ratings,
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


def evaluate_ranking(
    ranker: LightGBMRanker,
    ranking_data: pd.DataFrame,
) -> dict[str, float]:
    """
    Evaluate ranking quality per user.

    Each user is treated as one ranking query.
    """

    scored = ranking_data.copy()

    scored["ranking_score"] = (
        ranker.predict(scored)
    )

    k = 10

    precisions: list[float] = []
    recalls: list[float] = []
    ndcgs: list[float] = []

    users_evaluated = 0

    import math

    for _, group in scored.groupby(
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
                dcg += (
                    1.0
                    / math.log2(rank + 1)
                )

        ideal_hits = min(
            positives,
            k,
        )

        idcg = sum(
            1.0 / math.log2(rank + 1)
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
        "precision@10": (
            sum(precisions)
            / users_evaluated
        ),
        "recall@10": (
            sum(recalls)
            / users_evaluated
        ),
        "ndcg@10": (
            sum(ndcgs)
            / users_evaluated
        ),
        "users_evaluated": users_evaluated,
    }


def run_experiment() -> None:
    """
    Run a leakage-safe LightGBM ranking experiment.

    Inner split:
        Used to train LightGBM.

    Outer validation:
        Used only for final validation metrics.
    """

    start_time = time.perf_counter()

    loader = MovieLensLoader("data")

    ratings = preprocess_ratings(
        loader.load_ratings()
    )

    movies = loader.load_movies()
    tags = loader.load_tags()

    # --------------------------------------------------
    # Outer temporal split
    # --------------------------------------------------

    outer_train, outer_validation, outer_test = (
        temporal_split(ratings)
    )

    print()
    print("=" * 60)
    print("OUTER TEMPORAL SPLIT")
    print("=" * 60)

    print(
        f"Outer train rows: "
        f"{len(outer_train)}"
    )

    print(
        f"Outer validation rows: "
        f"{len(outer_validation)}"
    )

    print(
        f"Outer test rows: "
        f"{len(outer_test)}"
    )

    # --------------------------------------------------
    # Inner split
    # --------------------------------------------------

    inner_train, inner_holdout, _ = temporal_split(
        outer_train
    )

    print()
    print("=" * 60)
    print("INNER TEMPORAL SPLIT")
    print("=" * 60)

    print(
        f"Inner train rows: "
        f"{len(inner_train)}"
    )

    print(
        f"Inner holdout rows: "
        f"{len(inner_holdout)}"
    )

    # --------------------------------------------------
    # Build models for ranker training
    #
    # IMPORTANT:
    # Everything here uses only inner_train.
    # --------------------------------------------------

    print()
    print("=" * 60)
    print("PHASE 1: TRAIN RANKER")
    print("=" * 60)

    inner_hybrid = build_hybrid(
        train=inner_train,
        movies=movies,
        tags=tags,
    )

    feature_builder = RankingFeatureBuilder(
        positive_threshold=4.0
    )

    label_builder = RankingLabelBuilder(
        positive_threshold=4.0
    )

    inner_users = sorted(
        inner_train["userId"].unique()
    )

    inner_ranking_data = build_ranking_dataset(
        users=inner_users,
        hybrid=inner_hybrid,
        feature_builder=feature_builder,
        label_builder=label_builder,
        feature_ratings=inner_train,
        label_ratings=inner_holdout,
        movies=movies,
    )

    print()
    print("=" * 60)
    print("INNER RANKING DATASET")
    print("=" * 60)

    print(
        f"Rows: "
        f"{len(inner_ranking_data)}"
    )

    print(
        f"Users: "
        f"{inner_ranking_data['userId'].nunique()}"
    )

    print(
        f"Positive labels: "
        f"{int(inner_ranking_data['label'].sum())}"
    )

    print(
        f"Negative labels: "
        f"{int((inner_ranking_data['label'] == 0).sum())}"
    )

    inner_groups = (
        inner_ranking_data
        .groupby(
            "userId",
            sort=False,
        )
        .size()
        .tolist()
    )

    ranker = LightGBMRanker(
        n_estimators=200,
        learning_rate=0.05,
        num_leaves=31,
        random_seed=42,
    )

    ranker.fit(
        features=inner_ranking_data,
        groups=inner_groups,
    )

    print("\nRanker training complete.")

    # --------------------------------------------------
    # Build fresh outer-validation candidates
    #
    # IMPORTANT:
    # Retrieval models now use outer_train only.
    # --------------------------------------------------

    print()
    print("=" * 60)
    print("PHASE 2: OUTER VALIDATION")
    print("=" * 60)

    outer_hybrid = build_hybrid(
        train=outer_train,
        movies=movies,
        tags=tags,
    )

    outer_users = sorted(
        outer_train["userId"].unique()
    )

    outer_ranking_data = build_ranking_dataset(
        users=outer_users,
        hybrid=outer_hybrid,
        feature_builder=feature_builder,
        label_builder=label_builder,
        feature_ratings=outer_train,
        label_ratings=outer_validation,
        movies=movies,
    )

    print()
    print("=" * 60)
    print("OUTER VALIDATION DATASET")
    print("=" * 60)

    print(
        f"Rows: "
        f"{len(outer_ranking_data)}"
    )

    print(
        f"Users: "
        f"{outer_ranking_data['userId'].nunique()}"
    )

    print(
        f"Positive labels: "
        f"{int(outer_ranking_data['label'].sum())}"
    )

    print(
        f"Negative labels: "
        f"{int((outer_ranking_data['label'] == 0).sum())}"
    )

    # --------------------------------------------------
    # Final validation
    # --------------------------------------------------

    metrics = evaluate_ranking(
        ranker=ranker,
        ranking_data=outer_ranking_data,
    )

    print()
    print("=" * 60)
    print("PROPER LIGHTGBM VALIDATION RESULT")
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
        - start_time
    )

    # --------------------------------------------------
    # Save result
    # --------------------------------------------------

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    result_row = {
        "inner_train_rows": len(
            inner_train
        ),
        "inner_holdout_rows": len(
            inner_holdout
        ),
        "outer_train_rows": len(
            outer_train
        ),
        "outer_validation_rows": len(
            outer_validation
        ),
        "outer_test_rows": len(
            outer_test
        ),
        "inner_ranking_rows": len(
            inner_ranking_data
        ),
        "outer_ranking_rows": len(
            outer_ranking_data
        ),
        "positive_labels_inner": int(
            inner_ranking_data["label"].sum()
        ),
        "positive_labels_outer": int(
            outer_ranking_data["label"].sum()
        ),
        "users_evaluated": metrics[
            "users_evaluated"
        ],
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