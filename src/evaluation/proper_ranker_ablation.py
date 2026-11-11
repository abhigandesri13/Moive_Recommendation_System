from __future__ import annotations

import csv
import math
import time
from pathlib import Path

import pandas as pd

from src.data.loaders import MovieLensLoader
from src.data.preprocessing.rating_preprocessor import (
    preprocess_ratings,
)
from src.data.preprocessing.splitter import temporal_split

from src.recommenders.collaborative.bpr import (
    BPRRecommender,
)
from src.recommenders.semantic.content_recommender import (
    ContentRecommender,
)
from src.recommenders.semantic.embedding_recommender import (
    EmbeddingRecommender,
)
from src.recommenders.popularity.popularity_recommender import (
    PopularityRecommender,
)
from src.recommenders.hybrid.candidate_generator import (
    HybridCandidateGenerator,
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
    / "proper_ranker_ablation.csv"
)

CANDIDATES_PER_SOURCE = 100
K = 10


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
    Build candidate-generation models using only train data.
    """

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


def generate_candidates_for_users(
    users: list[int],
    bpr: BPRRecommender,
    content: ContentRecommender,
    embedding: EmbeddingRecommender,
    popularity: PopularityRecommender,
    ratings: pd.DataFrame,
    source_weights: dict[str, float],
    enabled_sources: list[str],
) -> pd.DataFrame:
    """
    Generate weighted hybrid candidates for multiple users.
    """

    generator = HybridCandidateGenerator(
        source_weights=source_weights
    )

    all_rows: list[pd.DataFrame] = []

    for position, user_id in enumerate(
        users,
        start=1,
    ):
        print(
            f"\rGenerating candidates: "
            f"{position}/{len(users)}",
            end="",
            flush=True,
        )

        user_ratings = ratings[
            ratings["userId"] == user_id
        ]

        already_rated = set(
            user_ratings["movieId"].astype(int)
        )

        candidate_sources = {}

        if "bpr" in enabled_sources:
            candidate_sources["bpr"] = (
                bpr.recommend(
                    user_id=user_id,
                    n=CANDIDATES_PER_SOURCE,
                )
            )

        if "content" in enabled_sources:
            candidate_sources["content"] = (
                content.recommend_for_user(
                    user_id=user_id,
                    ratings=ratings,
                    n=CANDIDATES_PER_SOURCE,
                )
            )

        if "semantic" in enabled_sources:
            candidate_sources["semantic"] = (
                embedding.recommend_for_user(
                    user_id=user_id,
                    ratings=ratings,
                    n=CANDIDATES_PER_SOURCE,
                )
            )

        if "popularity" in enabled_sources:
            candidate_sources["popularity"] = (
                popularity.recommend(
                    n=CANDIDATES_PER_SOURCE,
                    exclude_movies=already_rated,
                )
            )

        if not candidate_sources:
            continue

        candidates = generator.generate(
            candidate_sources=candidate_sources,
            candidates_per_source=CANDIDATES_PER_SOURCE,
        )

        if candidates.empty:
            continue

        candidates.insert(
            0,
            "userId",
            user_id,
        )

        all_rows.append(
            candidates
        )

    print()

    if not all_rows:
        raise RuntimeError(
            "No candidates were generated."
        )

    return pd.concat(
        all_rows,
        ignore_index=True,
    )


def build_ranking_data(
    candidates: pd.DataFrame,
    feature_ratings: pd.DataFrame,
    label_ratings: pd.DataFrame,
    movies: pd.DataFrame,
) -> pd.DataFrame:
    """
    Convert candidates into ranking features and labels.
    """

    feature_builder = RankingFeatureBuilder(
        positive_threshold=4.0
    )

    label_builder = RankingLabelBuilder(
        positive_threshold=4.0
    )

    features = feature_builder.build(
        candidates=candidates,
        ratings=feature_ratings,
        movies=movies,
    )

    return label_builder.build(
        candidates=features,
        labels_ratings=label_ratings,
    )


def evaluate_ranked_data(
    ranking_data: pd.DataFrame,
    scores: pd.Series,
) -> dict[str, float]:
    """
    Evaluate Precision@10, Recall@10 and NDCG@10.
    """

    data = ranking_data.copy()

    data["ranking_score"] = scores

    precisions: list[float] = []
    recalls: list[float] = []
    ndcgs: list[float] = []

    users_evaluated = 0

    for _, group in data.groupby(
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

        top_k = ranked.head(K)

        hits = int(
            top_k["label"].sum()
        )

        precision = hits / K

        recall = (
            hits / positives
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
            K,
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


def train_and_evaluate_ranker(
    train_data: pd.DataFrame,
    validation_data: pd.DataFrame,
) -> dict[str, float]:
    """
    Train LightGBM on train_data and evaluate it on
    independent validation_data.
    """

    groups = (
        train_data
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
        features=train_data,
        groups=groups,
    )

    validation_scores = (
        ranker.predict(
            validation_data
        )
    )

    return evaluate_ranked_data(
        ranking_data=validation_data,
        scores=validation_scores,
    )


def run_ablation() -> None:
    """
    Proper ranker ablation using an inner/outer temporal design.

    Ranker training:
        Inner train -> Inner holdout labels

    Ranker evaluation:
        Outer train retrieval -> Outer validation labels
    """

    start_time = time.perf_counter()

    loader = MovieLensLoader("data")

    ratings = preprocess_ratings(
        loader.load_ratings()
    )

    movies = loader.load_movies()
    tags = loader.load_tags()

    outer_train, outer_validation, _ = (
        temporal_split(ratings)
    )

    inner_train, inner_holdout, _ = (
        temporal_split(outer_train)
    )

    print()
    print("=" * 60)
    print("DATA SPLITS")
    print("=" * 60)

    print(
        f"Inner train: {len(inner_train)}"
    )

    print(
        f"Inner holdout: {len(inner_holdout)}"
    )

    print(
        f"Outer train: {len(outer_train)}"
    )

    print(
        f"Outer validation: "
        f"{len(outer_validation)}"
    )

    configurations = [
        (
            "BPR",
            ["bpr"],
            {"bpr": 1.0},
        ),
        (
            "BPR_Content",
            ["bpr", "content"],
            {
                "bpr": 0.7,
                "content": 0.3,
            },
        ),
        (
            "BPR_Content_Semantic",
            [
                "bpr",
                "content",
                "semantic",
            ],
            {
                "bpr": 0.55,
                "content": 0.20,
                "semantic": 0.25,
            },
        ),
        (
            "Full_Hybrid",
            [
                "bpr",
                "content",
                "semantic",
                "popularity",
            ],
            {
                "bpr": 0.45,
                "content": 0.20,
                "semantic": 0.25,
                "popularity": 0.10,
            },
        ),
    ]

    results: list[dict] = []

    for (
        configuration_name,
        enabled_sources,
        source_weights,
    ) in configurations:

        print()
        print("=" * 60)
        print(
            f"CONFIGURATION: "
            f"{configuration_name}"
        )
        print("=" * 60)

        # --------------------------------------------------
        # Phase A: Inner training data
        # --------------------------------------------------

        print(
            "\nBuilding inner candidate models..."
        )

        (
            inner_bpr,
            inner_content,
            inner_embedding,
            inner_popularity,
        ) = build_models(
            train=inner_train,
            movies=movies,
            tags=tags,
        )

        inner_users = sorted(
            inner_train["userId"].unique()
        )

        print(
            "\nGenerating inner candidates..."
        )

        inner_candidates = (
            generate_candidates_for_users(
                users=inner_users,
                bpr=inner_bpr,
                content=inner_content,
                embedding=inner_embedding,
                popularity=inner_popularity,
                ratings=inner_train,
                source_weights=source_weights,
                enabled_sources=enabled_sources,
            )
        )

        inner_data = build_ranking_data(
            candidates=inner_candidates,
            feature_ratings=inner_train,
            label_ratings=inner_holdout,
            movies=movies,
        )

        # --------------------------------------------------
        # Phase B: Outer validation data
        # --------------------------------------------------

        print(
            "\nBuilding outer candidate models..."
        )

        (
            outer_bpr,
            outer_content,
            outer_embedding,
            outer_popularity,
        ) = build_models(
            train=outer_train,
            movies=movies,
            tags=tags,
        )

        outer_users = sorted(
            outer_train["userId"].unique()
        )

        print(
            "\nGenerating outer candidates..."
        )

        outer_candidates = (
            generate_candidates_for_users(
                users=outer_users,
                bpr=outer_bpr,
                content=outer_content,
                embedding=outer_embedding,
                popularity=outer_popularity,
                ratings=outer_train,
                source_weights=source_weights,
                enabled_sources=enabled_sources,
            )
        )

        outer_data = build_ranking_data(
            candidates=outer_candidates,
            feature_ratings=outer_train,
            label_ratings=outer_validation,
            movies=movies,
        )

        print()
        print(
            f"Inner ranking rows: "
            f"{len(inner_data)}"
        )

        print(
            f"Outer ranking rows: "
            f"{len(outer_data)}"
        )

        print(
            f"Inner positives: "
            f"{int(inner_data['label'].sum())}"
        )

        print(
            f"Outer positives: "
            f"{int(outer_data['label'].sum())}"
        )

        metrics = train_and_evaluate_ranker(
            train_data=inner_data,
            validation_data=outer_data,
        )

        print()
        print(
            f"Precision@10: "
            f"{metrics['precision@10']:.6f}"
        )

        print(
            f"Recall@10: "
            f"{metrics['recall@10']:.6f}"
        )

        print(
            f"NDCG@10: "
            f"{metrics['ndcg@10']:.6f}"
        )

        results.append(
            {
                "configuration": (
                    configuration_name
                ),
                "model": "LightGBM_Ranker",
                "precision@10": (
                    metrics["precision@10"]
                ),
                "recall@10": (
                    metrics["recall@10"]
                ),
                "ndcg@10": (
                    metrics["ndcg@10"]
                ),
                "users_evaluated": (
                    metrics[
                        "users_evaluated"
                    ]
                ),
                "inner_rows": len(
                    inner_data
                ),
                "outer_rows": len(
                    outer_data
                ),
            }
        )

    # --------------------------------------------------
    # Save results
    # --------------------------------------------------

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    with RESULTS_FILE.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(
                results[0].keys()
            ),
        )

        writer.writeheader()
        writer.writerows(results)

    elapsed = (
        time.perf_counter()
        - start_time
    )

    print()
    print("=" * 60)
    print("PROPER RANKER ABLATION COMPLETE")
    print("=" * 60)

    for row in results:
        print()
        print(
            f"{row['configuration']}: "
            f"P@10={row['precision@10']:.6f}, "
            f"R@10={row['recall@10']:.6f}, "
            f"NDCG@10={row['ndcg@10']:.6f}"
        )

    print()
    print(
        f"Results saved to: "
        f"{RESULTS_FILE}"
    )

    print(
        f"Total time: "
        f"{elapsed:.2f} seconds"
    )


if __name__ == "__main__":
    run_ablation()