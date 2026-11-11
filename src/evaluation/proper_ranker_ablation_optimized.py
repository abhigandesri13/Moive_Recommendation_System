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
    / "proper_ranker_ablation_optimized.csv"
)

CANDIDATES_PER_SOURCE = 200
K = 10
RANKING_CANDIDATES_PER_USER = 100


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
    Build all candidate-generation models once for a given
    training split.
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


def generate_source_candidates(
    users: list[int],
    bpr: BPRRecommender,
    content: ContentRecommender,
    embedding: EmbeddingRecommender,
    popularity: PopularityRecommender,
    ratings: pd.DataFrame,
) -> dict[str, dict[int, pd.DataFrame]]:
    """
    Generate each source's candidates exactly once per user.

    Returned structure:
        source -> user_id -> DataFrame
    """

    source_candidates = {
        "bpr": {},
        "content": {},
        "semantic": {},
        "popularity": {},
    }

    for position, user_id in enumerate(
        users,
        start=1,
    ):
        print(
            f"\rGenerating base candidates: "
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

        source_candidates["bpr"][user_id] = (
            bpr.recommend(
                user_id=user_id,
                n=CANDIDATES_PER_SOURCE,
            )
        )

        source_candidates["content"][user_id] = (
            content.recommend_for_user(
                user_id=user_id,
                ratings=ratings,
                n=CANDIDATES_PER_SOURCE,
            )
        )

        source_candidates["semantic"][user_id] = (
            embedding.recommend_for_user(
                user_id=user_id,
                ratings=ratings,
                n=CANDIDATES_PER_SOURCE,
            )
        )

        source_candidates["popularity"][user_id] = (
            popularity.recommend(
                n=CANDIDATES_PER_SOURCE,
                exclude_movies=already_rated,
            )
        )

    print()

    return source_candidates


def build_candidate_pool(
    user_id: int,
    source_candidates: dict[str, dict[int, pd.DataFrame]],
    enabled_sources: list[str],
    source_weights: dict[str, float],
) -> pd.DataFrame:
    """
    Combine precomputed source candidates for one user.
    """

    candidate_sources = {}

    for source_name in enabled_sources:
        candidate_sources[source_name] = (
            source_candidates[source_name][user_id]
        )

    generator = HybridCandidateGenerator(
        source_weights=source_weights,
    )

    candidates = generator.generate(
        candidate_sources=candidate_sources,
        candidates_per_source=CANDIDATES_PER_SOURCE,
    )

    if not candidates.empty:
        candidates = (
            candidates
            .sort_values(
                "hybrid_candidate_score",
                ascending=False,
            )
            .head(RANKING_CANDIDATES_PER_USER)
            .reset_index(drop=True)
        )

        candidates.insert(
            0,
            "userId",
            user_id,
        )

    return candidates


def build_ranking_data(
    users: list[int],
    source_candidates: dict[str, dict[int, pd.DataFrame]],
    enabled_sources: list[str],
    source_weights: dict[str, float],
    feature_ratings: pd.DataFrame,
    label_ratings: pd.DataFrame,
    movies: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build ranking features and labels from precomputed candidates.
    """

    feature_builder = RankingFeatureBuilder(
        positive_threshold=4.0,
    )

    label_builder = RankingLabelBuilder(
        positive_threshold=4.0,
    )

    datasets: list[pd.DataFrame] = []

    for position, user_id in enumerate(
        users,
        start=1,
    ):
        print(
            f"\rBuilding ranking data: "
            f"{position}/{len(users)}",
            end="",
            flush=True,
        )

        candidates = build_candidate_pool(
            user_id=user_id,
            source_candidates=source_candidates,
            enabled_sources=enabled_sources,
            source_weights=source_weights,
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


def evaluate_ranker(
    ranker: LightGBMRanker,
    ranking_data: pd.DataFrame,
    relevant_ratings: pd.DataFrame,
) -> dict[str, float]:
    """
    Evaluate ranking performance at K=10.
    """

    data = ranking_data.copy()

    data["ranking_score"] = (
        ranker.predict(data)
    )

    precisions: list[float] = []
    recalls: list[float] = []
    ndcgs: list[float] = []

    users_evaluated = 0
    relevant_positive_counts = (
        relevant_ratings.loc[
            relevant_ratings["rating"] >= 4.0
        ]
        .groupby("userId")
        .size()
        .to_dict()
    )

    for user_id, group in data.groupby(
        "userId",
        sort=False,
    ):
        positives = int(
            relevant_positive_counts.get(
                user_id,
                0,
            )
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
        recall = hits / positives

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


def train_ranker(
    ranking_data: pd.DataFrame,
) -> LightGBMRanker:
    """
    Train LightGBM on one ranking dataset.
    """

    groups = (
        ranking_data
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
        features=ranking_data,
        groups=groups,
    )

    return ranker


def run_configuration(
    configuration_name: str,
    enabled_sources: list[str],
    source_weights: dict[str, float],
    inner_users: list[int],
    outer_users: list[int],
    inner_source_candidates: dict[
        str,
        dict[int, pd.DataFrame],
    ],
    outer_source_candidates: dict[
        str,
        dict[int, pd.DataFrame],
    ],
    inner_train: pd.DataFrame,
    inner_holdout: pd.DataFrame,
    outer_train: pd.DataFrame,
    outer_validation: pd.DataFrame,
    movies: pd.DataFrame,
) -> dict[str, float]:
    """
    Train on inner data and evaluate on outer validation.
    """

    print()
    print("=" * 60)
    print(
        f"CONFIGURATION: "
        f"{configuration_name}"
    )
    print("=" * 60)

    print("\nBuilding inner ranking dataset...")

    inner_data = build_ranking_data(
        users=inner_users,
        source_candidates=inner_source_candidates,
        enabled_sources=enabled_sources,
        source_weights=source_weights,
        feature_ratings=inner_train,
        label_ratings=inner_holdout,
        movies=movies,
    )

    print("\nBuilding outer validation dataset...")

    outer_data = build_ranking_data(
        users=outer_users,
        source_candidates=outer_source_candidates,
        enabled_sources=enabled_sources,
        source_weights=source_weights,
        feature_ratings=outer_train,
        label_ratings=outer_validation,
        movies=movies,
    )

    print()
    print(
        f"Inner rows: "
        f"{len(inner_data)}"
    )

    print(
        f"Outer rows: "
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

    print("\nTraining LightGBM...")

    ranker = train_ranker(
        inner_data
    )
    importance = ranker.feature_importance()

    print("\nTop ranking features:")
    print(
        importance.head(10).to_string(
            index=False
        )
    )

    metrics = evaluate_ranker(
        ranker=ranker,
        ranking_data=outer_data,
        relevant_ratings=outer_validation,
    )

    print(
        f"\nPrecision@10: "
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

    return {
        "configuration": configuration_name,
        "model": "LightGBM_Ranker",
        "precision@10": metrics[
            "precision@10"
        ],
        "recall@10": metrics[
            "recall@10"
        ],
        "ndcg@10": metrics[
            "ndcg@10"
        ],
        "users_evaluated": metrics[
            "users_evaluated"
        ],
        "inner_rows": len(inner_data),
        "outer_rows": len(outer_data),
    }


def run_experiment() -> None:
    """
    Run optimized, leakage-safe ranker ablation.

    Models are trained once per temporal split and
    candidate lists are generated once per user.
    """

    start_time = time.perf_counter()

    loader = MovieLensLoader("data")

    ratings = preprocess_ratings(
        loader.load_ratings()
    )

    movies = loader.load_movies()
    tags = loader.load_tags()

    outer_train, outer_validation, outer_test = (
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

    # --------------------------------------------------
    # Build INNER models once
    # --------------------------------------------------

    print()
    print("=" * 60)
    print("BUILDING INNER MODELS ONCE")
    print("=" * 60)

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

    print("\nGenerating inner candidates once...")

    inner_candidates = generate_source_candidates(
        users=inner_users,
        bpr=inner_bpr,
        content=inner_content,
        embedding=inner_embedding,
        popularity=inner_popularity,
        ratings=inner_train,
    )

    # --------------------------------------------------
    # Build OUTER models once
    # --------------------------------------------------

    print()
    print("=" * 60)
    print("BUILDING OUTER MODELS ONCE")
    print("=" * 60)

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

    print("\nGenerating outer candidates once...")

    outer_candidates = generate_source_candidates(
        users=outer_users,
        bpr=outer_bpr,
        content=outer_content,
        embedding=outer_embedding,
        popularity=outer_popularity,
        ratings=outer_train,
    )

    # --------------------------------------------------
    # Ablation configurations
    # --------------------------------------------------

    configurations = [
        (
            "BPR",
            ["bpr"],
            {
                "bpr": 1.0,
            },
        ),
        (
            "BPR_Content",
            [
                "bpr",
                "content",
            ],
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

    # --------------------------------------------------
    # Run all configurations using reused candidates
    # --------------------------------------------------

    for (
        configuration_name,
        enabled_sources,
        source_weights,
    ) in configurations:

        result = run_configuration(
            configuration_name=configuration_name,
            enabled_sources=enabled_sources,
            source_weights=source_weights,
            inner_users=inner_users,
            outer_users=outer_users,
            inner_source_candidates=inner_candidates,
            outer_source_candidates=outer_candidates,
            inner_train=inner_train,
            inner_holdout=inner_holdout,
            outer_train=outer_train,
            outer_validation=outer_validation,
            movies=movies,
        )

        results.append(result)

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
    print(
        "OPTIMIZED PROPER RANKER "
        "ABLATION COMPLETE"
    )
    print("=" * 60)

    for result in results:
        print(
            f"{result['configuration']}: "
            f"P@10={result['precision@10']:.6f}, "
            f"R@10={result['recall@10']:.6f}, "
            f"NDCG@10={result['ndcg@10']:.6f}"
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
    run_experiment()