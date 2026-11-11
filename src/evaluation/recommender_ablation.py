from __future__ import annotations

import csv
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
from src.recommenders.hybrid.hybrid_recommender import (
    HybridRecommender,
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
    RESULTS_DIR / "recommender_ablation.csv"
)

CANDIDATES_PER_SOURCE = 100

K = 10


def evaluate_candidates(
    candidate_data: pd.DataFrame,
) -> dict[str, float]:
    """
    Evaluate a candidate ranking directly using the
    candidate score supplied by the configuration.
    """

    precisions: list[float] = []
    recalls: list[float] = []
    ndcgs: list[float] = []

    users_evaluated = 0

    import math

    for _, group in candidate_data.groupby(
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

        recall = hits / positives

        dcg = 0.0

        for rank, label in enumerate(
            top_k["label"].tolist(),
            start=1,
        ):
            if label:
                dcg += 1.0 / math.log2(
                    rank + 1
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
    Build all candidate-generation models.
    """

    bpr = BPRRecommender(
        factors=32,
        learning_rate=0.005,
        regularization=0.001,
        epochs=10,
        random_seed=42,
    )

    bpr.fit(train)

    content = ContentRecommender()
    content.fit(
        movies,
        tags,
    )

    embedding = EmbeddingRecommender()
    embedding.fit(
        movies,
        tags,
    )

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


def generate_base_candidates(
    users: list[int],
    bpr: BPRRecommender,
    content: ContentRecommender,
    embedding: EmbeddingRecommender,
    popularity: PopularityRecommender,
    ratings: pd.DataFrame,
) -> dict[str, list[pd.DataFrame]]:
    """
    Generate reusable candidate sources for all users.

    This avoids retraining models between ablation settings.
    """

    results = {
        "bpr": [],
        "content": [],
        "semantic": [],
        "popularity": [],
    }

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

        results["bpr"].append(
            bpr.recommend(
                user_id=user_id,
                n=CANDIDATES_PER_SOURCE,
            )
        )

        results["content"].append(
            content.recommend_for_user(
                user_id=user_id,
                ratings=ratings,
                n=CANDIDATES_PER_SOURCE,
            )
        )

        results["semantic"].append(
            embedding.recommend_for_user(
                user_id=user_id,
                ratings=ratings,
                n=CANDIDATES_PER_SOURCE,
            )
        )

        results["popularity"].append(
            popularity.recommend(
                n=CANDIDATES_PER_SOURCE,
                exclude_movies=already_rated,
            )
        )

    print()

    return results


def build_configuration_candidates(
    user_id: int,
    index: int,
    base_candidates: dict[str, list[pd.DataFrame]],
    configuration: str,
    source_weights: dict[str, float],
) -> pd.DataFrame:
    """
    Reconstruct a candidate pool for one ablation configuration.
    """

    candidate_sources = {}

    if "bpr" in configuration:
        candidate_sources["bpr"] = (
            base_candidates["bpr"][index]
        )

    if "content" in configuration:
        candidate_sources["content"] = (
            base_candidates["content"][index]
        )

    if "semantic" in configuration:
        candidate_sources["semantic"] = (
            base_candidates["semantic"][index]
        )

    if "popularity" in configuration:
        candidate_sources["popularity"] = (
            base_candidates["popularity"][index]
        )

    generator = HybridCandidateGenerator(
        source_weights=source_weights,
    )

    candidates = generator.generate(
        candidate_sources=candidate_sources,
        candidates_per_source=CANDIDATES_PER_SOURCE,
    )

    if not candidates.empty:
        candidates.insert(
            0,
            "userId",
            user_id,
        )

    return candidates


def run_ablation() -> None:
    """
    Run component ablation experiments.

    Configurations:
        1. BPR
        2. BPR + Content
        3. BPR + Content + Semantic
        4. BPR + Content + Semantic + Popularity
        5. Full Hybrid + LightGBM
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

    print()
    print("=" * 60)
    print("BUILDING BASE MODELS")
    print("=" * 60)

    (
        bpr,
        content,
        embedding,
        popularity,
    ) = build_models(
        train=outer_train,
        movies=movies,
        tags=tags,
    )

    users = sorted(
        outer_train["userId"].unique()
    )

    print()
    print("=" * 60)
    print("GENERATING BASE CANDIDATES")
    print("=" * 60)

    base_candidates = generate_base_candidates(
        users=users,
        bpr=bpr,
        content=content,
        embedding=embedding,
        popularity=popularity,
        ratings=outer_train,
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

    feature_builder = RankingFeatureBuilder(
        positive_threshold=4.0,
    )

    label_builder = RankingLabelBuilder(
        positive_threshold=4.0,
    )

    rows = []

    # --------------------------------------------------
    # Candidate-only ablations
    # --------------------------------------------------

    for (
        configuration_name,
        sources,
        weights,
    ) in configurations:
        print()
        print("=" * 60)
        print(
            f"EVALUATING: "
            f"{configuration_name}"
        )
        print("=" * 60)

        datasets = []

        for index, user_id in enumerate(users):
            candidates = build_configuration_candidates(
                user_id=user_id,
                index=index,
                base_candidates=base_candidates,
                configuration=sources,
                source_weights=weights,
            )

            if candidates.empty:
                continue

            features = feature_builder.build(
                candidates=candidates,
                ratings=outer_train,
                movies=movies,
            )

            labeled = label_builder.build(
                candidates=features,
                labels_ratings=outer_validation,
            )

            if labeled.empty:
                continue

            datasets.append(labeled)

        if not datasets:
            continue

        data = pd.concat(
            datasets,
            ignore_index=True,
        )

        # For candidate-only ablation, the hybrid score
        # determines ranking.
        data["ranking_score"] = (
            data["hybrid_candidate_score"]
        )

        metrics = evaluate_candidates(
            data
        )

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

        rows.append(
            {
                "configuration": (
                    configuration_name
                ),
                "model": "candidate_score",
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
                "rows": len(data),
            }
        )

    # --------------------------------------------------
    # Full Hybrid + LightGBM
    # --------------------------------------------------

    print()
    print("=" * 60)
    print("TRAINING FULL HYBRID LIGHTGBM")
    print("=" * 60)

    full_datasets = []

    full_sources = [
        "bpr",
        "content",
        "semantic",
        "popularity",
    ]

    full_weights = {
        "bpr": 0.45,
        "content": 0.20,
        "semantic": 0.25,
        "popularity": 0.10,
    }

    for index, user_id in enumerate(users):
        candidates = build_configuration_candidates(
            user_id=user_id,
            index=index,
            base_candidates=base_candidates,
            configuration=full_sources,
            source_weights=full_weights,
        )

        if candidates.empty:
            continue

        features = feature_builder.build(
            candidates=candidates,
            ratings=outer_train,
            movies=movies,
        )

        labeled = label_builder.build(
            candidates=features,
            labels_ratings=outer_validation,
        )

        if not labeled.empty:
            full_datasets.append(labeled)

    full_data = pd.concat(
        full_datasets,
        ignore_index=True,
    )

    ranker = LightGBMRanker(
        n_estimators=200,
        learning_rate=0.05,
        num_leaves=31,
        random_seed=42,
    )

    groups = (
        full_data
        .groupby(
            "userId",
            sort=False,
        )
        .size()
        .tolist()
    )

    ranker.fit(
        features=full_data,
        groups=groups,
    )

    metrics = evaluate_candidates(
        full_data.assign(
            ranking_score=ranker.predict(
                full_data
            )
        )
    )

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

    rows.append(
        {
            "configuration": "Full_Hybrid",
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
                metrics["users_evaluated"]
            ),
            "rows": len(full_data),
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
                rows[0].keys()
            ),
        )

        writer.writeheader()
        writer.writerows(rows)

    elapsed = (
        time.perf_counter()
        - start_time
    )

    print()
    print("=" * 60)
    print("ABLATION COMPLETE")
    print("=" * 60)

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