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
from src.recommenders.hybrid.candidate_generator import (
    HybridCandidateGenerator,
)


RESULTS_DIR = Path("experiments") / "results"

RESULTS_FILE = (
    RESULTS_DIR
    / "candidate_recall_analysis.csv"
)

CANDIDATES_PER_SOURCE = 200


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
    Build all candidate-generation models using only training data.
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


def evaluate_candidate_recall(
    users: list[int],
    bpr: BPRRecommender,
    content: ContentRecommender,
    embedding: EmbeddingRecommender,
    popularity: PopularityRecommender,
    train: pd.DataFrame,
    validation: pd.DataFrame,
) -> pd.DataFrame:
    """
    Measure whether each candidate source retrieves future
    positive validation interactions.

    A positive interaction is validation rating >= 4.
    """

    rows: list[dict] = []

    for position, user_id in enumerate(
        users,
        start=1,
    ):
        print(
            f"\rAnalyzing candidates: "
            f"{position}/{len(users)}",
            end="",
            flush=True,
        )

        user_validation = validation[
            validation["userId"] == user_id
        ]

        positive_validation = set(
            user_validation.loc[
                user_validation["rating"] >= 4.0,
                "movieId",
            ].astype(int)
        )

        if not positive_validation:
            continue

        user_train = train[
            train["userId"] == user_id
        ]

        already_seen = set(
            user_train["movieId"].astype(int)
        )

        bpr_movies = set(
            bpr.recommend(
                user_id=user_id,
                n=CANDIDATES_PER_SOURCE,
            )["movieId"].astype(int)
        )

        content_movies = set(
            content.recommend_for_user(
                user_id=user_id,
                ratings=train,
                n=CANDIDATES_PER_SOURCE,
            )["movieId"].astype(int)
        )

        semantic_movies = set(
            embedding.recommend_for_user(
                user_id=user_id,
                ratings=train,
                n=CANDIDATES_PER_SOURCE,
            )["movieId"].astype(int)
        )

        popularity_movies = set(
            popularity.recommend(
                n=CANDIDATES_PER_SOURCE,
                exclude_movies=already_seen,
            )["movieId"].astype(int)
        )

        source_sets = {
            "bpr": bpr_movies,
            "content": content_movies,
            "semantic": semantic_movies,
            "popularity": popularity_movies,
        }

        combinations = {
            "BPR": {"bpr"},
            "Content": {"content"},
            "Semantic": {"semantic"},
            "Popularity": {"popularity"},
            "BPR_Content": {
                "bpr",
                "content",
            },
            "BPR_Semantic": {
                "bpr",
                "semantic",
            },
            "BPR_Popularity": {
                "bpr",
                "popularity",
            },
            "BPR_Content_Semantic": {
                "bpr",
                "content",
                "semantic",
            },
            "Full_Hybrid": {
                "bpr",
                "content",
                "semantic",
                "popularity",
            },
        }

        for combination_name, source_names in combinations.items():
            candidate_movies: set[int] = set()

            for source_name in source_names:
                candidate_movies.update(
                    source_sets[source_name]
                )

            hits = (
                candidate_movies
                & positive_validation
            )

            recall = (
                len(hits)
                / len(positive_validation)
            )

            rows.append(
                {
                    "userId": user_id,
                    "combination": combination_name,
                    "positive_validation_movies": len(
                        positive_validation
                    ),
                    "candidate_movies": len(
                        candidate_movies
                    ),
                    "hits": len(hits),
                    "recall": recall,
                }
            )

    print()

    return pd.DataFrame(rows)


def run_analysis() -> None:
    """
    Run candidate recall analysis.
    """

    start_time = time.perf_counter()

    loader = MovieLensLoader("data")

    ratings = preprocess_ratings(
        loader.load_ratings()
    )

    movies = loader.load_movies()
    tags = loader.load_tags()

    train, validation, _ = temporal_split(
        ratings
    )

    print()
    print("=" * 60)
    print("BUILDING CANDIDATE MODELS")
    print("=" * 60)

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

    users = sorted(
        train["userId"].unique()
    )

    print()
    print("=" * 60)
    print("CANDIDATE RECALL ANALYSIS")
    print("=" * 60)

    results = evaluate_candidate_recall(
        users=users,
        bpr=bpr,
        content=content,
        embedding=embedding,
        popularity=popularity,
        train=train,
        validation=validation,
    )

    if results.empty:
        raise RuntimeError(
            "No candidate recall results were generated."
        )

    summary = (
        results
        .groupby("combination")
        .agg(
            users_evaluated=(
                "userId",
                "nunique",
            ),
            total_hits=(
                "hits",
                "sum",
            ),
            average_recall=(
                "recall",
                "mean",
            ),
            average_candidate_count=(
                "candidate_movies",
                "mean",
            ),
        )
        .reset_index()
        .sort_values(
            "average_recall",
            ascending=False,
        )
        .reset_index(drop=True)
    )

    print()
    print("=" * 60)
    print("CANDIDATE RECALL SUMMARY")
    print("=" * 60)

    print(
        summary.to_string(
            index=False
        )
    )

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary.to_csv(
        RESULTS_FILE,
        index=False,
    )

    elapsed = (
        time.perf_counter()
        - start_time
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
    run_analysis()