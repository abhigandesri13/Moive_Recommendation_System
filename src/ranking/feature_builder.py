from __future__ import annotations

import numpy as np
import pandas as pd


class RankingFeatureBuilder:
    """
    Builds user-movie ranking features.

    The implementation is vectorized so that user-level,
    movie-level, and genre-level features are computed once
    and then merged into the candidate pool.
    """

    def __init__(
        self,
        positive_threshold: float = 4.0,
    ) -> None:
        if positive_threshold <= 0:
            raise ValueError(
                "positive_threshold must be greater than zero."
            )

        self.positive_threshold = positive_threshold

    def build(
        self,
        candidates: pd.DataFrame,
        ratings: pd.DataFrame,
        movies: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Build ranking features for a candidate pool.

        Required candidate columns:
            userId
            movieId
            hybrid_candidate_score
            bpr_score
            content_score
            semantic_score
            popularity_score
            sources

        Required rating columns:
            userId
            movieId
            rating

        Required movie columns:
            movieId
            title
            genres
        """

        required_candidate_columns = {
            "userId",
            "movieId",
            "hybrid_candidate_score",
            "bpr_score",
            "content_score",
            "semantic_score",
            "popularity_score",
            "sources",
        }

        required_rating_columns = {
            "userId",
            "movieId",
            "rating",
        }

        required_movie_columns = {
            "movieId",
            "title",
            "genres",
        }

        missing_candidates = (
            required_candidate_columns
            - set(candidates.columns)
        )

        if missing_candidates:
            raise ValueError(
                "candidates: missing required columns: "
                f"{sorted(missing_candidates)}"
            )

        missing_ratings = (
            required_rating_columns
            - set(ratings.columns)
        )

        if missing_ratings:
            raise ValueError(
                "ratings: missing required columns: "
                f"{sorted(missing_ratings)}"
            )

        missing_movies = (
            required_movie_columns
            - set(movies.columns)
        )

        if missing_movies:
            raise ValueError(
                "movies: missing required columns: "
                f"{sorted(missing_movies)}"
            )

        if candidates.empty:
            return pd.DataFrame()

        # --------------------------------------------------
        # Base candidate features
        # --------------------------------------------------

        features = candidates[
            [
                "userId",
                "movieId",
                "hybrid_candidate_score",
                "bpr_score",
                "content_score",
                "semantic_score",
                "popularity_score",
                "sources",
            ]
        ].copy()

        source_lists = (
            features["sources"]
            .astype(str)
            .str.split(",")
        )

        features["source_count"] = (
            source_lists.str.len()
        )

        features["from_bpr"] = (
            source_lists.apply(
                lambda values: int(
                    "bpr" in values
                )
            )
        )

        features["from_content"] = (
            source_lists.apply(
                lambda values: int(
                    "content" in values
                )
            )
        )

        features["from_semantic"] = (
            source_lists.apply(
                lambda values: int(
                    "semantic" in values
                )
            )
        )

        features["from_popularity"] = (
            source_lists.apply(
                lambda values: int(
                    "popularity" in values
                )
            )
        )
                # --------------------------------------------------
        # Source-relative rank features
        # --------------------------------------------------

        source_rank_columns = [
            ("bpr", "bpr_score", "bpr_rank_pct"),
            ("content", "content_score", "content_rank_pct"),
            ("semantic", "semantic_score", "semantic_rank_pct"),
            ("popularity", "popularity_score", "popularity_rank_pct"),
        ]

        for source_name, score_column, rank_column in source_rank_columns:

            present = (
                features[f"from_{source_name}"] == 1
            )

            ranks = (
                features.loc[present]
                .groupby("userId")[score_column]
                .rank(
                    ascending=False,
                    method="min",
                )
            )

            counts = (
                features.loc[present]
                .groupby("userId")[score_column]
                .transform("count")
            )

            rank_percent = pd.Series(
                0.0,
                index=features.index,
            )

            rank_percent.loc[present] = np.where(
                counts > 1,
                1.0
                - (
                    (ranks - 1.0)
                    / (counts - 1.0)
                ),
                1.0,
            )

            features[rank_column] = rank_percent

        features = features.drop(
            columns=["sources"]
        )

        # --------------------------------------------------
        # Movie-level statistics
        # --------------------------------------------------

        global_rating_average = (
            float(ratings["rating"].mean())
            if not ratings.empty
            else 0.0
        )

        movie_stats = (
            ratings.groupby("movieId")["rating"]
            .agg(
                movie_rating_count="count",
                movie_rating_average="mean",
            )
            .reset_index()
        )

        features = features.merge(
            movie_stats,
            on="movieId",
            how="left",
        )

        features["movie_rating_count"] = (
            features["movie_rating_count"]
            .fillna(0.0)
        )

        features["movie_rating_average"] = (
            features["movie_rating_average"]
            .fillna(global_rating_average)
        )

        # --------------------------------------------------
        # User-level statistics
        # --------------------------------------------------

        user_stats = (
            ratings.groupby("userId")["rating"]
            .agg(
                user_rating_count="count",
                user_average_rating="mean",
            )
            .reset_index()
        )

        positive_mask = (
            ratings["rating"]
            >= self.positive_threshold
        )

        positive_by_user = (
            ratings.loc[positive_mask]
            .groupby("userId")
            .size()
            .rename("positive_rating_count")
        )

        user_stats = user_stats.merge(
            positive_by_user,
            on="userId",
            how="left",
        )

        user_stats[
            "positive_rating_count"
        ] = user_stats[
            "positive_rating_count"
        ].fillna(0)

        user_stats[
            "user_positive_ratio"
        ] = (
            user_stats["positive_rating_count"]
            / user_stats["user_rating_count"]
        )

        user_stats[
            "user_positive_ratio"
        ] = user_stats[
            "user_positive_ratio"
        ].fillna(0.0)

        user_stats = user_stats[
            [
                "userId",
                "user_rating_count",
                "user_average_rating",
                "user_positive_ratio",
            ]
        ]

        features = features.merge(
            user_stats,
            on="userId",
            how="left",
        )

        features["user_rating_count"] = (
            features["user_rating_count"]
            .fillna(0.0)
        )

        features["user_average_rating"] = (
            features["user_average_rating"]
            .fillna(global_rating_average)
        )

        features["user_positive_ratio"] = (
            features["user_positive_ratio"]
            .fillna(0.0)
        )

        # --------------------------------------------------
        # Positive user history
        # --------------------------------------------------

        positive_ratings = ratings.loc[
            positive_mask,
            [
                "userId",
                "movieId",
                "rating",
            ],
        ].copy()

        # --------------------------------------------------
        # Positive-history movie genres
        # --------------------------------------------------

        positive_movies = positive_ratings.merge(
            movies[
                [
                    "movieId",
                    "genres",
                ]
            ],
            on="movieId",
            how="left",
        )

        positive_movies["genres"] = (
            positive_movies["genres"]
            .fillna("")
        )

        positive_movies["genre"] = (
            positive_movies["genres"]
            .str.split("|")
        )

        positive_movies = (
            positive_movies
            .explode("genre")
        )

        positive_movies["genre"] = (
            positive_movies["genre"]
            .fillna("")
            .astype(str)
        )

        positive_movies = positive_movies[
            positive_movies["genre"].ne("")
            & positive_movies["genre"].ne(
                "(no genres listed)"
            )
        ]

        # --------------------------------------------------
        # User genre count + rating weight
        # --------------------------------------------------

        user_genre = (
            positive_movies
            .groupby(
                [
                    "userId",
                    "genre",
                ]
            )
            .agg(
                genre_count=(
                    "movieId",
                    "nunique",
                ),
                genre_rating_sum=(
                    "rating",
                    "sum",
                ),
            )
            .reset_index()
        )

        positive_movie_counts = (
            positive_ratings
            .groupby("userId")
            .size()
            .rename(
                "positive_movie_count"
            )
            .reset_index()
        )

        user_genre = user_genre.merge(
            positive_movie_counts,
            on="userId",
            how="left",
        )

        # Original feature semantics:
        #
        # user_genre_affinity =
        # matching positive movie count
        # --------------------------------
        # positive movie count × candidate
        # genre count
        user_genre[
            "genre_count_ratio"
        ] = (
            user_genre["genre_count"]
            / user_genre[
                "positive_movie_count"
            ]
        )

        total_rating_weight = (
            user_genre.groupby("userId")[
                "genre_rating_sum"
            ]
            .sum()
            .rename(
                "total_rating_weight"
            )
            .reset_index()
        )

        user_genre = user_genre.merge(
            total_rating_weight,
            on="userId",
            how="left",
        )

        user_genre[
            "genre_rating_weight"
        ] = np.where(
            user_genre[
                "total_rating_weight"
            ]
            > 0,
            user_genre[
                "genre_rating_sum"
            ]
            / user_genre[
                "total_rating_weight"
            ],
            0.0,
        )

        # --------------------------------------------------
        # Candidate genres
        # --------------------------------------------------

        candidate_genres = candidates[
            [
                "userId",
                "movieId",
            ]
        ].copy()

        candidate_genres["_candidate_row"] = (
            np.arange(
                len(candidate_genres)
            )
        )

        candidate_genres = candidate_genres.merge(
            movies[
                [
                    "movieId",
                    "genres",
                ]
            ],
            on="movieId",
            how="left",
        )

        candidate_genres["genres"] = (
            candidate_genres["genres"]
            .fillna("")
        )

        candidate_genres["genre"] = (
            candidate_genres["genres"]
            .str.split("|")
        )

        candidate_genres = (
            candidate_genres
            .explode("genre")
        )

        candidate_genres["genre"] = (
            candidate_genres["genre"]
            .fillna("")
            .astype(str)
        )

        candidate_genres = candidate_genres[
            candidate_genres["genre"].ne("")
            & candidate_genres["genre"].ne(
                "(no genres listed)"
            )
        ]

        candidate_genre_counts = (
            candidate_genres
            .groupby("_candidate_row")
            .size()
            .rename(
                "candidate_genre_count"
            )
            .reset_index()
        )

        # --------------------------------------------------
        # Merge candidate genres with user preferences
        # --------------------------------------------------

        candidate_genres = candidate_genres.merge(
            user_genre[
                [
                    "userId",
                    "genre",
                    "genre_count_ratio",
                    "genre_rating_weight",
                ]
            ],
            on=[
                "userId",
                "genre",
            ],
            how="left",
        )

        candidate_genres[
            "genre_count_ratio"
        ] = candidate_genres[
            "genre_count_ratio"
        ].fillna(0.0)

        candidate_genres[
            "genre_rating_weight"
        ] = candidate_genres[
            "genre_rating_weight"
        ].fillna(0.0)

        # --------------------------------------------------
        # Aggregate genre-based candidate features
        # --------------------------------------------------

        candidate_genre_features = (
            candidate_genres
            .groupby(
                "_candidate_row"
            )
            .agg(
                user_genre_affinity=(
                    "genre_count_ratio",
                    "sum",
                ),
                user_movie_affinity=(
                    "genre_rating_weight",
                    "sum",
                ),
            )
            .reset_index()
        )

        candidate_genre_features = (
            candidate_genre_features.merge(
                candidate_genre_counts,
                on="_candidate_row",
                how="left",
            )
        )

        candidate_genre_features[
            "user_genre_affinity"
        ] = np.where(
            candidate_genre_features[
                "candidate_genre_count"
            ]
            > 0,
            candidate_genre_features[
                "user_genre_affinity"
            ]
            / candidate_genre_features[
                "candidate_genre_count"
            ],
            0.0,
        )

        # --------------------------------------------------
        # Attach candidate-level row ID
        # --------------------------------------------------

        features["_candidate_row"] = (
            np.arange(
                len(features)
            )
        )

        features = features.merge(
            candidate_genre_features[
                [
                    "_candidate_row",
                    "user_genre_affinity",
                    "user_movie_affinity",
                ]
            ],
            on="_candidate_row",
            how="left",
        )

        # --------------------------------------------------
        # Final cleanup
        # --------------------------------------------------

        numeric_columns = [
            "hybrid_candidate_score",
            "bpr_score",
            "content_score",
            "semantic_score",
            "popularity_score",
            "source_count",
            "from_bpr",
            "from_content",
            "from_semantic",
            "from_popularity",
            "movie_rating_count",
            "movie_rating_average",
            "user_rating_count",
            "user_average_rating",
            "user_positive_ratio",
            "user_genre_affinity",
            "user_movie_affinity",
            "bpr_rank_pct",
            "content_rank_pct",
            "semantic_rank_pct",
            "popularity_rank_pct",
        ]

        for column in numeric_columns:
            if column in features.columns:
                features[column] = (
                    pd.to_numeric(
                        features[column],
                        errors="coerce",
                    )
                    .fillna(0.0)
                )

        features = features.drop(
            columns=["_candidate_row"]
        )

        features = features.replace(
            [np.inf, -np.inf],
            0.0,
        )

        return features