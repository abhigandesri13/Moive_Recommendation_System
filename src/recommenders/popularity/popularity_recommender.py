from __future__ import annotations

import pandas as pd


class PopularityRecommender:
    """
    Popularity-based movie recommender.

    Ranking signal:
        Bayesian-style weighted rating using:
        - average rating
        - rating count

    This helps provide strong discovery and cold-start candidates.
    """

    def __init__(
        self,
        minimum_rating_count: int = 10,
    ) -> None:
        if minimum_rating_count < 0:
            raise ValueError(
                "minimum_rating_count cannot be negative."
            )

        self.minimum_rating_count = minimum_rating_count
        self.movies: pd.DataFrame | None = None

    def fit(
        self,
        movies: pd.DataFrame,
        ratings: pd.DataFrame,
    ) -> "PopularityRecommender":
        """
        Build movie popularity statistics.

        Required movie columns:
            movieId
            title
            genres

        Required rating columns:
            movieId
            rating
        """

        required_movie_columns = {
            "movieId",
            "title",
            "genres",
        }

        required_rating_columns = {
            "movieId",
            "rating",
        }

        missing_movies = (
            required_movie_columns - set(movies.columns)
        )

        if missing_movies:
            raise ValueError(
                "movies: missing required columns: "
                f"{sorted(missing_movies)}"
            )

        missing_ratings = (
            required_rating_columns - set(ratings.columns)
        )

        if missing_ratings:
            raise ValueError(
                "ratings: missing required columns: "
                f"{sorted(missing_ratings)}"
            )

        rating_stats = (
            ratings.groupby("movieId")["rating"]
            .agg(
                rating_count="count",
                rating_average="mean",
            )
            .reset_index()
        )

        movie_data = movies[
            ["movieId", "title", "genres"]
        ].copy()

        movie_data = movie_data.merge(
            rating_stats,
            on="movieId",
            how="left",
        )

        movie_data["rating_count"] = (
            movie_data["rating_count"]
            .fillna(0)
        )

        movie_data["rating_average"] = (
            movie_data["rating_average"]
            .fillna(0.0)
        )

        eligible = movie_data[
            movie_data["rating_count"]
            >= self.minimum_rating_count
        ].copy()

        if eligible.empty:
            self.movies = movie_data
            return self

        global_average = (
            ratings["rating"].mean()
        )

        mean_rating_count = (
            eligible["rating_count"].quantile(0.80)
        )

        if mean_rating_count <= 0:
            mean_rating_count = 1.0

        # Weighted rating:
        #
        # WR = (v / (v + m)) * R
        #      + (m / (v + m)) * C
        #
        # R = movie average rating
        # C = global average rating
        # v = movie rating count
        # m = minimum rating-count threshold
        v = eligible["rating_count"]
        R = eligible["rating_average"]
        C = global_average
        m = mean_rating_count

        eligible["score"] = (
            (v / (v + m)) * R
            + (m / (v + m)) * C
        )

        movie_data = movie_data.merge(
            eligible[
                ["movieId", "score"]
            ],
            on="movieId",
            how="left",
        )

        movie_data["score"] = (
            movie_data["score"]
            .fillna(0.0)
        )

        self.movies = movie_data

        return self

    def recommend(
        self,
        n: int = 10,
        exclude_movies: set[int] | None = None,
    ) -> pd.DataFrame:
        """
        Return the top-N popular movies.
        """

        if n <= 0:
            raise ValueError(
                "n must be greater than zero."
            )

        if self.movies is None:
            raise RuntimeError(
                "Model has not been fitted. "
                "Call fit() first."
            )

        recommendations = self.movies.copy()

        if exclude_movies:
            recommendations = recommendations[
                ~recommendations["movieId"].isin(
                    exclude_movies
                )
            ]

        recommendations = (
            recommendations
            .sort_values(
                "score",
                ascending=False,
            )
            .head(n)
        )

        return recommendations[
            [
                "movieId",
                "title",
                "genres",
                "score",
            ]
        ].reset_index(drop=True)