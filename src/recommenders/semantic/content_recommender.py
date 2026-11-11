from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class ContentRecommender:
    """
    Content-based movie recommender using TF-IDF.

    Movie representation:
        genres + tags

    Recommendation signal:
        cosine similarity between movie content vectors.
    """

    def __init__(self) -> None:
        self.vectorizer = TfidfVectorizer(
            stop_words="english"
        )

        self.movies: pd.DataFrame | None = None
        self.tfidf_matrix = None
        self.movie_index: dict[int, int] = {}

    def fit(
        self,
        movies: pd.DataFrame,
        tags: pd.DataFrame,
    ) -> "ContentRecommender":
        """
        Build TF-IDF representations for movies.

        Required movie columns:
            movieId
            title
            genres

        Required tag columns:
            movieId
            tag
        """

        required_movie_columns = {
            "movieId",
            "title",
            "genres",
        }

        required_tag_columns = {
            "movieId",
            "tag",
        }

        missing_movies = (
            required_movie_columns - set(movies.columns)
        )

        if missing_movies:
            raise ValueError(
                "movies: missing required columns: "
                f"{sorted(missing_movies)}"
            )

        missing_tags = (
            required_tag_columns - set(tags.columns)
        )

        if missing_tags:
            raise ValueError(
                "tags: missing required columns: "
                f"{sorted(missing_tags)}"
            )

        movies_copy = movies[
            ["movieId", "title", "genres"]
        ].copy()

        tags_grouped = (
            tags.groupby("movieId")["tag"]
            .apply(
                lambda values: " ".join(
                    values.astype(str)
                )
            )
            .reset_index()
        )

        movie_content = movies_copy.merge(
            tags_grouped,
            on="movieId",
            how="left",
        )

        movie_content["tag"] = (
            movie_content["tag"].fillna("")
        )

        movie_content["genres_clean"] = (
            movie_content["genres"]
            .apply(
                lambda value:
                value.replace("|", " ")
                if isinstance(value, str)
                else ""
            )
        )

        movie_content["content"] = (
            movie_content["genres_clean"]
            + " "
            + movie_content["tag"]
        )

        self.tfidf_matrix = (
            self.vectorizer.fit_transform(
                movie_content["content"]
            )
        )

        self.movies = movie_content.reset_index(
            drop=True
        )

        self.movie_index = {
            int(movie_id): index
            for index, movie_id
            in enumerate(
                self.movies["movieId"]
            )
        }

        return self

    def recommend(
        self,
        movie_id: int,
        n: int = 10,
    ) -> pd.DataFrame:
        """
        Return top-N movies similar to a given movie.
        """

        if n <= 0:
            raise ValueError(
                "n must be greater than zero."
            )

        if self.movies is None or self.tfidf_matrix is None:
            raise RuntimeError(
                "Model has not been fitted. "
                "Call fit() first."
            )

        if movie_id not in self.movie_index:
            return pd.DataFrame(
                columns=[
                    "movieId",
                    "title",
                    "genres",
                    "score",
                ]
            )

        movie_position = self.movie_index[movie_id]

        similarity_scores = cosine_similarity(
            self.tfidf_matrix[movie_position],
            self.tfidf_matrix,
        ).flatten()

        similarity_scores[movie_position] = -1.0

        top_indices = similarity_scores.argsort()[
            ::-1
        ][:n]

        recommendations = self.movies.iloc[
            top_indices
        ][
            ["movieId", "title", "genres"]
        ].copy()

        recommendations["score"] = (
            similarity_scores[top_indices]
        )

        return recommendations.reset_index(
            drop=True
        )

    def recommend_for_user(
        self,
        user_id: int,
        ratings: pd.DataFrame,
        n: int = 10,
        positive_threshold: float = 4.0,
    ) -> pd.DataFrame:
        """
        Recommend movies based on a user's positive rating history.

        The user's content profile is created by averaging the
        TF-IDF vectors of movies the user rated positively.

        Parameters
        ----------
        user_id:
            User identifier.

        ratings:
            Ratings DataFrame containing:
                userId
                movieId
                rating

        n:
            Number of recommendations.

        positive_threshold:
            Minimum rating considered a positive preference.
        """

        if n <= 0:
            raise ValueError(
                "n must be greater than zero."
            )

        if self.movies is None or self.tfidf_matrix is None:
            raise RuntimeError(
                "Model has not been fitted. "
                "Call fit() first."
            )

        required_columns = {
            "userId",
            "movieId",
            "rating",
        }

        missing = required_columns - set(ratings.columns)

        if missing:
            raise ValueError(
                "ratings: missing required columns: "
                f"{sorted(missing)}"
            )

        user_ratings = ratings[
            ratings["userId"] == user_id
        ].copy()

        positive_ratings = user_ratings[
            user_ratings["rating"] >= positive_threshold
        ]

        if positive_ratings.empty:
            return pd.DataFrame(
                columns=[
                    "movieId",
                    "title",
                    "genres",
                    "score",
                ]
            )

        positive_movie_ids = [
            int(movie_id)
            for movie_id in positive_ratings["movieId"]
            if int(movie_id) in self.movie_index
        ]

        if not positive_movie_ids:
            return pd.DataFrame(
                columns=[
                    "movieId",
                    "title",
                    "genres",
                    "score",
                ]
            )

        positive_indices = [
            self.movie_index[movie_id]
            for movie_id in positive_movie_ids
        ]

        user_profile = np.asarray(
            self.tfidf_matrix[
                positive_indices
            ].mean(axis=0)
        )

        similarity_scores = cosine_similarity(
            user_profile,
            self.tfidf_matrix,
        ).flatten()

        rated_movie_ids = set(
            user_ratings["movieId"].astype(int)
        )

        rated_indices = [
            self.movie_index[movie_id]
            for movie_id in rated_movie_ids
            if movie_id in self.movie_index
        ]

        similarity_scores[rated_indices] = -1.0

        top_indices = similarity_scores.argsort()[
            ::-1
        ][:n]

        recommendations = self.movies.iloc[
            top_indices
        ][
            ["movieId", "title", "genres"]
        ].copy()

        recommendations["score"] = (
            similarity_scores[top_indices]
        )

        return recommendations.reset_index(
            drop=True
        )