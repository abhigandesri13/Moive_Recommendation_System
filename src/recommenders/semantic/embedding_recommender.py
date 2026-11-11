from __future__ import annotations

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import normalize


class EmbeddingRecommender:
    """
    Semantic movie recommender using Sentence Transformers.

    Movie representation:
        title + genres + tags

    Recommendation signal:
        cosine similarity between dense semantic embeddings.
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
    ) -> None:
        self.model_name = model_name

        self.model = SentenceTransformer(
            model_name
        )

        self.movies: pd.DataFrame | None = None
        self.embeddings: np.ndarray | None = None
        self.movie_index: dict[int, int] = {}

    def fit(
        self,
        movies: pd.DataFrame,
        tags: pd.DataFrame,
    ) -> "EmbeddingRecommender":
        """
        Generate semantic embeddings for the movie catalog.

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
            required_movie_columns
            - set(movies.columns)
        )

        if missing_movies:
            raise ValueError(
                "movies: missing required columns: "
                f"{sorted(missing_movies)}"
            )

        missing_tags = (
            required_tag_columns
            - set(tags.columns)
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

        movie_content["semantic_text"] = (
            "Title: "
            + movie_content["title"].fillna("")
            + ". Genres: "
            + movie_content["genres_clean"]
            + ". Tags: "
            + movie_content["tag"]
        )

        self.embeddings = self.model.encode(
            movie_content["semantic_text"].tolist(),
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,
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
        Return top-N semantically similar movies.
        """

        if n <= 0:
            raise ValueError(
                "n must be greater than zero."
            )

        if (
            self.movies is None
            or self.embeddings is None
        ):
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

        movie_position = self.movie_index[
            movie_id
        ]

        query_embedding = self.embeddings[
            movie_position
        ]

        similarity_scores = (
            self.embeddings @ query_embedding
        )

        similarity_scores[
            movie_position
        ] = -1.0

        top_indices = similarity_scores.argsort()[
            ::-1
        ][:n]

        recommendations = self.movies.iloc[
            top_indices
        ][
            [
                "movieId",
                "title",
                "genres",
            ]
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
        Recommend movies based on a user's positive
        movie history using semantic embeddings.
        """

        if n <= 0:
            raise ValueError(
                "n must be greater than zero."
            )

        if (
            self.movies is None
            or self.embeddings is None
        ):
            raise RuntimeError(
                "Model has not been fitted. "
                "Call fit() first."
            )

        required_columns = {
            "userId",
            "movieId",
            "rating",
        }

        missing = (
            required_columns
            - set(ratings.columns)
        )

        if missing:
            raise ValueError(
                "ratings: missing required columns: "
                f"{sorted(missing)}"
            )

        user_ratings = ratings[
            ratings["userId"] == user_id
        ].copy()

        positive_ratings = user_ratings[
            user_ratings["rating"]
            >= positive_threshold
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
            for movie_id
            in positive_ratings["movieId"]
            if int(movie_id)
            in self.movie_index
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
            for movie_id
            in positive_movie_ids
        ]

        positive_embeddings = (
            self.embeddings[
                positive_indices
            ]
        )

        user_profile = positive_embeddings.mean(
            axis=0
        )

        user_profile = normalize(
            user_profile.reshape(1, -1)
        )[0]

        similarity_scores = (
            self.embeddings @ user_profile
        )

        rated_movie_ids = set(
            user_ratings["movieId"]
            .astype(int)
        )

        rated_indices = [
            self.movie_index[movie_id]
            for movie_id
            in rated_movie_ids
            if movie_id in self.movie_index
        ]

        similarity_scores[
            rated_indices
        ] = -1.0

        top_indices = similarity_scores.argsort()[
            ::-1
        ][:n]

        recommendations = self.movies.iloc[
            top_indices
        ][
            [
                "movieId",
                "title",
                "genres",
            ]
        ].copy()

        recommendations["score"] = (
            similarity_scores[top_indices]
        )

        return recommendations.reset_index(
            drop=True
        )