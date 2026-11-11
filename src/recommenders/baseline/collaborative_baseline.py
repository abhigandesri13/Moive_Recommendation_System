import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


class CollaborativeBaseline:
    """
    Baseline item-based collaborative filtering recommender.

    This preserves the core idea of the original project:
    user-item ratings -> item similarity -> personalized scoring.

    It is intended as an evaluation baseline, not the final production model.
    """

    def __init__(self) -> None:
        self.user_item_matrix: pd.DataFrame | None = None
        self.item_similarity: np.ndarray | None = None
        self.movie_ids: np.ndarray | None = None

    def fit(self, ratings: pd.DataFrame) -> "CollaborativeBaseline":
        """Build the user-item matrix and item similarity model."""

        required_columns = {"userId", "movieId", "rating"}

        missing_columns = required_columns - set(ratings.columns)

        if missing_columns:
            raise ValueError(
                f"ratings: missing required columns: {sorted(missing_columns)}"
            )

        matrix = ratings.pivot_table(
            index="userId",
            columns="movieId",
            values="rating",
            fill_value=0,
        )

        self.user_item_matrix = matrix
        self.movie_ids = matrix.columns.to_numpy()

        # Item-item similarity based on user-rating vectors.
        self.item_similarity = cosine_similarity(matrix.T)

        return self

    def recommend(
        self,
        user_id: int,
        n: int = 10,
        exclude_movies: set[int] | None = None,
    ) -> pd.DataFrame:
        """
        Generate personalized recommendations for a user.
        """

        if self.user_item_matrix is None or self.item_similarity is None:
            raise RuntimeError(
                "Model has not been fitted. Call fit() first."
            )

        if user_id not in self.user_item_matrix.index:
            return pd.DataFrame(
                columns=["movieId", "score"]
            )

        if n <= 0:
            raise ValueError("n must be greater than zero.")

        user_ratings = self.user_item_matrix.loc[user_id].to_numpy()

        # Weighted item scores based on the user's historical ratings.
        numerator = self.item_similarity @ user_ratings

        rated_mask = (user_ratings > 0).astype(float)
        denominator = (
            np.abs(self.item_similarity) @ rated_mask
        )

        with np.errstate(divide="ignore", invalid="ignore"):
            scores = numerator / denominator

        scores = np.nan_to_num(
            scores,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )

        score_series = pd.Series(
            scores,
            index=self.movie_ids,
            name="score",
        )

        # Exclude movies already rated by the user.
        rated_movies = set(
            self.user_item_matrix.columns[
                self.user_item_matrix.loc[user_id] > 0
            ]
        )

        excluded = rated_movies

        if exclude_movies:
            excluded = excluded.union(exclude_movies)

        score_series = score_series[
            ~score_series.index.isin(excluded)
        ]

        recommendations = (
            score_series
            .sort_values(ascending=False)
            .head(n)
            .reset_index()
        )

        recommendations.columns = ["movieId", "score"]

        return recommendations