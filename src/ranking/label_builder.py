from __future__ import annotations

import pandas as pd


class RankingLabelBuilder:
    """
    Creates binary relevance labels for ranking.

    label = 1:
        user gave the movie a rating >= positive_threshold

    label = 0:
        otherwise
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
        labels_ratings: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Attach relevance labels to candidate rows.

        Required candidate columns:
            userId
            movieId

        Required rating columns:
            userId
            movieId
            rating
        """

        required_candidate_columns = {
            "userId",
            "movieId",
        }

        required_rating_columns = {
            "userId",
            "movieId",
            "rating",
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
            - set(labels_ratings.columns)
        )

        if missing_ratings:
            raise ValueError(
                "labels_ratings: missing required columns: "
                f"{sorted(missing_ratings)}"
            )

        result = candidates.copy()

        if result.empty:
            result["label"] = pd.Series(
                dtype=int
            )
            return result

        positive_interactions = (
            labels_ratings[
                labels_ratings["rating"]
                >= self.positive_threshold
            ][
                [
                    "userId",
                    "movieId",
                ]
            ]
            .drop_duplicates()
            .copy()
        )

        positive_interactions["label"] = 1

        result = result.merge(
            positive_interactions,
            on=[
                "userId",
                "movieId",
            ],
            how="left",
        )

        result["label"] = (
            result["label"]
            .fillna(0)
            .astype(int)
        )

        return result