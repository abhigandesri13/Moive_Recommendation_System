from __future__ import annotations

import pandas as pd


class HybridCandidateGenerator:
    """
    Combines candidate recommendations from multiple sources.

    Each source must return:
        movieId
        score

    The generator:
        1. normalizes each source score
        2. applies source weights
        3. preserves source-specific scores
        4. creates one unified candidate pool
    """

    def __init__(
        self,
        source_weights: dict[str, float] | None = None,
    ) -> None:
        self.source_weights = source_weights or {}
        self.sources: list[str] = []

    @staticmethod
    def _validate_candidates(
        candidates: pd.DataFrame,
        source_name: str,
    ) -> None:
        required_columns = {
            "movieId",
            "score",
        }

        missing = (
            required_columns
            - set(candidates.columns)
        )

        if missing:
            raise ValueError(
                f"{source_name}: missing required columns: "
                f"{sorted(missing)}"
            )

    @staticmethod
    def _normalize_scores(
        candidates: pd.DataFrame,
    ) -> pd.DataFrame:
        result = candidates.copy()

        minimum = result["score"].min()
        maximum = result["score"].max()

        if maximum == minimum:
            result["normalized_score"] = 1.0
        else:
            result["normalized_score"] = (
                (result["score"] - minimum)
                / (maximum - minimum)
            )

        return result

    def generate(
        self,
        candidate_sources: dict[str, pd.DataFrame],
        candidates_per_source: int = 100,
    ) -> pd.DataFrame:
        """
        Combine candidates from multiple recommendation sources.

        Returns:
            movieId
            hybrid_candidate_score
            bpr_score
            content_score
            semantic_score
            popularity_score
            sources
        """

        if candidates_per_source <= 0:
            raise ValueError(
                "candidates_per_source must be greater than zero."
            )

        if not candidate_sources:
            return pd.DataFrame(
                columns=[
                    "movieId",
                    "hybrid_candidate_score",
                    "bpr_score",
                    "content_score",
                    "semantic_score",
                    "popularity_score",
                    "sources",
                ]
            )

        combined_frames: list[pd.DataFrame] = []

        for source_name, candidates in candidate_sources.items():
            self._validate_candidates(
                candidates,
                source_name,
            )

            if candidates.empty:
                continue

            prepared = (
                candidates[
                    [
                        "movieId",
                        "score",
                    ]
                ]
                .drop_duplicates(
                    subset=["movieId"]
                )
                .sort_values(
                    "score",
                    ascending=False,
                )
                .head(
                    candidates_per_source
                )
                .copy()
            )

            prepared = self._normalize_scores(
                prepared
            )

            weight = self.source_weights.get(
                source_name,
                1.0,
            )

            if weight < 0:
                raise ValueError(
                    f"Source weight for '{source_name}' "
                    "cannot be negative."
                )

            prepared["weighted_score"] = (
                prepared["normalized_score"]
                * weight
            )

            prepared["source"] = source_name

            combined_frames.append(
                prepared
            )

        if not combined_frames:
            return pd.DataFrame(
                columns=[
                    "movieId",
                    "hybrid_candidate_score",
                    "bpr_score",
                    "content_score",
                    "semantic_score",
                    "popularity_score",
                    "sources",
                ]
            )

        combined = pd.concat(
            combined_frames,
            ignore_index=True,
        )

        # --------------------------------------------------
        # Aggregate hybrid score
        # --------------------------------------------------

        hybrid_scores = (
            combined.groupby("movieId")[
                "weighted_score"
            ]
            .sum()
            .rename(
                "hybrid_candidate_score"
            )
        )

        # --------------------------------------------------
        # Preserve source-specific normalized scores
        # --------------------------------------------------

        source_score_frames = []

        for source_name in candidate_sources:
            source_data = combined[
                combined["source"]
                == source_name
            ][
                [
                    "movieId",
                    "normalized_score",
                ]
            ].copy()

            if source_data.empty:
                continue

            column_name = (
                f"{source_name}_score"
            )

            source_data = source_data.rename(
                columns={
                    "normalized_score":
                        column_name
                }
            )

            source_score_frames.append(
                source_data
            )

        result = (
            hybrid_scores
            .reset_index()
        )

        for source_data in source_score_frames:
            result = result.merge(
                source_data,
                on="movieId",
                how="left",
            )

        result["sources"] = (
            combined.groupby("movieId")[
                "source"
            ]
            .apply(
                lambda values:
                ",".join(
                    sorted(
                        set(values)
                    )
                )
            )
            .values
        )

        expected_source_columns = [
            "bpr_score",
            "content_score",
            "semantic_score",
            "popularity_score",
        ]

        for column in expected_source_columns:
            if column not in result.columns:
                result[column] = 0.0

        result = (
            result[
                [
                    "movieId",
                    "hybrid_candidate_score",
                    "bpr_score",
                    "content_score",
                    "semantic_score",
                    "popularity_score",
                    "sources",
                ]
            ]
            .fillna(0.0)
            .sort_values(
                "hybrid_candidate_score",
                ascending=False,
            )
            .reset_index(drop=True)
        )

        return result