from __future__ import annotations
from pathlib import Path

import pandas as pd
import lightgbm as lgb


class LightGBMRanker:
    """
    Learning-to-rank model for movie recommendation.

    LightGBM learns how to combine:
        - BPR score
        - TF-IDF content score
        - semantic score
        - popularity score
        - movie quality
        - user preference
        - source participation
    """

    DEFAULT_FEATURE_COLUMNS = [
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
        "bpr_rank_pct",
        "content_rank_pct",
        "semantic_rank_pct",
        "popularity_rank_pct",
        "movie_rating_count",
        "movie_rating_average",
        "user_rating_count",
        "user_average_rating",
        "user_positive_ratio",
        "user_genre_affinity",
        "user_movie_affinity",
    ]

    def __init__(
        self,
        feature_columns: list[str] | None = None,
        n_estimators: int = 200,
        learning_rate: float = 0.05,
        num_leaves: int = 31,
        max_depth: int = -1,
        random_seed: int = 42,
    ) -> None:
        self.feature_columns = (
            feature_columns
            if feature_columns is not None
            else self.DEFAULT_FEATURE_COLUMNS.copy()
        )

        self.model = lgb.LGBMRanker(
            objective="lambdarank",
            metric="ndcg",
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            num_leaves=num_leaves,
            max_depth=max_depth,
            random_state=random_seed,
            verbosity=-1,
        )

        self.is_fitted = False

    def _validate_features(
        self,
        features: pd.DataFrame,
    ) -> None:
        missing = (
            set(self.feature_columns)
            - set(features.columns)
        )

        if missing:
            raise ValueError(
                "features: missing required columns: "
                f"{sorted(missing)}"
            )

    def fit(
        self,
        features: pd.DataFrame,
        labels: pd.Series | None = None,
        groups: list[int] | None = None,
    ) -> "LightGBMRanker":
        """
        Train the learning-to-rank model.

        Parameters
        ----------
        features:
            Ranking feature DataFrame.

        labels:
            Relevance labels. If omitted, the method expects
            a 'label' column in features.

        groups:
            Number of candidate rows belonging to each query/user.
            For recommendation ranking, each user is one query.
        """

        self._validate_features(features)

        if labels is None:
            if "label" not in features.columns:
                raise ValueError(
                    "features must contain 'label' when "
                    "labels is not provided."
                )

            labels = features["label"]

        if len(features) != len(labels):
            raise ValueError(
                "features and labels must have the same length."
            )

        if groups is None:
            if "userId" not in features.columns:
                raise ValueError(
                    "features must contain 'userId' "
                    "when groups is not provided."
                )

            groups = (
                features.groupby(
                    "userId",
                    sort=False,
                )
                .size()
                .tolist()
            )

        if sum(groups) != len(features):
            raise ValueError(
                "Sum of groups must equal number of rows."
            )

        X = features[
            self.feature_columns
        ].copy()

        y = pd.Series(
            labels,
            index=features.index,
        ).astype(int)

        self.model.fit(
            X,
            y,
            group=groups,
        )

        self.is_fitted = True

        return self

    def predict(
        self,
        features: pd.DataFrame,
    ) -> pd.Series:
        """
        Predict ranking scores for candidate rows.
        """

        if not self.is_fitted:
            raise RuntimeError(
                "Ranker has not been fitted. "
                "Call fit() first."
            )

        self._validate_features(features)

        X = features[
            self.feature_columns
        ]

        predictions = self.model.predict(X)

        return pd.Series(
            predictions,
            index=features.index,
            name="ranking_score",
        )

    def rank(
        self,
        features: pd.DataFrame,
        n: int = 10,
    ) -> pd.DataFrame:
        """
        Return top-N candidates according to LightGBM score.
        """

        if n <= 0:
            raise ValueError(
                "n must be greater than zero."
            )

        result = features.copy()

        result["ranking_score"] = (
            self.predict(features)
        )

        return (
            result
            .sort_values(
                "ranking_score",
                ascending=False,
            )
            .head(n)
            .reset_index(drop=True)
        )

    def feature_importance(
        self,
    ) -> pd.DataFrame:
        """
        Return LightGBM feature importance.
        """

        if not self.is_fitted:
            raise RuntimeError(
                "Ranker has not been fitted yet."
            )

        importance = self.model.feature_importances_

        result = pd.DataFrame(
            {
                "feature": self.feature_columns,
                "importance": importance,
            }
        )

        return (
            result
            .sort_values(
                "importance",
                ascending=False,
            )
            .reset_index(drop=True)
        )
    def save(
        self,
        path: str | Path,
    ) -> None:
        """
        Save the trained LightGBM ranker.
        """

        if not self.is_fitted:
            raise RuntimeError(
                "Ranker has not been fitted. "
                "Call fit() before save()."
            )

        output_path = Path(path)

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.model.booster_.save_model(
            str(output_path)
        )

    @classmethod
    def load(
        cls,
        path: str | Path,
        feature_columns: list[str] | None = None,
    ) -> "LightGBMRanker":
        """
        Load a previously trained LightGBM ranker.
        """

        input_path = Path(path)

        if not input_path.exists():
            raise FileNotFoundError(
                f"Ranker model not found: {input_path}"
            )

        ranker = cls(
            feature_columns=feature_columns,
        )

        ranker.model._Booster = lgb.Booster(
            model_file=str(input_path)
        )

        ranker.model.fitted_ = True
        ranker.is_fitted = True
        return ranker