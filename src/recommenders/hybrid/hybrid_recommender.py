from __future__ import annotations
from src.ranking.feature_builder import RankingFeatureBuilder
from src.ranking.lgbm_ranker import LightGBMRanker
import pandas as pd

from src.recommenders.collaborative.bpr import BPRRecommender
from src.recommenders.hybrid.candidate_generator import (
    HybridCandidateGenerator,
)
from src.recommenders.popularity.popularity_recommender import (
    PopularityRecommender,
)
from src.recommenders.semantic.content_recommender import (
    ContentRecommender,
)
from src.recommenders.semantic.embedding_recommender import (
    EmbeddingRecommender,
)


class HybridRecommender:
    """
    Hybrid recommendation pipeline.

    Candidate sources:
        1. BPR collaborative filtering
        2. Personalized TF-IDF content filtering
        3. Personalized semantic embedding retrieval
        4. Popularity-based discovery
    """

    def __init__(
        self,
        bpr_model: BPRRecommender,
        content_model: ContentRecommender,
        embedding_model: EmbeddingRecommender,
        popularity_model: PopularityRecommender,
        ratings: pd.DataFrame,
        source_weights: dict[str, float] | None = None,
        movies: pd.DataFrame | None = None,
        ranker: LightGBMRanker | None = None,
        ranking_candidates_per_user: int = 100,
    ) -> None:
        self.bpr_model = bpr_model
        self.content_model = content_model
        self.embedding_model = embedding_model
        self.popularity_model = popularity_model
        self.ratings = ratings
        self.movies = movies
        self.ranker = ranker
        self.ranking_candidates_per_user = (
            ranking_candidates_per_user
        )

        self.candidate_generator = HybridCandidateGenerator(
            source_weights=source_weights
        )

    def generate_candidates(
        self,
        user_id: int,
        candidates_per_source: int = 100,
    ) -> pd.DataFrame:
        """
        Generate candidates from all recommendation sources.
        """

        if candidates_per_source <= 0:
            raise ValueError(
                "candidates_per_source must be greater than zero."
            )

        user_ratings = self.ratings[
            self.ratings["userId"] == user_id
        ]

        already_rated = set(
            user_ratings["movieId"].astype(int)
        )

        bpr_candidates = self.bpr_model.recommend(
            user_id=user_id,
            n=candidates_per_source,
        )

        content_candidates = (
            self.content_model.recommend_for_user(
                user_id=user_id,
                ratings=self.ratings,
                n=candidates_per_source,
            )
        )

        semantic_candidates = (
            self.embedding_model.recommend_for_user(
                user_id=user_id,
                ratings=self.ratings,
                n=candidates_per_source,
            )
        )

        popularity_candidates = (
            self.popularity_model.recommend(
                n=candidates_per_source,
                exclude_movies=already_rated,
            )
        )

        candidate_sources = {
            "bpr": bpr_candidates,
            "content": content_candidates,
            "semantic": semantic_candidates,
            "popularity": popularity_candidates,
        }

        candidates = self.candidate_generator.generate(
            candidate_sources=candidate_sources,
            candidates_per_source=candidates_per_source,
        )

        if not candidates.empty:
            candidates.insert(
                0,
                "userId",
                user_id,
            )

        return candidates

    def recommend(
        self,
        user_id: int,
        n: int = 10,
        candidates_per_source: int = 100,
    ) -> pd.DataFrame:
        """
        Return the top-N hybrid candidates.

        Final learning-to-rank will be added later.
        """

        if n <= 0:
            raise ValueError(
                "n must be greater than zero."
            )

        candidates = self.generate_candidates(
            user_id=user_id,
            candidates_per_source=candidates_per_source,
        )

        if candidates.empty:
            return candidates

        if self.ranker is None:
            return (
                candidates
                .head(n)
                .reset_index(drop=True)
            )

        if self.movies is None:
            raise RuntimeError(
                "movies must be provided when using "
                "the LightGBM ranker."
            )

        ranking_candidates = (
            candidates
            .sort_values(
                "hybrid_candidate_score",
                ascending=False,
            )
            .head(
                self.ranking_candidates_per_user
            )
            .reset_index(drop=True)
        )

        feature_builder = RankingFeatureBuilder(
            positive_threshold=4.0,
        )

        features = feature_builder.build(
            candidates=ranking_candidates,
            ratings=self.ratings,
            movies=self.movies,
        )

        ranked = self.ranker.rank(
            features,
            n=n,
        )

        return ranked