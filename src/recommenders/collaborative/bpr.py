from __future__ import annotations

import random

import numpy as np
import pandas as pd
import torch
from torch import nn


class BPRMatrixFactorization(nn.Module):
    """
    Bayesian Personalized Ranking (BPR) matrix-factorization model.

    The model learns user and item embeddings such that a user's
    positive items receive higher scores than sampled negative items.
    """

    def __init__(
        self,
        n_users: int,
        n_items: int,
        factors: int = 64,
    ) -> None:
        super().__init__()

        if n_users <= 0:
            raise ValueError("n_users must be greater than zero.")

        if n_items <= 0:
            raise ValueError("n_items must be greater than zero.")

        if factors <= 0:
            raise ValueError("factors must be greater than zero.")

        self.user_embedding = nn.Embedding(n_users, factors)
        self.item_embedding = nn.Embedding(n_items, factors)

        self.user_bias = nn.Embedding(n_users, 1)
        self.item_bias = nn.Embedding(n_items, 1)

        self._initialize_embeddings()

    def _initialize_embeddings(self) -> None:
        nn.init.normal_(self.user_embedding.weight, mean=0.0, std=0.01)
        nn.init.normal_(self.item_embedding.weight, mean=0.0, std=0.01)

        nn.init.zeros_(self.user_bias.weight)
        nn.init.zeros_(self.item_bias.weight)

    def score(
        self,
        user_indices: torch.Tensor,
        item_indices: torch.Tensor,
    ) -> torch.Tensor:
        """Calculate predicted preference scores."""

        user_vectors = self.user_embedding(user_indices)
        item_vectors = self.item_embedding(item_indices)

        interaction = (user_vectors * item_vectors).sum(dim=1)

        user_bias = self.user_bias(user_indices).squeeze(-1)
        item_bias = self.item_bias(item_indices).squeeze(-1)

        return interaction + user_bias + item_bias

    def forward(
        self,
        user_indices: torch.Tensor,
        positive_indices: torch.Tensor,
        negative_indices: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return positive and negative scores."""

        positive_scores = self.score(
            user_indices,
            positive_indices,
        )

        negative_scores = self.score(
            user_indices,
            negative_indices,
        )

        return positive_scores, negative_scores, positive_scores - negative_scores


class BPRRecommender:
    """
    Train and serve a BPR matrix-factorization recommender.
    """

    def __init__(
        self,
        factors: int = 64,
        learning_rate: float = 0.01,
        regularization: float = 0.001,
        epochs: int = 20,
        device: str = "cpu",
        random_seed: int = 42,
    ) -> None:

        if factors <= 0:
            raise ValueError("factors must be greater than zero.")

        if learning_rate <= 0:
            raise ValueError("learning_rate must be greater than zero.")

        if regularization < 0:
            raise ValueError("regularization cannot be negative.")

        if epochs <= 0:
            raise ValueError("epochs must be greater than zero.")

        self.factors = factors
        self.learning_rate = learning_rate
        self.regularization = regularization
        self.epochs = epochs
        self.device = torch.device(device)
        self.random_seed = random_seed

        self.model: BPRMatrixFactorization | None = None

        self.user_to_index: dict[int, int] = {}
        self.item_to_index: dict[int, int] = {}

        self.index_to_user: dict[int, int] = {}
        self.index_to_item: dict[int, int] = {}

        self.user_positive_items: dict[int, set[int]] = {}

    def _prepare_data(
        self,
        ratings: pd.DataFrame,
    ) -> pd.DataFrame:
        """Prepare positive interactions for BPR training."""

        required = {"userId", "movieId", "rating"}

        missing = required - set(ratings.columns)

        if missing:
            raise ValueError(
                f"ratings: missing required columns: {sorted(missing)}"
            )

        positives = ratings.loc[
            ratings["rating"] >= 4.0,
            ["userId", "movieId", "rating"],
        ].copy()

        if positives.empty:
            raise ValueError(
                "No positive interactions found for BPR training."
            )

        positives["userId"] = (
            pd.to_numeric(positives["userId"])
            .astype(int)
        )

        positives["movieId"] = (
            pd.to_numeric(positives["movieId"])
            .astype(int)
        )

        return positives

    def fit(self, ratings: pd.DataFrame) -> "BPRRecommender":
        """
        Train the BPR model.

        Positive interactions are ratings >= 4.0.
        Negative items are sampled from items the user
        has not positively interacted with.
        """

        positives = self._prepare_data(ratings)

        random.seed(self.random_seed)
        np.random.seed(self.random_seed)
        torch.manual_seed(self.random_seed)

        users = sorted(ratings["userId"].unique())
        items = sorted(ratings["movieId"].unique())
        if not items:
               raise ValueError("No movies found in training ratings.")
        self.user_to_index = {
            user_id: index
            for index, user_id in enumerate(users)
        }

        self.item_to_index = {
            movie_id: index
            for index, movie_id in enumerate(items)
        }

        self.index_to_user = {
            index: user_id
            for user_id, index in self.user_to_index.items()
        }

        self.index_to_item = {
            index: movie_id
            for movie_id, index in self.item_to_index.items()
        }

        self.user_positive_items = (
            positives.groupby("userId")["movieId"]
            .apply(set)
            .to_dict()
        )

        self.model = BPRMatrixFactorization(
            n_users=len(self.user_to_index),
            n_items=len(self.item_to_index),
            factors=self.factors,
        ).to(self.device)

        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.regularization,
        )

        training_samples = [
            (
                user_id,
                movie_id,
            )
            for user_id, movie_id
            in positives[["userId", "movieId"]].itertuples(
                index=False,
                name=None,
            )
        ]

        all_items = set(self.item_to_index.keys())

        self.model.train()

        batch_size = 1024

        user_indices = torch.tensor(
            [
                self.user_to_index[user_id]
                for user_id, _ in training_samples
            ],
            dtype=torch.long,
            device=self.device,
        )

        positive_indices = torch.tensor(
            [
                self.item_to_index[movie_id]
                for _, movie_id in training_samples
            ],
            dtype=torch.long,
            device=self.device,
        )

        negative_candidate_map = {
            user_id: tuple(
                all_items - self.user_positive_items[user_id]
            )
            for user_id in self.user_positive_items
        }

        n_samples = len(user_indices)

        for epoch in range(1, self.epochs + 1):

            # Resample negative items every epoch.
            negative_movie_ids = [
                random.choice(
                    negative_candidate_map[user_id]
                )
                for user_id, _ in training_samples
            ]

            negative_indices = torch.tensor(
                [
                    self.item_to_index[movie_id]
                    for movie_id in negative_movie_ids
                ],
                dtype=torch.long,
                device=self.device,
            )

            permutation = torch.randperm(
                n_samples,
                device=self.device,
            )

            total_loss = 0.0
            batch_count = 0

            for start in range(0, n_samples, batch_size):
                batch_idx = permutation[start:start + batch_size]

                batch_users = user_indices[batch_idx]
                batch_positive = positive_indices[batch_idx]
                batch_negative = negative_indices[batch_idx]

                optimizer.zero_grad()

                positive_scores, negative_scores, difference = self.model(
                    batch_users,
                    batch_positive,
                    batch_negative,
                )

                loss = -torch.nn.functional.logsigmoid(
                    difference
                ).mean()

                loss.backward()
                optimizer.step()

                total_loss += float(loss.item())
                batch_count += 1

            average_loss = (
                total_loss / batch_count
                if batch_count
                else 0.0
            )

            print(
                f"Epoch {epoch:02d}/{self.epochs} "
                f"- loss: {average_loss:.6f}"
            )

    def _score_all_items(
        self,
        user_id: int,
    ) -> pd.Series:
        """Score every known item for a user."""

        if self.model is None:
            raise RuntimeError(
                "Model has not been fitted. Call fit() first."
            )

        if user_id not in self.user_to_index:
            return pd.Series(dtype=float)

        user_index = self.user_to_index[user_id]

        item_indices = torch.arange(
            len(self.item_to_index),
            dtype=torch.long,
            device=self.device,
        )

        user_indices = torch.full(
            (len(self.item_to_index),),
            user_index,
            dtype=torch.long,
            device=self.device,
        )

        self.model.eval()

        with torch.no_grad():
            scores = self.model.score(
                user_indices,
                item_indices,
            )

        scores = scores.cpu().numpy()

        return pd.Series(
            scores,
            index=[
                self.index_to_item[index]
                for index in range(len(self.item_to_index))
            ],
            dtype=float,
        )

    def recommend(
        self,
        user_id: int,
        n: int = 10,
        exclude_movies: set[int] | None = None,
    ) -> pd.DataFrame:
        """Return top-N personalized recommendations."""

        if n <= 0:
            raise ValueError("n must be greater than zero.")

        scores = self._score_all_items(user_id)

        if scores.empty:
            return pd.DataFrame(
                columns=["movieId", "score"]
            )

        already_positive = self.user_positive_items.get(
            user_id,
            set(),
        )

        excluded = set(already_positive)

        if exclude_movies:
            excluded.update(exclude_movies)

        scores = scores[
            ~scores.index.isin(excluded)
        ]

        recommendations = (
            scores
            .sort_values(ascending=False)
            .head(n)
            .rename("score")
            .reset_index()
        )

        recommendations.columns = [
            "movieId",
            "score",
        ]

        return recommendations