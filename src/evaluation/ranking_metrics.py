import math
from collections.abc import Iterable

import numpy as np
import pandas as pd


def precision_at_k(
    recommended_items: Iterable[int],
    relevant_items: set[int],
    k: int,
) -> float:
    """
    Precision@K.

    Measures the fraction of the top-K recommendations
    that are relevant.
    """
    if k <= 0:
        raise ValueError("k must be greater than zero.")

    recommendations = list(recommended_items)[:k]

    if not recommendations:
        return 0.0

    hits = sum(item in relevant_items for item in recommendations)

    return hits / len(recommendations)


def recall_at_k(
    recommended_items: Iterable[int],
    relevant_items: set[int],
    k: int,
) -> float:
    """
    Recall@K.

    Measures the fraction of relevant test items
    that appear in the top-K recommendations.
    """
    if k <= 0:
        raise ValueError("k must be greater than zero.")

    if not relevant_items:
        return 0.0

    recommendations = list(recommended_items)[:k]

    hits = sum(item in relevant_items for item in recommendations)

    return hits / len(relevant_items)


def ndcg_at_k(
    recommended_items: Iterable[int],
    relevant_items: set[int],
    k: int,
) -> float:
    """
    NDCG@K using binary relevance.

    Relevant items contribute 1.
    Non-relevant items contribute 0.
    Higher scores are better.
    """
    if k <= 0:
        raise ValueError("k must be greater than zero.")

    recommendations = list(recommended_items)[:k]

    if not relevant_items or not recommendations:
        return 0.0

    dcg = 0.0

    for rank, item in enumerate(recommendations, start=1):
        if item in relevant_items:
            dcg += 1.0 / math.log2(rank + 1)

    ideal_hits = min(len(relevant_items), k)

    idcg = sum(
        1.0 / math.log2(rank + 1)
        for rank in range(1, ideal_hits + 1)
    )

    if idcg == 0.0:
        return 0.0

    return dcg / idcg


def catalog_coverage(
    recommended_items: Iterable[int],
    catalog_size: int,
) -> float:
    """
    Catalog coverage.

    Measures what fraction of the eligible catalog
    appears in the recommendation output.
    """
    if catalog_size <= 0:
        raise ValueError("catalog_size must be greater than zero.")

    unique_recommended = set(recommended_items)

    return len(unique_recommended) / catalog_size


def intra_list_diversity(
    recommended_items: Iterable[int],
    item_features: pd.DataFrame,
) -> float:
    """
    Calculate average pairwise cosine distance between
    recommended item feature vectors.

    Expected:
        item_features.index -> movie IDs
        item_features.values -> feature vectors

    A value closer to 1 means greater diversity.
    """
    items = list(dict.fromkeys(recommended_items))

    if len(items) < 2:
        return 0.0

    available_items = [
        item for item in items
        if item in item_features.index
    ]

    if len(available_items) < 2:
        return 0.0

    matrix = item_features.loc[available_items].to_numpy(dtype=float)

    norms = np.linalg.norm(matrix, axis=1, keepdims=True)

    # Prevent division by zero for zero vectors.
    normalized = np.divide(
        matrix,
        norms,
        out=np.zeros_like(matrix),
        where=norms != 0,
    )

    similarity_matrix = normalized @ normalized.T

    upper_triangle = np.triu_indices(
        len(available_items),
        k=1,
    )

    similarities = similarity_matrix[upper_triangle]

    distances = 1.0 - similarities

    return float(np.mean(distances))


def novelty(
    recommended_items: Iterable[int],
    item_popularity: pd.Series,
) -> float:
    """
    Average novelty of recommended items.

    Novelty is based on item popularity:
        - More popular item -> lower novelty
        - Less popular item -> higher novelty

    item_popularity should contain interaction counts,
    indexed by movieId.
    """
    items = list(dict.fromkeys(recommended_items))

    if not items:
        return 0.0

    popularity = item_popularity.reindex(items).fillna(0).astype(float)

    total_interactions = float(item_popularity.sum())

    if total_interactions <= 0:
        return 0.0

    probabilities = popularity / total_interactions

    novelty_scores = []

    for probability in probabilities:
        if probability > 0:
            novelty_scores.append(-math.log2(probability))
        else:
            # Completely unseen items are maximally novel.
            novelty_scores.append(
                -math.log2(1.0 / total_interactions)
            )

    return float(np.mean(novelty_scores))


def evaluate_user(
    recommended_items: Iterable[int],
    relevant_items: set[int],
    k: int,
    item_features: pd.DataFrame | None = None,
    item_popularity: pd.Series | None = None,
) -> dict[str, float]:
    """
    Evaluate one user's recommendation list.
    """
    recommendations = list(recommended_items)[:k]

    result = {
        f"precision@{k}": precision_at_k(
            recommendations,
            relevant_items,
            k,
        ),
        f"recall@{k}": recall_at_k(
            recommendations,
            relevant_items,
            k,
        ),
        f"ndcg@{k}": ndcg_at_k(
            recommendations,
            relevant_items,
            k,
        ),
    }

    if item_features is not None:
        result["intra_list_diversity"] = intra_list_diversity(
            recommendations,
            item_features,
        )

    if item_popularity is not None:
        result["novelty"] = novelty(
            recommendations,
            item_popularity,
        )

    return result


def aggregate_user_metrics(
    user_results: list[dict[str, float]],
) -> dict[str, float]:
    """
    Average metrics across users.
    """
    if not user_results:
        return {}

    metric_names = user_results[0].keys()

    return {
        metric: float(
            np.mean(
                [
                    result[metric]
                    for result in user_results
                    if metric in result
                ]
            )
        )
        for metric in metric_names
    }