import time
import numpy as np
import pandas as pd

from src.ranking.feature_builder import RankingFeatureBuilder

RATINGS_PATH = "data/ratings.csv"
MOVIES_PATH = "data/movies.csv"

ratings = pd.read_csv(RATINGS_PATH)
movies = pd.read_csv(MOVIES_PATH)

# Build a realistic candidate pool:
# 20 users × 500 candidate movies = 10,000 rows
users = ratings["userId"].drop_duplicates().head(20).tolist()
movie_ids = movies["movieId"].drop_duplicates().head(500).tolist()

rows = []
for user_id in users:
    for movie_id in movie_ids:
        rows.append(
            {
                "userId": user_id,
                "movieId": movie_id,
                "hybrid_candidate_score": np.random.random(),
                "bpr_score": np.random.random(),
                "content_score": np.random.random(),
                "semantic_score": np.random.random(),
                "popularity_score": np.random.random(),
                "sources": "bpr,content,semantic",
            }
        )

candidates = pd.DataFrame(rows)

builder = RankingFeatureBuilder()

# Warm-up
builder.build(candidates, ratings, movies)

times = []

for run in range(3):
    start = time.perf_counter()
    result = builder.build(candidates, ratings, movies)
    elapsed = time.perf_counter() - start
    times.append(elapsed)
    print(f"Run {run + 1}: {elapsed:.4f} seconds")

print()
print(f"Candidates: {len(candidates):,}")
print(f"Output rows: {len(result):,}")
print(f"Output columns: {len(result.columns)}")
print(f"Median time: {np.median(times):.4f} seconds")
print(f"Rows/second: {len(candidates) / np.median(times):,.0f}")