
from pathlib import Path

import pandas as pd


SOURCE_PATH = Path(
    "data/catalog/movielens_catalog_final.csv"
)

OUTPUT_PATH = Path(
    "data/catalog/movielens_catalog_runtime.csv.gz"
)

SOURCE_COLUMNS = [
    "movie_id",
    "tmdb_id",
    "imdb_id",
    "title",
    "release_year",
    "genres",
    "tags",
    "tmdb_metadata_available",
]

RUNTIME_COLUMNS = [
    "movie_id",
    "tmdb_id",
    "imdb_id",
    "title",
    "release_year",
    "genres",
    "tags",
]


def build_runtime_catalog() -> pd.DataFrame:

    if not SOURCE_PATH.exists():
        raise FileNotFoundError(
            f"Final catalog not found: {SOURCE_PATH}"
        )

    print("Loading verified final catalog...")

    catalog = pd.read_csv(
        SOURCE_PATH,
        usecols=SOURCE_COLUMNS,
        low_memory=False,
    )

    if len(catalog) != 87585:
        raise ValueError(
            f"Expected 87,585 movies, got {len(catalog)}."
        )

    if catalog["movie_id"].duplicated().any():
        raise ValueError("Duplicate movie IDs found.")

    catalog["movie_id"] = pd.to_numeric(
        catalog["movie_id"],
        errors="raise",
    ).astype(int)

    catalog["tmdb_id"] = pd.to_numeric(
        catalog["tmdb_id"],
        errors="coerce",
    )

    catalog["release_year"] = pd.to_numeric(
        catalog["release_year"],
        errors="coerce",
    )

    # Only retain TMDB IDs that passed identity validation
    # and have verified metadata in the final catalog.
    verified = (
        catalog["tmdb_metadata_available"]
        .fillna(False)
        .astype(bool)
    )

    catalog.loc[~verified, "tmdb_id"] = pd.NA

    catalog["genres"] = (
        catalog["genres"]
        .fillna("")
        .astype(str)
    )

    catalog["tags"] = (
        catalog["tags"]
        .fillna("")
        .astype(str)
    )

    catalog = catalog[RUNTIME_COLUMNS].copy()

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("Creating compressed runtime catalog...")

    catalog.to_csv(
        OUTPUT_PATH,
        index=False,
        encoding="utf-8",
        compression={
            "method": "gzip",
            "compresslevel": 6,
        },
    )

    size_mb = (
        OUTPUT_PATH.stat().st_size
        / (1024 ** 2)
    )

    verified_tmdb_count = (
        catalog["tmdb_id"].notna().sum()
    )

    print("\nRUNTIME CATALOG CREATED")
    print("-----------------------")
    print("Total movies:", len(catalog))
    print("Columns:", len(catalog.columns))
    print("Verified TMDB IDs:", verified_tmdb_count)
    print(
        "Movies without verified TMDB IDs:",
        len(catalog) - verified_tmdb_count,
    )
    print(f"Compressed size: {size_mb:.2f} MB")
    print("Output:", OUTPUT_PATH)

    return catalog


if __name__ == "__main__":
    build_runtime_catalog()
