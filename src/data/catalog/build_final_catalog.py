
from __future__ import annotations

from pathlib import Path

import pandas as pd


# ============================================================
# File paths
# ============================================================

BASE_CATALOG_PATH = Path(
    "data/catalog/movielens_catalog_base.csv"
)

TMDB_CACHE_PATH = Path(
    "data/catalog/tmdb_enrichment_cache.csv"
)

IDENTITY_REPAIR_PATH = Path(
    "data/catalog/tmdb_identity_repair.csv"
)

OUTPUT_PATH = Path(
    "data/catalog/movielens_catalog_final.csv"
)


# ============================================================
# Columns obtained from verified TMDB metadata
# ============================================================

TMDB_METADATA_COLUMNS = [
    "original_title",
    "release_date",
    "original_language",
    "language",
    "country",
    "available_languages",
    "dubbed_languages",
    "overview",
    "keywords",
    "runtime",
    "director",
    "cast",
    "vote_average",
    "vote_count",
    "popularity",
]

SUCCESS_STATUSES = {
    "resolved",
    "resolved_via_title_search",
}

UNRESOLVED_STATUSES = {
    "not_found",
    "wrong_media_type_tv",
    "retryable_error",
    "missing_imdb_id",
}


# ============================================================
# IMDb ID normalization
# ============================================================

def normalize_imdb_id(value: object) -> str:
    """Convert IMDb IDs into canonical ttXXXXXXXX format."""

    if pd.isna(value):
        return ""

    if isinstance(value, float):
        if value.is_integer():
            value = str(int(value))
        else:
            value = str(value)
    else:
        value = str(value).strip()

    if not value:
        return ""

    if value.lower().startswith("tt"):
        value = value[2:]

    if value.endswith(".0"):
        value = value[:-2]

    if not value.isdigit():
        return ""

    return f"tt{value.zfill(7)}"


# ============================================================
# File validation
# ============================================================

def validate_files() -> None:
    """Check that all required input files exist."""

    paths = [
        BASE_CATALOG_PATH,
        TMDB_CACHE_PATH,
        IDENTITY_REPAIR_PATH,
    ]

    for path in paths:
        if not path.exists():
            raise FileNotFoundError(
                f"Required file not found: {path}"
            )


# ============================================================
# Main catalog builder
# ============================================================

def build_final_catalog() -> pd.DataFrame:
    """
    Build the final MovieLens + verified TMDB catalog.

    Rules:
    1. Preserve every original MovieLens movie.
    2. Preserve only identity-safe TMDB mappings.
    3. Apply successful identity repairs.
    4. Clear unresolved or incorrect TMDB mappings.
    5. Attach TMDB metadata only when IMDb IDs agree.
    6. Never infer dubbed or available languages.
    7. Preserve original base catalog and cache files.
    """

    validate_files()

    print("=" * 65)
    print("BUILDING FINAL MOVIE CATALOG")
    print("=" * 65)

    # --------------------------------------------------------
    # Step 1: Load input data
    # --------------------------------------------------------

    base = pd.read_csv(
        BASE_CATALOG_PATH,
        low_memory=False,
    )

    cache = pd.read_csv(
        TMDB_CACHE_PATH,
        low_memory=False,
    )

    repairs = pd.read_csv(
        IDENTITY_REPAIR_PATH,
        low_memory=False,
    )

    print(f"\nBase movies: {len(base):,}")
    print(f"TMDB cache records: {len(cache):,}")
    print(f"Repair records: {len(repairs):,}")

    # --------------------------------------------------------
    # Step 2: Validate required columns
    # --------------------------------------------------------

    required_base = {
        "movie_id",
        "tmdb_id",
        "imdb_id",
        "title",
        "release_year",
    }

    required_cache = {
        "tmdb_id",
        "imdb_id",
        *TMDB_METADATA_COLUMNS,
    }

    required_repairs = {
        "movie_id",
        "imdb_id",
        "resolved_tmdb_id",
        "resolution_status",
    }

    if required_base - set(base.columns):
        raise ValueError(
            f"Base catalog missing columns: "
            f"{sorted(required_base - set(base.columns))}"
        )

    if required_cache - set(cache.columns):
        raise ValueError(
            f"TMDB cache missing columns: "
            f"{sorted(required_cache - set(cache.columns))}"
        )

    if required_repairs - set(repairs.columns):
        raise ValueError(
            f"Repair file missing columns: "
            f"{sorted(required_repairs - set(repairs.columns))}"
        )

    # --------------------------------------------------------
    # Step 3: Check uniqueness
    # --------------------------------------------------------

    if base["movie_id"].duplicated().any():
        raise ValueError(
            "Duplicate movie_id values found in base catalog."
        )

    if repairs["movie_id"].duplicated().any():
        raise ValueError(
            "Duplicate movie_id values found in repair file."
        )

    base["movie_id"] = pd.to_numeric(
        base["movie_id"],
        errors="raise",
    ).astype(int)

    base["tmdb_id"] = pd.to_numeric(
        base["tmdb_id"],
        errors="coerce",
    )

    cache["tmdb_id"] = pd.to_numeric(
        cache["tmdb_id"],
        errors="coerce",
    )

    repairs["movie_id"] = pd.to_numeric(
        repairs["movie_id"],
        errors="raise",
    ).astype(int)

    repairs["resolved_tmdb_id"] = pd.to_numeric(
        repairs["resolved_tmdb_id"],
        errors="coerce",
    )

    # Cache must have at most one record per TMDB ID.
    cache = cache.dropna(
        subset=["tmdb_id"]
    ).copy()

    if cache["tmdb_id"].duplicated().any():
        raise ValueError(
            "Duplicate TMDB IDs found in enrichment cache."
        )

    cache["tmdb_id"] = cache["tmdb_id"].astype(int)

    # --------------------------------------------------------
    # Step 4: Normalize IMDb identities
    # --------------------------------------------------------

    base["imdb_id"] = (
        base["imdb_id"]
        .map(normalize_imdb_id)
        .replace("", pd.NA)
    )

    cache["imdb_id"] = (
        cache["imdb_id"]
        .map(normalize_imdb_id)
        .replace("", pd.NA)
    )

    repairs["imdb_id"] = (
        repairs["imdb_id"]
        .map(normalize_imdb_id)
        .replace("", pd.NA)
    )

    base_imdb = (
        base["imdb_id"]
        .fillna("")
    )

    cache_imdb_by_id = (
        cache.set_index("tmdb_id")["imdb_id"]
        .fillna("")
        .to_dict()
    )

    # --------------------------------------------------------
    # Step 5: Identify existing incorrect mappings
    # --------------------------------------------------------

    original_tmdb = base["tmdb_id"].copy()

    cached_imdb_for_base = (
        original_tmdb.map(cache_imdb_by_id)
        .fillna("")
    )

    existing_cache_mismatch = (
        original_tmdb.notna()
        & cached_imdb_for_base.ne("")
        & cached_imdb_for_base.ne(base_imdb)
    )

    mismatched_movie_ids = set(
        base.loc[
            existing_cache_mismatch,
            "movie_id",
        ].astype(int)
    )

    repair_movie_ids = set(
        repairs["movie_id"].astype(int)
    )

    missing_repairs = (
        mismatched_movie_ids - repair_movie_ids
    )

    if missing_repairs:
        raise RuntimeError(
            "Some existing TMDB identity mismatches "
            "have no repair record: "
            f"{sorted(missing_repairs)[:20]}"
        )

    # --------------------------------------------------------
    # Step 6: Apply identity repair results
    # --------------------------------------------------------

    repair_status_by_movie = dict(
        zip(
            repairs["movie_id"],
            repairs["resolution_status"],
        )
    )

    base_index_by_movie = dict(
        zip(
            base["movie_id"],
            base.index,
        )
    )

    successful_repairs = 0
    unresolved_repairs = 0

    for row in repairs.itertuples(index=False):

        movie_id = int(row.movie_id)

        if movie_id not in base_index_by_movie:
            raise ValueError(
                f"Repair movie_id not found in base: {movie_id}"
            )

        index = base_index_by_movie[movie_id]
        status = str(row.resolution_status)

        if status in SUCCESS_STATUSES:

            if pd.isna(row.resolved_tmdb_id):
                raise ValueError(
                    f"Resolved movie {movie_id} has no TMDB ID."
                )

            base.at[index, "tmdb_id"] = int(
                row.resolved_tmdb_id
            )

            successful_repairs += 1

        elif status in UNRESOLVED_STATUSES:

            # Remove the old or incorrect TMDB mapping.
            # Keep the original MovieLens movie record.
            base.at[index, "tmdb_id"] = pd.NA

            unresolved_repairs += 1

        else:
            raise ValueError(
                f"Unexpected repair status for "
                f"movie {movie_id}: {status}"
            )

    print("\nIdentity repair application:")
    print(f"Successful repairs: {successful_repairs}")
    print(f"Unresolved repairs: {unresolved_repairs}")

    # --------------------------------------------------------
    # Step 7: Validate final TMDB mappings against cache
    # --------------------------------------------------------

    base["tmdb_id"] = pd.to_numeric(
        base["tmdb_id"],
        errors="coerce",
    )

    final_cached_imdb = (
        base["tmdb_id"]
        .map(cache_imdb_by_id)
        .fillna("")
    )

    expected_imdb = base["imdb_id"].fillna("")

    has_cached_metadata = (
        base["tmdb_id"].notna()
        & final_cached_imdb.ne("")
    )

    metadata_identity_mismatch = (
        has_cached_metadata
        & (
            final_cached_imdb.ne(expected_imdb)
            | expected_imdb.eq("")
        )
    )

    if metadata_identity_mismatch.any():

        bad_movies = base.loc[
            metadata_identity_mismatch,
            [
                "movie_id",
                "tmdb_id",
                "imdb_id",
            ],
        ]

        raise RuntimeError(
            "Identity validation failed. "
            "Incorrect TMDB metadata would otherwise be attached:\n"
            f"{bad_movies.head(20).to_string(index=False)}"
        )

    metadata_verified = (
        has_cached_metadata
        & final_cached_imdb.eq(expected_imdb)
        & expected_imdb.ne("")
    )

    # Successful repaired IDs must have verified metadata.
    successful_movie_mask = (
        base["movie_id"].isin(
            repairs.loc[
                repairs["resolution_status"].isin(
                    SUCCESS_STATUSES
                ),
                "movie_id",
            ]
        )
    )

    missing_repaired_metadata = (
        successful_movie_mask
        & ~metadata_verified
    )

    if missing_repaired_metadata.any():
        raise RuntimeError(
            "Some successful repairs have no verified "
            "metadata in the TMDB cache: "
            f"{base.loc[missing_repaired_metadata, 'movie_id'].tolist()[:20]}"
        )

    # --------------------------------------------------------
    # Step 8: Attach verified TMDB metadata
    # --------------------------------------------------------

    cache_lookup = cache.set_index("tmdb_id")

    for column in TMDB_METADATA_COLUMNS:

        mapped_values = (
            base["tmdb_id"]
            .map(cache_lookup[column])
        )

        # Only attach metadata after IMDb identity verification.
        mapped_values = mapped_values.where(
            metadata_verified
        )

        # Verified TMDB metadata takes priority.
        base[column] = mapped_values.combine_first(
            base[column]
        )

    # --------------------------------------------------------
    # Step 9: Add catalog quality indicators
    # --------------------------------------------------------

    base["tmdb_metadata_available"] = (
        metadata_verified.astype(bool)
    )

    base["tmdb_identity_status"] = (
        "unmapped"
    )

    base.loc[
        base["tmdb_id"].notna(),
        "tmdb_identity_status",
    ] = "metadata_unavailable"

    base.loc[
        metadata_verified,
        "tmdb_identity_status",
    ] = "verified_existing"

    base.loc[
        successful_movie_mask,
        "tmdb_identity_status",
    ] = "repaired_verified"

    base.loc[
        base["movie_id"].map(repair_status_by_movie)
        .eq("not_found"),
        "tmdb_identity_status",
    ] = "unresolved"

    base.loc[
        base["movie_id"].map(repair_status_by_movie)
        .eq("wrong_media_type_tv"),
        "tmdb_identity_status",
    ] = "unsupported_media_type_tv"

    # Store TMDB IDs as nullable integers.
    base["tmdb_id"] = base["tmdb_id"].round().astype("Int64")

    # --------------------------------------------------------
    # Step 10: Final data validation
    # --------------------------------------------------------

    if len(base) != 87585:
        raise RuntimeError(
            f"Expected 87,585 movies, got {len(base):,}."
        )

    if base["movie_id"].duplicated().any():
        raise RuntimeError(
            "Duplicate movie IDs detected after integration."
        )

    if (
        base.loc[
            base["tmdb_metadata_available"],
            "tmdb_id",
        ].isna().any()
    ):
        raise RuntimeError(
            "A record claims to have TMDB metadata "
            "but has no TMDB ID."
        )

    # --------------------------------------------------------
    # Step 11: Save final catalog
    # --------------------------------------------------------

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_path = OUTPUT_PATH.with_name(
        OUTPUT_PATH.stem + ".tmp.csv"
    )

    base.to_csv(
        temporary_path,
        index=False,
        encoding="utf-8",
    )

    temporary_path.replace(
        OUTPUT_PATH
    )

    # --------------------------------------------------------
    # Step 12: Final report
    # --------------------------------------------------------

    print("\n" + "=" * 65)
    print("FINAL CATALOG CREATED")
    print("=" * 65)

    print(f"Total movies: {len(base):,}")

    print(
        "Unique MovieLens IDs: "
        f"{base['movie_id'].nunique():,}"
    )

    print(
        "Movies with TMDB IDs: "
        f"{base['tmdb_id'].notna().sum():,}"
    )

    print(
        "Movies with verified TMDB metadata: "
        f"{base['tmdb_metadata_available'].sum():,}"
    )

    print(
        "Movies without TMDB metadata: "
        f"{(~base['tmdb_metadata_available']).sum():,}"
    )

    print(
        "Successful repaired identities: "
        f"{(base['tmdb_identity_status'] == 'repaired_verified').sum():,}"
    )

    print("\nIdentity status:")
    print(
        base["tmdb_identity_status"]
        .value_counts()
        .to_string()
    )

    print(
        f"\nSaved to: {OUTPUT_PATH}"
    )

    return base


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    build_final_catalog()
