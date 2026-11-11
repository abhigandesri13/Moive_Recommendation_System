
from __future__ import annotations

import re
import time

from datetime import datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path

import pandas as pd

from src.services.tmdb_client import TMDBClient


# ============================================================
# Configuration
# ============================================================

DEFAULT_CATALOG_PATH = Path(
    "data/catalog/movielens_catalog_base.csv"
)

DEFAULT_CACHE_PATH = Path(
    "data/catalog/tmdb_enrichment_cache.csv"
)

DEFAULT_OUTPUT_PATH = Path(
    "data/catalog/tmdb_identity_repair.csv"
)

TITLE_SIMILARITY_THRESHOLD = 0.80


# ============================================================
# IMDb ID normalization
# ============================================================

def normalize_imdb_id(value: object) -> str:
    """
    Normalize an IMDb ID to canonical ttXXXXXXXX form.

    Handles:
    - Numeric IMDb IDs
    - IDs already starting with tt
    - Floating-point values read from CSV
    - Missing leading zeros
    """

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

    # Preserve existing digits and restore leading zeros.
    value = value.zfill(7)

    return f"tt{value}"


# ============================================================
# Movie title normalization
# ============================================================

def _remove_release_year(title: object) -> str:
    """Remove a trailing four-digit year from a movie title."""

    if pd.isna(title):
        return ""

    return re.sub(
        r"\s*\(\d{4}\)\s*$",
        "",
        str(title).strip(),
    )


def normalize_title(value: object) -> str:
    """
    Normalize movie titles for conservative comparison.

    Handles:
    - MovieLens trailing release years
    - Article placement (Example, The -> The Example)
    - Punctuation and whitespace
    """

    text = _remove_release_year(value)

    if not text:
        return ""

    article_match = re.search(
        r",\s*(the|a|an)$",
        text,
        flags=re.IGNORECASE,
    )

    if article_match:
        article = article_match.group(1).lower()
        title_without_article = text[:article_match.start()].strip()

        text = f"{article} {title_without_article}"

    text = text.lower()

    text = re.sub(
        r"[^a-z0-9]+",
        " ",
        text,
    )

    return " ".join(text.split())


def title_similarity(
    left: object,
    right: object,
) -> float:
    """Calculate normalized title similarity."""

    left_normalized = normalize_title(left)
    right_normalized = normalize_title(right)

    if not left_normalized or not right_normalized:
        return 0.0

    return SequenceMatcher(
        None,
        left_normalized,
        right_normalized,
    ).ratio()


def _prepare_title_search_query(value: object) -> str:
    """
    Prepare a MovieLens title for TMDB title search.

    Removes the MovieLens release year and moves trailing
    articles to the beginning of the title.
    """

    text = _remove_release_year(value)

    if not text:
        return ""

    article_match = re.search(
        r",\s*(the|a|an)$",
        text,
        flags=re.IGNORECASE,
    )

    if article_match:
        article = article_match.group(1)
        title_without_article = text[:article_match.start()].strip()

        text = f"{article} {title_without_article}"

    return text.strip()


# ============================================================
# Candidate generation
# ============================================================

def build_identity_repair_candidates(
    catalog: pd.DataFrame,
    cache: pd.DataFrame,
) -> pd.DataFrame:
    """
    Identify MovieLens records requiring identity repair.

    A record becomes a repair candidate when:

    1. Its existing TMDB ID has cached metadata, but the
       cached IMDb ID disagrees with the MovieLens IMDb ID.

    OR

    2. Its TMDB ID is missing.

    The function does not modify either input DataFrame.
    """

    required_catalog_columns = {
        "movie_id",
        "tmdb_id",
        "imdb_id",
        "title",
        "release_year",
    }

    required_cache_columns = {
        "tmdb_id",
        "imdb_id",
    }

    missing_catalog = (
        required_catalog_columns - set(catalog.columns)
    )

    missing_cache = (
        required_cache_columns - set(cache.columns)
    )

    if missing_catalog:
        raise ValueError(
            "Catalog is missing columns: "
            f"{sorted(missing_catalog)}"
        )

    if missing_cache:
        raise ValueError(
            "Cache is missing columns: "
            f"{sorted(missing_cache)}"
        )

    catalog_df = catalog.copy()
    cache_df = cache.copy()

    # Normalize identifiers.
    catalog_df["tmdb_id"] = pd.to_numeric(
        catalog_df["tmdb_id"],
        errors="coerce",
    )

    cache_df["tmdb_id"] = pd.to_numeric(
        cache_df["tmdb_id"],
        errors="coerce",
    )

    catalog_df["imdb_norm"] = (
        catalog_df["imdb_id"]
        .map(normalize_imdb_id)
    )

    cache_identity = cache_df[
        ["tmdb_id", "imdb_id"]
    ].copy()

    cache_identity = cache_identity.rename(
        columns={
            "imdb_id": "cache_imdb_id",
        }
    )

    cache_identity["cache_imdb_norm"] = (
        cache_identity["cache_imdb_id"]
        .map(normalize_imdb_id)
    )

    # Null TMDB IDs must not participate in the cache join.
    cache_identity = cache_identity.dropna(
        subset=["tmdb_id"]
    )

    # A TMDB ID should identify one cached movie.
    cache_identity = cache_identity.drop_duplicates(
        subset=["tmdb_id"],
        keep="last",
    )

    merged = catalog_df.merge(
        cache_identity[
            [
                "tmdb_id",
                "cache_imdb_id",
                "cache_imdb_norm",
            ]
        ],
        on="tmdb_id",
        how="left",
        validate="many_to_one",
    )

    # Cached identity differs from the MovieLens identity.
    has_cached_metadata = (
        merged["cache_imdb_norm"].notna()
        & merged["cache_imdb_norm"].ne("")
    )

    cached_identity_mismatch = (
        has_cached_metadata
        & (
            merged["imdb_norm"]
            != merged["cache_imdb_norm"]
        )
    )

    # MovieLens does not have a TMDB mapping.
    missing_tmdb_mapping = (
        merged["tmdb_id"].isna()
    )

    repair_mask = (
        cached_identity_mismatch
        | missing_tmdb_mapping
    )

    candidates = merged.loc[
        repair_mask
    ].copy()

    candidates["repair_reason"] = (
        "identity_mismatch"
    )

    candidates.loc[
        missing_tmdb_mapping.loc[candidates.index],
        "repair_reason",
    ] = "missing_tmdb_id"

    output_columns = [
        "movie_id",
        "tmdb_id",
        "imdb_id",
        "title",
        "release_year",
        "imdb_norm",
        "repair_reason",
    ]

    candidates = candidates[
        output_columns
    ].copy()

    candidates["movie_id"] = pd.to_numeric(
        candidates["movie_id"],
        errors="raise",
    ).astype(int)

    candidates = candidates.sort_values(
        "movie_id"
    ).reset_index(drop=True)

    return candidates


def build_repair_candidates_from_catalog(
    catalog_path: str | Path = DEFAULT_CATALOG_PATH,
    cache_path: str | Path = DEFAULT_CACHE_PATH,
) -> pd.DataFrame:
    """Load catalog and cache, then generate repair candidates."""

    catalog_path = Path(catalog_path)
    cache_path = Path(cache_path)

    if not catalog_path.exists():
        raise FileNotFoundError(
            f"Catalog file not found: {catalog_path}"
        )

    if not cache_path.exists():
        raise FileNotFoundError(
            f"TMDB enrichment cache not found: {cache_path}"
        )

    catalog = pd.read_csv(
        catalog_path,
        low_memory=False,
    )

    cache = pd.read_csv(
        cache_path,
        low_memory=False,
    )

    return build_identity_repair_candidates(
        catalog=catalog,
        cache=cache,
    )


# ============================================================
# TMDB identity repair
# ============================================================

class TMDBIdentityRepair:
    """
    Resolve incorrect or missing TMDB mappings.

    Resolution order:

    1. IMDb ID lookup through TMDB /find.
    2. If IMDb lookup returns no movie, search by title.
    3. Require an exact release-year match.
    4. Require title similarity >= configured threshold.

    Every processed result is checkpointed immediately.
    """

    def __init__(
        self,
        client: TMDBClient | None = None,
        delay_seconds: float = 0.15,
    ) -> None:

        if delay_seconds < 0:
            raise ValueError(
                "delay_seconds cannot be negative."
            )

        self._owns_client = client is None

        self.client = (
            client
            if client is not None
            else TMDBClient()
        )

        self.delay_seconds = delay_seconds

    # --------------------------------------------------------
    # Close client
    # --------------------------------------------------------

    def close(self) -> None:
        """Close the TMDB client if this class created it."""

        if self._owns_client:
            self.client.close()

    # --------------------------------------------------------
    # Completed movie IDs
    # --------------------------------------------------------

    @staticmethod
    def load_completed_movie_ids(
        output_path: str | Path,
    ) -> set[int]:
        """Load previously processed MovieLens IDs."""

        output_path = Path(output_path)

        if not output_path.exists():
            return set()

        existing = pd.read_csv(
            output_path,
            usecols=["movie_id"],
        )

        completed_ids = pd.to_numeric(
            existing["movie_id"],
            errors="coerce",
        ).dropna()

        return set(
            completed_ids.astype(int).tolist()
        )

    # --------------------------------------------------------
    # Title-search fallback
    # --------------------------------------------------------


    def _resolve_by_title_search(
        self,
        title: object,
        release_year: object,
        imdb_id: str,
    ) -> dict | None:
        """
        Search by title and year, then verify the actual IMDb ID.

        A title/year match alone is not sufficient to establish
        movie identity.
        """

        if pd.isna(release_year):
            return None

        try:
            source_year = int(release_year)
        except (TypeError, ValueError, OverflowError):
            return None

        expected_imdb_id = normalize_imdb_id(imdb_id)

        if not expected_imdb_id:
            return None

        search_query = _prepare_title_search_query(title)

        if not search_query:
            return None

        search_data = self.client.search_movies(
            search_query,
            primary_release_year=source_year,
        )

        search_results = search_data.get("results", [])

        eligible_candidates = []

        for candidate in search_results:
            if not isinstance(candidate, dict):
                continue

            release_date = candidate.get("release_date") or ""

            try:
                candidate_year = int(release_date[:4])
            except (ValueError, TypeError):
                continue

            if candidate_year != source_year:
                continue

            similarity = title_similarity(
                title,
                candidate.get("title"),
            )

            if similarity < TITLE_SIMILARITY_THRESHOLD:
                continue

            candidate_id = candidate.get("id")

            if candidate_id is None:
                continue

            eligible_candidates.append(
                (similarity, int(candidate_id))
            )

        eligible_candidates.sort(
            key=lambda item: item[0],
            reverse=True,
        )

        # Verify the actual IMDb ID of each candidate.
        for similarity, tmdb_id in eligible_candidates:

            details = self.client.get_movie_details(
                tmdb_id
            )

            actual_imdb_id = normalize_imdb_id(
                details.get("imdb_id")
            )

            if actual_imdb_id == expected_imdb_id:
                return details

        return None


    # --------------------------------------------------------
    # Main repair
    # --------------------------------------------------------

    def repair(
        self,
        candidates: pd.DataFrame,
        output_path: str | Path = DEFAULT_OUTPUT_PATH,
    ) -> pd.DataFrame:
        """
        Resolve IMDb IDs and checkpoint every result.

        Existing output records are skipped, making this method
        resumable after interruptions.
        """

        required_columns = {
            "movie_id",
            "tmdb_id",
            "imdb_norm",
            "title",
            "release_year",
        }

        missing_columns = (
            required_columns - set(candidates.columns)
        )

        if missing_columns:
            raise ValueError(
                "Candidates are missing columns: "
                f"{sorted(missing_columns)}"
            )

        output_path = Path(output_path)

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        completed = self.load_completed_movie_ids(
            output_path
        )

        pending = candidates[
            ~candidates["movie_id"]
            .astype(int)
            .isin(completed)
        ].copy()

        print(
            f"Candidate movies: {len(candidates):,}"
        )

        print(
            f"Already processed: {len(completed):,}"
        )

        print(
            f"Remaining: {len(pending):,}"
        )

        records = []

        for index, row in enumerate(
            pending.itertuples(index=False),
            start=1,
        ):

            movie_id = int(row.movie_id)
            imdb_id = str(row.imdb_norm).strip()

            print(
                f"[{index}/{len(pending)}] "
                f"MovieLens {movie_id} | "
                f"{row.title}"
            )

            record = {
                "movie_id": movie_id,

                "source_tmdb_id": (
                    int(row.tmdb_id)
                    if pd.notna(row.tmdb_id)
                    else None
                ),

                "imdb_id": imdb_id,

                "title": row.title,

                "release_year": row.release_year,

                "resolved_tmdb_id": None,

                "resolved_title": None,

                "resolved_release_date": None,

                "resolution_status": None,

                "timestamp_utc": datetime.now(
                    timezone.utc
                ).isoformat(),
            }

            try:

                # --------------------------------------------
                # Step 1: IMDb ID lookup
                # --------------------------------------------

                if not imdb_id:

                    record["resolution_status"] = (
                        "missing_imdb_id"
                    )

                else:

                    data = self.client.find_by_imdb_id(
                        imdb_id
                    )

                    movies = data.get(
                        "movie_results",
                        [],
                    )

                    # ----------------------------------------
                    # Step 2: IMDb lookup was successful
                    # ----------------------------------------

                    if movies:


                       verified_movie = None

                    # Verify every IMDb lookup candidate using
                    # the actual IMDb ID stored in TMDB details.
                    for candidate in movies:
                        candidate_tmdb_id = candidate.get("id")

                        if candidate_tmdb_id is None:
                            continue

                        details = self.client.get_movie_details(
                            int(candidate_tmdb_id)
                        )

                        actual_imdb_id = normalize_imdb_id(
                            details.get("imdb_id")
                        )

                        if actual_imdb_id == normalize_imdb_id(imdb_id):
                            verified_movie = details
                            break

                    if verified_movie is not None:

                        record["resolved_tmdb_id"] = (
                            verified_movie.get("id")
                        )

                        record["resolved_title"] = (
                            verified_movie.get("title")
                        )

                        record["resolved_release_date"] = (
                            verified_movie.get("release_date")
                        )

                        record["resolution_status"] = (
                            "resolved"
                        )

                    else:

                        # IMDb lookup did not produce a verified
                        # identity. Try the conservative fallback.
                        matched_movie = (
                            self._resolve_by_title_search(
                                title=row.title,
                                release_year=row.release_year,
                                imdb_id=imdb_id,
                            )
                        )

                        if matched_movie is not None:

                            record["resolved_tmdb_id"] = (
                                matched_movie.get("id")
                            )

                            record["resolved_title"] = (
                                matched_movie.get("title")
                            )

                            record["resolved_release_date"] = (
                                matched_movie.get("release_date")
                            )

                            record["resolution_status"] = (
                                "resolved_via_title_search"
                            )

                        else:

                            record["resolution_status"] = (
                                "not_found"
                            )


            except Exception as exc:

                record["resolution_status"] = (
                    "retryable_error"
                )

                print(
                    f"  Failed: "
                    f"{type(exc).__name__}: {exc}"
                )

            # --------------------------------------------
            # Step 4: Save checkpoint immediately
            # --------------------------------------------

            write_header = (
                not output_path.exists()
                or output_path.stat().st_size == 0
            )

            pd.DataFrame(
                [record]
            ).to_csv(
                output_path,
                mode="a",
                header=write_header,
                index=False,
            )

            records.append(record)

            print(
                f"  Status: {record['resolution_status']}"
            )

            if index < len(pending):
                time.sleep(self.delay_seconds)

        return pd.DataFrame(records)


# ============================================================
# Module entry point
# ============================================================

def main() -> None:
    """Generate and display identity-repair candidates."""

    candidates = build_repair_candidates_from_catalog()

    print(
        f"Identity-repair candidates: {len(candidates)}"
    )

    display_columns = [
        "movie_id",
        "tmdb_id",
        "imdb_id",
        "title",
        "release_year",
    ]

    print(
        candidates[display_columns]
        .head(20)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
