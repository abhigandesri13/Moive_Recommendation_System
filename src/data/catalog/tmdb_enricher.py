from __future__ import annotations

from datetime import datetime, timezone
import time
from pathlib import Path
from typing import Any

import pandas as pd

from src.data.catalog.tmdb_parser import parse_tmdb_movie
from src.services.tmdb_client import TMDBClient


DEFAULT_CATALOG_PATH = Path("data/catalog/movielens_catalog_base.csv")
DEFAULT_CACHE_PATH = Path("data/catalog/tmdb_enrichment_cache.csv")
DEFAULT_FAILURE_PATH = Path("data/catalog/tmdb_enrichment_failures.csv")


class TMDBEnricher:
    """Resumable TMDB metadata enrichment."""

    def __init__(
        self,
        client: TMDBClient | None = None,
        delay_seconds: float = 0.1,
    ) -> None:
        self.client = client or TMDBClient()
        self.delay_seconds = delay_seconds

    def load_completed_ids(
        self,
        cache_path: str | Path,
    ) -> set[int]:
        """Return TMDB IDs that were successfully enriched."""

        cache_path = Path(cache_path)

        if not cache_path.exists():
            return set()

        cache = pd.read_csv(cache_path)

        if "tmdb_id" not in cache.columns:
            raise ValueError(
                "Enrichment cache must contain a tmdb_id column."
            )

        return set(
            pd.to_numeric(
                cache["tmdb_id"],
                errors="coerce",
            )
            .dropna()
            .astype(int)
        )

    def load_permanent_failures(
        self,
        failure_path: str | Path,
    ) -> set[int]:
        """Return TMDB IDs that should not be retried."""

        failure_path = Path(failure_path)

        if not failure_path.exists():
            return set()

        failures = pd.read_csv(failure_path)

        if failures.empty or "tmdb_id" not in failures.columns:
            return set()

        permanent = failures[
            failures["retryable"].astype(str).str.lower() == "false"
        ]

        return set(
            pd.to_numeric(
                permanent["tmdb_id"],
                errors="coerce",
            )
            .dropna()
            .astype(int)
        )

    def _record_failure(
        self,
        tmdb_id: int,
        failure_path: Path,
        status_code: int | None,
        retryable: bool,
        error_type: str,
        message: str,
    ) -> None:
        """Append one enrichment failure to the checkpoint file."""

        record = pd.DataFrame(
            [
                {
                    "tmdb_id": tmdb_id,
                    "status_code": status_code,
                    "retryable": retryable,
                    "error_type": error_type,
                    "message": message[:500],
                    "timestamp_utc": datetime.now(
                        timezone.utc
                    ).isoformat(),
                }
            ]
        )

        record.to_csv(
            failure_path,
            mode="a",
            header=not failure_path.exists(),
            index=False,
        )

    def _classify_failure(
        self,
        exc: Exception,
    ) -> tuple[int | None, bool]:
        """
        Classify an error.

        404 is treated as permanent.
        429 and 5xx are retryable.
        Connection/timeouts are retryable.
        """

        response: Any = getattr(exc, "response", None)
        status_code = getattr(response, "status_code", None)

        if status_code == 404:
            return 404, False

        if status_code == 429:
            return 429, True

        if isinstance(status_code, int) and 500 <= status_code <= 599:
            return status_code, True

        return status_code, True

    def enrich_tmdb_ids(
        self,
        tmdb_ids: list[int],
        cache_path: str | Path = DEFAULT_CACHE_PATH,
        failure_path: str | Path = DEFAULT_FAILURE_PATH,
    ) -> pd.DataFrame:
        """
        Enrich TMDB IDs and append successful records to a cache.

        Successfully enriched IDs and permanent failures are skipped
        automatically on future runs.
        """

        cache_path = Path(cache_path)
        failure_path = Path(failure_path)

        cache_path.parent.mkdir(parents=True, exist_ok=True)
        failure_path.parent.mkdir(parents=True, exist_ok=True)

        completed_ids = self.load_completed_ids(cache_path)
        permanent_failures = self.load_permanent_failures(
            failure_path
        )

        unique_ids = list(dict.fromkeys(int(x) for x in tmdb_ids))

        pending_ids = [
            tmdb_id
            for tmdb_id in unique_ids
            if tmdb_id not in completed_ids
            and tmdb_id not in permanent_failures
        ]

        skipped_ids = (
            completed_ids.intersection(unique_ids)
            | permanent_failures.intersection(unique_ids)
        )

        print(f"Requested IDs: {len(unique_ids):,}")
        print(f"Already cached: {len(completed_ids.intersection(unique_ids)):,}")
        print(
            "Permanent failures skipped: "
            f"{len(permanent_failures.intersection(unique_ids)):,}"
        )
        print(f"Remaining: {len(pending_ids):,}")

        records = []

        for index, tmdb_id in enumerate(pending_ids, start=1):
            print(
                f"[{index}/{len(pending_ids)}] "
                f"Fetching TMDB ID {tmdb_id}..."
            )

            try:
                data = self.client.get_movie_details(
                    tmdb_id,
                    append_to_response="credits,keywords",
                )

                parsed = parse_tmdb_movie(data)

                # Write successful records immediately.
                pd.DataFrame([parsed]).to_csv(
                    cache_path,
                    mode="a",
                    header=not cache_path.exists(),
                    index=False,
                )

                records.append(parsed)

            except Exception as exc:
                status_code, retryable = self._classify_failure(exc)

                self._record_failure(
                    tmdb_id=tmdb_id,
                    failure_path=failure_path,
                    status_code=status_code,
                    retryable=retryable,
                    error_type=type(exc).__name__,
                    message=str(exc),
                )

                category = (
                    "retryable"
                    if retryable
                    else "permanent"
                )

                print(
                    f"  Failed ({category}) | "
                    f"HTTP: {status_code} | "
                    f"Error: {type(exc).__name__}"
                )

            if index < len(pending_ids):
                time.sleep(self.delay_seconds)

        return pd.DataFrame(records)

    def close(self) -> None:
        """Close the HTTP session."""

        self.client.close()


def run_sample_enrichment(
    catalog_path: str | Path = DEFAULT_CATALOG_PATH,
    cache_path: str | Path = DEFAULT_CACHE_PATH,
    failure_path: str | Path = DEFAULT_FAILURE_PATH,
    sample_size: int = 5,
) -> pd.DataFrame:
    """Run a small resumable enrichment test."""

    catalog_path = Path(catalog_path)

    if not catalog_path.exists():
        raise FileNotFoundError(
            f"Catalog not found: {catalog_path}"
        )

    catalog = pd.read_csv(catalog_path)

    if "tmdb_id" not in catalog.columns:
        raise ValueError(
            "Catalog must contain a tmdb_id column."
        )

    tmdb_ids = (
        pd.to_numeric(
            catalog["tmdb_id"],
            errors="coerce",
        )
        .dropna()
        .astype(int)
        .drop_duplicates()
        .head(sample_size)
        .tolist()
    )

    enricher = TMDBEnricher()

    try:
        result = enricher.enrich_tmdb_ids(
            tmdb_ids,
            cache_path=cache_path,
            failure_path=failure_path,
        )
    finally:
        enricher.close()

    return result


if __name__ == "__main__":
    result = run_sample_enrichment()

    print()
    print(f"Newly enriched rows: {len(result)}")
    print(f"Cache: {DEFAULT_CACHE_PATH}")
    print(f"Failures: {DEFAULT_FAILURE_PATH}")