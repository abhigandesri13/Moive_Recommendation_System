from __future__ import annotations

import os
from typing import Any

from curl_cffi import requests
from dotenv import load_dotenv


class TMDBClient:
    """Reusable client for TMDB movie metadata."""

    BASE_URL = "https://api.themoviedb.org/3"

    def __init__(
        self,
        api_key: str | None = None,
        timeout: int = 20,
        max_retries: int = 3,
    ) -> None:
        load_dotenv()

        self.api_key = api_key or os.getenv("TMDB_API_KEY")
        self.timeout = timeout

        if not self.api_key:
            raise ValueError(
                "TMDB_API_KEY is not configured. "
                "Add it to the local .env file."
            )

        self.session = requests.Session(
            impersonate="chrome",
            retry=max_retries,
        )

        self.session.headers.update(
            {
                "Accept": "application/json",
            }
        )

    def get_movie_details(
        self,
        tmdb_id: int,
        append_to_response: str | None = None,
    ) -> dict[str, Any]:
        """Fetch metadata for one TMDB movie ID."""

        if tmdb_id <= 0:
            raise ValueError("tmdb_id must be a positive integer.")

        params = {
            "api_key": self.api_key,
            "language": "en-US",
        }

        if append_to_response:
            params["append_to_response"] = append_to_response

        response = self.session.get(
            f"{self.BASE_URL}/movie/{tmdb_id}",
            params=params,
            timeout=self.timeout,
        )

        response.raise_for_status()

        data = response.json()

        if not isinstance(data, dict):
            raise ValueError("TMDB returned an unexpected response.")

        return data
    def find_by_imdb_id(
        self,
        imdb_id: str,
    ) -> dict[str, Any]:
        """Find TMDB records associated with an IMDb ID."""

        if not imdb_id:
            raise ValueError("imdb_id must not be empty.")

        normalized_id = str(imdb_id).strip()

        if not normalized_id.startswith("tt"):
            normalized_id = f"tt{normalized_id}"

        response = self.session.get(
            f"{self.BASE_URL}/find/{normalized_id}",
            params={
                "api_key": self.api_key,
                "external_source": "imdb_id",
            },
            timeout=self.timeout,
        )

        response.raise_for_status()

        data = response.json()

        if not isinstance(data, dict):
            raise ValueError("TMDB returned an unexpected response.")

        return data
    def search_movies(
        self,
        query: str,
        primary_release_year: int | None = None,
        page: int = 1,
    ) -> dict[str, Any]:
        """Search TMDB movies by title, optionally filtered by release year."""

        if not query or not query.strip():
            raise ValueError("query must not be empty.")

        params = {
            "api_key": self.api_key,
            "query": query.strip(),
            "language": "en-US",
            "page": page,
            "include_adult": False,
        }

        if primary_release_year is not None:
            params["primary_release_year"] = int(primary_release_year)

        response = self.session.get(
            f"{self.BASE_URL}/search/movie",
            params=params,
            timeout=self.timeout,
        )

        response.raise_for_status()

        data = response.json()

        if not isinstance(data, dict):
            raise ValueError("TMDB returned an unexpected response.")

        return data
    def close(self) -> None:
        """Close the underlying HTTP session."""

        self.session.close()