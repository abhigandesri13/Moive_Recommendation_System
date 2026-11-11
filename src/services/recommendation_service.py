
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from src.recommenders.semantic.content_recommender import (
    ContentRecommender,
)

from src.recommenders.popularity.popularity_recommender import (
    PopularityRecommender,
)

from src.services.tmdb_client import TMDBClient


# ============================================================
# Default paths
# ============================================================

DEFAULT_CATALOG_PATH = Path(
    "data/catalog/movielens_catalog_runtime.csv.gz"
)

DEFAULT_RATINGS_PATH = Path(
    "data/ratings.csv"
)

DEFAULT_LINKS_PATH = Path(
    "data/links.csv"
)


# ============================================================
# IMDb ID normalization
# ============================================================

def normalize_imdb_id(value: object) -> str:
    """Normalize IMDb IDs into canonical ttXXXXXXXX format."""

    if pd.isna(value):
        return ""

    if isinstance(value, float):
        if value.is_integer():
            value = str(int(value))
        else:
            return ""
    else:
        value = str(value).strip()

    if not value:
        return ""

    if value.lower().startswith("tt"):
        value = value[2:]

    if value.endswith(".0"):
        value = value[:-2]

    if not value.isdigit() or set(value) == {"0"}:
        return ""

    return f"tt{value.zfill(7)}"


# ============================================================
# Movie Recommendation Service
# ============================================================

class MovieRecommendationService:
    """
    Central service for the CineMatch recommendation engine.

    Version 1 recommendation methods:

    1. Content-based similar movie recommendations.
    2. Personalized recommendations based on positive ratings.
    3. Popularity-based movie recommendations.

    Data architecture:

    - Expanded MovieLens movie catalog.
    - MovieLens-small rating history.
    - IMDb-based identity crosswalk.
    - Verified TMDB IDs for external movie details.
    - On-demand TMDB API metadata retrieval.
    """

    def __init__(
        self,
        catalog_path: str | Path = DEFAULT_CATALOG_PATH,
        ratings_path: str | Path = DEFAULT_RATINGS_PATH,
        links_path: str | Path = DEFAULT_LINKS_PATH,
        minimum_rating_count: int = 10,
    ) -> None:

        self.catalog_path = Path(catalog_path)
        self.ratings_path = Path(ratings_path)
        self.links_path = Path(links_path)

        self._validate_files()

        # ----------------------------------------------------
        # Load catalog
        # ----------------------------------------------------

        print("Loading final movie catalog...")

        self.catalog = pd.read_csv(
            self.catalog_path,
            low_memory=False,
        )

        # ----------------------------------------------------
        # Load MovieLens-small interaction data
        # ----------------------------------------------------

        print(
            "Loading MovieLens-small ratings and links..."
        )

        self.source_ratings = pd.read_csv(
            self.ratings_path
        )

        self.source_links = pd.read_csv(
            self.links_path
        )

        self._validate_columns()

        # ----------------------------------------------------
        # Normalize catalog fields
        # ----------------------------------------------------

        self.catalog["movie_id"] = pd.to_numeric(
            self.catalog["movie_id"],
            errors="raise",
        ).astype(int)

        self.catalog["tmdb_id"] = pd.to_numeric(
            self.catalog["tmdb_id"],
            errors="coerce",
        )

        self.catalog["release_year"] = pd.to_numeric(
            self.catalog["release_year"],
            errors="coerce",
        )

        self.catalog["genres"] = (
            self.catalog["genres"]
            .fillna("")
            .astype(str)
        )

        self.catalog["tags"] = (
            self.catalog["tags"]
            .fillna("")
            .astype(str)
        )

        if self.catalog["movie_id"].duplicated().any():
            raise ValueError(
                "Duplicate movie IDs exist in the runtime catalog."
            )

        # ----------------------------------------------------
        # Build IMDb identity crosswalk
        # ----------------------------------------------------

        print("Building IMDb identity crosswalk...")

        self.movie_bridge = self._build_movie_bridge()

        if self.movie_bridge.empty:
            raise RuntimeError(
                "No MovieLens-small movies could be mapped "
                "to the runtime catalog through IMDb IDs."
            )

        # ----------------------------------------------------
        # Map ratings to final catalog movie IDs
        # ----------------------------------------------------

        mapped_ratings = self.source_ratings.merge(
            self.movie_bridge,
            on="movieId",
            how="inner",
            validate="many_to_one",
        )

        mapped_ratings["movieId"] = (
            mapped_ratings["catalog_movie_id"]
            .astype(int)
        )

        mapped_ratings["userId"] = pd.to_numeric(
            mapped_ratings["userId"],
            errors="raise",
        ).astype(int)

        mapped_ratings["rating"] = pd.to_numeric(
            mapped_ratings["rating"],
            errors="raise",
        )

        self.ratings = mapped_ratings[
            [
                "userId",
                "movieId",
                "rating",
            ]
        ].copy()

        if self.ratings.empty:
            raise RuntimeError(
                "No rating records could be mapped "
                "to the runtime catalog."
            )

        # ----------------------------------------------------
        # Prepare recommender input
        # ----------------------------------------------------

        self.model_movies = self.catalog[
            [
                "movie_id",
                "title",
                "genres",
            ]
        ].rename(
            columns={
                "movie_id": "movieId",
            }
        ).copy()

        self.model_tags = self.catalog[
            [
                "movie_id",
                "tags",
            ]
        ].rename(
            columns={
                "movie_id": "movieId",
                "tags": "tag",
            }
        ).copy()

        # ----------------------------------------------------
        # Initialize content recommender
        # ----------------------------------------------------

        print(
            "Building TF-IDF content recommender..."
        )

        self.content_model = ContentRecommender()

        self.content_model.fit(
            movies=self.model_movies,
            tags=self.model_tags,
        )

        # ----------------------------------------------------
        # Initialize popularity recommender
        # ----------------------------------------------------

        print(
            "Building popularity recommender..."
        )

        self.popularity_model = PopularityRecommender(
            minimum_rating_count=minimum_rating_count,
        )

        self.popularity_model.fit(
            movies=self.model_movies,
            ratings=self.ratings,
        )

        # ----------------------------------------------------
        # Basic catalog metadata
        # ----------------------------------------------------

        self.metadata_columns = [
            "movie_id",
            "tmdb_id",
            "release_year",
        ]

        self.metadata = self.catalog[
            [
                column
                for column in self.metadata_columns
                if column in self.catalog.columns
            ]
        ].copy()

        # ----------------------------------------------------
        # User and catalog statistics
        # ----------------------------------------------------

        self.user_ids = sorted(
            self.ratings["userId"]
            .unique()
            .tolist()
        )

        self.movie_count = len(self.catalog)

        self.mapped_small_movie_count = (
            self.movie_bridge["movieId"].nunique()
        )

        self.mapped_rating_count = len(
            self.ratings
        )

        # ----------------------------------------------------
        # TMDB client and metadata cache
        # ----------------------------------------------------

        self._tmdb_client: TMDBClient | None = None

        self._tmdb_client_checked = False

        self._tmdb_details_cache: dict[int, dict] = {}

        # ----------------------------------------------------
        # Initialization summary
        # ----------------------------------------------------

        print("\nRecommendation service ready.")

        print(
            f"Catalog movies: {self.movie_count:,}"
        )

        print(
            "Mapped MovieLens-small movies: "
            f"{self.mapped_small_movie_count:,}"
        )

        print(
            "Mapped rating records: "
            f"{self.mapped_rating_count:,}"
        )

        print(
            f"Available users: {len(self.user_ids):,}"
        )

    # ========================================================
    # File validation
    # ========================================================

    def _validate_files(self) -> None:
        """Ensure all required input files exist."""

        paths = [
            self.catalog_path,
            self.ratings_path,
            self.links_path,
        ]

        for path in paths:

            if not path.exists():
                raise FileNotFoundError(
                    f"Required file not found: {path}"
                )

    def _validate_columns(self) -> None:
        """Validate required columns in all datasets."""

        required_catalog = {
            "movie_id",
            "tmdb_id",
            "imdb_id",
            "title",
            "genres",
            "tags",
            "release_year",
        }

        required_ratings = {
            "userId",
            "movieId",
            "rating",
        }

        required_links = {
            "movieId",
            "imdbId",
        }

        checks = [
            (
                "Catalog",
                required_catalog,
                set(self.catalog.columns),
            ),
            (
                "Ratings",
                required_ratings,
                set(self.source_ratings.columns),
            ),
            (
                "Links",
                required_links,
                set(self.source_links.columns),
            ),
        ]

        for name, required, actual in checks:

            missing = required - actual

            if missing:
                raise ValueError(
                    f"{name} missing columns: "
                    f"{sorted(missing)}"
                )

    # ========================================================
    # IMDb identity crosswalk
    # ========================================================

    def _build_movie_bridge(self) -> pd.DataFrame:
        """
        Connect MovieLens-small IDs to catalog IDs using IMDb IDs.

        Ambiguous IMDb mappings are excluded.
        """

        catalog_ids = self.catalog[
            [
                "movie_id",
                "imdb_id",
            ]
        ].copy()

        catalog_ids["imdb_key"] = (
            catalog_ids["imdb_id"]
            .map(normalize_imdb_id)
        )

        catalog_ids = catalog_ids[
            catalog_ids["imdb_key"] != ""
        ].copy()

        imdb_counts = (
            catalog_ids.groupby("imdb_key")["movie_id"]
            .nunique()
        )

        unique_imdb_ids = set(
            imdb_counts[
                imdb_counts == 1
            ].index
        )

        catalog_ids = catalog_ids[
            catalog_ids["imdb_key"].isin(
                unique_imdb_ids
            )
        ].copy()

        links = self.source_links[
            [
                "movieId",
                "imdbId",
            ]
        ].copy()

        links["imdb_key"] = (
            links["imdbId"]
            .map(normalize_imdb_id)
        )

        links = links[
            links["imdb_key"] != ""
        ].copy()

        bridge = links.merge(
            catalog_ids[
                [
                    "movie_id",
                    "imdb_key",
                ]
            ],
            on="imdb_key",
            how="inner",
            validate="many_to_one",
        )

        bridge = bridge.rename(
            columns={
                "movie_id": "catalog_movie_id",
            }
        )

        if bridge["movieId"].duplicated().any():
            raise RuntimeError(
                "A MovieLens-small movie maps to multiple "
                "runtime catalog records."
            )

        return bridge[
            [
                "movieId",
                "catalog_movie_id",
            ]
        ].copy()

    # ========================================================
    # Attach basic catalog metadata
    # ========================================================

    def _attach_metadata(
        self,
        recommendations: pd.DataFrame,
    ) -> pd.DataFrame:
        """Attach release year and verified TMDB ID."""

        if recommendations.empty:
            return recommendations.copy()

        result = recommendations.merge(
            self.metadata,
            left_on="movieId",
            right_on="movie_id",
            how="left",
            validate="one_to_one",
        )

        result = result.drop(
            columns=["movie_id"],
            errors="ignore",
        )

        return result

    # ========================================================
    # On-demand TMDB metadata
    # ========================================================

    def get_tmdb_details(
        self,
        catalog_movie_id: int,
    ) -> dict:
        """
        Retrieve detailed movie metadata from TMDB on demand.

        Retrieves:
        - Poster path
        - Overview
        - Rating
        - Runtime
        - Genres
        - Director
        - Cast
        - Release date

        Results are cached in memory for the lifetime
        of this recommendation service.
        """

        catalog_movie_id = int(catalog_movie_id)

        # Return previously retrieved metadata.
        if catalog_movie_id in self._tmdb_details_cache:
            return self._tmdb_details_cache[
                catalog_movie_id
            ]

        # Find movie in runtime catalog.
        movie = self.catalog.loc[
            self.catalog["movie_id"]
            == catalog_movie_id
        ]

        if movie.empty:
            return {}

        tmdb_id = pd.to_numeric(
            movie.iloc[0]["tmdb_id"],
            errors="coerce",
        )

        # No verified TMDB mapping.
        if pd.isna(tmdb_id):
            return {}

        # Initialize API client once.
        if not self._tmdb_client_checked:

            self._tmdb_client_checked = True

            try:
                self._tmdb_client = TMDBClient()

            except Exception as exc:

                print(
                    "TMDB client unavailable: "
                    f"{type(exc).__name__}"
                )

                return {}

        if self._tmdb_client is None:
            return {}

        try:

            details = self._tmdb_client.get_movie_details(
                int(tmdb_id),
                append_to_response="credits,keywords",
            )

            if not isinstance(details, dict):
                return {}

            # ------------------------------------------------
            # Extract directors
            # ------------------------------------------------

            credits = details.get(
                "credits",
                {},
            )

            crew = credits.get(
                "crew",
                [],
            )

            directors = [
                person.get("name", "")
                for person in crew
                if person.get("job") == "Director"
                and person.get("name")
            ]

            details["director"] = ", ".join(
                dict.fromkeys(directors)
            )

            # ------------------------------------------------
            # Extract main cast
            # ------------------------------------------------

            cast = credits.get(
                "cast",
                [],
            )

            details["cast_names"] = ", ".join(
                person.get("name", "")
                for person in cast[:10]
                if person.get("name")
            )

            # ------------------------------------------------
            # Cache successful response
            # ------------------------------------------------

            self._tmdb_details_cache[
                catalog_movie_id
            ] = details

            return details

        except Exception as exc:

            print(
                f"TMDB lookup failed for catalog movie "
                f"{catalog_movie_id}: {type(exc).__name__}"
            )

            return {}

    # ========================================================
    # Movie search
    # ========================================================

    def search_movies(
        self,
        query: str,
        limit: int = 30,
    ) -> pd.DataFrame:
        """Search movies by title."""

        if not query or not query.strip():

            return pd.DataFrame(
                columns=[
                    "movie_id",
                    "title",
                    "release_year",
                    "genres",
                ]
            )

        matches = self.catalog[
            self.catalog["title"]
            .astype(str)
            .str.contains(
                re.escape(query.strip()),
                case=False,
                regex=True,
                na=False,
            )
        ].copy()

        matches["_exact_match"] = (
            matches["title"]
            .str.lower()
            .eq(query.strip().lower())
        )

        matches = matches.sort_values(
            ["_exact_match", "title"],
            ascending=[False, True],
        )

        return matches[
            [
                "movie_id",
                "title",
                "release_year",
                "genres",
            ]
        ].head(limit).reset_index(drop=True)

    # ========================================================
    # Similar movie recommendations
    # ========================================================

    def recommend_similar_movies(
        self,
        movie_id: int,
    ) -> pd.DataFrame:
        """Return content-based movie recommendations."""

        recommendations = self.content_model.recommend(
            movie_id=int(movie_id),
            n=self.movie_count,
        )

        return self._attach_metadata(
            recommendations
        )

    # ========================================================
    # Personalized recommendations
    # ========================================================

    def recommend_for_user(
        self,
        user_id: int,
    ) -> pd.DataFrame:
        """Return personalized content recommendations."""

        recommendations = (
            self.content_model.recommend_for_user(
                user_id=int(user_id),
                ratings=self.ratings,
                n=self.movie_count,
                positive_threshold=4.0,
            )
        )

        return self._attach_metadata(
            recommendations
        )

    # ========================================================
    # Popular recommendations
    # ========================================================

    def recommend_popular(self) -> pd.DataFrame:
        """Return globally popular recommendations."""

        recommendations = (
            self.popularity_model.recommend(
                n=self.movie_count,
            )
        )

        return self._attach_metadata(
            recommendations
        )

    # ========================================================
    # Recommendation filtering
    # ========================================================

    @staticmethod
    def filter_recommendations(
        recommendations: pd.DataFrame,
        genre: str = "All",
        year_range: tuple[int, int] | None = None,
        limit: int = 10,
    ) -> pd.DataFrame:
        """Apply genre and release-year filters."""

        if recommendations.empty:
            return recommendations.copy()

        result = recommendations.copy()

        # Genre filter.
        if genre and genre != "All":

            result = result[
                result["genres"]
                .fillna("")
                .astype(str)
                .apply(
                    lambda value: (
                        genre in value.split("|")
                    )
                )
            ]

        # Release-year filter.
        if year_range is not None:

            minimum_year, maximum_year = year_range

            years = pd.to_numeric(
                result["release_year"],
                errors="coerce",
            )

            result = result[
                years.between(
                    minimum_year,
                    maximum_year,
                    inclusive="both",
                )
            ]

        # Preserve ranking order where possible.
        if "score" in result.columns:

            result = result.sort_values(
                "score",
                ascending=False,
            )

        return result.head(
            max(1, int(limit))
        ).reset_index(drop=True)

    # ========================================================
    # Filter options
    # ========================================================

    def get_genres(self) -> list[str]:
        """Return unique genres."""

        genres = set()

        for value in self.catalog["genres"].dropna():

            for genre in str(value).split("|"):

                genre = genre.strip()

                if (
                    genre
                    and genre.lower() != "(no genres listed)"
                ):
                    genres.add(genre)

        return sorted(genres)

    def get_year_bounds(self) -> tuple[int, int]:
        """Return minimum and maximum release years."""

        years = pd.to_numeric(
            self.catalog["release_year"],
            errors="coerce",
        ).dropna()

        if years.empty:
            return 1900, 2026

        return (
            int(years.min()),
            int(years.max()),
        )

    # ========================================================
    # Close external resources
    # ========================================================

    def close(self) -> None:
        """Close the TMDB API client when initialized."""

        if self._tmdb_client is not None:

            self._tmdb_client.close()

            self._tmdb_client = None
