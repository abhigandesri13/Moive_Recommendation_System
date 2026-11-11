from pathlib import Path

import pandas as pd


class MovieLensValidator:
    """Validate the structure and basic integrity of MovieLens datasets."""

    REQUIRED_COLUMNS = {
        "movies": {"movieId", "title", "genres"},
        "ratings": {"userId", "movieId", "rating", "timestamp"},
        "tags": {"userId", "movieId", "tag", "timestamp"},
        "links": {"movieId", "imdbId", "tmdbId"},
    }

    def __init__(self, data_dir: str | Path = "data") -> None:
        self.data_dir = Path(data_dir)

    def _load(self, filename: str) -> pd.DataFrame:
        path = self.data_dir / filename

        if not path.exists():
            raise FileNotFoundError(f"Dataset file not found: {path}")

        return pd.read_csv(path)

    def _validate_columns(
        self,
        dataset_name: str,
        dataframe: pd.DataFrame,
    ) -> None:
        required = self.REQUIRED_COLUMNS[dataset_name]
        actual = set(dataframe.columns)

        missing = required - actual

        if missing:
            raise ValueError(
                f"{dataset_name}: missing required columns: {sorted(missing)}"
            )

    def _validate_ratings(self, ratings: pd.DataFrame) -> None:
        invalid = ratings[
            ~ratings["rating"].between(0.5, 5.0, inclusive="both")
        ]

        if not invalid.empty:
            raise ValueError(
                f"ratings: found {len(invalid)} ratings outside the valid range."
            )

    def _validate_ids(
        self,
        movies: pd.DataFrame,
        ratings: pd.DataFrame,
        tags: pd.DataFrame,
        links: pd.DataFrame,
    ) -> None:
        movie_ids = set(movies["movieId"])

        rating_movie_ids = set(ratings["movieId"])
        tag_movie_ids = set(tags["movieId"])
        link_movie_ids = set(links["movieId"])

        unknown_rating_ids = rating_movie_ids - movie_ids
        unknown_tag_ids = tag_movie_ids - movie_ids
        unknown_link_ids = link_movie_ids - movie_ids

        if unknown_rating_ids:
            raise ValueError(
                f"ratings: {len(unknown_rating_ids)} unknown movieId values."
            )

        if unknown_tag_ids:
            raise ValueError(
                f"tags: {len(unknown_tag_ids)} unknown movieId values."
            )

        if unknown_link_ids:
            raise ValueError(
                f"links: {len(unknown_link_ids)} unknown movieId values."
            )

    def validate(self) -> dict[str, object]:
        """Run all validation checks and return a validation summary."""

        movies = self._load("movies.csv")
        ratings = self._load("ratings.csv")
        tags = self._load("tags.csv")
        links = self._load("links.csv")

        datasets = {
            "movies": movies,
            "ratings": ratings,
            "tags": tags,
            "links": links,
        }

        for name, dataframe in datasets.items():
            self._validate_columns(name, dataframe)

        self._validate_ratings(ratings)
        self._validate_ids(movies, ratings, tags, links)

        return {
            "status": "passed",
            "row_counts": {
                name: len(dataframe)
                for name, dataframe in datasets.items()
            },
            "column_counts": {
                name: len(dataframe.columns)
                for name, dataframe in datasets.items()
            },
        }