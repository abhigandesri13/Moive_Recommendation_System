from pathlib import Path

import pandas as pd


class MovieLensLoader:
    """Load and validate the MovieLens CSV files."""

    def __init__(self, data_dir: str | Path = "data") -> None:
        self.data_dir = Path(data_dir)

    def _load_csv(self, filename: str) -> pd.DataFrame:
        """Load one CSV file from the configured data directory."""
        file_path = self.data_dir / filename

        if not file_path.exists():
            raise FileNotFoundError(f"Dataset file not found: {file_path}")

        return pd.read_csv(file_path)

    def load_movies(self) -> pd.DataFrame:
        return self._load_csv("movies.csv")

    def load_ratings(self) -> pd.DataFrame:
        return self._load_csv("ratings.csv")

    def load_tags(self) -> pd.DataFrame:
        return self._load_csv("tags.csv")

    def load_links(self) -> pd.DataFrame:
        return self._load_csv("links.csv")

    def load_all(self) -> dict[str, pd.DataFrame]:
        """Load all MovieLens datasets."""
        return {
            "movies": self.load_movies(),
            "ratings": self.load_ratings(),
            "tags": self.load_tags(),
            "links": self.load_links(),
        }