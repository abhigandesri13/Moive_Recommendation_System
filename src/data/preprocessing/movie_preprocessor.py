import re

import pandas as pd


def extract_release_year(title: str) -> int | None:
    """Extract a four-digit release year from a MovieLens title."""
    if not isinstance(title, str):
        return None

    match = re.search(r"\((\d{4})\)\s*$", title)

    if match:
        return int(match.group(1))

    return None


def preprocess_movies(movies: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and enrich the MovieLens movie table.

    This function does not modify the original dataframe.
    """

    required_columns = {"movieId", "title", "genres"}

    missing_columns = required_columns - set(movies.columns)

    if missing_columns:
        raise ValueError(
            f"movies: missing required columns: {sorted(missing_columns)}"
        )

    df = movies.copy()

    # Basic string cleanup.
    df["title"] = df["title"].fillna("").astype(str).str.strip()
    df["genres"] = df["genres"].fillna("").astype(str).str.strip()

    # Clean title without removing the original MovieLens title.
    df["clean_title"] = (
        df["title"]
        .str.replace(r"\s*\(\d{4}\)\s*$", "", regex=True)
        .str.strip()
    )

    # Extract release year from the MovieLens title.
    df["release_year"] = df["title"].apply(extract_release_year)

    # Convert pipe-separated genres into Python lists.
    df["genre_list"] = df["genres"].apply(
        lambda value: [
            genre.strip()
            for genre in value.split("|")
            if genre.strip()
        ]
        if value and value != "(no genres listed)"
        else []
    )

    return df