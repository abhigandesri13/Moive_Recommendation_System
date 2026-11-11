import pandas as pd


def preprocess_ratings(ratings: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and standardize MovieLens ratings.

    This function does not modify the original dataframe.
    """

    required_columns = {"userId", "movieId", "rating", "timestamp"}

    missing_columns = required_columns - set(ratings.columns)

    if missing_columns:
        raise ValueError(
            f"ratings: missing required columns: {sorted(missing_columns)}"
        )

    df = ratings.copy()

    # Ensure identifiers use integer types.
    df["userId"] = pd.to_numeric(df["userId"], errors="raise").astype("int64")
    df["movieId"] = pd.to_numeric(df["movieId"], errors="raise").astype("int64")

    # Ratings must be numeric.
    df["rating"] = pd.to_numeric(df["rating"], errors="raise")

    # Convert Unix timestamp into UTC datetime.
    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        unit="s",
        utc=True,
    )

    # Initial positive-interaction flag.
    df["is_positive"] = df["rating"] >= 4.0

    # Sort chronologically for later temporal splitting.
    df = df.sort_values(
        ["timestamp", "userId", "movieId"]
    ).reset_index(drop=True)

    return df