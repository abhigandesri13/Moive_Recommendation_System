import pandas as pd


def temporal_split(
    ratings: pd.DataFrame,
    train_ratio: float = 0.8,
    validation_ratio: float = 0.1,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split ratings chronologically for each user.

    Each user's interactions are ordered by timestamp and divided into:
        - training: earliest interactions
        - validation: next interactions
        - test: latest interactions

    The remaining portion after train and validation becomes test.
    """

    required_columns = {
        "userId",
        "movieId",
        "rating",
        "timestamp",
    }

    missing_columns = required_columns - set(ratings.columns)

    if missing_columns:
        raise ValueError(
            f"ratings: missing required columns: {sorted(missing_columns)}"
        )

    if not 0 < train_ratio < 1:
        raise ValueError("train_ratio must be between 0 and 1.")

    if not 0 <= validation_ratio < 1:
        raise ValueError("validation_ratio must be between 0 and 1.")

    if train_ratio + validation_ratio >= 1:
        raise ValueError(
            "train_ratio + validation_ratio must be less than 1."
        )

    df = ratings.copy()

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        utc=True,
        errors="raise",
    )

    df = df.sort_values(
        ["userId", "timestamp", "movieId"]
    ).reset_index(drop=True)

    train_parts = []
    validation_parts = []
    test_parts = []

    for _, user_ratings in df.groupby("userId", sort=False):
        n = len(user_ratings)

        train_end = max(1, int(n * train_ratio))
        validation_end = max(
            train_end + 1,
            int(n * (train_ratio + validation_ratio)),
        )

        # Prevent validation_end from exceeding the available rows.
        validation_end = min(validation_end, n - 1)

        # Very small histories cannot provide all three splits.
        if n < 3:
            train_parts.append(user_ratings)
            continue

        train_parts.append(user_ratings.iloc[:train_end])
        validation_parts.append(
            user_ratings.iloc[train_end:validation_end]
        )
        test_parts.append(user_ratings.iloc[validation_end:])

    train = pd.concat(train_parts, ignore_index=True)

    validation = (
        pd.concat(validation_parts, ignore_index=True)
        if validation_parts
        else pd.DataFrame(columns=df.columns)
    )

    test = (
        pd.concat(test_parts, ignore_index=True)
        if test_parts
        else pd.DataFrame(columns=df.columns)
    )

    return train, validation, test