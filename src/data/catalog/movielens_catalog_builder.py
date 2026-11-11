from pathlib import Path

import pandas as pd


DEFAULT_DATA_DIR = Path("data/raw/ml-32m")
DEFAULT_OUTPUT_PATH = Path("data/catalog/movielens_catalog_base.csv")


def build_movielens_catalog(
    data_dir: str | Path = DEFAULT_DATA_DIR,
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
) -> pd.DataFrame:
    """
    Build the MovieLens catalog base from movies, links, and tags.

    This stage does not load ratings.csv and does not call any external API.
    """

    data_dir = Path(data_dir)
    output_path = Path(output_path)

    movies_path = data_dir / "movies.csv"
    links_path = data_dir / "links.csv"
    tags_path = data_dir / "tags.csv"

    for path in (movies_path, links_path, tags_path):
        if not path.exists():
            raise FileNotFoundError(f"Required file not found: {path}")

    movies = pd.read_csv(movies_path)
    links = pd.read_csv(links_path)
    tags = pd.read_csv(tags_path)

    required_movie_columns = {"movieId", "title", "genres"}
    required_link_columns = {"movieId", "imdbId", "tmdbId"}
    required_tag_columns = {"movieId", "tag"}

    if not required_movie_columns.issubset(movies.columns):
        raise ValueError("movies.csv is missing required columns.")

    if not required_link_columns.issubset(links.columns):
        raise ValueError("links.csv is missing required columns.")

    if not required_tag_columns.issubset(tags.columns):
        raise ValueError("tags.csv is missing required columns.")

    # Keep one normalized tag string per movie.
    tag_table = (
        tags.dropna(subset=["movieId", "tag"])
        .assign(tag=lambda df: df["tag"].astype(str).str.strip())
        .query("tag != ''")
        .groupby("movieId")["tag"]
        .agg(lambda values: "|".join(dict.fromkeys(values)))
        .reset_index(name="tags")
    )

    catalog = movies.merge(
        links[["movieId", "imdbId", "tmdbId"]],
        on="movieId",
        how="left",
        validate="one_to_one",
    )

    catalog = catalog.merge(
        tag_table,
        on="movieId",
        how="left",
        validate="one_to_one",
    )

    catalog["tags"] = catalog["tags"].fillna("")

    catalog["movie_id"] = catalog["movieId"].astype("int64")
    catalog["tmdb_id"] = pd.to_numeric(catalog["tmdbId"], errors="coerce")
    catalog["imdb_id"] = pd.to_numeric(catalog["imdbId"], errors="coerce")

    catalog["title"] = catalog["title"].fillna("").astype(str).str.strip()
    catalog["genres"] = catalog["genres"].fillna("").astype(str).str.strip()

    # Preserve the MovieLens title exactly as the initial catalog title.
    catalog["original_title"] = catalog["title"]

    # Release year will be enriched from TMDB later where MovieLens parsing
    # is unavailable or unreliable.
    catalog["release_year"] = pd.to_numeric(
        catalog["title"].str.extract(r"\((\d{4})\)\s*$")[0],
        errors="coerce",
    ).astype("Int64")

    catalog["release_date"] = pd.NA

    # Fields reserved for TMDB enrichment.
    catalog["industry"] = pd.NA
    catalog["original_language"] = pd.NA
    catalog["language"] = pd.NA
    catalog["country"] = pd.NA
    catalog["available_languages"] = pd.NA
    catalog["dubbed_languages"] = pd.NA
    catalog["overview"] = pd.NA
    catalog["keywords"] = pd.NA
    catalog["runtime"] = pd.NA
    catalog["director"] = pd.NA
    catalog["cast"] = pd.NA
    catalog["vote_average"] = pd.NA
    catalog["vote_count"] = pd.NA
    catalog["popularity"] = pd.NA

    catalog["content_embedding"] = pd.NA
    catalog["is_active"] = True
    catalog["metadata_last_updated"] = pd.NaT
    catalog["catalog_source"] = "movielens"

    output_columns = [
        "movie_id",
        "tmdb_id",
        "imdb_id",
        "title",
        "original_title",
        "release_date",
        "release_year",
        "industry",
        "original_language",
        "language",
        "country",
        "available_languages",
        "dubbed_languages",
        "genres",
        "overview",
        "keywords",
        "tags",
        "runtime",
        "director",
        "cast",
        "vote_average",
        "vote_count",
        "popularity",
        "content_embedding",
        "is_active",
        "metadata_last_updated",
        "catalog_source",
    ]

    catalog = catalog[output_columns].sort_values("movie_id").reset_index(drop=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    catalog.to_csv(output_path, index=False)

    return catalog


if __name__ == "__main__":
    result = build_movielens_catalog()

    print(f"Catalog rows: {len(result):,}")
    print(f"Catalog columns: {len(result.columns)}")
    print(f"Movies with TMDB ID: {result['tmdb_id'].notna().sum():,}")
    print(f"Movies with tags: {(result['tags'] != '').sum():,}")
    print(f"Movies with parsed year: {result['release_year'].notna().sum():,}")
    print(f"Saved to: {DEFAULT_OUTPUT_PATH}")