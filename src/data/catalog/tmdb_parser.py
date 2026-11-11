from __future__ import annotations

from typing import Any


def _join_names(items: list[dict[str, Any]], key: str = "name") -> str:
    """Join non-empty values from a TMDB list using a pipe separator."""
    values = []

    for item in items:
        value = item.get(key)

        if value:
            value = str(value).strip()

            if value and value not in values:
                values.append(value)

    return "|".join(values)


def _get_original_language_name(
    original_language: str | None,
    spoken_languages: list[dict[str, Any]],
) -> str | None:
    """Convert the original-language ISO code to a human-readable name."""

    if not original_language:
        return None

    for language in spoken_languages:
        if language.get("iso_639_1") == original_language:
            name = language.get("english_name") or language.get("name")

            if name:
                return str(name).strip()

    return original_language


def _get_directors(crew: list[dict[str, Any]]) -> str:
    """Extract director names from TMDB crew metadata."""
    directors = []

    for member in crew:
        if member.get("job") != "Director":
            continue

        name = member.get("name")

        if name and name not in directors:
            directors.append(str(name).strip())

    return "|".join(directors)


def _get_top_cast(
    cast: list[dict[str, Any]],
    limit: int = 20,
) -> str:
    """Keep the top billed cast members returned by TMDB."""
    names = []

    for member in cast[:limit]:
        name = member.get("name")

        if name and name not in names:
            names.append(str(name).strip())

    return "|".join(names)


def parse_tmdb_movie(data: dict[str, Any]) -> dict[str, Any]:
    """
    Convert one TMDB movie response into normalized catalog fields.

    Important:
    - translations are NOT treated as available audio languages.
    - dubbed_languages is intentionally left unset.
    - industry classification is handled separately.
    """

    spoken_languages = data.get("spoken_languages") or []
    production_countries = data.get("production_countries") or []

    credits = data.get("credits") or {}
    crew = credits.get("crew") or []
    cast = credits.get("cast") or []

    keywords_data = data.get("keywords") or {}
    keywords = keywords_data.get("keywords") or []

    original_language = data.get("original_language")

    return {
        "tmdb_id": data.get("id"),
        "imdb_id": data.get("imdb_id"),
        "title": data.get("title"),
        "original_title": data.get("original_title"),
        "release_date": data.get("release_date") or None,
        "original_language": original_language,
        "language": _get_original_language_name(
            original_language,
            spoken_languages,
        ),
        "country": _join_names(
            production_countries,
            key="name",
        ),
        "available_languages": None,
        "dubbed_languages": None,
        "overview": data.get("overview"),
        "keywords": _join_names(keywords),
        "runtime": data.get("runtime"),
        "director": _get_directors(crew),
        "cast": _get_top_cast(cast),
        "vote_average": data.get("vote_average"),
        "vote_count": data.get("vote_count"),
        "popularity": data.get("popularity"),
    }