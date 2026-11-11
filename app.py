
from __future__ import annotations

import pandas as pd
import streamlit as st

from src.services.recommendation_service import (
    MovieRecommendationService,
)


# ============================================================
# Page configuration
# ============================================================

st.set_page_config(
    page_title="CineMatch | Movie Recommendations",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# Styling
# ============================================================

st.markdown(
    """
    <style>
    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    .movie-card {
        border: 1px solid rgba(128,128,128,0.25);
        border-radius: 12px;
        padding: 18px;
        margin-bottom: 12px;
    }

    .movie-subtitle {
        color: #929292;
        font-size: 0.88rem;
    }

    div[data-testid="stMetric"] {
        background-color: rgba(128,128,128,0.06);
        padding: 12px;
        border-radius: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# Load recommendation service
# ============================================================

@st.cache_resource(show_spinner=False)
def load_recommendation_service():
    """Load and initialize the recommendation engine once."""

    return MovieRecommendationService()


try:
    with st.spinner(
        "Loading movie catalog and recommendation engine..."
    ):
        service = load_recommendation_service()

except Exception as exc:
    st.error(
        "The recommendation engine could not be initialized."
    )

    st.exception(exc)

    st.stop()


# ============================================================
# Session state
# ============================================================

if "raw_recommendations" not in st.session_state:
    st.session_state.raw_recommendations = None

if "recommendation_signature" not in st.session_state:
    st.session_state.recommendation_signature = None


# ============================================================
# Helper functions
# ============================================================

def is_missing(value) -> bool:
    if value is None:
        return True

    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def display_value(value, default="Not available") -> str:
    if is_missing(value):
        return default

    text = str(value).strip()

    return text if text else default


def display_year(value) -> str:
    if is_missing(value):
        return "Year unavailable"

    return str(int(float(value)))


def display_rating(value) -> str:
    if is_missing(value):
        return "Not available"

    try:
        return f"{float(value):.1f}/10"
    except (TypeError, ValueError):
        return "Not available"



def display_movie_card(movie) -> None:
    """Display a movie card with live TMDB metadata."""

    movie_id = int(movie["movieId"])

    title = str(movie.get("title", "Unknown movie"))
    genres = str(movie.get("genres", "Not available"))

    # Retrieve verified TMDB details on demand.
    details = service.get_tmdb_details(movie_id)

    poster_path = details.get("poster_path")
    poster_url = (
        f"https://image.tmdb.org/t/p/w500{poster_path}"
        if poster_path
        else None
    )

    rating = details.get("vote_average")
    runtime = details.get("runtime")
    overview = details.get("overview")
    director = details.get("director")
    cast = details.get("cast_names")
    release_date = details.get("release_date")
    vote_count = details.get("vote_count")

    with st.container(border=True):

        poster_col, info_col = st.columns(
            [1.1, 4],
            gap="large",
            vertical_alignment="top",
        )

        # ----------------------------------------------------
        # Poster
        # ----------------------------------------------------

        with poster_col:

            if poster_url:
                st.image(
                    poster_url,
                    width=170,
                )
            else:
                st.info("🎬 Poster unavailable")

        # ----------------------------------------------------
        # Movie information
        # ----------------------------------------------------

        with info_col:

            st.markdown(f"### {title}")

            st.caption(genres)

            # Ratings and runtime.
            col1, col2, col3 = st.columns(3)

            with col1:
                if rating is not None:
                    st.metric(
                        "TMDB Rating",
                        f"{float(rating):.1f}/10",
                    )
                else:
                    st.metric(
                        "TMDB Rating",
                        "Not rated",
                    )

            with col2:
                if runtime:
                    st.metric(
                        "Runtime",
                        f"{int(runtime)} min",
                    )
                else:
                    st.metric(
                        "Runtime",
                        "Unavailable",
                    )

            with col3:
                if vote_count is not None:
                    st.metric(
                        "Vote Count",
                        f"{int(vote_count):,}",
                    )
                else:
                    st.metric(
                        "Vote Count",
                        "Unavailable",
                    )

            # Overview.
            if overview:
                st.write(overview)
            else:
                st.write(
                    "Movie overview is not available."
                )

            # Additional information.
            if director:
                st.caption(f"**Director:** {director}")

            if release_date:
                st.caption(
                    f"**TMDB release date:** {release_date}"
                )

            if cast:

                with st.expander("View cast"):

                    st.write(cast)

            # Recommendation score.
            score = movie.get("score")

            if score is not None and pd.notna(score):

                st.caption(
                    f"Recommendation score: {float(score):.4f}"
                )

            # TMDB link.
            tmdb_id = details.get("id")

            if tmdb_id:

                st.markdown(
                    f"[View on TMDB]"
                    f"(https://www.themoviedb.org/movie/{tmdb_id})"
                )



def show_recommendations(
    raw_recommendations: pd.DataFrame,
    genre: str,
    year_range: tuple[int, int],
    top_n: int,
) -> None:

    recommendations = (
        service.filter_recommendations(
            recommendations=raw_recommendations,
            genre=genre,
            year_range=year_range,
            limit=top_n,
        )
    )

    if recommendations.empty:

        st.warning(
            "No movies match these filters. "
            "Try another genre or release-year range."
        )

        return

    st.subheader(
        f"Your recommendations ({len(recommendations)})"
    )

    st.caption(
        "Recommendations are ranked using the selected "
        "recommendation method."
    )

    for _, movie in recommendations.iterrows():

        display_movie_card(movie)


# ============================================================
# Header
# ============================================================

st.title("🎬 CineMatch")

st.markdown(
    """
    Discover movies based on your interests, viewing history
    and community ratings.
    """
)

st.caption(
    "Movie Recommendation System | Version 1 MVP"
)


# ============================================================
# Dashboard metrics
# ============================================================

metric1, metric2, metric3, metric4 = st.columns(4)

metric1.metric(
    "Movie catalog",
    f"{service.movie_count:,}",
)

metric2.metric(
    "Mapped movies",
    f"{service.mapped_small_movie_count:,}",
)

metric3.metric(
    "Rating records",
    f"{service.mapped_rating_count:,}",
)

metric4.metric(
    "Users",
    f"{len(service.user_ids):,}",
)


st.divider()


# ============================================================
# Sidebar controls
# ============================================================

st.sidebar.title("Recommendation Settings")

mode = st.sidebar.radio(
    "Choose recommendation mode",
    [
        "Similar Movies",
        "Personalized Recommendations",
        "Popular Movies",
    ],
)

st.sidebar.divider()

st.sidebar.subheader("Filters")

top_n = st.sidebar.slider(
    "Number of recommendations",
    min_value=5,
    max_value=30,
    value=10,
    step=5,
)

genre_options = [
    "All",
    *service.get_genres(),
]

genre_filter = st.sidebar.selectbox(
    "Filter by genre",
    genre_options,
)

year_min, year_max = service.get_year_bounds()

year_range = st.sidebar.slider(
    "Release year range",
    min_value=year_min,
    max_value=year_max,
    value=(year_min, year_max),
    step=1,
)

st.sidebar.divider()

st.sidebar.caption(
    "Recommendation methods: TF-IDF content similarity, "
    "user preference profiles and weighted popularity."
)


# ============================================================
# Recommendation selection
# ============================================================

current_signature = None
recommendation_button = False


if mode == "Similar Movies":

    st.subheader("🔎 Find Similar Movies")

    st.write(
        "Choose a movie to discover films with similar "
        "genres and descriptive tags."
    )

    search_query = st.text_input(
        "Search movie title",
        placeholder="Example: Toy Story, Titanic, Avatar...",
        key="movie_search",
    )

    matching_movies = service.search_movies(
        query=search_query,
        limit=50,
    )

    selected_movie_id = None

    if not matching_movies.empty:

        movie_lookup = (
            matching_movies
            .set_index("movie_id")
            .to_dict("index")
        )

        selected_movie_id = st.selectbox(
            "Select a movie",
            options=matching_movies["movie_id"].astype(int).tolist(),
            format_func=lambda movie_id: (
                movie_lookup[movie_id]["title"]
            ),
        )

        current_signature = (
            "similar",
            int(selected_movie_id),
        )

    elif search_query.strip():

        st.info(
            "No matching movies found. Try another title."
        )

    else:

        st.info(
            "Start by entering a movie title above."
        )

    recommendation_button = st.button(
        "🎬 Get Similar Movies",
        type="primary",
        disabled=selected_movie_id is None,
        use_container_width=True,
    )


elif mode == "Personalized Recommendations":

    st.subheader("👤 Personalized Recommendations")

    st.write(
        "Choose a MovieLens user to receive recommendations "
        "based on their positively rated movies."
    )

    selected_user_id = st.selectbox(
        "Select user ID",
        options=service.user_ids,
        format_func=lambda user_id: f"User {user_id}",
    )

    current_signature = (
        "personalized",
        int(selected_user_id),
    )

    recommendation_button = st.button(
        "✨ Get Personalized Recommendations",
        type="primary",
        use_container_width=True,
    )


elif mode == "Popular Movies":

    st.subheader("⭐ Popular Movies")

    st.write(
        "Discover highly rated movies based on the "
        "available MovieLens rating history."
    )

    current_signature = (
        "popular",
    )

    recommendation_button = st.button(
        "🔥 Explore Popular Movies",
        type="primary",
        use_container_width=True,
    )


# ============================================================
# Generate recommendations
# ============================================================

if recommendation_button:

    try:

        with st.spinner(
            "Finding the best movies for you..."
        ):

            if mode == "Similar Movies":

                raw = service.recommend_similar_movies(
                    movie_id=int(selected_movie_id)
                )

            elif mode == "Personalized Recommendations":

                raw = service.recommend_for_user(
                    user_id=int(selected_user_id)
                )

            else:

                raw = service.recommend_popular()

            st.session_state.raw_recommendations = raw

            st.session_state.recommendation_signature = (
                current_signature
            )

    except Exception as exc:

        st.error(
            "Unable to generate recommendations."
        )

        st.exception(exc)


# ============================================================
# Display results
# ============================================================

if (
    current_signature is not None
    and st.session_state.recommendation_signature
    == current_signature
    and st.session_state.raw_recommendations
    is not None
):

    st.divider()

    show_recommendations(
        raw_recommendations=(
            st.session_state.raw_recommendations
        ),
        genre=genre_filter,
        year_range=year_range,
        top_n=top_n,
    )


# ============================================================
# Footer
# ============================================================

st.divider()

st.caption(
    "CineMatch | Powered by Python, Pandas, "
    "Scikit-learn, TF-IDF and Streamlit."
)
