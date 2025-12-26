import streamlit as st
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

# --------------------------------------------------
# Page Configuration
# --------------------------------------------------
st.set_page_config(
    page_title="Movie Recommendation System",
    layout="wide"
)

# --------------------------------------------------
# 1. Load Data (Cached)
# --------------------------------------------------
@st.cache_data
def load_data():
    ratings = pd.read_csv("data/ratings.csv")
    movies = pd.read_csv("data/movies.csv")
    tags = pd.read_csv("data/tags.csv")
    links = pd.read_csv("data/links.csv")
    return ratings, movies, tags, links


ratings, movies, tags, links = load_data()

# --------------------------------------------------
# 2. Heavy Preprocessing (Cached for Cloud Stability)
# --------------------------------------------------
@st.cache_resource
def preprocess_data(ratings, movies, tags):
    # Merge ratings with movie titles
    ratings_movies = pd.merge(ratings, movies, on="movieId", how="left")

    # User–Item Matrix
    user_item_matrix = ratings_movies.pivot_table(
        index="userId",
        columns="movieId",
        values="rating"
    ).fillna(0)

    # Collaborative Filtering Similarity
    movie_similarity = cosine_similarity(user_item_matrix.T)
    movie_similarity_df = pd.DataFrame(
        movie_similarity,
        index=user_item_matrix.columns,
        columns=user_item_matrix.columns
    )

    # Content-Based Data Preparation
    tags_grouped = tags.groupby("movieId")["tag"].apply(
        lambda x: " ".join(x.astype(str))
    ).reset_index()

    movies_content = pd.merge(
        movies[["movieId", "title", "genres"]],
        tags_grouped,
        on="movieId",
        how="left"
    )

    movies_content["tag"] = movies_content["tag"].fillna("")
    movies_content["genres_clean"] = movies_content["genres"].apply(
        lambda x: x.replace("|", " ") if isinstance(x, str) else ""
    )
    movies_content["content"] = movies_content["genres_clean"] + " " + movies_content["tag"]

    # TF-IDF + Cosine Similarity
    tfidf = TfidfVectorizer(stop_words="english")
    tfidf_matrix = tfidf.fit_transform(movies_content["content"])

    content_similarity = cosine_similarity(tfidf_matrix)
    content_similarity_df = pd.DataFrame(
        content_similarity,
        index=movies_content["movieId"],
        columns=movies_content["movieId"]
    )

    return user_item_matrix, movie_similarity_df, content_similarity_df, movies_content


user_item_matrix_filled, movie_similarity_df, content_similarity_df, movies_content = preprocess_data(
    ratings, movies, tags
)

# --------------------------------------------------
# 3. Recommendation Functions
# --------------------------------------------------
def recommend_similar_movies_collab(movie_title, n=10):
    movie_row = movies[movies["title"].str.lower() == movie_title.lower()]
    if movie_row.empty:
        return pd.DataFrame(columns=["movieId", "title", "genres"])

    movie_id = movie_row.iloc[0]["movieId"]

    if movie_id not in movie_similarity_df.index:
        return pd.DataFrame(columns=["movieId", "title", "genres"])

    scores = movie_similarity_df[movie_id]
    similar_movie_ids = scores.sort_values(ascending=False).iloc[1:n + 1].index

    recs = movies[movies["movieId"].isin(similar_movie_ids)][
        ["movieId", "title", "genres"]
    ]
    return recs


def recommend_similar_movies_content(movie_title, n=10):
    movie_row = movies_content[movies_content["title"].str.lower() == movie_title.lower()]
    if movie_row.empty:
        return pd.DataFrame(columns=["movieId", "title", "genres"])

    movie_id = movie_row.iloc[0]["movieId"]

    if movie_id not in content_similarity_df.index:
        return pd.DataFrame(columns=["movieId", "title", "genres"])

    scores = content_similarity_df[movie_id]
    similar_movie_ids = scores.sort_values(ascending=False).iloc[1:n + 1].index

    recs = movies[movies["movieId"].isin(similar_movie_ids)][
        ["movieId", "title", "genres"]
    ]
    return recs


def recommend_for_user(user_id, n=10):
    if user_id not in user_item_matrix_filled.index:
        return pd.DataFrame(columns=["movieId", "title", "genres", "predicted_rating"])

    user_ratings = user_item_matrix_filled.loc[user_id]
    already_rated = user_ratings[user_ratings > 0].index.tolist()

    if len(already_rated) == 0:
        return pd.DataFrame(columns=["movieId", "title", "genres", "predicted_rating"])

    user_ratings_values = user_ratings.values
    similarity_matrix_values = movie_similarity_df.values

    numerator = similarity_matrix_values.dot(user_ratings_values)
    rated_mask = (user_ratings_values > 0).astype(int)
    denominator = (abs(similarity_matrix_values) * rated_mask).sum(axis=1)

    with pd.option_context("mode.use_inf_as_na", True):
        predicted_ratings = numerator / denominator

    predicted_ratings = pd.Series(
        predicted_ratings,
        index=movie_similarity_df.index
    ).fillna(0)

    predicted_ratings = predicted_ratings.drop(index=already_rated)

    top_movies = predicted_ratings.sort_values(ascending=False).head(n)

    recs = movies[movies["movieId"].isin(top_movies.index)][
        ["movieId", "title", "genres"]
    ].copy()

    recs["predicted_rating"] = recs["movieId"].map(top_movies)
    recs = recs.sort_values("predicted_rating", ascending=False)

    return recs

# --------------------------------------------------
# 4. Streamlit UI
# --------------------------------------------------
st.title("🎬 Movie Recommendation System")
st.caption(
    "Hybrid Recommendation Engine | Collaborative Filtering • Content-Based Filtering • Personalized Suggestions"
)

st.sidebar.header("Recommendation Controls")

mode = st.sidebar.radio(
    "Choose Recommendation Mode",
    [
        "Similar Movies (Collaborative Filtering)",
        "Similar Movies (Content-Based Filtering)",
        "Personalized User Recommendations"
    ]
)

all_titles = movies["title"].sort_values().unique()
all_user_ids = sorted(ratings["userId"].unique())

if mode == "Similar Movies (Collaborative Filtering)":
    st.subheader("Similar Movies – Collaborative Filtering")
    movie_title = st.selectbox("Select a movie:", all_titles)
    top_n = st.slider("Number of recommendations:", 3, 20, 10)

    if st.button("Get Recommendations"):
        recs = recommend_similar_movies_collab(movie_title, n=top_n)
        if recs.empty:
            st.warning("No recommendations found. Try a different movie.")
        else:
            st.dataframe(recs.reset_index(drop=True), use_container_width=True)

elif mode == "Similar Movies (Content-Based Filtering)":
    st.subheader("Similar Movies – Content-Based Filtering")
    movie_title = st.selectbox("Select a movie:", all_titles, key="content_movie")
    top_n = st.slider("Number of recommendations:", 3, 20, 10, key="content_slider")

    if st.button("Get Content-Based Recommendations"):
        recs = recommend_similar_movies_content(movie_title, n=top_n)
        if recs.empty:
            st.warning("No recommendations found. Try a different movie.")
        else:
            st.dataframe(recs.reset_index(drop=True), use_container_width=True)

elif mode == "Personalized User Recommendations":
    st.subheader("Personalized User Recommendations")
    user_id = st.selectbox("Select a user ID:", all_user_ids)
    top_n = st.slider("Number of recommendations:", 3, 20, 10, key="user_slider")

    if st.button("Get User Recommendations"):
        recs = recommend_for_user(user_id, n=top_n)
        if recs.empty:
            st.warning("No recommendations found for this user.")
        else:
            st.dataframe(recs.reset_index(drop=True), use_container_width=True)
