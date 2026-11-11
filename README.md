# 🎬 CineMatch — Movie Recommendation & Personalization Engine

CineMatch is a movie recommendation application that helps users discover movies through content similarity, personalized recommendations and popularity-based ranking.

The project combines an expanded MovieLens movie catalog with MovieLens rating history and on-demand TMDB movie metadata.

**Application:** Streamlit
**Language:** Python 3.13
**Project type:** Recommendation Systems / Machine Learning

---

## 1. Features

### Similar Movies
- Search movies by title.
- Find similar movies using TF-IDF content representations.
- Uses movie genres and tags.
- Ranks movies using cosine similarity.

### Personalized Recommendations
- Select an existing MovieLens user.
- Builds a user preference profile from positively rated movies.
- Treats ratings of 4.0 and above as positive preferences.
- Excludes movies the user has already rated.

### Popular Movies
- Uses average ratings and rating counts.
- Applies a weighted-rating approach to reduce the influence of movies with very few ratings.

### Movie Details
- Movie posters.
- TMDB ratings and vote counts.
- Movie descriptions.
- Runtime and director.
- Cast information.
- Links to the movie's TMDB page.

TMDB information is retrieved on demand when a movie is displayed.

### Filters
- Genre selection.
- Release-year range.
- Number of recommendations.

---

## 2. Technology Stack

| Category | Technology |
|---|---|
| Programming language | Python |
| Application | Streamlit |
| Data processing | Pandas, NumPy |
| Machine learning | Scikit-learn |
| Content representation | TF-IDF |
| Similarity | Cosine similarity |
| Movie metadata | TMDB API |
| HTTP client | curl-cffi |
| Environment management | python-dotenv |

---

## 3. Recommendation Architecture

The current Version 1 application uses three recommendation methods.

**Content-Based Filtering**

Movie genres and tags are converted into TF-IDF vectors. Cosine similarity identifies movies with similar content.

**Personalized Content Recommendations**

The system averages the TF-IDF representations of movies a user rated positively. It compares this profile against the movie catalog and excludes previously rated movies.

**Popularity-Based Recommendations**

A weighted-rating formula combines a movie's average rating with the global rating average, taking its rating count into account.

### System flow

```text
MovieLens Datasets
       |
       v
Movie Catalog and Tags
       |
       v
IMDb Identity Crosswalk
       |
       v
Final Runtime Catalog
       |
       v
Movie Recommendation Service
       |
       +-------------------------+
       |            |            |
       v            v            v
   Similar      Personalized   Popularity
    Movies        Movies        Ranking
       |            |            |
       +------------+------------+
                    |
                    v
             Genre / Year Filters
                    |
                    v
              Streamlit UI
                    |
                    v
               TMDB API
                    |
                    v
        Posters and Movie Details
```

---

## 4. Dataset Information

CineMatch uses two MovieLens datasets for different purposes.

### MovieLens 32M

Used to construct the expanded movie catalog.

- 87,585 movies
- 32,000,204 ratings in the original dataset
- 2,000,072 tag applications in the original dataset
- 200,948 users in the original dataset

Official source:

https://grouplens.org/datasets/movielens/32m/

### MovieLens Latest-Small

Used for user-rating history and collaborative user information in Version 1.

- 9,742 movies
- 100,836 ratings
- 610 users

Official source:

https://grouplens.org/datasets/movielens/latest/

### Movie Identity Mapping

The MovieLens datasets have different movie identifiers. CineMatch connects their records using normalized IMDb identifiers instead of assuming that their movie IDs are interchangeable.

The tested runtime integration maps:
- 9,466 MovieLens-small movies.
- 100,291 rating records.
- 610 users.

The remaining records without a reliable identity mapping are excluded from the rating-based recommendation history.

---

## 5. Project Structure

```text
Movie_Recommendation/
│
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
├── .env.example
│
├── data/
│   ├── ratings.csv
│   ├── links.csv
│   │
│   └── catalog/
│       └── movielens_catalog_runtime.csv.gz
│
├── src/
│   ├── data/
│   │   ├── catalog/
│   │   ├── loaders/
│   │   └── preprocessing/
│   │
│   ├── evaluation/
│   │
│   ├── ranking/
│   │
│   ├── recommenders/
│   │   ├── baseline/
│   │   ├── collaborative/
│   │   ├── hybrid/
│   │   ├── popularity/
│   │   └── semantic/
│   │
│   └── services/
│       ├── recommendation_service.py
│       └── tmdb_client.py
│
└── experiments/
    └── results/
```

The runtime catalog contains 87,585 movie records and seven columns. It is compressed using gzip to reduce deployment size.

The larger source catalogs, raw MovieLens archive, TMDB enrichment cache and intermediate repair files are kept outside the public application runtime.

---

## 6. Local Installation

### Prerequisites

- Python 3.13
- Git
- Internet connection for TMDB movie details

### Step 1 — Clone the repository

```bash
git clone https://github.com/abhigandesri13/Moive_Recommendation_System.git
cd Movie_Recommendation
```

### Step 2 — Create a virtual environment

```bash
python -m venv venv
```

Activate it on Windows:

```powershell
venv\Scripts\activate
```

### Step 3 — Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Step 4 — Configure the TMDB API key

Create a local `.env` file in the project root:

```text
TMDB_API_KEY=your_tmdb_api_key
```

Do not commit `.env` to GitHub.

Obtain an API key through:

https://www.themoviedb.org/settings/api

### Step 5 — Start the application

```bash
python -m streamlit run app.py
```

Open:

http://localhost:8501

---

## 7. Streamlit Cloud Deployment

**Live App:** https://ai-movie-recommendation-system.streamlit.app/

1. Push the project to GitHub.
2. Open https://share.streamlit.io/
3. Select **Create app**.
4. Select your GitHub repository and branch.
5. Set the main application file to `app.py`.
6. Open Advanced settings and select Python 3.13.
7. Add the following under Secrets:

```toml
TMDB_API_KEY = "your_tmdb_api_key"
```

8. Deploy the application.

The repository must contain:
- `app.py`
- `requirements.txt`
- `src/`
- `data/ratings.csv`
- `data/links.csv`
- `data/catalog/movielens_catalog_runtime.csv.gz`

Never upload `.env`, API keys or the TMDB enrichment cache.

Official deployment documentation:

https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy

---

## 8. Evaluation and Experiments

The repository includes experimental implementations and evaluation scripts for:

- Bayesian Personalized Ranking (BPR).
- Collaborative filtering.
- Content-based recommendation.
- Semantic embedding recommendation.
- Hybrid candidate generation.
- LightGBM ranking.
- Candidate recall analysis.
- Precision@K, Recall@K and NDCG@K evaluation.
- Model and hyperparameter experiments.

These experimental components are retained in the repository. The currently deployed Version 1 application uses TF-IDF content recommendations, personalized content profiles and weighted popularity. BPR and LightGBM are not part of the active Version 1 inference pipeline.

---

## 9. API Attribution

Movie metadata, posters and images are retrieved through The Movie Database (TMDB) API.

The application must include an approved TMDB logo and the following notice in its About or Credits section:

> This product uses the TMDB API but is not endorsed or certified by TMDB.

TMDB official attribution guidelines:

https://www.themoviedb.org/about/logos-attribution

TMDB developer documentation:

https://developer.themoviedb.org/

---

## 10. Dataset Acknowledgment

MovieLens datasets are provided by GroupLens Research.

Reference:

Harper, F. M., and Konstan, J. A. (2015).
*The MovieLens Datasets: History and Context.*
ACM Transactions on Interactive Intelligent Systems, 5(4), Article 19.

https://doi.org/10.1145/2827872

Review the applicable MovieLens dataset README and usage conditions before redistribution or commercial use. This project is intended as a non-commercial educational and research project.

---

## 11. Future Improvements

Planned enhancements include:

- BPR-based collaborative filtering integration.
- Full hybrid recommendation and ranking pipeline.
- Improved recommendation diversity.
- Bollywood, Tollywood and Hollywood classification.
- Verified language and dubbed-language filters.
- Advanced user personalization.
- Model persistence and loading.
- FastAPI integration.
- Docker-based deployment.
- Recommendation monitoring and evaluation.

---

## 12. Project Status

**Version:** 1.0 MVP

Implemented:
- Expanded movie catalog.
- IMDb identity crosswalk.
- TMDB metadata integration.
- Content-based recommendations.
- Personalized content recommendations.
- Popularity-based recommendations.
- Genre and release-year filters.
- Streamlit user interface.
- On-demand posters and movie details.

**Developed using Python, Scikit-learn, Streamlit, MovieLens and TMDB.**