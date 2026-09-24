# 🎬 CineSense — Movie Recommender & Rating Predictor

A machine learning-powered web app that recommends similar movies and predicts a movie's rating based on its features, built using the TMDB 5000 Movie Dataset.

## Features

- **Movie Recommender**: Content-based filtering using cosine similarity — select a movie, get 5 similar recommendations based on genre, cast, crew, and plot keywords.
- **Rating Predictor**: A Random Forest Regression model predicts a movie's rating (out of 10) from its budget, popularity, runtime, and vote count.

## Tech Stack

- **Data Processing**: Python, Pandas, NumPy
- **Machine Learning**: scikit-learn (CountVectorizer, Cosine Similarity, Random Forest Regressor)
- **Backend**: Flask
- **Frontend**: HTML, CSS (Jinja templating)
- **Development**: Google Colab (model training), VS Code (deployment)

## How It Works

1. **Data Cleaning**: Merged and cleaned the TMDB movies and credits datasets, extracting genres, keywords, top cast, and director from nested JSON fields.
2. **Feature Engineering**: Combined overview, genres, keywords, cast, and crew into a single "tags" field per movie.
3. **Recommendation Engine**: Vectorized tags using `CountVectorizer` (5000 features) and computed pairwise cosine similarity across all movies.
4. **Rating Prediction**: Trained a Random Forest Regressor on budget, popularity, runtime, and vote count (R² = 0.43, an improvement over a baseline Linear Regression's R² = 0.32).
5. **Deployment**: Wrapped both models in a Flask app with a simple web interface.

## Model Performance

| Model | MAE | R² Score |
|-------|-----|----------|
| Linear Regression | 0.54 | 0.32 |
| Random Forest | 0.49 | 0.43 |

## Running Locally

Model files (`.pkl`) are tracked with [Git LFS](https://git-lfs.com) due to their size (`similarity.pkl` is ~176MB). To run this project:

1. Clone this repo (make sure `git-lfs` is installed so the `.pkl` files download correctly: `git lfs install`)
2. Install dependencies: `pip install -r requirements.txt`
3. Run: `python app.py`
4. Visit `http://127.0.0.1:5000`

## Deployment

This app is set up to deploy on [Render](https://render.com) via `render.yaml`:

1. Push this repo to GitHub (with Git LFS objects included).
2. On Render, create a new **Blueprint** and point it at this repo — it will pick up `render.yaml` automatically (build: `git lfs pull && pip install -r requirements.txt`, start: `gunicorn app:app`).

## Dataset

[TMDB 5000 Movie Dataset](https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata) (Kaggle)
