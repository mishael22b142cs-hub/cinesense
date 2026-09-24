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

Model files (`.pkl`) are not included in this repo due to GitHub's file size limits. To run this project:

1. Clone this repo
2. Run the data preprocessing + model training steps in a Colab notebook (using the TMDB 5000 Dataset) to generate `movies.pkl`, `similarity.pkl`, and `rating_model.pkl`
3. Place the `.pkl` files in the project root
4. Install dependencies: `pip install flask pandas scikit-learn`
5. Run: `python app.py`
6. Visit `http://127.0.0.1:5000`

## Dataset

[TMDB 5000 Movie Dataset](https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata) (Kaggle)
