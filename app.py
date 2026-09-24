from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd

app = Flask(__name__)

# Saved files load cheyyuka
movies = pickle.load(open('movies.pkl', 'rb'))
similarity = pickle.load(open('similarity.pkl', 'rb'))
rating_model = pickle.load(open('rating_model.pkl', 'rb'))

# Recommend function (Colab-il undakkiyath thanne)
def recommend(movie):
    movie_index = movies[movies['title_x'] == movie].index[0]
    distances = similarity[movie_index]
    movies_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]
    return [movies.iloc[i[0]].title_x for i in movies_list]

# Homepage route
@app.route('/')
def home():
    movie_titles = movies['title_x'].tolist()
    return render_template('index.html', movies=movie_titles)

# Recommend route
@app.route('/recommend', methods=['POST'])
def get_recommendation():
    movie = request.form['movie']
    recommendations = recommend(movie)
    movie_titles = movies['title_x'].tolist()
    return render_template('index.html', movies=movie_titles, recommendations=recommendations, selected=movie)


@app.route('/predict', methods=['POST'])
def predict_rating():
    budget = float(request.form['budget'])
    popularity = float(request.form['popularity'])
    runtime = float(request.form['runtime'])
    vote_count = float(request.form['vote_count'])

    input_data = pd.DataFrame([[budget, popularity, runtime, vote_count]],
                                columns=['budget', 'popularity', 'runtime', 'vote_count'])
    
    predicted_rating = rating_model.predict(input_data)[0]
    
    movie_titles = movies['title_x'].tolist()
    return render_template('index.html', movies=movie_titles, predicted_rating=round(predicted_rating, 2))


if __name__ == '__main__':
    app.run(debug=True)
