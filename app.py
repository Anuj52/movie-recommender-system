import pandas as pd
import pickle
import streamlit as st
import requests


# Function to fetch movie poster
def fetch_poster(movie_id):
    url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key=8265bd1679663a7ea12ac168da84d2e8&language=en-US"
    data = requests.get(url)
    data = data.json()
    poster_path = data.get('poster_path')
    if poster_path:
        return f"https://image.tmdb.org/t/p/w500/{poster_path}"
    else:
        return "https://via.placeholder.com/500x750?text=No+Image+Available"  # Fallback image


# Recommendation function
def recommend(movie):
    # Ensure the movie exists in the dataset
    if movie not in movies['title'].values:
        return [], []

    # Find the index of the selected movie
    index = movies[movies['title'] == movie].index[0]

    # Sort the movies based on similarity
    distances = sorted(list(enumerate(similarity[index])), reverse=True, key=lambda x: x[1])

    recommended_movie_names = []
    recommended_movie_posters = []

    # Loop through the top 5 recommended movies
    for i in distances[1:6]:  # Skip the first as it's the selected movie itself
        movie_id = movies.iloc[i[0]].movie_id
        recommended_movie_posters.append(fetch_poster(movie_id))
        recommended_movie_names.append(movies.iloc[i[0]].title)

    return recommended_movie_names, recommended_movie_posters


# Streamlit App
st.header('Movie Recommender System')

# Load the movie dataset and similarity data
movies = pickle.load(open('movie_dict.pkl', 'rb'))

# Check if 'movies' is a dictionary and convert it to DataFrame
if isinstance(movies, dict):
    movies = pd.DataFrame(movies)

similarity = pickle.load(open('similarity.pkl', 'rb'))

# Ensure the movie dataset is valid
if 'title' not in movies or 'movie_id' not in movies:
    st.error("Movie data is invalid. Ensure the dataset is correctly loaded.")
    st.stop()

# Display the movie selection dropdown
movie_list = movies['title'].values
selected_movie = st.selectbox(
    "Type or select a movie from the dropdown",
    movie_list
)

# Show movie recommendations when button is pressed
if st.button('Show Recommendation'):
    recommended_movie_names, recommended_movie_posters = recommend(selected_movie)

    if not recommended_movie_names:
        st.warning(f"No recommendations found for '{selected_movie}'. Please select another movie.")
    else:
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.text(recommended_movie_names[0])
            st.image(recommended_movie_posters[0])
        with col2:
            st.text(recommended_movie_names[1])
            st.image(recommended_movie_posters[1])
        with col3:
            st.text(recommended_movie_names[2])
            st.image(recommended_movie_posters[2])
        with col4:
            st.text(recommended_movie_names[3])
            st.image(recommended_movie_posters[3])
        with col5:
            st.text(recommended_movie_names[4])
            st.image(recommended_movie_posters[4])
