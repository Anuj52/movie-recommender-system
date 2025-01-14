
# Movie Recommender System

This project is a **Movie Recommender System** built using **Streamlit**. The app uses a movie dataset to recommend similar movies based on a content-based filtering algorithm. Users can search for a movie, and the app will display its details along with recommendations for similar movies.

## Features

- **Search Movies**: Users can search for movies by title.
- **Movie Details**: Displays movie information such as genres, overview, keywords, cast, and more.
- **Movie Recommendations**: Recommends similar movies based on the selected movie.
- **Interactive UI**: Simple, clean, and user-friendly interface built with Streamlit.

## Technologies Used

- **Python** 3.x
- **Streamlit** for building the web application
- **Pandas** for data manipulation and analysis
- **NumPy** for numerical operations
- **Requests** for fetching movie posters from the API
- **Pickle** for saving and loading preprocessed movie data and similarity matrices

## Prerequisites

Before running the app, you need to install the necessary dependencies. Ensure you have Python 3.x installed on your system.

### Install Dependencies

To set up the environment, run the following command:

```bash
pip install pandas numpy requests streamlit
```

## Project Structure

The project structure should be as follows:

```
/Movie-Recommender-System
    /venv/                  # Virtual environment (optional)
    /app.py                 # Main Streamlit app file
    /movie_dict.pkl         # Pickled file containing preprocessed movie data
    /similarity.pkl         # Pickled file containing precomputed movie similarity matrix
    /tmdb_5000_movies.csv   # Raw movie dataset
    /tmdb_5000_credits.csv  # Raw movie credits dataset
    README.md               # Project documentation (this file)
```

- **`app.py`**: The main file that runs the Streamlit app.
- **`movie_dict.pkl`**: Pickle file storing processed movie data.
- **`similarity.pkl`**: Pickle file storing the precomputed similarity matrix.
- **`tmdb_5000_movies.csv`**: Raw dataset containing movie metadata.
- **`tmdb_5000_credits.csv`**: Raw dataset containing movie credits (cast and crew).

## Steps to Run the Application

### Step 1: Set Up the Virtual Environment

1. Create and activate a virtual environment:

   - On **Windows**:
     ```bash
     python -m venv venv
     venv\Scripts\activate
     ```

   - On **macOS/Linux**:
     ```bash
     python3 -m venv venv
     source venv/bin/activate
     ```

2. Install the required libraries:
   ```bash
   pip install pandas numpy requests streamlit
   ```

### Step 2: Prepare the Dataset

1. Download the datasets (`tmdb_5000_movies.csv` and `tmdb_5000_credits.csv`).
2. Use a Jupyter Notebook or Python script to preprocess the data by:
   - Merging the datasets on the movie title.
   - Handling missing values and duplicates.
   - Extracting relevant columns (movie_id, title, overview, genres, keywords, cast, and crew).
   - Saving the preprocessed movie data into `movie_dict.pkl` and the similarity matrix into `similarity.pkl`.

### Step 3: Run the Streamlit Application

1. Navigate to the project directory where `app.py` is located.
2. Run the Streamlit app using the following command:

   ```bash
   streamlit run app.py
   ```

   This will start the app and open it in your browser at `http://localhost:8501`.

### Step 4: Use the App

1. **Search for a Movie**: Type the movie title in the search box.
2. **View Movie Details**: The app will display the movie's genres, keywords, cast, and overview.
3. **View Recommendations**: After selecting a movie, the app will recommend similar movies based on content-based filtering.

## Code Explanation

1. **Movie Data Processing**:
   - Data is cleaned and preprocessed to handle missing values, remove duplicates, and convert certain columns (such as genres, keywords, and cast) into lists for easy handling.
   - The movie dataset is merged with the credits data (cast and crew).

2. **Similarity Calculation**:
   - The app uses a **content-based filtering** approach by computing similarities between movies based on their metadata (genres, keywords, cast).
   - Pre-computed similarity matrices are stored in a pickle file for fast retrieval.

3. **Fetching Movie Posters**:
   - The app fetches movie posters using the TMDb API, with error handling in case of a missing or incorrect poster.

4. **Streamlit UI**:
   - The app provides a simple interface to select a movie, view details, and get recommendations.

## Troubleshooting

- **Error: "Movie not found"**: If the selected movie is not found in the dataset, ensure that the movie title exists in the dataset or try another movie.
- **Error: "No recommendations found"**: If no recommendations are shown, this might be due to the similarity calculation being limited or incorrect.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- The datasets used are sourced from [The Movie Database (TMDb)]
- The content-based filtering algorithm is based on movie metadata such as genres, keywords, and cast.
- The **Streamlit** library was used for building the interactive web interface.

## Conclusion

This Movie Recommender System is a simple, yet powerful way to suggest movies to users based on their preferences. The content-based filtering method can be expanded or modified to include additional features such as runtime, director, and more.

```

This `README.md` file provides a complete and detailed guide on how to set up, use, and understand the Movie Recommender System. You can adjust the file further depending on your specific needs or project changes.
