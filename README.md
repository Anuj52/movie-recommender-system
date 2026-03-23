# Movie Matchmaker

Movie Matchmaker is a Streamlit movie discovery app built on a content-based recommender. It helps users explore similar movies, search by actor or director, open trailers, save favorites to a watchlist, and browse curated mood collections.

## Features

- **Movie, actor, and director search** to anchor recommendations from different entry points.
- **Rich movie profiles** with poster, overview, tagline, cast, director, runtime, year, rating, and trailer.
- **Advanced filters** for genre, mood, language, vote count, rating, year range, runtime, and sort order.
- **Similarity badges and explanations** so each match shows why it was recommended.
- **Watchlist support** with local persistence and JSON download.
- **Mood collections** for quick browsing across `Feel-Good`, `Action Night`, `Family Time`, and `Mind-Bending`.
- **Deployment-ready setup** with Streamlit theme config and Python runtime pinning.

## Tech Stack

- **Python**
- **Streamlit**
- **Pandas**
- **NumPy**
- **Requests**
- **scikit-learn**

## Setup

### 1. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 2. Add your TMDb API key

The app can read the TMDb key from any of these sources:

- Local `.env` file:
  `TMDB_API_KEY=YOUR_KEY`
- Environment variable:
  `TMDB_API_KEY`
- Streamlit secrets:
  `.streamlit/secrets.toml` with `TMDB_API_KEY = "YOUR_KEY"`

If no API key is set, the app still works, but posters and trailers fall back to limited behavior.

### 3. Run the app

```bash
python -m streamlit run app.py
```

The default local URL is:

```text
http://localhost:8501
```

## How It Works

### Recommender

- The app loads `movie_dict.pkl` and `similarity.pkl` when they are available.
- If those files are missing, it rebuilds recommendation vectors from `tmdb_5000_movies.csv` and `tmdb_5000_credits.csv`.
- Recommendations are based on content signals such as overview, genres, keywords, cast, and director.

### Discovery Flow

- Users can browse by **movie**, **actor**, or **director**.
- The sidebar applies filters before ranking the recommendation list.
- Results can be sorted by **similarity**, **rating**, **popularity**, or **newest**.

### Watchlist

- Saved movies are written to `watchlist.json`.
- The file is ignored by git through `.gitignore`.
- Users can also download the watchlist as a JSON export from inside the app.

## Project Structure

```text
/movie-recommender-system
    /app.py
    /requirements.txt
    /.env
    /.gitignore
    /.streamlit/config.toml
    /runtime.txt
    /movie_dict.pkl
    /similarity.pkl
    /tmdb_5000_movies.csv
    /tmdb_5000_credits.csv
    /watchlist.json
    README.md
```

## Deployment

### Streamlit Community Cloud

1. Push the project to GitHub.
2. Make sure `requirements.txt`, `.streamlit/config.toml`, and `runtime.txt` are included.
3. In Streamlit Community Cloud, create a new app from the repository.
4. Set `TMDB_API_KEY` in the app secrets.
5. Use `app.py` as the entry file.

### Notes for deployment

- Local `.env` is for local development only and should not be committed.
- `watchlist.json` is local server storage, so on hosted platforms it may be temporary or shared depending on the deployment setup.
- For production-grade user accounts and private watchlists, the next step would be adding a real database.

## Troubleshooting

- **`pip` launcher error on Windows**:
  Use `python -m pip install -r requirements.txt` instead of `pip install -r requirements.txt`.
- **No posters or trailers**:
  Make sure `TMDB_API_KEY` is set correctly.
- **Missing `similarity.pkl`**:
  Install `scikit-learn` so the app can rebuild vectors from the CSV files.
- **No recommendations after filtering**:
  Lower the minimum rating or vote threshold, or widen the year/runtime range.

## License

This project is licensed under the MIT License. See `LICENSE` for details.
