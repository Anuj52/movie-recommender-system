from __future__ import annotations

import ast
import configparser
import html
import json
import os
import pickle
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional
from urllib.parse import quote, urlencode

import numpy as np
import pandas as pd
import requests
import streamlit as st

st.set_page_config(page_title="Movie Matchmaker", layout="wide")

APP_DIR = Path(__file__).resolve().parent
MOVIE_DICT_PATH = APP_DIR / "movie_dict.pkl"
SIMILARITY_PATH = APP_DIR / "similarity.pkl"
MOVIES_CSV_PATH = APP_DIR / "tmdb_5000_movies.csv"
CREDITS_CSV_PATH = APP_DIR / "tmdb_5000_credits.csv"
ENV_PATH = APP_DIR / ".env"
WATCHLIST_PATH = APP_DIR / "watchlist.json"

NO_IMAGE_URL = "https://via.placeholder.com/500x750?text=No+Image+Available"
DEFAULT_RECOMMENDATIONS = 9
DEFAULT_MIN_RATING = 6.0
DEFAULT_MIN_VOTES = 200
SECTION_OPTIONS = ["Recommendations", "Mood Collections", "Watchlist"]
RECOMMENDATION_TYPES = [
    "Content-Based",
    "Collaborative",
    "Hybrid",
    "Popularity-Based",
]
AGE_GROUP_GENRE_BOOSTS: dict[str, set[str]] = {
    "All Ages": set(),
    "Kids & Family": {"Animation", "Family", "Adventure", "Fantasy", "Comedy"},
    "Teens": {"Adventure", "Action", "Fantasy", "Science Fiction", "Comedy"},
    "Adults": {"Drama", "Thriller", "Crime", "Mystery", "War"},
}
AGE_GROUP_MOOD_BOOSTS: dict[str, set[str]] = {
    "All Ages": set(),
    "Kids & Family": {"Feel-Good", "Family Time"},
    "Teens": {"Action Night", "Mind-Bending"},
    "Adults": {"Mind-Bending", "Action Night"},
}
SEASONAL_GENRE_BOOSTS: dict[str, set[str]] = {
    "Any": set(),
    "Spring": {"Romance", "Comedy", "Music"},
    "Summer": {"Adventure", "Action", "Family"},
    "Autumn": {"Drama", "Mystery", "Fantasy"},
    "Winter": {"Fantasy", "Family", "Drama"},
}
DAY_CONTEXT_MOOD_BOOSTS: dict[str, set[str]] = {
    "Any Day": set(),
    "Weekday": {"Feel-Good", "Mind-Bending"},
    "Weekend": {"Action Night", "Family Time"},
}
HOLIDAY_GENRE_BOOSTS = {"Family", "Animation", "Romance", "Fantasy", "Comedy"}

MOOD_RULES: dict[str, dict[str, Any]] = {
    "Feel-Good": {
        "description": "Warm, funny, comforting stories with romance, family, or friendship at the center.",
        "genres": {"Comedy", "Family", "Romance", "Animation", "Music"},
        "keywords": {"friendship", "love", "family", "wedding", "holiday", "coming of age", "road trip"},
        "overview_terms": {"friend", "love", "family", "wedding", "joy", "summer"},
    },
    "Action Night": {
        "description": "Fast, punchy picks for a big-screen action mood.",
        "genres": {"Action", "Adventure", "Thriller", "Crime", "War"},
        "keywords": {"battle", "mission", "hero", "weapon", "war", "police", "explosion"},
        "overview_terms": {"mission", "battle", "fight", "danger", "soldier", "agent"},
    },
    "Family Time": {
        "description": "Lighter adventures and all-ages stories that play well together.",
        "genres": {"Family", "Animation", "Adventure", "Fantasy"},
        "keywords": {"family", "child", "magic", "friendship", "animal", "fairy tale"},
        "overview_terms": {"child", "family", "magical", "adventure", "young"},
    },
    "Mind-Bending": {
        "description": "Twisty sci-fi, mystery, and thrillers with puzzle energy.",
        "genres": {"Science Fiction", "Mystery", "Thriller", "Fantasy"},
        "keywords": {"dream", "memory", "time travel", "future", "psychological", "alternate reality"},
        "overview_terms": {"memory", "reality", "future", "time", "mind", "mystery"},
    },
}


def load_env_file(env_path: Path) -> None:
    if not env_path.exists():
        return

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("'").strip('"')
        if key:
            os.environ.setdefault(key, value)


load_env_file(ENV_PATH)

try:
    TMDB_API_KEY = st.secrets.get("TMDB_API_KEY")
except Exception:
    TMDB_API_KEY = None

TMDB_API_KEY = TMDB_API_KEY or os.getenv("TMDB_API_KEY")


def get_github_repo_url() -> str:
    try:
        configured = st.secrets.get("GITHUB_REPO_URL")
    except Exception:
        configured = None

    configured = configured or os.getenv("GITHUB_REPO_URL")
    if configured:
        return str(configured)

    git_config_path = APP_DIR / ".git" / "config"
    if git_config_path.exists():
        parser = configparser.ConfigParser()
        parser.read(git_config_path, encoding="utf-8")
        remote_section = 'remote "origin"'
        if parser.has_section(remote_section):
            url = parser.get(remote_section, "url", fallback="").strip()
            if url.startswith("git@github.com:"):
                return "https://github.com/" + url.split(":", 1)[1].removesuffix(".git")
            if url:
                return url.removesuffix(".git")

    return "https://github.com"


def get_app_base_url() -> str:
    try:
        configured = st.secrets.get("APP_URL")
    except Exception:
        configured = None

    configured = configured or os.getenv("APP_URL")
    if configured:
        return str(configured).rstrip("/")

    try:
        context = getattr(st, "context", None)
        headers = getattr(context, "headers", {}) if context else {}
        host = headers.get("Host") or headers.get("host")
        proto = headers.get("X-Forwarded-Proto") or headers.get("x-forwarded-proto") or "https"
        if host:
            return f"{proto}://{host}".rstrip("/")
    except Exception:
        pass

    return ""


def get_share_url() -> str:
    base_url = get_app_base_url()
    params: dict[str, str] = {}
    panel = str(st.session_state.get("active_panel", "Recommendations"))
    if panel in SECTION_OPTIONS and panel != "Recommendations":
        params["panel"] = panel

    selected_movie_id = st.session_state.get("selected_movie_id_for_share")
    if selected_movie_id:
        params["movie"] = str(selected_movie_id)

    if not base_url:
        return get_github_repo_url()

    if not params:
        return base_url

    return base_url + "?" + urlencode(params)


def sync_query_state(movie_id_to_index: dict[int, int]) -> None:
    try:
        panel = str(st.query_params.get("panel", "")).strip()
        if panel in SECTION_OPTIONS:
            st.session_state["active_panel"] = panel

        movie_param = str(st.query_params.get("movie", "")).strip()
        if movie_param.isdigit():
            movie_id = int(movie_param)
            if movie_id in movie_id_to_index:
                st.session_state["browse_mode"] = "Movie"
                st.session_state["selected_movie_index"] = movie_id_to_index[movie_id]
    except Exception:
        return


def sync_query_params(selected_movie_id: int) -> None:
    st.session_state["selected_movie_id_for_share"] = int(selected_movie_id)
    try:
        active_panel = str(st.session_state.get("active_panel", "Recommendations"))
        if active_panel == "Recommendations":
            st.query_params.clear()
        else:
            st.query_params["panel"] = active_panel
        st.query_params["movie"] = str(int(selected_movie_id))
    except Exception:
        pass


def inject_styles() -> None:
    st.markdown(
        """
        <style>
            .stApp {
                background:
                    radial-gradient(circle at top right, rgba(212, 185, 143, 0.18), transparent 24%),
                    linear-gradient(180deg, #f5efe3 0%, #f9f5ed 60%, #fcfaf5 100%);
                color: #211912;
                font-family: "Segoe UI", "Inter", "Helvetica Neue", Arial, sans-serif;
            }
            .block-container {
                max-width: 1320px;
                padding-top: 3.6rem;
                padding-bottom: 2.5rem;
            }
            header[data-testid="stHeader"] {
                background: rgba(245, 239, 227, 0.92);
                backdrop-filter: blur(8px);
            }
            [data-testid="stSidebar"] {
                background: linear-gradient(180deg, #2b333f 0%, #313a46 100%);
                border-right: none;
            }
            [data-testid="stSidebar"] * {
                color: #f3f5f8 !important;
            }
            [data-testid="stSidebar"] [data-baseweb="base-input"] > div,
            [data-testid="stSidebar"] [data-baseweb="select"] > div,
            [data-testid="stSidebar"] .stTextInput input,
            [data-testid="stSidebar"] .stNumberInput input {
                background: rgba(255, 255, 255, 0.08);
                border: 1px solid rgba(255, 255, 255, 0.1);
                color: #ffffff;
            }
            [data-testid="stSidebar"] [data-baseweb="tag"] {
                background: rgba(255, 255, 255, 0.1);
                color: #ffffff;
            }
            [data-testid="stSidebar"] .stSlider [data-baseweb="slider"] {
                padding-top: 0.2rem;
            }
            h1, h2, h3, h4, h5, h6 {
                color: #23180f !important;
                font-family: "Segoe UI", "Inter", "Helvetica Neue", Arial, sans-serif !important;
                letter-spacing: -0.02em;
            }
            .topbar {
                align-items: center;
                background: linear-gradient(180deg, #252c36 0%, #1f2630 100%);
                border-radius: 16px;
                box-shadow: 0 12px 28px rgba(20, 24, 29, 0.2);
                display: flex;
                justify-content: space-between;
                margin-bottom: 1.15rem;
                overflow: hidden;
                padding: 0.85rem 1.2rem;
            }
            .topbar-brand {
                align-items: center;
                color: #f9fafb;
                display: flex;
                font-size: 1.05rem;
                font-weight: 600;
                gap: 0.7rem;
                min-width: 0;
            }
            .topbar-icon {
                font-size: 1rem;
            }
            .topbar-actions {
                color: #dbe1ea;
                display: flex;
                flex-wrap: wrap;
                gap: 1rem;
                font-size: 0.95rem;
                justify-content: flex-end;
                padding-right: 4.5rem;
            }
            .topbar-actions a {
                color: #dbe1ea !important;
                text-decoration: none;
            }
            .topbar-actions a:hover {
                text-decoration: underline;
            }
            .topbar-actions span {
                opacity: 0.9;
            }
            .hero-card {
                margin-bottom: 0.4rem;
            }
            .eyebrow {
                color: #3e3124;
                font-size: 0.84rem;
                font-weight: 500;
                margin-bottom: 0.4rem;
            }
            .hero-title {
                color: #2b1c0f;
                font-family: Georgia, "Times New Roman", serif;
                font-size: 3rem;
                font-weight: 700;
                line-height: 1.12;
                margin: 0;
            }
            .hero-tagline {
                color: #2f241b;
                font-size: 1rem;
                font-style: normal;
                margin-top: 0.45rem;
                margin-bottom: 0.85rem;
            }
            .hero-copy,
            .panel-copy {
                color: #34281d;
                font-size: 0.98rem;
                line-height: 1.6;
                margin: 1rem 0 0;
            }
            .results-note {
                color: #6e5d4b;
                margin-top: 0.15rem;
                margin-bottom: 1rem;
            }
            .token-row {
                margin: 0.35rem 0 0.15rem;
            }
            .token {
                display: inline-block;
                border-radius: 999px;
                padding: 0.33rem 0.72rem;
                margin: 0 0.38rem 0.38rem 0;
                font-size: 0.8rem;
                font-weight: 500;
                border: 1px solid rgba(181, 163, 132, 0.2);
            }
            .token-chip {
                background: #f2eadc;
                color: #58462f;
            }
            .token-badge {
                background: #efe3ce;
                color: #5f4930;
            }
            .token-mood {
                background: #e8efe9;
                color: #355243;
            }
            .mini-title {
                color: #24180e;
                font-size: 1.05rem;
                font-weight: 600;
                line-height: 1.3;
                margin: 0.55rem 0 0.15rem;
            }
            .meta-line {
                color: #6a5948;
                font-size: 0.85rem;
                margin: 0.15rem 0 0.4rem;
            }
            .section-kicker {
                color: #6e5d4b;
                font-size: 0.92rem;
                margin-top: -0.15rem;
                margin-bottom: 1rem;
            }
            [data-testid="stImage"] img {
                border-radius: 12px;
                box-shadow: 0 8px 20px rgba(55, 39, 18, 0.16);
            }
            [data-testid="metric-container"] {
                background: rgba(255, 250, 242, 0.88);
                border: 1px solid rgba(191, 173, 143, 0.35);
                border-radius: 12px;
                padding: 0.35rem 0.65rem;
                box-shadow: none;
            }
            .stat-grid {
                display: grid;
                gap: 0.65rem;
                grid-template-columns: repeat(5, minmax(0, 1fr));
                margin: 0.9rem 0 1rem;
            }
            .stat-card {
                background: rgba(255, 250, 242, 0.88);
                border: 1px solid rgba(191, 173, 143, 0.35);
                border-radius: 12px;
                padding: 0.7rem 0.85rem;
            }
            .stat-label {
                color: #6a5948;
                font-size: 0.86rem;
                margin-bottom: 0.3rem;
            }
            .stat-value {
                color: #24180e;
                font-size: 1.15rem;
                font-weight: 600;
                line-height: 1.2;
                white-space: nowrap;
            }
            .detail-stack {
                color: #3e3024;
                display: flex;
                flex-direction: column;
                gap: 0.35rem;
                margin-top: 0.85rem;
            }
            .detail-item {
                line-height: 1.45;
            }
            .detail-item strong {
                color: #24180e;
                font-weight: 600;
            }
            [data-testid="stButton"] > button,
            [data-testid="stLinkButton"] a {
                align-items: center;
                background: linear-gradient(180deg, #363d47 0%, #262d37 100%);
                border: 1px solid #252c35;
                border-radius: 10px;
                box-shadow: none;
                color: #ffffff !important;
                display: inline-flex;
                font-weight: 500;
                gap: 0.35rem;
                justify-content: center;
                min-height: 2.5rem;
                padding: 0.55rem 0.9rem;
                text-decoration: none !important;
                transition: transform 0.16s ease, opacity 0.16s ease;
            }
            [data-testid="stButton"] > button:hover,
            [data-testid="stLinkButton"] a:hover {
                transform: translateY(-1px);
            }
            [data-testid="stButton"] > button[kind="secondary"] {
                background: rgba(255, 250, 242, 0.9);
                border: 1px solid rgba(191, 173, 143, 0.45);
                color: #2d241c !important;
            }
            [data-testid="stLinkButton"] a p,
            [data-testid="stButton"] > button p {
                color: inherit !important;
                font-weight: inherit;
            }
            [data-testid="stVerticalBlockBorderWrapper"] {
                background: rgba(255, 251, 244, 0.92);
                border: 1px solid rgba(193, 177, 150, 0.4);
                border-radius: 18px;
                box-shadow: 0 12px 30px rgba(74, 54, 25, 0.08);
            }
            [data-baseweb="tab-list"] {
                gap: 0.5rem;
            }
            [data-baseweb="tab"] {
                background: rgba(255, 249, 241, 0.72);
                border-radius: 999px;
                color: #5a4938 !important;
                padding: 0.35rem 0.9rem;
            }
            [data-baseweb="tab"][aria-selected="true"] {
                background: #2d3641;
                color: #ffffff !important;
            }
            [data-testid="stPopover"] button {
                background: transparent;
                border: none;
                box-shadow: none;
                color: #5f4f3e !important;
                min-height: auto;
                padding: 0.1rem 0;
                text-decoration: underline;
            }
            @media (max-width: 1200px) {
                .topbar {
                    gap: 0.8rem;
                    padding-right: 1rem;
                }
                .topbar-actions {
                    display: none;
                }
                .hero-title {
                    font-size: 2.35rem;
                }
                .stat-grid {
                    grid-template-columns: repeat(2, minmax(0, 1fr));
                }
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_topbar() -> None:
    github_url = get_github_repo_url()
    share_url = get_share_url()
    watchlist_url = "?panel=Watchlist"
    share_href = (
        "mailto:?subject="
        + quote("Check out Movie Matchmaker")
        + "&body="
        + quote(f"Take a look at Movie Matchmaker: {share_url}")
    )
    st.markdown(
        f"""
        <div class='topbar'>
            <div class='topbar-brand'>
                <span class='topbar-icon'>MM</span>
                <span>Movie Matchmaker</span>
            </div>
            <div class='topbar-actions'>
                <a href="{html.escape(share_href, quote=True)}">Share</a>
                <a href="{html.escape(github_url, quote=True)}" target="_blank">GitHub</a>
                <a href="{html.escape(watchlist_url, quote=True)}">Watchlist</a>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


@st.cache_resource
def get_http_session() -> requests.Session:
    session = requests.Session()
    session.headers.update({"Accept": "application/json"})
    return session


@st.cache_data(show_spinner=False, ttl=24 * 60 * 60)
def _fetch_poster_cached(movie_id: int) -> str:
    if not TMDB_API_KEY:
        return NO_IMAGE_URL

    try:
        response = get_http_session().get(
            f"https://api.themoviedb.org/3/movie/{movie_id}",
            params={"api_key": TMDB_API_KEY, "language": "en-US"},
            timeout=10,
        )
        response.raise_for_status()
        data = response.json()
    except (requests.RequestException, ValueError):
        return NO_IMAGE_URL

    poster_path = data.get("poster_path")
    if not poster_path:
        return NO_IMAGE_URL

    return f"https://image.tmdb.org/t/p/w500{poster_path}"


def fetch_poster(movie_id: int) -> str:
    if not TMDB_API_KEY:
        return NO_IMAGE_URL
    return _fetch_poster_cached(int(movie_id))


@st.cache_data(show_spinner=False, ttl=12 * 60 * 60)
def fetch_trailer_url(movie_id: int) -> str:
    if not TMDB_API_KEY:
        return ""

    try:
        response = get_http_session().get(
            f"https://api.themoviedb.org/3/movie/{movie_id}/videos",
            params={"api_key": TMDB_API_KEY, "language": "en-US"},
            timeout=10,
        )
        response.raise_for_status()
        data = response.json()
    except (requests.RequestException, ValueError):
        return ""

    results = data.get("results", [])
    if not isinstance(results, list):
        return ""

    def sort_key(video: dict[str, Any]) -> tuple[int, int, int]:
        is_official = int(bool(video.get("official")))
        is_trailer = int(str(video.get("type", "")).lower() == "trailer")
        has_youtube = int(str(video.get("site", "")).lower() == "youtube")
        return has_youtube, is_trailer, is_official

    for video in sorted(results, key=sort_key, reverse=True):
        if str(video.get("site", "")).lower() != "youtube":
            continue
        key = str(video.get("key", "")).strip()
        if key:
            return f"https://www.youtube.com/watch?v={key}"
    return ""


def safe_rerun() -> None:
    try:
        st.rerun()
    except AttributeError:
        st.experimental_rerun()


def load_watchlist_ids() -> set[int]:
    if not WATCHLIST_PATH.exists():
        return set()

    try:
        payload = json.loads(WATCHLIST_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return set()

    if not isinstance(payload, list):
        return set()

    watchlist: set[int] = set()
    for item in payload:
        try:
            watchlist.add(int(item))
        except (TypeError, ValueError):
            continue
    return watchlist


def save_watchlist_ids(movie_ids: set[int]) -> None:
    WATCHLIST_PATH.write_text(
        json.dumps(sorted(movie_ids), indent=2),
        encoding="utf-8",
    )


def ensure_session_state(default_movie_index: int) -> None:
    st.session_state.setdefault("watchlist_ids", load_watchlist_ids())
    st.session_state.setdefault("watchlist_notice", "")
    st.session_state.setdefault("browse_mode", "Movie")
    st.session_state.setdefault("selected_movie_index", int(default_movie_index))
    st.session_state.setdefault("selected_actor", "")
    st.session_state.setdefault("selected_director", "")
    st.session_state.setdefault("actor_anchor_index", int(default_movie_index))
    st.session_state.setdefault("director_anchor_index", int(default_movie_index))
    st.session_state.setdefault("active_panel", "Recommendations")
    st.session_state.setdefault("selected_movie_id_for_share", None)
    st.session_state.setdefault("interaction_history_ids", [])


def toggle_watchlist(movie: pd.Series) -> None:
    movie_id = int(movie["movie_id"])
    title = str(movie["title"])
    watchlist_ids = set(st.session_state.get("watchlist_ids", set()))

    if movie_id in watchlist_ids:
        watchlist_ids.remove(movie_id)
        st.session_state["watchlist_notice"] = f"Removed {title} from your watchlist."
    else:
        watchlist_ids.add(movie_id)
        st.session_state["watchlist_notice"] = f"Saved {title} to your watchlist."

    st.session_state["watchlist_ids"] = watchlist_ids
    save_watchlist_ids(watchlist_ids)


def clear_watchlist() -> None:
    st.session_state["watchlist_ids"] = set()
    save_watchlist_ids(set())
    st.session_state["watchlist_notice"] = "Cleared your watchlist."


def show_watchlist_notice() -> None:
    notice = str(st.session_state.get("watchlist_notice", "")).strip()
    if notice:
        st.success(notice)
        st.session_state["watchlist_notice"] = ""


def is_in_watchlist(movie_id: int) -> bool:
    return int(movie_id) in set(st.session_state.get("watchlist_ids", set()))


def _safe_parse_list(value: Any) -> list[Any]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, dict):
        return [value]
    if not isinstance(value, str):
        return []

    text = value.strip()
    if not text:
        return []

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        try:
            parsed = ast.literal_eval(text)
        except (ValueError, SyntaxError):
            return []

    return parsed if isinstance(parsed, list) else []


def _extract_names(value: Any, max_items: Optional[int] = None) -> list[str]:
    names: list[str] = []
    for item in _safe_parse_list(value):
        if not isinstance(item, dict):
            continue
        name = str(item.get("name", "")).strip()
        if not name:
            continue
        names.append(name)
        if max_items is not None and len(names) >= max_items:
            break
    return names


def _extract_director(value: Any) -> str:
    for item in _safe_parse_list(value):
        if not isinstance(item, dict):
            continue
        if item.get("job") == "Director":
            return str(item.get("name", "")).strip()
    return ""


def _coerce_name_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    return []


def _normalize_terms_for_tags(items: list[str]) -> list[str]:
    return [item.replace(" ", "").lower() for item in items if item]


def infer_moods(row: pd.Series) -> list[str]:
    genres = {genre.lower() for genre in _coerce_name_list(row.get("genres"))}
    keywords = {keyword.lower() for keyword in _coerce_name_list(row.get("keywords"))}
    overview = str(row.get("overview", "") or "").lower()

    moods: list[str] = []
    for mood, rules in MOOD_RULES.items():
        genre_hits = genres.intersection({genre.lower() for genre in rules["genres"]})
        keyword_hits = keywords.intersection({term.lower() for term in rules["keywords"]})
        overview_hits = {
            term.lower()
            for term in rules["overview_terms"]
            if term.lower() in overview
        }
        score = (2 * len(genre_hits)) + (2 * len(keyword_hits)) + len(overview_hits)
        if score >= 2:
            moods.append(mood)

    if moods:
        return moods[:2]

    if "comedy" in genres or "romance" in genres:
        return ["Feel-Good"]
    if "action" in genres or "thriller" in genres:
        return ["Action Night"]
    if "family" in genres or "animation" in genres:
        return ["Family Time"]
    if "science fiction" in genres or "mystery" in genres:
        return ["Mind-Bending"]
    return []


def _build_tag_string(row: pd.Series) -> str:
    tokens: list[str] = []

    overview = str(row.get("overview", "") or "").lower().strip()
    if overview:
        tokens.extend(overview.split())

    tokens.extend(_normalize_terms_for_tags(_coerce_name_list(row.get("genres"))))
    tokens.extend(_normalize_terms_for_tags(_coerce_name_list(row.get("keywords"))))
    tokens.extend(_normalize_terms_for_tags(_coerce_name_list(row.get("cast"))[:3]))

    director = str(row.get("director", "") or "").strip()
    if director:
        tokens.append(director.replace(" ", "").lower())

    return " ".join(tokens)


@st.cache_data(show_spinner=False)
def load_metadata() -> pd.DataFrame:
    if not MOVIES_CSV_PATH.exists():
        return pd.DataFrame()

    movies_raw = pd.read_csv(MOVIES_CSV_PATH)
    if "id" in movies_raw.columns and "movie_id" not in movies_raw.columns:
        movies_raw = movies_raw.rename(columns={"id": "movie_id"})

    base_columns = [
        "movie_id",
        "title",
        "overview",
        "genres",
        "keywords",
        "release_date",
        "runtime",
        "vote_average",
        "vote_count",
        "popularity",
        "tagline",
        "original_language",
    ]
    movies_raw = movies_raw[base_columns].copy()

    if CREDITS_CSV_PATH.exists():
        credits_raw = pd.read_csv(CREDITS_CSV_PATH, usecols=["movie_id", "cast", "crew"])
        movies_raw = movies_raw.merge(credits_raw, on="movie_id", how="left")
    else:
        movies_raw["cast"] = None
        movies_raw["crew"] = None

    metadata = pd.DataFrame(
        {
            "movie_id": pd.to_numeric(movies_raw["movie_id"], errors="coerce"),
            "title": movies_raw["title"].fillna("").astype(str).str.strip(),
            "overview": movies_raw["overview"].fillna("").astype(str).str.strip(),
            "tagline": movies_raw["tagline"].fillna("").astype(str).str.strip(),
            "genres": movies_raw["genres"].apply(_extract_names),
            "keywords": movies_raw["keywords"].apply(lambda value: _extract_names(value, max_items=6)),
            "cast": movies_raw["cast"].apply(lambda value: _extract_names(value, max_items=5)),
            "director": movies_raw["crew"].apply(_extract_director),
            "release_year": pd.to_datetime(movies_raw["release_date"], errors="coerce").dt.year,
            "runtime": pd.to_numeric(movies_raw["runtime"], errors="coerce"),
            "vote_average": pd.to_numeric(movies_raw["vote_average"], errors="coerce"),
            "vote_count": pd.to_numeric(movies_raw["vote_count"], errors="coerce"),
            "popularity": pd.to_numeric(movies_raw["popularity"], errors="coerce"),
            "original_language": movies_raw["original_language"].fillna("").astype(str).str.upper(),
        }
    )

    metadata = metadata.dropna(subset=["movie_id"]).copy()
    metadata = metadata[metadata["title"] != ""].copy()
    metadata["movie_id"] = metadata["movie_id"].astype(int)
    metadata["moods"] = metadata.apply(infer_moods, axis=1)
    metadata["tags"] = metadata.apply(_build_tag_string, axis=1)
    metadata = metadata.drop_duplicates(subset=["movie_id"]).reset_index(drop=True)
    return metadata


def ensure_movie_columns(movies_df: pd.DataFrame) -> pd.DataFrame:
    movies_df = movies_df.copy()

    for column in ("genres", "keywords", "cast", "moods"):
        if column not in movies_df.columns:
            movies_df[column] = [[] for _ in range(len(movies_df))]
        else:
            movies_df[column] = movies_df[column].apply(_coerce_name_list)

    string_columns = ("overview", "tagline", "director", "original_language", "title", "tags")
    for column in string_columns:
        if column not in movies_df.columns:
            movies_df[column] = ""
        else:
            movies_df[column] = movies_df[column].fillna("").astype(str).str.strip()

    numeric_columns = ("release_year", "runtime", "vote_average", "vote_count", "popularity")
    for column in numeric_columns:
        if column not in movies_df.columns:
            movies_df[column] = np.nan
        else:
            movies_df[column] = pd.to_numeric(movies_df[column], errors="coerce")

    mood_mask = movies_df["moods"].apply(len).eq(0)
    if mood_mask.any():
        movies_df.loc[mood_mask, "moods"] = movies_df.loc[mood_mask].apply(infer_moods, axis=1)

    movies_df["search_label"] = movies_df.apply(
        lambda row: (
            f"{row['title']} ({int(row['release_year'])})"
            if pd.notna(row["release_year"])
            else row["title"]
        ),
        axis=1,
    )
    return movies_df


def merge_movie_metadata(core_df: pd.DataFrame, metadata_df: pd.DataFrame) -> pd.DataFrame:
    movies_df = core_df.copy()

    if not metadata_df.empty:
        joined = metadata_df.drop(columns=["title"]).set_index("movie_id")
        movies_df = movies_df.join(joined, on="movie_id", rsuffix="_meta")

        if "tags_meta" in movies_df.columns:
            meta_tags = movies_df["tags_meta"].fillna("").astype(str).str.strip()
            if "tags" not in movies_df.columns:
                movies_df["tags"] = meta_tags
            else:
                base_tags = movies_df["tags"].fillna("").astype(str).str.strip()
                movies_df["tags"] = np.where(base_tags.eq(""), meta_tags, base_tags)
            movies_df = movies_df.drop(columns=["tags_meta"])

    return ensure_movie_columns(movies_df)


@dataclass(frozen=True)
class RecommenderModel:
    kind: str
    similarity: Optional[np.ndarray] = None
    vectors: Optional[Any] = None


@st.cache_resource
def load_recommender() -> tuple[pd.DataFrame, RecommenderModel]:
    metadata_df = load_metadata()

    movies_obj: Any = None
    if MOVIE_DICT_PATH.exists():
        with open(MOVIE_DICT_PATH, "rb") as file:
            movies_obj = pickle.load(file)

    if movies_obj is None:
        if metadata_df.empty:
            raise FileNotFoundError(
                "Missing data. Add movie_dict.pkl or the TMDb CSV files to this folder."
            )
        core_df = metadata_df[["movie_id", "title", "tags"]].copy()
    else:
        core_df = pd.DataFrame(movies_obj) if isinstance(movies_obj, dict) else movies_obj.copy()

    if "title" not in core_df.columns or "movie_id" not in core_df.columns:
        raise ValueError("Movie data is invalid. Expected 'title' and 'movie_id' columns.")

    core_df["movie_id"] = pd.to_numeric(core_df["movie_id"], errors="coerce")
    core_df = core_df.dropna(subset=["movie_id"]).copy()
    core_df["movie_id"] = core_df["movie_id"].astype(int)
    core_df["title"] = core_df["title"].fillna("").astype(str).str.strip()
    core_df = core_df[core_df["title"] != ""].copy()
    core_df = core_df.drop_duplicates(subset=["movie_id"]).reset_index(drop=True)

    movies_df = merge_movie_metadata(core_df, metadata_df)
    if movies_df.empty:
        raise ValueError("No movie records were loaded.")

    if SIMILARITY_PATH.exists():
        with open(SIMILARITY_PATH, "rb") as file:
            similarity = np.asarray(pickle.load(file))
        if similarity.shape != (len(movies_df), len(movies_df)):
            raise ValueError(
                "similarity.pkl does not match the loaded movie dataset. Rebuild it or remove it."
            )
        return movies_df, RecommenderModel(kind="similarity", similarity=similarity)

    tag_series = movies_df["tags"].fillna("").astype(str)
    if tag_series.str.strip().eq("").all():
        raise ValueError("Movie tags are empty. Rebuild the dataset with metadata tags.")

    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "scikit-learn is required when similarity.pkl is missing. Install it with pip."
        ) from exc

    vectorizer = TfidfVectorizer(max_features=5000, stop_words="english", ngram_range=(1, 2))
    vectors = vectorizer.fit_transform(tag_series)
    return movies_df, RecommenderModel(kind="vectors", vectors=vectors)


def get_similarity_scores(model: RecommenderModel, selected_index: int) -> np.ndarray:
    if model.kind == "similarity" and model.similarity is not None:
        return np.asarray(model.similarity[int(selected_index)], dtype=float)

    try:
        from sklearn.metrics.pairwise import cosine_similarity
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "scikit-learn is required to compute recommendations. Install it with pip."
        ) from exc

    return cosine_similarity(model.vectors[int(selected_index)], model.vectors).ravel()


def number_or_zero(value: Any) -> float:
    if pd.isna(value):
        return 0.0
    return float(value)


def normalize_scores(values: Any) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.size == 0:
        return array
    finite_mask = np.isfinite(array)
    if not finite_mask.any():
        return np.zeros_like(array, dtype=float)
    finite_values = array[finite_mask]
    minimum = float(finite_values.min())
    maximum = float(finite_values.max())
    if maximum - minimum < 1e-9:
        return np.zeros_like(array, dtype=float)
    normalized = (array - minimum) / (maximum - minimum)
    normalized[~finite_mask] = 0.0
    return normalized


def update_interaction_history(movie_id: int) -> None:
    history = [int(item) for item in st.session_state.get("interaction_history_ids", []) if str(item).isdigit()]
    history = [item for item in history if item != int(movie_id)]
    history.insert(0, int(movie_id))
    st.session_state["interaction_history_ids"] = history[:8]


def get_history_indices(
    movie_id_to_index: dict[int, int],
    selected_index: int,
) -> list[int]:
    history_ids = list(st.session_state.get("interaction_history_ids", []))
    watchlist_ids = list(st.session_state.get("watchlist_ids", set()))
    combined_ids = [
        int(movie_id)
        for movie_id in [*history_ids, *watchlist_ids]
        if str(movie_id).isdigit() and int(movie_id) in movie_id_to_index
    ]
    unique_indices: list[int] = []
    seen: set[int] = set()
    for movie_id in combined_ids:
        index = int(movie_id_to_index[int(movie_id)])
        if index == int(selected_index) or index in seen:
            continue
        seen.add(index)
        unique_indices.append(index)
    return unique_indices


def build_session_profile(movies_df: pd.DataFrame, history_indices: list[int]) -> dict[str, list[str]]:
    genre_counter: Counter[str] = Counter()
    mood_counter: Counter[str] = Counter()
    language_counter: Counter[str] = Counter()

    for history_index in history_indices:
        movie = movies_df.iloc[int(history_index)]
        genre_counter.update(_coerce_name_list(movie.get("genres"))[:3])
        mood_counter.update(_coerce_name_list(movie.get("moods"))[:2])
        language = str(movie.get("original_language", "") or "").strip().upper()
        if language:
            language_counter.update([language])

    return {
        "genres": [name for name, _ in genre_counter.most_common(3)],
        "moods": [name for name, _ in mood_counter.most_common(2)],
        "languages": [name for name, _ in language_counter.most_common(2)],
    }


def score_preference_alignment(
    movie: pd.Series,
    session_profile: dict[str, list[str]],
    preferred_languages: list[str],
    preferred_runtime: int,
    age_group: str,
    day_context: str,
    season_context: str,
    holiday_mode: bool,
) -> float:
    score = 0.0
    genres = set(_coerce_name_list(movie.get("genres")))
    moods = set(_coerce_name_list(movie.get("moods")))
    language = str(movie.get("original_language", "") or "").strip().upper()
    runtime = movie.get("runtime")

    profile_genres = set(session_profile.get("genres", []))
    profile_moods = set(session_profile.get("moods", []))
    profile_languages = set(session_profile.get("languages", []))

    if profile_genres:
        score += 0.24 * len(genres.intersection(profile_genres))
    if profile_moods:
        score += 0.18 * len(moods.intersection(profile_moods))
    if preferred_languages and language in preferred_languages:
        score += 0.35
    elif profile_languages and language in profile_languages:
        score += 0.18

    if pd.notna(runtime):
        runtime_delta = abs(float(runtime) - float(preferred_runtime))
        score += max(0.0, 1.0 - (runtime_delta / 100.0)) * 0.28

    score += 0.12 * len(genres.intersection(AGE_GROUP_GENRE_BOOSTS.get(age_group, set())))
    score += 0.12 * len(moods.intersection(AGE_GROUP_MOOD_BOOSTS.get(age_group, set())))
    score += 0.1 * len(genres.intersection(SEASONAL_GENRE_BOOSTS.get(season_context, set())))
    score += 0.12 * len(moods.intersection(DAY_CONTEXT_MOOD_BOOSTS.get(day_context, set())))

    if holiday_mode:
        score += 0.12 * len(genres.intersection(HOLIDAY_GENRE_BOOSTS))
        if "Feel-Good" in moods or "Family Time" in moods:
            score += 0.15

    return float(score)


def build_mode_reason(
    mode: str,
    selected_movie: pd.Series,
    candidate_movie: pd.Series,
    session_profile: dict[str, list[str]],
    profile_score: float,
) -> str:
    if mode == "Content-Based":
        return build_reason(selected_movie, candidate_movie)

    if mode == "Popularity-Based":
        return "Popular choice boosted by strong ratings, vote count, and your current filters."

    if mode == "Collaborative":
        genre_note = ", ".join(session_profile.get("genres", [])[:2])
        if genre_note:
            return f"Based on your saved and recent titles, you often lean toward {genre_note} movies."
        return "Build up your watchlist or browse a few more titles to strengthen collaborative suggestions."

    base_reason = build_reason(selected_movie, candidate_movie)
    if profile_score > 0.6 and session_profile.get("moods"):
        return base_reason + f" | Also fits your session mood: {session_profile['moods'][0]}."
    return base_reason + " | Blended with popularity and session preference signals."


def build_popularity_scores(movies_df: pd.DataFrame) -> np.ndarray:
    weighted = (
        movies_df["vote_average"].fillna(0) * 0.5
        + np.log1p(movies_df["vote_count"].fillna(0)) * 0.25
        + np.log1p(movies_df["popularity"].fillna(0)) * 0.25
    )
    return normalize_scores(weighted.to_numpy(dtype=float))


def get_mode_description(mode: str) -> str:
    descriptions = {
        "Content-Based": "Uses genres, cast, director, keywords, and plot similarity to find movies like the one you selected.",
        "Collaborative": "Uses your watchlist and recent browsing history as implicit preference signals to approximate collaborative suggestions.",
        "Hybrid": "Blends content similarity, session history, and popularity for more balanced recommendations.",
        "Popularity-Based": "Surfaces broadly popular, high-rated titles that still respect your active filters and profile.",
    }
    return descriptions.get(mode, "")


def build_people_index(
    movies_df: pd.DataFrame,
    column: str,
    min_titles: int = 1,
) -> dict[str, list[int]]:
    index_map: dict[str, list[int]] = {}

    for idx, movie in movies_df.iterrows():
        if column == "director":
            people = [str(movie.get("director", "") or "").strip()]
        else:
            people = _coerce_name_list(movie.get(column))

        for person in people:
            if not person:
                continue
            index_map.setdefault(person, []).append(int(idx))

    filtered: dict[str, list[int]] = {}
    for person, indices in index_map.items():
        unique_indices = sorted(set(indices))
        if len(unique_indices) < min_titles:
            continue
        filtered[person] = sorted(
            unique_indices,
            key=lambda idx: (
                number_or_zero(movies_df.at[idx, "vote_average"]),
                number_or_zero(movies_df.at[idx, "vote_count"]),
                number_or_zero(movies_df.at[idx, "popularity"]),
                number_or_zero(movies_df.at[idx, "release_year"]),
            ),
            reverse=True,
        )
    return filtered


def shared_items(left: Any, right: Any, limit: int = 2) -> list[str]:
    left_items = _coerce_name_list(left)
    right_items = _coerce_name_list(right)
    right_lookup = {item.casefold(): item for item in right_items}

    matches: list[str] = []
    seen: set[str] = set()
    for item in left_items:
        key = item.casefold()
        if key in right_lookup and key not in seen:
            matches.append(right_lookup[key])
            seen.add(key)
        if len(matches) >= limit:
            break
    return matches


def build_similarity_badges(selected_movie: pd.Series, candidate_movie: pd.Series) -> list[str]:
    badges: list[str] = []

    if shared_items(selected_movie.get("genres"), candidate_movie.get("genres"), limit=1):
        badges.append("Shared genre")
    if shared_items(selected_movie.get("keywords"), candidate_movie.get("keywords"), limit=1):
        badges.append("Shared theme")
    if shared_items(selected_movie.get("cast"), candidate_movie.get("cast"), limit=1):
        badges.append("Shared cast")

    selected_director = str(selected_movie.get("director", "") or "").strip()
    candidate_director = str(candidate_movie.get("director", "") or "").strip()
    if selected_director and candidate_director and selected_director.casefold() == candidate_director.casefold():
        badges.append("Same director")

    if shared_items(selected_movie.get("moods"), candidate_movie.get("moods"), limit=1):
        badges.append("Same mood")

    return badges[:3] or ["Similar vibe"]


def build_reason(selected_movie: pd.Series, candidate_movie: pd.Series) -> str:
    reasons: list[str] = []

    common_genres = shared_items(selected_movie.get("genres"), candidate_movie.get("genres"), limit=2)
    if common_genres:
        reasons.append("Shared genres: " + ", ".join(common_genres))

    common_keywords = shared_items(
        selected_movie.get("keywords"),
        candidate_movie.get("keywords"),
        limit=2,
    )
    if common_keywords:
        reasons.append("Shared themes: " + ", ".join(common_keywords))

    common_cast = shared_items(selected_movie.get("cast"), candidate_movie.get("cast"), limit=1)
    if common_cast:
        reasons.append("Shared cast: " + ", ".join(common_cast))

    selected_director = str(selected_movie.get("director", "") or "").strip()
    candidate_director = str(candidate_movie.get("director", "") or "").strip()
    if selected_director and candidate_director and selected_director.casefold() == candidate_director.casefold():
        reasons.append(f"Same director: {candidate_director}")

    return " | ".join(reasons) or "Close match on plot, themes, and cast signals."


def format_year(value: Any) -> str:
    return str(int(value)) if pd.notna(value) else "N/A"


def format_runtime(value: Any) -> str:
    if pd.isna(value):
        return "N/A"
    total_minutes = int(round(float(value)))
    hours, minutes = divmod(total_minutes, 60)
    if hours:
        return f"{hours}h {minutes:02d}m"
    return f"{minutes}m"


def format_rating(movie: pd.Series) -> str:
    vote_average = movie.get("vote_average")
    vote_count = movie.get("vote_count")
    if pd.isna(vote_average):
        return "Rating unavailable"
    count = int(vote_count) if pd.notna(vote_count) else 0
    return f"{float(vote_average):.1f}/10 from {count:,} votes"


def truncate_text(text: Any, max_chars: int = 170) -> str:
    clean_text = str(text or "").strip()
    if not clean_text:
        return ""
    if len(clean_text) <= max_chars:
        return clean_text
    return clean_text[: max_chars - 3].rstrip() + "..."


def render_token_row(items: Any, variant: str, max_items: int = 6) -> None:
    labels = _coerce_name_list(items)[:max_items]
    if not labels:
        return
    tokens = "".join(
        f"<span class='token token-{variant}'>{html.escape(label)}</span>"
        for label in labels
    )
    st.markdown(f"<div class='token-row'>{tokens}</div>", unsafe_allow_html=True)


def render_watchlist_button(movie: pd.Series, key: str, compact: bool = False) -> None:
    movie_id = int(movie["movie_id"])
    label = "Saved" if is_in_watchlist(movie_id) else "Save"
    if not compact:
        label = "In Watchlist" if is_in_watchlist(movie_id) else "Add to Watchlist"

    if st.button(label, key=key, use_container_width=True):
        toggle_watchlist(movie)
        safe_rerun()


def render_trailer_button(movie_id: int, label: str = "Trailer", kind: str = "primary") -> None:
    trailer_url = fetch_trailer_url(movie_id)
    if not trailer_url:
        st.caption("Trailer unavailable")
        return

    if hasattr(st, "link_button"):
        try:
            st.link_button(label, trailer_url, use_container_width=True, type=kind)
        except TypeError:
            st.link_button(label, trailer_url, use_container_width=True)
    else:
        st.markdown(f"[{label}]({trailer_url})")


def render_stat_grid(movie: pd.Series) -> None:
    stat_items = [
        ("Release", format_year(movie.get("release_year"))),
        ("Runtime", format_runtime(movie.get("runtime"))),
        (
            "Rating",
            f"{float(movie['vote_average']):.1f}/10" if pd.notna(movie["vote_average"]) else "N/A",
        ),
        ("Votes", f"{int(movie['vote_count']):,}" if pd.notna(movie["vote_count"]) else "N/A"),
        (
            "Popularity",
            f"{int(round(float(movie['popularity']))):,}" if pd.notna(movie["popularity"]) else "N/A",
        ),
    ]
    cards = "".join(
        "<div class='stat-card'>"
        f"<div class='stat-label'>{html.escape(label)}</div>"
        f"<div class='stat-value'>{html.escape(value)}</div>"
        "</div>"
        for label, value in stat_items
    )
    st.markdown(f"<div class='stat-grid'>{cards}</div>", unsafe_allow_html=True)


def render_selected_movie(movie: pd.Series) -> None:
    with st.container(border=True):
        poster_col, detail_col = st.columns([0.9, 1.7], gap="large")

        with poster_col:
            st.image(fetch_poster(int(movie["movie_id"])), use_container_width=True)

        with detail_col:
            title = html.escape(movie["title"])
            tagline = str(movie.get("tagline", "") or "").strip()
            overview = html.escape(movie.get("overview", "") or "Overview not available.")
            card_parts = [
                "<div class='hero-card'>",
                "<div class='eyebrow'>Selected Movie</div>",
                f"<div class='hero-title'>{title}</div>",
            ]
            if tagline:
                card_parts.append(f"<p class='hero-tagline'>{html.escape(tagline)}</p>")
            card_parts.append("</div>")
            st.markdown("".join(card_parts), unsafe_allow_html=True)

            render_stat_grid(movie)

            st.markdown(f"<p class='hero-copy'>{overview}</p>", unsafe_allow_html=True)

            action_cols = st.columns([1.05, 1.15, 2.8])
            with action_cols[0]:
                render_trailer_button(int(movie["movie_id"]), label="Play trailer")
            with action_cols[1]:
                render_watchlist_button(movie, key=f"selected_watch_{int(movie['movie_id'])}")

            render_token_row(movie.get("genres"), variant="chip", max_items=5)
            render_token_row(movie.get("moods"), variant="mood", max_items=4)

            details: list[str] = []
            director = str(movie.get("director", "") or "").strip()
            if director:
                details.append(f"Director: {director}")

            cast = _coerce_name_list(movie.get("cast"))
            if cast:
                details.append("Cast: " + ", ".join(cast[:3]))

            language = str(movie.get("original_language", "") or "").strip()
            if language:
                details.append(f"Language: {language}")

            if details:
                detail_rows = "".join(
                    f"<div class='detail-item'><strong>{html.escape(item.split(':', 1)[0])}:</strong> {html.escape(item.split(':', 1)[1].strip())}</div>"
                    for item in details
                    if ":" in item
                )
                st.markdown(f"<div class='detail-stack'>{detail_rows}</div>", unsafe_allow_html=True)

            trailer_url = fetch_trailer_url(int(movie["movie_id"]))
            if trailer_url:
                with st.expander("Open embedded trailer"):
                    st.video(trailer_url)


def render_recommendation_details(recommendation: dict[str, Any]) -> None:
    movie = recommendation["movie"]
    movie_id = int(movie["movie_id"])
    st.caption(format_rating(movie))
    reason = recommendation.get("reason", "")
    if reason:
        st.write(reason)

    short_overview = truncate_text(movie.get("overview"), max_chars=130)
    if short_overview:
        st.write(short_overview)

    action_cols = st.columns(2)
    with action_cols[0]:
        render_watchlist_button(movie, key=f"rec_watch_{recommendation['rank']}_{movie_id}", compact=True)
    with action_cols[1]:
        render_trailer_button(movie_id, label="Trailer", kind="secondary")


def render_recommendation_card(recommendation: dict[str, Any]) -> None:
    movie = recommendation["movie"]
    st.image(fetch_poster(int(movie["movie_id"])), use_container_width=True)
    st.markdown(f"<div class='mini-title'>{html.escape(movie['title'])}</div>", unsafe_allow_html=True)
    st.markdown(
        f"<div class='meta-line'>Runtime | {format_runtime(movie.get('runtime'))}</div>",
        unsafe_allow_html=True,
    )
    render_token_row(movie.get("genres"), variant="chip", max_items=3)
    render_token_row(recommendation["badges"], variant="badge", max_items=2)

    if hasattr(st, "popover"):
        with st.popover("Details"):
            render_recommendation_details(recommendation)
    else:
        with st.expander("Details"):
            render_recommendation_details(recommendation)


def filter_candidate(
    movie: pd.Series,
    genre_filter: str,
    mood_filter: str,
    min_rating: float,
    min_votes: int,
    year_range: tuple[int, int],
    runtime_range: tuple[int, int],
    languages: list[str],
) -> bool:
    genres = _coerce_name_list(movie.get("genres"))
    moods = _coerce_name_list(movie.get("moods"))
    rating = number_or_zero(movie.get("vote_average"))
    votes = int(number_or_zero(movie.get("vote_count")))
    year = movie.get("release_year")
    runtime = movie.get("runtime")
    language = str(movie.get("original_language", "") or "").strip().upper()

    if genre_filter != "All" and genre_filter not in genres:
        return False
    if mood_filter != "All" and mood_filter not in moods:
        return False
    if rating < float(min_rating):
        return False
    if votes < int(min_votes):
        return False
    if pd.isna(year) or not (int(year_range[0]) <= int(year) <= int(year_range[1])):
        return False
    if pd.isna(runtime) or not (int(runtime_range[0]) <= int(runtime) <= int(runtime_range[1])):
        return False
    if languages and language not in languages:
        return False
    return True


def sort_recommendations(recommendations: list[dict[str, Any]], sort_by: str) -> list[dict[str, Any]]:
    if sort_by == "Similarity":
        recommendations.sort(key=lambda item: item["score"], reverse=True)
    elif sort_by == "Rating":
        recommendations.sort(
            key=lambda item: (
                number_or_zero(item["movie"].get("vote_average")),
                number_or_zero(item["movie"].get("vote_count")),
                item["score"],
            ),
            reverse=True,
        )
    elif sort_by == "Popularity":
        recommendations.sort(
            key=lambda item: (
                number_or_zero(item["movie"].get("popularity")),
                number_or_zero(item["movie"].get("vote_average")),
                item["score"],
            ),
            reverse=True,
        )
    elif sort_by == "Newest":
        recommendations.sort(
            key=lambda item: (
                number_or_zero(item["movie"].get("release_year")),
                number_or_zero(item["movie"].get("vote_average")),
                item["score"],
            ),
            reverse=True,
        )
    return recommendations


def get_recommendations(
    movies_df: pd.DataFrame,
    model: RecommenderModel,
    movie_id_to_index: dict[int, int],
    selected_index: int,
    limit: int,
    genre_filter: str,
    mood_filter: str,
    min_rating: float,
    min_votes: int,
    year_range: tuple[int, int],
    runtime_range: tuple[int, int],
    languages: list[str],
    sort_by: str,
    recommendation_mode: str,
    preferred_runtime: int,
    age_group: str,
    day_context: str,
    season_context: str,
    holiday_mode: bool,
) -> list[dict[str, Any]]:
    content_scores = normalize_scores(get_similarity_scores(model, selected_index))
    selected_movie = movies_df.iloc[int(selected_index)]
    history_indices = get_history_indices(movie_id_to_index, selected_index)
    session_profile = build_session_profile(movies_df, history_indices)
    popularity_scores = build_popularity_scores(movies_df)

    history_scores = np.zeros(len(movies_df), dtype=float)
    if history_indices:
        stacked_scores = np.vstack(
            [normalize_scores(get_similarity_scores(model, history_index)) for history_index in history_indices]
        )
        history_scores = stacked_scores.mean(axis=0)

    recommendations: list[dict[str, Any]] = []
    for candidate_index in range(len(movies_df)):
        if candidate_index == int(selected_index):
            continue

        candidate_movie = movies_df.iloc[int(candidate_index)]
        if not filter_candidate(
            candidate_movie,
            genre_filter=genre_filter,
            mood_filter=mood_filter,
            min_rating=min_rating,
            min_votes=min_votes,
            year_range=year_range,
            runtime_range=runtime_range,
            languages=languages,
        ):
            continue

        profile_score = score_preference_alignment(
            candidate_movie,
            session_profile=session_profile,
            preferred_languages=languages,
            preferred_runtime=preferred_runtime,
            age_group=age_group,
            day_context=day_context,
            season_context=season_context,
            holiday_mode=holiday_mode,
        )

        if recommendation_mode == "Popularity-Based":
            final_score = (0.8 * popularity_scores[candidate_index]) + (0.2 * profile_score)
        elif recommendation_mode == "Collaborative":
            final_score = (0.65 * history_scores[candidate_index]) + (0.2 * popularity_scores[candidate_index]) + (0.15 * profile_score)
        elif recommendation_mode == "Hybrid":
            final_score = (
                0.5 * content_scores[candidate_index]
                + 0.25 * history_scores[candidate_index]
                + 0.15 * popularity_scores[candidate_index]
                + 0.1 * profile_score
            )
        else:
            final_score = (0.88 * content_scores[candidate_index]) + (0.12 * profile_score)

        recommendations.append(
            {
                "movie": candidate_movie,
                "score": float(final_score),
                "reason": build_mode_reason(
                    recommendation_mode,
                    selected_movie,
                    candidate_movie,
                    session_profile=session_profile,
                    profile_score=profile_score,
                ),
                "badges": build_similarity_badges(selected_movie, candidate_movie),
                "mode": recommendation_mode,
            }
        )

    recommendations = sort_recommendations(recommendations, sort_by)
    trimmed = recommendations[: int(limit)]
    for rank, item in enumerate(trimmed, start=1):
        item["rank"] = rank
    return trimmed


def get_default_movie_index(movies_df: pd.DataFrame) -> int:
    avatar_match = movies_df.index[movies_df["title"].str.casefold() == "avatar"]
    if len(avatar_match):
        return int(avatar_match[0])
    return int(movies_df.index[0])


def get_sorted_movie_options(movies_df: pd.DataFrame) -> list[int]:
    label_lookup = movies_df["search_label"].to_dict()
    return sorted(label_lookup, key=lambda index: label_lookup[index].casefold())


def resolve_selected_movie(
    movies_df: pd.DataFrame,
    actor_map: dict[str, list[int]],
    director_map: dict[str, list[int]],
) -> tuple[int, str]:
    label_lookup = movies_df["search_label"].to_dict()
    movie_options = get_sorted_movie_options(movies_df)
    browse_mode = st.session_state.get("browse_mode", "Movie")

    if browse_mode == "Movie" or (browse_mode == "Actor" and not actor_map) or (browse_mode == "Director" and not director_map):
        if st.session_state.get("selected_movie_index") not in movie_options:
            st.session_state["selected_movie_index"] = movie_options[0]
        st.selectbox(
            "Movie",
            options=movie_options,
            format_func=lambda index: label_lookup[int(index)],
            key="selected_movie_index",
        )
        return int(st.session_state["selected_movie_index"]), ""

    if browse_mode == "Actor":
        actor_options = sorted(actor_map, key=str.casefold)
        if not st.session_state.get("selected_actor") or st.session_state["selected_actor"] not in actor_options:
            st.session_state["selected_actor"] = actor_options[0]
        st.selectbox("Actor", options=actor_options, key="selected_actor")

        actor_indices = actor_map[st.session_state["selected_actor"]]
        if st.session_state.get("actor_anchor_index") not in actor_indices:
            st.session_state["actor_anchor_index"] = actor_indices[0]
        st.selectbox(
            "Movie from this actor",
            options=actor_indices,
            format_func=lambda index: label_lookup[int(index)],
            key="actor_anchor_index",
        )
        context = (
            f"Recommendations are anchored to {st.session_state['selected_actor']}'s filmography. "
            f"This dataset includes {len(actor_indices)} matching titles."
        )
        return int(st.session_state["actor_anchor_index"]), context

    director_options = sorted(director_map, key=str.casefold)
    if not st.session_state.get("selected_director") or st.session_state["selected_director"] not in director_options:
        st.session_state["selected_director"] = director_options[0]
    st.selectbox("Director", options=director_options, key="selected_director")

    director_indices = director_map[st.session_state["selected_director"]]
    if st.session_state.get("director_anchor_index") not in director_indices:
        st.session_state["director_anchor_index"] = director_indices[0]
    st.selectbox(
        "Movie from this director",
        options=director_indices,
        format_func=lambda index: label_lookup[int(index)],
        key="director_anchor_index",
    )
    context = (
        f"Recommendations are anchored to {st.session_state['selected_director']}'s directing work. "
        f"This dataset includes {len(director_indices)} matching titles."
    )
    return int(st.session_state["director_anchor_index"]), context


def get_filter_bounds(movies_df: pd.DataFrame) -> dict[str, Any]:
    valid_years = movies_df["release_year"].dropna().astype(int)
    year_min = int(valid_years.min()) if not valid_years.empty else 1950
    year_max = int(valid_years.max()) if not valid_years.empty else 2025

    valid_runtime = movies_df["runtime"].dropna().astype(int)
    runtime_min = int(valid_runtime.min()) if not valid_runtime.empty else 60
    runtime_max = int(valid_runtime.max()) if not valid_runtime.empty else 240

    vote_count_max = int(movies_df["vote_count"].fillna(0).max()) if not movies_df.empty else 0
    vote_slider_max = max(100, min(10000, vote_count_max))

    return {
        "year_min": year_min,
        "year_max": year_max,
        "runtime_min": runtime_min,
        "runtime_max": runtime_max,
        "vote_slider_max": vote_slider_max,
    }


def get_top_mood_picks(movies_df: pd.DataFrame, mood: str, limit: int = 6) -> pd.DataFrame:
    mood_movies = movies_df[movies_df["moods"].apply(lambda moods: mood in _coerce_name_list(moods))].copy()
    if mood_movies.empty:
        return mood_movies

    mood_movies = mood_movies.sort_values(
        by=["vote_average", "vote_count", "popularity", "release_year"],
        ascending=False,
        na_position="last",
    )
    return mood_movies.head(limit)


def render_mood_sections(movies_df: pd.DataFrame, movie_id_to_index: dict[int, int]) -> None:
    st.subheader("Mood Collections")
    st.markdown(
        "<p class='section-kicker'>Pick a vibe first, then jump straight into recommendations for one of the top titles.</p>",
        unsafe_allow_html=True,
    )
    mood_tabs = st.tabs(list(MOOD_RULES))

    for mood_name, tab in zip(MOOD_RULES, mood_tabs):
        with tab:
            st.caption(MOOD_RULES[mood_name]["description"])
            picks = get_top_mood_picks(movies_df, mood_name, limit=6)
            if picks.empty:
                st.info("No movies matched this mood in the current dataset.")
                continue

            pick_columns = st.columns(3, gap="large")
            for column, (_, movie) in zip(pick_columns * 2, picks.iterrows()):
                with column:
                    with st.container(border=True):
                        st.image(fetch_poster(int(movie["movie_id"])), use_container_width=True)
                        st.markdown(
                            f"<div class='mini-title'>{html.escape(movie['title'])}</div>",
                            unsafe_allow_html=True,
                        )
                        st.markdown(
                            f"<div class='meta-line'>{format_year(movie.get('release_year'))} | {format_rating(movie)}</div>",
                            unsafe_allow_html=True,
                        )
                        short_overview = truncate_text(movie.get("overview"), max_chars=120)
                        if short_overview:
                            st.write(short_overview)
                        if st.button(
                            "Use for recommendations",
                            key=f"mood_pick_{mood_name}_{int(movie['movie_id'])}",
                            use_container_width=True,
                        ):
                            st.session_state["browse_mode"] = "Movie"
                            st.session_state["selected_movie_index"] = movie_id_to_index[int(movie["movie_id"])]
                            safe_rerun()


def render_watchlist_section(movies_df: pd.DataFrame) -> None:
    st.subheader("Watchlist")
    watchlist_ids = set(st.session_state.get("watchlist_ids", set()))
    if not watchlist_ids:
        st.info("Save movies from the selected title or recommendation cards to build your watchlist.")
        return

    watchlist_movies = movies_df[movies_df["movie_id"].isin(watchlist_ids)].copy()
    order = {movie_id: idx for idx, movie_id in enumerate(sorted(watchlist_ids))}
    watchlist_movies["watchlist_order"] = watchlist_movies["movie_id"].map(order)
    watchlist_movies = watchlist_movies.sort_values("watchlist_order")

    export_payload = [
        {
            "movie_id": int(movie["movie_id"]),
            "title": movie["title"],
            "release_year": int(movie["release_year"]) if pd.notna(movie["release_year"]) else None,
        }
        for _, movie in watchlist_movies.iterrows()
    ]

    action_cols = st.columns([1, 1])
    with action_cols[0]:
        st.download_button(
            "Download watchlist",
            data=json.dumps(export_payload, indent=2),
            file_name="movie_watchlist.json",
            mime="application/json",
            use_container_width=True,
        )
    with action_cols[1]:
        if st.button("Clear watchlist", use_container_width=True):
            clear_watchlist()
            safe_rerun()

    watchlist_columns = st.columns(3, gap="large")
    repeated_columns = watchlist_columns * max(1, len(watchlist_movies))
    for column, (_, movie) in zip(repeated_columns, watchlist_movies.iterrows()):
        with column:
            with st.container(border=True):
                st.image(fetch_poster(int(movie["movie_id"])), use_container_width=True)
                st.markdown(f"<div class='mini-title'>{html.escape(movie['title'])}</div>", unsafe_allow_html=True)
                st.markdown(
                    f"<div class='meta-line'>{format_year(movie.get('release_year'))} | {format_rating(movie)}</div>",
                    unsafe_allow_html=True,
                )
                action_cols = st.columns(2)
                with action_cols[0]:
                    if st.button(
                        "Remove",
                        key=f"watchlist_remove_{int(movie['movie_id'])}",
                        use_container_width=True,
                    ):
                        toggle_watchlist(movie)
                        safe_rerun()
                with action_cols[1]:
                    render_trailer_button(int(movie["movie_id"]), label="Trailer", kind="secondary")


def render_recommendation_grid(recommendations: list[dict[str, Any]]) -> None:
    st.subheader("Top Matches")
    st.markdown(
        "<p class='results-note'>The list updates automatically as you change filters, sorting, or browse mode.</p>",
        unsafe_allow_html=True,
    )

    if not recommendations:
        st.warning("No movies matched the current filters. Try widening the year, runtime, or language range.")
        return

    columns_per_row = min(5, max(1, len(recommendations)))
    for start in range(0, len(recommendations), columns_per_row):
        columns = st.columns(columns_per_row, gap="medium")
        for column, recommendation in zip(columns, recommendations[start : start + columns_per_row]):
            with column:
                render_recommendation_card(recommendation)


def render_panel_switcher() -> str:
    if hasattr(st, "segmented_control"):
        selected = st.segmented_control(
            "Section",
            options=SECTION_OPTIONS,
            default=st.session_state.get("active_panel", "Recommendations"),
            selection_mode="single",
            label_visibility="collapsed",
        )
        if selected:
            st.session_state["active_panel"] = selected
    else:
        st.radio(
            "Section",
            options=SECTION_OPTIONS,
            key="active_panel",
            horizontal=True,
            label_visibility="collapsed",
        )

    return str(st.session_state.get("active_panel", "Recommendations"))


def main() -> None:
    inject_styles()

    with st.spinner("Loading movie data..."):
        try:
            movies, model = load_recommender()
        except Exception as exc:
            st.error(str(exc))
            st.stop()

    default_movie_index = get_default_movie_index(movies)
    ensure_session_state(default_movie_index)
    show_watchlist_notice()

    if not TMDB_API_KEY:
        st.info("Set `TMDB_API_KEY` in `.env`, your environment, or Streamlit secrets to display posters and trailers.")

    actor_map = build_people_index(movies, column="cast", min_titles=2)
    director_map = build_people_index(movies, column="director", min_titles=1)
    filter_bounds = get_filter_bounds(movies)
    movie_id_to_index = {
        int(movie_id): int(index)
        for index, movie_id in movies["movie_id"].items()
    }
    sync_query_state(movie_id_to_index)

    all_genres = sorted(
        {
            genre
            for genres in movies["genres"]
            for genre in _coerce_name_list(genres)
        }
    )
    all_languages = sorted(
        {
            language
            for language in movies["original_language"].fillna("").astype(str)
            if language.strip()
        }
    )

    with st.sidebar:
        st.subheader("Discover")
        st.radio(
            "Browse by",
            options=["Movie", "Actor", "Director"],
            key="browse_mode",
            horizontal=True,
        )
        selected_index, context = resolve_selected_movie(movies, actor_map, director_map)
        st.markdown("---")

        st.subheader("Recommendation engine")
        recommendation_mode = st.selectbox(
            "Recommendation type",
            options=RECOMMENDATION_TYPES,
            index=2,
        )
        st.caption(get_mode_description(recommendation_mode))
        age_group = st.selectbox(
            "Age group",
            options=["All Ages", "Kids & Family", "Teens", "Adults"],
            index=0,
        )
        preferred_runtime = st.slider(
            "Average viewing duration",
            min_value=filter_bounds["runtime_min"],
            max_value=filter_bounds["runtime_max"],
            value=min(120, filter_bounds["runtime_max"]),
            step=5,
        )
        day_context = st.selectbox(
            "Day of the week context",
            options=["Any Day", "Weekday", "Weekend"],
            index=0,
        )
        season_context = st.selectbox(
            "Season context",
            options=["Any", "Spring", "Summer", "Autumn", "Winter"],
            index=0,
        )
        holiday_mode = st.toggle("Holiday mood boost", value=False)
        with st.expander("Recommendation system notes"):
            st.markdown(
                "- `Content-Based`: looks at movie metadata like genre, cast, director, and plot.\n"
                "- `Collaborative`: uses your watchlist and recent browsing as implicit interaction signals.\n"
                "- `Hybrid`: combines metadata, session history, and popularity.\n"
                "- `Popularity-Based`: falls back to top-rated / widely watched movies.\n\n"
                "The engine also adapts using language preferences, average viewing duration, and seasonal/day context."
            )
        st.markdown("---")

        st.subheader("Tune the results")
        num_recommendations = st.slider(
            "How many matches",
            min_value=3,
            max_value=12,
            value=DEFAULT_RECOMMENDATIONS,
            step=1,
        )
        genre_filter = st.selectbox("Genre", options=["All", *all_genres])
        mood_filter = st.selectbox("Mood", options=["All", *MOOD_RULES])
        min_rating = st.slider(
            "Minimum audience rating",
            min_value=0.0,
            max_value=10.0,
            value=DEFAULT_MIN_RATING,
            step=0.5,
        )
        min_votes = st.slider(
            "Minimum vote count",
            min_value=0,
            max_value=filter_bounds["vote_slider_max"],
            value=min(DEFAULT_MIN_VOTES, filter_bounds["vote_slider_max"]),
            step=50,
        )
        year_range = st.slider(
            "Release year",
            min_value=filter_bounds["year_min"],
            max_value=filter_bounds["year_max"],
            value=(filter_bounds["year_min"], filter_bounds["year_max"]),
            step=1,
        )
        runtime_range = st.slider(
            "Runtime (minutes)",
            min_value=filter_bounds["runtime_min"],
            max_value=filter_bounds["runtime_max"],
            value=(filter_bounds["runtime_min"], filter_bounds["runtime_max"]),
            step=5,
        )
        selected_languages = st.multiselect("Languages", options=all_languages, default=[])
        sort_by = st.selectbox(
            "Sort recommendations by",
            options=["Similarity", "Rating", "Popularity", "Newest"],
        )
        st.markdown("---")

        st.subheader("Saved")
        st.metric("Watchlist items", len(st.session_state.get("watchlist_ids", set())))

    selected_movie = movies.iloc[int(selected_index)]
    update_interaction_history(int(selected_movie["movie_id"]))
    st.session_state["selected_movie_id_for_share"] = int(selected_movie["movie_id"])
    render_topbar()
    if context:
        st.caption(context)

    render_selected_movie(selected_movie)
    st.caption(f"{recommendation_mode} mode: {get_mode_description(recommendation_mode)}")

    recommendations = get_recommendations(
        movies_df=movies,
        model=model,
        movie_id_to_index=movie_id_to_index,
        selected_index=int(selected_index),
        limit=int(num_recommendations),
        genre_filter=genre_filter,
        mood_filter=mood_filter,
        min_rating=float(min_rating),
        min_votes=int(min_votes),
        year_range=(int(year_range[0]), int(year_range[1])),
        runtime_range=(int(runtime_range[0]), int(runtime_range[1])),
        languages=[language.upper() for language in selected_languages],
        sort_by=sort_by,
        recommendation_mode=recommendation_mode,
        preferred_runtime=int(preferred_runtime),
        age_group=age_group,
        day_context=day_context,
        season_context=season_context,
        holiday_mode=bool(holiday_mode),
    )

    active_panel = render_panel_switcher()
    sync_query_params(int(selected_movie["movie_id"]))

    if active_panel == "Recommendations":
        render_recommendation_grid(recommendations)
    elif active_panel == "Mood Collections":
        render_mood_sections(movies, movie_id_to_index)
    else:
        render_watchlist_section(movies)


if __name__ == "__main__":
    main()
