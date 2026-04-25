import ast
import base64
import html as html_lib
import time
from pathlib import Path

import pandas as pd
import streamlit as st

from src.movie_recommenders import build_recommender

# Config
ROOT_DIR = Path(__file__).resolve().parents[2]

CSV_PATH = ROOT_DIR / "movies" / "tmdb_dataset_full.csv"
EMBEDDINGS_PATH = ROOT_DIR / "data" / "embeddings" / "tmdb_embeddings.parquet"
USERS_PATH = ROOT_DIR / "data" / "users.csv"
INTERACTIONS_PATH = ROOT_DIR / "data" / "interactions.csv"
CACHE_DIR = ROOT_DIR / "cache" / "reco_cache"

BG_PATH = ROOT_DIR / "assets" / "fondo-login.png"
LOGO_PATH = ROOT_DIR / "assets" / "logo-streaming.png"

FEATURED_LIMIT = 12
CARD_HEIGHT = 370


# Recommendercall
@st.cache_resource
def get_recommender(users_mtime, interactions_mtime):
    return build_recommender(
        movies_csv=str(CSV_PATH),
        embeddings_parquet=str(EMBEDDINGS_PATH),
        users_csv=str(USERS_PATH),
        interactions_csv=str(INTERACTIONS_PATH),
        cache_dir=str(CACHE_DIR),
        force_rebuild_cache=False,
    )


# Helpers
def get_base64_of_bin_file(bin_file):
    with open(bin_file, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()


def get_bg_base64():
    try:
        return get_base64_of_bin_file(BG_PATH)
    except Exception:
        return ""


def get_logo_base64():
    try:
        return get_base64_of_bin_file(LOGO_PATH)
    except Exception:
        return ""


def placeholder_poster():
    svg = """
    <svg xmlns="http://www.w3.org/2000/svg" width="900" height="1350" viewBox="0 0 900 1350">
      <defs>
        <linearGradient id="g" x1="0" y1="0" x2="1" y2="1">
          <stop offset="0%" stop-color="#1e293b"/>
          <stop offset="100%" stop-color="#0f172a"/>
        </linearGradient>
      </defs>
      <rect width="900" height="1350" fill="url(#g)"/>
      <circle cx="450" cy="520" r="140" fill="#6366f1" opacity="0.18"/>
      <circle cx="450" cy="520" r="85"  fill="#a855f7" opacity="0.24"/>
      <text x="50%" y="78%" font-family="Arial, sans-serif" font-size="56"
            fill="#cbd5e1" text-anchor="middle">Sin póster</text>
    </svg>
    """
    return "data:image/svg+xml;base64," + base64.b64encode(svg.encode("utf-8")).decode("utf-8")


def escape_text(value, default=""):
    if value is None:
        return default
    try:
        if pd.isna(value):
            return default
    except Exception:
        pass
    return html_lib.escape(str(value), quote=True)


def safe_poster_url(value):
    if value is None:
        return placeholder_poster()
    try:
        if pd.isna(value):
            return placeholder_poster()
    except Exception:
        pass
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null"}:
        return placeholder_poster()
    return text


def parse_genres(value):
    if pd.isna(value):
        return "Desconocido"
    if isinstance(value, list):
        out = []
        for item in value:
            if isinstance(item, dict):
                name = item.get("name")
                if name:
                    out.append(str(name))
            else:
                out.append(str(item))
        return ", ".join(out[:3]) if out else "Desconocido"
    if isinstance(value, dict):
        name = value.get("name")
        return str(name) if name else "Desconocido"

    text = str(value).strip()
    if not text:
        return "Desconocido"

    try:
        parsed = ast.literal_eval(text)
        return parse_genres(parsed)
    except Exception:
        pass

    text = text.replace("|", ",")
    text = text.replace("[", "").replace("]", "")
    text = text.replace("{", "").replace("}", "")
    parts = [p.strip().strip("'").strip('"') for p in text.split(",")]
    parts = [p for p in parts if p]
    return ", ".join(parts[:3]) if parts else "Desconocido"


def safe_float(v, default=0.0):
    try:
        return float(v)
    except Exception:
        return default


def safe_int(v, default=0):
    try:
        return int(float(v))
    except Exception:
        return default


def get_query_param(name: str):
    value = st.query_params.get(name)
    if isinstance(value, list):
        return value[0] if value else None
    return value


def set_query_params(**kwargs):
    for key, value in kwargs.items():
        if value is None:
            if key in st.query_params:
                del st.query_params[key]
        else:
            st.query_params[key] = str(value)


def go_to_page(page: str, tmdb_id: int | None = None):
    if tmdb_id is None:
        st.session_state.selected_tmdb_id = None
    set_query_params(page=page, tmdb_id=tmdb_id)
    st.rerun()


def resolve_user_id(recommender) -> int | None:
    direct = st.session_state["logged_user"]["user_id"]
    if direct is not None:
        try:
            return int(direct)
        except Exception:
            pass

    username = st.session_state.get("user")
    if not username:
        username = st.session_state.get("username")
    if not username:
        return None

    try:
        if recommender.users is not None and not recommender.users.empty:
            matches = recommender.users[
                recommender.users["username"].astype(str).str.lower() == str(username).lower()
            ]
            if not matches.empty:
                return int(matches.iloc[0]["user_id"])
    except Exception:
        pass

    return None


# Title lookup  exact match (case-insensitive) against the CSV
# Returns tmdb_id (int) if found, None otherwise
@st.cache_data
def load_title_index() -> pd.DataFrame:
    """
    Loads only the columns needed for title lookup from the CSV.
    Cached so it is only read once per session.
    """
    needed = ["title", "tmdbId", "tmdb_id", "id"]
    try:
        df = pd.read_csv(CSV_PATH, usecols=lambda c: c in needed, low_memory=False)
    except Exception:
        return pd.DataFrame(columns=["title", "tmdbId"])

    # Normalise the id column name
    if "tmdbId" not in df.columns:
        for alt in ("tmdb_id", "id"):
            if alt in df.columns:
                df = df.rename(columns={alt: "tmdbId"})
                break

    df["tmdbId"] = pd.to_numeric(df.get("tmdbId", pd.Series(dtype=float)), errors="coerce").fillna(0).astype("int64")
    df["title_lower"] = df["title"].astype(str).str.strip().str.lower()
    return df[["title", "title_lower", "tmdbId"]]


def find_movie_by_exact_title(query: str) -> int | None:
    """
    Returns the tmdbId of the first movie whose title matches `query`
    exactly (case-insensitive), or None if there is no match.
    """
    if not query or not query.strip():
        return None
    index = load_title_index()
    if index.empty:
        return None
    q = query.strip().lower()
    matches = index[index["title_lower"] == q]
    if matches.empty:
        return None
    tmdb_id = int(matches.iloc[0]["tmdbId"])
    return tmdb_id if tmdb_id != 0 else None


# Data normalization
def normalize_movie_df(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df

    df = df.copy()

    if "tmdbId" not in df.columns:
        for alt in ("tmdb_id", "movieId", "movie_id", "itemId", "item_id", "id"):
            if alt in df.columns:
                df = df.rename(columns={alt: "tmdbId"})
                break

    if "tmdbId" in df.columns:
        df["tmdbId"] = pd.to_numeric(df["tmdbId"], errors="coerce").fillna(0).astype("int64")

    if "year" not in df.columns:
        if "release_date" in df.columns:
            df["year"] = (
                pd.to_datetime(df["release_date"], errors="coerce")
                .dt.year.fillna(0).astype(int)
            )
        else:
            df["year"] = 0

    if "genres_text" not in df.columns:
        if "genres" in df.columns:
            df["genres_text"] = df["genres"].apply(parse_genres)
        else:
            df["genres_text"] = "Desconocido"

    for col in ("vote_average", "vote_count", "popularity", "runtime"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    return df


# Card builder
def build_movie_card(movie: pd.Series, card_id: str = "") -> str:
    poster = safe_poster_url(movie.get("poster_url"))
    poster = escape_text(poster, default=placeholder_poster())
    title = escape_text(movie.get("title", "Sin título"), "Sin título")
    year = int(movie.get("year", 0)) if pd.notna(movie.get("year", 0)) else 0
    rating = safe_float(movie.get("vote_average", 0))
    runtime = safe_int(movie.get("runtime", 0))
    runtime_text = f"{runtime}m" if runtime > 0 else "N/D"
    rating_label = f"★ {rating:.1f}"
    cid = f"mc_{card_id}"

    return f"""
<style>
#{cid} {{
    height: {CARD_HEIGHT}px;
    display: flex;
    flex-direction: column;
    border-radius: 14px;
    overflow: hidden;
    border: 1px solid rgba(255,255,255,0.08);
    box-shadow: 0 4px 16px rgba(0,0,0,0.4);
    background: #0f172a;
    transition: transform .2s ease, box-shadow .2s ease, border-color .2s ease;
    font-family: 'Outfit', Arial, sans-serif;
}}
#{cid}:hover {{
    transform: translateY(-4px);
    box-shadow: 0 14px 34px rgba(0,0,0,0.56);
    border-color: rgba(129,140,248,0.45);
}}
#{cid} .sv-pw {{
    flex: 0 0 72%;
    position: relative;
    overflow: hidden;
    background: #1e293b;
}}
#{cid} .sv-pw img {{
    width: 100%;
    height: 100%;
    object-fit: cover;
    display: block;
}}
#{cid} .sv-pw::after {{
    content: '';
    position: absolute;
    inset: 0;
    background: linear-gradient(to top,
        rgba(10,15,30,0.88) 0%,
        rgba(10,15,30,0.10) 62%,
        transparent 100%);
}}
#{cid} .sv-bdg {{
    position: absolute;
    top: 8px;
    right: 8px;
    background: rgba(17,24,39,0.86);
    border: 1px solid rgba(255,255,255,0.12);
    color: #fde68a;
    font-size: 11px;
    font-weight: 800;
    padding: 4px 8px;
    border-radius: 999px;
}}
#{cid} .sv-inf {{
    flex: 1;
    padding: 10px;
    display: flex;
    flex-direction: column;
    gap: 5px;
    background: linear-gradient(180deg, #111827 0%, #0f172a 100%);
}}
#{cid} .sv-ttl {{
    color: #f8fafc;
    font-size: 14px;
    font-weight: 800;
}}
#{cid} .sv-met {{
    display: flex;
    gap: 6px;
    color: #94a3b8;
    font-size: 11px;
}}
</style>

<div id="{cid}">
  <div class="sv-pw">
    <img src="{poster}" loading="lazy" />
    <div class="sv-bdg">{rating_label}</div>
  </div>
  <div class="sv-inf">
    <div class="sv-ttl">{title}</div>
    <div class="sv-met">
      <span>{year if year else "N/D"}</span>
      <span>•</span>
      <span>{runtime_text}</span>
    </div>
  </div>
</div>
"""


# Section renderer
def render_section(title: str, df: pd.DataFrame):
    count = len(df)
    st.markdown(
        f"""
        <div class="sv-section-bar">
          <h2>{title}</h2>
          <span class="sv-tab sv-tab-active">▶ Películas</span>
          <span class="sv-count-badge">{count} títulos</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if df.empty:
        st.markdown(
            '<div class="sv-empty">No hay títulos disponibles.</div>',
            unsafe_allow_html=True,
        )
        return

    per_row = 6
    for start in range(0, len(df), per_row):
        row = df.iloc[start:start + per_row]
        n_cols = min(per_row, len(row))
        cols = st.columns(n_cols, gap="small")
        for i, (_, movie) in enumerate(row.iterrows()):
            raw_id = movie.get("tmdbId", 0)
            tmdb_id = int(raw_id) if pd.notna(raw_id) else 0
            if tmdb_id == 0:
                continue

            with cols[i]:
                st.markdown(
                    build_movie_card(movie, card_id=f"{title}_{tmdb_id}"),
                    unsafe_allow_html=True,
                )
                if st.button(
                    "Ver Detalles",
                    key=f"btn_{title}_{tmdb_id}",
                    use_container_width=True,
                ):
                    go_to_page("moviecard", tmdb_id)


# Main dashboard
def show_dashboard(logout_callback=None):
    if not st.session_state.get("logged_in"):
        st.query_params["page"] = "login"
        st.rerun()
        return

    if "random_seed" not in st.session_state:
        st.session_state.random_seed = int(time.time()) % 100_000

    # Track the last search query that was processed so we don't
    # re-trigger navigation on every rerun caused by other widgets.
    if "_last_search_processed" not in st.session_state:
        st.session_state._last_search_processed = ""

    bg_b64 = get_bg_base64()
    logo_b64 = get_logo_base64()

    # Global CSS
    st.markdown(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&display=swap');

        :root {{ --primary-gradient: linear-gradient(135deg, #6366f1 0%, #a855f7 100%); }}

        .stApp {{
            background:
                linear-gradient(rgba(10,14,26,0.80), rgba(10,14,26,0.80)),
                url("data:image/png;base64,{bg_b64}");
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
            font-family: 'Outfit', sans-serif;
        }}

        #MainMenu, footer, header {{ visibility: hidden; }}

        .main .block-container {{
            max-width: 1600px !important;
            padding: 0 2vw 4vh 2vw !important;
        }}

        [data-testid="column"]:has([id^="mc_"]) [data-testid="stButton"],
        [data-testid="column"]:has([id^="mc_"]) [data-testid="stButton"] > button {{
            height: 0 !important;
            min-height: 0 !important;
            max-height: 0 !important;
            overflow: hidden !important;
            margin: 0 !important;
            padding: 0 !important;
            border: none !important;
            opacity: 0 !important;
            pointer-events: none !important;
            display: block !important;
        }}

        .sv-hero {{
            text-align: center;
            padding: 2.8vh 0 2.4vh 0;
        }}

        .sv-hero h1 {{
            margin: 0;
            font-size: clamp(24px, 4vh, 46px);
            font-weight: 900;
            color: #f8fafc;
            letter-spacing: -.025em;
        }}

        div[data-testid="stTextInput"] > div {{
            border-radius: 999px !important;
            background: rgba(15,23,42,0.75) !important;
            box-shadow: 0 10px 48px rgba(0,0,0,0.42),
                        inset 0 0 0 1.5px rgba(99,102,241,0.35) !important;
            border: none !important;
        }}

        div[data-testid="stTextInput"] input {{
            font-size: 2.2vh !important;
            padding: 1.8vh 2.4vw !important;
            color: #f8fafc !important;
            font-family: 'Outfit', sans-serif !important;
            font-weight: 600 !important;
            background: transparent !important;
            caret-color: #818cf8 !important;
        }}

        div[data-testid="stTextInput"] input::placeholder {{
            color: rgba(148,163,184,0.7) !important;
            font-weight: 400 !important;
        }}

        div.stTextInput {{
            margin-bottom: 3vh !important;
        }}

        .sv-section-bar {{
            display: flex;
            align-items: center;
            gap: 1vw;
            margin: 3.5vh 0 1.6vh 0;
            flex-wrap: wrap;
        }}

        .sv-section-bar h2 {{
            margin: 0;
            font-size: 2.4vh;
            font-weight: 900;
            color: #f8fafc;
            letter-spacing: -.01em;
        }}

        .sv-tab {{
            display: inline-flex;
            align-items: center;
            padding: 0.5vh 1.1vw;
            border-radius: 999px;
            font-size: 1.25vh;
            font-weight: 700;
            cursor: default;
            border: none;
        }}

        .sv-tab-active {{
            background: var(--primary-gradient);
            color: #fff;
        }}

        .sv-count-badge {{
            color: #475569;
            font-size: 1.25vh;
            margin-left: auto;
        }}

        .sv-empty {{
            text-align: center;
            padding: 4vh;
            color: #475569;
            font-size: 1.6vh;
        }}

        .sv-footer {{
            margin-top: 4vh;
            color: rgba(100,116,139,0.6);
            font-size: 1.2vh;
            text-align: center;
        }}

        input {{
            font-family: 'Outfit', sans-serif !important;
        }}

        [data-testid="column"] > div:first-child {{
            background: transparent !important;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )

    users_mtime = USERS_PATH.stat().st_mtime
    interactions_mtime = INTERACTIONS_PATH.stat().st_mtime
    recommender = get_recommender(users_mtime, interactions_mtime)
    user_id = resolve_user_id(recommender)
    seed = st.session_state.random_seed

    reco_warning = None

    # Recomendaciones
    try:
        if user_id is not None:
            df_recommended = recommender.recommend_with_explanations(
                user_id=int(user_id),
                n=FEATURED_LIMIT,
            )
            df_recommended = normalize_movie_df(df_recommended)
        else:
            reco_warning = "No se pudo resolver el usuario; se muestran populares como respaldo."
            df_recommended = pd.DataFrame()
    except Exception as e:
        st.error(f"Error en recomendaciones: {type(e).__name__} - {e}")
        reco_warning = "Error en recomendaciones; se muestran populares como respaldo."
        df_recommended = pd.DataFrame()

    # Populars
    try:
        if user_id is not None:
            df_popular = recommender.recommend_popular_with_explanations(
                user_id=int(user_id),
                n=FEATURED_LIMIT,
            )
            df_popular = normalize_movie_df(df_popular)
        else:
            df_popular = pd.DataFrame()
    except Exception as e:
        st.error(f"Error en populares: {type(e).__name__} - {e}")
        df_popular = pd.DataFrame()

    # Fallbacks
    if df_recommended is None or df_recommended.empty:
        df_recommended = df_popular.head(FEATURED_LIMIT).copy()

    if df_popular is None or df_popular.empty:
        df_popular = df_recommended.head(FEATURED_LIMIT).copy()

    # Random
    try:
        df_random = recommender.recommend_random(
            n=FEATURED_LIMIT,
            user_id=int(user_id) if user_id is not None else None,
            seed=seed,
        )
        df_random = normalize_movie_df(df_random)
    except Exception as e:
        st.error(f"Error en random: {type(e).__name__} - {e}")
        df_random = pd.DataFrame()

    # Topbar
    col_brand, col_user, col_nav = st.columns([1.2, 0.8, 2.0], gap="small")

    with col_brand:
        if logo_b64:
            st.image(f"data:image/png;base64,{logo_b64}", width=110)
        else:
            st.markdown("### StreamVortex")

    with col_user:
        st.write("")
        st.write(f"**{st.session_state.get('user', 'Usuario')}**")

    with col_nav:
        cn1, cn2, cn3 = st.columns(3, gap="small")
        with cn1:
            st.button("Películas", use_container_width=True, disabled=True, key="nav_movies_dashboard")
        with cn2:
            if st.button("Audit", key="nav_audit_dashboard", use_container_width=True):
                go_to_page("audit")
        with cn3:
            if logout_callback is not None:
                if st.button("Salir", key="nav_logout_dashboard", use_container_width=True):
                    logout_callback()

    st.divider()

    st.markdown(
        '<div class="sv-hero"><h1>Plataforma de recomendación de películas</h1></div>',
        unsafe_allow_html=True,
    )

    if reco_warning:
        st.info(reco_warning)

    _, sc, _ = st.columns([1, 4, 1])
    with sc:
        search = st.text_input(
            "search",
            placeholder="Buscar película por título exacto y pulsa Enter...",
            label_visibility="collapsed",
            key="dashboard_search",
        )

    # Search: exact title match navigate to moviecard
    # Only triggers when the value changes (new Enter press), so it
    # does NOT interfere with the rest of the dashboard on re-renders.
    search_query = (search or "").strip()
    if search_query and search_query != st.session_state._last_search_processed:
        st.session_state._last_search_processed = search_query
        tmdb_id = find_movie_by_exact_title(search_query)
        if tmdb_id is not None:
            go_to_page("moviecard", tmdb_id)
        # If no match do nothing, dashboard renders normally below

    # Sections — always shown unfiltered
    render_section("Recomendaciones", df_recommended)
    render_section("Populares", df_popular)
    render_section("Random", df_random)

    st.markdown(
        '<div class="sv-footer">© 2026 StreamVortex Platform · Dashboard de películas</div>',
        unsafe_allow_html=True,
    )