import ast
import base64
import os
import time
import html as html_lib
from pathlib import Path

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components


# ------------------------------------------------------------
# Config
# ------------------------------------------------------------
st.set_page_config(
    page_title="StreamVortex | Dashboard",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

ROOT_DIR = Path(__file__).resolve().parents[2]
CSV_PATH = ROOT_DIR / "movies" / "tmdb_dataset_full.csv"
BG_PATH = ROOT_DIR / "assets" / "fondo-login.png"

FEATURED_LIMIT = 24
POOL_LIMIT_PER_CHUNK = 80
CHUNKSIZE = 50000


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
def get_base64_of_bin_file(bin_file):
    with open(bin_file, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()


def get_bg_base64():
    try:
        return get_base64_of_bin_file(BG_PATH)
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
        <circle cx="450" cy="520" r="85" fill="#a855f7" opacity="0.24"/>
        <text x="50%" y="78%" font-family="Arial, sans-serif" font-size="56" fill="#cbd5e1" text-anchor="middle">
            Sin póster
        </text>
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


# ---------------------------------------------------------------
# NOTE: We do NOT cache this function (or use a per-seed cache)
# so that each "shuffle" call produces a fresh random sample.
# ---------------------------------------------------------------
@st.cache_data(show_spinner=True)
def load_candidate_pool(csv_path: str) -> pd.DataFrame:
    """
    Carga y preprocesa el dataset completo una sola vez, devolviendo
    un pool amplio ordenado por score. El muestreo aleatorio se hace
    después, fuera de la caché.
    """
    usecols = [
        "tmdbId",
        "title",
        "genres",
        "poster_url",
        "vote_average",
        "vote_count",
        "popularity",
        "runtime",
        "release_date",
        "overview",
    ]

    if not os.path.exists(csv_path):
        return pd.DataFrame(columns=usecols + ["genres_text", "year", "score"])

    best_chunks = []

    reader = pd.read_csv(
        csv_path,
        usecols=lambda c: c in usecols,
        chunksize=CHUNKSIZE,
        low_memory=False,
    )

    for chunk in reader:
        if chunk.empty:
            continue

        for col in ["vote_average", "vote_count", "popularity", "runtime"]:
            if col in chunk.columns:
                chunk[col] = pd.to_numeric(chunk[col], errors="coerce").fillna(0)

        if "title" in chunk.columns:
            chunk = chunk[chunk["title"].notna()]
            chunk["title"] = chunk["title"].astype(str).str.strip()

        if chunk.empty:
            continue

        chunk["genres_text"] = (
            chunk["genres"].apply(parse_genres) if "genres" in chunk.columns else "Desconocido"
        )

        if "release_date" in chunk.columns:
            chunk["year"] = (
                pd.to_datetime(chunk["release_date"], errors="coerce").dt.year.fillna(0).astype(int)
            )
        else:
            chunk["year"] = 0

        vote_scaled = (chunk["vote_average"].fillna(0) / 10.0) * 35
        pop_scaled = chunk["popularity"].fillna(0).rank(pct=True) * 40
        count_scaled = chunk["vote_count"].fillna(0).rank(pct=True) * 25
        chunk["score"] = vote_scaled + pop_scaled + count_scaled

        keep_n = min(POOL_LIMIT_PER_CHUNK * 4, len(chunk))
        best_chunks.append(chunk.nlargest(keep_n, "score"))

    if not best_chunks:
        return pd.DataFrame(columns=usecols + ["genres_text", "year", "score"])

    pool = pd.concat(best_chunks, ignore_index=True)
    pool = pool.drop_duplicates(subset=["tmdbId"], keep="first")
    pool = pool.sort_values(["score", "vote_count", "popularity"], ascending=False)
    return pool


def get_random_movies(csv_path: str, seed: int) -> pd.DataFrame:
    """
    A partir del pool cacheado, devuelve una muestra aleatoria ponderada
    por score para que siempre salgan películas de calidad pero variadas.
    """
    pool = load_candidate_pool(csv_path)
    if pool.empty:
        return pool

    candidates = pool.head(400).copy()

    weights = candidates["score"].clip(lower=0.01)
    n = min(FEATURED_LIMIT, len(candidates))

    sampled = candidates.sample(n=n, weights=weights, random_state=seed, replace=False)
    sampled = sampled.sort_values(["score", "vote_count"], ascending=False)
    return sampled.reset_index(drop=True)


def build_movie_card(movie: pd.Series, selected: bool = False) -> str:
    """
    Tarjeta autocontenida para usar dentro de components.html().
    Importante: incluye su propio CSS porque el iframe no hereda el CSS global de Streamlit.
    """
    border = "rgba(99, 102, 241, 0.95)" if selected else "rgba(255,255,255,0.10)"
    glow = "0 0 0 0.2vh rgba(99, 102, 241, 0.25)" if selected else "none"
    badge = "Destacada" if selected else "Película"

    poster = safe_poster_url(movie.get("poster_url"))
    poster = escape_text(poster, default=placeholder_poster())

    title = escape_text(movie.get("title", "Sin título"), "Sin título")
    genres = escape_text(movie.get("genres_text", "Desconocido"), "Desconocido")
    year = int(movie.get("year", 0)) if pd.notna(movie.get("year", 0)) else 0
    rating = float(movie.get("vote_average", 0) or 0)
    runtime = safe_int(movie.get("runtime", 0))
    runtime_text = f"{runtime} min" if runtime > 0 else "N/D"

    return f"""
    <!doctype html>
    <html>
    <head>
        <meta charset="utf-8">
        <style>
            html, body {{
                margin: 0;
                padding: 0;
                background: transparent;
                font-family: Arial, sans-serif;
                overflow: hidden;
            }}

            .movie-card {{
                box-sizing: border-box;
                width: 100%;
                height: 100%;
                background: rgba(30, 41, 59, 0.78);
                border: 1px solid {border};
                border-radius: 18px;
                overflow: hidden;
                box-shadow: {glow};
                display: flex;
                flex-direction: column;
                color: #f8fafc;
            }}

            .poster-wrap {{
                position: relative;
                width: 100%;
                aspect-ratio: 2 / 3;
                background: #0f172a;
                overflow: hidden;
                flex: 0 0 auto;
            }}

            .poster-img {{
                position: absolute;
                inset: 0;
                width: 100%;
                height: 100%;
                object-fit: cover;
                display: block;
            }}

            .poster-overlay {{
                position: absolute;
                inset: 0;
                background: linear-gradient(to top, rgba(15,23,42,0.9), transparent 45%);
                pointer-events: none;
            }}

            .movie-badge {{
                position: absolute;
                top: 10px;
                right: 10px;
                padding: 4px 10px;
                border-radius: 999px;
                background: rgba(15,23,42,0.82);
                border: 1px solid rgba(255,255,255,0.12);
                color: #f8fafc;
                font-size: 11px;
                font-weight: 700;
                z-index: 2;
            }}

            .movie-body {{
                padding: 12px 12px 14px 12px;
                display: flex;
                flex-direction: column;
                gap: 8px;
                flex: 1 1 auto;
                min-height: 0;
            }}

            .movie-head {{
                display: flex;
                justify-content: space-between;
                gap: 8px;
                align-items: flex-start;
            }}

            .movie-title {{
                margin: 0;
                color: #f8fafc;
                font-size: 14px;
                line-height: 1.2;
                font-weight: 700;
                flex: 1;
                min-width: 0;
                display: -webkit-box;
                -webkit-line-clamp: 2;
                -webkit-box-orient: vertical;
                overflow: hidden;
            }}

            .movie-rating {{
                flex-shrink: 0;
                font-size: 11px;
                color: #fde68a;
                background: rgba(250, 204, 21, 0.10);
                border: 1px solid rgba(250, 204, 21, 0.15);
                border-radius: 999px;
                padding: 4px 8px;
                white-space: nowrap;
            }}

            .movie-meta {{
                display: flex;
                flex-wrap: wrap;
                gap: 6px;
                color: #94a3b8;
                font-size: 11px;
                line-height: 1.35;
            }}

            .movie-meta span {{
                display: inline-flex;
                align-items: center;
                max-width: 100%;
                overflow: hidden;
                text-overflow: ellipsis;
                white-space: nowrap;
            }}
        </style>
    </head>
    <body>
        <div class="movie-card">
            <div class="poster-wrap">
                <img src="{poster}" class="poster-img" loading="lazy" />
                <div class="poster-overlay"></div>
                <div class="movie-badge">{badge}</div>
            </div>

            <div class="movie-body">
                <div class="movie-head">
                    <h3 class="movie-title">{title}</h3>
                    <div class="movie-rating">★ {rating:.1f}</div>
                </div>

                <div class="movie-meta">
                    <span>{genres}</span>
                    <span>{year if year else "Año N/D"}</span>
                    <span>{runtime_text}</span>
                </div>
            </div>
        </div>
    </body>
    </html>
    """


def show_dashboard(logout_callback=None):
    # ------------------------------------------------------------
    # Seed de aleatoriedad (persiste en sesión hasta que el usuario
    # pulse "Aleatorizar")
    # ------------------------------------------------------------
    if "random_seed" not in st.session_state:
        st.session_state.random_seed = int(time.time()) % 100_000

    # ------------------------------------------------------------
    # CSS global
    # ------------------------------------------------------------
    bg_b64 = get_bg_base64()

    st.markdown(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&display=swap');

        :root {{
            --primary-gradient: linear-gradient(135deg, #6366f1 0%, #a855f7 100%);
            --glass-bg: rgba(30, 41, 59, 0.72);
            --glass-border: rgba(255, 255, 255, 0.10);
            --text-main: #f8fafc;
            --text-muted: #94a3b8;
        }}

        .stApp {{
            background:
                linear-gradient(rgba(15, 23, 42, 0.72), rgba(15, 23, 42, 0.72)),
                url("data:image/png;base64,{bg_b64}");
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
            font-family: 'Outfit', sans-serif;
        }}

        #MainMenu, footer, header {{ visibility: hidden; }}

        .main .block-container {{
            max-width: 1600px !important;
            padding: 2vh 2.2vw 3vh 2.2vw !important;
        }}

        [data-testid="column"] > div:first-child {{
            background: transparent !important;
        }}

        [data-testid="stVerticalBlock"] > [data-testid="stVerticalBlockBorderWrapper"] {{
            background: transparent !important;
        }}

        .topbar {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 2vh;
            padding: 1.2vh 1.4vw;
            background: rgba(15, 23, 42, 0.35);
            border: 1px solid rgba(255,255,255,0.08);
            border-radius: 2vh;
            backdrop-filter: blur(18px);
        }}

        .brand h1 {{
            margin: 0;
            font-size: 2.9vh;
            font-weight: 800;
            background: var(--primary-gradient);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }}

        .brand p {{
            margin: 0;
            color: var(--text-muted);
            font-size: 1.55vh;
        }}

        .user-pill {{
            padding: 0.9vh 1vw;
            border-radius: 999px;
            background: rgba(99, 102, 241, 0.16);
            border: 1px solid rgba(129, 140, 248, 0.28);
            color: var(--text-main);
            font-size: 1.5vh;
            font-weight: 600;
        }}

        .hero {{
            background: linear-gradient(135deg, rgba(30,41,59,0.90), rgba(15,23,42,0.80));
            border: 1px solid rgba(255,255,255,0.10);
            border-radius: 3vh;
            padding: 2.2vh 2vw;
            box-shadow: 0 3vh 6vh -2vh rgba(0,0,0,0.55);
            margin-bottom: 2vh;
        }}

        .hero h2 {{
            margin: 0 0 0.8vh 0;
            font-size: 3.2vh;
            color: var(--text-main);
            font-weight: 800;
        }}

        .hero p {{
            margin: 0;
            color: var(--text-muted);
            font-size: 1.65vh;
        }}

        .metric-box {{
            margin-top: 1.4vh;
            padding: 1.2vh 1vw;
            border-radius: 2vh;
            background: rgba(15, 23, 42, 0.48);
            border: 1px solid rgba(255,255,255,0.08);
        }}

        .metric-label {{
            color: var(--text-muted);
            font-size: 1.3vh;
            margin-bottom: 0.4vh;
        }}

        .metric-value {{
            color: var(--text-main);
            font-size: 2.1vh;
            font-weight: 700;
        }}

        div[data-baseweb="input"],
        div[data-baseweb="select"] > div {{
            background-color: rgba(15, 23, 42, 0.78) !important;
            border-radius: 1.2vh !important;
            border: 0.1vh solid var(--glass-border) !important;
            color: var(--text-main) !important;
        }}

        input {{
            color: var(--text-main) !important;
            font-family: 'Outfit', sans-serif !important;
        }}

        .stButton button {{
            width: 100%;
            background: var(--primary-gradient) !important;
            color: white !important;
            border: none !important;
            border-radius: 1.2vh !important;
            font-weight: 700 !important;
            padding: 1vh 1.2vw !important;
            transition: all 0.3s ease !important;
            box-shadow: 0 1.2vh 2vh -0.8vh rgba(99, 102, 241, 0.5);
        }}

        .stButton button:hover {{
            transform: translateY(-0.2vh);
            box-shadow: 0 1.6vh 2.4vh -0.8vh rgba(99, 102, 241, 0.65);
        }}

        .stButton button:disabled {{
            opacity: 0.45 !important;
            transform: none !important;
            cursor: not-allowed !important;
        }}

        .section-title {{
            display: flex;
            justify-content: space-between;
            align-items: end;
            margin: 1.5vh 0 1vh;
        }}

        .section-title h3 {{
            margin: 0;
            font-size: 2.1vh;
            color: var(--text-main);
            font-weight: 700;
        }}

        .section-title span {{
            color: var(--text-muted);
            font-size: 1.4vh;
        }}

        .details-panel {{
            background: rgba(30,41,59,0.78);
            border: 1px solid rgba(255,255,255,0.12);
            border-radius: 2.4vh;
            padding: 1.5vh 1.1vw;
            backdrop-filter: blur(16px);
            box-shadow: 0 2vh 4vh -1.2vh rgba(0,0,0,0.35);
        }}

        .details-panel h3 {{
            margin: 0 0 1vh 0;
            color: var(--text-main);
            font-size: 2.05vh;
        }}

        .details-row {{
            display: flex;
            gap: 1vw;
            align-items: flex-start;
        }}

        .details-poster {{
            width: 38%;
            min-width: 100px;
            border-radius: 1.8vh;
            overflow: hidden;
            border: 1px solid rgba(255,255,255,0.10);
            flex-shrink: 0;
            position: relative;
            padding-top: 57%;
            background: #0f172a;
        }}

        .details-poster img {{
            position: absolute;
            inset: 0;
            width: 100%;
            height: 100%;
            display: block;
            object-fit: cover;
        }}

        .details-info {{
            flex: 1;
            min-width: 0;
        }}

        .details-title {{
            font-size: 1.8vh;
            font-weight: 800;
            margin-bottom: 0.4vh;
            color: var(--text-main);
            word-break: break-word;
        }}

        .details-sub {{
            color: var(--text-muted);
            font-size: 1.3vh;
            margin-bottom: 0.8vh;
        }}

        .details-overview {{
            color: #cbd5e1;
            font-size: 1.35vh;
            line-height: 1.5;
            margin-bottom: 1vh;
        }}

        .details-stats {{
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 0.6vw;
            margin-top: 0.8vh;
        }}

        .detail-box {{
            background: rgba(15,23,42,0.48);
            border: 1px solid rgba(255,255,255,0.08);
            border-radius: 1.5vh;
            padding: 0.8vh 0.7vw;
        }}

        .detail-box .k {{
            display: block;
            color: var(--text-muted);
            font-size: 1.1vh;
            margin-bottom: 0.3vh;
        }}

        .detail-box .v {{
            color: var(--text-main);
            font-size: 1.4vh;
            font-weight: 700;
        }}

        .footer-note {{
            margin-top: 2vh;
            color: rgba(148,163,184,0.75);
            font-size: 1.25vh;
            text-align: center;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )

    # ------------------------------------------------------------
    # Data — carga aleatoria usando la seed de sesión
    # ------------------------------------------------------------
    movies_df = get_random_movies(str(CSV_PATH), seed=st.session_state.random_seed)

    if "selected_tmdb_id" not in st.session_state:
        st.session_state.selected_tmdb_id = None

    if st.session_state.selected_tmdb_id is None and not movies_df.empty:
        st.session_state.selected_tmdb_id = int(movies_df.iloc[0]["tmdbId"])

    # ------------------------------------------------------------
    # Header
    # ------------------------------------------------------------
    top_left, top_right = st.columns([5, 2])
    with top_left:
        st.markdown(
            """
            <div class="topbar">
                <div class="brand">
                    <h1>StreamVortex</h1>
                    <p>Selecciona una película destacada del catálogo</p>
                </div>
                <div class="user-pill">Dashboard de selección</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with top_right:
        st.markdown(
            """
            <div class="topbar" style="justify-content:center; gap:0.8vw;">
                <div class="user-pill">Catálogo curado</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        btn_col1, btn_col2 = st.columns(2)
        with btn_col1:
            if st.button("🔀 Aleatorizar", key="shuffle_btn", use_container_width=True):
                st.session_state.random_seed = int(time.time()) % 100_000
                st.session_state.selected_tmdb_id = None
                st.rerun()
        with btn_col2:
            if logout_callback:
                if st.button("Cerrar sesión", key="logout_btn", use_container_width=True):
                    logout_callback()

    # ------------------------------------------------------------
    # Hero / Metrics
    # ------------------------------------------------------------
    hero_l, hero_r = st.columns([3, 2], gap="large")

    with hero_l:
        st.markdown(
            """
            <div class="hero">
                <h2>Explora películas destacadas</h2>
                <p>
                    Cada vez que pulses <strong style="color:#a5b4fc">🔀 Aleatorizar</strong>
                    se muestra una selección distinta y representativa del dataset,
                    con póster, género, año, duración y valoración.
                </p>
                <div class="metric-box">
                    <div class="metric-label">Modo catálogo</div>
                    <div class="metric-value">Selección aleatoria ponderada por calidad</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with hero_r:
        total_movies = len(movies_df)
        avg_rating = float(movies_df["vote_average"].mean()) if not movies_df.empty else 0
        total_genres = (
            len(
                {
                    g.strip()
                    for row in movies_df["genres_text"].dropna().astype(str).tolist()
                    for g in row.split(",")
                    if g.strip()
                }
            )
            if not movies_df.empty
            else 0
        )

        m1, m2, m3 = st.columns(3)
        with m1:
            st.markdown(
                f"""
                <div class="metric-box">
                    <div class="metric-label">Películas curadas</div>
                    <div class="metric-value">{total_movies}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with m2:
            st.markdown(
                f"""
                <div class="metric-box">
                    <div class="metric-label">Rating medio</div>
                    <div class="metric-value">{avg_rating:.1f}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with m3:
            st.markdown(
                f"""
                <div class="metric-box">
                    <div class="metric-label">Géneros</div>
                    <div class="metric-value">{total_genres}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # ------------------------------------------------------------
    # Filters
    # ------------------------------------------------------------
    if movies_df.empty:
        st.warning(f"No se pudo cargar el CSV en: {CSV_PATH}")
        st.stop()

    all_genres = sorted(
        {
            g.strip()
            for row in movies_df["genres_text"].dropna().astype(str).tolist()
            for g in row.split(",")
            if g.strip()
        }
    )

    years = movies_df["year"].fillna(0).astype(int)
    min_year = int(years[years > 0].min()) if (years > 0).any() else 1900
    max_year = int(years.max()) if len(years) else 2025

    f1, f2, f3 = st.columns([3, 2, 2])
    with f1:
        search = st.text_input(
            "Buscar película",
            placeholder="Título o parte del nombre...",
            label_visibility="visible",
        )
    with f2:
        genre_filter = st.selectbox("Género", ["Todos"] + all_genres)
    with f3:
        sort_by = st.selectbox("Ordenar por", ["Recomendadas", "Rating", "Popularidad", "Año"])

    year_col_1, year_col_2 = st.columns(2)
    with year_col_1:
        year_range = st.slider(
            "Rango de año",
            min_value=min_year,
            max_value=max_year,
            value=(max(min_year, max_year - 25), max_year),
        )
    with year_col_2:
        only_with_poster = st.checkbox("Solo con póster", value=True)

    # ------------------------------------------------------------
    # Apply filters
    # ------------------------------------------------------------
    filtered = movies_df.copy()

    if search.strip():
        q = search.strip().lower()
        filtered = filtered[filtered["title"].astype(str).str.lower().str.contains(q, na=False)]

    if genre_filter != "Todos":
        filtered = filtered[
            filtered["genres_text"].astype(str).str.lower().str.contains(genre_filter.lower(), na=False)
        ]

    filtered = filtered[(filtered["year"] >= year_range[0]) & (filtered["year"] <= year_range[1])]

    if only_with_poster:
        filtered = filtered[
            filtered["poster_url"].notna() & (filtered["poster_url"].astype(str).str.strip() != "")
        ]

    if sort_by == "Rating":
        filtered = filtered.sort_values(["vote_average", "vote_count"], ascending=False)
    elif sort_by == "Popularidad":
        filtered = filtered.sort_values(["popularity", "vote_count"], ascending=False)
    elif sort_by == "Año":
        filtered = filtered.sort_values(["year", "vote_average"], ascending=False)
    else:
        filtered = filtered.sort_values(["score", "vote_average", "vote_count"], ascending=False)

    display_df = filtered.head(FEATURED_LIMIT).copy()

    st.markdown(
        f"""
        <div class="section-title">
            <h3>Películas destacadas</h3>
            <span>{len(display_df)} resultados visibles · seed {st.session_state.random_seed}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ------------------------------------------------------------
    # Main content: grid + panel de detalles
    # ------------------------------------------------------------
    grid_col, detail_col = st.columns([3.2, 1.4], gap="large")

    with grid_col:
        if display_df.empty:
            st.markdown(
                """
                <div class="details-panel" style="text-align:center;">
                    No hay resultados con los filtros actuales.
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            per_row = 4 if len(display_df) >= 4 else max(1, len(display_df))

            for start in range(0, len(display_df), per_row):
                row = display_df.iloc[start : start + per_row]
                cols = st.columns(per_row, gap="medium")

                for i, (_, movie) in enumerate(row.iterrows()):
                    with cols[i]:
                        selected = int(movie["tmdbId"]) == int(st.session_state.selected_tmdb_id or -1)

                        components.html(
                            build_movie_card(movie, selected=selected),
                            height=410,
                            scrolling=False,
                        )

                        if st.button(
                            "Seleccionar",
                            key=f"select_{movie['tmdbId']}_{st.session_state.random_seed}",
                            use_container_width=True,
                            disabled=selected,
                        ):
                            st.session_state.selected_tmdb_id = int(movie["tmdbId"])
                            st.rerun()

    with detail_col:
        st.markdown(
            """
            <div class="section-title">
                <h3>Selección actual</h3>
            </div>
            """,
            unsafe_allow_html=True,
        )

        selected_movie = display_df[
            display_df["tmdbId"] == st.session_state.selected_tmdb_id
        ]
        if selected_movie.empty:
            selected_movie = movies_df[
                movies_df["tmdbId"] == st.session_state.selected_tmdb_id
            ]

        if not selected_movie.empty:
            movie = selected_movie.iloc[0]

            poster = safe_poster_url(movie.get("poster_url"))
            poster = escape_text(poster, default=placeholder_poster())

            title = escape_text(movie.get("title", "Sin título"), "Sin título")
            genres = escape_text(movie.get("genres_text", "Desconocido"), "Desconocido")
            year = int(movie.get("year", 0)) if pd.notna(movie.get("year", 0)) else 0
            runtime = safe_int(movie.get("runtime", 0))
            vote_average = safe_float(movie.get("vote_average", 0))
            vote_count = safe_int(movie.get("vote_count", 0))
            popularity = safe_float(movie.get("popularity", 0))
            overview = escape_text(
                movie.get("overview", "Vista rápida del contenido para selección."),
                "Vista rápida del contenido para selección.",
            )

            st.markdown(
                f"""
                <div class="details-panel">
                    <h3>{title}</h3>
                    <div class="details-row">
                        <div class="details-poster">
                            <img src="{poster}" />
                        </div>
                        <div class="details-info">
                            <div class="details-title">{title}</div>
                            <div class="details-sub">{genres} · {year if year else "Año N/D"}</div>
                            <div class="details-overview">
                                {overview}
                            </div>
                        </div>
                    </div>

                    <div class="details-stats">
                        <div class="detail-box">
                            <span class="k">Valoración</span>
                            <span class="v">★ {vote_average:.1f}</span>
                        </div>
                        <div class="detail-box">
                            <span class="k">Duración</span>
                            <span class="v">{runtime if runtime else "N/D"} min</span>
                        </div>
                        <div class="detail-box">
                            <span class="k">Votos</span>
                            <span class="v">{vote_count:,}</span>
                        </div>
                        <div class="detail-box">
                            <span class="k">Popularidad</span>
                            <span class="v">{popularity:.1f}</span>
                        </div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                """
                <div class="details-panel" style="text-align:center;">
                    Selecciona una película para verla aquí.
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown(
        """
        <div class="footer-note">
            © 2026 StreamVortex Platform · Dashboard de selección de películas
        </div>
        """,
        unsafe_allow_html=True,
    )