import ast
import base64
import html as html_lib
import time
from pathlib import Path
from datetime import datetime

import pandas as pd
import streamlit as st

from src.pages.dashboard import USERS_PATH
from src.movie_recommenders import build_recommender


# Config

ROOT_DIR = Path(__file__).resolve().parents[2]
CSV_PATH = ROOT_DIR / "movies" / "tmdb_dataset_full.csv"
INTERACTIONS_PATH = ROOT_DIR / "data" / "interactions.csv"
BG_PATH = ROOT_DIR / "assets" / "fondo-login.png"
LOGO_PATH = ROOT_DIR / "assets" / "logo-streaming.png"

# Paths required by the recommender (embeddings parquet is used by the engine)
USERS_CSV = ROOT_DIR / "data" / "users.csv"
EMBEDDINGS_CSV = ROOT_DIR / "data" / "embeddings" / "tmdb_embeddings.parquet"
RECO_CACHE_DIR = ROOT_DIR / "cache" / "reco_cache"

FEATURED_LIMIT = 12
CARD_HEIGHT = 370

STAR_OPTIONS = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]
STAR_LABELS = {
    0.0: "Sin valorar",
    0.5: "★½ Pésima",
    1.0: "★ Muy mala",
    1.5: "★½ Mala",
    2.0: "★★ Regular",
    2.5: "★★½ Mediocre",
    3.0: "★★★ Pasable",
    3.5: "★★★½ Bien",
    4.0: "★★★★ Muy bien",
    4.5: "★★★★½ Excelente",
    5.0: "★★★★★ Obra maestra",
}

INTERACTION_COLUMNS = ["interaction_id", "user_id", "movie_id", "rating", "valoration_date"]



# MODE 4 — Similar recommender loader (cached across Streamlit reruns)

@st.cache_resource(show_spinner=False)
def _load_similar_recommender(users_mtime, interactions_mtime):
    """
    Loads and fits the MovieRecommender once per Streamlit process.
    Used exclusively by get_similar_movies() to power the
    "Relacionado" section on the moviecard page (MODE 4 — Similar).
    """
    try:
        recommender = build_recommender(
            movies_csv=str(CSV_PATH),
            embeddings_parquet=str(EMBEDDINGS_CSV),
            users_csv=str(USERS_CSV),
            interactions_csv=str(INTERACTIONS_PATH),
            cache_dir=str(RECO_CACHE_DIR),
            force_rebuild_cache=False,
        )
        return recommender
    except Exception:
        return None


def get_similar_movies(reference_tmdb_id: int, user_id: str) -> pd.DataFrame:
    """
    Returns up to FEATURED_LIMIT movies that are genre-similar to the
    movie currently open on the detail page.
    """
    users_mtime = USERS_PATH.stat().st_mtime
    interactions_mtime = INTERACTIONS_PATH.stat().st_mtime
    recommender = _load_similar_recommender(users_mtime, interactions_mtime)
    if recommender is None:
        return pd.DataFrame()

    uid: int | None = None
    try:
        if user_id:
            uid = int(user_id)
    except (ValueError, TypeError):
        uid = None

    try:
        df = recommender.recommend_similar_with_explanations(
            reference_tmdb_id=reference_tmdb_id,
            user_id=uid,
            n=FEATURED_LIMIT,
        )
        return df
    except Exception:
        return pd.DataFrame()



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


def fmt_rating(rating: float) -> str:
    if rating <= 0 or rating >= 10:
        return "★ —"
    return f"★ {rating:.1f}"


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


def go_to_dashboard():
    st.session_state.selected_tmdb_id = None
    set_query_params(page="dashboard", tmdb_id=None)
    st.rerun()


def go_to_page(page: str, tmdb_id=None):
    if tmdb_id is None:
        st.session_state.selected_tmdb_id = None
    set_query_params(page=page, tmdb_id=tmdb_id)
    st.rerun()


def go_to_moviecard(tmdb_id: int):
    st.session_state.selected_tmdb_id = int(tmdb_id)
    set_query_params(page="moviecard", tmdb_id=int(tmdb_id))
    st.rerun()


def get_current_user_id() -> str:
    user = st.session_state.get("logged_user")

    if not user:
        return ""

    if isinstance(user, dict):
        return str(user.get("user_id", "")).strip()

    return ""



# Interactions CSV helpers

def _empty_interactions_df() -> pd.DataFrame:
    return pd.DataFrame(columns=INTERACTION_COLUMNS)


def _ensure_interactions_csv():
    """Crea el CSV de interacciones si no existe."""
    path = INTERACTIONS_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        _empty_interactions_df().to_csv(path, index=False)


def _load_interactions_df() -> pd.DataFrame:
    _ensure_interactions_csv()
    try:
        df = pd.read_csv(INTERACTIONS_PATH)
    except Exception:
        return _empty_interactions_df()

    for col in INTERACTION_COLUMNS:
        if col not in df.columns:
            df[col] = pd.NA

    df = df[INTERACTION_COLUMNS].copy()

    df["interaction_id"] = pd.to_numeric(df["interaction_id"], errors="coerce").astype("Int64")
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce")

    for col in ["user_id", "movie_id", "valoration_date"]:
        df[col] = df[col].where(df[col].notna(), "")
        df[col] = df[col].astype(str).str.strip()
        df[col] = df[col].replace({"nan": "", "None": "", "<NA>": ""})

    return df


def get_user_rating(user_id: str, tmdb_id: int) -> float | None:
    """
    Devuelve la valoración (0.0–5.0) que user_id dio a tmdb_id,
    o None si no existe ninguna.
    """
    if not user_id:
        return None

    try:
        df = _load_interactions_df()
        mask = (df["user_id"].astype(str) == str(user_id)) & (
            df["movie_id"].astype(str) == str(tmdb_id)
        )
        row = df[mask]
        if row.empty:
            return None
        rating = row.iloc[-1]["rating"]
        if pd.isna(rating):
            return None
        return float(rating)
    except Exception:
        return None


def save_user_rating(user_id: str, tmdb_id: int, rating: float):
    """
    Guarda o actualiza la valoración de user_id para tmdb_id.
    """
    _ensure_interactions_csv()

    try:
        df = _load_interactions_df()
    except Exception:
        df = _empty_interactions_df()

    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    if df.empty:
        max_id = 0
    else:
        max_id = pd.to_numeric(df["interaction_id"], errors="coerce").fillna(0).max()
        try:
            max_id = int(max_id)
        except Exception:
            max_id = 0

    mask = (df["user_id"].astype(str) == str(user_id)) & (
        df["movie_id"].astype(str) == str(tmdb_id)
    )

    if mask.any():
        idx = df[mask].index[-1]
        df.at[idx, "rating"] = float(rating)
        df.at[idx, "valoration_date"] = now_str
        df.at[idx, "user_id"] = str(user_id)
        df.at[idx, "movie_id"] = str(tmdb_id)
        if pd.isna(df.at[idx, "interaction_id"]):
            df.at[idx, "interaction_id"] = max_id + 1
    else:
        new_row = {
            "interaction_id": max_id + 1,
            "user_id": str(user_id),
            "movie_id": str(tmdb_id),
            "rating": float(rating),
            "valoration_date": now_str,
        }
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

    df.to_csv(INTERACTIONS_PATH, index=False)


def delete_user_rating(user_id: str, tmdb_id: int):
    """Elimina la valoración del usuario para esa película."""
    if not user_id:
        return

    _ensure_interactions_csv()
    try:
        df = _load_interactions_df()
        mask = (df["user_id"].astype(str) == str(user_id)) & (
            df["movie_id"].astype(str) == str(tmdb_id)
        )
        df = df[~mask]
        df.to_csv(INTERACTIONS_PATH, index=False)
    except Exception:
        pass



# Star rating renderer

def render_star_widget(tmdb_id: int, user_id: str):
    """
    Renderiza el widget de valoración por estrellas (0–5, paso 0.5).
    """
    current = get_user_rating(user_id, tmdb_id)
    has_rating = current is not None
    current_rating = current if has_rating else 0.0

    slider_key = f"star_slider_{tmdb_id}"
    if slider_key not in st.session_state:
        st.session_state[slider_key] = current_rating if current_rating in STAR_OPTIONS else 0.0
    elif st.session_state[slider_key] not in STAR_OPTIONS:
        st.session_state[slider_key] = current_rating if current_rating in STAR_OPTIONS else 0.0

    selected = st.select_slider(
        label="Valorar",
        options=STAR_OPTIONS,
        format_func=lambda v: f"{v} ★" if v > 0 else "—",
        key=slider_key,
    )

    preview_rating = selected if selected in STAR_OPTIONS else current_rating
    label = STAR_LABELS.get(preview_rating, "")

    def stars_html(value: float, uid: str) -> str:
        full = int(value)
        half = 1 if (value - full) >= 0.5 else 0
        empty = 5 - full - half

        grad_id = f"hg_{uid}"

        star_full = (
            '<svg viewBox="0 0 24 24" width="30" height="30" aria-hidden="true">'
            '<path d="M12 2l3.09 6.26L22 9.27l-5 4.87 1.18 6.88L12 17.77l-6.18 3.25'
            'L7 14.14 2 9.27l6.91-1.01L12 2z" fill="#f59e0b" stroke="#f59e0b" stroke-width="1"/>'
            "</svg>"
        )
        star_half = (
            f'<svg viewBox="0 0 24 24" width="30" height="30" aria-hidden="true">'
            f'<defs><linearGradient id="{grad_id}"><stop offset="50%" stop-color="#f59e0b"/>'
            f'<stop offset="50%" stop-color="rgba(148,163,184,0.18)"/></linearGradient></defs>'
            f'<path d="M12 2l3.09 6.26L22 9.27l-5 4.87 1.18 6.88L12 17.77l-6.18 3.25'
            f'L7 14.14 2 9.27l6.91-1.01L12 2z" fill="url(#{grad_id})" stroke="#f59e0b" stroke-width="1"/>'
            "</svg>"
        )
        star_empty = (
            '<svg viewBox="0 0 24 24" width="30" height="30" aria-hidden="true">'
            '<path d="M12 2l3.09 6.26L22 9.27l-5 4.87 1.18 6.88L12 17.77l-6.18 3.25'
            'L7 14.14 2 9.27l6.91-1.01L12 2z" fill="rgba(148,163,184,0.18)" stroke="rgba(148,163,184,0.65)" stroke-width="1.3"/>'
            "</svg>"
        )
        return star_full * full + star_half * half + star_empty * empty

    if preview_rating and preview_rating > 0:
        status_text = (
            f"Valoración actual: <strong style='color:#f59e0b'>{preview_rating} / 5</strong> — {label}"
        )
    else:
        status_text = "<span style='color:#94a3b8'>Aún no has valorado esta película</span>"

    st.markdown(
        f"""
        <div style="
            background: rgba(15,23,42,0.82);
            border: 1px solid rgba(255,255,255,0.16);
            border-radius: 16px;
            padding: 16px 18px 12px 18px;
            margin-top: 4px;
            box-shadow: 0 10px 26px rgba(0,0,0,0.30);
            backdrop-filter: blur(10px);
            -webkit-backdrop-filter: blur(10px);
        ">
          <div style="font-size:12px;font-weight:800;color:#cbd5e1;
                      letter-spacing:.06em;text-transform:uppercase;margin-bottom:10px;">
            Tu valoración
          </div>
          <div style="display:flex;align-items:center;gap:4px;margin-bottom:8px;">
            {stars_html(preview_rating, str(tmdb_id))}
          </div>
          <div style="font-size:13px;color:#e2e8f0;min-height:18px;">{status_text}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <style>
        div[data-testid="stSlider"] > label { display: none !important; }
        div[data-testid="stSlider"] .st-emotion-cache-1dp5vir,
        div[data-testid="stSlider"] [class*="thumb"] {
            background: #6366f1 !important;
        }
        div[data-testid="stSlider"] [class*="track"] {
            background: linear-gradient(90deg,#6366f1,#a855f7) !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    col_save, col_del = st.columns([3, 2], gap="small")

    with col_save:
        save_label = "Actualizar valoración" if has_rating else "Guardar valoración"
        if st.button(
            save_label,
            key=f"star_save_{tmdb_id}",
            use_container_width=True,
            disabled=(selected == 0.0) or (not bool(user_id)),
            type="primary",
        ):
            save_user_rating(user_id, tmdb_id, selected)
            st.success(f"Valoración guardada: {selected} ★ — {STAR_LABELS[selected]}")
            st.rerun()



# Full movie catalog lookup

@st.cache_data(show_spinner=False)
def load_movie_catalog(csv_path: str) -> pd.DataFrame:
    """
    Loads the full CSV used for movie lookup on the detail page.
    This is intentionally NOT filtered by score or chunk ranking.
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

    if not Path(csv_path).exists():
        return pd.DataFrame(columns=usecols + ["genres_text", "year"])

    df = pd.read_csv(
        csv_path,
        usecols=lambda c: c in usecols,
        low_memory=False,
    )

    if "tmdbId" in df.columns:
        df["tmdbId"] = pd.to_numeric(df["tmdbId"], errors="coerce").fillna(0).astype("int64")

    for col in ["vote_average", "vote_count", "popularity", "runtime"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    if "title" in df.columns:
        df["title"] = df["title"].astype(str).str.strip()

    if "genres" in df.columns:
        df["genres_text"] = df["genres"].apply(parse_genres)
    else:
        df["genres_text"] = "Desconocido"

    if "release_date" in df.columns:
        df["year"] = (
            pd.to_datetime(df["release_date"], errors="coerce")
            .dt.year.fillna(0)
            .astype(int)
        )
    else:
        df["year"] = 0

    return df


def get_movie_by_tmdb_id(csv_path: str, tmdb_id: int):
    """
    Looks up the movie in the FULL CSV, not a recency / score-trimmed pool.
    """
    catalog = load_movie_catalog(csv_path)
    if catalog.empty:
        return None

    matches = catalog[catalog["tmdbId"] == tmdb_id]
    if matches.empty:
        return None
    return matches.iloc[0]



# Shared helper: coerce recommender output to moviecard shape

def _normalise_recommender_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    The recommender returns columns like genres_list (a Python list).
    render_section / build_movie_card expect genres_text (a string)
    and a plain integer tmdbId.
    """
    if df.empty:
        return df

    out = df.copy()

    if "genres_list" in out.columns and "genres_text" not in out.columns:
        out["genres_text"] = out["genres_list"].apply(
            lambda gl: ", ".join(str(g) for g in gl[:3]) if gl else "Desconocido"
        )

    if "year" not in out.columns and "release_date" in out.columns:
        out["year"] = (
            pd.to_datetime(out["release_date"], errors="coerce")
            .dt.year.fillna(0)
            .astype(int)
        )
    elif "year" not in out.columns:
        out["year"] = 0

    if "tmdbId" in out.columns:
        out["tmdbId"] = pd.to_numeric(out["tmdbId"], errors="coerce").fillna(0).astype("int64")

    return out



# Card builder

def build_movie_card(movie: pd.Series, card_id: str = "") -> str:
    poster = safe_poster_url(movie.get("poster_url"))
    poster = escape_text(poster, default=placeholder_poster())
    title = escape_text(movie.get("title", "Sin título"), "Sin título")
    year = int(movie.get("year", 0)) if pd.notna(movie.get("year", 0)) else 0
    rating = safe_float(movie.get("vote_average", 0))
    runtime = safe_int(movie.get("runtime", 0))
    runtime_text = f"{runtime}m" if runtime > 0 else "N/D"
    rating_label = fmt_rating(rating)
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
    cursor: default;
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
    pointer-events: none;
}}
#{cid} .sv-ph {{
    position: absolute;
    left: 10px;
    bottom: 10px;
    z-index: 2;
    padding: 4px 8px;
    border-radius: 999px;
    background: rgba(15,23,42,0.72);
    border: 1px solid rgba(255,255,255,0.10);
    color: #e2e8f0;
    font-size: 10px;
    font-weight: 800;
    opacity: 0;
    transform: translateY(4px);
    transition: opacity .2s ease, transform .2s ease;
    pointer-events: none;
}}
#{cid}:hover .sv-ph {{
    opacity: 1;
    transform: translateY(0);
}}
#{cid} .sv-bdg {{
    position: absolute;
    top: 8px;
    right: 8px;
    background: rgba(17,24,39,0.86);
    backdrop-filter: blur(10px);
    border: 1px solid rgba(255,255,255,0.12);
    color: #fde68a;
    font-size: 11px;
    font-weight: 800;
    padding: 4px 8px;
    border-radius: 999px;
    z-index: 2;
}}
#{cid} .sv-inf {{
    flex: 1;
    padding: 10px 10px 9px;
    display: flex;
    flex-direction: column;
    gap: 5px;
    background: linear-gradient(180deg, #111827 0%, #0f172a 100%);
}}
#{cid} .sv-ttl {{
    color: #f8fafc;
    font-size: 14px;
    font-weight: 800;
    line-height: 1.28;
    display: -webkit-box;
    -webkit-line-clamp: 2;
    -webkit-box-orient: vertical;
    overflow: hidden;
    letter-spacing: -0.01em;
}}
#{cid} .sv-met {{
    display: flex;
    gap: 6px;
    flex-wrap: wrap;
    color: #94a3b8;
    font-size: 11px;
    font-weight: 600;
    align-items: center;
}}
#{cid} .sv-dot {{
    width: 3px;
    height: 3px;
    border-radius: 50%;
    background: #334155;
    flex-shrink: 0;
    display: inline-block;
}}
#{cid} .sv-tag {{
    border: 1px solid #334155;
    color: #cbd5e1;
    font-size: 10px;
    font-weight: 700;
    padding: 2px 6px;
    border-radius: 999px;
}}
</style>
<div id="{cid}" data-cardid="{cid}">
  <div class="sv-pw">
    <img src="{poster}" loading="lazy" />
    <div class="sv-bdg">{rating_label}</div>
    <div class="sv-ph">▶ Vista previa</div>
  </div>
  <div class="sv-inf">
    <div class="sv-ttl">{title}</div>
    <div class="sv-met">
      <span>{year if year else "N/D"}</span>
      <span class="sv-dot"></span>
      <span>{runtime_text}</span>
      <span class="sv-dot"></span>
      <span class="sv-tag">Movie</span>
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
            '<div class="sv-empty">No se encontró ningún título con ese nombre.</div>',
            unsafe_allow_html=True,
        )
        return

    per_row = 6
    for start in range(0, len(df), per_row):
        row = df.iloc[start:start + per_row]
        n_cols = min(per_row, len(row))
        cols = st.columns(n_cols, gap="small")
        for i, (_, movie) in enumerate(row.iterrows()):
            tmdb_id = int(movie["tmdbId"])
            with cols[i]:
                st.markdown(
                    build_movie_card(movie, card_id=f"{title}_{tmdb_id}"),
                    unsafe_allow_html=True,
                )
                if st.button("▶ Abrir", key=f"btn_{title}_{tmdb_id}", use_container_width=True):
                    go_to_moviecard(tmdb_id)



# Detail page renderer

def render_movie_detail(movie: pd.Series, logout_callback=None):
    bg_b64 = get_bg_base64()
    logo_b64 = get_logo_base64()

    poster_url = escape_text(
        safe_poster_url(movie.get("poster_url")),
        default=placeholder_poster(),
    )
    title = escape_text(movie.get("title", "Sin título"), "Sin título")
    genres = escape_text(movie.get("genres_text", "Desconocido"), "Desconocido")
    overview = escape_text(
        movie.get("overview", "Sin descripción disponible."),
        "Sin descripción disponible.",
    )
    year = int(movie.get("year", 0)) if pd.notna(movie.get("year", 0)) else 0
    runtime = safe_int(movie.get("runtime", 0))
    vote_avg = safe_float(movie.get("vote_average", 0))
    vote_count = safe_int(movie.get("vote_count", 0))
    popularity = safe_float(movie.get("popularity", 0))
    tmdb_id = safe_int(movie.get("tmdbId", -1), -1)
    release = escape_text(movie.get("release_date", "N/D"), "N/D")

    user_id = get_current_user_id()

    st.markdown(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800;900&display=swap');

        :root {{
            --primary-gradient: linear-gradient(135deg, #6366f1 0%, #a855f7 100%);
            --text-main: #f8fafc;
            --text-muted: #94a3b8;
        }}

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
            padding: 0 2vw 1vh 2vw !important;
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
        .sv-tab-active {{ background: var(--primary-gradient); color: #fff; }}
        .sv-count-badge {{ color: #475569; font-size: 1.25vh; margin-left: auto; }}
        .sv-empty {{ text-align: center; padding: 4vh; color: #475569; font-size: 1.6vh; }}

        .detail-shell {{ min-height: 100vh; color: var(--text-main); }}

        .detail-content {{
            padding: 0 18px 18px 18px;
            margin-top: 0 !important;
        }}

        .detail-grid {{
            display: grid;
            grid-template-columns: 260px 1fr;
            gap: 28px;
            align-items: start;
        }}

        .poster-panel {{
            background: rgba(18,18,18,0.78);
            backdrop-filter: blur(12px);
            border: 1px solid rgba(255,255,255,0.08);
            border-radius: 22px;
            padding: 14px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.40);
        }}
        .poster-panel img {{
            width: 100%;
            border-radius: 18px;
            display: block;
            object-fit: cover;
            aspect-ratio: 2 / 3;
        }}

        .watch-pill {{
            display: inline-flex;
            align-items: center;
            gap: 8px;
            background: var(--primary-gradient);
            color: #fff;
            font-weight: 900;
            border-radius: 999px;
            padding: 12px 20px;
            box-shadow: 0 10px 24px rgba(251,67,104,0.35);
            margin-bottom: 10px;
        }}

        .main-title {{
            font-size: clamp(32px, 4vw, 62px);
            line-height: 1.02;
            font-weight: 900;
            margin: 0 0 16px 0;
            letter-spacing: -0.04em;
            color: #fff;
        }}

        .meta-row {{
            display: flex;
            gap: 10px;
            flex-wrap: wrap;
            align-items: center;
            margin-bottom: 16px;
        }}

        .pill {{
            display: inline-flex;
            align-items: center;
            gap: 6px;
            padding: 8px 12px;
            border-radius: 10px;
            font-size: 13px;
            font-weight: 800;
            background: rgba(255,255,255,0.95);
            color: #111827;
        }}
        .pill.dark {{
            background: rgba(255,255,255,0.10);
            color: #fff;
            border: 1px solid rgba(255,255,255,0.12);
        }}
        .pill.orange {{
            background: transparent;
            color: #f59e0b;
            padding-left: 0;
        }}

        .overview {{
            font-size: 17px;
            line-height: 1.72;
            color: rgba(255,255,255,0.93);
            max-width: 1100px;
            margin-bottom: 18px;
        }}

        .info-grid {{
            display: grid;
            grid-template-columns: repeat(2, minmax(0, 1fr));
            gap: 10px 24px;
            max-width: 1100px;
        }}
        .info-item {{
            font-size: 17px;
            line-height: 1.5;
            color: rgba(255,255,255,0.94);
        }}
        .info-item strong {{ color: #fff; font-weight: 800; }}

        [data-testid="stVerticalBlock"] > [data-testid="element-container"],
        [data-testid="stVerticalBlock"] > [data-testid="stMarkdownContainer"],
        [data-testid="stVerticalBlock"] > div > [data-testid="stMarkdownContainer"] {{
            margin-top: 0 !important;
            margin-bottom: 0 !important;
            padding-top: 0 !important;
            padding-bottom: 0 !important;
        }}
        .main .block-container > div > div > div > div[data-testid="stVerticalBlock"] {{
            gap: 0.25rem !important;
        }}

        @media (max-width: 1100px) {{
            .detail-grid {{ grid-template-columns: 1fr; }}
            .info-grid {{ grid-template-columns: 1fr; }}
        }}
        @media (max-width: 700px) {{
            .detail-content {{ padding: 0 10px 14px 10px; }}
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Topbar
    col_brand, col_user, col_nav = st.columns([1.2, 0.8, 2.0], gap="small")

    with col_brand:
        if logo_b64:
            st.image(f"data:image/png;base64,{logo_b64}", width=110)
        else:
            st.markdown("### StreamVortex")

    with col_user:
        st.write("")
        st.write(f"👤 **{st.session_state.get('user', 'Usuario')}**")

    with col_nav:
        cn1, cn2, cn3 = st.columns(3, gap="small")
        with cn1:
            if st.button("Películas", key="nav_movies_moviecard", use_container_width=True):
                go_to_dashboard()
        with cn2:
            if st.button("Audit", key="nav_audit_moviecard", use_container_width=True):
                go_to_page("audit")
        with cn3:
            if logout_callback is not None:
                if st.button("Salir", key="nav_logout_moviecard", use_container_width=True):
                    logout_callback()

    st.divider()

    left, right = st.columns([0.9, 3.1], gap="large")

    with left:
        st.markdown(
            f"""
            <div class="poster-panel">
                <img src="{poster_url}" />
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
        render_star_widget(tmdb_id, user_id)

    with right:
        st.markdown(
            f"""
            <div class="watch-pill">Ver ahora</div>
            <div class="main-title">{title}</div>
            <div class="meta-row">
                <span class="pill">Tráiler</span>
                <span class="pill dark">FHD</span>
                <span class="pill orange">IMDb: {fmt_rating(vote_avg)}</span>
            </div>
            <div class="overview">{overview}</div>
            <div class="info-grid">
                <div class="info-item"><strong>Estreno:</strong> {release}</div>
                <div class="info-item"><strong>Duración:</strong> {runtime if runtime else "N/D"} min</div>
                <div class="info-item"><strong>Género:</strong> {genres}</div>
                <div class="info-item"><strong>Votos:</strong> {vote_count:,}</div>
                <div class="info-item"><strong>Popularidad:</strong> {popularity:.0f}</div>
                <div class="info-item"><strong>TMDB ID:</strong> {tmdb_id}</div>
                <div class="info-item"><strong>Nota media:</strong> {vote_avg:.1f} / 10</div>
                <div class="info-item"><strong>Año:</strong> {year if year else "N/D"}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Related movies (uses recommender)
    df_related = get_similar_movies(
        reference_tmdb_id=tmdb_id,
        user_id=user_id,
    )

    df_related = _normalise_recommender_df(df_related)
    render_section("Relacionado", df_related)



# Entry point called from app.py

def show_movie_detail(logout_callback=None):
    if not st.session_state.get("logged_in"):
        st.session_state.selected_tmdb_id = None
        set_query_params(page="login", tmdb_id=None)
        st.rerun()
        return

    csv = str(CSV_PATH)

    tmdb_id = get_query_param("tmdb_id")
    if not tmdb_id:
        tmdb_id = st.session_state.get("selected_tmdb_id")

    if not tmdb_id:
        set_query_params(page="dashboard", tmdb_id=None)
        st.rerun()
        return

    try:
        tmdb_id = int(tmdb_id)
    except Exception:
        st.error("TMDB ID inválido.")
        st.session_state.selected_tmdb_id = None
        set_query_params(page="dashboard", tmdb_id=None)
        st.rerun()
        return

    movie = get_movie_by_tmdb_id(csv, tmdb_id)

    if movie is None:
        st.error("Película no encontrada.")
        st.session_state.selected_tmdb_id = None
        set_query_params(page="dashboard", tmdb_id=None)
        st.rerun()
        return

    st.session_state.selected_tmdb_id = tmdb_id
    render_movie_detail(movie, logout_callback=logout_callback)