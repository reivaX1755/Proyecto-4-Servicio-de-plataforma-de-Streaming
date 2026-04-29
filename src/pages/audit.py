# src/pages/audit.py
"""
Página de auditoría del pipeline de recomendación.

Objetivos:
- ver el perfil del usuario
- ver las 12 recomendaciones del modelo principal
- ver las 12 películas populares
- mostrar el breakdown de cada score
- en el perfil, ver qué películas ha valorado el usuario y a qué géneros pertenecen
- explicar cómo se calcula la puntuación
- mostrar el impacto de embeddings en el score de recomendaciones

Añade esta página a tu app principal con:
    elif st.session_state.page == "audit":
        from src.pages.audit import show_audit
        show_audit()
"""

from __future__ import annotations

import ast
import base64
from copy import deepcopy
from pathlib import Path
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from src.movie_recommenders import build_recommender


# Paths

ROOT_DIR = Path(__file__).resolve().parents[2]
CSV_PATH = ROOT_DIR / "movies" / "tmdb_dataset_full.csv"
EMBEDDINGS_PATH = ROOT_DIR / "data" / "embeddings" / "tmdb_embeddings.parquet"
USERS_PATH = ROOT_DIR / "data" / "users.csv"
INTERACTIONS_PATH = ROOT_DIR / "data" / "interactions.csv"
CACHE_DIR = ROOT_DIR / "cache" / "reco_cache"



# Recommender compartido con el dashboard

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



# Navegación

def set_query_params(**kwargs):
    for key, value in kwargs.items():
        if value is None:
            if key in st.query_params:
                del st.query_params[key]
        else:
            st.query_params[key] = str(value)


def go_to_page(page: str, tmdb_id: int | None = None):
    set_query_params(page=page, tmdb_id=tmdb_id)
    st.rerun()



# Utils de parsing / formato

def normalize_genres_list(value) -> list[str]:
    """
    Devuelve una lista plana de géneros sin disparar el ValueError de pd.isna
    con arrays o listas.
    """
    if value is None:
        return []

    if isinstance(value, list):
        out: list[str] = []
        for item in value:
            if isinstance(item, dict):
                name = item.get("name")
                if name:
                    out.append(str(name).strip())
            else:
                s = str(item).strip()
                if s:
                    out.append(s)
        return out

    if isinstance(value, dict):
        name = value.get("name")
        return [str(name).strip()] if name else []

    if isinstance(value, (np.ndarray, pd.Series)):
        try:
            return normalize_genres_list(list(value))
        except Exception:
            return []

    try:
        if pd.isna(value):
            return []
    except Exception:
        pass

    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null"}:
        return []

    try:
        parsed = ast.literal_eval(text)
        return normalize_genres_list(parsed)
    except Exception:
        pass

    text = text.replace("|", ",")
    text = text.replace("[", "").replace("]", "")
    text = text.replace("{", "").replace("}", "")
    parts = [p.strip().strip("'").strip('"') for p in text.split(",")]
    return [p for p in parts if p]


def parse_genres(value):
    genres = normalize_genres_list(value)
    if not genres:
        return "Desconocido"
    return ", ".join(genres[:3])


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


def _minmax(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce").fillna(0.0)
    if s.empty:
        return s
    mn = s.min()
    mx = s.max()
    if pd.isna(mn) or pd.isna(mx) or mx == mn:
        return pd.Series([0.0] * len(s), index=s.index, dtype="float32")
    return ((s - mn) / (mx - mn)).astype("float32")


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


def normalize_movie_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normaliza los dataframes devueltos por el recomendador para que el
    dashboard/auditoría puedan trabajar siempre con las mismas columnas.
    """
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
            df["year"] = pd.to_datetime(df["release_date"], errors="coerce").dt.year.fillna(0).astype(int)
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


def get_user_id_from_session(rec) -> int | None:
    direct = st.session_state.get("logged_user", {}).get("user_id")
    if direct is not None:
        try:
            return int(direct)
        except Exception:
            pass

    username = st.session_state.get("user") or st.session_state.get("username")
    if not username:
        return None

    try:
        if rec.users is not None and not rec.users.empty:
            matches = rec.users[
                rec.users["username"].astype(str).str.lower() == str(username).lower()
            ]
            if not matches.empty:
                return int(matches.iloc[0]["user_id"])
    except Exception:
        pass

    return None



# Helpers de UI

def _section_title(title: str, icon: str = "◆") -> None:
    st.markdown(
        f"""
        <div style="display:flex;align-items:center;gap:10px;
                    margin:2rem 0 1rem;border-bottom:1px solid #1e293b;
                    padding-bottom:10px;">
          <span style="color:#6366f1;font-size:18px;">{icon}</span>
          <span style="color:#f8fafc;font-size:17px;font-weight:800;
                       letter-spacing:-.01em;">{title}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _metric_card(label: str, value: str, sub: str = "", color: str = "#6366f1") -> str:
    return f"""
    <div style="background:#0f172a;border:1px solid #1e293b;border-radius:12px;
                padding:16px 20px;text-align:center;">
      <div style="color:#64748b;font-size:11px;font-weight:700;
                  text-transform:uppercase;letter-spacing:.08em;">{label}</div>
      <div style="color:{color};font-size:28px;font-weight:900;
                  margin:6px 0 2px;">{value}</div>
      <div style="color:#475569;font-size:11px;">{sub}</div>
    </div>
    """


def _badge(text: str, color: str = "#6366f1") -> str:
    return (
        f'<span style="background:{color}22;color:{color};border:1px solid {color}55;'
        f'padding:2px 10px;border-radius:999px;font-size:12px;font-weight:700;">'
        f"{text}</span>"
    )


def _plot_group(df_all: pd.DataFrame, df_top: pd.DataFrame, x_col: str, y_col: str, title: str):
    if df_all is None or df_all.empty or x_col not in df_all.columns or y_col not in df_all.columns:
        st.info("Sin datos para graficar.")
        return

    all_df = df_all.copy()
    top_ids = set(df_top["tmdbId"].tolist()) if df_top is not None and not df_top.empty and "tmdbId" in df_top.columns else set()

    all_df[x_col] = pd.to_numeric(all_df[x_col], errors="coerce").fillna(0)
    all_df[y_col] = pd.to_numeric(all_df[y_col], errors="coerce").fillna(0)

    top_df = all_df[all_df["tmdbId"].isin(top_ids)].copy()
    base_df = all_df[~all_df["tmdbId"].isin(top_ids)].copy()

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.scatter(base_df[x_col], base_df[y_col], s=22, alpha=0.22)
    ax.scatter(top_df[x_col], top_df[y_col], s=70, alpha=0.95)

    if not top_df.empty:
        top_df = top_df.sort_values(y_col, ascending=False).reset_index(drop=True)
        for i, row in top_df.iterrows():
            ax.annotate(
                f"{i + 1}",
                (row[x_col], row[y_col]),
                textcoords="offset points",
                xytext=(5, 5),
                fontsize=9,
                fontweight="bold",
            )

    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.grid(True, alpha=0.2)
    st.pyplot(fig, clear_figure=True)


def _movie_card(movie: pd.Series, score_label: str, score_col: str, accent: str = "#6366f1") -> str:
    poster = safe_poster_url(movie.get("poster_url"))
    title = str(movie.get("title", "Sin título"))
    year = int(movie.get("year", 0)) if pd.notna(movie.get("year", 0)) else 0
    vote_average = safe_float(movie.get("vote_average", 0))
    vote_count = safe_int(movie.get("vote_count", 0))
    popularity = safe_float(movie.get("popularity", 0))
    runtime = safe_int(movie.get("runtime", 0))
    score = safe_float(movie.get(score_col, 0))

    return f"""
    <div style="
        height: 410px;
        display:flex;
        flex-direction:column;
        overflow:hidden;
        border-radius:16px;
        border:1px solid rgba(255,255,255,0.08);
        background:#0f172a;
        box-shadow:0 8px 24px rgba(0,0,0,0.35);
        font-family:'Outfit', Arial, sans-serif;">
      <div style="height:68%;position:relative;background:#1e293b;">
        <img src="{poster}" style="width:100%;height:100%;object-fit:cover;display:block;" />
        <div style="position:absolute;top:8px;right:8px;background:rgba(17,24,39,0.9);
                    border:1px solid rgba(255,255,255,0.12);color:#fde68a;
                    font-size:11px;font-weight:800;padding:4px 8px;border-radius:999px;">
          {score_label}: {score:.4f}
        </div>
        <div style="position:absolute;left:0;right:0;bottom:0;height:48%;
                    background:linear-gradient(to top, rgba(10,15,30,0.92), rgba(10,15,30,0.06));"></div>
      </div>
      <div style="padding:10px 10px 12px 10px;display:flex;flex-direction:column;gap:8px;background:#111827;flex:1;">
        <div style="color:#f8fafc;font-size:14px;font-weight:900;line-height:1.15;">{title}</div>
        <div style="color:#94a3b8;font-size:11px;display:flex;gap:6px;flex-wrap:wrap;">
          <span>{year if year else "N/D"}</span>
          <span>•</span>
          <span>{runtime if runtime else "N/D"}m</span>
          <span>•</span>
          <span>★ {vote_average:.1f}</span>
          <span>•</span>
          <span>{vote_count:,} votos</span>
        </div>
        <div style="color:#94a3b8;font-size:11px;">Popularidad: {popularity:.2f}</div>
      </div>
    </div>
    """


def _render_movie_grid(
    df: pd.DataFrame,
    score_col: str,
    title: str,
    accent: str = "#6366f1",
    columns_per_row: int = 3,
):
    _section_title(title, "🎬")

    if df is None or df.empty:
        st.info("No hay datos para mostrar.")
        return

    for start in range(0, len(df), columns_per_row):
        row = df.iloc[start:start + columns_per_row]
        cols = st.columns(len(row), gap="small")
        for i, (_, movie) in enumerate(row.iterrows()):
            with cols[i]:
                st.markdown(
                    _movie_card(movie, score_label=score_col, score_col=score_col, accent=accent),
                    unsafe_allow_html=True,
                )
                tmdb_id = int(movie.get("tmdbId", 0)) if pd.notna(movie.get("tmdbId", 0)) else 0
                if tmdb_id:
                    if st.button("Ver detalle", key=f"{title}_{tmdb_id}", use_container_width=True):
                        go_to_page("moviecard", tmdb_id)


def _clone_without_embeddings(rec):
    """
    Clona el recomendador desactivando la contribución de embeddings.
    Esto se usa solo para comparar el ranking con vs sin embeddings.
    """
    alt = deepcopy(rec)
    alt.config.embedding_candidate_weight = 0.0
    alt.config.embedding_balanced_weight = 0.0
    alt.config.embedding_popular_weight = 0.0
    alt.config.similar_embedding_weight = 0.0
    return alt


def _score_breakdown_recommendations(
    ranked_with_emb: pd.DataFrame,
    profile: dict,
    rec,
    ranked_without_emb: pd.DataFrame | None = None,
) -> pd.DataFrame:
    if ranked_with_emb is None or ranked_with_emb.empty:
        return ranked_with_emb

    out = ranked_with_emb.copy()

    out["genre_term"] = 0.28 * pd.to_numeric(out.get("genre_affinity", 0), errors="coerce").fillna(0)
    out["rating_term"] = 0.28 * pd.to_numeric(out.get("rating_scaled", 0), errors="coerce").fillna(0)
    out["pop_term"] = 0.22 * pd.to_numeric(out.get("pop_scaled", 0), errors="coerce").fillna(0)
    out["votes_term"] = 0.15 * pd.to_numeric(out.get("votes_scaled", 0), errors="coerce").fillna(0)
    out["year_term"] = 0.05 * pd.to_numeric(out.get("year_scaled", 0), errors="coerce").fillna(0)

    out["fav_bonus"] = 0.10 * pd.to_numeric(out.get("fav_genre_hit", 0), errors="coerce").fillna(0)
    if profile and not pd.isna(profile.get("mean_rating", float("nan"))) and profile.get("mean_rating", 0) >= 4.0:
        out["mean_rating_bonus"] = 0.03 * pd.to_numeric(out.get("rating_scaled", 0), errors="coerce").fillna(0)
    else:
        out["mean_rating_bonus"] = 0.0

    out["watch_penalty"] = -1.5 * pd.to_numeric(out.get("watched_penalty", 0), errors="coerce").fillna(0)

    emb_weight = getattr(getattr(rec, "config", None), "embedding_balanced_weight", 0.0) or 0.0
    if "embedding_sim" in out.columns:
        out["embedding_term"] = emb_weight * _minmax(pd.to_numeric(out["embedding_sim"], errors="coerce").fillna(0))
    else:
        out["embedding_term"] = 0.0

    out["reconstructed_score"] = (
        out["genre_term"]
        + out["rating_term"]
        + out["pop_term"]
        + out["votes_term"]
        + out["year_term"]
        + out["fav_bonus"]
        + out["mean_rating_bonus"]
        + out["watch_penalty"]
        + out["embedding_term"]
    )

    out["formula"] = (
        out["genre_term"].round(4).astype(str)
        + " + "
        + out["rating_term"].round(4).astype(str)
        + " + "
        + out["pop_term"].round(4).astype(str)
        + " + "
        + out["votes_term"].round(4).astype(str)
        + " + "
        + out["year_term"].round(4).astype(str)
        + " + "
        + out["fav_bonus"].round(4).astype(str)
        + " + "
        + out["mean_rating_bonus"].round(4).astype(str)
        + " + "
        + out["watch_penalty"].round(4).astype(str)
        + " + "
        + out["embedding_term"].round(4).astype(str)
    )

    if ranked_without_emb is not None and not ranked_without_emb.empty:
        no_emb = ranked_without_emb.copy().reset_index(drop=True)
        no_emb["rank_position_no_emb"] = np.arange(1, len(no_emb) + 1)

        no_emb_cols = ["tmdbId", "rank_position_no_emb", "rank_score"]
        no_emb_cols = [c for c in no_emb_cols if c in no_emb.columns]
        no_emb = no_emb[no_emb_cols].rename(columns={"rank_score": "rank_score_no_emb"})

        out = out.merge(no_emb, on="tmdbId", how="left")
        if "rank_position_with_emb" not in out.columns:
            out["rank_position_with_emb"] = np.arange(1, len(out) + 1)
        out["delta_pos"] = out["rank_position_no_emb"] - out["rank_position_with_emb"]
        out["delta_score"] = out["rank_score"] - out["rank_score_no_emb"]
    else:
        out["rank_position_with_emb"] = np.arange(1, len(out) + 1)
        out["rank_position_no_emb"] = np.nan
        out["rank_score_no_emb"] = np.nan
        out["delta_pos"] = np.nan
        out["delta_score"] = np.nan

    cols = [
        c for c in [
            "tmdbId", "title",
            "rank_position_with_emb", "rank_position_no_emb",
            "rank_score", "rank_score_no_emb",
            "delta_pos", "delta_score",
            "genre_term", "rating_term", "pop_term", "votes_term", "year_term",
            "fav_bonus", "mean_rating_bonus", "watch_penalty", "embedding_term",
            "embedding_sim", "reconstructed_score", "formula",
            "vote_average", "vote_count", "popularity",
            "genres_list", "genre_affinity",
        ]
        if c in out.columns
    ]
    return out[cols].copy()


def _score_breakdown_popular(ranked: pd.DataFrame, profile: dict) -> pd.DataFrame:
    if ranked is None or ranked.empty:
        return ranked

    out = ranked.copy()
    out["genre_term"] = 0.08 * pd.to_numeric(out.get("genre_affinity", 0), errors="coerce").fillna(0)
    out["rating_term"] = 0.10 * pd.to_numeric(out.get("rating_scaled", 0), errors="coerce").fillna(0)
    out["pop_term"] = 0.07 * pd.to_numeric(out.get("pop_scaled", 0), errors="coerce").fillna(0)
    out["votes_term"] = 0.70 * pd.to_numeric(out.get("votes_scaled", 0), errors="coerce").fillna(0)
    out["year_term"] = 0.05 * pd.to_numeric(out.get("year_scaled", 0), errors="coerce").fillna(0)

    out["fav_bonus"] = 0.05 * pd.to_numeric(out.get("fav_genre_hit", 0), errors="coerce").fillna(0)
    if profile and not pd.isna(profile.get("mean_rating", float("nan"))) and profile.get("mean_rating", 0) >= 4.0:
        out["mean_rating_bonus"] = 0.01 * pd.to_numeric(out.get("rating_scaled", 0), errors="coerce").fillna(0)
    else:
        out["mean_rating_bonus"] = 0.0

    out["watch_penalty"] = -1.0 * pd.to_numeric(out.get("watched_penalty", 0), errors="coerce").fillna(0)
    out["reconstructed_score"] = (
        out["genre_term"]
        + out["rating_term"]
        + out["pop_term"]
        + out["votes_term"]
        + out["year_term"]
        + out["fav_bonus"]
        + out["mean_rating_bonus"]
        + out["watch_penalty"]
    )

    out["formula"] = (
        out["genre_term"].round(4).astype(str)
        + " + "
        + out["rating_term"].round(4).astype(str)
        + " + "
        + out["pop_term"].round(4).astype(str)
        + " + "
        + out["votes_term"].round(4).astype(str)
        + " + "
        + out["year_term"].round(4).astype(str)
        + " + "
        + out["fav_bonus"].round(4).astype(str)
        + " + "
        + out["mean_rating_bonus"].round(4).astype(str)
        + " + "
        + out["watch_penalty"].round(4).astype(str)
    )

    cols = [
        c for c in [
            "tmdbId", "title", "genre_term", "rating_term", "pop_term", "votes_term",
            "year_term", "fav_bonus", "mean_rating_bonus", "watch_penalty",
            "reconstructed_score", "popular_score", "formula",
        ]
        if c in out.columns
    ]
    return out[cols].copy()



# Perfil del usuario

def _render_user_profile(rec, user_id: int) -> dict:
    _section_title("Perfil del usuario", "👤")

    user_row = rec.users[rec.users["user_id"] == user_id]
    if user_row.empty:
        st.error(f"user_id={user_id} no existe en users.csv")
        st.stop()

    u = user_row.iloc[0]
    hist = rec.interactions[rec.interactions["user_id"] == user_id] if rec.interactions is not None else pd.DataFrame()

    mean_r = hist["rating"].dropna().mean() if not hist.empty and "rating" in hist.columns else None
    mean_r_text = f"{mean_r:.2f}" if mean_r is not None and pd.notna(mean_r) else "—"

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(
            _metric_card("Usuario", str(u.get("username", "N/D")), f"ID {user_id}", "#6366f1"),
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            _metric_card("Edad", str(u.get("age", "N/D")), str(u.get("gender", "")), "#a855f7"),
            unsafe_allow_html=True,
        )
    with col3:
        st.markdown(
            _metric_card("Interacciones", str(len(hist)), "películas valoradas", "#22c55e"),
            unsafe_allow_html=True,
        )
    with col4:
        st.markdown(
            _metric_card("Rating medio", mean_r_text, "sobre 5", "#f59e0b"),
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    profile = rec.build_user_profile(user_id)
    genre_profile = profile["genre_profile"]

    left, right = st.columns(2)

    with left:
        st.markdown("**Géneros favoritos declarados**")
        fav = u.get("favorite_genres_list", [])
        if not isinstance(fav, list):
            fav = list(fav) if hasattr(fav, "__iter__") else []
        if fav:
            st.markdown(" ".join(_badge(g) for g in fav), unsafe_allow_html=True)
        else:
            st.caption("Sin géneros declarados")

    with right:
        st.markdown("**Distribución de ratings dados**")
        if not hist.empty and "rating" in hist.columns:
            ratings = hist["rating"].dropna()
            if not ratings.empty:
                dist = ratings.value_counts().sort_index()
                chart_data = pd.DataFrame({"rating": dist.index, "count": dist.values})
                st.bar_chart(chart_data.set_index("rating"), height=180, use_container_width=True)
            else:
                st.caption("Sin ratings válidos")
        else:
            st.caption("Sin historial de ratings")

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown("**Genre profile calculado por el modelo**")
    if genre_profile:
        gp_df = pd.DataFrame(
            sorted(genre_profile.items(), key=lambda x: x[1], reverse=True),
            columns=["genre", "weight"],
        ).head(12)

        c1, c2 = st.columns([1.2, 1])
        with c1:
            st.bar_chart(gp_df.set_index("genre"), y="weight", use_container_width=True)
        with c2:
            st.dataframe(
                gp_df.rename(columns={"genre": "Género", "weight": "Peso"}),
                use_container_width=True,
                hide_index=True,
            )
    else:
        st.warning("Genre profile vacío: no hay suficiente historial o favoritos.")

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown("**Películas valoradas por el usuario y a qué géneros pertenecen**")
    rated_df = hist.copy()
    if not rated_df.empty and "movie_id" in rated_df.columns:
        movies_cols = ["tmdbId", "title", "genres", "vote_average", "vote_count", "popularity"]
        movies_cols = [c for c in movies_cols if c in rec.movies.columns] if rec.movies is not None else []

        if rec.movies is not None and movies_cols:
            movies_df = rec.movies[movies_cols].copy()
            rated_df = rated_df.merge(
                movies_df,
                left_on="movie_id",
                right_on="tmdbId",
                how="left",
            )

            rated_df["genres_text"] = rated_df["genres"].apply(parse_genres) if "genres" in rated_df.columns else "Desconocido"

            if "valoration_date" in rated_df.columns:
                rated_df = rated_df.sort_values(["valoration_date", "rating"], ascending=[False, False])
            else:
                rated_df = rated_df.sort_values("rating", ascending=False)

            cols_to_show = []
            for c in ["title", "rating", "genres_text", "vote_average", "vote_count", "popularity", "valoration_date"]:
                if c in rated_df.columns:
                    cols_to_show.append(c)

            display_df = rated_df[cols_to_show].copy()
            rename_map = {
                "title": "Película",
                "rating": "Nota dada",
                "genres_text": "Géneros",
                "vote_average": "TMDB avg",
                "vote_count": "TMDB votos",
                "popularity": "Popularidad",
                "valoration_date": "Fecha",
            }
            display_df = display_df.rename(columns=rename_map)

            st.dataframe(display_df.head(20), use_container_width=True, hide_index=True)

            genre_rows = rated_df[["rating", "genres"]].dropna().copy()
            genre_rows["genres_list"] = genre_rows["genres"].apply(normalize_genres_list)
            genre_rows = genre_rows.explode("genres_list")
            genre_rows["genres_list"] = genre_rows["genres_list"].astype(str).str.strip().str.lower()
            genre_rows = genre_rows[
                genre_rows["genres_list"].ne("")
                & genre_rows["genres_list"].ne("nan")
                & genre_rows["genres_list"].ne("none")
                & genre_rows["genres_list"].ne("null")
            ]

            if not genre_rows.empty:
                genre_summary = (
                    genre_rows.groupby("genres_list")
                    .agg(
                        mean_rating=("rating", "mean"),
                        n_ratings=("rating", "count"),
                    )
                    .reset_index()
                    .sort_values(["mean_rating", "n_ratings"], ascending=False)
                )

                c1, c2 = st.columns(2)
                with c1:
                    st.markdown("**Nota media por género**")
                    st.bar_chart(
                        genre_summary.set_index("genres_list")[["mean_rating"]],
                        use_container_width=True,
                        height=260,
                    )

                with c2:
                    st.markdown("**Cuántas veces ha puntuado cada género**")
                    st.bar_chart(
                        genre_summary.set_index("genres_list")[["n_ratings"]],
                        use_container_width=True,
                        height=260,
                    )

                st.markdown("**Detalle agrupado por género**")
                st.dataframe(
                    genre_summary.rename(columns={
                        "genres_list": "Género",
                        "mean_rating": "Nota media",
                        "n_ratings": "Nº valoraciones",
                    }),
                    use_container_width=True,
                    hide_index=True,
                )
            else:
                st.caption("No hay géneros suficientes para agrupar el historial.")
        else:
            st.caption("No se pudo acceder a la tabla de películas para cruzar géneros.")
    else:
        st.info("Este usuario todavía no tiene valoraciones registradas.")

    return profile



# Recomendaciones

def _render_recommendations(rec, user_id: int, profile: dict) -> pd.DataFrame:
    _section_title("Recomendaciones del modelo", "🏆")

    candidates = rec.generate_candidates(user_id=user_id)
    ranked_with_emb = rec.rank_candidates_balanced(user_id=user_id, candidates=candidates.copy())
    ranked_without_emb = _clone_without_embeddings(rec).rank_candidates_balanced(
        user_id=user_id,
        candidates=candidates.copy(),
    )

    ranked_with_emb = normalize_movie_df(ranked_with_emb)
    ranked_without_emb = normalize_movie_df(ranked_without_emb)

    top_12 = ranked_with_emb.head(12).copy()

    if ranked_with_emb.empty:
        st.warning("El recomendador no devolvió resultados.")
        return ranked_with_emb

    top_12_no_emb = ranked_without_emb.head(12).copy() if ranked_without_emb is not None else pd.DataFrame()

    compare_df = _score_breakdown_recommendations(
        top_12,
        profile,
        rec,
        ranked_without_emb=ranked_without_emb,
    )

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(
            _metric_card("Recomendaciones", str(len(top_12)), "Top 12 del modelo", "#6366f1"),
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            _metric_card("Score medio", f"{top_12['rank_score'].mean():.4f}", "rank_score", "#a855f7"),
            unsafe_allow_html=True,
        )
    with col3:
        st.markdown(
            _metric_card("Candidatos", f"{len(ranked_with_emb):,}", "pool total", "#22c55e"),
            unsafe_allow_html=True,
        )
    with col4:
        overlap = len(set(top_12["tmdbId"].tolist()) & set(top_12_no_emb["tmdbId"].tolist())) if not top_12_no_emb.empty else 0
        st.markdown(
            _metric_card("Solapamiento", f"{overlap}/12", "con vs sin embeddings", "#f59e0b"),
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    st.info(
        """
        El score final combina varias señales:
        afinidad de géneros, calidad media de TMDB, popularidad, número de votos,
        año, preferencias del usuario y penalización por películas ya vistas.

        Los embeddings añaden una señal semántica: comparan el perfil del usuario
        con la representación textual de cada película (overview + géneros) y
        empujan hacia arriba títulos que son parecidos en significado, aunque no
        coincidan exactamente en popularidad o género.
        """
    )

    with st.expander("Ver cómo se construye la puntuación", expanded=True):
        st.markdown(
            f"""
            **Con embeddings**
            - `genre_term` → afinidad por géneros
            - `rating_term` → calidad TMDB
            - `pop_term` → popularidad
            - `votes_term` → número de votos
            - `year_term` → cercanía temporal
            - `fav_bonus` → bonus por géneros favoritos declarados
            - `mean_rating_bonus` → bonus si el usuario tiende a valorar alto
            - `watch_penalty` → penalización por películas ya vistas
            - `embedding_term` → similitud semántica, ponderada por `embedding_balanced_weight = {getattr(rec.config, 'embedding_balanced_weight', 0.0):.3f}`

            **Sin embeddings**
            - se usa exactamente el mismo pool de candidatos
            - pero con el peso de embeddings a 0
            - así puedes ver qué películas suben o bajan solo por la señal semántica
            """
        )

    st.markdown("**Plot del pool completo de candidatos**")
    st.caption("Puntos grises = todas las películas del grupo. Puntos resaltados = Top 12 recomendadas.")
    _plot_group(ranked_with_emb, top_12, x_col="vote_count", y_col="rank_score", title="Recomendaciones: vote_count vs rank_score")

    st.markdown("**Desglose del score por película**")
    breakdown = compare_df.copy()

    if not breakdown.empty:
        breakdown = breakdown.sort_values("rank_score", ascending=False).reset_index(drop=True)

        display_cols = [
            c for c in [
                "title",
                "rank_position_with_emb",
                "rank_position_no_emb",
                "delta_pos",
                "rank_score",
                "rank_score_no_emb",
                "delta_score",
                "embedding_sim",
                "embedding_term",
                "genre_term",
                "rating_term",
                "pop_term",
                "votes_term",
                "year_term",
                "fav_bonus",
                "mean_rating_bonus",
                "watch_penalty",
                "formula",
            ]
            if c in breakdown.columns
        ]

        st.dataframe(
            breakdown[display_cols].rename(columns={
                "title": "Título",
                "rank_position_with_emb": "Pos. con emb.",
                "rank_position_no_emb": "Pos. sin emb.",
                "delta_pos": "Cambio pos.",
                "rank_score": "Score con emb.",
                "rank_score_no_emb": "Score sin emb.",
                "delta_score": "Δ score",
                "embedding_sim": "Emb. sim.",
                "embedding_term": "Término emb.",
                "genre_term": "Género",
                "rating_term": "Rating",
                "pop_term": "Popularidad",
                "votes_term": "Votos",
                "year_term": "Año",
                "fav_bonus": "Bonus fav.",
                "mean_rating_bonus": "Bonus rating medio",
                "watch_penalty": "Penalización vistas",
                "formula": "Fórmula",
            }),
            use_container_width=True,
            hide_index=True,
        )

        st.markdown("**Cómo impactan los embeddings en el score**")
        fig, ax = plt.subplots(figsize=(11, 5))
        ax.scatter(
            breakdown["embedding_term"] if "embedding_term" in breakdown.columns else np.zeros(len(breakdown)),
            breakdown["delta_score"] if "delta_score" in breakdown.columns else np.zeros(len(breakdown)),
            alpha=0.75,
        )
        ax.axhline(0, linestyle="--", alpha=0.4)
        ax.set_xlabel("Término de embeddings")
        ax.set_ylabel("Δ score (con emb - sin emb)")
        ax.set_title("Relación entre similitud semántica y cambio real en la puntuación")
        ax.grid(True, alpha=0.2)
        st.pyplot(fig, clear_figure=True)
    else:
        st.info("No hay datos suficientes para mostrar el breakdown.")

    st.markdown("**Vista tabular de las 12 recomendadas**")
    show_cols = [c for c in ["title", "vote_average", "vote_count", "popularity", "genre_affinity", "rank_score"] if c in top_12.columns]
    st.dataframe(top_12[show_cols], use_container_width=True, hide_index=True)

    _render_movie_grid(
        top_12.sort_values("rank_score", ascending=False).reset_index(drop=True),
        "rank_score",
        "Top 12 Recomendaciones",
        accent="#6366f1",
        columns_per_row=3,
    )

    return ranked_with_emb



# Populares

def _render_popular(rec, user_id: int, profile: dict) -> pd.DataFrame:
    _section_title("Películas populares", "📈")

    candidates = rec.generate_candidates(user_id=user_id)
    ranked = rec.rank_candidates_popular(user_id=user_id, candidates=candidates.copy())
    top_12 = ranked.head(12).copy()

    if ranked.empty:
        st.warning("No se pudieron generar populares.")
        return ranked

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(
            _metric_card("Populares", str(len(top_12)), "Top 12 por popularidad", "#f59e0b"),
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            _metric_card("Score medio", f"{top_12['popular_score'].mean():.4f}", "popular_score", "#f97316"),
            unsafe_allow_html=True,
        )
    with col3:
        st.markdown(
            _metric_card("Votos medios", f"{top_12['vote_count'].mean():.0f}", "vote_count", "#22c55e"),
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown("**Plot del pool completo de candidatos**")
    st.caption("Puntos grises = todas las películas del grupo. Puntos resaltados = Top 12 populares.")
    _plot_group(ranked, top_12, x_col="vote_count", y_col="popular_score", title="Populares: vote_count vs popular_score")

    breakdown = _score_breakdown_popular(top_12, profile)
    st.markdown("**Desglose del score por película**")
    st.dataframe(
        breakdown.rename(columns={
            "tmdbId": "tmdbId",
            "title": "Título",
            "genre_term": "Género",
            "rating_term": "Rating",
            "pop_term": "Popularidad",
            "votes_term": "Votos",
            "year_term": "Año",
            "fav_bonus": "Bonus fav.",
            "mean_rating_bonus": "Bonus rating medio",
            "watch_penalty": "Penalización vistas",
            "reconstructed_score": "Score reconstruido",
            "popular_score": "Popular score",
            "formula": "Fórmula",
        }),
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("**Vista tabular de las 12 populares**")
    show_cols = [c for c in ["title", "vote_average", "vote_count", "popularity", "genre_affinity", "popular_score"] if c in top_12.columns]
    st.dataframe(top_12[show_cols], use_container_width=True, hide_index=True)

    _render_movie_grid(
        top_12.sort_values("popular_score", ascending=False).reset_index(drop=True),
        "popular_score",
        "Top 12 Populares",
        accent="#f59e0b",
        columns_per_row=3,
    )

    return ranked



# Entry point

def show_audit() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700;800;900&display=swap');

        .stApp {
            background: #060d1a;
            font-family: 'Outfit', sans-serif;
        }

        #MainMenu, footer, header { visibility: hidden; }

        .main .block-container {
            max-width: 1480px !important;
            padding: 1.5rem 2vw 3rem !important;
        }

        div[data-testid="stSelectbox"] > div,
        div[data-testid="stNumberInput"] > div {
            background: #0f172a !important;
            border: 1px solid #1e293b !important;
            border-radius: 10px !important;
        }

        .audit-topbar {
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 14px 20px;
            margin-bottom: 24px;
            background: #0a1020;
            border: 1px solid #1e293b;
            border-radius: 16px;
        }

        .audit-topbar-title {
            font-size: 20px;
            font-weight: 900;
            color: #f8fafc;
            letter-spacing: -.02em;
        }

        .audit-topbar-sub {
            font-size: 12px;
            color: #64748b;
            margin-top: 2px;
        }

        div[data-testid="stTabs"] button {
            font-family: 'Outfit', sans-serif !important;
            font-weight: 800 !important;
            font-size: 13px !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="audit-topbar">
          <div>
            <div class="audit-topbar-title">🔬 Auditoría del Recomendador</div>
            <div class="audit-topbar-sub">Inspección del usuario, del ranking y de la lógica popular</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    users_mtime = USERS_PATH.stat().st_mtime
    interactions_mtime = INTERACTIONS_PATH.stat().st_mtime
    rec = get_recommender(users_mtime, interactions_mtime)

    if rec.users is None or rec.users.empty:
        st.error("No hay usuarios cargados en el recomendador.")
        return

    available_users = sorted(rec.users["user_id"].tolist())
    user_labels = {}
    for uid in available_users:
        row = rec.users[rec.users["user_id"] == uid].iloc[0]
        uname = row.get("username", f"user_{uid}")
        n_inter = len(rec.interactions[rec.interactions["user_id"] == uid]) if rec.interactions is not None else 0
        user_labels[uid] = f"{uname} (id={uid}, {n_inter} interacc.)"

    with st.sidebar:
        st.markdown("### ⚙️ Configuración")
        user_id = st.selectbox(
            "Usuario a auditar",
            options=available_users,
            format_func=lambda uid: user_labels.get(uid, str(uid)),
            index=0,
        )

        st.markdown("---")
        if st.button("🔄 Reconstruir caché", use_container_width=True):
            st.cache_resource.clear()
            st.rerun()

        if st.button("← Volver al dashboard", use_container_width=True):
            go_to_page("dashboard")

    tabs = st.tabs(
        [
            "👤 Perfil",
            "🏆 Recomendaciones",
            "📈 Populares",
        ]
    )

    with st.spinner("Calculando datos de auditoría..."):
        profile = rec.build_user_profile(user_id)
        _ = normalize_movie_df(rec.recommend_with_explanations(user_id=user_id, n=12))
        _ = normalize_movie_df(rec.recommend_popular_with_explanations(user_id=user_id, n=12))

        if "recommender_ready_time" not in st.session_state:
            st.session_state.recommender_ready_time = time.time()

    login_time  = st.session_state.get("login_time")
    ready_time  = st.session_state.get("recommender_ready_time")

    if login_time and ready_time:
        elapsed = ready_time - login_time
        elapsed_str = f"{elapsed:.1f} s" if elapsed < 60 else f"{int(elapsed)//60}m {int(elapsed)%60}s"
        st.markdown(
            f"""
            <div style="display:inline-flex;align-items:center;gap:10px;
                        background:#0f172a;border:1px solid #1e293b;
                        border-radius:12px;padding:10px 20px;margin-bottom:16px;">
              <div>
                <div style="color:#64748b;font-size:11px;font-weight:700;
                            text-transform:uppercase;letter-spacing:.08em;">
                  Login → recomendador listo
                </div>
                <div style="color:#fde68a;font-size:22px;font-weight:900;">{elapsed_str}</div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with tabs[0]:
        _render_user_profile(rec, user_id)

    with tabs[1]:
        _render_recommendations(rec, user_id, profile)

    with tabs[2]:
        _render_popular(rec, user_id, profile)