
from __future__ import annotations

import ast
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set

import numpy as np
import pandas as pd


# Config
@dataclass
class RecommenderPaths:
    embeddings_parquet: Path
    users_csv: Path
    interactions_csv: Path
    cache_dir: Path
    movies_csv: Optional[Path] = None
    embeddings_csv: Optional[Path] = None


@dataclass
class RecommenderConfig:
    candidate_k: int = 400
    recommendations_n: int = 12

    # Ratings thresholds
    min_positive_rating: float = 3.5
    min_negative_rating: float = 2.5

    user_history_limit: int = 200
    bayesian_m: int = 500
    meta_name: str = "movie_meta_v3_emb.parquet"

    # Embedding weights
    embedding_candidate_weight: float = 0.30
    embedding_balanced_weight: float = 0.15
    embedding_popular_weight: float = 0.05

    # How strongly negative feedback repels the profile embedding.
    negative_embedding_weight: float = 0.85

    # Minimum number of samples required to trust each side of the profile.
    embedding_min_positive: int = 2
    embedding_min_negative: int = 2

    # How much negative genre evidence subtracts from the genre profile.
    negative_genre_weight: float = 1.0

    # Similar mode tuning
    similar_genre_weight: float = 0.38
    similar_shared_genre_weight: float = 0.10
    similar_embedding_weight: float = 0.28
    similar_title_weight: float = 0.05
    similar_year_weight: float = 0.08
    similar_runtime_weight: float = 0.05
    similar_quality_weight: float = 0.04
    similar_popularity_weight: float = 0.01
    similar_votes_weight: float = 0.01
    similar_exact_genre_boost: float = 0.06
    similar_exact_title_boost: float = 0.03


# Embedding index
@dataclass
class EmbeddingIndex:
    matrix: np.ndarray
    id_to_idx: Dict[int, int]

    def get_vector(self, tmdb_id: int) -> Optional[np.ndarray]:
        idx = self.id_to_idx.get(int(tmdb_id))
        if idx is None:
            return None
        return self.matrix[idx]

    def cosine_similarity_to_vector(
        self,
        query_vec: np.ndarray,
        tmdb_ids: Sequence[int],
    ) -> np.ndarray:
        indices = [self.id_to_idx.get(int(tid)) for tid in tmdb_ids]
        sims = np.zeros(len(tmdb_ids), dtype="float32")
        for i, idx in enumerate(indices):
            if idx is not None:
                sims[i] = float(np.dot(query_vec, self.matrix[idx]))
        return sims

    @classmethod
    def build(cls, df: pd.DataFrame, id_col: str = "tmdbId") -> "EmbeddingIndex":
        valid = df[df["embedding"].notna()].copy()
        if valid.empty:
            return cls(matrix=np.zeros((0, 1), dtype="float32"), id_to_idx={})

        vecs = np.array([np.asarray(v, dtype="float32") for v in valid["embedding"]])
        ids = valid[id_col].astype("int64").tolist()

        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        vecs = (vecs / norms).astype("float32")

        id_to_idx = {int(tid): i for i, tid in enumerate(ids)}
        return cls(matrix=vecs, id_to_idx=id_to_idx)


# Generic utils

def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _safe_str(x, default: str = "") -> str:
    if x is None:
        return default
    try:
        if pd.isna(x):
            return default
    except Exception:
        pass
    return str(x)


def _parse_genres(value) -> List[str]:
    if value is None:
        return []
    if isinstance(value, (list, np.ndarray)):
        out: List[str] = []
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

    text = _safe_str(value).strip()
    if not text:
        return []
    try:
        parsed = ast.literal_eval(text)
        return _parse_genres(parsed)
    except Exception:
        pass

    text = text.replace("|", ",").replace("[", "").replace("]", "")
    text = text.replace("{", "").replace("}", "")
    parts = [p.strip().strip("'").strip('"') for p in text.split(",")]
    return [p for p in parts if p]


def _parse_favorite_genres(value) -> List[str]:
    return _parse_genres(value)


def _to_numeric(series: pd.Series, default: float = 0.0) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").fillna(default)


def _safe_year(release_date: pd.Series) -> pd.Series:
    return pd.to_datetime(release_date, errors="coerce").dt.year.fillna(0).astype(int)


def _minmax(series: pd.Series) -> pd.Series:
    if series.empty:
        return series
    mn = series.min()
    mx = series.max()
    if pd.isna(mn) or pd.isna(mx) or mx == mn:
        return pd.Series([0.0] * len(series), index=series.index, dtype="float32")
    return ((series - mn) / (mx - mn)).astype("float32")


def _to_list(value) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    try:
        return [str(item) for item in value if item is not None]
    except TypeError:
        return []


def _normalize_text(text: object) -> str:
    raw = _safe_str(text).lower().strip()
    if not raw:
        return ""
    raw = re.sub(r"[^\w\s]", " ", raw, flags=re.UNICODE)
    raw = re.sub(r"\s+", " ", raw, flags=re.UNICODE).strip()
    return raw


_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "for",
    "from", "has", "he", "in", "into", "is", "it", "its", "of", "on",
    "or", "s", "she", "that", "the", "their", "them", "then", "there",
    "this", "to", "was", "were", "will", "with", "you", "your", "i",
    "we", "they", "his", "her", "had", "have", "not", "than", "who",
    "what", "when", "where", "why", "how", "which", "do", "does", "did",
    "over", "under", "out", "up", "down", "after", "before", "into", "about",
    "movie", "film", "story", "one", "two", "three", "new", "old", "life"
}


def _tokenize_text(text: object, min_len: int = 3) -> Set[str]:
    normalized = _normalize_text(text)
    if not normalized:
        return set()
    return {
        tok
        for tok in normalized.split()
        if len(tok) >= min_len and tok not in _STOPWORDS and not tok.isdigit()
    }


def _jaccard_from_sets(a: Set[str], b: Set[str]) -> float:
    if not a or not b:
        return 0.0
    union = a | b
    return len(a & b) / len(union) if union else 0.0


def _overlap_coefficient(a: Set[str], b: Set[str]) -> float:
    if not a or not b:
        return 0.0
    denom = min(len(a), len(b))
    return len(a & b) / denom if denom > 0 else 0.0


def _closeness_score(x: float, y: float, scale: float) -> float:
    if scale <= 0:
        return 0.0
    return float(math.exp(-abs(float(x) - float(y)) / scale))


def _l2_normalise(vec: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 0 else vec


def _weighted_genre_add(
    genre_profile: Dict[str, float],
    genres: Sequence[str],
    weight: float,
) -> None:
    for g in genres:
        key = str(g).strip().lower()
        if key:
            genre_profile[key] = genre_profile.get(key, 0.0) + weight



# Main recommender

class MovieRecommender:
    """
    Four-mode movie recommender backed by pre-computed text+genre embeddings.

    MODE 1  Balanced
    MODE 2  Popular
    MODE 3  Random
    MODE 4  Similar
    """

    def __init__(self, paths: RecommenderPaths, config: Optional[RecommenderConfig] = None):
        self.paths = paths
        self.config = config or RecommenderConfig()

        _ensure_dir(self.paths.cache_dir)

        self.movies: Optional[pd.DataFrame] = None
        self.users: Optional[pd.DataFrame] = None
        self.interactions: Optional[pd.DataFrame] = None
        self.meta: Optional[pd.DataFrame] = None
        self.emb_index: Optional[EmbeddingIndex] = None

    # -----------------------------------------------------------------------
    # Data loading
    # -----------------------------------------------------------------------
    def load_static_tables(self) -> None:
        parquet_path = self.paths.embeddings_parquet
        self.movies = pd.read_parquet(parquet_path)

        self.movies["tmdbId"] = pd.to_numeric(self.movies["tmdbId"], errors="coerce").astype("Int64")
        self.movies = self.movies[self.movies["tmdbId"].notna()].copy()
        self.movies["tmdbId"] = self.movies["tmdbId"].astype("int64")

        for col in ["vote_average", "vote_count", "popularity", "runtime"]:
            if col in self.movies.columns:
                self.movies[col] = _to_numeric(self.movies[col], 0.0)

        self.movies["year"] = (
            _safe_year(self.movies["release_date"]) if "release_date" in self.movies.columns else 0
        )
        self.movies["genres_list"] = (
            self.movies["genres"].apply(_parse_genres)
            if "genres" in self.movies.columns
            else [[] for _ in range(len(self.movies))]
        )

        if "embedding" in self.movies.columns:
            self.emb_index = EmbeddingIndex.build(self.movies, id_col="tmdbId")
        else:
            self.emb_index = None

        user_cols = [
            "user_id", "username", "email", "password",
            "gender", "age", "favorite_genres", "created_at",
        ]
        self.users = pd.read_csv(
            self.paths.users_csv,
            usecols=lambda c: c in user_cols,
            low_memory=False,
        )
        self.users["user_id"] = pd.to_numeric(self.users["user_id"], errors="coerce").astype("Int64")
        self.users = self.users[self.users["user_id"].notna()].copy()
        self.users["user_id"] = self.users["user_id"].astype("int64")
        self.users["favorite_genres_list"] = (
            self.users["favorite_genres"].apply(_parse_favorite_genres)
            if "favorite_genres" in self.users.columns
            else [[] for _ in range(len(self.users))]
        )

        interaction_cols = [
            "interaction_id", "user_id", "movie_id", "rating", "valoration_date",
        ]
        self.interactions = pd.read_csv(
            self.paths.interactions_csv,
            usecols=lambda c: c in interaction_cols,
            low_memory=False,
        )
        self.interactions["user_id"] = pd.to_numeric(self.interactions["user_id"], errors="coerce").astype("Int64")
        self.interactions["movie_id"] = pd.to_numeric(self.interactions["movie_id"], errors="coerce").astype("Int64")
        self.interactions["rating"] = _to_numeric(self.interactions["rating"], np.nan)
        self.interactions = self.interactions[
            self.interactions["user_id"].notna() & self.interactions["movie_id"].notna()
        ].copy()
        self.interactions["user_id"] = self.interactions["user_id"].astype("int64")
        self.interactions["movie_id"] = self.interactions["movie_id"].astype("int64")

    def _bayesian_weighted_rating(self, df: pd.DataFrame) -> pd.Series:
        v = df["vote_count"].fillna(0.0).astype(float)
        R = df["vote_average"].fillna(0.0).astype(float)

        valid = df.loc[df["vote_count"].fillna(0) > 0, "vote_average"]
        C = float(valid.mean()) if not valid.empty else float(R.mean())
        m = float(self.config.bayesian_m)

        confidence = v / (v + m)
        weighted_rating = confidence * R + (1.0 - confidence) * C
        return weighted_rating

    # -----------------------------------------------------------------------
    # Cache
    # -----------------------------------------------------------------------
    def _cache_file(self) -> Path:
        return self.paths.cache_dir / self.config.meta_name

    def build_cache(self, force: bool = False) -> None:
        cache_file = self._cache_file()
        if not force and cache_file.exists():
            return

        if self.movies is None:
            self.load_static_tables()

        meta = self.movies.drop(columns=["embedding"], errors="ignore").copy()

        meta["pop_scaled"] = _minmax(meta["popularity"].fillna(0.0))
        meta["votes_scaled"] = _minmax(np.log1p(meta["vote_count"].fillna(0.0)))
        meta["rating_scaled"] = _minmax(meta["vote_average"].fillna(0.0))
        meta["year_scaled"] = _minmax(meta["year"].fillna(0))

        meta["vote_confidence"] = (
            meta["vote_count"].fillna(0.0).astype(float)
            / (meta["vote_count"].fillna(0.0).astype(float) + float(self.config.bayesian_m))
        ).clip(0.0, 1.0)

        meta["weighted_rating"] = self._bayesian_weighted_rating(meta)
        meta["quality_score_raw"] = meta["weighted_rating"].fillna(0.0) * meta["vote_confidence"].fillna(0.0)
        meta["quality_score"] = _minmax(meta["quality_score_raw"])

        meta["catalogue_score"] = (
            0.65 * meta["quality_score"].fillna(0.0)
            + 0.20 * meta["pop_scaled"].fillna(0.0)
            + 0.15 * meta["year_scaled"].fillna(0.0)
        )

        meta = meta.sort_values(["catalogue_score", "vote_count", "popularity"], ascending=False).reset_index(drop=True)
        meta.to_parquet(cache_file, index=False)
        self.meta = meta

    _REQUIRED_CACHE_COLS = {
        "tmdbId",
        "genres_list",
        "catalogue_score",
        "popularity",
        "vote_count",
        "weighted_rating",
        "vote_confidence",
        "quality_score",
    }

    def load_cache(self) -> None:
        cache_file = self._cache_file()
        self.meta = pd.read_parquet(cache_file)
        self.meta["tmdbId"] = self.meta["tmdbId"].astype("int64")

        if "genres_list" in self.meta.columns:
            self.meta["genres_list"] = self.meta["genres_list"].apply(_to_list)

        for col in [
            "vote_average", "vote_count", "popularity", "runtime", "year",
            "pop_scaled", "votes_scaled", "rating_scaled", "year_scaled",
            "vote_confidence", "weighted_rating", "quality_score_raw",
            "quality_score", "catalogue_score",
        ]:
            if col in self.meta.columns:
                self.meta[col] = pd.to_numeric(self.meta[col], errors="coerce").fillna(0.0)

    def _cache_is_valid(self) -> bool:
        cache_file = self._cache_file()
        if not cache_file.exists():
            return False
        try:
            import pyarrow.parquet as pq
            schema_cols = set(pq.read_schema(cache_file).names)
        except Exception:
            try:
                schema_cols = set(pd.read_parquet(cache_file, columns=[]).columns)
            except Exception:
                return False
        return self._REQUIRED_CACHE_COLS.issubset(schema_cols)

    def fit(self, force_rebuild_cache: bool = False) -> None:
        self.load_static_tables()
        if force_rebuild_cache or not self._cache_is_valid():
            self.build_cache(force=True)
        else:
            self.load_cache()

    # -----------------------------------------------------------------------
    # Embedding helpers
    # -----------------------------------------------------------------------
    def _build_user_embedding(
        self,
        positive_ids: Set[int],
        negative_ids: Optional[Set[int]] = None,
    ) -> Optional[np.ndarray]:
        if self.emb_index is None or len(self.emb_index.id_to_idx) == 0:
            return None

        positive_ids = set(positive_ids or set())
        negative_ids = set(negative_ids or set())

        pos_vecs: List[np.ndarray] = []
        neg_vecs: List[np.ndarray] = []

        for tid in positive_ids:
            v = self.emb_index.get_vector(tid)
            if v is not None:
                pos_vecs.append(v)

        for tid in negative_ids:
            v = self.emb_index.get_vector(tid)
            if v is not None:
                neg_vecs.append(v)

        if len(pos_vecs) < self.config.embedding_min_positive and len(neg_vecs) < self.config.embedding_min_negative:
            return None

        components: List[np.ndarray] = []

        if len(pos_vecs) >= self.config.embedding_min_positive:
            pos_mean = np.mean(np.stack(pos_vecs, axis=0), axis=0)
            components.append(pos_mean.astype("float32"))

        if len(neg_vecs) >= self.config.embedding_min_negative:
            neg_mean = np.mean(np.stack(neg_vecs, axis=0), axis=0)
            components.append((-self.config.negative_embedding_weight * neg_mean).astype("float32"))

        if not components:
            return None

        profile_vec = np.sum(np.stack(components, axis=0), axis=0)
        return _l2_normalise(profile_vec).astype("float32")

    def _embedding_similarity_column(
        self,
        df: pd.DataFrame,
        query_vec: np.ndarray,
        col_name: str = "embedding_sim",
    ) -> pd.Series:
        sims = self.emb_index.cosine_similarity_to_vector(query_vec, df["tmdbId"].tolist())
        return pd.Series(sims, index=df.index, dtype="float32", name=col_name)

    # -----------------------------------------------------------------------
    # User profile
    # -----------------------------------------------------------------------
    def _user_row(self, user_id: int) -> pd.Series:
        if self.users is None:
            raise RuntimeError("Usuarios no cargados.")
        row = self.users[self.users["user_id"] == user_id]
        if row.empty:
            raise KeyError(f"Usuario {user_id} no encontrado.")
        return row.iloc[0]

    def _user_history(self, user_id: int) -> pd.DataFrame:
        if self.interactions is None:
            raise RuntimeError("Interacciones no cargadas.")
        hist = self.interactions[self.interactions["user_id"] == user_id].copy()
        if hist.empty:
            return hist
        if "valoration_date" in hist.columns:
            hist = hist.sort_values("valoration_date", ascending=False)
        return hist.head(self.config.user_history_limit).copy()

    def build_user_profile(self, user_id: int) -> Dict[str, object]:
        if self.meta is None:
            raise RuntimeError("Primero llama a fit().")

        user = self._user_row(user_id)
        hist = self._user_history(user_id)

        fav_genres: List[str] = _to_list(user.get("favorite_genres_list", []))
        positive_genre_profile: Dict[str, float] = {}
        negative_genre_profile: Dict[str, float] = {}

        if hist.empty:
            mean_rating = float("nan")
            watched_ids: Set[int] = set()
            positive_ids: Set[int] = set()
            negative_ids: Set[int] = set()
        else:
            watched_ids = set(hist["movie_id"].astype("int64").tolist())
            merged = hist.merge(
                self.meta[["tmdbId", "genres_list"]],
                left_on="movie_id",
                right_on="tmdbId",
                how="inner",
            ).copy()

            rated = merged["rating"].fillna(0) if "rating" in merged.columns else pd.Series([0] * len(merged), index=merged.index)

            positives = merged[rated >= self.config.min_positive_rating].copy()
            negatives = merged[rated <= self.config.min_negative_rating].copy()

            # Fallbacks keep the profile usable if the user has only one side.
            if positives.empty and not merged.empty:
                positives = merged.copy()
            if negatives.empty and not merged.empty:
                negatives = merged.iloc[0:0].copy()

            if not positives.empty and "genres_list" in positives.columns:
                for _, row in positives.iterrows():
                    weight = 1.0
                    if "rating" in row and pd.notna(row["rating"]):
                        weight = max(0.5, min(2.0, (float(row["rating"]) - 2.5) / 2.0))
                    _weighted_genre_add(positive_genre_profile, _to_list(row["genres_list"]), weight)

            if not negatives.empty and "genres_list" in negatives.columns:
                for _, row in negatives.iterrows():
                    weight = 1.0
                    if "rating" in row and pd.notna(row["rating"]):
                        # stronger penalty for lower ratings
                        weight = max(0.5, min(2.5, (3.0 - float(row["rating"])) / 1.5))
                    _weighted_genre_add(negative_genre_profile, _to_list(row["genres_list"]), weight)

            mean_rating = (
                float(hist["rating"].dropna().mean())
                if "rating" in hist.columns and hist["rating"].notna().any()
                else float("nan")
            )

            positive_ids = (
                set(
                    hist.loc[hist["rating"].fillna(0) >= self.config.min_positive_rating, "movie_id"]
                    .astype("int64")
                    .tolist()
                )
                if "rating" in hist.columns
                else watched_ids
            )

            negative_ids = (
                set(
                    hist.loc[hist["rating"].fillna(0) <= self.config.min_negative_rating, "movie_id"]
                    .astype("int64")
                    .tolist()
                )
                if "rating" in hist.columns
                else set()
            )

        # Explicit favorites still matter, but positive/negative history is the primary signal.
        for g in fav_genres:
            key = str(g).strip().lower()
            if key:
                positive_genre_profile[key] = positive_genre_profile.get(key, 0.0) + 1.5

        if positive_genre_profile:
            total = sum(positive_genre_profile.values())
            if total > 0:
                positive_genre_profile = {k: v / total for k, v in positive_genre_profile.items()}

        if negative_genre_profile:
            total_neg = sum(negative_genre_profile.values())
            if total_neg > 0:
                negative_genre_profile = {k: v / total_neg for k, v in negative_genre_profile.items()}

        user_embedding = self._build_user_embedding(positive_ids, negative_ids)

        return {
            "watched_ids": watched_ids,
            "positive_ids": positive_ids,
            "negative_ids": negative_ids,
            "favorite_genres": fav_genres,
            "mean_rating": mean_rating,
            "genre_profile": positive_genre_profile,
            "negative_genre_profile": negative_genre_profile,
            "user_embedding": user_embedding,
        }

    # -----------------------------------------------------------------------
    # Shared helper: genre affinity
    # -----------------------------------------------------------------------
    def _genre_overlap(
        self,
        movie_genres,
        favorite_genres: Sequence[str],
        genre_profile: Dict[str, float],
        negative_genre_profile: Optional[Dict[str, float]] = None,
    ) -> float:
        genres_list = _to_list(movie_genres)
        if not genres_list:
            return 0.0

        movie_genres_norm = {str(g).strip().lower() for g in genres_list if str(g).strip()}
        overlap = 0.0

        if favorite_genres:
            fav = {str(g).strip().lower() for g in favorite_genres if str(g).strip()}
            if fav:
                overlap += len(movie_genres_norm & fav) / max(len(fav), 1)

        if genre_profile:
            overlap += sum(genre_profile.get(g, 0.0) for g in movie_genres_norm)

        if negative_genre_profile:
            overlap -= self.config.negative_genre_weight * sum(
                negative_genre_profile.get(g, 0.0) for g in movie_genres_norm
            )

        return float(overlap)

    # -----------------------------------------------------------------------
    # MODE 4 helper
    # -----------------------------------------------------------------------
    def _movie_similarity_to_reference(
        self,
        movie_row: pd.Series,
        reference_row: pd.Series,
    ) -> Dict[str, float]:
        candidate_genres = {
            str(g).strip().lower()
            for g in _to_list(movie_row.get("genres_list", []))
            if str(g).strip()
        }
        reference_genres = {
            str(g).strip().lower()
            for g in _to_list(reference_row.get("genres_list", []))
            if str(g).strip()
        }

        genre_jaccard = _jaccard_from_sets(candidate_genres, reference_genres)
        genre_overlap = _overlap_coefficient(candidate_genres, reference_genres)
        shared_genres = len(candidate_genres & reference_genres)
        exact_genre_hit = 1.0 if candidate_genres and candidate_genres == reference_genres else 0.0

        cand_title_tok = _tokenize_text(movie_row.get("title", ""))
        ref_title_tok = _tokenize_text(reference_row.get("title", ""))
        title_jaccard = _jaccard_from_sets(cand_title_tok, ref_title_tok)
        exact_title_hit = 1.0 if (
            _normalize_text(movie_row.get("title", "")) == _normalize_text(reference_row.get("title", ""))
        ) else 0.0

        cand_year = float(movie_row.get("year", 0) or 0)
        ref_year = float(reference_row.get("year", 0) or 0)
        if cand_year > 0 and ref_year > 0:
            year_similarity = _closeness_score(cand_year, ref_year, scale=8.0)
            year_gap = abs(cand_year - ref_year)
            same_decade = 1.0 if int(cand_year // 10) == int(ref_year // 10) else 0.0
        else:
            year_similarity = year_gap = same_decade = 0.0

        cand_rt = float(movie_row.get("runtime", 0) or 0)
        ref_rt = float(reference_row.get("runtime", 0) or 0)
        if cand_rt > 0 and ref_rt > 0:
            runtime_similarity = _closeness_score(cand_rt, ref_rt, scale=25.0)
            runtime_gap = abs(cand_rt - ref_rt)
        else:
            runtime_similarity = runtime_gap = 0.0

        return {
            "genre_jaccard": float(genre_jaccard),
            "genre_overlap": float(genre_overlap),
            "shared_genres": float(shared_genres),
            "exact_genre_hit": float(exact_genre_hit),
            "title_jaccard": float(title_jaccard),
            "exact_title_hit": float(exact_title_hit),
            "overview_jaccard": 0.0,
            "overview_overlap": 0.0,
            "year_similarity": float(year_similarity),
            "year_gap": float(year_gap),
            "same_decade": float(same_decade),
            "runtime_similarity": float(runtime_similarity),
            "runtime_gap": float(runtime_gap),
        }

    # -----------------------------------------------------------------------
    # Candidate generation
    # -----------------------------------------------------------------------
    def generate_candidates(
        self,
        user_id: int,
        candidate_k: Optional[int] = None,
    ) -> pd.DataFrame:
        if self.meta is None:
            raise RuntimeError("Primero llama a fit().")

        candidate_k = candidate_k or self.config.candidate_k
        profile = self.build_user_profile(user_id)
        watched_ids = profile["watched_ids"]
        fav_genres = profile["favorite_genres"]
        genre_profile = profile["genre_profile"]
        negative_genre_profile = profile.get("negative_genre_profile", {})
        user_emb = profile["user_embedding"]

        pool = self.meta.copy()
        if watched_ids:
            pool = pool[~pool["tmdbId"].isin(watched_ids)].copy()
        if pool.empty:
            return pool

        pool["genre_affinity"] = pool["genres_list"].apply(
            lambda gs: self._genre_overlap(gs, fav_genres, genre_profile, negative_genre_profile)
        )

        if user_emb is not None and self.emb_index is not None:
            pool["embedding_sim"] = self._embedding_similarity_column(pool, user_emb)
            emb_weight = self.config.embedding_candidate_weight
            genre_weight = 0.30 - emb_weight * 0.5
        else:
            pool["embedding_sim"] = 0.0
            emb_weight = 0.0
            genre_weight = 0.30

        pool["candidate_score"] = (
            0.70 * pool["catalogue_score"].fillna(0.0)
            + genre_weight * _minmax(pool["genre_affinity"].fillna(0.0))
            + emb_weight * _minmax(pool["embedding_sim"].fillna(0.0))
        )

        pool = pool.sort_values(["candidate_score", "vote_count", "popularity"], ascending=False).reset_index(drop=True)
        return pool.head(candidate_k).copy()

    # -----------------------------------------------------------------------
    # Ranking: BALANCED
    # -----------------------------------------------------------------------
    def rank_candidates_balanced(self, user_id: int, candidates: pd.DataFrame) -> pd.DataFrame:
        if candidates.empty:
            return candidates

        profile = self.build_user_profile(user_id)
        watched_ids = profile["watched_ids"]
        fav_genres = profile["favorite_genres"]
        genre_profile = profile["genre_profile"]
        negative_genre_profile = profile.get("negative_genre_profile", {})
        mean_rating = profile["mean_rating"]
        user_emb = profile["user_embedding"]

        out = candidates.copy()

        if "genre_affinity" not in out.columns:
            out["genre_affinity"] = out["genres_list"].apply(
                lambda gs: self._genre_overlap(gs, fav_genres, genre_profile, negative_genre_profile)
            )

        out["pop_scaled"] = _minmax(out["popularity"].fillna(0.0))
        out["votes_scaled"] = _minmax(np.log1p(out["vote_count"].fillna(0.0)))
        out["rating_scaled"] = _minmax(out["vote_average"].fillna(0.0))
        out["year_scaled"] = _minmax(out["year"].fillna(0))
        out["quality_scaled"] = _minmax(out.get("quality_score", pd.Series([0.0] * len(out), index=out.index)))

        if user_emb is not None and self.emb_index is not None:
            out["embedding_sim"] = self._embedding_similarity_column(out, user_emb)
            emb_weight = self.config.embedding_balanced_weight
            genre_weight = max(0.35 - emb_weight * 0.5, 0.20)
        else:
            out["embedding_sim"] = 0.0
            emb_weight = 0.0
            genre_weight = 0.35

        out["rank_score"] = (
            genre_weight * out["genre_affinity"].fillna(0.0)
            + 0.40 * out["quality_scaled"].fillna(0.0)
            + 0.15 * out["pop_scaled"].fillna(0.0)
            + 0.10 * out["year_scaled"].fillna(0.0)
            + emb_weight * _minmax(out["embedding_sim"].fillna(0.0))
        )

        if fav_genres:
            fav_norm = {str(g).strip().lower() for g in fav_genres if str(g).strip()}
            if fav_norm:
                out["fav_genre_hit"] = out["genres_list"].apply(
                    lambda gs: 1.0 if any(str(g).strip().lower() in fav_norm for g in _to_list(gs)) else 0.0
                )
                out["rank_score"] += 0.10 * out["fav_genre_hit"]

        if not math.isnan(mean_rating) and mean_rating >= 4.0:
            out["rank_score"] += 0.03 * out["quality_scaled"]

        out["watched_penalty"] = out["tmdbId"].astype("int64").isin(watched_ids).astype("float32")
        out["rank_score"] -= 1.5 * out["watched_penalty"]

        return out.sort_values(["rank_score", "vote_count", "popularity"], ascending=False).reset_index(drop=True)

    # -----------------------------------------------------------------------
    # Ranking: POPULAR
    # -----------------------------------------------------------------------
    def rank_candidates_popular(self, user_id: int, candidates: pd.DataFrame) -> pd.DataFrame:
        if candidates.empty:
            return candidates

        profile = self.build_user_profile(user_id)
        watched_ids = profile["watched_ids"]
        fav_genres = profile["favorite_genres"]
        genre_profile = profile["genre_profile"]
        negative_genre_profile = profile.get("negative_genre_profile", {})
        mean_rating = profile["mean_rating"]
        user_emb = profile["user_embedding"]

        out = candidates.copy()

        if "genre_affinity" not in out.columns:
            out["genre_affinity"] = out["genres_list"].apply(
                lambda gs: self._genre_overlap(gs, fav_genres, genre_profile, negative_genre_profile)
            )

        out["pop_scaled"] = _minmax(out["popularity"].fillna(0.0))
        out["votes_scaled"] = _minmax(np.log1p(out["vote_count"].fillna(0.0)))
        out["rating_scaled"] = _minmax(out["vote_average"].fillna(0.0))
        out["year_scaled"] = _minmax(out["year"].fillna(0))
        out["quality_scaled"] = _minmax(out.get("quality_score", pd.Series([0.0] * len(out), index=out.index)))

        if user_emb is not None and self.emb_index is not None:
            out["embedding_sim"] = self._embedding_similarity_column(out, user_emb)
            emb_weight = self.config.embedding_popular_weight
        else:
            out["embedding_sim"] = 0.0
            emb_weight = 0.0

        out["popular_score"] = (
            0.10 * out["genre_affinity"].fillna(0.0)
            + 0.25 * out["quality_scaled"].fillna(0.0)
            + 0.40 * out["votes_scaled"].fillna(0.0)
            + 0.15 * out["pop_scaled"].fillna(0.0)
            + 0.10 * out["year_scaled"].fillna(0.0)
            + emb_weight * _minmax(out["embedding_sim"].fillna(0.0))
        )

        if fav_genres:
            fav_norm = {str(g).strip().lower() for g in fav_genres if str(g).strip()}
            if fav_norm:
                out["fav_genre_hit"] = out["genres_list"].apply(
                    lambda gs: 1.0 if any(str(g).strip().lower() in fav_norm for g in _to_list(gs)) else 0.0
                )
                out["popular_score"] += 0.05 * out["fav_genre_hit"]

        if not math.isnan(mean_rating) and mean_rating >= 4.0:
            out["popular_score"] += 0.01 * out["quality_scaled"]

        out["watched_penalty"] = out["tmdbId"].astype("int64").isin(watched_ids).astype("float32")
        out["popular_score"] -= 1.0 * out["watched_penalty"]

        return out.sort_values(["popular_score", "vote_count", "popularity"], ascending=False).reset_index(drop=True)

    # -----------------------------------------------------------------------
    # Ranking: SIMILAR
    # -----------------------------------------------------------------------
    def rank_candidates_similar(
        self,
        reference_genres: List[str],
        candidates: pd.DataFrame,
        user_id: Optional[int] = None,
        reference_row: Optional[pd.Series] = None,
    ) -> pd.DataFrame:
        if candidates.empty:
            return candidates

        if reference_row is None:
            reference_row = pd.Series({
                "title": "", "overview": "", "year": 0,
                "runtime": 0, "genres_list": reference_genres,
            })

        out = candidates.copy()

        sim_parts = out.apply(
            lambda row: self._movie_similarity_to_reference(row, reference_row),
            axis=1,
            result_type="expand",
        )
        out = pd.concat([out, sim_parts], axis=1)
        out["genre_similarity"] = out["genre_jaccard"].fillna(0.0)

        ref_tmdb_id = int(reference_row.get("tmdbId", -1)) if reference_row is not None else -1
        ref_emb = self.emb_index.get_vector(ref_tmdb_id) if self.emb_index is not None else None

        if ref_emb is not None and self.emb_index is not None:
            out["embedding_sim"] = self._embedding_similarity_column(out, ref_emb, "embedding_sim")
        else:
            out["embedding_sim"] = 0.0

        out["pop_scaled"] = _minmax(out["popularity"].fillna(0.0))
        out["votes_scaled"] = _minmax(np.log1p(out["vote_count"].fillna(0.0)))
        out["rating_scaled"] = _minmax(out["vote_average"].fillna(0.0))
        out["quality_scaled"] = _minmax(out.get("quality_score", pd.Series([0.0] * len(out), index=out.index)))

        out["similar_score"] = (
            self.config.similar_genre_weight * out["genre_jaccard"].fillna(0.0)
            + self.config.similar_shared_genre_weight * _minmax(out["shared_genres"].fillna(0.0))
            + self.config.similar_embedding_weight * _minmax(out["embedding_sim"].fillna(0.0))
            + self.config.similar_title_weight * out["title_jaccard"].fillna(0.0)
            + self.config.similar_year_weight * out["year_similarity"].fillna(0.0)
            + self.config.similar_runtime_weight * out["runtime_similarity"].fillna(0.0)
            + self.config.similar_quality_weight * out["quality_scaled"].fillna(0.0)
            + self.config.similar_popularity_weight * out["pop_scaled"].fillna(0.0)
            + self.config.similar_votes_weight * out["votes_scaled"].fillna(0.0)
        )

        out["similar_score"] += self.config.similar_exact_genre_boost * out["exact_genre_hit"].fillna(0.0)
        out["similar_score"] += self.config.similar_exact_title_boost * out["exact_title_hit"].fillna(0.0)
        out["similar_score"] += 0.02 * out["same_decade"].fillna(0.0)

        if user_id is not None:
            try:
                profile = self.build_user_profile(user_id)
                watched_ids = profile["watched_ids"]
                if watched_ids:
                    out["watched_penalty"] = out["tmdbId"].astype("int64").isin(watched_ids).astype("float32")
                    out["similar_score"] -= 1.5 * out["watched_penalty"]
            except Exception:
                pass

        out["similar_score"] += 0.03 * out["quality_scaled"].fillna(0.0)
        out["similar_score"] += 0.02 * out["rating_scaled"].fillna(0.0)

        return out.sort_values(
            ["similar_score", "genre_jaccard", "vote_count", "popularity"],
            ascending=False,
        ).reset_index(drop=True)

    # -----------------------------------------------------------------------
    # Public API: SIMILAR
    # -----------------------------------------------------------------------
    def recommend_similar(
        self,
        reference_tmdb_id: int,
        user_id: Optional[int] = None,
        n: Optional[int] = None,
        candidate_k: Optional[int] = None,
    ) -> pd.DataFrame:
        if self.meta is None:
            raise RuntimeError("Primero llama a fit().")

        n = n or self.config.recommendations_n
        candidate_k = candidate_k or self.config.candidate_k

        ref_row = self.meta[self.meta["tmdbId"] == reference_tmdb_id]
        if ref_row.empty:
            return self.recommend_random(n=n, user_id=user_id)

        reference_row = ref_row.iloc[0]
        reference_genres: List[str] = _to_list(reference_row["genres_list"])

        if user_id is not None:
            try:
                candidates = self.generate_candidates(
                    user_id=user_id,
                    candidate_k=max(candidate_k, self.config.candidate_k),
                )
            except Exception:
                candidates = self.meta.copy().head(candidate_k)
        else:
            candidates = self.meta.copy().head(max(candidate_k, self.config.candidate_k))

        candidates = candidates[candidates["tmdbId"] != reference_tmdb_id].copy()
        if candidates.empty:
            return candidates

        ranked = self.rank_candidates_similar(
            reference_genres=reference_genres,
            candidates=candidates,
            user_id=user_id,
            reference_row=reference_row,
        )
        return ranked.head(n).copy()

    def recommend_similar_with_explanations(
        self,
        reference_tmdb_id: int,
        user_id: Optional[int] = None,
        n: Optional[int] = None,
        candidate_k: Optional[int] = None,
    ) -> pd.DataFrame:
        recs = self.recommend_similar(
            reference_tmdb_id=reference_tmdb_id,
            user_id=user_id,
            n=n,
            candidate_k=candidate_k,
        )
        if recs.empty:
            return recs

        cols = [
            "tmdbId", "title", "poster_url", "year", "runtime",
            "vote_average", "vote_count", "popularity",
            "genres_list", "genre_similarity", "similar_score",
            "quality_score", "weighted_rating", "vote_confidence",
            "embedding_sim", "title_jaccard", "year_similarity",
            "runtime_similarity", "shared_genres", "same_decade",
        ]
        return recs[[c for c in cols if c in recs.columns]].copy()

    # -----------------------------------------------------------------------
    # Public API: BALANCED
    # -----------------------------------------------------------------------
    def recommend(
        self,
        user_id: int,
        n: Optional[int] = None,
        candidate_k: Optional[int] = None,
    ) -> pd.DataFrame:
        if self.meta is None:
            raise RuntimeError("Primero llama a fit().")

        n = n or self.config.recommendations_n
        candidate_k = candidate_k or self.config.candidate_k
        candidates = self.generate_candidates(user_id=user_id, candidate_k=candidate_k)
        ranked = self.rank_candidates_balanced(user_id=user_id, candidates=candidates)
        return ranked.head(n).copy()

    def recommend_with_explanations(
        self,
        user_id: int,
        n: Optional[int] = None,
        candidate_k: Optional[int] = None,
    ) -> pd.DataFrame:
        recs = self.recommend(user_id=user_id, n=n, candidate_k=candidate_k)
        if recs.empty:
            return recs

        cols = [
            "tmdbId", "title", "poster_url", "year", "runtime",
            "vote_average", "vote_count", "popularity",
            "genres_list", "genre_affinity", "rank_score",
            "embedding_sim", "weighted_rating", "vote_confidence",
            "quality_score", "votes_scaled", "pop_scaled", "year_scaled",
        ]
        return recs[[c for c in cols if c in recs.columns]].copy()

    # -----------------------------------------------------------------------
    # Public API: POPULAR
    # -----------------------------------------------------------------------
    def recommend_popular(
        self,
        user_id: int,
        n: Optional[int] = None,
        candidate_k: Optional[int] = None,
    ) -> pd.DataFrame:
        if self.meta is None:
            raise RuntimeError("Primero llama a fit().")

        n = n or self.config.recommendations_n
        candidate_k = candidate_k or self.config.candidate_k
        candidates = self.generate_candidates(user_id=user_id, candidate_k=candidate_k)
        ranked = self.rank_candidates_popular(user_id=user_id, candidates=candidates)
        return ranked.head(n).copy()

    def recommend_popular_with_explanations(
        self,
        user_id: int,
        n: Optional[int] = None,
        candidate_k: Optional[int] = None,
    ) -> pd.DataFrame:
        recs = self.recommend_popular(user_id=user_id, n=n, candidate_k=candidate_k)
        if recs.empty:
            return recs

        cols = [
            "tmdbId", "title", "poster_url", "year", "runtime",
            "vote_average", "vote_count", "popularity",
            "genres_list", "genre_affinity", "popular_score",
            "embedding_sim", "weighted_rating", "vote_confidence",
            "quality_score", "votes_scaled", "pop_scaled", "year_scaled",
        ]
        return recs[[c for c in cols if c in recs.columns]].copy()

    # -----------------------------------------------------------------------
    # Public API: RANDOM
    # -----------------------------------------------------------------------
    def recommend_random(
        self,
        n: Optional[int] = None,
        user_id: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> pd.DataFrame:
        if self.meta is None:
            raise RuntimeError("Primero llama a fit().")

        n = n or self.config.recommendations_n
        pool = self.meta.copy()

        if user_id is not None:
            try:
                profile = self.build_user_profile(user_id)
                watched_ids = profile["watched_ids"]
                if watched_ids:
                    pool = pool[~pool["tmdbId"].isin(watched_ids)].copy()
            except Exception:
                pass

        if pool.empty:
            return pool

        sample_n = min(n, len(pool))
        rng = np.random.default_rng(seed)
        idx = rng.choice(pool.index.to_numpy(), size=sample_n, replace=False)
        return pool.loc[idx].copy().reset_index(drop=True)

    # -----------------------------------------------------------------------
    # Convenience helper
    # -----------------------------------------------------------------------
    def recommend_dashboard_triplet(
        self,
        user_id: int,
        n: int = 12,
        random_seed: Optional[int] = None,
        candidate_k: Optional[int] = None,
    ) -> Dict[str, pd.DataFrame]:
        return {
            "recomendaciones": self.recommend_with_explanations(user_id=user_id, n=n, candidate_k=candidate_k),
            "populares": self.recommend_popular_with_explanations(user_id=user_id, n=n, candidate_k=candidate_k),
            "random": self.recommend_random(n=n, user_id=user_id, seed=random_seed),
        }



# Entry point / factory

def build_recommender(
    embeddings_parquet: str,
    users_csv: str,
    interactions_csv: str,
    cache_dir: str,
    force_rebuild_cache: bool = False,
    movies_csv: Optional[str] = None,
    embeddings_csv: Optional[str] = None,
) -> MovieRecommender:
    paths = RecommenderPaths(
        embeddings_parquet=Path(embeddings_parquet),
        users_csv=Path(users_csv),
        interactions_csv=Path(interactions_csv),
        cache_dir=Path(cache_dir),
        movies_csv=Path(movies_csv) if movies_csv else None,
        embeddings_csv=Path(embeddings_csv) if embeddings_csv else None,
    )
    recommender = MovieRecommender(paths=paths)
    recommender.fit(force_rebuild_cache=force_rebuild_cache)
    return recommender


if __name__ == "__main__":
    PROJECT_ROOT = Path(__file__).resolve().parents[1]

    recommender = build_recommender(
        embeddings_parquet=str(PROJECT_ROOT / "data" / "embeddings" / "tmdb_embeddings.parquet"),
        users_csv=str(PROJECT_ROOT / "data" / "users.csv"),
        interactions_csv=str(PROJECT_ROOT / "data" / "interactions.csv"),
        cache_dir=str(PROJECT_ROOT / "cache" / "reco_cache"),
        force_rebuild_cache=False,
    )

    result_balanced = recommender.recommend_with_explanations(user_id=1, n=12)
    print("\n=== BALANCED (MODE 1) ===")
    print(result_balanced.to_string(index=False))

    result_popular = recommender.recommend_popular_with_explanations(user_id=1, n=12)
    print("\n=== POPULAR (MODE 2) ===")
    print(result_popular.to_string(index=False))

    result_random = recommender.recommend_random(n=12, user_id=1, seed=42)
    print("\n=== RANDOM (MODE 3) ===")
    print(result_random.to_string(index=False))

    REFERENCE_ID = 550
    result_similar = recommender.recommend_similar_with_explanations(
        reference_tmdb_id=REFERENCE_ID, user_id=1, n=12
    )
    print(f"\n=== SIMILAR to tmdbId={REFERENCE_ID} (MODE 4) ===")
    print(result_similar.to_string(index=False))
