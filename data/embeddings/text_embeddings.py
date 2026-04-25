"""
generate_embeddings.py

Genera embeddings ligeros para películas usando:
- genres + overview
- Modelo: all-MiniLM-L6-v2 (rápido y eficiente)
- Reducción de dimensionalidad con PCA
- Compresión eficiente (Parquet + float16)

Salida:
Parquet con columna adicional: "embedding"
"""

from pathlib import Path
import ast
import pandas as pd
import numpy as np

from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA


# Paths
ROOT_DIR = Path(__file__).resolve().parents[2]

INPUT_CSV = ROOT_DIR / "movies" / "tmdb_dataset_full.csv"
OUTPUT_FILE = ROOT_DIR / "data" / "embeddings" / "tmdb_embeddings.parquet"


# Config
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
BATCH_SIZE = 256
TARGET_DIM = 128  # Reducimos de 384  128


# Utils
def parse_genres(value):
    if pd.isna(value):
        return ""

    if isinstance(value, list):
        return " ".join([str(x) for x in value])

    text = str(value)
    try:
        parsed = ast.literal_eval(text)
        if isinstance(parsed, list):
            return " ".join([
                g.get("name", "") if isinstance(g, dict) else str(g)
                for g in parsed
            ])
    except Exception:
        pass

    return text.replace("|", " ")


def build_text(row):
    genres = parse_genres(row.get("genres"))
    overview = str(row.get("overview") or "")
    return f"{genres}. {overview}".strip()


# Main
def main():
    print("Cargando dataset...")
    df = pd.read_csv(INPUT_CSV, low_memory=False)

    print(f"Registros: {len(df)}")

    # Generar texto
    print("Generando texto combinado...")
    df["text"] = df.apply(build_text, axis=1)

    # Modelo
    print("Cargando modelo...")
    model = SentenceTransformer(MODEL_NAME)

    # Embeddings
    print("Generando embeddings...")
    embeddings = model.encode(
        df["text"].tolist(),
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    print(f"Dim original: {embeddings.shape}")

    # PCA
    print(f"Aplicando PCA  {TARGET_DIM} dimensiones...")
    pca = PCA(n_components=TARGET_DIM)
    reduced_embeddings = pca.fit_transform(embeddings)

    # Reducir a float16
    print("🪶 Convirtiendo a float16 (reducción de tamaño)...")
    reduced_embeddings = reduced_embeddings.astype(np.float16)

    # Guardar embeddings como lista
    print("Preparando columna embedding...")
    df["embedding"] = [emb.tolist() for emb in reduced_embeddings]

    # Eliminar columna temporal
    df.drop(columns=["text"], inplace=True)

    # Guardado optimizado
    print("Guardando en Parquet (compresión ZSTD)...")

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    df.to_parquet(
        OUTPUT_FILE,
        index=False,
        engine="pyarrow",
        compression="zstd"  # mejor ratio + velocidad
    )

    print(f"Archivo guardado en: {OUTPUT_FILE}")
    print(f"Shape embeddings: {reduced_embeddings.shape}")

    # Estimación tamaño en memoria
    size_mb = reduced_embeddings.nbytes / (1024 * 1024)
    print(f"Tamaño embeddings en RAM (float16): ~{size_mb:.2f} MB")


if __name__ == "__main__":
    main()