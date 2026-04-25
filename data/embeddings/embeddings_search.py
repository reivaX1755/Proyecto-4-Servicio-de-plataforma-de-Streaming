"""
search_parquet_movies.py

Script para buscar películas dentro de un archivo .parquet
por:
- tmdbId
- título (exacto o parcial)

Uso:

1. Buscar por ID:
python search_parquet_movies.py --id 27

2. Buscar por título exacto:
python search_parquet_movies.py --title "9 Songs"

3. Buscar por coincidencia parcial:
python search_parquet_movies.py --title "songs"

4. Especificar otro parquet:
python search_parquet_movies.py --file "ruta/al/archivo.parquet" --id 27
"""

from pathlib import Path
import argparse
import pandas as pd


# --------------------------------------------------
# CONFIG
# --------------------------------------------------

DEFAULT_PARQUET_PATH = "tmdb_embeddings.parquet"


# --------------------------------------------------
# HELPERS
# --------------------------------------------------

def load_parquet(parquet_path: str) -> pd.DataFrame:
    path = Path(parquet_path)

    if not path.exists():
        raise FileNotFoundError(f"No existe el archivo: {path}")

    print(f"\nCargando parquet: {path}")
    df = pd.read_parquet(path)

    print(f"Registros encontrados: {len(df):,}")
    print(f"Columnas: {list(df.columns)}\n")

    return df


def print_movie_row(row: pd.Series):
    print("=" * 80)
    print(f"tmdbId:        {row.get('tmdbId', '')}")
    print(f"title:         {row.get('title', '')}")
    print(f"genres:        {row.get('genres', '')}")
    print(f"release_date:  {row.get('release_date', '')}")
    print(f"runtime:       {row.get('runtime', '')}")
    print(f"vote_average:  {row.get('vote_average', '')}")
    print(f"vote_count:    {row.get('vote_count', '')}")
    print(f"popularity:    {row.get('popularity', '')}")
    print(f"poster_url:    {row.get('poster_url', '')}")
    print()
    print("overview:")
    print(row.get("overview", ""))
    print()

    embedding = row.get("embedding", [])
    if isinstance(embedding, list):
        print(f"embedding size: {len(embedding)}")
        print(f"embedding preview: {embedding[:10]}")
    else:
        print("embedding: no disponible")

    print("=" * 80)
    print()


def search_by_id(df: pd.DataFrame, movie_id: int):
    if "tmdbId" not in df.columns:
        print("La columna 'tmdbId' no existe en el parquet.")
        return

    df["tmdbId"] = pd.to_numeric(df["tmdbId"], errors="coerce")

    result = df[df["tmdbId"] == movie_id]

    if result.empty:
        print(f"No se encontró ninguna película con tmdbId = {movie_id}")
        return

    print(f"Encontradas {len(result)} coincidencias para tmdbId = {movie_id}\n")

    for _, row in result.iterrows():
        print_movie_row(row)


def search_by_title(df: pd.DataFrame, title: str):
    if "title" not in df.columns:
        print("La columna 'title' no existe en el parquet.")
        return

    df["title"] = df["title"].astype(str)

    result = df[
        df["title"].str.lower().str.contains(title.lower(), na=False)
    ]

    if result.empty:
        print(f"No se encontró ninguna película con título que contenga: {title}")
        return

    print(f"Encontradas {len(result)} coincidencias para título = '{title}'\n")

    # Limitar salida para no explotar consola
    max_results = 20

    for i, (_, row) in enumerate(result.iterrows()):
        if i >= max_results:
            print(f"... mostrando solo las primeras {max_results} coincidencias")
            break

        print_movie_row(row)


# --------------------------------------------------
# MAIN
# --------------------------------------------------

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--file",
        type=str,
        default=DEFAULT_PARQUET_PATH,
        help="Ruta del archivo parquet"
    )

    parser.add_argument(
        "--id",
        type=int,
        help="Buscar por tmdbId"
    )

    parser.add_argument(
        "--title",
        type=str,
        help="Buscar por título (parcial o exacto)"
    )

    args = parser.parse_args()

    if args.id is None and args.title is None:
        print("Debes indicar --id o --title")
        return

    df = load_parquet(args.file)

    if args.id is not None:
        search_by_id(df, args.id)

    if args.title is not None:
        search_by_title(df, args.title)


if __name__ == "__main__":
    main()