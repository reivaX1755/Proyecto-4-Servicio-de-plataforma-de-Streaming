import pandas as pd
import random
from datetime import datetime, timedelta

# -----------------------------
# CONFIG
# -----------------------------
USERS = list(range(1, 11))  # usuarios 1–10
MOVIES_CSV = "../movies/tmdb_dataset_full.csv"
OUTPUT_FILE = "interactions.csv"

INTERACTIONS_PER_USER = (10, 30)  # min, max por usuario

# Ratings posibles (Letterboxd style)
RATINGS = [x * 0.5 for x in range(0, 11)]  # 0 → 5 en pasos de 0.5


# -----------------------------
# FUNCIONES
# -----------------------------
def random_date():
    now = datetime.now()
    days_ago = random.randint(0, 365)
    return (now - timedelta(days=days_ago)).strftime("%Y-%m-%d %H:%M:%S")


def weighted_rating(match=True):
    if match:
        return random.choices(
            [3.0, 3.5, 4.0, 4.5, 5.0],
            weights=[10, 15, 30, 25, 20]
        )[0]
    else:
        return random.choices(
            [0.5, 1.0, 1.5, 2.0, 2.5, 3.0],
            weights=[15, 20, 20, 20, 15, 10]
        )[0]


# -----------------------------
# LOAD MOVIES (CLAVE)
# -----------------------------
print("Cargando dataset de películas...")

movies_df = pd.read_csv(
    MOVIES_CSV,
    usecols=["tmdbId", "genres"],
    low_memory=False
)

# Limpiar IDs válidos
movies_df = movies_df.dropna(subset=["tmdbId"])
movies_df["tmdbId"] = movies_df["tmdbId"].astype(int)

# Lista REAL de IDs válidos
movie_ids = movies_df["tmdbId"].tolist()

print(f"Películas válidas cargadas: {len(movie_ids)}")


# -----------------------------
# GENERAR INTERACCIONES
# -----------------------------
interactions = []
interaction_id = 1

for user_id in USERS:
    num_interactions = random.randint(*INTERACTIONS_PER_USER)

    # Seleccionar películas únicas para este usuario
    selected_movies = random.sample(movie_ids, num_interactions)

    for movie_id in selected_movies:
        # Simulación simple (puedes mejorar con géneros luego)
        match = random.random() < 0.6  # 60% afinidad

        rating = weighted_rating(match)
        date = random_date()

        interactions.append([
            interaction_id,
            user_id,
            movie_id,
            rating,
            date
        ])

        interaction_id += 1


# -----------------------------
# GUARDAR CSV
# -----------------------------
df = pd.DataFrame(
    interactions,
    columns=[
        "interaction_id",
        "user_id",
        "movie_id",
        "rating",
        "valoration_date"
    ]
)

df.to_csv(OUTPUT_FILE, index=False)

print(f"Dataset generado: {OUTPUT_FILE}")
print(f"Total interacciones: {len(df)}")