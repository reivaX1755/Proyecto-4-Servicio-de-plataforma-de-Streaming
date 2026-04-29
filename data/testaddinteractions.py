import pandas as pd
import random
from datetime import datetime, timedelta
from pathlib import Path


# CONFIG

USER_ID = 23
RATING = 5.0
OUTPUT_FILE = Path(__file__).resolve().parent / "interactions.csv"

MOVIE_IDS = [
    346364, 694, 419430, 381288, 72190, 348, 44214, 447332, 19908, 138843,
    578, 539, 405774, 176, 474350, 126889, 300668, 747, 259693, 158015,
    333371, 9552, 22970, 493922, 170, 396535, 345940, 458723, 530385, 300669,
    561, 395992, 1091, 4232, 282035, 310131, 2668, 520763, 49018, 439079,
    1933, 60304, 381283, 927, 270303, 1576, 238636, 1813, 36647, 565
]


# FUNCIONES

def random_date():
    now = datetime.now()
    days_ago = random.randint(0, 365)
    return (now - timedelta(days=days_ago)).strftime("%Y-%m-%d %H:%M:%S")

def get_next_interaction_id(df: pd.DataFrame) -> int:
    if df.empty or "interaction_id" not in df.columns:
        return 1
    return int(pd.to_numeric(df["interaction_id"], errors="coerce").max()) + 1


# CARGA DEL CSV EXISTENTE

if OUTPUT_FILE.exists():
    df_existing = pd.read_csv(OUTPUT_FILE, low_memory=False)
    next_id = get_next_interaction_id(df_existing)
else:
    df_existing = pd.DataFrame(columns=[
        "interaction_id",
        "user_id",
        "movie_id",
        "rating",
        "valoration_date"
    ])
    next_id = 1


# GENERAR NUEVAS INTERACCIONES

new_rows = []

for movie_id in MOVIE_IDS:
    new_rows.append([
        next_id,
        USER_ID,
        movie_id,
        RATING,
        random_date()
    ])
    next_id += 1

df_new = pd.DataFrame(
    new_rows,
    columns=[
        "interaction_id",
        "user_id",
        "movie_id",
        "rating",
        "valoration_date"
    ]
)


# UNIR Y GUARDAR
df_final = pd.concat([df_existing, df_new], ignore_index=True)
df_final.to_csv(OUTPUT_FILE, index=False)

print(f"Archivo actualizado: {OUTPUT_FILE}")
print(f"Nuevas interacciones añadidas: {len(df_new)}")
print(f"Último interaction_id usado: {next_id - 1}")