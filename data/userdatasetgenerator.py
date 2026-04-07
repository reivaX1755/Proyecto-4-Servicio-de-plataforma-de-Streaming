import csv
import random
from datetime import datetime, timedelta

# Configuración
num_users = 10
output_file = "users_test.csv"

# Posibles géneros
genres = [
    "Comedy", "Drama", "Romance", "Crime", "Action", "Thriller", "Documentary",
    "Adventure", "Science Fiction", "Animation", "Family", "Mystery", "Horror",
    "Fantasy", "War", "Music", "Western", "History", "TV Movie"
]

# Preferencias por usuario
min_genres = 1
max_genres = 4

# Género (normalizado)
genders = ["male", "female", "other"]

# Generación de fechas
def random_date():
    now = datetime.now()
    delta_days = random.randint(0, 365)
    random_time = now - timedelta(days=delta_days)
    return random_time.strftime("%Y-%m-%d %H:%M:%S")

# Crear usuarios
users = []
for i in range(1, num_users + 1):
    user_id = i
    username = f"user{i}"
    email = f"user{i}@example.com"
    password = f"pass{i}"  # simple (sin hash por ahora)

    age = random.randint(15, 70)
    gender = random.choice(genders)

    num_preferences = random.randint(min_genres, max_genres)
    preferences = random.sample(genres, num_preferences)
    preferences_str = "|".join(preferences)

    created_at = random_date()

    users.append([
        user_id,
        username,
        email,
        password,
        gender,
        age,
        preferences_str,
        created_at
    ])

# Guardar CSV
with open(output_file, mode='w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)

    # Cabecera FINAL (la importante)
    writer.writerow([
        "user_id",
        "username",
        "email",
        "password",
        "gender",
        "age",
        "favorite_genres",
        "created_at"
    ])

    writer.writerows(users)

print(f"Dataset generado: {output_file} con {num_users} usuarios")