import pandas as pd

# --------------------------------------------------
# Cargar CSV
# --------------------------------------------------
df = pd.read_csv(
    r"C:/Users/Jordi/Desktop/Master IABiGData/Proyecto 4 Plataforma de streaming/Proyecto-4-Servicio-de-plataforma-de-Streaming/movies/tmdb_dataset_full.csv"
)

print(f"Total de registros cargados: {len(df)}\n")

# --------------------------------------------------
# Menú interactivo
# --------------------------------------------------
print("Selecciona una opción:")
print("1: Buscar por ID")
print("2: Buscar por título")

opcion = input("Opción: ").strip()

# --------------------------------------------------
# Buscar por ID
# --------------------------------------------------
if opcion == "1":
    
    try:
        buscar_id = int(input("Introduce el tmdbId: ").strip())
    except ValueError:
        print("ID inválido")
        exit()

    resultado = df[df["tmdbId"] == buscar_id]

    print("\n===== RESULTADO POR ID =====")
    print(f"Resultados encontrados: {len(resultado)}\n")

    if len(resultado) > 0:
        print(resultado[["tmdbId", "title", "release_date", "vote_average", "vote_count"]])
    else:
        print("No se encontró ninguna película con ese ID.")

# --------------------------------------------------
# Buscar por título
# --------------------------------------------------
elif opcion == "2":

    buscar_titulo = input("Introduce el título: ").strip().lower()

    resultado = df[df["title"].str.lower().str.contains(buscar_titulo, na=False)]

    print("\n===== RESULTADO POR TÍTULO =====")
    print(f"Resultados encontrados: {len(resultado)}\n")

    if len(resultado) > 0:
        print(resultado[["tmdbId", "title", "release_date", "vote_average", "vote_count"]])
    else:
        print("No se encontró ninguna película con ese título.")

# --------------------------------------------------
# Opción inválida
# --------------------------------------------------
else:
    print("No seleccionaste ninguna opción válida.")