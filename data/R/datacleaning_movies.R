library(dplyr)

# Cargar dataset original
df <- read.csv("C:/Users/Jordi/Desktop/Master IABiGData/Proyecto 4 Plataforma de streaming/Proyecto-4-Servicio-de-plataforma-de-Streaming/movies/tmdb_dataset_full.csv",
               stringsAsFactors = FALSE)

#todos los registros
total_inicial <- nrow(df)
cat("Total inicial de registros:", total_inicial, "\n\n")

#registros sin poster
df_sin_poster <- df %>%
  filter(is.na(poster_url) | poster_url == "")

cat("Películas sin poster:", nrow(df_sin_poster), "\n")

df <- df %>%
  filter(!(is.na(poster_url) | poster_url == ""))

cat("Quedan tras quitar sin poster:", nrow(df), "\n\n")

#registros con 1 voto o menos
df_1_voto <- df %>%
  filter(vote_count <= 1 | is.na(vote_count))

cat("Películas con 1 voto o menos:", nrow(df_1_voto), "\n")

df <- df %>%
  filter(vote_count > 1)

cat("Quedan tras quitar películas con 1 voto:", nrow(df), "\n\n")

#campos 
df_campos_faltantes <- df %>%
  filter(if_any(everything(), ~ is.na(.) | . == ""))

cat("Películas con algún campo faltante:", nrow(df_campos_faltantes), "\n")

df_totalmente_limpio <- df %>%
  filter(if_all(everything(), ~ !is.na(.) & . != ""))

cat("Quedan tras limpieza completa:", nrow(df_totalmente_limpio), "\n\n")

#resumen
total_final <- nrow(df_totalmente_limpio)

cat("TOTAL FINAL:", total_final, "\n")
cat("TOTAL ELIMINADO:", total_inicial - total_final, "\n\n")

# guardad csv
ruta_salida <- "C:/Users/Jordi/Desktop/tmdb_dataset_totalmente_limpio.csv"

write.csv(df_totalmente_limpio, ruta_salida, row.names = FALSE)

cat("Archivo guardado en:", ruta_salida, "\n")