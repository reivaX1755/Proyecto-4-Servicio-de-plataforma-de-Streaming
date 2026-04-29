# Proyecto 4 - Servicio de plataforma de streaming

## Descripción del proyecto

Este proyecto es una aplicación de recomendación y descubrimiento de películas desarrollada con Streamlit. La plataforma combina datos de interacciones de usuarios, procesamiento de contenidos y recomendaciones de películas para ofrecer una experiencia de exploración personalizada.

El sistema incluye:

- Gestión de datos de películas y metadatos de TMDB.
- Recomendaciones basadas en interacciones de usuarios y embeddings.
- Interfaz web con Streamlit para visualización y navegación de películas.
- Módulos de limpieza y preparación de datos.

## Estructura principal del repositorio

- `app.py` - Punto de entrada de la aplicación Streamlit.
- `requirements.txt` - Dependencias necesarias para ejecutar el proyecto.
- `data/` - Carpeta de datos con `interactions.csv`, `users.csv` y subcarpetas de embeddings.
- `movies/` - Carpeta con el dataset principal de películas (`tmdb_dataset_full.csv`).
- `src/` - Código fuente de la aplicación y módulos de recomendación.
- `assets/` - Scripts y utilidades de procesamiento de datos.

## Requisitos previos

- Python 3.14 o superior.
- Conexión a Internet para descargar los datasets.
- Recomendado usar un entorno virtual (venv, conda, etc.).

## Manual de instalación y ejecución

1. Clona el repositorio:

```bash
git clone https://github.com/tu-usuario/tu-repositorio.git
cd Proyecto-4-Servicio-de-plataforma-de-Streaming
```

2. Descarga los datasets desde el siguiente enlace:

https://drive.google.com/drive/folders/1kuFtIGV_fTRWykWmEs7PKzE6W7gbXM1E?usp=sharing

3. Copia los archivos descargados en las carpetas del proyecto:

- `tmdb.embeddings.parquet` → `data/embeddings/`
- `tmdb_dataset_full.csv` → `movies/`

4. Instala las dependencias del proyecto:

```bash
pip install -r requirements.txt
```

5. Ejecuta la aplicación Streamlit:

```bash
python -m streamlit run app.py
```

O también:

```bash
python streamlit run app.py
```

## Uso

- Abre la URL que Streamlit indique en la terminal.
- Navega por la interfaz para explorar las recomendaciones de películas.
- Busca o filtra contenido según la lógica implementada en la aplicación.

## Notas

- Asegúrate de que los archivos `tmdb.embeddings.parquet` y `tmdb_dataset_full.csv` estén presentes en las rutas correctas antes de ejecutar la aplicación.
- Si agregas nuevos datos o actualizas los embeddings, revisa los scripts dentro de `assets/` para generar o procesar los archivos correctamente.

## Videos al final de la documentación
