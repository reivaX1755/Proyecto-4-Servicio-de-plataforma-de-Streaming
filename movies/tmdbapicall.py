import os
import time
import logging
import requests
import pandas as pd
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

#  CONFIG 
API_KEY = "f8b713317cafa651474dc1aa3360cea2"
OUTPUT_CSV = "tmdb_dataset_full.csv"
CHECKPOINT_FILE = "tmdb_last_id.txt"
PAUSE_SEC = 0.3
TIMEOUT = 10
SAVE_EVERY = 100

#  LOGGING 
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

#  SESSION CON RETRIES 
session = requests.Session()
retries = Retry(
    total=5,
    backoff_factor=1,
    status_forcelist=[429, 500, 502, 503, 504],
    allowed_methods=["GET"]
)
adapter = HTTPAdapter(max_retries=retries)
session.mount("https://", adapter)

#  FUNCIONES 
def fetch_movie(tmdb_id):
    url = f"https://api.themoviedb.org/3/movie/{tmdb_id}"
    params = {"api_key": API_KEY, "language": "en-US"}
    try:
        resp = session.get(url, params=params, timeout=TIMEOUT)
        if resp.status_code == 200:
            d = resp.json()
            poster_path = d.get("poster_path")
            poster_url = f"https://image.tmdb.org/t/p/w500{poster_path}" if poster_path else None
            genres = "|".join([g.get("name", "") for g in d.get("genres", [])])
            return {
                "tmdbId": d.get("id"),
                "title": d.get("title"),
                "genres": genres,
                "overview": d.get("overview"),
                "poster_url": poster_url,
                "vote_average": d.get("vote_average"),
                "vote_count": d.get("vote_count"),
                "popularity": d.get("popularity"),
                "runtime": d.get("runtime"),
                "release_date": d.get("release_date")
            }
        elif resp.status_code == 404:
            return None
        else:
            logging.error(f"Error {resp.status_code} for ID {tmdb_id}")
            return None
    except requests.RequestException as e:
        logging.error(f"Request failed for ID {tmdb_id}: {e}")
        return None

def load_last_id():
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE, "r") as f:
            return int(f.read().strip())
    return 1

def save_last_id(last_id):
    with open(CHECKPOINT_FILE, "w") as f:
        f.write(str(last_id))

#  MAIN 

# Cargar registros ya guardados
if os.path.exists(OUTPUT_CSV):
    df_existing = pd.read_csv(OUTPUT_CSV)
    existing_ids = set(df_existing["tmdbId"].tolist())
    results = df_existing.to_dict(orient="records")
    logging.info(f"Loaded {len(results)} existing records")
else:
    existing_ids = set()
    results = []

current_id = load_last_id()
logging.info(f"Starting from tmdbId={current_id}")

try:
    while True:
        if current_id not in existing_ids:
            record = fetch_movie(current_id)
            if record:
                results.append(record)
                existing_ids.add(record["tmdbId"])

        current_id += 1

        # Guardado incremental
        if len(results) % SAVE_EVERY == 0 and len(results) > 0:
            pd.DataFrame(results).to_csv(OUTPUT_CSV, index=False)
            save_last_id(current_id)
            logging.info(f"Saved {len(results)} records. Last ID: {current_id}")

        time.sleep(PAUSE_SEC)

except KeyboardInterrupt:
    logging.info("Stopped manually. Saving progress...")
    pd.DataFrame(results).to_csv(OUTPUT_CSV, index=False)
    save_last_id(current_id)
    logging.info(f"Final save: {len(results)} records. Last ID: {current_id}")