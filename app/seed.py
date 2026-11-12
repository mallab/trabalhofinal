"""
Carrega o MovieLens no banco na primeira vez que a aplicação sobe.

Procura os CSVs nessa ordem:
1. data/ml-latest-small/  (MovieLens completo, baixado pelo scripts/download_data.py)
2. data/                  (movies.csv e ratings.csv soltos)
3. data/sample/           (base pequena que fica no repositório)
"""

from __future__ import annotations

import io
import logging
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd
from sqlalchemy import func, insert, select, text
from sqlalchemy.orm import Session

from app.config import DATA_DIR, DOWNLOAD_DATA
from app.models import Item, Rating, User
from app.recommender import NO_GENRE

logger = logging.getLogger(__name__)

MOVIELENS_URL = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"
CHUNK = 5000  # insere de 5000 em 5000 linhas


def download_movielens(data_dir: Path = DATA_DIR, url: str = MOVIELENS_URL) -> Path:
    # baixa o zip (~1 MB) e extrai só os 2 CSVs que a gente usa
    target = data_dir / "ml-latest-small"
    if (target / "movies.csv").exists() and (target / "ratings.csv").exists():
        return target
    data_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Baixando MovieLens de %s", url)
    with urllib.request.urlopen(url, timeout=60) as resp:
        payload = resp.read()
    with zipfile.ZipFile(io.BytesIO(payload)) as zf:
        for name in ("movies.csv", "ratings.csv"):
            zf.extract(f"ml-latest-small/{name}", data_dir)
    return target


def find_dataset(data_dir: Path = DATA_DIR) -> Path | None:
    for folder in (data_dir / "ml-latest-small", data_dir, data_dir / "sample"):
        if (folder / "movies.csv").exists() and (folder / "ratings.csv").exists():
            return folder
    return None


def load_movielens(folder: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    # lê os CSVs e já deixa com os nomes de coluna das nossas tabelas
    movies = pd.read_csv(folder / "movies.csv")
    ratings = pd.read_csv(folder / "ratings.csv")

    items = pd.DataFrame({
        "id": movies["movieId"].astype(int),
        "title": movies["title"].str.strip(),
        "genres": movies["genres"].replace(NO_GENRE, "").fillna(""),
        # o ano vem no final do título, ex: "Toy Story (1995)"
        "year": pd.to_numeric(movies["title"].str.extract(r"\((\d{4})\)\s*$")[0], errors="coerce"),
    })
    ratings = pd.DataFrame({
        "user_id": ratings["userId"].astype(int),
        "item_id": ratings["movieId"].astype(int),
        "rating": ratings["rating"].astype(float),
        "timestamp": pd.to_datetime(ratings["timestamp"], unit="s", utc=True),
    })
    ratings = ratings[ratings["item_id"].isin(items["id"])]
    # se tiver nota repetida pro mesmo filme, fica a mais recente
    ratings = ratings.sort_values("timestamp").drop_duplicates(["user_id", "item_id"], keep="last")
    return items, ratings.reset_index(drop=True)


def _bulk_insert(db: Session, model, records: list[dict]) -> None:
    for start in range(0, len(records), CHUNK):
        db.execute(insert(model), records[start:start + CHUNK])


def _reset_sequences(db: Session) -> None:
    # No Postgres, como a gente insere os ids do MovieLens na mão, o contador
    # de id fica parado no 1. Aqui ajusto pra ele continuar depois do maior id.
    if db.get_bind().dialect.name != "postgresql":
        return
    for table in ("users", "items", "ratings"):
        db.execute(text(
            f"SELECT setval(pg_get_serial_sequence('{table}', 'id'), "
            f"COALESCE((SELECT MAX(id) FROM {table}), 0) + 1, false)"
        ))


def seed_database(db: Session, data_dir: Path = DATA_DIR, try_download: bool = DOWNLOAD_DATA) -> dict:
    # se o banco já tem filme, não faz nada (pode reiniciar sem duplicar)
    if db.scalar(select(func.count(Item.id))):
        return {"seeded": False, "reason": "banco já populado"}

    folder = data_dir / "ml-latest-small"
    if try_download and not (folder / "ratings.csv").exists():
        try:
            download_movielens(data_dir)
        except Exception as exc:
            # sem internet segue com a base pequena
            logger.warning("Não foi possível baixar o MovieLens (%s). Usando dados locais.", exc)

    folder = find_dataset(data_dir)
    if folder is None:
        logger.warning("Nenhum dataset encontrado em %s; banco começa vazio.", data_dir)
        return {"seeded": False, "reason": "dataset não encontrado"}

    items, ratings = load_movielens(folder)
    users = sorted(ratings["user_id"].unique())

    items_records = [
        {**r, "year": None if pd.isna(r["year"]) else int(r["year"])}
        for r in items.to_dict("records")
    ]
    _bulk_insert(db, Item, items_records)
    # o MovieLens não tem nome de usuário, então crio um genérico
    _bulk_insert(db, User, [
        {"id": int(u), "name": f"Usuário MovieLens {u}", "preferred_genres": "", "disliked_genres": ""}
        for u in users
    ])
    ratings_records = ratings.to_dict("records")
    for r in ratings_records:
        r["timestamp"] = r["timestamp"].to_pydatetime()
    _bulk_insert(db, Rating, ratings_records)
    _reset_sequences(db)
    db.commit()

    summary = {"seeded": True, "source": str(folder), "items": len(items),
               "users": len(users), "ratings": len(ratings)}
    logger.info("Banco populado: %s", summary)
    return summary
