"""
Configuração dos testes.

As variáveis de ambiente precisam ser definidas ANTES de importar a API, pra
os testes usarem um banco temporário e a base pequena de data/sample
(assim roda rápido e sem internet).
"""

import os
import tempfile
from pathlib import Path

SAMPLE_DIR = Path(__file__).resolve().parent.parent / "data" / "sample"
_tmp = tempfile.mkdtemp(prefix="recsys-test-")
os.environ["DATABASE_URL"] = f"sqlite:///{_tmp}/test.db"
os.environ["DATA_DIR"] = str(SAMPLE_DIR)
os.environ["DOWNLOAD_DATA"] = "false"
os.environ["AUTO_SEED"] = "true"

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from app.recommender import HybridRecommender  # noqa: E402
from app.seed import load_movielens  # noqa: E402


@pytest.fixture(scope="session")
def sample_data():
    # base pequena: 40 filmes e 32 usuários divididos em 4 grupos de gosto
    items, ratings = load_movielens(SAMPLE_DIR)
    return items, ratings[["user_id", "item_id", "rating"]]


@pytest.fixture(scope="session")
def model(sample_data):
    items, ratings = sample_data
    return HybridRecommender().fit(items, ratings)


@pytest.fixture(scope="session")
def client():
    # cliente que faz requisições pra API (já sobe o banco e treina o modelo)
    from app.main import app

    with TestClient(app) as c:
        yield c


# ids dos filmes da base pequena (data/sample/movies.csv)
SCIFI = set(range(1, 11))  # Matrix, Star Wars, Terminator 2, ...
KIDS = set(range(11, 19))  # Toy Story, Procurando Nemo, Shrek, ...
ROMANCE = set(range(19, 27))
CRIME = set(range(27, 36))
SHAWSHANK = 29
TOY_STORY = 11
