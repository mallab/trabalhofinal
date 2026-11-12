"""
Configurações do projeto, lidas das variáveis de ambiente.

Tudo tem valor padrão, então roda local sem precisar de .env.
No Docker o docker-compose.yml troca o DATABASE_URL pro Postgres.
"""

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

# local usa SQLite, no Docker usa Postgres
DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{BASE_DIR / 'recsys.db'}")

# pasta dos CSVs do MovieLens
DATA_DIR = Path(os.getenv("DATA_DIR", str(BASE_DIR / "data")))

# popula o banco com o MovieLens na primeira vez
AUTO_SEED = os.getenv("AUTO_SEED", "true").lower() == "true"

# tenta baixar o MovieLens se ele não estiver na pasta data
DOWNLOAD_DATA = os.getenv("DOWNLOAD_DATA", "true").lower() == "true"

# parâmetros do modelo (explicados no docs/MODELO.md)
SVD_COMPONENTS = int(os.getenv("SVD_COMPONENTS", "20"))
CF_FULL_WEIGHT_AT = int(os.getenv("CF_FULL_WEIGHT_AT", "20"))
POPULARITY_PRIOR = int(os.getenv("POPULARITY_PRIOR", "10"))
