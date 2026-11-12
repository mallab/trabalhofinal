"""
Baixa o MovieLens small (100 mil notas, 9.742 filmes, 610 usuários).

Como rodar: python -m scripts.download_data
Fonte: https://grouplens.org/datasets/movielens/
"""

import logging
import sys

from app.config import DATA_DIR
from app.seed import download_movielens


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    try:
        folder = download_movielens(DATA_DIR)
    except Exception as exc:
        print(f"Falha ao baixar o MovieLens: {exc}", file=sys.stderr)
        return 1
    print(f"MovieLens disponível em {folder}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
