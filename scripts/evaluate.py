"""
Compara as 4 estratégias no MovieLens.

Como rodar: python -m scripts.evaluate
Opções: --k 10 (tamanho da lista) e --users 0 (0 = todos os usuários)
"""

import argparse
import sys

from app.config import DATA_DIR
from app.evaluation import evaluate
from app.seed import find_dataset, load_movielens


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--k", type=int, default=10, help="tamanho da lista (top K)")
    parser.add_argument("--users", type=int, default=300, help="quantos usuários avaliar (0 = todos)")
    parser.add_argument("--min-ratings", type=int, default=20)
    args = parser.parse_args()

    folder = find_dataset(DATA_DIR)
    if folder is None:
        print("Dataset não encontrado. Rode: python -m scripts.download_data", file=sys.stderr)
        return 1

    items, ratings = load_movielens(folder)
    print(f"Dataset: {folder}  ({len(items)} itens, {len(ratings)} notas)\n")
    report = evaluate(items, ratings, k=args.k, max_users=args.users or None,
                      min_ratings=args.min_ratings)
    print(report.to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
