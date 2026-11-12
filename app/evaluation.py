"""
Avaliação offline do modelo.

Pra cada usuário com pelo menos 20 notas, escondo 20% delas (teste) e treino
com o resto. Depois vejo se os filmes escondidos que ele curtiu (nota >= 4)
aparecem no top 10.

Métricas:
- precision@K: quantos do top K eram filmes que ele curtiu
- recall@K: quantos dos filmes escondidos apareceram no top K
- hit_rate@K: % de usuários com pelo menos 1 acerto
- coverage: % do catálogo que apareceu em alguma recomendação
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from app.recommender import STRATEGIES, HybridRecommender, RecommenderConfig


def train_test_split_by_user(
    ratings: pd.DataFrame, test_frac: float = 0.2, min_ratings: int = 20, seed: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    counts = ratings["user_id"].value_counts()
    # só entra no teste quem tem notas suficientes
    eligible = ratings["user_id"].isin(counts[counts >= min_ratings].index)
    is_test = np.zeros(len(ratings), dtype=bool)
    is_test[eligible.to_numpy()] = rng.random(int(eligible.sum())) < test_frac
    return ratings[~is_test], ratings[is_test]


def evaluate(
    items: pd.DataFrame,
    ratings: pd.DataFrame,
    k: int = 10,
    strategies: tuple[str, ...] = STRATEGIES,
    config: RecommenderConfig | None = None,
    max_users: int | None = 300,
    min_ratings: int = 20,
    seed: int = 42,
) -> pd.DataFrame:
    train, test = train_test_split_by_user(ratings, min_ratings=min_ratings, seed=seed)
    model = HybridRecommender(config).fit(items, train)

    # filmes escondidos que o usuário gostou = o que o modelo deveria acertar
    relevant = test[test["rating"] >= model.config.like_threshold]
    relevant_by_user = relevant.groupby("user_id")["item_id"].apply(set)
    users = relevant_by_user.index.to_numpy()
    if len(users) == 0:
        raise ValueError("Nenhum usuário elegível para avaliação (reduza min_ratings).")
    # pra rodar mais rápido, uso uma amostra de usuários
    if max_users and len(users) > max_users:
        users = np.random.default_rng(seed).choice(users, max_users, replace=False)
    train_by_user = {
        u: dict(zip(g["item_id"], g["rating"])) for u, g in train.groupby("user_id")
    }

    rows = []
    for strategy in strategies:
        precisions, recalls, hits, shown = [], [], [], set()
        for user in users:
            result = model.recommend(train_by_user.get(user, {}), n=k, strategy=strategy)
            recs = [r.item_id for r in result.items]
            shown.update(recs)
            n_hit = len(set(recs) & relevant_by_user[user])
            precisions.append(n_hit / k)
            recalls.append(n_hit / len(relevant_by_user[user]))
            hits.append(n_hit > 0)
        rows.append({
            "estrategia": strategy,
            f"precision@{k}": round(float(np.mean(precisions)), 4),
            f"recall@{k}": round(float(np.mean(recalls)), 4),
            f"hit_rate@{k}": round(float(np.mean(hits)), 4),
            "coverage": round(len(shown) / len(model.item_ids), 4),
            "usuarios": len(users),
        })
    return pd.DataFrame(rows)
