"""
Modelo de recomendação híbrido.

A ideia é juntar 3 formas de recomendar e somar as notas de cada uma:

1. Colaborativa: usa SVD na matriz usuário x filme. Pra recomendar, a gente
   projeta as notas do usuário no espaço do SVD (fold-in). Assim não precisa
   retreinar quando o usuário avalia um filme novo.
2. Conteúdo: TF-IDF com os gêneros do filme + a década. O perfil do usuário
   é a média dos filmes que ele avaliou (pesando pela nota).
3. Popularidade: média bayesiana, igual ao ranking do IMDb. Serve pra quem
   ainda não tem nenhuma nota (cold start).

O peso de cada parte muda conforme o usuário tem mais ou menos notas.
Mais detalhes no docs/MODELO.md.

Esse arquivo não mexe com banco nem com a API, só recebe DataFrames.
Fiz assim pra ficar fácil de testar separado.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse.linalg import norm as sparse_norm
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

NO_GENRE = "(no genres listed)"
STRATEGIES = ("hybrid", "collaborative", "content", "popular")


# ---------------------------------------------------------------------------
# Funções de apoio
# ---------------------------------------------------------------------------
def split_genres(genres: str | None) -> list[str]:
    # No MovieLens os gêneros vêm como "Action|Sci-Fi"
    if not genres or not isinstance(genres, str):
        return []
    return [g.strip() for g in genres.split("|") if g.strip() and g.strip() != NO_GENRE]


def decade_token(year: float | int | None) -> str | None:
    # 1994 vira "decada_1990". Entra no TF-IDF como se fosse mais um gênero
    if year is None or pd.isna(year):
        return None
    return f"decada_{int(year) // 10 * 10}"


def _identity(tokens: list[str]) -> list[str]:
    # o TfidfVectorizer já recebe a lista de tokens pronta
    return tokens


def _minmax(values: np.ndarray | None) -> np.ndarray | None:
    # coloca tudo entre 0 e 1 pra poder somar os sinais
    if values is None:
        return None
    lo, hi = float(values.min()), float(values.max())
    if hi - lo < 1e-12:
        return np.zeros_like(values, dtype=float)
    return (values - lo) / (hi - lo)


# ---------------------------------------------------------------------------
# Configuração e saída
# ---------------------------------------------------------------------------
@dataclass
class RecommenderConfig:
    n_components: int = 20  # dimensões do SVD (testei 20, 50 e 100, 20 foi melhor)
    cf_full_weight_at: int = 20  # com 20 notas a colaborativa chega no peso máximo
    max_cf_weight: float = 0.7  # peso máximo da colaborativa
    popularity_weight: float = 0.1  # peso mínimo da popularidade
    cold_start_popularity_boost: float = 0.2  # peso extra de popularidade pra quem tem poucas notas
    popularity_prior: int = 10  # o "C" da média bayesiana
    mean_shrinkage: int = 5  # puxa a média do usuário pra média geral quando ele tem poucas notas
    genre_preference_weight: float = 2.0  # força dos gêneros favoritos no perfil
    like_threshold: float = 4.0  # a partir de 4 considero que o usuário gostou
    neutral_rating: float = 3.0  # nota 3 não puxa o perfil pra lado nenhum
    random_state: int = 42


@dataclass
class Recommendation:
    item_id: int
    score: float
    reason: str
    components: dict[str, float] = field(default_factory=dict)


@dataclass
class RecommendationResult:
    strategy: str  # estratégia que foi usada de fato (pode cair pra "popular")
    weights: dict[str, float]
    items: list[Recommendation]


# ---------------------------------------------------------------------------
# Modelo
# ---------------------------------------------------------------------------
class HybridRecommender:

    def __init__(self, config: RecommenderConfig | None = None):
        self.config = config or RecommenderConfig()
        self.is_fitted = False

    # ----------------------------- treino ---------------------------------
    def fit(self, items: pd.DataFrame, ratings: pd.DataFrame) -> HybridRecommender:
        """
        items: colunas id, genres (e title/year se tiver)
        ratings: colunas user_id, item_id, rating
        """
        if items.empty:
            raise ValueError("O catálogo de itens está vazio.")

        items = items.reset_index(drop=True)
        self.item_ids = items["id"].astype(int).to_numpy()
        # dicionário pra achar a posição do filme na matriz pelo id
        self.item_pos = {int(item_id): pos for pos, item_id in enumerate(self.item_ids)}
        self.titles = (
            items["title"].fillna("").tolist() if "title" in items else [""] * len(items)
        )
        self.item_genres = [split_genres(g) for g in items["genres"].fillna("")]
        self.known_genres = sorted({g for gs in self.item_genres for g in gs})
        years = items["year"] if "year" in items else pd.Series([None] * len(items))

        # ignora notas de filmes que não estão no catálogo
        ratings = ratings[ratings["item_id"].isin(self.item_pos.keys())]
        self.global_mean = float(ratings["rating"].mean()) if len(ratings) else 3.0

        self._fit_content(years)
        self._fit_popularity(ratings)
        self._fit_collaborative(ratings)

        self.n_ratings = len(ratings)
        self.n_users = int(ratings["user_id"].nunique()) if len(ratings) else 0
        self.is_fitted = True
        return self

    def _fit_content(self, years: pd.Series) -> None:
        # cada filme vira uma lista: gêneros + década
        docs = []
        for genres, year in zip(self.item_genres, years):
            tokens = list(genres)
            decade = decade_token(year)
            if decade:
                tokens.append(decade)
            docs.append(tokens)

        self.vectorizer = TfidfVectorizer(analyzer=_identity, lowercase=False)
        try:
            # as linhas já saem normalizadas, então produto escalar = cosseno
            self.item_matrix = self.vectorizer.fit_transform(docs).tocsr()
        except ValueError:
            # acontece se nenhum filme tiver gênero nem ano
            self.vectorizer = None
            self.item_matrix = sparse.csr_matrix((len(docs), 1))

    def _fit_popularity(self, ratings: pd.DataFrame) -> None:
        n_items = len(self.item_ids)
        self.item_counts = np.zeros(n_items)
        self.item_means = np.full(n_items, np.nan)
        if len(ratings):
            stats = ratings.groupby("item_id")["rating"].agg(["count", "mean"])
            pos = stats.index.map(self.item_pos).to_numpy()
            self.item_counts[pos] = stats["count"].to_numpy()
            self.item_means[pos] = stats["mean"].to_numpy()

        # média bayesiana: (n * média + C * média_geral) / (n + C)
        # filme com 1 nota 5 não vai pro topo, ele é puxado pra média geral
        c, m = self.config.popularity_prior, self.global_mean
        sums = np.nan_to_num(self.item_means) * self.item_counts
        self.weighted_rating = (sums + c * m) / (self.item_counts + c)

        # misturei com a quantidade de notas porque só com a média os filmes
        # sem nenhuma nota ficavam acima de muito filme bom
        self.popularity = 0.7 * _minmax(self.weighted_rating) + 0.3 * _minmax(np.log1p(self.item_counts))

    def _user_mean(self, ratings: Iterable[float]) -> float:
        # média do usuário puxada pra média geral (fica estável com poucas notas)
        values = list(ratings)
        k = self.config.mean_shrinkage
        return (sum(values) + k * self.global_mean) / (len(values) + k)

    def _fit_collaborative(self, ratings: pd.DataFrame) -> None:
        self.Vt = None
        self.item_factors = None
        if ratings.empty:
            return

        user_codes, _ = pd.factorize(ratings["user_id"])
        item_pos = ratings["item_id"].map(self.item_pos).to_numpy()

        # tira a média de cada usuário das notas dele
        # (quem dá 5 pra tudo não pode distorcer o modelo)
        k = self.config.mean_shrinkage
        grp = ratings.groupby("user_id")["rating"]
        user_mean = (grp.transform("sum") + k * self.global_mean) / (grp.transform("count") + k)
        centered = (ratings["rating"] - user_mean).to_numpy()

        n_users = int(user_codes.max()) + 1
        matrix = sparse.csr_matrix(
            (centered, (user_codes, item_pos)), shape=(n_users, len(self.item_ids))
        )

        # limito o número de dimensões pelo tamanho dos dados. Na base pequena
        # de teste, com 20 dimensões o SVD "decorava" a matriz e errava tudo.
        # No MovieLens o limite dá 76, então fica o 20 da config mesmo.
        n_comp = min(self.config.n_components, max(1, min(matrix.shape) // 8), min(matrix.shape) - 1)
        if n_comp < 1:
            return

        svd = TruncatedSVD(n_components=n_comp, random_state=self.config.random_state)
        svd.fit(matrix)
        self.Vt = svd.components_  # (k, n_filmes)
        # fatores dos filmes, usados pra achar filmes parecidos
        factors = (svd.components_ * svd.singular_values_[:, None]).T
        norms = np.linalg.norm(factors, axis=1, keepdims=True)
        self.item_factors = np.divide(factors, norms, out=np.zeros_like(factors), where=norms > 0)

    # ---------------------- notas de cada parte ----------------------------
    def _collaborative_scores(self, ratings: dict[int, float]) -> np.ndarray | None:
        # fold-in: projeta as notas do usuário no SVD e volta
        if self.Vt is None or not ratings:
            return None
        mean = self._user_mean(ratings.values())
        r = np.zeros(len(self.item_ids))
        for item_id, value in ratings.items():
            r[self.item_pos[item_id]] = value - mean
        if not r.any():
            return None
        latent = self.Vt @ r
        return self.Vt.T @ latent

    def _genre_vector(self, genres: Iterable[str]) -> sparse.csr_matrix | None:
        genres = list(genres)
        if not genres or self.vectorizer is None:
            return None
        return self.vectorizer.transform([genres])

    def _content_scores(
        self, ratings: dict[int, float], preferred_genres: Iterable[str]
    ) -> np.ndarray | None:
        n_features = self.item_matrix.shape[1]
        profile = sparse.csr_matrix((1, n_features))

        # perfil = soma dos filmes avaliados * (nota - 3)
        # nota abaixo de 3 empurra o perfil pra longe daquele tipo de filme
        if ratings:
            pos = [self.item_pos[i] for i in ratings]
            weights = np.array([r - self.config.neutral_rating for r in ratings.values()])
            profile = profile + sparse.csr_matrix(weights) @ self.item_matrix[pos]

        # soma os gêneros favoritos que o usuário escolheu
        prefs = self._genre_vector(preferred_genres)
        if prefs is not None:
            profile = profile + self.config.genre_preference_weight * prefs

        norm = sparse_norm(profile)
        if norm == 0:
            return None
        # cosseno entre o perfil e cada filme
        return np.asarray((self.item_matrix @ (profile / norm).T).todense()).ravel()

    def _weights(self, strategy: str, n_ratings: int, has_cf: bool, has_content: bool) -> dict:
        # decide o peso de cada parte. Se faltar informação, cai pra popularidade
        cfg = self.config
        if strategy == "collaborative" and has_cf:
            return {"collaborative": 1.0}
        if strategy == "content" and has_content:
            # popularidade só pra desempatar (tem muito filme com os mesmos gêneros)
            return {"content": 0.9, "popular": 0.1}
        if strategy == "hybrid" and (has_cf or has_content):
            maturity = min(1.0, n_ratings / cfg.cf_full_weight_at)  # 0 = novato, 1 = já tem bastante nota
            w_cf = cfg.max_cf_weight * maturity if has_cf else 0.0
            # quem tem pouca nota recebe mais popularidade, pra não sugerir filme obscuro
            w_pop = cfg.popularity_weight + cfg.cold_start_popularity_boost * (1 - maturity)
            w_content = 1.0 - w_cf - w_pop if has_content else 0.0
            if not has_content:
                w_pop = 1.0 - w_cf
            return {k: round(v, 4) for k, v in
                    {"collaborative": w_cf, "content": w_content, "popular": w_pop}.items() if v > 0}
        return {"popular": 1.0}

    # ---------------------------- recomendar -------------------------------
    def recommend(
        self,
        user_ratings: dict[int, float],
        preferred_genres: Iterable[str] = (),
        disliked_genres: Iterable[str] = (),
        n: int = 10,
        strategy: str = "hybrid",
        exclude: Iterable[int] = (),
    ) -> RecommendationResult:
        """
        Retorna os top N filmes pro usuário.
        user_ratings é um dicionário {id_do_filme: nota}, pode ser vazio.
        Nunca recomenda filme que ele já avaliou nem de gênero que ele não gosta.
        """
        if not self.is_fitted:
            raise RuntimeError("Modelo não treinado. Chame fit() antes.")
        if strategy not in STRATEGIES:
            raise ValueError(f"Estratégia inválida: {strategy}. Use uma de {STRATEGIES}.")

        ratings = {int(i): float(r) for i, r in user_ratings.items() if int(i) in self.item_pos}
        preferred = list(preferred_genres)
        disliked = set(disliked_genres)

        signals = {
            "collaborative": _minmax(self._collaborative_scores(ratings)),
            "content": _minmax(self._content_scores(ratings, preferred)),
            "popular": self.popularity,
        }
        weights = self._weights(
            strategy, len(ratings), signals["collaborative"] is not None,
            signals["content"] is not None,
        )
        used_strategy = "popular" if weights == {"popular": 1.0} else strategy

        total = np.zeros(len(self.item_ids))
        for name, w in weights.items():
            total += w * signals[name]

        # tira o que não pode aparecer
        blocked = np.zeros(len(self.item_ids), dtype=bool)
        for item_id in list(ratings) + [int(i) for i in exclude]:
            if item_id in self.item_pos:
                blocked[self.item_pos[item_id]] = True
        if disliked:
            for pos, genres in enumerate(self.item_genres):
                if disliked.intersection(genres):
                    blocked[pos] = True
        total[blocked] = -np.inf

        n = max(0, min(n, int((~blocked).sum())))
        if n == 0:
            return RecommendationResult(used_strategy, weights, [])
        top = np.argpartition(-total, n - 1)[:n]
        top = top[np.argsort(-total[top], kind="stable")]

        items = []
        for pos in top:
            parts = {k: round(float(w * signals[k][pos]), 4) for k, w in weights.items()}
            items.append(Recommendation(
                item_id=int(self.item_ids[pos]),
                score=round(float(total[pos]), 4),
                reason=self._explain(pos, parts, ratings, preferred),
                components=parts,
            ))
        return RecommendationResult(used_strategy, weights, items)

    def _explain(self, pos: int, parts: dict, ratings: dict[int, float], preferred: list[str]) -> str:
        # monta a frase explicando o motivo, olhando qual parte pesou mais
        dominant = max(parts, key=parts.get)
        liked = [self.item_pos[i] for i, r in ratings.items() if r >= self.config.like_threshold]

        if dominant == "collaborative":
            if liked and self.item_factors is not None:
                sims = self.item_factors[liked] @ self.item_factors[pos]
                anchor = liked[int(np.argmax(sims))]
                return f"Quem curtiu '{self.titles[anchor]}' também curtiu este filme"
            return "Usuários com gosto parecido com o seu avaliaram bem este filme"

        if dominant == "content":
            overlap = [g for g in preferred if g in self.item_genres[pos]]
            if liked:
                sims = (self.item_matrix[liked] @ self.item_matrix[pos].T).toarray().ravel()
                if sims.max() >= 0.5:
                    anchor = liked[int(np.argmax(sims))]
                    return f"Parecido com '{self.titles[anchor]}', que você avaliou bem"
            if overlap:
                return f"Combina com seus gêneros favoritos: {', '.join(overlap)}"
            genres = ", ".join(self.item_genres[pos][:3]) or "sem gênero"
            return f"Tem o estilo dos filmes que você curte ({genres})"

        count = int(self.item_counts[pos])
        if count:
            return (f"Popular e bem avaliado no catálogo "
                    f"({count} avaliações, média {self.item_means[pos]:.1f})")
        return "Novidade no catálogo"

    def similar_items(self, item_id: int, n: int = 10) -> list[tuple[int, float]]:
        # filmes parecidos: mistura gêneros (conteúdo) com os fatores do SVD
        if item_id not in self.item_pos:
            raise KeyError(item_id)
        pos = self.item_pos[item_id]

        content = (self.item_matrix @ self.item_matrix[pos].T).toarray().ravel()
        score = 0.9 * content
        if self.item_factors is not None and self.item_counts[pos] > 0:
            # filme com poucas notas tem fator do SVD pouco confiável, então pesa menos
            confidence = self.item_counts / (self.item_counts + 10)
            collab = (self.item_factors @ self.item_factors[pos]) * confidence
            score = 0.5 * collab + 0.4 * content
        score = score + 0.1 * self.popularity
        score[pos] = -np.inf

        n = max(0, min(n, len(score) - 1))
        if n == 0:
            return []
        top = np.argpartition(-score, n - 1)[:n]
        top = top[np.argsort(-score[top], kind="stable")]
        return [(int(self.item_ids[p]), round(float(score[p]), 4)) for p in top]

    def item_stats(self, item_id: int) -> tuple[int, float | None]:
        # quantidade de notas e média do filme (do momento do treino)
        pos = self.item_pos.get(item_id)
        if pos is None or self.item_counts[pos] == 0:
            return 0, None
        return int(self.item_counts[pos]), round(float(self.item_means[pos]), 2)
