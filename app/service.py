"""
Liga o modelo com o banco.

O modelo fica carregado na memória e esse arquivo decide quando retreinar:
- cadastrou filme novo: marca pra retreinar na próxima consulta
  (o filme precisa entrar na matriz do modelo)
- deu nota: NÃO precisa retreinar na hora, porque o fold-in já usa as notas
  atuais do usuário direto do banco. Só depois de 1000 notas novas eu retreino
  pra atualizar o SVD e a popularidade.
"""

from __future__ import annotations

import logging
import threading
from datetime import datetime, timezone

import pandas as pd
from sqlalchemy import select
from sqlalchemy.orm import Session

from app import config
from app.models import Item, Rating
from app.recommender import HybridRecommender, RecommenderConfig

logger = logging.getLogger(__name__)


class CatalogEmptyError(RuntimeError):
    # não tem nenhum filme no banco
    pass


class RecommenderService:
    def __init__(self, model_config: RecommenderConfig | None = None, retrain_every: int = 1000):
        self.model_config = model_config or RecommenderConfig()
        self.retrain_every = retrain_every
        self._model: HybridRecommender | None = None
        # lock pra duas requisições não retreinarem ao mesmo tempo
        self._lock = threading.RLock()
        self._stale = True
        self.version = 0
        self.trained_at: datetime | None = None
        self.ratings_since_training = 0

    def mark_stale(self) -> None:
        # chamado quando cadastra filme novo
        self._stale = True

    def register_rating_change(self) -> None:
        # chamado a cada nota nova, alterada ou removida
        self.ratings_since_training += 1
        if self.ratings_since_training >= self.retrain_every:
            self._stale = True

    def retrain(self, db: Session) -> HybridRecommender:
        with self._lock:
            # lê os filmes e as notas do banco e treina o modelo
            conn = db.connection()
            items = pd.read_sql(select(Item.id, Item.title, Item.genres, Item.year), conn)
            ratings = pd.read_sql(select(Rating.user_id, Rating.item_id, Rating.rating), conn)
            if items.empty:
                raise CatalogEmptyError("Nenhum item cadastrado.")

            model = HybridRecommender(self.model_config).fit(items, ratings)
            self._model = model
            self._stale = False
            self.version += 1
            self.trained_at = datetime.now(timezone.utc)
            self.ratings_since_training = 0
            logger.info("Modelo v%s treinado: %s itens, %s notas",
                        self.version, len(items), len(ratings))
            return model

    def get_model(self, db: Session) -> HybridRecommender:
        # devolve o modelo, retreinando antes se precisar
        with self._lock:
            if self._model is None or self._stale:
                return self.retrain(db)
            return self._model

    def status(self) -> dict:
        m = self._model
        return {
            "trained": m is not None,
            "version": self.version,
            "trained_at": self.trained_at,
            "n_items": len(m.item_ids) if m else 0,
            "n_users": m.n_users if m else 0,
            "n_ratings": m.n_ratings if m else 0,
            "ratings_since_training": self.ratings_since_training,
            "stale": self._stale,
        }

    def reset(self) -> None:
        # usado nos testes pra começar do zero
        with self._lock:
            self._model = None
            self._stale = True
            self.version = 0
            self.trained_at = None
            self.ratings_since_training = 0


# uma instância só, usada pela API inteira
recommender_service = RecommenderService(
    RecommenderConfig(
        n_components=config.SVD_COMPONENTS,
        cf_full_weight_at=config.CF_FULL_WEIGHT_AT,
        popularity_prior=config.POPULARITY_PRIOR,
    )
)
