"""
Schemas do Pydantic: definem o formato do que entra e sai da API.
O FastAPI usa isso pra validar os dados e montar o Swagger.
"""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

Strategy = Literal["hybrid", "collaborative", "content", "popular"]


# ------------------------------- Usuários ----------------------------------
class UserCreate(BaseModel):
    name: str = Field(min_length=1, max_length=120, examples=["Dudu"])
    preferred_genres: list[str] = Field(default_factory=list, examples=[["Sci-Fi", "Action"]])
    disliked_genres: list[str] = Field(default_factory=list, examples=[["Horror"]])


class PreferencesUpdate(BaseModel):
    # o campo que não for enviado continua como está

    preferred_genres: list[str] | None = Field(default=None, examples=[["Comedy", "Animation"]])
    disliked_genres: list[str] | None = Field(default=None, examples=[["Horror"]])


class UserOut(BaseModel):
    id: int
    name: str
    preferred_genres: list[str]
    disliked_genres: list[str]
    n_ratings: int
    created_at: datetime


# -------------------------------- Itens ------------------------------------
class ItemCreate(BaseModel):
    title: str = Field(min_length=1, max_length=300, examples=["Duna: Parte Dois (2024)"])
    genres: list[str] = Field(default_factory=list, examples=[["Action", "Adventure", "Sci-Fi"]])
    year: int | None = Field(default=None, ge=1870, le=2100, examples=[2024])


class ItemOut(BaseModel):
    id: int
    title: str
    genres: list[str]
    year: int | None = None


class ItemDetail(ItemOut):
    n_ratings: int
    avg_rating: float | None


class SimilarItem(BaseModel):
    item: ItemOut
    similarity: float


# ------------------------------ Avaliações ---------------------------------
class RatingIn(BaseModel):
    item_id: int = Field(examples=[2571])
    rating: float = Field(ge=0.5, le=5.0, multiple_of=0.5, examples=[4.5])


class RatingOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    user_id: int
    item_id: int
    rating: float
    timestamp: datetime


class RatedItem(BaseModel):
    item: ItemOut
    rating: float
    timestamp: datetime


# ---------------------------- Recomendações --------------------------------
class RecommendationOut(BaseModel):
    item: ItemOut
    score: float = Field(description="Score final (0 a 1), soma ponderada dos sinais")
    reason: str = Field(description="Explicação em português do porquê da recomendação")
    components: dict[str, float] = Field(description="Contribuição de cada sinal para o score")


class RecommendationResponse(BaseModel):
    user_id: int
    strategy_requested: Strategy
    strategy_used: Strategy = Field(
        description="Pode cair para 'popular' quando o usuário ainda não tem histórico"
    )
    weights: dict[str, float]
    recommendations: list[RecommendationOut]


# -------------------------------- Sistema ----------------------------------
class ModelStatus(BaseModel):
    trained: bool
    version: int
    trained_at: datetime | None
    n_items: int
    n_users: int
    n_ratings: int
    ratings_since_training: int
    stale: bool
