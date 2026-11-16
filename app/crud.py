"""Funções que mexem no banco e convertem os dados pro formato da API."""

from __future__ import annotations

from collections.abc import Iterable

from fastapi import HTTPException, status
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app import schemas
from app.models import Item, Rating, User, utcnow
from app.recommender import split_genres


# ------------------------------ Gêneros ------------------------------------
def join_genres(genres: Iterable[str]) -> str:
    # ["Action", "Sci-Fi"] vira "Action|Sci-Fi" (tira repetidos e vazios)
    seen: dict[str, None] = {}
    for g in genres:
        g = g.strip()
        if g and "|" not in g:
            seen.setdefault(g, None)
    return "|".join(seen)


def canonical_genres(names: Iterable[str], known: Iterable[str], strict: bool) -> list[str]:
    """
    Arruma maiúscula/minúscula: "sci-fi" vira "Sci-Fi".
    strict=True (preferências do usuário): gênero que não existe dá erro 422.
    strict=False (cadastro de filme): aceita gênero novo.
    """
    lookup = {g.lower(): g for g in known}
    result, unknown = [], []
    for name in names:
        name = name.strip()
        if not name:
            continue
        if name.lower() in lookup:
            result.append(lookup[name.lower()])
        elif strict:
            unknown.append(name)
        else:
            result.append(name)
    if unknown:
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail={"msg": f"Gêneros desconhecidos: {unknown}", "valid_genres": sorted(lookup.values())},
        )
    return list(dict.fromkeys(result))


# ------------------------------ Conversões ---------------------------------
def item_out(item: Item) -> schemas.ItemOut:
    return schemas.ItemOut(id=item.id, title=item.title, genres=split_genres(item.genres), year=item.year)


def count_ratings(db: Session, user_id: int) -> int:
    return db.scalar(select(func.count(Rating.id)).where(Rating.user_id == user_id)) or 0


def user_out(db: Session, user: User) -> schemas.UserOut:
    return schemas.UserOut(
        id=user.id,
        name=user.name,
        preferred_genres=split_genres(user.preferred_genres),
        disliked_genres=split_genres(user.disliked_genres),
        n_ratings=count_ratings(db, user.id),
        created_at=user.created_at,
    )


# ------------------------------- Consultas ---------------------------------
def get_user_or_404(db: Session, user_id: int) -> User:
    user = db.get(User, user_id)
    if user is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"Usuário {user_id} não encontrado")
    return user


def get_item_or_404(db: Session, item_id: int) -> Item:
    item = db.get(Item, item_id)
    if item is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"Item {item_id} não encontrado")
    return item


def user_ratings_dict(db: Session, user_id: int) -> dict[int, float]:
    # notas do usuário no formato que o modelo espera: {id_do_filme: nota}
    rows = db.execute(select(Rating.item_id, Rating.rating).where(Rating.user_id == user_id))
    return {item_id: rating for item_id, rating in rows}


def upsert_rating(db: Session, user_id: int, item_id: int, value: float) -> tuple[Rating, bool]:
    # se já tem nota pra esse filme atualiza, senão cria. Retorna (nota, se_criou)
    rating = db.scalar(select(Rating).where(Rating.user_id == user_id, Rating.item_id == item_id))
    created = rating is None
    if created:
        rating = Rating(user_id=user_id, item_id=item_id, rating=value)
        db.add(rating)
    else:
        rating.rating = value
        rating.timestamp = utcnow()
    db.commit()
    db.refresh(rating)
    return rating, created
