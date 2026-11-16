"""Rotas de usuários: cadastro, preferências, notas e recomendações."""

from fastapi import APIRouter, Depends, HTTPException, Query, Response, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from app import crud, schemas
from app.database import get_db
from app.models import Item, Rating, User
from app.recommender import split_genres
from app.service import recommender_service

router = APIRouter(prefix="/users", tags=["Usuários"])


def _known_genres(db: Session) -> list[str]:
    # lista de gêneros que existem no catálogo, pra validar as preferências
    return recommender_service.get_model(db).known_genres


@router.post("", response_model=schemas.UserOut, status_code=status.HTTP_201_CREATED,
             summary="Cadastrar novo usuário")
def create_user(payload: schemas.UserCreate, db: Session = Depends(get_db)):
    """Cria um usuário. Se ele informar gêneros favoritos, já recebe
    recomendações personalizadas antes de dar a primeira nota."""
    known = _known_genres(db)
    user = User(
        name=payload.name.strip(),
        preferred_genres=crud.join_genres(crud.canonical_genres(payload.preferred_genres, known, strict=True)),
        disliked_genres=crud.join_genres(crud.canonical_genres(payload.disliked_genres, known, strict=True)),
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return crud.user_out(db, user)


@router.get("/{user_id}", response_model=schemas.UserOut, summary="Consultar usuário")
def get_user(user_id: int, db: Session = Depends(get_db)):
    return crud.user_out(db, crud.get_user_or_404(db, user_id))


@router.put("/{user_id}/preferences", response_model=schemas.UserOut,
            summary="Atualizar preferências de gênero")
def update_preferences(user_id: int, payload: schemas.PreferencesUpdate, db: Session = Depends(get_db)):
    """Atualiza os gêneros favoritos e/ou os gêneros que o usuário não quer ver.
    O campo que não for enviado continua igual."""
    user = crud.get_user_or_404(db, user_id)
    known = _known_genres(db)
    if payload.preferred_genres is not None:
        user.preferred_genres = crud.join_genres(
            crud.canonical_genres(payload.preferred_genres, known, strict=True))
    if payload.disliked_genres is not None:
        user.disliked_genres = crud.join_genres(
            crud.canonical_genres(payload.disliked_genres, known, strict=True))
    db.commit()
    return crud.user_out(db, user)


@router.get("/{user_id}/ratings", response_model=list[schemas.RatedItem],
            summary="Listar avaliações do usuário")
def list_ratings(user_id: int, limit: int = Query(50, ge=1, le=500), db: Session = Depends(get_db)):
    crud.get_user_or_404(db, user_id)
    rows = db.execute(
        select(Rating, Item).join(Item, Rating.item_id == Item.id)
        .where(Rating.user_id == user_id)
        .order_by(Rating.rating.desc(), Rating.timestamp.desc())
        .limit(limit)
    ).all()
    return [schemas.RatedItem(item=crud.item_out(item), rating=r.rating, timestamp=r.timestamp)
            for r, item in rows]


@router.post("/{user_id}/ratings", response_model=schemas.RatingOut,
             summary="Avaliar item (cria ou atualiza a nota)")
def rate_item(user_id: int, payload: schemas.RatingIn, response: Response, db: Session = Depends(get_db)):
    """Dá nota pra um filme (de 0.5 a 5, de meio em meio). Se o usuário já
    tinha dado nota pra esse filme, ela é atualizada. A próxima recomendação
    já considera a nota nova, sem retreinar o modelo."""
    crud.get_user_or_404(db, user_id)
    crud.get_item_or_404(db, payload.item_id)
    rating, created = crud.upsert_rating(db, user_id, payload.item_id, payload.rating)
    recommender_service.register_rating_change()
    response.status_code = status.HTTP_201_CREATED if created else status.HTTP_200_OK
    return rating


@router.delete("/{user_id}/ratings/{item_id}", status_code=status.HTTP_204_NO_CONTENT,
               summary="Remover avaliação")
def delete_rating(user_id: int, item_id: int, db: Session = Depends(get_db)):
    crud.get_user_or_404(db, user_id)
    rating = db.scalar(select(Rating).where(Rating.user_id == user_id, Rating.item_id == item_id))
    if rating is None:
        crud.get_item_or_404(db, item_id)  # se o filme nem existe, dá 404 do filme
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Avaliação não encontrada")
    db.delete(rating)
    db.commit()
    recommender_service.register_rating_change()


@router.get("/{user_id}/recommendations", response_model=schemas.RecommendationResponse,
            summary="Obter recomendações para o usuário")
def recommend(
    user_id: int,
    n: int = Query(10, ge=1, le=100, description="Quantidade de recomendações"),
    strategy: schemas.Strategy = Query("hybrid", description="Algoritmo a usar"),
    db: Session = Depends(get_db),
):
    """Retorna os N melhores filmes pro usuário, com a nota de cada parte do
    modelo e o motivo da recomendação. Filme que ele já avaliou ou de gênero
    que ele marcou como "não gosto" nunca aparece."""
    user = crud.get_user_or_404(db, user_id)
    model = recommender_service.get_model(db)
    result = model.recommend(
        user_ratings=crud.user_ratings_dict(db, user_id),
        preferred_genres=split_genres(user.preferred_genres),
        disliked_genres=split_genres(user.disliked_genres),
        n=n,
        strategy=strategy,
    )
    # busca os dados dos filmes recomendados no banco
    ids = [r.item_id for r in result.items]
    items = {i.id: i for i in db.scalars(select(Item).where(Item.id.in_(ids)))} if ids else {}
    return schemas.RecommendationResponse(
        user_id=user_id,
        strategy_requested=strategy,
        strategy_used=result.strategy,
        weights=result.weights,
        recommendations=[
            schemas.RecommendationOut(item=crud.item_out(items[r.item_id]), score=r.score,
                                      reason=r.reason, components=r.components)
            for r in result.items if r.item_id in items
        ],
    )
