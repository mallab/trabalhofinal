"""Rotas dos filmes: cadastro, busca, detalhes e filmes parecidos."""

from fastapi import APIRouter, Depends, Query, status
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app import crud, schemas
from app.database import get_db
from app.models import Item, Rating
from app.service import recommender_service

router = APIRouter(tags=["Itens"])


@router.post("/items", response_model=schemas.ItemOut, status_code=status.HTTP_201_CREATED,
             summary="Cadastrar novo item")
def create_item(payload: schemas.ItemCreate, db: Session = Depends(get_db)):
    """Cadastra um filme novo. Mesmo sem nenhuma nota ele já pode ser
    recomendado pela parte de conteúdo (gêneros + década)."""
    known = recommender_service.get_model(db).known_genres
    item = Item(
        title=payload.title.strip(),
        genres=crud.join_genres(crud.canonical_genres(payload.genres, known, strict=False)),
        year=payload.year,
    )
    db.add(item)
    db.commit()
    db.refresh(item)
    recommender_service.mark_stale()  # o filme novo precisa entrar no modelo
    return crud.item_out(item)


@router.get("/items", response_model=list[schemas.ItemOut], summary="Listar / buscar itens")
def list_items(
    search: str | None = Query(None, description="Trecho do título (sem diferenciar maiúsculas)"),
    genre: str | None = Query(None, description="Filtra por gênero, ex.: Comedy"),
    limit: int = Query(20, ge=1, le=200),
    offset: int = Query(0, ge=0),
    db: Session = Depends(get_db),
):
    stmt = select(Item)
    if search:
        stmt = stmt.where(func.lower(Item.title).contains(search.lower()))
    if genre:
        # procura "|Action|" dentro de "|Action|Sci-Fi|" pra não pegar pedaço de outro gênero
        stmt = stmt.where(("|" + Item.genres + "|").contains(f"|{genre}|"))
    stmt = stmt.order_by(Item.id).limit(limit).offset(offset)
    return [crud.item_out(i) for i in db.scalars(stmt)]


@router.get("/items/{item_id}", response_model=schemas.ItemDetail, summary="Detalhes do item")
def get_item(item_id: int, db: Session = Depends(get_db)):
    item = crud.get_item_or_404(db, item_id)
    n, avg = db.execute(
        select(func.count(Rating.id), func.avg(Rating.rating)).where(Rating.item_id == item_id)
    ).one()
    return schemas.ItemDetail(**crud.item_out(item).model_dump(), n_ratings=n,
                              avg_rating=round(float(avg), 2) if avg is not None else None)


@router.get("/items/{item_id}/similar", response_model=list[schemas.SimilarItem],
            summary="Itens parecidos")
def similar_items(item_id: int, n: int = Query(10, ge=1, le=50), db: Session = Depends(get_db)):
    """Filmes parecidos com esse. Junta gênero parecido com o que o SVD aprendeu
    das notas (quem gostou de um também gostou do outro)."""
    crud.get_item_or_404(db, item_id)
    model = recommender_service.get_model(db)
    pairs = model.similar_items(item_id, n)
    items = {i.id: i for i in db.scalars(select(Item).where(Item.id.in_([p for p, _ in pairs])))}
    return [schemas.SimilarItem(item=crud.item_out(items[p]), similarity=s)
            for p, s in pairs if p in items]


@router.get("/genres", response_model=list[str], summary="Gêneros disponíveis")
def list_genres(db: Session = Depends(get_db)):
    return recommender_service.get_model(db).known_genres
