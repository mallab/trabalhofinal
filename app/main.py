"""
Arquivo principal da API.

Pra rodar local: uvicorn app.main:app --reload
Documentação (Swagger): http://localhost:8000/docs
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, RedirectResponse

from app import __version__, config
from app.database import Base, SessionLocal, engine
from app.routers import items, system, users
from app.seed import seed_database
from app.service import CatalogEmptyError, recommender_service

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("recsys")


@asynccontextmanager
async def lifespan(_: FastAPI):
    # roda quando a API sobe: cria as tabelas, carrega o MovieLens e treina o modelo
    Base.metadata.create_all(bind=engine)
    with SessionLocal() as db:
        if config.AUTO_SEED:
            seed_database(db)
        try:
            recommender_service.retrain(db)
        except CatalogEmptyError:
            logger.warning("Catálogo vazio: cadastre itens via POST /items.")
    yield


# esse texto aparece no topo do Swagger
DESCRIPTION = """
API de recomendação de filmes usando o dataset MovieLens.

O modelo é híbrido e junta 3 coisas:
- **Filtragem colaborativa**: SVD com fold-in (atualiza na hora, sem precisar retreinar)
- **Filtragem por conteúdo**: TF-IDF dos gêneros + década do filme
- **Popularidade**: média bayesiana, usada pra quem ainda não tem nota (cold start)

O peso de cada parte muda conforme o histórico do usuário e cada recomendação
vem com o motivo em português.

**Ordem pra testar:** `POST /users` → `POST /users/{id}/ratings` →
`GET /users/{id}/recommendations`
"""

app = FastAPI(
    title="Sistema de Recomendação de Filmes",
    description=DESCRIPTION,
    version=__version__,
    lifespan=lifespan,
    openapi_tags=[
        {"name": "Usuários", "description": "Cadastro, preferências, avaliações e recomendações"},
        {"name": "Itens", "description": "Catálogo de filmes"},
        {"name": "Sistema", "description": "Status da API e do modelo"},
    ],
)


@app.exception_handler(CatalogEmptyError)
async def catalog_empty_handler(_: Request, exc: CatalogEmptyError):
    # se não tiver nenhum filme cadastrado, devolve 503 com uma mensagem clara
    return JSONResponse(status_code=503, content={"detail": f"{exc} Cadastre itens via POST /items."})


@app.get("/", include_in_schema=False)
def root():
    # quem abrir a raiz cai direto no Swagger
    return RedirectResponse(url="/docs")


app.include_router(users.router)
app.include_router(items.router)
app.include_router(system.router)
