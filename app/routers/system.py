"""Rotas de sistema: status da API e do modelo."""

from fastapi import APIRouter, Depends
from sqlalchemy import text
from sqlalchemy.orm import Session

from app import schemas
from app.database import get_db
from app.service import recommender_service

router = APIRouter(tags=["Sistema"])


@router.get("/health", summary="Health check")
def health(db: Session = Depends(get_db)):
    db.execute(text("SELECT 1"))
    return {"status": "ok", "database": "ok", "model_trained": recommender_service.status()["trained"]}


@router.get("/model/status", response_model=schemas.ModelStatus, summary="Status do modelo")
def model_status():
    return recommender_service.status()


@router.post("/model/retrain", response_model=schemas.ModelStatus, summary="Retreinar o modelo")
def retrain(db: Session = Depends(get_db)):
    """Retreina o modelo na hora com os dados atuais do banco."""
    recommender_service.retrain(db)
    return recommender_service.status()
