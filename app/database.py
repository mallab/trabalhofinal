"""Conexão com o banco usando SQLAlchemy. Funciona com SQLite e Postgres."""

from collections.abc import Iterator

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from app.config import DATABASE_URL

_is_sqlite = DATABASE_URL.startswith("sqlite")

engine = create_engine(
    DATABASE_URL,
    # o SQLite precisa disso pra funcionar com várias threads (o FastAPI usa)
    connect_args={"check_same_thread": False} if _is_sqlite else {},
    pool_pre_ping=True,
)

SessionLocal = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)


class Base(DeclarativeBase):
    pass


def get_db() -> Iterator[Session]:
    # abre uma sessão por requisição e fecha no final
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
