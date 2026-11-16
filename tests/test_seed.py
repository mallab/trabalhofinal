"""Testes da carga do MovieLens no banco."""

from pathlib import Path

from sqlalchemy import create_engine, func, select
from sqlalchemy.orm import Session

from app.database import Base
from app.models import Item, Rating, User
from app.seed import find_dataset, load_movielens, seed_database
from tests.conftest import SAMPLE_DIR


def test_load_movielens_parses_year_and_genres():
    items, ratings = load_movielens(SAMPLE_DIR)
    matrix = items.set_index("id").loc[1]
    assert matrix["title"] == "Matrix, The (1999)"
    assert matrix["year"] == 1999
    assert matrix["genres"] == "Action|Sci-Fi|Thriller"
    assert not ratings.duplicated(["user_id", "item_id"]).any()
    assert str(ratings["timestamp"].dt.tz) == "UTC"


def test_find_dataset(tmp_path: Path):
    assert find_dataset(tmp_path) is None
    assert find_dataset(SAMPLE_DIR) == SAMPLE_DIR


def test_seed_is_idempotent(tmp_path: Path):
    engine = create_engine(f"sqlite:///{tmp_path / 'seed.db'}")
    Base.metadata.create_all(engine)
    with Session(engine) as db:
        first = seed_database(db, data_dir=SAMPLE_DIR, try_download=False)
        assert first["seeded"] is True
        second = seed_database(db, data_dir=SAMPLE_DIR, try_download=False)
        assert second["seeded"] is False

        assert db.scalar(select(func.count(Item.id))) == first["items"]
        assert db.scalar(select(func.count(User.id))) == first["users"]
        assert db.scalar(select(func.count(Rating.id))) == first["ratings"]

        # usuário novo tem que pegar o id seguinte, sem bater com os do MovieLens
        db.add(User(name="novo"))
        db.commit()
        assert db.scalar(select(func.max(User.id))) == first["users"] + 1


def test_seed_without_dataset(tmp_path: Path):
    engine = create_engine(f"sqlite:///{tmp_path / 'empty.db'}")
    Base.metadata.create_all(engine)
    with Session(engine) as db:
        result = seed_database(db, data_dir=tmp_path / "nada", try_download=False)
        assert result == {"seeded": False, "reason": "dataset não encontrado"}
