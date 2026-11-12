"""
Tabelas do banco.

Os gêneros ficam salvos como texto separado por "|" (ex: "Action|Sci-Fi"),
igual vem no MovieLens. Na API eles viram lista.
"""

from datetime import datetime, timezone

from sqlalchemy import DateTime, Float, ForeignKey, Integer, String, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(120))
    preferred_genres: Mapped[str] = mapped_column(String(500), default="")
    disliked_genres: Mapped[str] = mapped_column(String(500), default="")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    ratings: Mapped[list["Rating"]] = relationship(
        back_populates="user", cascade="all, delete-orphan"
    )


class Item(Base):
    __tablename__ = "items"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    title: Mapped[str] = mapped_column(String(300), index=True)
    genres: Mapped[str] = mapped_column(String(300), default="")
    year: Mapped[int | None] = mapped_column(Integer, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    ratings: Mapped[list["Rating"]] = relationship(
        back_populates="item", cascade="all, delete-orphan"
    )


class Rating(Base):
    __tablename__ = "ratings"
    # um usuário só pode ter uma nota por filme
    __table_args__ = (UniqueConstraint("user_id", "item_id", name="uq_rating_user_item"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), index=True)
    item_id: Mapped[int] = mapped_column(ForeignKey("items.id"), index=True)
    rating: Mapped[float] = mapped_column(Float)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    user: Mapped[User] = relationship(back_populates="ratings")
    item: Mapped[Item] = relationship(back_populates="ratings")
