"""Testes unitários do modelo, sem banco e sem API."""

import numpy as np
import pandas as pd
import pytest

from app.evaluation import evaluate, train_test_split_by_user
from app.recommender import HybridRecommender, decade_token, split_genres
from tests.conftest import KIDS, ROMANCE, SCIFI, SHAWSHANK, TOY_STORY


# ------------------------------ auxiliares --------------------------------- #
def test_split_genres():
    assert split_genres("Action|Sci-Fi") == ["Action", "Sci-Fi"]
    assert split_genres("(no genres listed)") == []
    assert split_genres("") == []
    assert split_genres(None) == []


def test_decade_token():
    assert decade_token(1994) == "decada_1990"
    assert decade_token(2001.0) == "decada_2000"
    assert decade_token(None) is None
    assert decade_token(float("nan")) is None


# -------------------------------- treino ----------------------------------- #
def test_fit_learns_catalog(model, sample_data):
    items, ratings = sample_data
    assert model.is_fitted
    assert len(model.item_ids) == len(items)
    assert "Sci-Fi" in model.known_genres and "Animation" in model.known_genres
    assert model.n_ratings == len(ratings)
    assert model.Vt is not None  # SVD treinado


def test_fit_empty_catalog_raises():
    with pytest.raises(ValueError):
        HybridRecommender().fit(pd.DataFrame(columns=["id", "genres"]),
                                pd.DataFrame(columns=["user_id", "item_id", "rating"]))


def test_fit_without_ratings_still_works(sample_data):
    # sem nenhuma nota no catálogo: só a parte de conteúdo funciona
    items, _ = sample_data
    m = HybridRecommender().fit(items, pd.DataFrame(columns=["user_id", "item_id", "rating"]))
    assert m.Vt is None
    result = m.recommend({}, preferred_genres=["Animation"], n=3)
    assert all("Animation" in m.item_genres[m.item_pos[r.item_id]] for r in result.items)


def test_popularity_prefers_well_rated_classics(model):
    assert int(np.argmax(model.popularity)) == model.item_pos[SHAWSHANK]


# --------------------------- recomendação ---------------------------------- #
def test_cold_start_user_gets_popular(model):
    result = model.recommend({}, n=5)
    assert result.strategy == "popular"
    assert result.weights == {"popular": 1.0}
    assert result.items[0].item_id == SHAWSHANK
    assert "Popular" in result.items[0].reason


def test_cold_start_with_preferred_genres_uses_content(model):
    result = model.recommend({}, preferred_genres=["Animation"], n=5)
    assert result.strategy == "hybrid"
    assert "collaborative" not in result.weights
    for rec in result.items:
        assert "Animation" in model.item_genres[model.item_pos[rec.item_id]]


def test_scifi_fan_gets_scifi(model):
    fan = {1: 5.0, 2: 5.0, 3: 4.5, 6: 5.0, 19: 1.0, 20: 1.5}
    result = model.recommend(fan, n=4)
    hits = sum(r.item_id in SCIFI for r in result.items)
    assert hits >= 3, [model.titles[model.item_pos[r.item_id]] for r in result.items]


def test_collaborative_alone_learns_taste(model):
    kid = {11: 5.0, 12: 5.0, 13: 4.5, 1: 1.0, 27: 1.0}
    result = model.recommend(kid, n=3, strategy="collaborative")
    assert result.strategy == "collaborative"
    assert sum(r.item_id in KIDS for r in result.items) >= 2


def test_rated_items_are_never_recommended(model):
    rated = {i: 4.0 for i in range(1, 30)}
    recs = model.recommend(rated, n=40).items
    assert not set(rated) & {r.item_id for r in recs}
    assert len(recs) == 40 - len(rated)  # pediu 40 mas só sobram 11


def test_disliked_genres_are_filtered(model):
    recs = model.recommend({}, disliked_genres=["Drama", "Comedy"], n=40).items
    assert recs
    for r in recs:
        genres = model.item_genres[model.item_pos[r.item_id]]
        assert "Drama" not in genres and "Comedy" not in genres


def test_exclude_parameter(model):
    recs = model.recommend({}, n=5, exclude=[SHAWSHANK]).items
    assert SHAWSHANK not in {r.item_id for r in recs}


def test_weights_adapt_to_history_size(model):
    few = model.recommend({1: 5.0, 2: 1.0}, n=1).weights
    many = model.recommend({i: (5.0 if i in SCIFI else 1.0) for i in range(1, 25)}, n=1).weights
    assert few["collaborative"] < many["collaborative"]
    assert many["collaborative"] == model.config.max_cf_weight
    assert few["popular"] > many["popular"]  # quem tem pouca nota recebe mais popularidade
    for w in (few, many):
        assert sum(w.values()) == pytest.approx(1.0)


def test_strategy_falls_back_to_popular_without_signal(model):
    assert model.recommend({}, strategy="collaborative").strategy == "popular"
    assert model.recommend({}, strategy="content").strategy == "popular"


def test_invalid_strategy_raises(model):
    with pytest.raises(ValueError):
        model.recommend({}, strategy="magica")


def test_unfitted_model_raises():
    with pytest.raises(RuntimeError):
        HybridRecommender().recommend({})


def test_unknown_items_in_history_are_ignored(model):
    result = model.recommend({999999: 5.0}, n=3)
    assert result.strategy == "popular"


def test_scores_sorted_and_explained(model):
    recs = model.recommend({19: 5.0, 20: 5.0, 21: 4.5}, n=6).items
    scores = [r.score for r in recs]
    assert scores == sorted(scores, reverse=True)
    assert all(r.reason for r in recs)
    assert all(r.score == pytest.approx(sum(r.components.values()), abs=1e-3) for r in recs)


def test_new_item_without_ratings_can_be_recommended(sample_data):
    # filme novo sem nenhuma nota tem que aparecer pela parte de conteúdo
    items, ratings = sample_data
    new = pd.DataFrame([{"id": 100, "title": "Filme Novo (2024)",
                         "genres": "Animation|Children|Comedy", "year": 2024}])
    m = HybridRecommender().fit(pd.concat([items, new], ignore_index=True), ratings)
    result = m.recommend({}, preferred_genres=["Animation", "Children"], n=8, strategy="content")
    assert 100 in {r.item_id for r in result.items}


def test_similar_items(model):
    similar = model.similar_items(TOY_STORY, n=4)
    ids = [i for i, _ in similar]
    assert TOY_STORY not in ids
    assert sum(i in KIDS for i in ids) >= 3
    with pytest.raises(KeyError):
        model.similar_items(999999)


def test_item_stats(model, sample_data):
    _, ratings = sample_data
    n, mean = model.item_stats(SHAWSHANK)
    expected = ratings[ratings.item_id == SHAWSHANK]["rating"]
    assert n == len(expected)
    assert mean == pytest.approx(expected.mean(), abs=0.01)
    assert model.item_stats(999999) == (0, None)


# ------------------------------- avaliação --------------------------------- #
def test_train_test_split_keeps_all_ratings(sample_data):
    _, ratings = sample_data
    train, test = train_test_split_by_user(ratings, min_ratings=5)
    assert len(train) + len(test) == len(ratings)
    assert len(test) > 0


def test_evaluate_returns_metrics(sample_data):
    items, ratings = sample_data
    report = evaluate(items, ratings, k=5, max_users=None, min_ratings=5)
    assert set(report["estrategia"]) == {"hybrid", "collaborative", "content", "popular"}
    for col in ("precision@5", "recall@5", "hit_rate@5", "coverage"):
        assert report[col].between(0, 1).all()


def test_romance_fan_explanation_mentions_liked_movie(model):
    recs = model.recommend({19: 5.0, 20: 5.0, 22: 5.0}, n=3, strategy="content").items
    assert any("Parecido com" in r.reason for r in recs)
    assert sum(r.item_id in ROMANCE for r in recs) >= 2
