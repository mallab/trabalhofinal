"""
Testes de integração: sobe a API inteira (banco SQLite + modelo) e faz
requisições de verdade, do mesmo jeito que o Swagger faz.
"""

import pytest

from tests.conftest import KIDS, SCIFI, SHAWSHANK


def _new_user(client, **extra):
    resp = client.post("/users", json={"name": "Teste", **extra})
    assert resp.status_code == 201, resp.text
    return resp.json()


# -------------------------------- sistema ---------------------------------- #
def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok", "database": "ok", "model_trained": True}


def test_swagger_docs_available(client):
    assert client.get("/docs").status_code == 200
    spec = client.get("/openapi.json").json()
    for path in ("/users", "/items", "/users/{user_id}/recommendations",
                 "/users/{user_id}/preferences", "/users/{user_id}/ratings"):
        assert path in spec["paths"]


def test_root_redirects_to_docs(client):
    resp = client.get("/", follow_redirects=False)
    assert resp.status_code in (302, 307)
    assert resp.headers["location"] == "/docs"


def test_database_was_seeded(client):
    status = client.get("/model/status").json()
    assert status["trained"] is True
    assert status["n_items"] >= 40
    assert status["n_users"] == 32


def test_retrain_increments_version(client):
    before = client.get("/model/status").json()["version"]
    resp = client.post("/model/retrain")
    assert resp.status_code == 200
    assert resp.json()["version"] == before + 1
    assert resp.json()["stale"] is False


# ------------------------------- usuários ---------------------------------- #
def test_create_and_get_user(client):
    user = _new_user(client, preferred_genres=["sci-fi", "Action"], disliked_genres=["horror"])
    assert user["preferred_genres"] == ["Sci-Fi", "Action"]  # "sci-fi" virou "Sci-Fi"
    assert user["disliked_genres"] == ["Horror"]
    assert user["n_ratings"] == 0

    fetched = client.get(f"/users/{user['id']}").json()
    assert fetched == user


def test_create_user_validation(client):
    assert client.post("/users", json={"name": ""}).status_code == 422
    resp = client.post("/users", json={"name": "X", "preferred_genres": ["Telenovela"]})
    assert resp.status_code == 422
    assert "valid_genres" in resp.json()["detail"]


def test_user_not_found(client):
    assert client.get("/users/999999").status_code == 404
    assert client.get("/users/999999/recommendations").status_code == 404
    assert client.post("/users/999999/ratings", json={"item_id": 1, "rating": 4}).status_code == 404


def test_update_preferences_partial(client):
    user = _new_user(client, preferred_genres=["Comedy"], disliked_genres=["Horror"])
    resp = client.put(f"/users/{user['id']}/preferences", json={"preferred_genres": ["Animation"]})
    assert resp.status_code == 200
    assert resp.json()["preferred_genres"] == ["Animation"]
    assert resp.json()["disliked_genres"] == ["Horror"]  # não enviado => mantido


# -------------------------------- itens ------------------------------------ #
def test_list_and_search_items(client):
    assert len(client.get("/items", params={"limit": 5}).json()) == 5
    found = client.get("/items", params={"search": "matrix"}).json()
    assert [i["title"] for i in found] == ["Matrix, The (1999)"]
    animation = client.get("/items", params={"genre": "Animation", "limit": 100}).json()
    assert animation and all("Animation" in i["genres"] for i in animation)


def test_item_detail_and_404(client):
    item = client.get(f"/items/{SHAWSHANK}").json()
    assert item["title"].startswith("Shawshank")
    assert item["year"] == 1994
    assert item["n_ratings"] > 0 and 0.5 <= item["avg_rating"] <= 5
    assert client.get("/items/999999").status_code == 404


def test_genres(client):
    genres = client.get("/genres").json()
    assert "Sci-Fi" in genres and "Comedy" in genres


def test_similar_items(client):
    resp = client.get("/items/1/similar", params={"n": 4})  # Matrix
    assert resp.status_code == 200
    ids = [s["item"]["id"] for s in resp.json()]
    assert 1 not in ids and len(ids) == 4
    assert sum(i in SCIFI for i in ids) >= 3


def test_create_item_and_recommend_it_without_ratings(client):
    # filme novo tem que entrar no modelo e já poder ser recomendado
    version = client.get("/model/status").json()["version"]
    resp = client.post("/items", json={"title": "Animação Inédita (2025)",
                                       "genres": ["animation", "Children", "Comedy"], "year": 2025})
    assert resp.status_code == 201
    item = resp.json()
    assert item["genres"] == ["Animation", "Children", "Comedy"]
    assert client.get("/model/status").json()["stale"] is True

    user = _new_user(client, preferred_genres=["Animation", "Children"])
    recs = client.get(f"/users/{user['id']}/recommendations",
                      params={"n": 10, "strategy": "content"}).json()
    assert item["id"] in [r["item"]["id"] for r in recs["recommendations"]]
    assert client.get("/model/status").json()["version"] == version + 1


def test_create_item_validation(client):
    assert client.post("/items", json={"title": ""}).status_code == 422
    assert client.post("/items", json={"title": "X", "year": 1500}).status_code == 422


# ------------------------------ avaliações --------------------------------- #
def test_rate_create_then_update(client):
    user = _new_user(client)
    first = client.post(f"/users/{user['id']}/ratings", json={"item_id": 1, "rating": 3.5})
    assert first.status_code == 201
    second = client.post(f"/users/{user['id']}/ratings", json={"item_id": 1, "rating": 5})
    assert second.status_code == 200
    assert second.json()["rating"] == 5.0

    ratings = client.get(f"/users/{user['id']}/ratings").json()
    assert len(ratings) == 1 and ratings[0]["rating"] == 5.0
    assert client.get(f"/users/{user['id']}").json()["n_ratings"] == 1


@pytest.mark.parametrize("value", [0, 5.5, 3.3, -1])
def test_rate_invalid_value(client, value):
    user = _new_user(client)
    resp = client.post(f"/users/{user['id']}/ratings", json={"item_id": 1, "rating": value})
    assert resp.status_code == 422


def test_rate_unknown_item(client):
    user = _new_user(client)
    resp = client.post(f"/users/{user['id']}/ratings", json={"item_id": 999999, "rating": 4})
    assert resp.status_code == 404


def test_delete_rating(client):
    user = _new_user(client)
    client.post(f"/users/{user['id']}/ratings", json={"item_id": 2, "rating": 4})
    assert client.delete(f"/users/{user['id']}/ratings/2").status_code == 204
    assert client.delete(f"/users/{user['id']}/ratings/2").status_code == 404


# ---------------------------- recomendações -------------------------------- #
def test_new_user_without_history_gets_popular(client):
    user = _new_user(client)
    data = client.get(f"/users/{user['id']}/recommendations", params={"n": 5}).json()
    assert data["strategy_requested"] == "hybrid"
    assert data["strategy_used"] == "popular"
    assert data["recommendations"][0]["item"]["id"] == SHAWSHANK
    assert len(data["recommendations"]) == 5


def test_recommendations_follow_ratings_in_real_time(client):
    # fluxo completo: dar nota muda a recomendação na hora, sem retreinar
    user = _new_user(client)
    uid = user["id"]
    version = client.get("/model/status").json()["version"]

    for item_id, rating in [(11, 5), (12, 5), (13, 4.5), (14, 5), (1, 1), (27, 1)]:
        client.post(f"/users/{uid}/ratings", json={"item_id": item_id, "rating": rating})

    data = client.get(f"/users/{uid}/recommendations", params={"n": 4}).json()
    assert data["strategy_used"] == "hybrid"
    assert set(data["weights"]) == {"collaborative", "content", "popular"}
    ids = [r["item"]["id"] for r in data["recommendations"]]
    assert not {11, 12, 13, 14, 1, 27} & set(ids)  # já avaliados não voltam
    assert sum(i in KIDS for i in ids) >= 3
    for rec in data["recommendations"]:
        assert rec["reason"]
        assert 0 <= rec["score"] <= 1
    assert client.get("/model/status").json()["version"] == version  # sem retreino


def test_disliked_genres_never_recommended(client):
    user = _new_user(client)
    client.put(f"/users/{user['id']}/preferences", json={"disliked_genres": ["Drama", "Crime"]})
    data = client.get(f"/users/{user['id']}/recommendations", params={"n": 15}).json()
    for rec in data["recommendations"]:
        assert "Drama" not in rec["item"]["genres"] and "Crime" not in rec["item"]["genres"]


@pytest.mark.parametrize("strategy", ["hybrid", "collaborative", "content", "popular"])
def test_all_strategies(client, strategy):
    user = _new_user(client, preferred_genres=["Romance"])
    for item_id in (19, 20, 21):
        client.post(f"/users/{user['id']}/ratings", json={"item_id": item_id, "rating": 5})
    resp = client.get(f"/users/{user['id']}/recommendations", params={"strategy": strategy, "n": 3})
    assert resp.status_code == 200
    assert resp.json()["strategy_used"] == strategy
    assert len(resp.json()["recommendations"]) == 3


def test_invalid_strategy_and_n(client):
    assert client.get("/users/1/recommendations", params={"strategy": "x"}).status_code == 422
    assert client.get("/users/1/recommendations", params={"n": 0}).status_code == 422
