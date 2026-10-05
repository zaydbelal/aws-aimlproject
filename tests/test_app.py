"""Web API tests against a model trained on tiny synthetic data."""

import pandas as pd
import pytest

from hybridrec.config import ALSConfig, Config, ContentConfig, SplitConfig
from hybridrec.data import Dataset
from hybridrec.pipeline import build_experiment, save_experiment
from tests.test_core import synthetic


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    import app.server as server

    ratings, items = synthetic()
    users = pd.DataFrame({"user_id": ratings["user_id"].unique(), "gender": "F", "age": 25,
                          "occupation": 0})
    cfg = Config(split=SplitConfig(cold_user_frac=0.2, cold_item_frac=0.1,
                                   min_ratings_to_coldify_item=5),
                 als=ALSConfig(factors=4, iterations=3), content=ContentConfig(text_dims=4))
    experiment = build_experiment(cfg, Dataset("ml-1m", ratings, items, users), verbose=False)
    folder = tmp_path_factory.mktemp("artifacts")
    save_experiment(experiment, folder / "ml-1m.pkl")
    server.ARTIFACTS_DIR = folder
    server.RESULTS_DIR = folder
    return server.create_app("ml-1m").test_client()


def test_meta_and_search(client):
    meta = client.get("/api/meta").get_json()
    assert meta["movies"] == 40 and meta["demographics"] is True
    hits = client.get("/api/search?q=tears").get_json()
    assert hits and all("Tears" in h["title"] for h in hits)
    assert client.get("/api/search?q=t").get_json() == []


def test_recommend_existing_user(client):
    data = client.post("/api/recommend", json={"user_id": 1, "k": 5}).get_json()
    assert len(data["recommendations"]) == 5
    first = data["recommendations"][0]
    assert set(first["parts"]) == {"cf", "content", "popularity"}
    user = client.get("/api/user/1").get_json()
    seen = {h["item_id"] for h in user["history"]}
    assert not seen & {r["item_id"] for r in data["recommendations"]}


def test_recommend_new_user_and_diversify(client):
    body = {"ratings": [{"item_id": 25, "rating": 5}], "gender": "F", "age": 25,
            "k": 5, "diversify": True}
    data = client.post("/api/recommend", json=body).get_json()
    assert data["history_size"] == 1
    assert 25 not in {r["item_id"] for r in data["recommendations"]}
    for model in ["fixed", "switch", "cf", "content", "popularity"]:
        response = client.post("/api/recommend", json={**body, "model": model})
        assert response.status_code == 200


@pytest.mark.parametrize("body, status", [
    ({"user_id": 99999}, 404),
    ({"user_id": "abc"}, 400),
    ({"user_id": 1, "k": 0}, 400),
    ({"ratings": [{"item_id": 99999, "rating": 4}]}, 400),
    ({"ratings": [{"item_id": 1, "rating": 9}]}, 400),
    ({"ratings": [], "gender": "X"}, 400),
    ({"model": "nope", "ratings": []}, 400),
])
def test_validation_errors(client, body, status):
    response = client.post("/api/recommend", json=body)
    assert response.status_code == status
    assert "error" in response.get_json()


def test_results_missing_is_clear(client):
    response = client.get("/api/results")
    assert response.status_code == 404
    assert "evaluate" in response.get_json()["error"]
