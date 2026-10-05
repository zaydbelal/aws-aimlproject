"""Fast unit tests on tiny synthetic data (no download needed): pytest -q"""

import numpy as np
import pandas as pd
import pytest

from hybridrec import metrics
from hybridrec.config import ALSConfig, ContentConfig, HybridConfig, SplitConfig
from hybridrec.data import split_title
from hybridrec.interactions import Interactions
from hybridrec.models import ALSModel, ContentModel, HybridModel, PopularityModel, zscore_rows
from hybridrec.split import make_split


# ----------------------------------------------------------------- fixtures

def synthetic(n_users=60, n_items=40, seed=0):
    """Two taste groups: users 0-29 like items 0-19 (Action), users 30-59 like 20-39 (Drama)."""
    rng = np.random.default_rng(seed)
    rows = []
    t = 0
    for u in range(n_users):
        group = u < n_users // 2
        liked = range(0, n_items // 2) if group else range(n_items // 2, n_items)
        for i in rng.choice(list(liked), size=12, replace=False):
            t += 1
            rows.append((u + 1, int(i) + 1, float(rng.integers(4, 6)), t))
    ratings = pd.DataFrame(rows, columns=["user_id", "item_id", "rating", "timestamp"])
    items = pd.DataFrame({
        "item_id": np.arange(1, n_items + 1),
        "title": [f"{'Explosion' if i < n_items // 2 else 'Tears'} Story {i}" for i in range(n_items)],
        "year": [1990.0 + (i % 10) for i in range(n_items)],
        "genres": [["Action"] if i < n_items // 2 else ["Drama"] for i in range(n_items)],
        "tags": ["" for _ in range(n_items)],
    })
    return ratings, items


# ------------------------------------------------------------------ metrics

def test_metrics_hand_computed():
    ranked, relevant = [5, 1, 9, 3], {1, 3, 7}
    assert metrics.precision_at_k(ranked, relevant, 4) == pytest.approx(2 / 4)
    assert metrics.recall_at_k(ranked, relevant, 4) == pytest.approx(2 / 3)
    dcg = 1 / np.log2(3) + 1 / np.log2(5)
    idcg = 1 + 1 / np.log2(3) + 1 / np.log2(4)
    assert metrics.ndcg_at_k(ranked, relevant, 4) == pytest.approx(dcg / idcg)
    assert metrics.hit_rate_at_k(ranked, relevant, 1) == 0.0
    assert metrics.ndcg_at_k([1, 3, 7], relevant, 3) == pytest.approx(1.0)


def test_beyond_accuracy_metrics():
    lists = [np.array([0, 1]), np.array([1, 2])]
    assert metrics.coverage(lists, 4) == pytest.approx(0.75)
    features = np.eye(3)
    assert metrics.intra_list_diversity(lists, features) == pytest.approx(1.0)


# --------------------------------------------------------------------- data

def test_split_title():
    assert split_title("American President, The (1995)") == ("The American President", 1995.0)
    title, year = split_title("No Year Here")
    assert title == "No Year Here" and np.isnan(year)


def test_split_creates_cold_users_and_items():
    ratings, _ = synthetic()
    cfg = SplitConfig(cold_user_frac=0.2, cold_item_frac=0.2, min_ratings_to_coldify_item=5)
    split = make_split(ratings, cfg)
    # Nothing lost, nothing duplicated.
    assert len(split.train) + len(split.test) == len(ratings)
    train_user_counts = split.train["user_id"].value_counts()
    for u in split.coldified_users:
        assert train_user_counts.get(u, 0) < cfg.cold_threshold
    train_item_counts = split.train["item_id"].value_counts()
    for i in split.coldified_items:
        assert train_item_counts.get(i, 0) < cfg.cold_threshold
    # Warm users' test ratings are newer than their training ratings.
    warm = ratings["user_id"].unique()
    warm = [u for u in warm if u not in set(split.coldified_users)]
    for u in warm[:10]:
        tr = split.train[(split.train.user_id == u) & ~split.train.item_id.isin(split.coldified_items)]
        te = split.test[(split.test.user_id == u) & ~split.test.item_id.isin(split.coldified_items)]
        if len(tr) and len(te):
            assert tr.timestamp.max() <= te.timestamp.min()


def test_split_is_deterministic():
    ratings, _ = synthetic()
    a = make_split(ratings, SplitConfig(seed=3))
    b = make_split(ratings, SplitConfig(seed=3))
    pd.testing.assert_frame_equal(a.train, b.train)


# ------------------------------------------------------------------- models

@pytest.fixture(scope="module")
def fitted():
    ratings, items = synthetic()
    inter = Interactions(ratings, ratings["user_id"].unique(), items["item_id"])
    als = ALSModel(ALSConfig(factors=4, regularization=0.1, alpha=1.0, iterations=8)).fit(inter)
    content = ContentModel(ContentConfig(text_dims=4)).fit(inter, items)
    pop = PopularityModel().fit(inter)
    return inter, als, content, pop


def test_als_learns_the_two_groups(fitted):
    inter, als, _, _ = fitted
    scores = als.score_users(np.array([0, 59]))
    # User 0 is in the Action group (items 0-19), user 59 in Drama (20-39).
    assert scores[0, :20].mean() > scores[0, 20:].mean()
    assert scores[1, 20:].mean() > scores[1, :20].mean()


def test_als_fold_in_new_user(fitted):
    _, als, _, _ = fitted
    scores = als.score_new_user([25, 26, 27], [5, 5, 4])
    assert scores[20:].mean() > scores[:20].mean()
    assert np.allclose(als.score_new_user([], []), 0)


def test_content_handles_items_without_ratings(fitted):
    _, _, content, _ = fitted
    # One liked Action movie -> other Action movies score higher than Drama ones.
    scores = content.score_new_user([3], [5])
    assert scores[:20].mean() > scores[20:].mean()


def test_zscore_rows():
    z = zscore_rows(np.array([[1.0, 2.0, 3.0], [5.0, 5.0, 5.0]]))
    assert z[0].mean() == pytest.approx(0) and z[0].std() == pytest.approx(1)
    assert np.all(z[1] == 0)   # no information -> no opinion


def test_adaptive_weight_shifts_with_history(fitted):
    inter, als, content, pop = fitted
    hybrid = HybridModel(als, content, pop, inter.user_counts, inter.item_counts,
                         HybridConfig(k_user=10, k_item=10), mode="adaptive")
    w = hybrid.cf_weight([0, 2, 10, 100]).mean(axis=1)
    assert w[0] == 0                       # brand-new user: no CF at all
    assert np.all(np.diff(w) > 0)          # more history -> more CF
    assert w[-1] < 1

    weights = hybrid.weights([0, 3, 50])
    total = weights["cf"] + weights["content"] + weights["popularity"]
    assert np.allclose(total, 1.0)                       # always a proper split
    assert np.all(weights["popularity"][0] == 1.0)       # new user -> popularity only
    assert np.all(weights["popularity"][0] > weights["popularity"][2])

    fixed = HybridModel(als, content, pop, inter.user_counts, inter.item_counts,
                        HybridConfig(), mode="fixed")
    assert np.all(fixed.cf_weight([0, 100]) == 0.5)


def test_new_movie_is_scored_by_content(fitted):
    inter, als, content, pop = fitted
    item_counts = inter.item_counts.copy()
    item_counts[5] = 0                                   # pretend movie 5 was just released
    hybrid = HybridModel(als, content, pop, inter.user_counts, item_counts,
                         HybridConfig(k_content=10.0))
    weights = hybrid.weights([50])
    assert weights["cf"][0, 5] == 0
    assert weights["content"][0, 5] > 0.8               # mostly content (k_content = 10)


def test_new_user_with_no_history_gets_popularity(fitted):
    inter, als, content, pop = fitted
    hybrid = HybridModel(als, content, pop, inter.user_counts, inter.item_counts, HybridConfig())
    final, parts = hybrid.score_new_user([], [])
    assert np.all(parts["weight"] == 0)
    assert np.argmax(final) == np.argmax(pop.global_scores)
