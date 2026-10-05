"""Train/test split that deliberately creates cold-start users and items.

MovieLens 1M has no cold users: everyone rated at least 20 movies. So we
make some, the same way a real platform meets them:

* Warm users: the oldest 80% of their ratings go to train and the newest
  20% to test (we predict the future from the past).
* Cold users (10% of users): we keep only their first n ratings,
  n drawn at random from {0, 1, 2, 3, 4}, and hide the rest in test. n = 0
  is a brand-new sign-up.
* Cold items (5% of items): we keep only the first m ratings an item
  received, m drawn from {0, ..., 4}, and hide the rest in test. This is a
  newly released movie.

When a cold user rates a cold item, that rating always goes to test, so
neither of them sneaks past the "< 5 interactions" limit.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import SplitConfig


@dataclass
class Split:
    train: pd.DataFrame
    test: pd.DataFrame
    coldified_users: np.ndarray   # user ids we turned into cold-start users
    coldified_items: np.ndarray   # item ids we turned into cold-start items


def _pick(counts: pd.Series, min_count: int, share: float, total: int, rng) -> np.ndarray:
    eligible = np.sort(counts[counts >= min_count].index.to_numpy())
    size = min(len(eligible), int(round(share * total)))
    return np.sort(rng.choice(eligible, size=size, replace=False))


def make_split(ratings: pd.DataFrame, cfg: SplitConfig) -> Split:
    rng = np.random.default_rng(cfg.seed)
    # Chronological order. user_id and item_id break timestamp ties so the
    # result is identical on every machine.
    r = ratings.sort_values(["timestamp", "user_id", "item_id"], kind="mergesort")
    r = r.reset_index(drop=True)

    user_counts = r["user_id"].value_counts()
    item_counts = r["item_id"].value_counts()
    cold_users = _pick(user_counts, cfg.min_ratings_to_coldify_user,
                       cfg.cold_user_frac, len(user_counts), rng)
    cold_items = _pick(item_counts, cfg.min_ratings_to_coldify_item,
                       cfg.cold_item_frac, len(item_counts), rng)

    # How many training interactions each cold user / item keeps: 0..4.
    keep_user = pd.Series(rng.integers(0, cfg.cold_threshold, len(cold_users)), index=cold_users)
    keep_item = pd.Series(rng.integers(0, cfg.cold_threshold, len(cold_items)), index=cold_items)

    is_cold_user = r["user_id"].isin(cold_users).to_numpy()
    is_cold_item = r["item_id"].isin(cold_items).to_numpy()
    in_train = np.zeros(len(r), dtype=bool)

    # Case 1: cold user x cold item -> always test (in_train stays False).

    # Case 2: warm user x cold item -> keep the item's first m ratings.
    mask = is_cold_item & ~is_cold_user
    part = r[mask]
    position = part.groupby("item_id").cumcount().to_numpy()
    in_train[mask] = position < part["item_id"].map(keep_item).to_numpy()

    # Case 3: cold user x warm item -> keep the user's first n ratings.
    mask = is_cold_user & ~is_cold_item
    part = r[mask]
    position = part.groupby("user_id").cumcount().to_numpy()
    in_train[mask] = position < part["user_id"].map(keep_user).to_numpy()

    # Case 4: warm user x warm item -> newest test_frac of each user is test.
    mask = ~is_cold_user & ~is_cold_item
    part = r[mask]
    position = part.groupby("user_id").cumcount().to_numpy()
    size = part.groupby("user_id")["item_id"].transform("size").to_numpy()
    n_test = np.floor(size * cfg.test_frac).astype(int)
    in_train[mask] = position < size - n_test

    return Split(
        train=r[in_train].reset_index(drop=True),
        test=r[~in_train].reset_index(drop=True),
        coldified_users=cold_users,
        coldified_items=cold_items,
    )
