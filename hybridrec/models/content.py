"""Content-based filtering: recommend movies that look like ones you liked.

Step 1: describe every movie with a vector built from its metadata:

    [ genres (multi-hot) | decade (one-hot) | text embedding ]

The text embedding is TF-IDF over the title words (plus user tags on ML-25M
and ml-latest-small) compressed to 64 dims with truncated SVD, i.e. Latent
Semantic Analysis. Words that appear together, like "star" and "trek",
end up in the same direction, so sequels and franchises land close together.
Each block is scaled to unit length and multiplied by its weight, so no
block dominates just because it has more columns.

Step 2: describe every user as the weighted sum of the movies they rated:

    profile_u = sum_i (rating_ui - 2.5) * item_vector_i

A 5-star rating pulls the profile towards that movie; a 1-star rating
pushes it away. Score = cosine(profile_u, item_vector_i).

Many movies share exactly the same vector (same genres, same decade, no
title word in common with anything), so their cosines tie. We break those
exact ties by popularity with a tiny 1e-6 * log(1 + ratings) nudge;
otherwise the order would fall back to the arbitrary order of movie ids.
A user with no ratings has no profile, and the model gives no opinion
(all scores 0) rather than quietly turning into a popularity ranking.

Why this helps cold start: a new movie has metadata on day one, so it can
be scored without a single rating. A user with one rating already has a
usable profile.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer, normalize

from ..config import ContentConfig
from ..interactions import Interactions

RATING_MIDPOINT = 2.5
TIE_BREAK = 1e-6


def build_item_features(items: pd.DataFrame, cfg: ContentConfig, seed: int = 0) -> np.ndarray:
    """Items must already be in the model's item order. Returns unit-length rows."""
    blocks = []

    genres = MultiLabelBinarizer().fit_transform(items["genres"]).astype(np.float64)
    blocks.append(cfg.genre_weight * normalize(genres))

    decade = (items["year"] // 10 * 10).fillna(-1).astype(int)
    decade_onehot = pd.get_dummies(decade).drop(columns=-1, errors="ignore")
    blocks.append(cfg.year_weight * normalize(decade_onehot.to_numpy(np.float64)))

    text = (items["title"].fillna("") + " " + items["tags"].fillna("")).str.lower()
    tfidf = TfidfVectorizer(stop_words="english", min_df=2, sublinear_tf=True,
                            token_pattern=r"(?u)\b[a-z][a-z0-9']+\b").fit_transform(text)
    dims = min(cfg.text_dims, tfidf.shape[1] - 1)
    embedding = TruncatedSVD(n_components=dims, random_state=seed).fit_transform(tfidf)
    blocks.append(cfg.text_weight * normalize(embedding))

    return normalize(np.hstack(blocks))


class ContentModel:
    def __init__(self, cfg: ContentConfig):
        self.cfg = cfg

    def fit(self, inter: Interactions, items: pd.DataFrame) -> "ContentModel":
        ordered = items.set_index("item_id").loc[inter.item_ids].reset_index()
        self.item_features = build_item_features(ordered, self.cfg)

        weights = inter.matrix.copy()
        weights.data = weights.data - RATING_MIDPOINT
        self.user_profiles = normalize(np.asarray(weights @ self.item_features))
        self.tie_break = TIE_BREAK * np.log1p(inter.item_counts)
        return self

    def score_users(self, user_index: np.ndarray) -> np.ndarray:
        # Both sides have unit length, so the dot product is the cosine.
        profiles = self.user_profiles[user_index]
        has_profile = np.abs(profiles).sum(axis=1, keepdims=True) > 0
        return profiles @ self.item_features.T + has_profile * self.tie_break

    def profile_from_history(self, item_index, ratings) -> np.ndarray:
        weights = np.asarray(ratings, dtype=np.float64) - RATING_MIDPOINT
        profile = weights @ self.item_features[np.asarray(item_index, dtype=int)]
        norm = np.linalg.norm(profile)
        return profile / norm if norm > 0 else profile

    def score_new_user(self, item_index, ratings) -> np.ndarray:
        profile = self.profile_from_history(item_index, ratings) if len(item_index) else None
        if profile is None or not profile.any():
            return np.zeros(len(self.item_features))
        return self.item_features @ profile + self.tie_break

    def similar_items(self, item: int, n: int = 10) -> np.ndarray:
        sims = self.item_features @ self.item_features[item]
        sims[item] = -np.inf
        return np.argsort(-sims)[:n]
