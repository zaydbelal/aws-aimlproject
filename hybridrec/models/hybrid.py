"""The hybrid: trust each signal only as far as the data behind it allows.

Three signals score every (user, movie) pair:

    CF   - ALS matrix factorization   needs history on the user AND the movie
    CB   - content similarity         needs a few ratings from the user, none on the movie
    POP  - (demographic) popularity   needs nothing from the user

They are on different scales, so each is z-scored per user first (minus the
user's mean, divided by the user's standard deviation). Then three weights
that always add up to 1:

    c_user = n_user / (n_user + k_user)       trust in the user's CF vector
    c_item = n_item / (n_item + k_item)       trust in the movie's CF vector
    t_user = n_user / (n_user + k_content)    trust in the user's content profile

    w_cf   = c_user * c_item
    w_cb   = (1 - w_cf) * t_user              what CF can't cover goes to content...
    w_pop  = (1 - w_cf) * (1 - t_user)        ...and what content can't cover to popularity

    final  = w_cf * CF + w_cb * CB + w_pop * POP

The weights are set separately for every user-movie pair and shift smoothly
as history thins out:

    brand-new user                 -> 100% popularity
    user with a handful of ratings -> mostly popularity, some content and CF
    regular user, brand-new movie  -> c_item = 0: content and popularity decide
    regular user, well-known movie -> mostly CF

For comparison the same class can run two simpler strategies:
    "fixed"  - 50% CF + 50% content everywhere (the naive mix the brief warns against)
    "switch" - hard rules: CF if user and movie both have >= 5 ratings,
               else content if the user has any rating, else popularity
"""

from __future__ import annotations

import numpy as np

from ..config import HybridConfig


def zscore_rows(scores: np.ndarray) -> np.ndarray:
    """Standardise each row; a constant row (no information) becomes all zeros."""
    mean = scores.mean(axis=1, keepdims=True)
    std = scores.std(axis=1, keepdims=True)
    return np.divide(scores - mean, std, out=np.zeros_like(scores), where=std > 1e-12)


class HybridModel:
    MODES = ("adaptive", "fixed", "switch")

    def __init__(self, cf, content, popularity, user_counts, item_counts,
                 cfg: HybridConfig, mode: str = "adaptive", cold_threshold: int = 5):
        if mode not in self.MODES:
            raise ValueError(f"mode must be one of {self.MODES}")
        self.cf, self.content, self.popularity = cf, content, popularity
        self.user_counts = np.asarray(user_counts)
        self.item_counts = np.asarray(item_counts)
        self.cfg, self.mode, self.cold_threshold = cfg, mode, cold_threshold

    # --- the weighting rule --------------------------------------------

    def user_confidence(self, n_user) -> np.ndarray:
        n_user = np.asarray(n_user, dtype=np.float64)
        return n_user / (n_user + self.cfg.k_user)

    def item_confidence(self) -> np.ndarray:
        return self.item_counts / (self.item_counts + self.cfg.k_item)

    def content_confidence(self, n_user) -> np.ndarray:
        n_user = np.asarray(n_user, dtype=np.float64)
        return n_user / (n_user + self.cfg.k_content)

    def weights(self, n_users) -> dict:
        """Three (users x items) weight matrices, one per signal; they sum to 1."""
        n_users = np.atleast_1d(np.asarray(n_users, dtype=np.float64))
        shape = (len(n_users), len(self.item_counts))

        if self.mode == "fixed":
            return {"cf": np.full(shape, 0.5), "content": np.full(shape, 0.5),
                    "popularity": np.zeros(shape)}

        if self.mode == "switch":
            warm = ((n_users >= self.cold_threshold)[:, None]
                    & (self.item_counts >= self.cold_threshold)[None, :])
            has_history = np.broadcast_to((n_users > 0)[:, None], shape)
            return {"cf": warm.astype(np.float64),
                    "content": (~warm & has_history).astype(np.float64),
                    "popularity": (~warm & ~has_history).astype(np.float64)}

        w_cf = self.user_confidence(n_users)[:, None] * self.item_confidence()[None, :]
        t_user = self.content_confidence(n_users)[:, None]
        return {"cf": w_cf,
                "content": (1 - w_cf) * t_user,
                "popularity": (1 - w_cf) * (1 - t_user)}

    def cf_weight(self, n_users) -> np.ndarray:
        return self.weights(n_users)["cf"]

    # --- scoring -------------------------------------------------------

    def combine(self, cf_z, cb_z, pop_z, n_users, return_parts: bool = False):
        """Blend already z-scored signals (the tuner calls this directly)."""
        w = self.weights(n_users)
        parts = {"cf": w["cf"] * cf_z, "content": w["content"] * cb_z,
                 "popularity": w["popularity"] * pop_z}
        final = parts["cf"] + parts["content"] + parts["popularity"]
        if return_parts:
            parts["weight"] = w["cf"]
            parts["weights"] = w
            return final, parts
        return final

    def blend(self, cf_raw, cb_raw, pop_raw, n_users, return_parts: bool = False):
        return self.combine(zscore_rows(cf_raw), zscore_rows(cb_raw), zscore_rows(pop_raw),
                            n_users, return_parts)

    def score_users(self, user_index: np.ndarray) -> np.ndarray:
        return self.blend(self.cf.score_users(user_index),
                          self.content.score_users(user_index),
                          self.popularity.score_users(user_index),
                          self.user_counts[user_index])

    def score_new_user(self, item_index, ratings, gender=None, age=None):
        """Score a user who is not in the training data (web app "new user" mode)."""
        item_index = np.asarray(item_index, dtype=int)
        final, parts = self.blend(self.cf.score_new_user(item_index, ratings)[None, :],
                                  self.content.score_new_user(item_index, ratings)[None, :],
                                  self.popularity.score_new_user(gender, age)[None, :],
                                  [len(item_index)], return_parts=True)
        return final[0], _first_row(parts)

    def explain_user(self, user_index: int):
        """Final scores plus each signal's contribution, for one training user."""
        u = np.array([user_index])
        final, parts = self.blend(self.cf.score_users(u), self.content.score_users(u),
                                  self.popularity.score_users(u), self.user_counts[u],
                                  return_parts=True)
        return final[0], _first_row(parts)


def _first_row(parts: dict) -> dict:
    out = {name: value[0] for name, value in parts.items() if name != "weights"}
    out["weights"] = {name: value[0] for name, value in parts["weights"].items()}
    return out
