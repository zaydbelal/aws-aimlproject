"""Popularity: recommend what most people watched.

It sounds dumb, but on MovieLens it is a strong baseline and the only thing
that works for a user with zero history. Every model we build has to beat it.

With ML-1M we also know each user's gender and age group, so the
"demographic" variant recommends what people like *them* watched:

    p_segment(i) = (count_in_segment(i) + m * p_global(i)) / (segment_total + m)

The m * p_global term is smoothing: a small segment leans on the global
ranking instead of trusting a handful of ratings.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..interactions import Interactions


class PopularityModel:
    def __init__(self, use_demographics: bool = False, smoothing: float = 1000.0):
        self.use_demographics = use_demographics
        self.smoothing = smoothing

    @staticmethod
    def segment_of(gender, age) -> str:
        return f"{gender}-{age}"

    def fit(self, inter: Interactions, users: pd.DataFrame | None = None) -> "PopularityModel":
        counts = inter.item_counts.astype(np.float64)
        self.global_share = (counts + 1.0) / (counts.sum() + len(counts))
        self.global_scores = np.log(self.global_share)

        self.segment_scores = {}
        self.user_segment = np.full(inter.n_users, None, dtype=object)
        if self.use_demographics and users is not None:
            users = users.set_index("user_id")
            known = [u for u in inter.user_ids if u in users.index]
            for u in known:
                row = users.loc[u]
                self.user_segment[inter.user_pos[u]] = self.segment_of(row["gender"], row["age"])

            binary = inter.matrix.copy()
            binary.data[:] = 1.0
            for segment in sorted(set(s for s in self.user_segment if s is not None)):
                members = np.flatnonzero(self.user_segment == segment)
                seg_counts = np.asarray(binary[members].sum(axis=0)).ravel()
                share = (seg_counts + self.smoothing * self.global_share) / (
                    seg_counts.sum() + self.smoothing)
                self.segment_scores[segment] = np.log(share)
        return self

    def score_users(self, user_index: np.ndarray) -> np.ndarray:
        rows = [self.segment_scores.get(self.user_segment[u], self.global_scores)
                for u in user_index]
        return np.vstack(rows)

    def score_new_user(self, gender=None, age=None) -> np.ndarray:
        if self.use_demographics and gender is not None and age is not None:
            return self.segment_scores.get(self.segment_of(gender, age), self.global_scores)
        return self.global_scores
