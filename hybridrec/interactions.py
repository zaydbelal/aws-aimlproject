"""Turning the ratings table into a user x item matrix.

Models work with row/column positions (0, 1, 2, ...) instead of MovieLens
ids, so `Interactions` keeps the mapping in both directions plus the sparse
training matrix R where R[u, i] = rating, and 0 = "not watched".
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.sparse as sp


class Interactions:
    def __init__(self, train: pd.DataFrame, user_ids, item_ids):
        # Every user and every catalog item gets a row/column, including
        # people and movies with zero training data (the coldest cases).
        self.user_ids = np.asarray(sorted(set(user_ids)))
        self.item_ids = np.asarray(sorted(set(item_ids)))
        self.user_pos = {u: p for p, u in enumerate(self.user_ids)}
        self.item_pos = {i: p for p, i in enumerate(self.item_ids)}

        rows = train["user_id"].map(self.user_pos).to_numpy()
        cols = train["item_id"].map(self.item_pos).to_numpy()
        values = train["rating"].to_numpy(dtype=np.float64)
        self.matrix = sp.csr_matrix((values, (rows, cols)),
                                    shape=(len(self.user_ids), len(self.item_ids)))
        self.matrix.sum_duplicates()

        # How much history each user / item has: this is what the hybrid
        # blend looks at to decide whom to trust.
        self.user_counts = np.diff(self.matrix.indptr)
        self.item_counts = np.diff(self.matrix.tocsc().indptr)

    @property
    def n_users(self) -> int:
        return len(self.user_ids)

    @property
    def n_items(self) -> int:
        return len(self.item_ids)

    def user_history(self, user_index: int) -> tuple[np.ndarray, np.ndarray]:
        """(item positions, ratings) the user has in the training data."""
        start, end = self.matrix.indptr[user_index], self.matrix.indptr[user_index + 1]
        return self.matrix.indices[start:end], self.matrix.data[start:end]
