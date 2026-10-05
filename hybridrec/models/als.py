"""Collaborative filtering with matrix factorization (implicit ALS).

The idea: give every user a vector x_u and every movie a vector y_i (128
numbers each, see config.py) so that x_u . y_i is high when u is likely to
watch i. Nobody labels what the dimensions mean; the model discovers
"taste directions" such as action-vs-romance purely from who watched what.

We use the implicit-feedback version from Hu, Koren & Volinsky (2008),
"Collaborative Filtering for Implicit Feedback Datasets":

* preference p_ui = 1 if u rated i, else 0    (did they watch it?)
* confidence c_ui = 1 + alpha * rating        (how sure we are about p_ui)

and minimise   sum_ui c_ui (p_ui - x_u . y_i)^2 + reg * (|x|^2 + |y|^2).

Alternating Least Squares: if the item vectors are fixed, each user vector
is an ordinary ridge regression with a closed-form answer, and vice versa.
So we alternate: solve all users, solve all items, repeat.

Why this fails for cold start: a user with 0 ratings has nothing to regress
on, so x_u = 0 and every score is 0. With 1-4 ratings x_u is a rough guess;
an item with 0-4 ratings has a rough y_i. The hybrid fixes this.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from ..config import ALSConfig
from ..interactions import Interactions


class ALSModel:
    def __init__(self, cfg: ALSConfig):
        self.cfg = cfg

    def fit(self, inter: Interactions, verbose: bool = False) -> "ALSModel":
        cfg = self.cfg
        # Store (confidence - 1) = alpha * rating; the "+1" part is handled
        # by the shared Gram matrix below.
        conf_users = inter.matrix.copy()
        conf_users.data = cfg.alpha * conf_users.data
        conf_items = sp.csr_matrix(conf_users.T)

        rng = np.random.default_rng(cfg.seed)
        self.user_factors = rng.normal(0, 0.01, (inter.n_users, cfg.factors))
        self.item_factors = rng.normal(0, 0.01, (inter.n_items, cfg.factors))

        for iteration in range(cfg.iterations):
            self.user_factors = self._solve_side(conf_users, self.item_factors)
            self.item_factors = self._solve_side(conf_items, self.user_factors)
            if verbose:
                print(f"  ALS iteration {iteration + 1}/{cfg.iterations}")
        self._item_gram = self._gram(self.item_factors)
        return self

    def _gram(self, fixed: np.ndarray) -> np.ndarray:
        """Y^T Y + reg * I: the part of the equation shared by every row."""
        return fixed.T @ fixed + self.cfg.regularization * np.eye(fixed.shape[1])

    def _solve_side(self, conf: sp.csr_matrix, fixed: np.ndarray) -> np.ndarray:
        gram = self._gram(fixed)
        solved = np.zeros((conf.shape[0], fixed.shape[1]))
        for row in range(conf.shape[0]):
            start, end = conf.indptr[row], conf.indptr[row + 1]
            solved[row] = self._solve_one(gram, fixed, conf.indices[start:end],
                                          conf.data[start:end])
        return solved

    @staticmethod
    def _solve_one(gram, fixed, index, conf_minus_one) -> np.ndarray:
        """Closed-form ridge regression for one user (or one item).

        x = (Y^T C Y + reg I)^-1 Y^T C p, written so it only touches the few
        items this user actually rated.
        """
        if len(index) == 0:
            return np.zeros(fixed.shape[1])   # no data -> no opinion
        Y = fixed[index]
        A = gram + (Y.T * conf_minus_one) @ Y
        b = Y.T @ (1.0 + conf_minus_one)       # p = 1 for every rated item
        return np.linalg.solve(A, b)

    # --- scoring --------------------------------------------------------

    def score_users(self, user_index: np.ndarray) -> np.ndarray:
        return self.user_factors[user_index] @ self.item_factors.T

    def fold_in(self, item_index: np.ndarray, ratings: np.ndarray) -> np.ndarray:
        """User vector for someone not in the training data (used by the web app).

        Same ridge regression as training, item vectors kept fixed.
        """
        conf = self.cfg.alpha * np.asarray(ratings, dtype=np.float64)
        return self._solve_one(self._item_gram, self.item_factors,
                               np.asarray(item_index, dtype=int), conf)

    def score_new_user(self, item_index, ratings) -> np.ndarray:
        return self.item_factors @ self.fold_in(item_index, ratings)
