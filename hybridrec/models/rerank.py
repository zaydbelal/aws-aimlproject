"""Bonus: re-ranking for diversity and novelty (Maximal Marginal Relevance).

A pure relevance ranking often returns ten near-identical blockbusters. MMR
(Carbonell & Goldstein, 1998) builds the list one movie at a time. At each
step it picks the candidate with the best

    lambda * relevance  -  (1 - lambda) * (similarity to what is already picked)

where similarity is cosine similarity of the content vectors. On top of
that we subtract a small popularity penalty to nudge less-famous movies up
(novelty). lambda = 1 gives back the original ranking.
"""

from __future__ import annotations

import numpy as np

from ..config import RerankConfig


class MMRReranker:
    def __init__(self, item_features: np.ndarray, item_counts: np.ndarray, cfg: RerankConfig):
        self.features = item_features
        self.cfg = cfg
        # Popularity as a percentile in [0, 1]: 1 = the most-watched movie.
        order = np.argsort(np.argsort(item_counts))
        self.popularity_pct = order / max(len(order) - 1, 1)

    def rerank(self, scores: np.ndarray, k: int) -> np.ndarray:
        """scores: one user's scores (already -inf for excluded items)."""
        n_candidates = min(self.cfg.candidates, int(np.isfinite(scores).sum()))
        if n_candidates == 0:
            return np.array([], dtype=int)
        candidates = np.argpartition(-scores, n_candidates - 1)[:n_candidates]
        relevance = scores[candidates]
        spread = relevance.max() - relevance.min()
        relevance = (relevance - relevance.min()) / spread if spread > 0 else np.ones_like(relevance)
        relevance = relevance - self.cfg.popularity_penalty * self.popularity_pct[candidates]

        vectors = self.features[candidates]
        max_similarity = np.zeros(n_candidates)
        chosen = np.zeros(n_candidates, dtype=bool)
        picked = []
        lam = self.cfg.mmr_lambda
        for _ in range(min(k, n_candidates)):
            mmr = lam * relevance - (1 - lam) * max_similarity
            mmr[chosen] = -np.inf
            best = int(np.argmax(mmr))
            chosen[best] = True
            picked.append(candidates[best])
            max_similarity = np.maximum(max_similarity, vectors @ vectors[best])
        return np.array(picked, dtype=int)
