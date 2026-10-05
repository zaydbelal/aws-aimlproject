"""Ranking metrics. All take a ranked list of item positions and a set of
relevant ones (test movies the user rated >= 4 stars).

Precision@K  share of the K recommendations that were relevant.
Recall@K     share of the user's relevant movies that made it into the top K.
NDCG@K       like recall, but a hit at rank 1 is worth more than a hit at
             rank 10 (gain 1/log2(rank + 1)), divided by the best possible
             score so 1.0 = perfect ordering.
HitRate@K    1 if at least one recommendation was relevant.

Beyond accuracy (computed over everyone's lists):
Coverage     share of the catalog that appears in anybody's top K.
Novelty      mean self-information -log2(share of users who watched the
             item). Higher = less obvious picks.
Diversity    average (1 - cosine similarity) between pairs of movies in a
             list, using the content vectors. Higher = more varied lists.
"""

from __future__ import annotations

import numpy as np


def precision_at_k(ranked, relevant, k):
    return len(set(ranked[:k]) & relevant) / k


def recall_at_k(ranked, relevant, k):
    return len(set(ranked[:k]) & relevant) / len(relevant) if relevant else 0.0


def ndcg_at_k(ranked, relevant, k):
    gains = [1.0 / np.log2(rank + 2) for rank, item in enumerate(ranked[:k]) if item in relevant]
    ideal = sum(1.0 / np.log2(rank + 2) for rank in range(min(len(relevant), k)))
    return sum(gains) / ideal if ideal > 0 else 0.0


def hit_rate_at_k(ranked, relevant, k):
    return float(bool(set(ranked[:k]) & relevant))


def coverage(lists, n_items):
    if len(lists) == 0:
        return 0.0
    return len(np.unique(np.concatenate(lists))) / n_items


def novelty(lists, item_counts, n_users):
    if len(lists) == 0:
        return 0.0
    share = (np.asarray(item_counts, dtype=np.float64) + 1) / (n_users + 1)
    return float(np.mean([-np.log2(share[items]).mean() for items in lists if len(items)]))


def intra_list_diversity(lists, item_features):
    values = []
    for items in lists:
        if len(items) < 2:
            continue
        vectors = item_features[items]
        sims = vectors @ vectors.T
        n = len(items)
        values.append(1 - (sims.sum() - np.trace(sims)) / (n * (n - 1)))
    return float(np.mean(values)) if values else 0.0
