"""Offline evaluation: rank, compare with what users really did, slice the results.

For each test user and each model:
  1. score every movie in the catalog,
  2. remove movies the user already rated in training,
  3. keep the top K,
  4. compare with the movies the user rated >= 4 stars in the test period.

Slices:
  warm users         >= 5 training ratings
  cold users         < 5 training ratings (also split by exactly 0, 1, 2, 3, 4)
  cold items         every candidate is a movie with < 5 training ratings,
                     so the question becomes "which new movie will you like?"
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import metrics
from .pipeline import Experiment, eval_users, relevant_test_items

BATCH = 512


def rank(scores: np.ndarray, users: np.ndarray, experiment: Experiment, k: int,
         candidate_mask: np.ndarray | None = None, reranker=None) -> np.ndarray:
    """Turn a (users x items) score matrix into top-k item positions, best first."""
    scores = np.array(scores, dtype=np.float64)           # copy: we write into it
    seen = experiment.inter.matrix[users].tocoo()
    scores[seen.row, seen.col] = -np.inf                  # never re-recommend
    if candidate_mask is not None:
        scores[:, ~candidate_mask] = -np.inf
    lists = np.zeros((len(users), k), dtype=int)
    for row in range(len(users)):
        if reranker is not None:
            picked = reranker.rerank(scores[row], k)
        else:
            best = np.argpartition(-scores[row], k)[:k]
            picked = best[np.argsort(-scores[row][best], kind="stable")]
        lists[row, :len(picked)] = picked
    return lists


def top_k(scorer, experiment: Experiment, users: np.ndarray, k: int,
          candidate_mask: np.ndarray | None = None, reranker=None) -> np.ndarray:
    """Score users in batches (keeps memory flat) and return their top-k lists."""
    lists = [rank(scorer.score_users(users[start:start + BATCH]), users[start:start + BATCH],
                  experiment, k, candidate_mask, reranker)
             for start in range(0, len(users), BATCH)]
    return np.vstack(lists) if lists else np.zeros((0, k), dtype=int)


def per_user_metrics(lists: np.ndarray, users: np.ndarray, truth: dict, ks) -> pd.DataFrame:
    rows = []
    for ranked, u in zip(lists, users):
        relevant = truth[int(u)]
        ranked = list(ranked)
        row = {"user": int(u), "n_relevant": len(relevant)}
        for k in ks:
            row[f"precision@{k}"] = metrics.precision_at_k(ranked, relevant, k)
            row[f"recall@{k}"] = metrics.recall_at_k(ranked, relevant, k)
            row[f"ndcg@{k}"] = metrics.ndcg_at_k(ranked, relevant, k)
            row[f"hit@{k}"] = metrics.hit_rate_at_k(ranked, relevant, k)
        rows.append(row)
    return pd.DataFrame(rows)


def run_models(experiment: Experiment, models: dict, truth: dict, users: np.ndarray,
               candidate_mask=None, include_rerank: bool = True, verbose: bool = True):
    """Evaluate every model on the same users. Returns (per-user metrics, top-K lists)."""
    cfg = experiment.cfg
    ks = sorted({cfg.k, *cfg.extra_ks})
    frames, all_lists = [], {}
    jobs = [(name, model, None) for name, model in models.items()]
    if include_rerank:
        jobs.append(("Hybrid adaptive + MMR re-rank", experiment.hybrids["adaptive"],
                     experiment.reranker))
    for name, model, reranker in jobs:
        if verbose:
            print(f"  ranking with {name} ...", flush=True)
        lists = top_k(model, experiment, users, max(ks), candidate_mask, reranker)
        frame = per_user_metrics(lists, users, truth, ks)
        frame.insert(0, "model", name)
        frame["n_train"] = experiment.inter.user_counts[users]
        frames.append(frame)
        all_lists[name] = lists
    return pd.concat(frames, ignore_index=True), all_lists


def beyond_accuracy(experiment: Experiment, lists: np.ndarray, k: int) -> dict:
    lists = [row[:k] for row in lists]
    inter = experiment.inter
    return {
        "coverage": metrics.coverage(lists, inter.n_items),
        "novelty": metrics.novelty(lists, inter.item_counts, inter.n_users),
        "diversity": metrics.intra_list_diversity(lists, experiment.content.item_features),
    }


def summarise(per_user: pd.DataFrame, lists: dict, users: np.ndarray, experiment: Experiment,
              slice_mask: np.ndarray | None = None) -> dict:
    """Average metrics per model over the users selected by slice_mask."""
    k = experiment.cfg.k
    summary = {}
    for name, frame in per_user.groupby("model", sort=False):
        frame = frame.reset_index(drop=True)
        mask = np.ones(len(frame), dtype=bool) if slice_mask is None else slice_mask
        chosen = frame[mask]
        result = {"users": int(len(chosen))}
        for column in frame.columns:
            if "@" in column:
                result[column] = float(chosen[column].mean()) if len(chosen) else float("nan")
        result.update(beyond_accuracy(experiment, lists[name][mask], k))
        summary[name] = result
    return summary


def evaluate_all(experiment: Experiment, verbose: bool = True) -> dict:
    """The full evaluation used for the report."""
    cfg = experiment.cfg
    threshold = cfg.split.cold_threshold
    truth = relevant_test_items(experiment)
    users = eval_users(experiment, truth)
    models = experiment.models()

    if verbose:
        print(f"Evaluating {len(users)} users with relevant test items ...")
    per_user, lists = run_models(experiment, models, truth, users, verbose=verbose)
    n_train = experiment.inter.user_counts[users]

    slices = {
        "all users": None,
        f"warm users (>= {threshold} ratings)": n_train >= threshold,
        f"cold users (< {threshold} ratings)": n_train < threshold,
    }
    for n in range(threshold):
        slices[f"cold users with exactly {n}"] = n_train == n

    results = {"slices": {}, "config": cfg.to_dict()}
    for slice_name, mask in slices.items():
        results["slices"][slice_name] = summarise(per_user, lists, users, experiment, mask)

    # Cold-item slice: candidates restricted to movies with < 5 training ratings.
    cold_items = experiment.inter.item_counts < threshold
    cold_truth = {u: {i for i in items if cold_items[i]} for u, items in truth.items()}
    cold_truth = {u: items for u, items in cold_truth.items() if items}
    cold_users = eval_users(experiment, cold_truth)
    if verbose:
        print(f"Cold-item slice: {cold_items.sum()} cold movies, {len(cold_users)} users "
              f"who liked one of them in test")
    cold_item_per_user, cold_item_lists = run_models(
        experiment, models, cold_truth, cold_users, candidate_mask=cold_items,
        include_rerank=False, verbose=verbose)
    results["slices"][f"cold items (< {threshold} ratings)"] = summarise(
        cold_item_per_user, cold_item_lists, cold_users, experiment)

    results["dataset"] = {
        "name": cfg.dataset,
        "users": int(experiment.inter.n_users),
        "items": int(experiment.inter.n_items),
        "train_ratings": int(len(experiment.split.train)),
        "test_ratings": int(len(experiment.split.test)),
        "eval_users": int(len(users)),
        "cold_eval_users": int((n_train < threshold).sum()),
        "cold_items": int(cold_items.sum()),
        "cold_item_eval_users": int(len(cold_users)),
    }
    results["_per_user"] = per_user          # used by the failure analysis, not saved to JSON
    results["_lists"] = lists
    results["_users"] = users
    results["_truth"] = truth
    return results
