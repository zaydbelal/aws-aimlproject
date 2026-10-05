"""Hyperparameter search on a validation split (the test set is never touched).

We split the *training* data again with the same procedure, which gives an
inner train / validation pair with its own cold users and cold items. Then:

Stage 1 (optional, --als): grid over ALS factors / regularization / alpha,
    scored by CF-only NDCG@10 on warm validation users.
Stage 2: grid over the blending knobs k_user, k_item and k_content,
    scored by the geometric mean of NDCG@10 on three slices: warm users,
    cold users and cold items. Scoring slices separately (instead of pooling
    all users) stops the large warm group from drowning out the cold-start
    cases. The geometric mean makes a 10% gain count the same in every
    slice, even though cold-user NDCG is about ten times larger than
    cold-item NDCG.

The winners are written to results/best_params.json, which `evaluate` and
`train` pick up automatically.
"""

from __future__ import annotations

import itertools
import json
import time
from dataclasses import replace

import numpy as np

from .config import RESULTS_DIR, Config
from .data import load_dataset
from .evaluate import per_user_metrics, rank
from .models import ALSModel, zscore_rows
from .pipeline import eval_users, fit_models, rebuild_hybrids, relevant_test_items
from .split import make_split

ALS_GRID = {"factors": [32, 64, 128], "regularization": [30.0, 100.0, 300.0], "alpha": [0.25, 1.0]}
HYBRID_GRID = {
    "k_user": [3.0, 10.0, 30.0, 100.0, 300.0],
    "k_item": [3.0, 10.0, 30.0, 100.0, 300.0, 1000.0],
    "k_content": [3.0, 10.0, 30.0, 100.0, 300.0],
}

BEST_PARAMS_PATH = RESULTS_DIR / "best_params.json"


def validation_experiment(cfg: Config, dataset, verbose=True):
    outer = make_split(dataset.ratings, cfg.split)
    inner = make_split(outer.train, replace(cfg.split, seed=cfg.split.seed + 1))
    if verbose:
        print(f"Validation split: {len(inner.train):,} train / {len(inner.test):,} validation ratings")
    return fit_models(cfg, dataset, inner, verbose=verbose)


def _ndcg(lists, users, truth, k) -> float:
    if len(users) == 0:
        return float("nan")
    return float(per_user_metrics(lists, users, truth, [k])[f"ndcg@{k}"].mean())


class SliceScorer:
    """Pre-computes the z-scored signals once so each grid point is just a re-weighting."""

    def __init__(self, experiment):
        self.e = experiment
        cfg = experiment.cfg
        threshold = cfg.split.cold_threshold
        self.k = cfg.k
        truth = relevant_test_items(experiment)
        users = eval_users(experiment, truth)
        n_train = experiment.inter.user_counts[users]

        cold_items = experiment.inter.item_counts < threshold
        cold_truth = {u: {i for i in items if cold_items[i]} for u, items in truth.items()}
        cold_truth = {u: s for u, s in cold_truth.items() if s}
        cold_item_users = eval_users(experiment, cold_truth)

        self.groups = {
            "warm users": (users[n_train >= threshold], truth, None),
            "cold users": (users[n_train < threshold], truth, None),
            "cold items": (cold_item_users, cold_truth, cold_items),
        }
        self.cache = {}
        for name, (group_users, _, _) in self.groups.items():
            self.cache[name] = {
                "cf": zscore_rows(experiment.als.score_users(group_users)),
                "cb": zscore_rows(experiment.content.score_users(group_users)),
                "pop": zscore_rows(experiment.demo_popularity.score_users(group_users)),
            }

    def score(self, hybrid) -> dict:
        out = {}
        for name, (users, truth, mask) in self.groups.items():
            c = self.cache[name]
            final = hybrid.combine(c["cf"], c["cb"], c["pop"], self.e.inter.user_counts[users])
            out[name] = _ndcg(rank(final, users, self.e, self.k, mask), users, truth, self.k)
        values = np.array([out[n] for n in self.groups])
        out["objective"] = float(np.exp(np.log(np.maximum(values, 1e-9)).mean()))
        return out


def tune_als(cfg: Config, dataset, verbose=True) -> dict:
    experiment = validation_experiment(cfg, dataset, verbose=False)
    truth = relevant_test_items(experiment)
    users = eval_users(experiment, truth)
    warm = users[experiment.inter.user_counts[users] >= cfg.split.cold_threshold]
    rows = []
    for factors, reg, alpha in itertools.product(*ALS_GRID.values()):
        start = time.time()
        als_cfg = replace(cfg.als, factors=factors, regularization=reg, alpha=alpha)
        als = ALSModel(als_cfg).fit(experiment.inter)
        lists = np.vstack([rank(als.score_users(warm[s:s + 512]), warm[s:s + 512], experiment, cfg.k)
                           for s in range(0, len(warm), 512)])
        score = _ndcg(lists, warm, truth, cfg.k)
        rows.append({"factors": factors, "regularization": reg, "alpha": alpha, "ndcg": score})
        if verbose:
            print(f"  ALS factors={factors:4d} reg={reg:5g} alpha={alpha:4g} -> "
                  f"NDCG@{cfg.k} {score:.4f}  ({time.time() - start:.0f}s)", flush=True)
    best = max(rows, key=lambda r: r["ndcg"])
    return {"grid": rows, "best": {k: best[k] for k in ("factors", "regularization", "alpha")}}


def tune_hybrid(cfg: Config, dataset, verbose=True) -> dict:
    experiment = validation_experiment(cfg, dataset, verbose=verbose)
    scorer = SliceScorer(experiment)
    rows = []
    for k_user, k_item, k_content in itertools.product(*HYBRID_GRID.values()):
        experiment.cfg = replace(cfg, hybrid=replace(cfg.hybrid, k_user=k_user, k_item=k_item,
                                                     k_content=k_content))
        rebuild_hybrids(experiment)
        result = scorer.score(experiment.hybrids["adaptive"])
        rows.append({"k_user": k_user, "k_item": k_item, "k_content": k_content, **result})
    rows.sort(key=lambda r: -r["objective"])
    if verbose:
        print("Top 5 blending settings (validation):")
        for r in rows[:5]:
            print(f"  k_user={r['k_user']:5g} k_item={r['k_item']:5g} "
                  f"k_content={r['k_content']:5g} -> warm {r['warm users']:.4f} "
                  f"cold users {r['cold users']:.4f} cold items {r['cold items']:.4f} "
                  f"geo-mean {r['objective']:.4f}")
    best = {k: rows[0][k] for k in HYBRID_GRID}
    return {"grid": rows, "best": best}


def run_tuning(cfg: Config, include_als: bool = False, verbose: bool = True) -> dict:
    dataset = load_dataset(cfg.dataset)
    output = {"dataset": cfg.dataset}
    if include_als:
        print("Stage 1: ALS grid")
        output["als"] = tune_als(cfg, dataset, verbose)
        cfg = replace(cfg, als=replace(cfg.als, **output["als"]["best"]))
    print("Stage 2: blending grid")
    output["hybrid"] = tune_hybrid(cfg, dataset, verbose)

    best = {"als": {k: getattr(cfg.als, k) for k in ALS_GRID}, "hybrid": output["hybrid"]["best"]}
    RESULTS_DIR.mkdir(exist_ok=True)
    log_path = RESULTS_DIR / f"tuning_{cfg.dataset}.json"
    if not include_als and log_path.exists():
        # Keep the record of an earlier ALS search, if there was one.
        previous = json.loads(log_path.read_text())
        if "als" in previous:
            output["als"] = previous["als"]
    saved = json.loads(BEST_PARAMS_PATH.read_text()) if BEST_PARAMS_PATH.exists() else {}
    saved[cfg.dataset] = best
    BEST_PARAMS_PATH.write_text(json.dumps(saved, indent=2))
    log_path.write_text(json.dumps(output, indent=2))
    print(f"Best parameters: {best}\nSaved to {BEST_PARAMS_PATH}")
    return best


def apply_best_params(cfg: Config, verbose: bool = True) -> Config:
    """Return cfg with tuned values from results/best_params.json, if any."""
    if not BEST_PARAMS_PATH.exists():
        return cfg
    params = json.loads(BEST_PARAMS_PATH.read_text()).get(cfg.dataset)
    if not params:
        return cfg
    if verbose:
        print(f"Using tuned parameters from {BEST_PARAMS_PATH.name}: {params}")
    return replace(cfg, als=replace(cfg.als, **params["als"]),
                   hybrid=replace(cfg.hybrid, **params["hybrid"]))
