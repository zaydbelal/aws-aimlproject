"""Glue: load data -> split -> fit every model. Used by evaluation, tuning and the app."""

from __future__ import annotations

import pickle
import time
from dataclasses import dataclass, field

import numpy as np

from .config import ARTIFACTS_DIR, Config
from .data import Dataset, load_dataset
from .interactions import Interactions
from .models import ALSModel, ContentModel, HybridModel, MMRReranker, PopularityModel
from .split import Split, make_split


@dataclass
class Experiment:
    cfg: Config
    dataset: Dataset
    split: Split
    inter: Interactions
    popularity: PopularityModel          # global popularity
    demo_popularity: PopularityModel     # demographic popularity (= global without user data)
    als: ALSModel
    content: ContentModel
    hybrids: dict = field(default_factory=dict)   # mode -> HybridModel
    reranker: MMRReranker | None = None

    def models(self) -> dict:
        """Every recommender we compare, by display name -> object with score_users()."""
        models = {"Popularity": self.popularity}
        if self.dataset.users is not None:
            models["Popularity (demographic)"] = self.demo_popularity
        models.update({
            "CF only (ALS)": self.als,
            "Content only": self.content,
            "Hybrid 50/50 fixed": self.hybrids["fixed"],
            "Hybrid switch at 5": self.hybrids["switch"],
            "Hybrid adaptive": self.hybrids["adaptive"],
        })
        return models


def _log(verbose, message):
    if verbose:
        print(message, flush=True)


def fit_models(cfg: Config, dataset: Dataset, split: Split, verbose: bool = True) -> Experiment:
    all_users = dataset.ratings["user_id"].unique()
    inter = Interactions(split.train, user_ids=all_users, item_ids=dataset.items["item_id"])
    _log(verbose, f"Matrix: {inter.n_users} users x {inter.n_items} items, "
                  f"{inter.matrix.nnz:,} training ratings")

    popularity = PopularityModel(use_demographics=False).fit(inter)
    demo_popularity = PopularityModel(use_demographics=True).fit(inter, dataset.users)

    start = time.time()
    als = ALSModel(cfg.als).fit(inter, verbose=verbose)
    _log(verbose, f"ALS trained in {time.time() - start:.1f}s")

    content = ContentModel(cfg.content).fit(inter, dataset.items)
    _log(verbose, f"Content features: {content.item_features.shape[1]} dims per item")

    experiment = Experiment(cfg, dataset, split, inter, popularity, demo_popularity, als, content)
    rebuild_hybrids(experiment)
    return experiment


def rebuild_hybrids(experiment: Experiment) -> None:
    """(Re)create the blends; cheap, so tuning can call it for every setting."""
    e = experiment
    for mode in HybridModel.MODES:
        e.hybrids[mode] = HybridModel(e.als, e.content, e.demo_popularity,
                                      e.inter.user_counts, e.inter.item_counts,
                                      e.cfg.hybrid, mode=mode,
                                      cold_threshold=e.cfg.split.cold_threshold)
    e.reranker = MMRReranker(e.content.item_features, e.inter.item_counts, e.cfg.rerank)


def build_experiment(cfg: Config, dataset: Dataset | None = None, verbose: bool = True) -> Experiment:
    dataset = dataset or load_dataset(cfg.dataset)
    split = make_split(dataset.ratings, cfg.split)
    _log(verbose, f"Split: {len(split.train):,} train / {len(split.test):,} test ratings; "
                  f"{len(split.coldified_users)} cold users, {len(split.coldified_items)} cold items")
    return fit_models(cfg, dataset, split, verbose)


# --- saving the trained model for the web app ---------------------------

def save_experiment(experiment: Experiment, path=None):
    path = path or ARTIFACTS_DIR / f"{experiment.cfg.dataset}.pkl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(experiment, f, protocol=pickle.HIGHEST_PROTOCOL)
    return path


def load_experiment(dataset_name: str = "ml-1m", path=None) -> Experiment:
    path = path or ARTIFACTS_DIR / f"{dataset_name}.pkl"
    with open(path, "rb") as f:
        return pickle.load(f)


def relevant_test_items(experiment: Experiment) -> dict[int, set]:
    """user position -> set of item positions the user rated >= 4 in test."""
    e = experiment
    liked = e.split.test[e.split.test["rating"] >= e.cfg.split.relevance_threshold]
    users = liked["user_id"].map(e.inter.user_pos).to_numpy()
    items = liked["item_id"].map(e.inter.item_pos).to_numpy()
    truth: dict[int, set] = {}
    for u, i in zip(users, items):
        truth.setdefault(int(u), set()).add(int(i))
    return truth


def eval_users(experiment: Experiment, truth: dict, seed: int = 0) -> np.ndarray:
    users = np.array(sorted(truth))
    limit = experiment.cfg.max_eval_users
    if limit is not None and len(users) > limit:
        users = np.sort(np.random.default_rng(seed).choice(users, limit, replace=False))
    return users
