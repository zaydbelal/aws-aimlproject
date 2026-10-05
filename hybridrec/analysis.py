"""Failure analysis: where does the hybrid still recommend poorly, and why?

Every function returns plain data (dicts / DataFrames), so the same numbers
feed the markdown report, the figures and the web app's Evaluation page.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .evaluate import top_k
from .pipeline import Experiment

HYBRID = "Hybrid adaptive"
HISTORY_BUCKETS = [(0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (5, 19), (20, 49),
                   (50, 99), (100, 199), (200, 10**9)]


def bucket_label(low: int, high: int) -> str:
    if low == high:
        return str(low)
    return f"{low}+" if high >= 10**9 else f"{low}-{high}"


def by_history(per_user: pd.DataFrame, models: list[str], k: int) -> pd.DataFrame:
    """NDCG@k per model for users grouped by how many training ratings they have."""
    rows = []
    for low, high in HISTORY_BUCKETS:
        chosen = per_user[(per_user["n_train"] >= low) & (per_user["n_train"] <= high)]
        row = {"history": bucket_label(low, high),
               "users": int(chosen["user"].nunique())}
        for model in models:
            row[model] = float(chosen.loc[chosen["model"] == model, f"ndcg@{k}"].mean())
        rows.append(row)
    return pd.DataFrame(rows)


def weights_by_history(experiment: Experiment) -> pd.DataFrame:
    """Average trust the adaptive hybrid puts in each signal, per history bucket.

    Averaged over movies in proportion to how often they are watched, so the
    numbers describe the movies people actually meet, not the obscure tail.
    """
    hybrid = experiment.hybrids["adaptive"]
    watched = experiment.inter.item_counts / experiment.inter.item_counts.sum()
    rows = []
    for low, high in HISTORY_BUCKETS:
        n = low if high >= 10**9 else (low + high) / 2
        w = hybrid.weights([n])
        rows.append({"history": bucket_label(low, high),
                     **{f"{name} weight": float(w[name][0] @ watched) for name in w}})
    return pd.DataFrame(rows)


def mainstreamness(experiment: Experiment, per_user: pd.DataFrame, k: int) -> pd.DataFrame:
    """Does the hybrid fail people with unusual taste?

    A user's mainstreamness = average popularity percentile of the movies they
    rated in training. We cut warm users into five equal groups by it.
    """
    inter = experiment.inter
    pct = np.argsort(np.argsort(inter.item_counts)) / (inter.n_items - 1)
    hybrid = per_user[(per_user["model"] == HYBRID) & (per_user["n_train"] >= 5)].copy()
    hybrid["mainstream"] = [pct[inter.user_history(u)[0]].mean() for u in hybrid["user"]]
    hybrid["group"] = pd.qcut(hybrid["mainstream"], 5, labels=[
        "1 most niche", "2", "3", "4", "5 most mainstream"])
    table = hybrid.groupby("group", observed=True).agg(
        users=("user", "size"), mainstream=("mainstream", "mean"),
        ndcg=(f"ndcg@{k}", "mean"), recall=(f"recall@{k}", "mean")).reset_index()
    return table


def popularity_bias(experiment: Experiment, results: dict, k: int) -> dict:
    """Recall of relevant movies split into the popular head vs the long tail."""
    inter = experiment.inter
    order = np.argsort(-inter.item_counts)
    head = np.zeros(inter.n_items, dtype=bool)
    # Head = the most-watched movies that together hold 50% of all ratings.
    cumulative = np.cumsum(inter.item_counts[order]) / inter.item_counts.sum()
    head[order[: int(np.searchsorted(cumulative, 0.5)) + 1]] = True

    users, truth = results["_users"], results["_truth"]
    out = {"head_items": int(head.sum()), "tail_items": int((~head).sum()), "models": {}}
    for model in ["Popularity (demographic)", "CF only (ALS)", HYBRID,
                  "Hybrid adaptive + MMR re-rank"]:
        if model not in results["_lists"]:
            continue
        lists = results["_lists"][model][:, :k]
        hits = {"head": [0, 0], "tail": [0, 0]}
        for ranked, u in zip(lists, users):
            ranked = set(ranked)
            for item in truth[int(u)]:
                side = "head" if head[item] else "tail"
                hits[side][1] += 1
                hits[side][0] += item in ranked
        recommended_head = float(head[lists].mean())
        out["models"][model] = {
            "recall_head": hits["head"][0] / max(hits["head"][1], 1),
            "recall_tail": hits["tail"][0] / max(hits["tail"][1], 1),
            "share_of_recs_from_head": recommended_head,
        }
    out["share_of_relevant_in_head"] = float(
        np.mean([head[i] for u in users for i in truth[int(u)]]))
    return out


def genre_recall(experiment: Experiment, results: dict, k: int) -> pd.DataFrame:
    """For relevant test movies of each genre: how often did the hybrid surface them?"""
    items = experiment.dataset.items.set_index("item_id").loc[experiment.inter.item_ids]
    genres = items["genres"].to_numpy()
    lists = results["_lists"][HYBRID][:, :k]
    hits, totals, recommended = {}, {}, {}
    for ranked, u in zip(lists, results["_users"]):
        ranked_set = set(ranked)
        for item in ranked:
            for g in genres[item]:
                recommended[g] = recommended.get(g, 0) + 1
        for item in results["_truth"][int(u)]:
            for g in genres[item]:
                totals[g] = totals.get(g, 0) + 1
                hits[g] = hits.get(g, 0) + (item in ranked_set)
    all_relevant = sum(totals.values())
    all_recommended = sum(recommended.values())
    rows = [{"genre": g, "relevant_pairs": totals[g], "recall": hits[g] / totals[g],
             "share_of_relevant": totals[g] / all_relevant,
             "share_of_recommendations": recommended.get(g, 0) / all_recommended}
            for g in totals if totals[g] >= 200]
    return pd.DataFrame(rows).sort_values("recall").reset_index(drop=True)


def cold_user_first_rating(experiment: Experiment, per_user: pd.DataFrame, k: int) -> pd.DataFrame:
    """Cold users whose few ratings are low: a dislike is still 'watched' to CF."""
    inter = experiment.inter
    cold = per_user[(per_user["model"] == HYBRID) & per_user["n_train"].between(1, 4)].copy()
    cold["mean_rating"] = [inter.user_history(u)[1].mean() for u in cold["user"]]
    cold["their ratings"] = pd.cut(cold["mean_rating"], [0, 2.99, 3.99, 5],
                                   labels=["mostly dislikes (<3)", "lukewarm (3-4)",
                                           "mostly likes (>=4)"])
    return cold.groupby("their ratings", observed=True).agg(
        users=("user", "size"), ndcg=(f"ndcg@{k}", "mean")).reset_index()


def titles(experiment: Experiment, positions) -> list[str]:
    items = experiment.dataset.items.set_index("item_id")
    out = []
    for p in positions:
        row = items.loc[experiment.inter.item_ids[p]]
        year = "" if pd.isna(row["year"]) else f" ({int(row['year'])})"
        out.append(f"{row['title']}{year}")
    return out


def case_studies(experiment: Experiment, results: dict, k: int, n: int = 3) -> list[dict]:
    """The cold users (1-4 ratings) for whom the hybrid did worst."""
    per_user = results["_per_user"]
    cold = per_user[(per_user["model"] == HYBRID) & per_user["n_train"].between(1, 4)]
    # Among the zero-NDCG cases prefer users with many liked test movies:
    # plenty of targets and still a miss is the clearest failure.
    worst = cold.sort_values([f"ndcg@{k}", "n_relevant"], ascending=[True, False]).head(n)
    users = list(results["_users"])
    cases = []
    for _, row in worst.iterrows():
        u = int(row["user"])
        history, ratings = experiment.inter.user_history(u)
        recs = results["_lists"][HYBRID][users.index(u)][:5]
        liked = sorted(results["_truth"][u], key=lambda i: -experiment.inter.item_counts[i])[:5]
        cases.append({
            "user_id": int(experiment.inter.user_ids[u]),
            "history": [f"{t} - {r:g} stars" for t, r in zip(titles(experiment, history), ratings)],
            "recommended": titles(experiment, recs),
            "actually_liked": titles(experiment, liked),
            "liked_count": len(results["_truth"][u]),
        })
    return cases


def hardest_cold_items(experiment: Experiment, k: int, n: int = 6) -> list[dict]:
    """Cold movies that many users liked in test but the hybrid ranks badly."""
    inter = experiment.inter
    test = experiment.split.test
    liked = test[test["rating"] >= experiment.cfg.split.relevance_threshold]
    cold_mask = inter.item_counts < experiment.cfg.split.cold_threshold
    liked = liked.assign(item=liked["item_id"].map(inter.item_pos),
                         user=liked["user_id"].map(inter.user_pos))
    liked = liked[cold_mask[liked["item"].to_numpy()]]
    popular = liked["item"].value_counts().head(40).index.to_numpy()

    hybrid = experiment.hybrids["adaptive"]
    rows = []
    for item in popular:
        all_fans = liked.loc[liked["item"] == item, "user"].to_numpy()
        fans = all_fans[:200]                  # a sample keeps this step quick
        lists = top_k(hybrid, experiment, fans, k, candidate_mask=cold_mask)
        hit_rate = float(np.mean([item in row for row in lists]))
        meta = experiment.dataset.items.set_index("item_id").loc[inter.item_ids[item]]
        similar = experiment.content.similar_items(item, 3)
        rows.append({"title": titles(experiment, [item])[0], "genres": ", ".join(meta["genres"]),
                     "train_ratings": int(inter.item_counts[item]), "fans_in_test": len(all_fans),
                     "hit_rate": hit_rate,
                     "nearest_by_content": titles(experiment, similar)})
    rows.sort(key=lambda r: (r["hit_rate"], -r["fans_in_test"]))
    return rows[:n]


def content_ablation(experiment: Experiment, k: int) -> pd.DataFrame:
    """Which metadata block helps the cold-item slice? Content-only NDCG with blocks removed."""
    from dataclasses import replace

    from .evaluate import per_user_metrics
    from .models import ContentModel
    from .pipeline import eval_users, relevant_test_items

    inter = experiment.inter
    cold_mask = inter.item_counts < experiment.cfg.split.cold_threshold
    truth = relevant_test_items(experiment)
    truth = {u: {i for i in s if cold_mask[i]} for u, s in truth.items()}
    truth = {u: s for u, s in truth.items() if s}
    users = eval_users(experiment, truth)
    base = experiment.cfg.content
    variants = {
        "all blocks": base,
        "genres only": replace(base, year_weight=0.0, text_weight=0.0),
        "without decade": replace(base, year_weight=0.0),
        "without title/tag text": replace(base, text_weight=0.0),
    }
    rows = []
    for name, cfg in variants.items():
        model = ContentModel(cfg).fit(inter, experiment.dataset.items)
        lists = top_k(model, experiment, users, k, candidate_mask=cold_mask)
        score = per_user_metrics(lists, users, truth, [k])[f"ndcg@{k}"].mean()
        rows.append({"content features": name, f"ndcg@{k} on cold items": float(score)})
    return pd.DataFrame(rows)


def run_analysis(experiment: Experiment, results: dict, verbose: bool = True) -> dict:
    k = experiment.cfg.k
    models = [m for m in ["Popularity (demographic)", "Popularity", "CF only (ALS)",
                          "Content only", HYBRID] if m in results["_lists"]]
    if verbose:
        print("Failure analysis ...")
    return {
        "by_history": by_history(results["_per_user"], models, k),
        "weights": weights_by_history(experiment),
        "mainstreamness": mainstreamness(experiment, results["_per_user"], k),
        "popularity_bias": popularity_bias(experiment, results, k),
        "genres": genre_recall(experiment, results, k),
        "cold_user_ratings": cold_user_first_rating(experiment, results["_per_user"], k),
        "case_studies": case_studies(experiment, results, k),
        "hardest_cold_items": hardest_cold_items(experiment, k),
        "content_ablation": content_ablation(experiment, k),
    }
