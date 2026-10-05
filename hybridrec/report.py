"""Writes results/: metrics.json, results.md (tables), failure_analysis.md and figures."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from .config import RESULTS_DIR

# Four hues from the validated chart palette, checked for colour-blind safety in
# this order (blue, violet, aqua, yellow). Orange is skipped so it never clashes
# with the app's burnt-orange action colour. Each model keeps its colour everywhere.
SERIES_COLORS = {
    "Hybrid adaptive": "#2a78d6",
    "CF only (ALS)": "#4a3aa7",
    "Popularity (demographic)": "#1baf7a",
    "Popularity": "#1baf7a",
    "Content only": "#eda100",
}
INK, INK_MUTED, GRID, SURFACE = "#141413", "#6b6a64", "#e4e2dc", "#f9f8f6"

HEADLINE_METRICS = ["precision@{k}", "recall@{k}", "ndcg@{k}", "hit@{k}",
                    "coverage", "novelty", "diversity"]


def _table(frame: pd.DataFrame, floatfmt: str = "{:.4f}") -> str:
    """Tiny markdown table writer (avoids needing the 'tabulate' package)."""
    header = "| " + " | ".join(str(c) for c in frame.columns) + " |"
    rule = "|" + "|".join("---" if frame[c].dtype == object else "---:" for c in frame.columns) + "|"
    lines = [header, rule]
    for _, row in frame.iterrows():
        cells = [floatfmt.format(v) if isinstance(v, (float, np.floating)) else str(v)
                 for v in row]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def slice_table(results: dict, slice_name: str, k: int) -> pd.DataFrame:
    rows = []
    for model, values in results["slices"][slice_name].items():
        row = {"model": model}
        for metric in HEADLINE_METRICS:
            name = metric.format(k=k)
            row[name] = values.get(name, np.nan)
        rows.append(row)
    return pd.DataFrame(rows)


def _clean(value):
    """Make values JSON-safe: NaN -> null (browsers reject NaN), numpy -> python."""
    if isinstance(value, dict):
        return {str(k): _clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(v) for v in value]
    if isinstance(value, pd.DataFrame):
        return _clean(value.to_dict(orient="records"))
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return None if np.isnan(value) else float(value)
    return value


MAIN_SLICES = {"all users": "All users", "warm users": "Warm users",
               "cold users (": "Cold users", "cold items": "Cold movies"}


def robustness_table(results: dict, k: int) -> pd.DataFrame:
    """NDCG@k per main slice, plus each model's worst slice relative to the best model there.

    A model that is great on average but collapses on one slice gets a low
    'worst vs best' value; that is exactly the failure the brief is about.
    """
    metric = f"ndcg@{k}"
    names = {label: next(s for s in results["slices"] if s.startswith(prefix))
             for prefix, label in MAIN_SLICES.items()}
    models = list(results["slices"][names["Cold movies"]])   # models scored on every slice
    rows = []
    for model in models:
        row = {"model": model}
        for label, name in names.items():
            row[label] = results["slices"][name][model][metric]
        rows.append(row)
    table = pd.DataFrame(rows)
    ratios = [table[label] / table[label].max() for label in names if label != "All users"]
    table["worst slice vs best"] = pd.concat(ratios, axis=1).min(axis=1)
    return table


def json_ready(results: dict, analysis: dict) -> dict:
    out = {key: value for key, value in results.items() if not key.startswith("_")}
    out["analysis"] = analysis
    out["summary"] = robustness_table(results, results["config"]["k"])
    return _clean(out)


def plot_history(by_history: pd.DataFrame, k: int, path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
    fig, ax = plt.subplots(figsize=(8, 4.2), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    x = np.arange(len(by_history))
    models = [m for m in SERIES_COLORS if m in by_history.columns and m != "Popularity"]
    for model in models:
        y = by_history[model].to_numpy()
        ax.plot(x, y, color=SERIES_COLORS[model], linewidth=2, solid_capstyle="round",
                marker="o", markersize=5, markeredgecolor=SURFACE, markeredgewidth=1.5,
                label=model)
        ax.annotate(f"{model}  {y[-1]:.2f}", (x[-1], y[-1]), xytext=(8, 0),
                    textcoords="offset points", va="center", color=INK, fontsize=8.5)
    ax.set_xticks(x, by_history["history"])
    ax.set_xlabel("Training ratings the user has", color=INK_MUTED)
    ax.set_ylabel(f"NDCG@{k}", color=INK_MUTED)
    ax.axvline(4.5, color=GRID, linewidth=1)
    ax.text(4.4, ax.get_ylim()[1] * 0.97, "cold  ", ha="right", va="top", color=INK_MUTED, fontsize=8.5)
    ax.text(4.6, ax.get_ylim()[1] * 0.97, "  warm", ha="left", va="top", color=INK_MUTED, fontsize=8.5)
    ax.grid(axis="y", color=GRID, linewidth=1)
    for side in ["top", "right", "left"]:
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=INK_MUTED, length=0)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=4, frameon=False, fontsize=8.5)
    ax.set_xlim(-0.3, len(x) + 1.8)
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def write_reports(experiment, results: dict, analysis: dict) -> None:
    k = experiment.cfg.k
    RESULTS_DIR.mkdir(exist_ok=True)
    figures = RESULTS_DIR / "figures"
    figures.mkdir(exist_ok=True)
    name = experiment.cfg.dataset

    (RESULTS_DIR / f"metrics_{name}.json").write_text(
        json.dumps(json_ready(results, analysis), indent=2))

    # ---- results.md --------------------------------------------------
    d = results["dataset"]
    lines = [f"# Results - {name}", "",
             "Generated by `python -m hybridrec evaluate`. Do not edit by hand.", "",
             f"{d['users']:,} users, {d['items']:,} movies, {d['train_ratings']:,} training / "
             f"{d['test_ratings']:,} test ratings. Evaluated on {d['eval_users']:,} users "
             f"({d['cold_eval_users']:,} cold). Cold-item slice: {d['cold_items']:,} movies "
             f"with < 5 training ratings, {d['cold_item_eval_users']:,} users.", "",
             f"Relevant = rated >= {experiment.cfg.split.relevance_threshold:g} stars in the test "
             f"period. Lists of K = {k}. Coverage/novelty/diversity describe the lists, "
             "not accuracy.", "",
             f"## Summary: NDCG@{k} by slice", "",
             "`worst slice vs best` = the model's lowest NDCG across the warm, cold-user and "
             "cold-movie slices, as a share of the best model on that slice. 1.00 = never "
             "beaten anywhere.", "",
             _table(robustness_table(results, k), "{:.3f}"), ""]
    for slice_name in results["slices"]:
        lines += [f"## {slice_name[0].upper() + slice_name[1:]}", "",
                  _table(slice_table(results, slice_name, k)), ""]
    (RESULTS_DIR / f"results_{name}.md").write_text("\n".join(lines))

    # ---- failure_analysis.md -----------------------------------------
    plot_history(analysis["by_history"], k, figures / f"ndcg_by_history_{name}.png")
    a = analysis
    pb = a["popularity_bias"]
    pb_rows = pd.DataFrame([{"model": m, "recall on head movies": v["recall_head"],
                             "recall on long-tail movies": v["recall_tail"],
                             "share of recs from head": v["share_of_recs_from_head"]}
                            for m, v in pb["models"].items()])
    lines = [f"# Failure analysis - {name}", "",
             "Generated by `python -m hybridrec evaluate`. The interpretation lives in REPORT.md.", "",
             f"## 1. Accuracy vs. how much history a user has (NDCG@{k})", "",
             f"![NDCG by history](figures/ndcg_by_history_{name}.png)", "",
             _table(a["by_history"]), "",
             "Average trust the adaptive hybrid puts in each signal at each history level "
             "(averaged over movies in proportion to how often they are watched; the three "
             "add up to 1):", "",
             _table(a["weights"], "{:.3f}"), "",
             "## 2. Popularity bias: head vs. long tail", "",
             f"Head = the {pb['head_items']} most-watched movies that hold half of all training "
             f"ratings; tail = the other {pb['tail_items']}. "
             f"{pb['share_of_relevant_in_head']:.0%} of the relevant test movies are head movies.", "",
             _table(pb_rows), "",
             "## 3. Users with unusual taste (warm users, by mainstreamness quintile)", "",
             _table(a["mainstreamness"]), "",
             "## 4. Genres the hybrid misses (recall of relevant movies per genre, lowest first)", "",
             _table(a["genres"]), "",
             "## 5. Cold users whose few ratings are dislikes", "",
             _table(a["cold_user_ratings"]), "",
             "## 6. Content features ablation (content-only model, cold-item slice)", "",
             _table(a["content_ablation"]), "",
             "## 7. Cold movies the hybrid struggles to surface", "",
             "Among cold movies with the most fans in the test period. Hit rate = share of "
             f"those fans who got the movie in their top {k} cold-movie list.", ""]
    for row in a["hardest_cold_items"]:
        lines.append(f"- **{row['title']}** ({row['genres']}) - {row['train_ratings']} training "
                     f"ratings, {row['fans_in_test']} fans, hit rate {row['hit_rate']:.0%}. "
                     f"Nearest by content: {'; '.join(row['nearest_by_content'])}")
    lines += ["", "## 8. Worst cold-user cases", ""]
    for case in a["case_studies"]:
        lines += [f"**User {case['user_id']}** ({case['liked_count']} liked movies in test)", "",
                  f"- Training history: {'; '.join(case['history'])}",
                  f"- Hybrid recommended: {'; '.join(case['recommended'])}",
                  f"- Actually liked (most popular first): {'; '.join(case['actually_liked'])}", ""]
    (RESULTS_DIR / f"failure_analysis_{name}.md").write_text("\n".join(lines))
    print(f"Wrote results to {RESULTS_DIR}")
