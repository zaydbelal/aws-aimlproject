"""A small Flask app to explore the recommender.

Run with:  python -m hybridrec serve   (after `python -m hybridrec evaluate` or `train`)

JSON API:
    GET  /api/meta              dataset facts, example users, demographic options
    GET  /api/search?q=matrix   movie lookup for the "new user" mode
    GET  /api/user/<id>         a user's training history
    POST /api/recommend         recommendations + per-movie score breakdown
    GET  /api/results           evaluation results written by `evaluate`
"""

from __future__ import annotations

import os

for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from flask import Flask, jsonify, render_template, request  # noqa: E402

from hybridrec.config import ARTIFACTS_DIR, RESULTS_DIR  # noqa: E402
from hybridrec.pipeline import load_experiment  # noqa: E402

AGE_GROUPS = {1: "Under 18", 18: "18-24", 25: "25-34", 35: "35-44", 45: "45-49",
              50: "50-55", 56: "56+"}
MODELS = {
    "adaptive": "Hybrid, adaptive",
    "fixed": "Hybrid, fixed 50/50",
    "switch": "Hybrid, switch at 5",
    "cf": "Collaborative filtering only",
    "content": "Content only",
    "popularity": "Popularity only",
}
MAX_K = 50
SIGNALS = ("cf", "content", "popularity")


class ApiError(Exception):
    def __init__(self, message, status=400):
        super().__init__(message)
        self.message, self.status = message, status


class Recommender:
    """Wraps a trained Experiment with the lookups the web app needs."""

    def __init__(self, dataset_name: str, path=None):
        self.e = load_experiment(dataset_name, path)
        e = self.e
        items = e.dataset.items.set_index("item_id").loc[e.inter.item_ids].reset_index()
        self.items = items
        self.search_titles = items["title"].str.lower().to_numpy()
        self.has_demographics = e.dataset.users is not None
        self.watched_share = e.inter.item_counts / max(e.inter.item_counts.sum(), 1)

        liked = e.split.test[e.split.test["rating"] >= e.cfg.split.relevance_threshold]
        self.liked_later = liked.groupby("user_id")["item_id"].apply(set).to_dict()

    # --- lookups -------------------------------------------------------

    def movie(self, position: int) -> dict:
        row = self.items.iloc[position]
        return {
            "item_id": int(row["item_id"]),
            "title": row["title"],
            "year": None if pd.isna(row["year"]) else int(row["year"]),
            "genres": list(row["genres"]),
            "ratings_in_train": int(self.e.inter.item_counts[position]),
        }

    def search(self, query: str, limit: int = 8) -> list[dict]:
        query = query.strip().lower()
        if len(query) < 2:
            return []
        matches = np.flatnonzero(np.char.find(self.search_titles.astype(str), query) >= 0)
        # Most-rated first: the movie people mean is usually the famous one.
        matches = matches[np.argsort(-self.e.inter.item_counts[matches], kind="stable")]
        return [self.movie(p) for p in matches[:limit]]

    def example_users(self) -> list[dict]:
        """A few users at different history levels, for one-click demos."""
        e = self.e
        counts = e.inter.user_counts
        examples = []
        for label, n in [("new sign-up", 0), ("1 rating", 1), ("3 ratings", 3),
                         ("warm", 40), ("power user", 300)]:
            pool = [u for u in np.flatnonzero(counts == n if n < 5 else counts >= n)
                    if e.inter.user_ids[u] in self.liked_later]
            if pool:
                u = pool[len(pool) // 2]
                examples.append({"label": label, "user_id": int(e.inter.user_ids[u]),
                                 "ratings": int(counts[u])})
        return examples

    def user(self, user_id: int) -> dict:
        if user_id not in self.e.inter.user_pos:
            raise ApiError(f"No user with id {user_id}. Ids run from "
                           f"{self.e.inter.user_ids.min()} to {self.e.inter.user_ids.max()}.", 404)
        u = self.e.inter.user_pos[user_id]
        positions, ratings = self.e.inter.user_history(u)
        order = np.argsort(-ratings, kind="stable")
        profile = {}
        if self.has_demographics:
            row = self.e.dataset.users.set_index("user_id").loc[user_id]
            profile = {"gender": row["gender"], "age": AGE_GROUPS.get(int(row["age"]), "")}
        return {
            "user_id": user_id,
            "n_ratings": int(len(positions)),
            "liked_later": len(self.liked_later.get(user_id, ())),
            "profile": profile,
            "history": [{**self.movie(p), "rating": float(r)}
                        for p, r in zip(positions[order][:40], ratings[order][:40])],
        }

    # --- recommending --------------------------------------------------

    def _scores(self, model: str, user_index=None, history=None, gender=None, age=None):
        """Return (final scores, parts or None, history length)."""
        e = self.e
        if user_index is not None:
            n = int(e.inter.user_counts[user_index])
            u = np.array([user_index])
            if model in ("adaptive", "fixed", "switch"):
                final, parts = e.hybrids[model].explain_user(user_index)
                return final, parts, n
            single = {"cf": e.als, "content": e.content, "popularity": e.demo_popularity}[model]
            return single.score_users(u)[0], None, n

        items, ratings = history
        n = len(items)
        if model in ("adaptive", "fixed", "switch"):
            final, parts = e.hybrids[model].score_new_user(items, ratings, gender, age)
            return final, parts, n
        if model == "cf":
            return e.als.score_new_user(items, ratings), None, n
        if model == "content":
            return e.content.score_new_user(items, ratings), None, n
        return e.demo_popularity.score_new_user(gender, age).copy(), None, n

    def recommend(self, payload: dict) -> dict:
        model = payload.get("model", "adaptive")
        if model not in MODELS:
            raise ApiError(f"Unknown model {model!r}.")
        k = int(payload.get("k", 10))
        if not 1 <= k <= MAX_K:
            raise ApiError(f"k must be between 1 and {MAX_K}.")
        diversify = bool(payload.get("diversify", False))

        user_id = payload.get("user_id")
        if user_id is not None:
            try:
                user_id = int(user_id)
            except (TypeError, ValueError):
                raise ApiError("User id must be a whole number.")
            if user_id not in self.e.inter.user_pos:
                raise ApiError(f"No user with id {user_id}.", 404)
            u = self.e.inter.user_pos[user_id]
            final, parts, n = self._scores(model, user_index=u)
            seen = self.e.inter.user_history(u)[0]
            liked_later = self.liked_later.get(user_id, set())
        else:
            items, ratings = self._parse_history(payload.get("ratings", []))
            gender, age = payload.get("gender") or None, payload.get("age")
            if gender is not None and gender not in ("M", "F"):
                raise ApiError("Gender must be M or F.")
            age = int(age) if age not in (None, "") else None
            if age is not None and age not in AGE_GROUPS:
                raise ApiError("Unknown age group.")
            final, parts, n = self._scores(model, history=(items, ratings), gender=gender, age=age)
            seen, liked_later = items, set()

        scores = np.array(final, dtype=np.float64)
        scores[seen] = -np.inf
        if diversify:
            picked = self.e.reranker.rerank(scores, k)
        else:
            picked = np.argsort(-scores, kind="stable")[:k]

        results = []
        for rank, p in enumerate(picked, start=1):
            entry = {**self.movie(int(p)), "rank": rank, "score": float(scores[p]),
                     "liked_later": int(self.items.iloc[p]["item_id"]) in liked_later}
            if parts is not None:
                entry["parts"] = {name: float(parts[name][p]) for name in SIGNALS}
                entry["weights"] = {name: float(parts["weights"][name][p]) for name in SIGNALS}
            results.append(entry)

        return {
            "model": model,
            "model_label": MODELS[model],
            "history_size": n,
            # Average trust in each signal, weighting movies by how often they are watched.
            "mean_weights": ({name: float(parts["weights"][name] @ self.watched_share)
                              for name in SIGNALS} if parts is not None else None),
            "hits": int(sum(r["liked_later"] for r in results)),
            "liked_later_total": len(liked_later),
            "recommendations": results,
        }

    def _parse_history(self, raw) -> tuple[np.ndarray, np.ndarray]:
        if not isinstance(raw, list) or len(raw) > 200:
            raise ApiError("Ratings must be a list of at most 200 movies.")
        items, ratings = [], []
        for entry in raw:
            try:
                item_id, rating = int(entry["item_id"]), float(entry["rating"])
            except (KeyError, TypeError, ValueError):
                raise ApiError("Each rating needs an item_id and a rating.")
            if item_id not in self.e.inter.item_pos:
                raise ApiError(f"Unknown movie id {item_id}.")
            if not 0.5 <= rating <= 5:
                raise ApiError("Ratings go from 0.5 to 5 stars.")
            items.append(self.e.inter.item_pos[item_id])
            ratings.append(rating)
        return np.array(items, dtype=int), np.array(ratings, dtype=np.float64)

    def meta(self) -> dict:
        e = self.e
        return {
            "dataset": e.cfg.dataset,
            "users": int(e.inter.n_users),
            "movies": int(e.inter.n_items),
            "train_ratings": int(e.inter.matrix.nnz),
            "examples": self.example_users(),
            "models": MODELS,
            "demographics": self.has_demographics,
            "age_groups": [{"value": k, "label": v} for k, v in AGE_GROUPS.items()],
            "hybrid": {"k_user": e.cfg.hybrid.k_user, "k_item": e.cfg.hybrid.k_item,
                       "k_content": e.cfg.hybrid.k_content},
        }


def create_app(dataset_name: str = "ml-1m") -> Flask:
    app = Flask(__name__, static_folder="static", template_folder="templates")
    model_path = ARTIFACTS_DIR / f"{dataset_name}.pkl"
    if not model_path.exists():
        raise SystemExit(f"No trained model at {model_path}.\n"
                         f"Run: python -m hybridrec evaluate --dataset {dataset_name}")
    print(f"Loading {model_path} ...")
    rec = Recommender(dataset_name, model_path)
    results_path = RESULTS_DIR / f"metrics_{dataset_name}.json"

    @app.errorhandler(ApiError)
    def handle_api_error(error):
        return jsonify({"error": error.message}), error.status

    @app.get("/")
    def index():
        return render_template("index.html")

    @app.get("/api/meta")
    def meta():
        return jsonify(rec.meta())

    @app.get("/api/search")
    def search():
        return jsonify(rec.search(request.args.get("q", "")))

    @app.get("/api/user/<int:user_id>")
    def user(user_id):
        return jsonify(rec.user(user_id))

    @app.post("/api/recommend")
    def recommend():
        payload = request.get_json(silent=True)
        if not isinstance(payload, dict):
            raise ApiError("Send a JSON object.")
        return jsonify(rec.recommend(payload))

    @app.get("/api/results")
    def results():
        if not results_path.exists():
            raise ApiError("No evaluation results yet. Run: python -m hybridrec evaluate", 404)
        return app.response_class(results_path.read_text(), mimetype="application/json")

    return app
