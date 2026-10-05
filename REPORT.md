# Write-up: a hybrid recommender that degrades gracefully

*AIML-01 · MovieLens 1M · all numbers are NDCG@10 on the held-out test set
unless stated. Full tables: [`results/results_ml-1m.md`](results/results_ml-1m.md).*

## 1. Approach

**Evaluation first.** MovieLens 1M has no cold users (everyone rated ≥ 20
movies), so `split.py` creates them, the way a real service meets them.
Warm users train on the oldest 80% of their ratings and are tested on the
newest 20%. 10% of users keep only their first 0-4 ratings, and 5% of
movies keep only their first 0-4 ratings. A rating of ≥ 4★ in the test
period counts as relevant. We rank the whole catalog minus seen movies
and report Precision/Recall/NDCG/Hit-rate@10 per slice: warm users, cold
users (also by exactly 0…4 ratings), and **cold movies**, where only movies
with < 5 training ratings are candidates. RMSE is deliberately absent: a
product shows a top-10 list, not predicted stars.

**Three signals.**
* *Collaborative filtering:* implicit-feedback ALS (Hu, Koren & Volinsky
  2008), written from scratch. Confidence = 1 + α·rating; 128 factors.
  New users are folded in with one ridge regression.
* *Content:* each movie is `[genres | decade | 64-d LSA embedding of title
  (+ tag) text]`. A user's profile is the sum of rated movies weighted by
  `rating − 2.5`, so dislikes push away. Score = cosine.
* *Popularity:* within the user's age × gender segment, smoothed towards
  global popularity.

**The blend.** Each signal is z-scored per user, then mixed with weights
that are recomputed for every user-movie pair and always sum to 1:

```
w_cf  = c_user · c_item                   c = n / (n + k): "how well do we know it"
w_cb  = (1 − w_cf) · t_user               t_user = n_user / (n_user + k_content)
w_pop = (1 − w_cf) · (1 − t_user)
```

CF is used only where **both** sides have history. The rest goes to
content once the user has a few ratings, and to popularity before that.
Tuned values: `k_user = 30`, `k_item = 300`, `k_content = 30`.

## 2. Key decisions

1. **Tune on validation, never on test.** `tune.py` re-splits the training
   data with the same procedure (it has its own cold users and movies). The
   blend objective is the **geometric mean** of NDCG on warm users, cold
   users and cold movies. A plain average let the cold-user slice (NDCG
   ≈ 0.3) drown out the cold-movie slice (≈ 0.03); with the geometric mean,
   a 10% gain counts the same everywhere. The blend optimum is interior to
   its grid (we widened `k_item` when it first landed on the edge). For ALS,
   128 factors and α = 1 sit at the edge of the grid, but neighbouring
   settings are within about 1%.
2. **The formula was chosen by evidence.** Version 1, CF plus one fixed
   "cold score" (content + popularity), could not serve cold users (who
   need popularity) and cold movies (where popularity hurts) at once.
   Version 2 gated popularity by the movie's history as well. It sounded
   principled but lowered validation NDCG on cold movies from 0.032 to
   0.028: scaling z-scores by per-movie weights quietly *rewards* the
   least-known movies. Version 3 (above) scored best (geo-mean 0.101 vs
   0.090).
3. **Ties matter.** 1,017 cold movies have only 833 distinct content
   vectors (same genres and decade, no shared title words). Tie order alone
   moved content-only validation NDCG on cold movies from 0.029 to 0.042,
   so exact ties are broken by popularity.
4. **Implicit ALS over rating-prediction SVD,** because the task is choosing
   which unseen movies to show, not predicting stars. Low ratings still
   count as "watched", but with less confidence.

## 3. Results

| Model | All | Warm | Cold users | Cold movies | Worst slice / best |
|---|---:|---:|---:|---:|---:|
| Popularity (global / demographic) | 0.117 / 0.124 | 0.088 / 0.094 | 0.376 / **0.389** | 0.015 / 0.018 | 0.28 / 0.32 |
| CF only (ALS) | 0.120 | **0.108** | 0.229 | **0.056** | 0.59 |
| Content only | 0.031 | 0.028 | 0.063 | 0.050 | 0.16 |
| Hybrid fixed 50/50 | 0.105 | 0.092 | 0.223 | 0.051 | 0.57 |
| Hybrid hard switch at 5 | 0.111 | 0.107 | 0.148 | 0.050 | 0.38 |
| **Hybrid adaptive** | **0.131** | 0.105 | 0.367 | 0.054 | **0.94** |

* **No slice breaks it.** Every alternative collapses somewhere. CF gets
  0.006 for brand-new users (its scores are all zero). Popularity gets
  0.018 on new movies. The 50/50 mix is dragged down by content (0.223
  for cold users). The hard switch hands 1-4-rating users to content,
  which ranks badly (0.148). The adaptive hybrid stays within 94% of the
  best model on every slice and is best overall (+9% over CF, +6% over
  demographic popularity, +12% over plain popularity).
* **Brand-new users:** 0.432, exactly demographic popularity, the best
  available signal there. Demographics add +4% over global popularity.
* **Beyond accuracy:** catalog coverage is 22.7% vs 2.9% for popularity,
  and novelty is 2.25 vs 1.52 bits.
* **Bonus, MMR re-ranking** (λ = 0.7 plus a small popularity penalty):
  intra-list diversity rises from 0.62 to 0.67, and NDCG slightly *improves*
  (0.131 → 0.134), because near-duplicate sequels make way for other
  relevant movies. It did **not** raise novelty or coverage (2.25 → 2.20,
  22.7% → 21.9%): the top-100 candidate pool is already dominated by
  popular titles, so re-ranking cannot reach the tail.
* The same code on ml-latest-small (with user tags), untuned: the adaptive
  hybrid is again best overall (0.087) with the best worst-slice ratio
  (0.87 vs ≤ 0.75 for the rest).

**Caveat:** absolute NDCG is not comparable *across* history buckets. Cold
users keep *all* their remaining ratings as test targets, so they have far
more relevant movies, and that inflates their scores. Compare models within
a bucket.

## 4. Where the hybrid still fails, and why

From [`results/failure_analysis_ml-1m.md`](results/failure_analysis_ml-1m.md):

1. **Popularity bias / long tail.** 96% of recommendations come from the
   388 "head" movies that hold half of all ratings. Recall is 9.8% on
   relevant head movies and 0.14% on the tail, yet 61% of relevant test
   movies are tail movies. Implicit ALS, popularity and the z-scored blend
   all favour what is already popular, and MMR only re-orders a popular
   candidate pool.
2. **Heavy users are under-served.** For users with 200+ ratings the hybrid
   scores 0.170, against 0.188 for CF and 0.208 for popularity. Because
   `k_item = 300`, a moderately popular movie (~100 ratings) gets only ~22%
   CF trust even from a user with 200 ratings, and content (the weakest
   ranker for warm users) picks up much of the rest. The single `k_item`
   compromise suits cold movies but over-shrinks CF for the mid-catalogue.
3. **Blockbusters treated as new releases.** The cold movies with the most
   fans (A Few Good Men, 764 fans; Top Gun, 622; Bound; Body Heat) reach
   **0%** of them. Content sees "Action, Romance, 1980s" and finds
   Excalibur and The Jewel of the Nile. Metadata cannot tell a hit from a
   flop, and with 0-3 ratings CF and popularity have no evidence either.
4. **Genres systematically missed.** Recall of liked movies is 0.06% for
   Documentary, 0.5% Western, 1.5% Horror and Musical, against 8-10% for
   Action, Sci-Fi and War. Action and Sci-Fi are over-recommended (16% and
   10% of recommendations vs 11% and 7% of liked movies). Niche genres are
   small, so both popularity and CF under-rank them.
5. **Niche taste.** Recall@10 for the most niche fifth of warm users is
   0.053, against 0.087 for the most mainstream fifth. (NDCG is slightly
   *higher* for niche users, 0.110 vs 0.098: we get their top few right
   but find fewer of their favourites.)
6. **Misleading first ratings.** Cold users whose few ratings are mostly
   dislikes score 0.330, against 0.369 when they are likes. Content uses
   the dislike as a negative signal, but implicit CF counts "watched and
   hated" as a positive one.
7. **Weak text features.** Removing the title-text embedding *improves*
   content-only NDCG on cold movies (0.050 → 0.055). On ML-1M titles
   carry almost no signal; they help only with sequels.
8. **Two ratings cannot reveal breadth.** User 998 rated *E.T.* 5★ and got
   Star Wars, Raiders and Back to the Future. In the test period they liked
   American Beauty, Saving Private Ryan and The Matrix. The hybrid
   extrapolates "family sci-fi" from one data point.

## 5. Next steps

* **Learn the blend.** Replace the three hand-shaped confidences with a
  small model (e.g. gradient-boosted trees on n_user, n_item, the three
  scores and the genre) trained on validation. That directly addresses
  failures 2 and 3.
* **Better item content.** Plot descriptions and sentence embeddings (e.g.
  TMDB synopses through a text-embedding model), cast and director, or
  the ML-25M tag genome. Also learn a mapping from content to the ALS
  item space, so a new movie gets a CF vector on day one.
* **Explicit negatives in CF.** Down-weight or exclude 1-2★ ratings, or
  use a model with a dislike signal (failure 6).
* **Long-tail exposure.** Calibrated re-ranking (match each user's genre
  mix), or a larger, popularity-debiased candidate pool before MMR.
* **Online validation.** Offline NDCG on a 20-year-old dataset is a proxy.
  An A/B test of click-through and long-term retention is the real test.
