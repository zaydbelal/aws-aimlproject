# Study guide: how this project works

Read this if you need to understand or explain the project, for example to a
reviewer. It walks through the code in the order the data flows, then
answers the questions you are most likely to be asked.

---

## The problem in one paragraph

A streaming service wants to show each person a short list of movies they
will like. The classic method, **collaborative filtering (CF)**, learns from
"people who watched what you watched also watched…". It needs history. A
person who just signed up has none, and a movie released yesterday has
none. Plain CF has nothing to say about either. Our job is a recommender
that **gets gradually worse** as history disappears, instead of breaking.

## The idea in one picture

Three signals, each trusted only as far as the data behind it allows:

| Signal | Needs |
|---|---|
| Collaborative filtering (CF) | history on the user **and** on the movie |
| Content similarity | a few ratings from the user; nothing on the movie |
| Popularity | nothing from the user |

```
c_user = n_user / (n_user + k_user)       CF trust in the user's vector
c_item = n_item / (n_item + k_item)       CF trust in the movie's vector
t_user = n_user / (n_user + k_content)    trust in the user's content profile

w_cf   = c_user · c_item
w_cb   = (1 − w_cf) · t_user              what CF can't cover goes to content…
w_pop  = (1 − w_cf) · (1 − t_user)        …and what content can't cover to popularity

final  = w_cf · CF  +  w_cb · Content  +  w_pop · Popularity      (weights add up to 1)
```

`n_user` is how many movies the user rated, `n_item` how many people rated
the movie, and each `k` reads "how many ratings until we trust this 50%".
The weights are computed **separately for every user-movie pair**:

* brand-new user → 100% popularity (what people of their age and gender watch)
* user with a handful of ratings → mostly popularity, some content and CF
* regular user, movie released yesterday → `c_item = 0`, so content and popularity decide
* regular user, well-known movie → mostly CF

---

## Walking through the code

| Step | File | What happens |
|---|---|---|
| 1 | `hybridrec/data.py` | Download MovieLens, parse it into three tables: ratings, items (title, year, genres, tags), users (gender, age). |
| 2 | `hybridrec/split.py` | Build train/test and **manufacture** cold users and cold movies. |
| 3 | `hybridrec/interactions.py` | Turn the ratings table into a sparse user × movie matrix. |
| 4 | `hybridrec/models/popularity.py` | Baseline: most-watched movies (optionally within your age/gender group). |
| 5 | `hybridrec/models/als.py` | Collaborative filtering by matrix factorization (ALS). |
| 6 | `hybridrec/models/content.py` | Content-based scores from genres, decade and title/tag text. |
| 7 | `hybridrec/models/hybrid.py` | The adaptive blend above. |
| 8 | `hybridrec/models/rerank.py` | Bonus: MMR re-ranking for diversity and novelty. |
| 9 | `hybridrec/metrics.py` | Precision@K, Recall@K, NDCG@K, hit rate, coverage, novelty, diversity. |
| 10 | `hybridrec/evaluate.py` | Rank, compare with what users actually did, slice the results. |
| 11 | `hybridrec/tune.py` | Choose hyperparameters on a validation split. |
| 12 | `hybridrec/analysis.py` | Failure analysis. |
| 13 | `hybridrec/report.py` | Write `results/`. |
| 14 | `app/` | The web app. |

### Step 2: the split (the part reviewers ask about most)

MovieLens 1M has no cold users: everyone rated at least 20 movies. So we
**hide** data to create them:

* **Warm users:** sort each user's ratings by time; the oldest 80% train,
  the newest 20% test. We predict the future from the past, as a real
  service must.
* **Cold users (10% of users):** keep only their first `n` ratings in
  training, `n` drawn at random from 0-4. Everything else goes to test.
  `n = 0` is a person who just signed up.
* **Cold movies (5% of movies):** keep only the first `m` ratings each
  movie received, `m` from 0-4. Everything else goes to test.
* A cold user's rating of a cold movie always goes to test, so neither one
  ends up with 5 or more training interactions.

"Relevant" in the test set means **rated 4 or 5 stars**. A movie you
watched and hated is not a success.

### Step 5: ALS in plain words

Give every user a list of 128 numbers (a vector `x_u`) and every movie
another 128 numbers (`y_i`), and predict the score as their dot product.
Nobody tells the model what the numbers mean; it discovers "taste
directions" on its own.

We use **implicit** ALS (Hu, Koren & Volinsky 2008): instead of predicting
the star rating, predict *did the user watch it* (1 or 0). Higher ratings
count as more *confidence* that the answer is 1:
`confidence = 1 + alpha × rating`.

**Why "alternating":** with all movie vectors frozen, finding the best user
vector is ordinary ridge regression with an exact formula. So we freeze
the movies and solve every user, then freeze the users and solve every
movie, and repeat about 12 times.

**Why it breaks for cold start:** with 0 ratings there is nothing to
regress on, so the vector is all zeros and every score is 0. With 1-4
ratings the vector is a noisy guess.

**Fold-in:** for a brand-new user in the web app we run the same ridge
regression on their few ratings, keeping the movie vectors fixed.

### Step 6: content-based in plain words

Each movie becomes a vector:

```
[ genres: Action=1, Comedy=0, ... | decade: 1990s=1, ... | 64-number text embedding ]
```

The text embedding is **TF-IDF** of the title words (plus user tags in the
newer MovieLens versions), squeezed to 64 numbers with **truncated SVD**.
That pairing is called Latent Semantic Analysis. Words that appear
together end up in the same direction, so "Star Trek II" lands near "Star
Trek III".

A user's profile is the sum of the movies they rated, weighted by
`rating − 2.5`: a 5-star rating pulls the profile towards a movie and a
1-star rating pushes it away. The score is the cosine similarity between
the profile and each movie.

A new movie has a genre and a title on day one, so content scores it
immediately. One rating is enough to give a user a profile.

### Step 7: why z-scores before blending

CF scores might range from −0.1 to 1.2, cosine similarities from −1 to 1,
log-popularity from −14 to −5. Adding raw numbers would let one signal
dominate by accident of scale. So for each user we **standardise each
signal**: subtract that user's mean score and divide by the standard
deviation. Now "+2" means "two standard deviations above this user's
average" for every signal. A signal with no information (all zeros, e.g.
CF for a new user) becomes all zeros and drops out of the blend.

### Step 7b: how we arrived at this formula (worth telling a reviewer)

Every candidate was compared on the **validation** split, never on test:

1. **CF + one "cold score" (content + a fixed dose of popularity).** One
   popularity dose could not suit both cold users (who need a lot) and
   cold movies (where it hurts).
2. **Gate popularity by the movie's history as well** ("a movie with 3
   ratings has no meaningful popularity"). This sounded principled but
   scored *worse* on cold movies. Multiplying z-scores by per-movie weights
   quietly *rewards* the least-known movies, and among new movies a few
   early ratings are actually a useful hint.
3. **The final formula:** only CF is gated by the movie's history; the
   leftover weight is split between content and popularity by how much
   the *user* has rated. Best validation score, and each weight has a
   plain-English meaning.

We also found that many movies have **identical content vectors** (same
genres and decade, no shared title words), so tie-breaking alone moved
content-only NDCG on cold movies from 0.029 to 0.042. The content model now
breaks exact ties by popularity.

### Step 8: MMR re-ranking (bonus)

Take the hybrid's top 100 and build the final 10 one at a time. At each
step pick the movie with the best

`λ × relevance − (1 − λ) × (highest similarity to anything already picked)`

minus a small popularity penalty. This trades a little accuracy for more
varied and less obvious lists. We report the trade-off in coverage,
novelty and diversity.

### Step 9: the metrics

For one user with 3 relevant test movies and a top-10 list holding 2 of them:

* **Precision@10** = 2/10. How much of the list was good.
* **Recall@10** = 2/3. How much of the good stuff we found.
* **NDCG@10**: hits near the top are worth more (gain = 1/log2(rank+1)),
  divided by the best achievable score. 1.0 = perfect order.
* **Hit rate@10** = 1. Was there at least one good movie?
* **Coverage**: share of the catalog that appears in anyone's list.
* **Novelty**: average −log2(share of users who watched the movie).
  Higher = less obvious.
* **Diversity**: average dissimilarity between movies within a list.

Why not RMSE? RMSE checks how close predicted stars are to real stars, on
movies the user *already chose to watch*. A product shows a short list
chosen from thousands of movies, so ordering is what matters. A model can
have great RMSE and useless top-10s.

### Step 11: tuning without cheating

Hyperparameters are **never chosen on the test set**. `tune.py` re-splits
the training data with the same procedure, which gives an inner
train/validation pair that has its own cold users and movies. It grid-searches:

1. ALS: factors, regularization, alpha (by CF NDCG on warm validation users).
2. Blend: `k_user`, `k_item` and `k_content`, by the **geometric mean** of
   NDCG on warm users, cold users and cold movies. Scoring the three slices
   separately stops the large warm group (90% of users) from drowning out
   the cold cases. The geometric mean makes a 10% gain count the same in
   every slice, even though cold-user NDCG is about ten times larger than
   cold-movie NDCG. (A plain average let the cold-user slice dominate.)

---

## Questions you may be asked

**Why not a fixed 50/50 blend?** For a user with 300 ratings, CF is far
more accurate than content, so half the weight on content is wasted. For a
new user, CF is all zeros, so half the weight sits on noise. The results
table compares the two directly.

**Why not a hard switch at 5 ratings?** Information grows smoothly: a user
with 4 ratings and one with 6 are nearly the same person, but a hard switch
treats them completely differently. We evaluate exactly that switch (CF if
both sides have ≥ 5 ratings, else content, else popularity) as a baseline.

**Why does popularity beat everything for cold users?** With 0-4 ratings
there is very little to personalise from, and MovieLens tastes are
concentrated on famous movies. That is a real, well-known result, not a
bug. The hybrid's job is to *match* popularity there (it uses it as a
fallback) and then pull ahead as history grows.

**Why implicit ALS rather than rating-prediction SVD?** We are ranking
movies the user has *not* seen. Implicit ALS learns from what people choose
to watch, which is exactly that question. Rating-prediction SVD learns
"given you watched it, how many stars", which ignores the choice.

**Why rating − 2.5 for content profiles?** It makes 1-2 star ratings
negative (push away) and 3-5 positive (pull towards), so a dislike carries
information.

**How would you deploy it?** Precompute item factors and content vectors;
at request time fold in the user (one 128×128 solve), score with a matrix-vector
product, blend and re-rank. Everything runs in milliseconds. Retrain ALS
nightly.

**What would you improve?** See "Next steps" in REPORT.md: real plot
descriptions with sentence embeddings, learning the blend weights with a
small model instead of a formula, mapping content features into the CF
space so new movies get a CF vector, and an online A/B test.
