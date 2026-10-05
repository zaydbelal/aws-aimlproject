"""All the knobs of the project in one place.

Every number that changes the results lives here, so you can answer the
question "why is it set to that?" by looking at one file. `python -m hybridrec
tune` searches over the blending knobs and saves the winners to
results/best_params.json; `evaluate` uses that file when it exists.
"""

from dataclasses import asdict, dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"


@dataclass
class SplitConfig:
    seed: int = 42
    # Share of users we turn into cold-start users: we keep only their first
    # 0-4 ratings for training and hide the rest.
    cold_user_frac: float = 0.10
    # Share of items we turn into cold-start items the same way.
    cold_item_frac: float = 0.05
    # "Cold" means fewer than this many training interactions (the brief says <5).
    cold_threshold: int = 5
    # For warm users, the most recent 20% of their ratings become the test set.
    test_frac: float = 0.20
    # A test rating counts as "relevant" (a movie the user liked) at >= 4 stars.
    relevance_threshold: float = 4.0
    # Only users/items with at least this many ratings can be picked to be
    # turned cold, so there is something left over to test on.
    min_ratings_to_coldify_user: int = 10
    min_ratings_to_coldify_item: int = 20


@dataclass
class ALSConfig:
    # Defaults = the winners of `python -m hybridrec tune --als` on ML-1M.
    factors: int = 128         # length of each user/item vector
    regularization: float = 100.0  # L2 penalty; stops vectors from overfitting
    alpha: float = 1.0         # how much a higher rating boosts confidence
    iterations: int = 12
    seed: int = 7


@dataclass
class ContentConfig:
    # Relative importance of each metadata block in the item vector.
    genre_weight: float = 1.0
    year_weight: float = 0.5
    text_weight: float = 0.7
    # Size of the text embedding (TF-IDF squeezed with truncated SVD).
    text_dims: int = 64


@dataclass
class HybridConfig:
    # Each k reads "how many ratings until we trust this 50%".
    # Defaults = the winners of `python -m hybridrec tune` on ML-1M.
    # CF trust in the user: c_user = n_user / (n_user + k_user).
    k_user: float = 30.0
    # CF trust in the movie: c_item = n_item / (n_item + k_item).
    k_item: float = 300.0
    # Trust in the user's content profile: t_user = n_user / (n_user + k_content).
    # Whatever CF and content can't cover goes to popularity.
    k_content: float = 30.0


@dataclass
class RerankConfig:
    # MMR: 1.0 = pure relevance, lower = more diversity.
    mmr_lambda: float = 0.7
    # Extra penalty for very popular items (pushes towards novelty).
    popularity_penalty: float = 0.1
    candidates: int = 100


@dataclass
class Config:
    dataset: str = "ml-1m"
    k: int = 10                      # length of the recommendation list
    extra_ks: tuple = (5, 20)        # also reported in the JSON results
    split: SplitConfig = field(default_factory=SplitConfig)
    als: ALSConfig = field(default_factory=ALSConfig)
    content: ContentConfig = field(default_factory=ContentConfig)
    hybrid: HybridConfig = field(default_factory=HybridConfig)
    rerank: RerankConfig = field(default_factory=RerankConfig)
    # Evaluate on at most this many users (None = all). Handy for ML-25M.
    max_eval_users: int | None = None

    def to_dict(self) -> dict:
        return asdict(self)
