"""Downloading and loading MovieLens into three tidy tables.

Whatever the source dataset, `load_dataset` returns:

    ratings  user_id | item_id | rating | timestamp
    items    item_id | title | year | genres (list of str) | tags (str)
    users    user_id | gender | age | occupation   (ML-1M only, else None)

so the rest of the code never needs to know which MovieLens version it got.
"""

from __future__ import annotations

import io
import re
import shutil
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .config import DATA_DIR

OFFICIAL_URLS = {
    "ml-1m": "https://files.grouplens.org/datasets/movielens/ml-1m.zip",
    "ml-latest-small": "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip",
    "ml-25m": "https://files.grouplens.org/datasets/movielens/ml-25m.zip",
}

# Public GitHub copies of the same files, used only if grouplens.org is
# unreachable (some university / corporate networks block it).
MIRRORS = {
    "ml-1m": "https://raw.githubusercontent.com/khanhnamle1994/movielens/master/",
    "ml-latest-small": "https://raw.githubusercontent.com/smanihwr/ml-latest-small/master/",
}


@dataclass
class Dataset:
    name: str
    ratings: pd.DataFrame
    items: pd.DataFrame
    users: pd.DataFrame | None


# --------------------------------------------------------------------------
# Download
# --------------------------------------------------------------------------

def _fetch(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "hybridrec/1.0"})
    with urllib.request.urlopen(request, timeout=120) as response:
        return response.read()


def _expected_files(name: str) -> list[str]:
    if name == "ml-1m":
        return ["ratings.dat", "movies.dat", "users.dat"]
    return ["ratings.csv", "movies.csv", "tags.csv"]


def is_downloaded(name: str) -> bool:
    folder = DATA_DIR / name
    return all((folder / f).exists() for f in _expected_files(name))


def download(name: str) -> Path:
    """Download a dataset into data/<name>/ (skips if already there)."""
    if name not in OFFICIAL_URLS:
        raise ValueError(f"Unknown dataset {name!r}. Choose from {list(OFFICIAL_URLS)}")
    folder = DATA_DIR / name
    if is_downloaded(name):
        print(f"{name} already in {folder}")
        return folder
    folder.mkdir(parents=True, exist_ok=True)

    try:
        print(f"Downloading {OFFICIAL_URLS[name]} ...")
        archive = zipfile.ZipFile(io.BytesIO(_fetch(OFFICIAL_URLS[name])))
        for member in archive.namelist():
            filename = Path(member).name
            if filename in _expected_files(name):
                with archive.open(member) as src, open(folder / filename, "wb") as dst:
                    shutil.copyfileobj(src, dst)
    except Exception as error:  # network blocked, timeout, ...
        if name not in MIRRORS:
            raise RuntimeError(
                f"Could not download {name}: {error}. Download the zip by hand from "
                f"{OFFICIAL_URLS[name]} and unzip its files into {folder}"
            ) from error
        print(f"Official download failed ({error}); trying the GitHub mirror ...")
        _download_from_mirror(name, folder)

    print(f"Saved to {folder}")
    return folder


def _download_from_mirror(name: str, folder: Path) -> None:
    base = MIRRORS[name]
    if name == "ml-latest-small":
        for filename in _expected_files(name):
            (folder / filename).write_bytes(_fetch(base + filename))
        return

    # The ML-1M mirror stores tab-separated CSVs; convert them back to the
    # official "::"-separated .dat layout so there is one loader.
    def tsv(filename):
        return pd.read_csv(io.BytesIO(_fetch(base + filename)), sep="\t", index_col=0,
                           encoding="latin-1")

    ratings = tsv("ratings.csv")[["user_id", "movie_id", "rating", "timestamp"]]
    movies = tsv("movies.csv")[["movie_id", "title", "genres"]]
    users = tsv("users.csv")[["user_id", "gender", "age", "occupation", "zipcode"]]
    for frame, filename in [(ratings, "ratings.dat"), (movies, "movies.dat"), (users, "users.dat")]:
        lines = frame.astype(str).agg("::".join, axis=1)
        (folder / filename).write_text("\n".join(lines) + "\n", encoding="latin-1", errors="replace")


# --------------------------------------------------------------------------
# Load
# --------------------------------------------------------------------------

_ARTICLE = re.compile(r"^(.*), (The|A|An)$")
_YEAR = re.compile(r"^(.*?)\s*\((\d{4})\)\s*$")


def split_title(raw: str) -> tuple[str, float]:
    """'American President, The (1995)' -> ('The American President', 1995)."""
    raw = raw.strip()
    year = np.nan
    match = _YEAR.match(raw)
    if match:
        raw, year = match.group(1), float(match.group(2))
    # Some titles carry an alternative name in brackets; the article fix
    # applies to the main part only.
    main, bracket, rest = raw.partition(" (")
    article = _ARTICLE.match(main)
    if article:
        main = f"{article.group(2)} {article.group(1)}"
    return main + bracket + rest, year


def _read_dat(path: Path, columns: list[str]) -> pd.DataFrame:
    # "::" separated; swapping it for a tab lets pandas use its fast C parser.
    text = path.read_text(encoding="latin-1").replace("::", "\t")
    return pd.read_csv(io.StringIO(text), sep="\t", names=columns, header=None, quoting=3)


def _build_items(movies: pd.DataFrame, tags: pd.DataFrame | None) -> pd.DataFrame:
    titles = movies["title"].map(split_title)
    items = pd.DataFrame({
        "item_id": movies["item_id"].astype(int),
        "title": [t for t, _ in titles],
        "year": [y for _, y in titles],
        "genres": movies["genres"].map(
            lambda g: [] if g == "(no genres listed)" else str(g).split("|")
        ),
    })
    if tags is not None and len(tags):
        joined = (tags.assign(tag=tags["tag"].astype(str).str.lower())
                  .groupby("item_id")["tag"].apply(" ".join))
        items["tags"] = items["item_id"].map(joined).fillna("")
    else:
        items["tags"] = ""
    return items.reset_index(drop=True)


def load_dataset(name: str = "ml-1m") -> Dataset:
    folder = DATA_DIR / name
    if not is_downloaded(name):
        raise FileNotFoundError(
            f"{name} not found in {folder}. Run: python -m hybridrec download --dataset {name}"
        )

    if name == "ml-1m":
        ratings = _read_dat(folder / "ratings.dat", ["user_id", "item_id", "rating", "timestamp"])
        movies = _read_dat(folder / "movies.dat", ["item_id", "title", "genres"])
        users = _read_dat(folder / "users.dat", ["user_id", "gender", "age", "occupation", "zip"])
        users = users.drop(columns="zip")
        tags = None
    else:
        ratings = pd.read_csv(folder / "ratings.csv").rename(
            columns={"userId": "user_id", "movieId": "item_id"})
        movies = pd.read_csv(folder / "movies.csv").rename(columns={"movieId": "item_id"})
        tags = pd.read_csv(folder / "tags.csv").rename(columns={"movieId": "item_id"})
        users = None

    ratings = ratings.astype({"user_id": int, "item_id": int, "rating": float, "timestamp": int})
    items = _build_items(movies, tags)
    # Drop ratings that point at movies missing from the catalog (none in the
    # official files, but cheap insurance).
    ratings = ratings[ratings["item_id"].isin(items["item_id"])].reset_index(drop=True)
    return Dataset(name=name, ratings=ratings, items=items, users=users)


def sample_users(dataset: Dataset, n_users: int, seed: int = 0) -> Dataset:
    """Keep a random subset of users (makes ML-25M fit a laptop)."""
    all_users = dataset.ratings["user_id"].unique()
    if n_users >= len(all_users):
        return dataset
    keep = np.random.default_rng(seed).choice(all_users, size=n_users, replace=False)
    ratings = dataset.ratings[dataset.ratings["user_id"].isin(keep)].reset_index(drop=True)
    users = None if dataset.users is None else dataset.users[dataset.users["user_id"].isin(keep)]
    return Dataset(dataset.name, ratings, dataset.items, users)
