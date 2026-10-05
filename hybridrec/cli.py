"""Command line entry point: python -m hybridrec <command>.

    download   fetch a MovieLens dataset into data/
    tune       grid-search blending (and optionally ALS) knobs on a validation split
    evaluate   train on the train split, evaluate every model, run the failure
               analysis, write results/ and save the model for the web app
    train      only train and save the model for the web app (no evaluation)
    serve      start the web app
"""

from __future__ import annotations

import argparse
import time
from dataclasses import replace

from .config import Config


def _config(args) -> Config:
    from .tune import apply_best_params

    cfg = Config(dataset=args.dataset)
    if getattr(args, "max_eval_users", None):
        cfg = replace(cfg, max_eval_users=args.max_eval_users)
    if getattr(args, "defaults", False):
        print("Ignoring results/best_params.json (--defaults)")
        return cfg
    return apply_best_params(cfg)


def _dataset(args):
    from .data import load_dataset, sample_users

    dataset = load_dataset(args.dataset)
    if getattr(args, "sample_users", None):
        dataset = sample_users(dataset, args.sample_users)
        print(f"Sampled {args.sample_users:,} users")
    return dataset


def cmd_download(args):
    from .data import download

    download(args.dataset)


def cmd_tune(args):
    from .tune import run_tuning

    run_tuning(Config(dataset=args.dataset), include_als=args.als)


def cmd_evaluate(args):
    from .analysis import run_analysis
    from .evaluate import evaluate_all
    from .pipeline import build_experiment, save_experiment
    from .report import write_reports

    start = time.time()
    cfg = _config(args)
    experiment = build_experiment(cfg, _dataset(args))
    results = evaluate_all(experiment)
    analysis = run_analysis(experiment, results)
    write_reports(experiment, results, analysis)
    print(f"Model saved to {save_experiment(experiment)}")
    print(f"Done in {time.time() - start:.0f}s")


def cmd_train(args):
    from .pipeline import build_experiment, save_experiment

    experiment = build_experiment(_config(args), _dataset(args))
    print(f"Model saved to {save_experiment(experiment)}")


def cmd_serve(args):
    from app.server import create_app

    create_app(args.dataset).run(host=args.host, port=args.port, debug=False)


def main(argv=None):
    parser = argparse.ArgumentParser(prog="python -m hybridrec", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    def add(name, func, help_text):
        p = sub.add_parser(name, help=help_text)
        p.add_argument("--dataset", default="ml-1m",
                       choices=["ml-1m", "ml-latest-small", "ml-25m"])
        p.set_defaults(func=func)
        return p

    add("download", cmd_download, "download a MovieLens dataset")
    p = add("tune", cmd_tune, "tune hyperparameters on a validation split")
    p.add_argument("--als", action="store_true", help="also grid-search ALS (slower)")
    for name, func, text in [("evaluate", cmd_evaluate, "train, evaluate, analyse"),
                             ("train", cmd_train, "train and save the model only")]:
        p = add(name, func, text)
        p.add_argument("--defaults", action="store_true",
                       help="ignore tuned parameters, use config.py defaults")
        p.add_argument("--sample-users", type=int, help="random subset of users (ML-25M)")
        p.add_argument("--max-eval-users", type=int, help="evaluate on at most N users")
    p = add("serve", cmd_serve, "start the web app")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
