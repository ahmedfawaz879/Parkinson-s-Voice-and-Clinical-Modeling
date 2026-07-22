"""Command-line entry point for the Parkinson's voice/clinical pipeline.

Reconstructed/orchestrated from notebook cells 12-17 (processed-data
loading, train/eval, robustness sweep, artifact saving), wired to
configs/default.yaml.

Usage
-----
    parkinsons-voice train --updrs
    parkinsons-voice robustness

Both commands require processed PPMI tables to already exist locally under
``data.processed_dir`` (default ``./processed``); PPMI data is gated by a
Data Use Agreement and is not bundled with this repo (see README.md).
``--help`` works standalone without any data present.
"""

from __future__ import annotations

import argparse
import logging
import sys

from .config import ensure_dirs, load_config, set_global_seed
from .data import PPMIDataNotFoundError, load_processed_sources, require_data_dir
from .features import compute_updrs_total
from .robustness import robustness_sweep
from .train import save_artifacts, train_classifier, train_updrs_regressor

logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="parkinsons-voice",
        description="PPMI clinical-data pipeline: PD classification, UPDRS progression "
        "modeling, SHAP/LIME explainability, and feature-level noise-robustness testing.",
    )
    parser.add_argument("--config", default=None, help="Path to a YAML config overriding configs/default.yaml")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable DEBUG-level logging")

    sub = parser.add_subparsers(dest="command", required=True)

    train_p = sub.add_parser("train", help="Train the PD classifier on processed PPMI clinical data")
    train_p.add_argument("--processed-dir", default=None, help="Override data.processed_dir from config")
    train_p.add_argument(
        "--updrs", action="store_true", help="Also fit the UPDRS TOTAL-score regressor from updrs_longitudinal"
    )

    robust_p = sub.add_parser(
        "robustness", help="Train the PD classifier and run the feature-level noise-robustness SNR sweep"
    )
    robust_p.add_argument("--processed-dir", default=None, help="Override data.processed_dir from config")

    return parser


def _load_clinical_sources(processed_dir_override, cfg):
    processed_dir = processed_dir_override or cfg.data.processed_dir
    require_data_dir(processed_dir)
    sources = load_processed_sources(processed_dir)
    if sources["clinical"] is None:
        raise PPMIDataNotFoundError(
            f"No clinical_clean*.csv found under {processed_dir}. See README.md for the "
            "expected processed-data layout and the PPMI Data Use Agreement requirement."
        )
    return sources


def cmd_train(args, cfg) -> int:
    sources = _load_clinical_sources(args.processed_dir, cfg)

    clf, metrics, _split = train_classifier(sources["clinical"], cfg)
    print("Classification metrics:")
    for key, value in metrics.items():
        if key == "confusion_matrix":
            continue
        print(f"  {key}: {value}")
    print("Confusion matrix:")
    print(metrics["confusion_matrix"])

    artifacts = {"classifier": clf}

    if args.updrs:
        if sources["updrs"] is None:
            logger.warning("--updrs requested but no updrs_longitudinal*.csv found; skipping regressor")
        else:
            updrs_total = compute_updrs_total(sources["updrs"])
            reg, reg_metrics, _ = train_updrs_regressor(updrs_total, cfg)
            print("UPDRS regression metrics:")
            for key, value in reg_metrics.items():
                print(f"  {key}: {value}")
            artifacts["updrs_regressor"] = reg

    saved = save_artifacts(artifacts, cfg.output.artifacts_dir)
    print("Saved artifacts:", saved)
    return 0


def cmd_robustness(args, cfg) -> int:
    sources = _load_clinical_sources(args.processed_dir, cfg)

    clf, _metrics, split = train_classifier(sources["clinical"], cfg)
    _X_train, X_test, _y_train, y_test = split

    sweep = robustness_sweep(clf, X_test, y_test, cfg.robustness.snr_levels_db, random_state=cfg.seed)
    print("Feature-level noise-robustness sweep (Gaussian noise on extracted features;")
    print("audio-level perturbation is out of scope - see robustness.py):")
    print(sweep.to_string(index=False))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="[%(levelname)s] %(message)s")

    cfg = load_config(args.config)
    set_global_seed(cfg.seed)
    ensure_dirs(cfg)

    try:
        if args.command == "train":
            return cmd_train(args, cfg)
        if args.command == "robustness":
            return cmd_robustness(args, cfg)
    except PPMIDataNotFoundError as exc:
        logger.error(str(exc))
        return 1

    parser.error(f"Unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    sys.exit(main())
