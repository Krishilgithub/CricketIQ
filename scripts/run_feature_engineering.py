from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cricketiq.feature_engineering.full_pipeline import load_config, run_feature_engineering


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run CricketIQ feature engineering pipeline")
    parser.add_argument(
        "--config",
        default="configs/feature_engineering.json",
        help="Path to feature engineering config file",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = (ROOT / args.config).resolve() if not Path(args.config).is_absolute() else Path(args.config)
    cfg = load_config(config_path)

    if not cfg.input_dir.is_absolute():
        cfg.input_dir = (ROOT / cfg.input_dir).resolve()
    if not cfg.output_dir.is_absolute():
        cfg.output_dir = (ROOT / cfg.output_dir).resolve()

    summary = run_feature_engineering(cfg)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
