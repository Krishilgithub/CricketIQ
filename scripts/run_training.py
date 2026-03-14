from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cricketiq.training.train_baseline import train_baseline_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run CricketIQ baseline model training")
    parser.add_argument("--config", default="configs/training.json", help="Training config path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = Path(args.config)
    cfg = (ROOT / cfg).resolve() if not cfg.is_absolute() else cfg
    result = train_baseline_model(cfg)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
