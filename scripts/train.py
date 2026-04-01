from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cclm_cira.train import train_experiment


def main() -> None:
    parser = argparse.ArgumentParser(description="Train CCLM+CIRA model")
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--dataset", default="data/cira_lab_dataset.json")
    parser.add_argument("--output", default="outputs/run_01")
    args = parser.parse_args()

    train_experiment(config_path=args.config, dataset_path=args.dataset, output_dir=args.output)


if __name__ == "__main__":
    main()
