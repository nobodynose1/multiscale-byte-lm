"""Create a deliberately unsafe checkpoint for a negative resume e2e test."""

from argparse import ArgumentParser
from pathlib import Path

import torch


def run() -> None:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    snapshot = torch.load(args.source, map_location="cpu", weights_only=True)
    snapshot["CUM_BATCH"] += 1
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(snapshot, args.out)


if __name__ == "__main__":
    run()
