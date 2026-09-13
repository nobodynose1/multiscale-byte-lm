"""Build an explicit-resume input config for a chained run.

A run restores only from `resume.checkpoint_file`, and the training position
lives inside that checkpoint, so a chained run cannot reuse a previous run's
output config as its input. This helper takes a plain sample config and points
its `resume` block at the checkpoint of the run it continues.
"""

from argparse import ArgumentParser
from pathlib import Path
from typing import Any

import yaml


def build_resume_config(
    config: dict[str, Any], checkpoint: Path, target_elements: int
) -> dict[str, Any]:
    """Add an explicit checkpoint and a larger cumulative training target."""
    train = config.get("train")
    if not isinstance(train, dict) or not isinstance(train.get("target_elements"), int):
        raise ValueError("base config must define train.target_elements as an integer")

    base_target = train["target_elements"]
    if target_elements <= base_target:
        raise ValueError(
            "target_elements for a resumed run must exceed the base config target "
            f"({base_target}), got {target_elements}"
        )

    train["target_elements"] = target_elements
    config["resume"] = {"checkpoint_file": str(checkpoint)}
    return config


def run() -> None:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True, help="sample config to base on")
    parser.add_argument("--checkpoint", type=Path, required=True, help="checkpoint to resume from")
    parser.add_argument(
        "--target-elements",
        type=int,
        required=True,
        help="cumulative target_elements for the resumed run (must exceed the base target)",
    )
    parser.add_argument("--out", type=Path, required=True, help="config to write")
    args = parser.parse_args()

    with args.base.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    build_resume_config(config, args.checkpoint, args.target_elements)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as file:
        yaml.safe_dump(config, file)
    print(f"Wrote {args.out} resuming from {args.checkpoint}")


if __name__ == "__main__":
    run()
