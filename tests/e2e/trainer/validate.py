from argparse import ArgumentParser, BooleanOptionalAction
from datetime import datetime
from pathlib import Path

import polars as pl
import torch

from mblm.train.core.config import (
    CoreIoConfig,
    CoreModelParams,
    CoreTrainConfig,
    GenericOutputConfig,
)
from mblm.utils.io import load_yml


class TrainOutputConfig(GenericOutputConfig[CoreModelParams, CoreTrainConfig, CoreIoConfig]):
    pass


def ensure_no_error_logs(log_file: Path):
    with Path.open(log_file) as log:
        for line in log.readlines():
            _, _, level, msg = line.split(" - ")
            if level == "CRITICAL" or level == "ERROR":
                raise AssertionError(msg)


def assert_on_model_run_output(output_dir: Path) -> None:
    checkpoints = list(output_dir.rglob("*.pth"))
    latest_checkpoint = output_dir / "latest.pth"
    ranked_checkpoints = [
        checkpoint for checkpoint in checkpoints if checkpoint != latest_checkpoint
    ]
    csv_loss_file = output_dir / "loss.csv"
    yml_config_file = output_dir / "config.yaml"
    log_file = output_dir / "train.log"

    assert csv_loss_file.exists(), "Expected a CSV loss file"
    assert yml_config_file.exists(), "Expected a YAML config file"
    assert log_file.exists(), "Expected a log file"

    ensure_no_error_logs(log_file)

    run_config = load_yml(yml_config_file, parse_to=TrainOutputConfig)

    assert latest_checkpoint.exists(), "Expected a latest checkpoint for resuming"
    assert (
        len(ranked_checkpoints) == run_config.io.num_models_to_save
    ), f"Expected {run_config.io.num_models_to_save} ranked model checkpoints"

    # assert that we get the specified number of training loss logs
    csv_log = pl.read_csv(csv_loss_file)
    is_resumed = run_config.resume is not None and run_config.resume.parent_checkpoint is not None

    num_train_loss_entries = csv_log.filter(pl.col("kind") == "train").select(pl.len()).item()
    if is_resumed:
        assert 0 < num_train_loss_entries <= run_config.io.log_train_loss_amount, (
            "Expected resumed training to record a non-empty partial train segment, "
            f"received {num_train_loss_entries} entries"
        )
    else:
        assert num_train_loss_entries == run_config.io.log_train_loss_amount, (
            f"Expected {run_config.io.log_train_loss_amount} train loss entries, "
            f"received {num_train_loss_entries}"
        )

    # assert that we get the specified number of validation loss logs
    num_valid_loss_entries = csv_log.filter(pl.col("kind") == "valid").select(pl.len()).item()
    if is_resumed:
        assert 0 < num_valid_loss_entries <= run_config.io.validate_amount, (
            "Expected resumed training to record a non-empty partial validation segment, "
            f"received {num_valid_loss_entries} entries"
        )
    else:
        assert num_valid_loss_entries == run_config.io.validate_amount, (
            f"Expected {run_config.io.validate_amount} validation loss entries, "
            f"received {num_valid_loss_entries}"
        )

    if is_resumed and run_config.io.num_models_to_save > 0:
        resume_evaluations = csv_log.filter(
            (pl.col("kind") == "valid") & (pl.col("source") == "resume_evaluation")
        )
        assert resume_evaluations.height == 1, "Expected one resume validation evaluation"
        assert resume_evaluations.select(
            pl.col("complete").all()
        ).item(), "Resume evaluation must be complete before it can contribute to TopN"
        assert resume_evaluations.select(pl.col("skipped_batches").max()).item() == 0

    # assert that we the test loss is logged
    num_test_loss_entries = csv_log.filter(pl.col("kind") == "test").select(pl.len()).item()
    assert num_test_loss_entries == 1

    # assert we log valid dates in the summary
    try:
        datetime.fromisoformat(run_config.summary.training_start)
        datetime.fromisoformat(run_config.summary.training_end)
    except ValueError:
        raise AssertionError("Failed to parse training start/end dates")


def assert_on_chained_logs(csv_loss_files: list[Path], assert_equal_epochs: bool):
    for file_idx in range(0, len(csv_loss_files) - 1):
        file_1 = csv_loss_files[file_idx]
        file_2 = csv_loss_files[file_idx + 1]

        prev = pl.read_csv(file_1).filter(pl.col("kind") == "train")
        curr = pl.read_csv(file_2).filter(pl.col("kind") == "train")

        # every single run was exactly one epoch
        if assert_equal_epochs:
            # The resumed segment starts at the cursor where the preceding
            # segment ended. Train rows are sampled at different positions,
            # so their full epoch series are intentionally not row-aligned.
            assert (
                prev.select(pl.col("epoch").max()).item()
                == curr.select(pl.col("epoch").min()).item()
            ), "Training log epoch cursor did not resume continuously"

        # Train rows are intentionally sampled, not emitted per micro-batch. The
        # exact checkpoint cursor continuity is checked separately below; these
        # CSV observations must only be strictly ordered.
        assert (
            prev.select(pl.col("cum_batch").max()).item()
            < curr.select(pl.col("cum_batch").min()).item()
        ), "Training log cumulative batches did not advance"
        assert (
            prev.select(pl.col("elements_seen").max()).item()
            < curr.select(pl.col("elements_seen").min()).item()
        ), "Training log elements_seen did not advance"

        for df in [prev, curr]:
            # the chained test is designed to run for around 3 epochs, adjust if necessary
            assert df.select(pl.col("epoch").max()).item() < 4, "Expected less than 3 epochs"


def assert_on_grad_acc(csv_loss_files: list[Path]):
    for file_idx in range(0, len(csv_loss_files) - 1):
        file_1 = csv_loss_files[file_idx]
        file_2 = csv_loss_files[file_idx + 1]

        prev = pl.read_csv(file_1).sort("timestamp")
        curr = pl.read_csv(file_2).sort("timestamp")

        # make sure same loss on test
        assert (
            curr.select(pl.col("loss").last().round(3)).item()
            == prev.select(pl.col("loss").last().round(3)).item()
        )


def _state_step(snapshot: dict) -> int:
    steps = {
        int(value["step"].item() if isinstance(value["step"], torch.Tensor) else value["step"])
        for value in snapshot["OPTIMIZER"]["state"].values()
        if "step" in value
    }
    assert len(steps) == 1, "Expected every optimizer parameter state to share one step"
    return steps.pop()


def _expected_cum_batch(config: TrainOutputConfig) -> int:
    per_rank_micro_batch_elements = config.train.batch_size * config.params.input_seq_len
    global_micro_batch_elements = per_rank_micro_batch_elements * config.summary.num_workers
    assert config.train.target_elements % global_micro_batch_elements == 0
    return config.train.target_elements // global_micro_batch_elements


def assert_on_resume_checkpoint_chain(output_dirs: list[Path]) -> None:
    """Check complete checkpoint-state continuity from real e2e artifacts."""
    assert len(output_dirs) >= 2, "Expected at least two chained outputs"
    snapshots: list[dict] = []
    configs: list[TrainOutputConfig] = []

    for output_dir in output_dirs:
        checkpoint = output_dir / "latest.pth"
        assert checkpoint.exists(), f"Expected resumable checkpoint {checkpoint}"
        snapshot = torch.load(checkpoint, map_location="cpu", weights_only=True)
        required_keys = {
            "MODEL",
            "OPTIMIZER",
            "SCHEDULER",
            "GRAD_SCALER",
            "EPOCH",
            "BATCH",
            "CUM_BATCH",
        }
        assert required_keys.issubset(snapshot), f"Incomplete checkpoint {checkpoint}"
        config = load_yml(output_dir / "config.yaml", parse_to=TrainOutputConfig)
        assert snapshot["CUM_BATCH"] == _expected_cum_batch(config)
        assert snapshot["CUM_BATCH"] % config.train.gradient_accumulate_every == 0
        snapshots.append(snapshot)
        configs.append(config)

    for previous_dir, current_dir, previous, current, previous_config, current_config in zip(
        output_dirs[:-1],
        output_dirs[1:],
        snapshots[:-1],
        snapshots[1:],
        configs[:-1],
        configs[1:],
        strict=True,
    ):
        assert current_config.resume is not None
        assert (
            Path(current_config.resume.parent_checkpoint).resolve()
            == (previous_dir / "latest.pth").resolve()
        ), f"{current_dir} did not record {previous_dir} as its explicit parent"

        expected_new_micro_batches = current["CUM_BATCH"] - previous["CUM_BATCH"]
        assert expected_new_micro_batches > 0
        expected_new_optimizer_steps = (
            expected_new_micro_batches // current_config.train.gradient_accumulate_every
        )
        assert _state_step(current) - _state_step(previous) == expected_new_optimizer_steps
        assert (
            current["SCHEDULER"]["last_epoch"] - previous["SCHEDULER"]["last_epoch"]
            == expected_new_optimizer_steps
        )
        assert (
            current["GRAD_SCALER"]["_growth_tracker"] - previous["GRAD_SCALER"]["_growth_tracker"]
            == expected_new_optimizer_steps
        )
        assert any(
            not torch.equal(previous["MODEL"][name], current["MODEL"][name])
            for name in previous["MODEL"]
        ), "Model weights did not change during resumed training"


def run_test():
    parser = ArgumentParser()
    parser.add_argument("--check-output", type=Path, dest="check_output")
    parser.add_argument("--check-chained-csv", action="append", type=Path, dest="check_chained_csv")
    parser.add_argument(
        "--check-resume-chain", action="append", type=Path, dest="check_resume_chain"
    )
    parser.add_argument(
        "--check-grad-acc-csv", action="append", type=Path, dest="check_grad_acc_csv"
    )
    parser.add_argument("--assert-equal-epochs", action=BooleanOptionalAction, default=False)
    args = parser.parse_args()

    check_output: Path | None = args.check_output
    check_chained_csv: list[Path] | None = args.check_chained_csv
    check_resume_chain: list[Path] | None = args.check_resume_chain
    check_grad_acc_csv: list[Path] | None = args.check_grad_acc_csv
    assert_equal_epochs: bool = args.assert_equal_epochs

    checks = [check_output, check_chained_csv, check_resume_chain, check_grad_acc_csv]
    if sum(check is not None for check in checks) != 1:
        raise AssertionError("Specify exactly one validation mode")
    if check_output:
        print(f"Validating {check_output}")
        return assert_on_model_run_output(check_output)
    if check_chained_csv:
        print(f"Validating {len(check_chained_csv)} chained runs")
        return assert_on_chained_logs(check_chained_csv, assert_equal_epochs)
    if check_resume_chain:
        print(f"Validating {len(check_resume_chain)} resume checkpoints")
        return assert_on_resume_checkpoint_chain(check_resume_chain)
    if check_grad_acc_csv:
        print("Asserting on gradient accumulation")
        return assert_on_grad_acc(check_grad_acc_csv)
    raise AssertionError("No tests ran")


if __name__ == "__main__":
    run_test()
