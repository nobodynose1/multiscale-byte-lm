"""The training cursor counts the micro-batches this rank has completed: it
moves once per micro-batch, it is restored from the checkpoint instead of being
recomputed from a data loader, and the points the run logs, validates and saves
at are decided on that count."""

import csv
from collections.abc import Iterator
from pathlib import Path
from typing import Any, cast

import torch
from pytest_mock import MockerFixture

import mblm.train.core.trainer as trainer_module
from mblm.data.datasets import DistributedDataset
from mblm.train.core.config import (
    CoreIoConfig,
    CoreModelParams,
    CoreTrainConfig,
    GenericEntryConfig,
    ResumeConfig,
)
from mblm.train.core.startup import start_trainer
from mblm.train.core.trainer import CoreTrainer, CoreTrainerOptions
from mblm.utils.distributed import ElasticRunVars
from mblm.utils.io import (
    CheckpointCursor,
    load_checkpoint_snapshot,
    load_training_checkpoint_state,
)


class RecordingLogger:
    def __init__(self) -> None:
        self.infos: list[str] = []
        self.warnings: list[str] = []
        self.fatals: list[str] = []

    def debug(self, *_args: Any, **_kwargs: Any) -> None:
        pass

    def info(self, message: str, *_args: Any, **_kwargs: Any) -> None:
        self.infos.append(message)

    def warning(self, message: str, *_args: Any, **_kwargs: Any) -> None:
        self.warnings.append(message)

    def error(self, *_args: Any, **_kwargs: Any) -> None:
        pass

    def fatal(self, message: Any, *_args: Any, **_kwargs: Any) -> None:
        self.fatals.append(str(message))


class TinyParams(CoreModelParams):
    pass


class TinyEntryConfig(GenericEntryConfig[TinyParams, CoreTrainConfig, CoreIoConfig]):
    pass


class TinyModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor([1.0]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.weight


class TinyTrainer(CoreTrainer[TinyModel, torch.Tensor, TinyParams, CoreTrainConfig, CoreIoConfig]):
    def init_model(self) -> TinyModel:
        return TinyModel()

    def model_forward(self, model: TinyModel, batch: torch.Tensor, device: str) -> torch.Tensor:
        return model(batch.to(device)).sum()

    def configure_optimizer(
        self, parameters: Iterator[torch.nn.Parameter]
    ) -> torch.optim.Optimizer:
        return torch.optim.SGD(parameters, lr=0.1)


class FakeDataset:
    def __init__(self) -> None:
        self.offsets: list[int] = []

    def offset_to(self, epoch: int) -> None:
        self.offsets.append(epoch)


class FakeLoader:
    """A loader that is re-iterated for every epoch of the cycler."""

    def __init__(self, batches: int) -> None:
        self._batches = batches

    def __len__(self) -> int:
        return self._batches

    def __iter__(self) -> Iterator[torch.Tensor]:
        return iter([torch.full((1,), 0.0) for _ in range(self._batches)])


class CountingSGD(torch.optim.SGD):
    """SGD that appends to `steps` every time the run takes an optimizer step."""

    def __init__(self, parameters: Iterator[torch.nn.Parameter], steps: list[int]) -> None:
        super().__init__(parameters, lr=0.1)
        self._steps = steps

    def step(self, closure: Any = None) -> Any:
        self._steps.append(1)
        return super().step(closure)


def entry_config(
    tmp_path: Path,
    *,
    target_elements: int,
    log_train_loss_amount: int,
    resume: ResumeConfig | None = None,
    gradient_accumulate_every: int = 1,
) -> TinyEntryConfig:
    return TinyEntryConfig(
        params=TinyParams(input_seq_len=1),
        train=CoreTrainConfig(
            target_elements=target_elements,
            target_elements_strategy="batch",
            batch_size=1,
            learning_rate=0.1,
            gradient_accumulate_every=gradient_accumulate_every,
        ),
        io=CoreIoConfig(
            name_model="cursor",
            output_dir=str(tmp_path / "runs"),
            num_models_to_save=0,
            validate_amount=1,
            log_train_loss_amount=log_train_loss_amount,
        ),
        resume=resume,
    )


class CursorHarness:
    """Run the real `_train` loop and read back what it recorded."""

    def __init__(
        self,
        mocker: MockerFixture,
        config: TinyEntryConfig,
        *,
        loader_batches: int,
    ) -> None:
        self.log = RecordingLogger()
        self.forward_batches = 0
        self.optimizer_steps: list[int] = []
        self.dataset = FakeDataset()
        loader = FakeLoader(batches=loader_batches)

        def count_forward(
            _trainer: Any, model: TinyModel, batch: torch.Tensor, device: str
        ) -> torch.Tensor:
            self.forward_batches += 1
            return model(batch.to(device)).sum()

        mocker.patch.object(TinyTrainer, "configure_logger", lambda _self, *_a, **_kw: self.log)
        mocker.patch.object(
            TinyTrainer,
            "configure_optimizer",
            lambda _self, parameters: CountingSGD(parameters, self.optimizer_steps),
        )
        mocker.patch.object(TinyTrainer, "_init_distributed_model", lambda _self, model: model)
        mocker.patch.object(
            TinyTrainer, "get_train_dataloader", lambda _self, _dataset, **_kwargs: loader
        )
        mocker.patch.object(
            TinyTrainer, "get_valid_dataloader", lambda _self, _dataset, **_kwargs: loader
        )
        mocker.patch.object(TinyTrainer, "model_forward", count_forward)

        self.trainer = TinyTrainer(
            config,
            run_vars=ElasticRunVars(local_rank=0, global_rank=0, world_size=1, is_cuda=False),
            options=CoreTrainerOptions(
                display_progress=False,
                track_first_fw_bw_exec_times=None,
                skip_validation=True,
                amp_dtype=torch.bfloat16,
            ),
        )
        start_trainer(self.trainer, world_size=1)

    def train(self) -> TinyModel:
        dataset = cast(DistributedDataset[torch.Tensor], self.dataset)
        return self.trainer._train(dataset, dataset)

    def loss_rows(self, kind: str = "train") -> list[dict[str, str]]:
        with (self.trainer.output_dir / "loss.csv").open(encoding="utf-8", newline="") as file:
            return [row for row in csv.DictReader(file) if row["kind"] == kind]

    def latest_snapshot(self) -> dict[str, Any]:
        return load_checkpoint_snapshot(self.trainer.output_dir / "latest.pth")


class TestTheCursorCountsCompletedMicroBatches:
    def test_a_fresh_run_counts_its_micro_batches_from_one(
        self, mocker: MockerFixture, tmp_path: Path
    ):
        harness = CursorHarness(
            mocker,
            entry_config(tmp_path, target_elements=3, log_train_loss_amount=3),
            loader_batches=3,
        )

        harness.train()

        assert [row["cum_batch"] for row in harness.loss_rows()] == ["1", "2", "3"]
        assert harness.forward_batches == 3

    def test_the_epoch_and_batch_columns_follow_the_cycler(
        self, mocker: MockerFixture, tmp_path: Path
    ):
        harness = CursorHarness(
            mocker,
            entry_config(tmp_path, target_elements=4, log_train_loss_amount=4),
            loader_batches=2,
        )

        harness.train()

        rows = harness.loss_rows()
        assert [(row["epoch"], row["batch"]) for row in rows] == [
            ("0", "0"),
            ("0", "1"),
            ("1", "0"),
            ("1", "1"),
        ]

    def test_the_saved_checkpoint_carries_the_completed_count(
        self, mocker: MockerFixture, tmp_path: Path
    ):
        harness = CursorHarness(
            mocker,
            entry_config(tmp_path, target_elements=3, log_train_loss_amount=1),
            loader_batches=3,
        )

        harness.train()

        snapshot = harness.latest_snapshot()
        # the cursor is the completed count, the epoch and batch point at the
        # next micro-batch to run
        assert (snapshot["EPOCH"], snapshot["BATCH"], snapshot["CUM_BATCH"]) == (1, 0, 3)


class TestAResumedRunContinuesTheCursor:
    @staticmethod
    def harness(mocker: MockerFixture, tmp_path: Path, cursor: CheckpointCursor) -> CursorHarness:
        mocker.patch.object(
            trainer_module,
            "load_training_checkpoint_state",
            lambda _path, _model, **_kwargs: (cursor, 2.5),
        )
        return CursorHarness(
            mocker,
            entry_config(
                tmp_path,
                target_elements=6,
                log_train_loss_amount=6,
                resume=ResumeConfig(checkpoint_file="unused.pth"),
            ),
            loader_batches=2,
        )

    def test_a_resumed_run_only_walks_the_micro_batches_left_to_it(
        self, mocker: MockerFixture, tmp_path: Path
    ):
        harness = self.harness(mocker, tmp_path, CheckpointCursor(epoch=1, batch=0, cum_batch=4))

        harness.train()

        # four of the six micro-batches were already completed before the run
        assert harness.forward_batches == 2

    def test_a_resumed_run_logs_on_the_count_it_restored(
        self, mocker: MockerFixture, tmp_path: Path
    ):
        harness = self.harness(mocker, tmp_path, CheckpointCursor(epoch=1, batch=0, cum_batch=4))

        harness.train()

        rows = harness.loss_rows()
        # the points below the resume position are never revisited
        assert [row["cum_batch"] for row in rows] == ["5", "6"]
        assert [(row["epoch"], row["batch"]) for row in rows] == [("1", "0"), ("1", "1")]

    def test_a_resumed_run_saves_the_next_cursor(self, mocker: MockerFixture, tmp_path: Path):
        harness = self.harness(mocker, tmp_path, CheckpointCursor(epoch=1, batch=0, cum_batch=4))

        harness.train()

        snapshot = harness.latest_snapshot()
        assert (snapshot["EPOCH"], snapshot["BATCH"], snapshot["CUM_BATCH"]) == (2, 0, 6)

    def test_a_cursor_past_the_target_ends_the_run_without_a_single_batch(
        self, mocker: MockerFixture, tmp_path: Path
    ):
        harness = self.harness(mocker, tmp_path, CheckpointCursor(epoch=0, batch=0, cum_batch=7))

        harness.train()

        assert harness.forward_batches == 0
        assert harness.log.warnings[-1] == (
            "Resume position 7 exceeds target iterations 6; no further training will run"
        )
        assert "Remaining batch iterations: 0" in harness.log.infos
        assert "Finished training" in harness.log.infos
        assert harness.log.fatals == []


class TestAnAccumulationWindowSpansEpochBoundaries:
    def test_a_window_that_crosses_an_epoch_still_takes_one_optimizer_step(
        self, mocker: MockerFixture, tmp_path: Path
    ):
        # four micro-batches in epochs of three: the second window opens on the
        # last micro-batch of epoch 0 and closes on the first micro-batch of
        # epoch 1, and the epoch boundary itself takes no step
        harness = CursorHarness(
            mocker,
            entry_config(
                tmp_path,
                target_elements=4,
                log_train_loss_amount=4,
                gradient_accumulate_every=2,
            ),
            loader_batches=3,
        )

        harness.train()

        assert harness.forward_batches == 4
        assert harness.optimizer_steps == [1, 1]
        assert [(row["epoch"], row["batch"]) for row in harness.loss_rows()] == [
            ("0", "0"),
            ("0", "1"),
            ("0", "2"),
            ("1", "0"),
        ]


class TestAWholeWindowRunSavesAResumePoint:
    def test_the_checkpoint_of_a_divisible_run_loads_back_as_a_resume_point(
        self, mocker: MockerFixture, tmp_path: Path
    ):
        harness = CursorHarness(
            mocker,
            entry_config(
                tmp_path,
                target_elements=6,
                log_train_loss_amount=6,
                gradient_accumulate_every=2,
            ),
            loader_batches=2,
        )

        harness.train()

        # six micro-batches fill three whole windows, so the run ends on an
        # accumulation boundary with no window left pending
        assert harness.optimizer_steps == [1, 1, 1]
        snapshot = harness.latest_snapshot()
        assert snapshot["CUM_BATCH"] == 6

        cursor, _ = load_training_checkpoint_state(
            harness.trainer.output_dir / "latest.pth",
            harness.trainer._model,
            optimizer=harness.trainer._optimizer,
            scheduler=harness.trainer._scheduler,
            grad_scaler=harness.trainer._grad_scaler,
            gradient_accumulate_every=2,
        )
        assert cursor == CheckpointCursor(epoch=3, batch=0, cum_batch=6)
