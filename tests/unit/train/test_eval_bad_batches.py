"""A validation pass that meets a batch the model cannot score skips that batch
instead of killing the run, and records the pass as incomplete: the loss stays
the mean over the batches that succeeded, and an incomplete pass never enters
the model candidate board. Only the forward pass of one batch may be skipped -
every other failure, a pass that skipped too much of its data, and the
training-side forward pass itself stay fatal."""

import csv
from collections.abc import Iterator
from pathlib import Path
from typing import Any, cast

import pytest
import torch
from pytest_mock import MockerFixture

import mblm.train.core.startup as startup_module
import mblm.train.core.trainer as trainer_module
from mblm.data.datasets import DistributedDataset
from mblm.train.core.config import (
    CoreIoConfig,
    CoreModelParams,
    CoreTrainConfig,
    GenericEntryConfig,
)
from mblm.train.core.startup import (
    EVAL_COMPLETE,
    EVAL_INCOMPLETE,
    EVAL_SCHEDULED,
    EvalOutcome,
    start_trainer,
)
from mblm.train.core.trainer import (
    DELIVERY_TOPN_BEST,
    CoreTrainer,
    CoreTrainerOptions,
)
from mblm.utils.distributed import ElasticRunVars


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


def noop(*_args: Any, **_kwargs: Any) -> None:
    """Stand-in for a collective call in the single-rank tests."""


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


class LossLoader:
    """A loader whose batches carry the value the model scores them with."""

    def __init__(self, batches: int) -> None:
        self._batches = batches

    def __len__(self) -> int:
        return self._batches

    def __iter__(self) -> Iterator[torch.Tensor]:
        return iter([torch.full((1,), float(index)) for index in range(self._batches)])


class LoaderThatFailsMidIteration(LossLoader):
    """A loader whose own iteration raises, after one batch."""

    def __iter__(self) -> Iterator[torch.Tensor]:
        def batches() -> Iterator[torch.Tensor]:
            yield torch.zeros(1)
            raise FloatingPointError("the dataset cannot be read")

        return batches()


class FakeDataset:
    def __init__(self) -> None:
        self.offsets: list[int] = []

    def offset_to(self, epoch: int) -> None:
        self.offsets.append(epoch)


def entry_config(
    tmp_path: Path,
    *,
    num_models_to_save: int,
    validate_amount: int,
    target_elements: int = 2,
) -> TinyEntryConfig:
    return TinyEntryConfig(
        params=TinyParams(input_seq_len=1),
        train=CoreTrainConfig(
            target_elements=target_elements,
            target_elements_strategy="batch",
            batch_size=1,
            learning_rate=0.1,
            gradient_accumulate_every=1,
        ),
        io=CoreIoConfig(
            name_model="bad-batches",
            output_dir=str(tmp_path / "runs"),
            num_models_to_save=num_models_to_save,
            validate_amount=validate_amount,
            log_train_loss_amount=1,
        ),
    )


class BadBatchHarness:
    """Run the real `_train`, or a single real `_evaluate` pass, against a model
    whose forward pass fails for the batches a test configures."""

    def __init__(
        self,
        mocker: MockerFixture,
        config: TinyEntryConfig,
        *,
        loader_batches: int,
        valid_loader_batches: int | None = None,
        bad_valid_batches: set[int] | None = None,
        bad_valid_once: bool = False,
        valid_failure: type[Exception] = FloatingPointError,
        valid_loader_fails: bool = False,
        bad_train_value: int | None = None,
    ) -> None:
        self.log = RecordingLogger()
        self.forward_batches = 0
        self.bad_valid_batches = set(bad_valid_batches or ())
        self.bad_valid_once = bad_valid_once
        self.valid_failure = valid_failure
        self.bad_train_value = bad_train_value
        self.dataset = FakeDataset()
        scored_batches = loader_batches if valid_loader_batches is None else valid_loader_batches
        train_loader: Any = LossLoader(batches=loader_batches)
        self.valid_loader: Any = (
            LoaderThatFailsMidIteration(batches=scored_batches)
            if valid_loader_fails
            else LossLoader(batches=scored_batches)
        )

        def forward(
            _trainer: Any, model: TinyModel, batch: torch.Tensor, device: str
        ) -> torch.Tensor:
            value = int(batch.item())
            self.forward_batches += 1
            if model.training:
                if value == self.bad_train_value:
                    raise FloatingPointError("non-finite loss elements detected")
                return model(batch.to(device)).sum()
            if value in self.bad_valid_batches:
                if self.bad_valid_once:
                    # the batch is bad once: the pass after it scores normally
                    self.bad_valid_batches.discard(value)
                raise self.valid_failure(f"non-finite loss elements detected: {value}")
            return model(batch.to(device)).sum()

        mocker.patch.object(TinyTrainer, "configure_logger", lambda _self, *_a, **_kw: self.log)
        mocker.patch.object(TinyTrainer, "_init_distributed_model", lambda _self, model: model)
        mocker.patch.object(
            TinyTrainer, "get_train_dataloader", lambda _self, _dataset, **_kwargs: train_loader
        )
        mocker.patch.object(
            TinyTrainer,
            "get_valid_dataloader",
            lambda _self, _dataset, **_kwargs: self.valid_loader,
        )
        mocker.patch.object(TinyTrainer, "model_forward", forward)
        # the candidate board is entered under a barrier; a single rank has no
        # process group to enter it with
        mocker.patch.object(trainer_module.dist, "barrier", noop)

        self.trainer = TinyTrainer(
            config,
            run_vars=ElasticRunVars(local_rank=0, global_rank=0, world_size=1, is_cuda=False),
            options=CoreTrainerOptions(
                display_progress=False,
                track_first_fw_bw_exec_times=None,
                amp_dtype=torch.bfloat16,
            ),
        )
        start_trainer(self.trainer, world_size=1)

    def train(self) -> TinyModel:
        dataset = cast(DistributedDataset[torch.Tensor], self.dataset)
        return self.trainer._train(dataset, dataset)

    def evaluate(self) -> EvalOutcome:
        return self.trainer._evaluate(
            self.trainer._model_dist,
            self.valid_loader,
            items_seen_so_far=0,
            cumulative_batch_idx=1,
            stage=EVAL_SCHEDULED,
        )

    def loss_rows(self, kind: str = "train") -> list[dict[str, str]]:
        with (self.trainer.output_dir / "loss.csv").open(encoding="utf-8", newline="") as file:
            return [row for row in csv.DictReader(file) if row["kind"] == kind]

    def board(self) -> list[tuple[float, Any]]:
        return self.trainer._top_n_models.get_top()


class TestAPassThatSkipsABadBatch:
    def test_the_loss_stays_the_mean_over_the_batches_that_succeeded(
        self, mocker: MockerFixture, tmp_path: Path
    ):
        harness = BadBatchHarness(
            mocker,
            entry_config(tmp_path, num_models_to_save=1, validate_amount=1),
            loader_batches=10,
            bad_valid_batches={9},
        )

        outcome = harness.evaluate()

        assert outcome.state == EVAL_INCOMPLETE
        assert (outcome.ok_batches, outcome.skipped_batches) == (9, 1)
        # batches 0 to 8 scored their own value; batch 9 was left out
        assert outcome.loss == 4.0
        assert any("Skipping batch 9" in warning for warning in harness.log.warnings)

    def test_a_pass_that_skips_nothing_is_still_complete(
        self, mocker: MockerFixture, tmp_path: Path
    ):
        harness = BadBatchHarness(
            mocker,
            entry_config(tmp_path, num_models_to_save=1, validate_amount=1),
            loader_batches=3,
        )

        outcome = harness.evaluate()

        assert outcome.state == EVAL_COMPLETE
        assert (outcome.ok_batches, outcome.skipped_batches) == (3, 0)
        assert outcome.loss == 1.0


class TestTheThresholdsThatFailAPass:
    def test_a_pass_whose_every_batch_fails_reports_no_loss(
        self, mocker: MockerFixture, tmp_path: Path
    ):
        log = RecordingLogger()
        mocker.patch.object(startup_module, "_log", log)
        harness = BadBatchHarness(
            mocker,
            entry_config(tmp_path, num_models_to_save=1, validate_amount=1),
            loader_batches=3,
            bad_valid_batches={0, 1, 2},
        )

        with pytest.raises(SystemExit) as exit_info:
            harness.evaluate()

        assert exit_info.value.code == 1
        assert log.fatals == [
            "evaluation 'scheduled_validation' failed: rank 0: no batch of the evaluation "
            "succeeded"
        ]

    def test_a_pass_that_skips_more_than_a_tenth_of_its_batches_fails(
        self, mocker: MockerFixture, tmp_path: Path
    ):
        log = RecordingLogger()
        mocker.patch.object(startup_module, "_log", log)
        harness = BadBatchHarness(
            mocker,
            entry_config(tmp_path, num_models_to_save=1, validate_amount=1),
            loader_batches=10,
            bad_valid_batches={0, 1},
        )

        with pytest.raises(SystemExit) as exit_info:
            harness.evaluate()

        assert exit_info.value.code == 1
        assert log.fatals == [
            "evaluation 'scheduled_validation' failed: rank 0: skipped 2 of 10 batches, more "
            "than the 10% a pass may skip"
        ]

    def test_a_pass_that_skips_exactly_a_tenth_of_its_batches_is_incomplete(
        self, mocker: MockerFixture, tmp_path: Path
    ):
        harness = BadBatchHarness(
            mocker,
            entry_config(tmp_path, num_models_to_save=1, validate_amount=1),
            loader_batches=10,
            bad_valid_batches={0},
        )

        outcome = harness.evaluate()

        # the threshold is a boundary, not a count: exactly a tenth passes
        assert outcome.state == EVAL_INCOMPLETE
        assert (outcome.ok_batches, outcome.skipped_batches) == (9, 1)


class TestWhatIsNotSkipped:
    def test_an_error_that_is_not_a_floating_point_failure_is_not_skipped(
        self, mocker: MockerFixture, tmp_path: Path
    ):
        log = RecordingLogger()
        mocker.patch.object(startup_module, "_log", log)
        harness = BadBatchHarness(
            mocker,
            entry_config(tmp_path, num_models_to_save=1, validate_amount=1),
            loader_batches=3,
            bad_valid_batches={1},
            valid_failure=RuntimeError,
        )

        with pytest.raises(SystemExit) as exit_info:
            harness.evaluate()

        assert exit_info.value.code == 1
        assert log.fatals == [
            "evaluation 'scheduled_validation' failed: rank 0: RuntimeError: non-finite loss "
            "elements detected: 1"
        ]

    def test_a_failure_of_the_data_iteration_is_not_skipped(
        self, mocker: MockerFixture, tmp_path: Path
    ):
        log = RecordingLogger()
        mocker.patch.object(startup_module, "_log", log)
        harness = BadBatchHarness(
            mocker,
            entry_config(tmp_path, num_models_to_save=1, validate_amount=1),
            loader_batches=2,
            valid_loader_fails=True,
        )

        with pytest.raises(SystemExit) as exit_info:
            harness.evaluate()

        assert exit_info.value.code == 1
        assert log.fatals == [
            "evaluation 'scheduled_validation' failed: rank 0: FloatingPointError: the dataset "
            "cannot be read"
        ]
        assert harness.log.warnings == []

    def test_a_failure_in_training_is_not_skipped(self, mocker: MockerFixture, tmp_path: Path):
        harness = BadBatchHarness(
            mocker,
            entry_config(tmp_path, num_models_to_save=1, validate_amount=1),
            loader_batches=2,
            bad_train_value=0,
        )

        with pytest.raises(FloatingPointError):
            harness.train()

        assert not any("Skipping batch" in warning for warning in harness.log.warnings)
        assert harness.loss_rows("valid") == []


class TestAnIncompletePassNeverReachesTheBoard:
    def test_the_run_records_the_skips_and_offers_only_the_complete_pass(
        self, mocker: MockerFixture, tmp_path: Path
    ):
        # the same batch fails once, so the first scheduled validation is
        # incomplete and the second one is complete
        harness = BadBatchHarness(
            mocker,
            entry_config(tmp_path, num_models_to_save=1, validate_amount=2),
            loader_batches=2,
            valid_loader_batches=10,
            bad_valid_batches={9},
            bad_valid_once=True,
        )

        harness.train()

        rows = harness.loss_rows("valid")
        assert [row["source"] for row in rows] == [EVAL_SCHEDULED, EVAL_SCHEDULED]
        assert [row["complete"] for row in rows] == ["False", "True"]
        assert [row["skipped_batches"] for row in rows] == ["1", "0"]
        # the incomplete pass is recorded, and it is not a candidate
        assert [loss for loss, _state in harness.board()] == [float(rows[1]["loss"])]
        assert harness.trainer._delivery_source == DELIVERY_TOPN_BEST

    def test_a_run_whose_validations_are_all_incomplete_does_not_deliver(
        self, mocker: MockerFixture, tmp_path: Path
    ):
        log = RecordingLogger()
        mocker.patch.object(startup_module, "_log", log)
        harness = BadBatchHarness(
            mocker,
            entry_config(tmp_path, num_models_to_save=1, validate_amount=1),
            loader_batches=2,
            valid_loader_batches=10,
            bad_valid_batches={9},
        )

        with pytest.raises(SystemExit) as exit_info:
            harness.train()

        assert exit_info.value.code == 1
        assert log.fatals == [
            "startup stage 'delivery' failed: RuntimeError: the run holds no complete validation "
            "candidate to deliver (num_models_to_save=1, resumed=False, skip_validation=False)"
        ]
        assert harness.board() == []
        rows = harness.loss_rows("valid")
        assert [row["complete"] for row in rows] == ["False"]
        assert [row["skipped_batches"] for row in rows] == ["1"]
        assert [row["source"] for row in rows] == [EVAL_SCHEDULED]
