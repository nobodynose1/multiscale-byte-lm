"""The model a run delivers for testing is chosen in four states: a run that
stores no candidate and a resumed run whose internal option skipped validation
deliver their final training state, a run whose board holds a complete
validation delivers the best candidate, and a run that should have filled its
board and did not fails instead of delivering an unranked model as if it had
been chosen."""

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
    ResumeConfig,
)
from mblm.train.core.startup import EVAL_RESUME, EVAL_SCHEDULED, start_trainer
from mblm.train.core.trainer import (
    DELIVERY_FINAL_UNRANKED,
    DELIVERY_TOPN_BEST,
    SOURCE_TRAIN,
    CoreTrainer,
    CoreTrainerOptions,
)
from mblm.utils.distributed import ElasticRunVars
from mblm.utils.io import CheckpointCursor
from mblm.utils.top_n import TopN


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


class FakeDataset:
    def __init__(self) -> None:
        self.offsets: list[int] = []

    def offset_to(self, epoch: int) -> None:
        self.offsets.append(epoch)


class FakeLoader:
    """A loader that is re-iterated for every epoch of the cycler, and whose
    every batch carries a different loss."""

    def __init__(self, batches: int) -> None:
        self._batches = batches

    def __len__(self) -> int:
        return self._batches

    def __iter__(self) -> Iterator[torch.Tensor]:
        return iter([torch.full((1,), float(index)) for index in range(self._batches)])


def entry_config(
    tmp_path: Path,
    *,
    num_models_to_save: int,
    validate_amount: int,
    target_elements: int = 2,
    resume: ResumeConfig | None = None,
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
            name_model="delivery",
            output_dir=str(tmp_path / "runs"),
            num_models_to_save=num_models_to_save,
            validate_amount=validate_amount,
            log_train_loss_amount=1,
        ),
        resume=resume,
    )


class DeliveryHarness:
    """Run the real `_train` loop with validation on, and read back what it
    delivered."""

    def __init__(
        self,
        mocker: MockerFixture,
        config: TinyEntryConfig,
        *,
        loader_batches: int,
        skip_validation: bool = False,
        cursor: CheckpointCursor | None = None,
    ) -> None:
        self.log = RecordingLogger()
        self.forward_batches = 0
        self.get_top_calls: list[int | None] = []
        self.dataset = FakeDataset()
        loader = FakeLoader(batches=loader_batches)

        def count_forward(
            _trainer: Any, model: TinyModel, batch: torch.Tensor, device: str
        ) -> torch.Tensor:
            self.forward_batches += 1
            return model(batch.to(device)).sum()

        real_get_top = TopN.get_top

        def record_get_top(instance: TopN[Any], best: int | None = None) -> Any:
            self.get_top_calls.append(best)
            return real_get_top(instance, best)

        mocker.patch.object(TopN, "get_top", record_get_top)
        mocker.patch.object(TinyTrainer, "configure_logger", lambda _self, *_a, **_kw: self.log)
        mocker.patch.object(TinyTrainer, "_init_distributed_model", lambda _self, model: model)
        mocker.patch.object(
            TinyTrainer, "get_train_dataloader", lambda _self, _dataset, **_kwargs: loader
        )
        mocker.patch.object(
            TinyTrainer, "get_valid_dataloader", lambda _self, _dataset, **_kwargs: loader
        )
        mocker.patch.object(
            TinyTrainer, "get_test_dataloader", lambda _self, _dataset, **_kwargs: loader
        )
        mocker.patch.object(TinyTrainer, "model_forward", count_forward)
        # the candidate board is entered under a barrier; a single rank has no
        # process group to enter it with
        mocker.patch.object(trainer_module.dist, "barrier", noop)
        if cursor is not None:
            mocker.patch.object(
                trainer_module,
                "load_training_checkpoint_state",
                lambda _path, _model, **_kwargs: (cursor, 2.5),
            )

        self.trainer = TinyTrainer(
            config,
            run_vars=ElasticRunVars(local_rank=0, global_rank=0, world_size=1, is_cuda=False),
            options=CoreTrainerOptions(
                display_progress=False,
                track_first_fw_bw_exec_times=None,
                skip_validation=skip_validation,
                amp_dtype=torch.bfloat16,
            ),
        )
        start_trainer(self.trainer, world_size=1)

    def train(self) -> TinyModel:
        dataset = cast(DistributedDataset[torch.Tensor], self.dataset)
        return self.trainer._train(dataset, dataset)

    def test(self) -> None:
        dataset = cast(DistributedDataset[torch.Tensor], self.dataset)
        self.trainer.test(dataset, self.trainer._model)

    def loss_rows(self, kind: str = "train") -> list[dict[str, str]]:
        with (self.trainer.output_dir / "loss.csv").open(encoding="utf-8", newline="") as file:
            return [row for row in csv.DictReader(file) if row["kind"] == kind]

    def board(self) -> list[tuple[float, Any]]:
        return self.trainer._top_n_models.get_top()


class TestTheDeliveredState:
    def test_a_run_that_stores_no_candidate_delivers_its_final_state(
        self, mocker: MockerFixture, tmp_path: Path
    ):
        harness = DeliveryHarness(
            mocker,
            entry_config(tmp_path, num_models_to_save=0, validate_amount=1),
            loader_batches=2,
        )

        harness.train()

        assert harness.trainer._delivery_source == DELIVERY_FINAL_UNRANKED
        assert 1 not in harness.get_top_calls
        assert [row["source"] for row in harness.loss_rows()] == [SOURCE_TRAIN]
        assert harness.loss_rows() == [
            {**row, "complete": "True", "skipped_batches": "0"} for row in harness.loss_rows()
        ]

    def test_a_resumed_run_that_stores_no_candidate_does_not_re_evaluate(
        self, mocker: MockerFixture, tmp_path: Path
    ):
        harness = DeliveryHarness(
            mocker,
            entry_config(
                tmp_path,
                num_models_to_save=0,
                validate_amount=2,
                resume=ResumeConfig(checkpoint_file="unused.pth"),
            ),
            loader_batches=2,
            cursor=CheckpointCursor(epoch=0, batch=0, cum_batch=1),
        )

        harness.train()

        # the re-evaluation only exists to fill a board this run does not keep
        assert [row["source"] for row in harness.loss_rows("valid")] == [EVAL_SCHEDULED]
        assert harness.trainer._delivery_source == DELIVERY_FINAL_UNRANKED
        assert 1 not in harness.get_top_calls

    def test_a_resumed_run_whose_validation_is_skipped_delivers_its_final_state(
        self, mocker: MockerFixture, tmp_path: Path
    ):
        harness = DeliveryHarness(
            mocker,
            entry_config(
                tmp_path,
                num_models_to_save=1,
                validate_amount=1,
                resume=ResumeConfig(checkpoint_file="unused.pth"),
            ),
            loader_batches=2,
            skip_validation=True,
            cursor=CheckpointCursor(epoch=0, batch=0, cum_batch=1),
        )

        harness.train()

        assert harness.trainer._delivery_source == DELIVERY_FINAL_UNRANKED
        assert harness.loss_rows("valid") == []
        assert any("internal skip" in warning for warning in harness.log.warnings)
        assert 1 not in harness.get_top_calls

    def test_a_run_whose_board_holds_a_complete_validation_delivers_the_best(
        self, mocker: MockerFixture, tmp_path: Path
    ):
        harness = DeliveryHarness(
            mocker,
            entry_config(tmp_path, num_models_to_save=1, validate_amount=2),
            loader_batches=2,
        )

        harness.train()

        assert harness.trainer._delivery_source == DELIVERY_TOPN_BEST
        assert 1 in harness.get_top_calls
        valid_losses = [float(row["loss"]) for row in harness.loss_rows("valid")]
        # the board holds validation-side numbers, and only the best of them
        assert [loss for loss, _state in harness.board()] == [min(valid_losses)]

    def test_a_fresh_run_that_skips_validation_fails_instead_of_delivering(
        self, mocker: MockerFixture, tmp_path: Path
    ):
        log = RecordingLogger()
        mocker.patch.object(startup_module, "_log", log)
        harness = DeliveryHarness(
            mocker,
            entry_config(tmp_path, num_models_to_save=1, validate_amount=1),
            loader_batches=2,
            skip_validation=True,
        )

        with pytest.raises(SystemExit) as exit_info:
            harness.train()

        assert exit_info.value.code == 1
        assert log.fatals == [
            "startup stage 'delivery' failed: RuntimeError: the run holds no complete "
            "validation candidate to deliver (num_models_to_save=1, resumed=False, "
            "skip_validation=True)"
        ]


class TestTheReEvaluationOfARestoredModel:
    def test_the_restored_model_enters_the_board_with_its_validation_loss(
        self, mocker: MockerFixture, tmp_path: Path
    ):
        # a run resumed at the end of its target trains nothing at all: the
        # re-evaluation is the only thing that can put its board on the map
        harness = DeliveryHarness(
            mocker,
            entry_config(
                tmp_path,
                num_models_to_save=1,
                validate_amount=1,
                target_elements=6,
                resume=ResumeConfig(checkpoint_file="unused.pth"),
            ),
            loader_batches=2,
            cursor=CheckpointCursor(epoch=0, batch=0, cum_batch=6),
        )

        harness.train()

        # the two forward passes are the validation batches: not one training
        # batch ran, and the board still holds the restored model
        assert harness.forward_batches == 2
        assert harness.loss_rows() == []
        assert [row["source"] for row in harness.loss_rows("valid")] == [EVAL_RESUME]
        assert harness.trainer._delivery_source == DELIVERY_TOPN_BEST
        assert [loss for loss, _state in harness.board()] == [
            float(harness.loss_rows("valid")[0]["loss"])
        ]

    def test_a_re_evaluated_run_keeps_the_restored_model_when_training_makes_it_worse(
        self, mocker: MockerFixture, tmp_path: Path
    ):
        # the checkpoint is restored from a state whose loss the re-evaluation
        # measures; the training that follows is free to lose to it
        harness = DeliveryHarness(
            mocker,
            entry_config(
                tmp_path,
                num_models_to_save=1,
                validate_amount=2,
                target_elements=4,
                resume=ResumeConfig(checkpoint_file="unused.pth"),
            ),
            loader_batches=2,
            cursor=CheckpointCursor(epoch=0, batch=0, cum_batch=2),
        )

        harness.train()

        rows = harness.loss_rows("valid")
        assert [row["source"] for row in rows] == [EVAL_RESUME, EVAL_SCHEDULED]
        assert [row["complete"] for row in rows] == ["True", "True"]
        assert [row["skipped_batches"] for row in rows] == ["0", "0"]
        assert [loss for loss, _state in harness.board()] == [
            min(float(row["loss"]) for row in rows)
        ]


class TestTheTestRowNamesTheDeliveredState:
    def test_the_row_of_a_delivered_candidate_names_the_board(
        self, mocker: MockerFixture, tmp_path: Path
    ):
        harness = DeliveryHarness(
            mocker,
            entry_config(tmp_path, num_models_to_save=1, validate_amount=2),
            loader_batches=2,
        )
        harness.train()

        harness.test()

        rows = harness.loss_rows("test")
        assert [row["source"] for row in rows] == [DELIVERY_TOPN_BEST]
        assert [row["complete"] for row in rows] == ["True"]

    def test_the_row_of_an_unranked_delivery_names_that_too(
        self, mocker: MockerFixture, tmp_path: Path
    ):
        harness = DeliveryHarness(
            mocker,
            entry_config(tmp_path, num_models_to_save=0, validate_amount=1),
            loader_batches=2,
        )
        harness.train()

        harness.test()

        rows = harness.loss_rows("test")
        assert [row["source"] for row in rows] == [DELIVERY_FINAL_UNRANKED]
        assert [row["skipped_batches"] for row in rows] == ["0"]
