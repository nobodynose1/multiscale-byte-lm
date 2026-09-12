"""A training failure must leave the process instead of re-entering `_train`
inside it: recovery is a new process started from an explicit checkpoint."""

from collections.abc import Iterator
from pathlib import Path
from typing import Any, cast

import pytest
import torch
from pytest_mock import MockerFixture

from mblm.data.datasets import DistributedDataset
from mblm.train.core.config import (
    CoreIoConfig,
    CoreModelParams,
    CoreTrainConfig,
    GenericEntryConfig,
)
from mblm.train.core.startup import start_trainer
from mblm.train.core.trainer import CoreTrainer, CoreTrainerOptions
from mblm.utils.distributed import ElasticRunVars


class RecordingLogger:
    def __init__(self) -> None:
        self.fatals: list[str] = []

    def debug(self, *_args: Any, **_kwargs: Any) -> None:
        pass

    def info(self, *_args: Any, **_kwargs: Any) -> None:
        pass

    def warning(self, *_args: Any, **_kwargs: Any) -> None:
        pass

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
    def offset_to(self, epoch: int) -> None:
        pass


class TrainCalls:
    """Stand-in for `_train` counting how often the trainer runs it."""

    def __init__(self, *, model: TinyModel | None = None, error: Exception | None = None) -> None:
        self.model = model
        self.error = error
        self.count = 0

    def __call__(self, *_args: Any, **_kwargs: Any) -> TinyModel:
        self.count += 1
        if self.error is not None:
            raise self.error
        assert self.model is not None
        return self.model


def entry_config(tmp_path: Path) -> TinyEntryConfig:
    return TinyEntryConfig(
        params=TinyParams(input_seq_len=1),
        train=CoreTrainConfig(
            target_elements=1,
            target_elements_strategy="batch",
            batch_size=1,
            learning_rate=0.1,
            gradient_accumulate_every=1,
        ),
        io=CoreIoConfig(
            name_model="no-retry",
            output_dir=str(tmp_path),
            num_models_to_save=0,
            validate_amount=1,
            log_train_loss_amount=1,
        ),
    )


class TrainerHarness:
    """Drive the real `CoreTrainer.train()` over a stubbed `_train`."""

    def __init__(self, mocker: MockerFixture, tmp_path: Path, calls: TrainCalls) -> None:
        self.log = RecordingLogger()
        self.dump_output_config = mocker.patch.object(TinyTrainer, "_dump_output_config")
        mocker.patch.object(TinyTrainer, "create_output_dir", lambda _self: str(tmp_path))
        mocker.patch.object(
            TinyTrainer, "configure_logger", lambda _self, *_args, **_kwargs: self.log
        )
        mocker.patch.object(TinyTrainer, "_init_distributed_model", lambda _self, model: model)
        mocker.patch.object(TinyTrainer, "_train", calls)
        self.trainer = TinyTrainer(
            entry_config(tmp_path),
            run_vars=ElasticRunVars(local_rank=0, global_rank=0, world_size=1, is_cuda=False),
            options=CoreTrainerOptions(display_progress=False),
        )
        start_trainer(self.trainer, world_size=1)
        self.dataset = FakeDataset()

    def run(self) -> TinyModel | None:
        dataset = cast(DistributedDataset[torch.Tensor], self.dataset)
        return self.trainer.train(dataset, dataset)


class TestNoInProcessRetry:
    def test_a_failed_run_is_not_re_entered(self, mocker: MockerFixture, tmp_path: Path):
        calls = TrainCalls(error=RuntimeError("boom"))
        harness = TrainerHarness(mocker, tmp_path, calls)

        with pytest.raises(RuntimeError, match="boom"):
            harness.run()

        assert calls.count == 1

    def test_a_failed_run_reports_nothing_and_dumps_no_summary(
        self, mocker: MockerFixture, tmp_path: Path
    ):
        calls = TrainCalls(error=RuntimeError("boom"))
        harness = TrainerHarness(mocker, tmp_path, calls)

        with pytest.raises(RuntimeError, match="boom"):
            harness.run()

        assert harness.log.fatals == []
        assert harness.trainer._running_summary_stats.error is None
        assert harness.trainer._running_summary_stats.training_end == ""
        # `__init__` dumps the config once; a failed run dumps no summary
        assert harness.dump_output_config.call_count == 1

    def test_a_successful_run_still_finishes(self, mocker: MockerFixture, tmp_path: Path):
        model = TinyModel()
        calls = TrainCalls(model=model)
        harness = TrainerHarness(mocker, tmp_path, calls)

        best_model = harness.run()

        assert best_model is model
        assert calls.count == 1
        assert harness.trainer._running_summary_stats.training_end != ""
