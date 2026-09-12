"""A run that finishes without a single optimizer step has trained nothing and
must not be reported as a successful run. Only a resume cursor that already
reached the target is a legitimate zero-step run."""

from collections.abc import Iterator
from pathlib import Path
from typing import Any, cast

import pytest
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
from mblm.train.core.trainer import CoreTrainer, CoreTrainerOptions
from mblm.utils.distributed import ElasticRunVars


class RecordingLogger:
    def __init__(self) -> None:
        self.infos: list[str] = []
        self.fatals: list[str] = []

    def debug(self, *_args: Any, **_kwargs: Any) -> None:
        pass

    def info(self, message: str) -> None:
        self.infos.append(message)

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
    def __init__(self) -> None:
        self.offsets: list[int] = []

    def offset_to(self, epoch: int) -> None:
        self.offsets.append(epoch)


class FakeLoader:
    def __init__(self, batches: int) -> None:
        self._batches = batches

    def __len__(self) -> int:
        return self._batches

    def __iter__(self) -> Iterator[torch.Tensor]:
        return iter([torch.zeros(1) for _ in range(self._batches)])


def entry_config(
    tmp_path: Path,
    *,
    gradient_accumulate_every: int,
    resume: ResumeConfig | None = None,
) -> TinyEntryConfig:
    return TinyEntryConfig(
        params=TinyParams(input_seq_len=1),
        train=CoreTrainConfig(
            target_elements=1,
            target_elements_strategy="batch",
            batch_size=1,
            learning_rate=0.1,
            gradient_accumulate_every=gradient_accumulate_every,
        ),
        io=CoreIoConfig(
            name_model="zero-step",
            output_dir=str(tmp_path),
            num_models_to_save=0,
            validate_amount=1,
            log_train_loss_amount=1,
        ),
        resume=resume,
    )


class TrainerHarness:
    """Run the real `_train` loop over a tiny in-memory model."""

    def __init__(self, mocker: MockerFixture, tmp_path: Path, config: TinyEntryConfig) -> None:
        self.log = RecordingLogger()
        self.trainer = self._build(mocker, tmp_path, config)
        self.dataset = FakeDataset()
        loader = FakeLoader(batches=1)
        mocker.patch.object(
            TinyTrainer, "get_train_dataloader", lambda _self, _dataset, **_kwargs: loader
        )
        mocker.patch.object(
            TinyTrainer, "get_valid_dataloader", lambda _self, _dataset, **_kwargs: loader
        )

    def _build(self, mocker: MockerFixture, tmp_path: Path, config: TinyEntryConfig) -> TinyTrainer:
        log = self.log
        mocker.patch.object(TinyTrainer, "_create_output_dir", lambda _self, _io: tmp_path)
        mocker.patch.object(TinyTrainer, "configure_logger", lambda _self, *_args, **_kwargs: log)
        mocker.patch.object(TinyTrainer, "_init_distributed_model", lambda _self, model: model)
        mocker.patch.object(TinyTrainer, "_dump_output_config")

        trainer = TinyTrainer(
            config,
            run_vars=ElasticRunVars(local_rank=0, world_size=1, is_cuda=False),
            options=CoreTrainerOptions(
                display_progress=False,
                track_first_fw_bw_exec_times=None,
                skip_validation=True,
                amp_dtype=torch.bfloat16,
            ),
        )
        return trainer

    def train(self) -> TinyModel:
        dataset = cast(DistributedDataset[torch.Tensor], self.dataset)
        return self.trainer._train(dataset, dataset)


class TestZeroStepGuard:
    def test_a_fresh_run_without_a_single_optimizer_step_fails(
        self, mocker: MockerFixture, tmp_path: Path
    ):
        # one batch iteration, accumulated over two: the optimizer is never stepped
        harness = TrainerHarness(
            mocker, tmp_path, entry_config(tmp_path, gradient_accumulate_every=2)
        )

        with pytest.raises(RuntimeError, match="without a single optimizer step"):
            harness.train()

        assert harness.log.fatals == [
            "Finished training without a single optimizer step: "
            "1 batch iterations, gradient accumulation of 2"
        ]

    def test_a_resume_that_already_reached_the_target_may_finish_without_steps(
        self, mocker: MockerFixture, tmp_path: Path
    ):
        mocker.patch.object(
            trainer_module,
            "load_training_checkpoint_state",
            lambda _path, model, **_kwargs: (model, 12.5, {}),
        )
        # the checkpoint cursor sits at epoch 1 of a loader holding a single
        # batch: the target is already reached, nothing is left to train
        harness = TrainerHarness(
            mocker,
            tmp_path,
            entry_config(
                tmp_path,
                gradient_accumulate_every=2,
                resume=ResumeConfig(
                    checkpoint_file="unused.pth", next_epoch_index=1, next_batch_index=0
                ),
            ),
        )

        best_model = harness.train()

        assert harness.log.fatals == []
        assert isinstance(best_model, TinyModel)
        assert "Finished training" in harness.log.infos

    def test_a_run_that_completed_an_optimizer_step_succeeds(
        self, mocker: MockerFixture, tmp_path: Path
    ):
        # the single batch iteration lands on an accumulation boundary
        harness = TrainerHarness(
            mocker, tmp_path, entry_config(tmp_path, gradient_accumulate_every=1)
        )

        best_model = harness.train()

        assert harness.log.fatals == []
        assert isinstance(best_model, TinyModel)
