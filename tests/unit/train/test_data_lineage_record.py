"""The output config of a run attests the enumeration rule the code applied and
the digest of the inputs the run actually read."""

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import torch
import yaml
from pytest_mock import MockerFixture

from mblm.data.utils import DATA_ENUMERATION_VERSION
from mblm.train.core.config import (
    CoreIoConfig,
    CoreModelParams,
    CoreTrainConfig,
    DataLineage,
    GenericEntryConfig,
    GenericOutputConfig,
)
from mblm.train.core.startup import start_trainer
from mblm.train.core.trainer import CoreTrainer
from mblm.utils.distributed import ElasticRunVars


class TinyParams(CoreModelParams):
    pass


class TinyEntryConfig(GenericEntryConfig[TinyParams, CoreTrainConfig, CoreIoConfig]):
    pass


class TinyOutputConfig(GenericOutputConfig[TinyParams, CoreTrainConfig, CoreIoConfig]):
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


class SilentLogger:
    def debug(self, *_args: Any, **_kwargs: Any) -> None:
        pass

    def info(self, *_args: Any, **_kwargs: Any) -> None:
        pass

    def warning(self, *_args: Any, **_kwargs: Any) -> None:
        pass

    def error(self, *_args: Any, **_kwargs: Any) -> None:
        pass

    def fatal(self, *_args: Any, **_kwargs: Any) -> None:
        pass


LINEAGE = DataLineage(
    enumeration=DATA_ENUMERATION_VERSION,
    train="a" * 64,
    validation="b" * 64,
)


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
            name_model="data-lineage",
            output_dir=str(tmp_path / "outputs"),
            num_models_to_save=0,
            validate_amount=1,
            log_train_loss_amount=1,
        ),
    )


def build_trainer(
    mocker: MockerFixture, tmp_path: Path, lineage: DataLineage | None = None
) -> TinyTrainer:
    mocker.patch.object(TinyTrainer, "create_output_dir", lambda _self: str(tmp_path))
    mocker.patch.object(
        TinyTrainer, "configure_logger", lambda _self, *_args, **_kwargs: SilentLogger()
    )
    mocker.patch.object(TinyTrainer, "_init_distributed_model", lambda _self, model: model)
    trainer = TinyTrainer(
        entry_config(tmp_path),
        run_vars=ElasticRunVars(local_rank=0, global_rank=0, world_size=1, is_cuda=False),
    )
    if lineage is not None:
        # the entries record the lineage before the run owns an output directory
        trainer.record_data_lineage(lineage)
    start_trainer(trainer, world_size=1)
    return trainer


def read_dumped_config(tmp_path: Path) -> dict[str, Any]:
    return yaml.safe_load((tmp_path / "config.yaml").read_text(encoding="utf-8"))


class TestDataLineageRecord:
    def test_a_recorded_lineage_reaches_the_output_config(
        self, mocker: MockerFixture, tmp_path: Path
    ):
        build_trainer(mocker, tmp_path, LINEAGE)

        dumped = read_dumped_config(tmp_path)

        assert dumped["data_lineage"] == {
            "enumeration": DATA_ENUMERATION_VERSION,
            "train": "a" * 64,
            "validation": "b" * 64,
        }

    def test_a_run_that_records_no_lineage_writes_a_null_lineage(
        self, mocker: MockerFixture, tmp_path: Path
    ):
        build_trainer(mocker, tmp_path)

        assert read_dumped_config(tmp_path)["data_lineage"] is None

    def test_the_recorded_lineage_is_an_output_field_of_the_run(
        self, mocker: MockerFixture, tmp_path: Path
    ):
        trainer = build_trainer(mocker, tmp_path, LINEAGE)

        assert trainer._data_lineage == LINEAGE
        assert "data_lineage" in TinyOutputConfig.model_fields

    def test_the_enumeration_rule_is_a_named_version(self):
        assert DATA_ENUMERATION_VERSION == "sorted-name-bytes-v1"
