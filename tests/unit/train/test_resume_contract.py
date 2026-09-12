"""A run restores state only from an explicitly configured checkpoint, and the
training position is read from that checkpoint rather than from a config file."""

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
import torch
import yaml
from pydantic import ValidationError
from pytest_mock import MockerFixture

import mblm.train.core.trainer as trainer_module
from mblm.train.core.config import (
    CoreIoConfig,
    CoreModelParams,
    CoreTrainConfig,
    GenericEntryConfig,
    GenericOutputConfig,
    ResumeConfig,
    ResumeMetadata,
    SummaryStats,
)
from mblm.train.core.trainer import CoreTrainer
from mblm.utils.distributed import ElasticRunVars
from mblm.utils.io import CheckpointCursor, save_training_checkpoint_state


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


def entry_payload() -> dict[str, Any]:
    return {
        "params": {"input_seq_len": 1},
        "train": {
            "target_elements": 1,
            "target_elements_strategy": "batch",
            "batch_size": 1,
            "learning_rate": 0.1,
            "gradient_accumulate_every": 1,
        },
        "io": {
            "name_model": "resume-contract",
            "output_dir": "unused",
            "num_models_to_save": 0,
            "validate_amount": 1,
            "log_train_loss_amount": 1,
        },
    }


def output_payload(parent_checkpoint: str | None) -> dict[str, Any]:
    payload = entry_payload()
    payload["resume"] = ResumeMetadata(parent_checkpoint=parent_checkpoint).model_dump()
    payload["summary"] = SummaryStats(
        training_start="2026-01-01T00:00:00",
        training_end="2026-01-01T00:00:01",
        num_workers=1,
        cuda_devices=[],
        parameter_count=1,
        error=None,
    ).model_dump()
    return payload


def entry_config(tmp_path: Path, *, resume: ResumeConfig | None = None) -> TinyEntryConfig:
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
            name_model="resume-contract",
            output_dir=str(tmp_path / "outputs"),
            num_models_to_save=0,
            validate_amount=1,
            log_train_loss_amount=1,
        ),
        resume=resume,
    )


def build_trainer(mocker: MockerFixture, tmp_path: Path, config: TinyEntryConfig) -> TinyTrainer:
    mocker.patch.object(TinyTrainer, "_create_output_dir", lambda _self, _io: tmp_path)
    mocker.patch.object(
        TinyTrainer, "configure_logger", lambda _self, *_args, **_kwargs: SilentLogger()
    )
    mocker.patch.object(TinyTrainer, "_init_distributed_model", lambda _self, model: model)
    return TinyTrainer(
        config,
        run_vars=ElasticRunVars(local_rank=0, world_size=1, is_cuda=False),
    )


def read_dumped_config(tmp_path: Path) -> dict[str, Any]:
    return yaml.safe_load((tmp_path / "config.yaml").read_text(encoding="utf-8"))


class TestResumeInputContract:
    @pytest.mark.parametrize(
        "legacy_key,value",
        [
            ("next_epoch_index", 0),
            ("next_batch_index", 0),
            ("migrate_embeddings", False),
            ("rename_modules", True),
            ("resumed_from", None),
        ],
    )
    def test_a_legacy_resume_key_fails_to_parse(self, legacy_key: str, value: Any):
        payload = entry_payload()
        payload["resume"] = {"checkpoint_file": "ckpt.pth", legacy_key: value}

        with pytest.raises(ValidationError, match=legacy_key):
            TinyEntryConfig.model_validate(payload)

    def test_auto_resume_latest_fails_to_parse(self):
        payload = entry_payload()
        payload["train"]["auto_resume_latest"] = False

        with pytest.raises(ValidationError, match="auto_resume_latest"):
            TinyEntryConfig.model_validate(payload)

    def test_an_unknown_train_key_fails_to_parse(self):
        payload = entry_payload()
        payload["train"]["not_a_train_option"] = 1

        with pytest.raises(ValidationError, match="not_a_train_option"):
            TinyEntryConfig.model_validate(payload)

    def test_the_checkpoint_file_is_the_only_resume_key(self):
        payload = entry_payload()
        payload["resume"] = {"checkpoint_file": "ckpt.pth"}

        config = TinyEntryConfig.model_validate(payload)

        assert set(ResumeConfig.model_fields) == {"checkpoint_file"}
        assert config.resume == ResumeConfig(checkpoint_file="ckpt.pth")

    def test_the_output_config_records_provenance_not_a_cursor(self):
        output = TinyOutputConfig.model_validate(output_payload("run1/latest.pth"))

        assert output.model_dump()["resume"] == {"parent_checkpoint": "run1/latest.pth"}

    def test_an_output_config_is_not_a_valid_input_config(self):
        output = TinyOutputConfig.model_validate(output_payload("run1/latest.pth"))

        with pytest.raises(ValidationError, match="parent_checkpoint"):
            TinyEntryConfig.model_validate(output.model_dump())

    def test_the_trainer_exposes_no_migration_hooks(self):
        assert not hasattr(CoreTrainer, "migrate_embeddings_if_enabled")
        assert not hasattr(CoreTrainer, "rename_modules_if_enabled")


class TestResumeRuntimeContract:
    def test_a_fresh_run_never_looks_for_a_checkpoint(self, mocker: MockerFixture, tmp_path: Path):
        # a run directory that the deleted auto-resume looked for: an unreadable
        # config plus a checkpoint that is not a checkpoint
        stale_run = tmp_path / "outputs" / "resume-contract_1"
        stale_run.mkdir(parents=True)
        (stale_run / "latest.pth").write_bytes(b"not a checkpoint")
        (stale_run / "config.yaml").write_text("resume: [", encoding="utf-8")
        load = mocker.patch.object(
            trainer_module,
            "load_training_checkpoint_state",
            side_effect=AssertionError("a fresh run must not load a checkpoint"),
        )

        build_trainer(mocker, tmp_path, entry_config(tmp_path))

        assert load.call_count == 0
        assert read_dumped_config(tmp_path)["resume"] == {"parent_checkpoint": None}

    def test_an_explicit_checkpoint_supplies_the_cursor(
        self, mocker: MockerFixture, tmp_path: Path
    ):
        checkpoint = tmp_path / "run1" / "latest.pth"
        save_training_checkpoint_state(
            checkpoint.parent,
            "latest",
            model=TinyModel(),
            loss=4.5,
            epoch=2,
            batch=5,
            cum_batch=17,
        )

        trainer = build_trainer(
            mocker,
            tmp_path,
            entry_config(tmp_path, resume=ResumeConfig(checkpoint_file=str(checkpoint))),
        )

        assert trainer._resume_cursor == CheckpointCursor(epoch=2, batch=5, cum_batch=17)
        dumped = read_dumped_config(tmp_path)
        assert dumped["resume"] == {"parent_checkpoint": str(checkpoint)}
        assert set(dumped["resume"]) == {"parent_checkpoint"}
