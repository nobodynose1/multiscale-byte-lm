from contextlib import contextmanager
from typing import Any, Iterator

import pytest
from pytest_mock import MockerFixture

import mblm.train.mblm as train_module
from mblm.model.config import MBLMModelConfig
from mblm.model.mamba_admission import MambaAdmissionState
from mblm.model.transformer import TransformerBlock
from mblm.train.core.config import CoreTrainConfig, TrainMaskedConfig
from mblm.train.mblm import (
    TrainEntryConfig,
    TrainMaskedEntryConfig,
    TrainMaskedMBLMParams,
    TrainMBLMIoConfig,
    TrainMBLMParams,
    train_encoder_mblm,
    train_mblm,
)
from mblm.utils.distributed import ElasticRunVars


class RecordingLogger:
    def __init__(self) -> None:
        self.infos: list[str] = []
        self.fatals: list[str] = []

    def info(self, message: str) -> None:
        self.infos.append(message)

    def fatal(self, message: Any, *_args: Any, **_kwargs: Any) -> None:
        self.fatals.append(str(message))


class TrainingError(RuntimeError):
    """The failure the training loop hits after the run has started."""


def block() -> TransformerBlock:
    return TransformerBlock(
        attn_head_dims=64,
        attn_num_heads=16,
        attn_use_rot_embs=True,
        use_flash_attn=True,
        pos_emb_type="fixed",
    )


def model_config() -> MBLMModelConfig:
    return MBLMModelConfig(
        num_tokens=257,
        pad_token_id=256,
        hidden_dims=[64],
        num_layers=[1],
        seq_lens=[8],
        train_checkpoint_chunks=None,
        block=block(),
    )


def io_config() -> TrainMBLMIoConfig:
    return TrainMBLMIoConfig(
        name_model="fatal-exit",
        output_dir="unused",
        num_models_to_save=0,
        validate_amount=1,
        log_train_loss_amount=1,
        dataset_dir="unused",
        dataset_id="stubbed",
    )


def plain_entry_config() -> TrainEntryConfig:
    return TrainEntryConfig(
        params=TrainMBLMParams(
            num_tokens=257,
            pad_token_id=256,
            hidden_dims=[64],
            num_layers=[1],
            seq_lens=[8],
            train_checkpoint_chunks=None,
            block=block(),
            input_seq_len=8,
        ),
        train=CoreTrainConfig(
            target_elements=1,
            target_elements_strategy="batch",
            batch_size=1,
            learning_rate=0.001,
            gradient_accumulate_every=1,
        ),
        io=io_config(),
    )


def masked_entry_config() -> TrainMaskedEntryConfig:
    return TrainMaskedEntryConfig(
        params=TrainMaskedMBLMParams(
            mask_token_id=256,
            mblm_config=model_config(),
            input_seq_len=8,
        ),
        train=TrainMaskedConfig(
            target_elements=1,
            target_elements_strategy="batch",
            batch_size=1,
            learning_rate=0.001,
            gradient_accumulate_every=1,
            masking_proba=0.15,
        ),
        io=io_config(),
    )


@contextmanager
def process_group_stub(**_kwargs: Any) -> Iterator[ElasticRunVars]:
    yield ElasticRunVars(local_rank=0, world_size=1, is_cuda=False)


class DatasetStub:
    @staticmethod
    def from_train_entry_config(*_args: Any, **_kwargs: Any) -> object:
        return object()

    @staticmethod
    def supports_test_mode() -> bool:
        return True


class FailingTrainerStub:
    """A trainer whose training loop fails after the run has started."""

    def __init__(self, _config: Any, **_kwargs: Any) -> None:
        self.output_dir = "unused"

    def train(self, *_args: Any, **_kwargs: Any) -> None:
        raise TrainingError("the training loop failed")

    def test(self, *_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("a run that failed during training must not reach testing")


class TestFatalTrainingErrorExitsNonZero:
    @pytest.mark.parametrize("masked", [False, True])
    def test_a_fatal_training_error_exits_non_zero(self, mocker: MockerFixture, masked: bool):
        entry_log = RecordingLogger()
        shutdown = mocker.patch.object(train_module, "shutdown_log_handlers")

        mocker.patch.object(train_module, "create_logger", lambda *_args, **_kwargs: entry_log)
        mocker.patch.object(train_module, "process_group", process_group_stub)
        mocker.patch.object(
            train_module,
            "run_mamba_admission",
            lambda *_args, **_kwargs: MambaAdmissionState(required=False, ok=True),
        )
        mocker.patch.object(train_module, "seed_run", lambda *_args, **_kwargs: None)
        mocker.patch.object(train_module, "write_mamba_impl_marker", lambda *_args, **_kwargs: None)
        mocker.patch.object(train_module.torch.distributed, "get_rank", lambda: 0)
        mocker.patch.object(train_module.dataset_registry, "retrieve", lambda *_args: DatasetStub)
        mocker.patch.object(
            train_module.masked_dataset_registry, "retrieve", lambda *_args: DatasetStub
        )
        mocker.patch.object(train_module, "MegabyteTrainer", FailingTrainerStub)
        mocker.patch.object(train_module, "MaskedTrainer", FailingTrainerStub)

        with pytest.raises(SystemExit) as exit_info:
            if masked:
                train_encoder_mblm(masked_entry_config())
            else:
                train_mblm(plain_entry_config())

        assert exit_info.value.code == 1
        assert entry_log.fatals == ["the training loop failed"]
        shutdown.assert_called_once_with()
