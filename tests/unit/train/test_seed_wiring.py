from contextlib import contextmanager
from datetime import timedelta
from typing import Any, Iterator

import pytest
from pytest_mock import MockerFixture

import mblm.train.mblm as train_module
from mblm.data.types import ModelMode
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
    def __init__(self, messages: list[str]):
        self._messages = messages

    def info(self, message: str) -> None:
        self._messages.append(message)

    def fatal(self, *args: Any, **_kwargs: Any) -> None:
        raise AssertionError(f"the entry reported a failure: {args}")


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
        name_model="seed-wiring",
        output_dir="unused",
        num_models_to_save=0,
        validate_amount=1,
        log_train_loss_amount=1,
        dataset_dir="unused",
        dataset_id="stubbed",
    )


def mbplm_entry_config(seed: int | None, **train_overrides: Any) -> TrainEntryConfig:
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
            seed=seed,
            **train_overrides,
        ),
        io=io_config(),
    )


def masked_entry_config(seed: int | None, **train_overrides: Any) -> TrainMaskedEntryConfig:
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
            seed=seed,
            **train_overrides,
        ),
        io=io_config(),
    )


def run_entry(
    mocker: MockerFixture, *, masked: bool, seed: int | None, **train_overrides: Any
) -> tuple[list[str], list[str], dict[str, Any]]:
    """
    Drive one training entry with every run piece stubbed, and record the order
    in which the entry touches them, plus the arguments it hands to the process group.
    """
    calls: list[str] = []
    messages: list[str] = []
    process_group_kwargs: dict[str, Any] = {}

    @contextmanager
    def fake_process_group(**kwargs: Any) -> Iterator[ElasticRunVars]:
        process_group_kwargs.update(kwargs)
        yield ElasticRunVars(local_rank=0, global_rank=0, world_size=1, is_cuda=False)

    def fake_admission(_params: Any, **_kwargs: Any) -> MambaAdmissionState:
        calls.append("admission")
        return MambaAdmissionState(required=False, ok=True)

    def fake_seed_run(base_seed: int | None, *, rank: int) -> int | None:
        calls.append(f"seed:{base_seed}:{rank}")
        return None if base_seed is None else base_seed + rank

    def fake_marker(*_args: Any, **_kwargs: Any) -> None:
        calls.append("marker")

    def fake_start_trainer(*_args: Any, **_kwargs: Any) -> None:
        calls.append("startup")

    class TrainerStub:
        def __init__(self, _config: Any, **_kwargs: Any):
            calls.append("trainer")
            self.output_dir = "unused"

        def train(self, _train_dataset: Any, _valid_dataset: Any) -> None:
            calls.append("train")

    class DatasetStub:
        @staticmethod
        def from_train_entry_config(*_args: Any, mode: ModelMode, **_kwargs: Any) -> object:
            calls.append(f"dataset:{mode.value}")
            return object()

        @staticmethod
        def supports_test_mode() -> bool:
            return False

    mocker.patch.object(
        train_module, "create_logger", lambda *_args, **_kwargs: RecordingLogger(messages)
    )
    mocker.patch.object(train_module, "bootstrap_log", lambda: RecordingLogger(messages))
    mocker.patch.object(train_module, "start_trainer", fake_start_trainer)
    mocker.patch.object(train_module, "process_group", fake_process_group)
    mocker.patch.object(train_module, "run_mamba_admission", fake_admission)
    mocker.patch.object(train_module, "seed_run", fake_seed_run)
    mocker.patch.object(train_module, "write_mamba_impl_marker", fake_marker)
    mocker.patch.object(train_module.torch.distributed, "get_rank", lambda: 0)
    mocker.patch.object(train_module.dataset_registry, "retrieve", lambda *_args: DatasetStub)
    mocker.patch.object(
        train_module.masked_dataset_registry, "retrieve", lambda *_args: DatasetStub
    )
    mocker.patch.object(train_module, "MegabyteTrainer", TrainerStub)
    mocker.patch.object(train_module, "MaskedTrainer", TrainerStub)

    if masked:
        train_encoder_mblm(masked_entry_config(seed, **train_overrides))
    else:
        train_mblm(mbplm_entry_config(seed, **train_overrides))
    return calls, messages, process_group_kwargs


class TestSeedWiring:
    def test_train_mblm_seeds_after_admission_and_before_the_datasets(self, mocker: MockerFixture):
        calls, messages, _ = run_entry(mocker, masked=False, seed=11)

        assert calls == [
            "admission",
            "seed:11:0",
            "dataset:train",
            "dataset:valid",
            "trainer",
            "startup",
            "marker",
            "train",
        ]
        assert messages == ["Distributed timeout: 600 seconds", "Effective seed: 11"]

    def test_train_encoder_mblm_seeds_after_admission_and_before_the_datasets(
        self, mocker: MockerFixture
    ):
        calls, messages, _ = run_entry(mocker, masked=True, seed=11)

        assert calls == [
            "admission",
            "seed:11:0",
            "dataset:train",
            "dataset:valid",
            "trainer",
            "startup",
            "marker",
            "train",
        ]
        assert messages == ["Distributed timeout: 600 seconds", "Effective seed: 11"]

    def test_without_a_base_seed_the_entries_do_not_seed(self, mocker: MockerFixture):
        calls, messages, _ = run_entry(mocker, masked=False, seed=None)

        assert calls == [
            "admission",
            "seed:None:0",
            "dataset:train",
            "dataset:valid",
            "trainer",
            "startup",
            "marker",
            "train",
        ]
        assert messages == ["Distributed timeout: 600 seconds"]


class TestDistributedTimeoutWiring:
    @pytest.mark.parametrize("masked", [False, True])
    def test_the_entries_pass_the_default_timeout_to_the_process_group(
        self, mocker: MockerFixture, masked: bool
    ):
        _, messages, process_group_kwargs = run_entry(mocker, masked=masked, seed=None)

        assert process_group_kwargs["timeout"] == timedelta(seconds=600)
        assert messages == ["Distributed timeout: 600 seconds"]

    @pytest.mark.parametrize("masked", [False, True])
    def test_the_configured_timeout_reaches_the_process_group_and_the_log(
        self, mocker: MockerFixture, masked: bool
    ):
        _, messages, process_group_kwargs = run_entry(
            mocker, masked=masked, seed=None, distributed_timeout_seconds=1234
        )

        assert process_group_kwargs["timeout"] == timedelta(seconds=1234)
        assert messages == ["Distributed timeout: 1234 seconds"]
