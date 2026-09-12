from contextlib import contextmanager
from typing import Any, Iterator

import pytest
from pytest_mock import MockerFixture

import mblm.train.core.startup as startup_module
import mblm.train.mblm as train_module
from mblm.data.types import ModelMode
from mblm.model.config import MBLMModelConfig
from mblm.model.mamba_admission import MambaAdmissionState
from mblm.model.transformer import TransformerBlock
from mblm.train.core.config import CoreTrainConfig, TrainMaskedConfig
from mblm.train.core.startup import StageOutcome
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
        self.messages: list[str] = []
        self.fatals: list[str] = []

    def info(self, message: str) -> None:
        self.messages.append(message)

    def fatal(self, message: Any, *_args: Any, **_kwargs: Any) -> None:
        self.fatals.append(str(message))


def noop(*_args: Any, **_kwargs: Any) -> None:
    """Stand-in for a collective call in the single-rank tests."""


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
        name_model="dataset-stage",
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


def process_group_stub(world_size: int):
    @contextmanager
    def process_group(**_kwargs: Any) -> Iterator[ElasticRunVars]:
        yield ElasticRunVars(local_rank=0, global_rank=0, world_size=world_size, is_cuda=False)

    return process_group


class EntryHarness:
    """
    Drive one training entry with every run piece stubbed, and record which of
    them the entry touched.
    """

    def __init__(
        self,
        mocker: MockerFixture,
        *,
        world_size: int = 1,
        fail_registry: bool = False,
        fail_dataset_build: bool = False,
    ) -> None:
        self.calls: list[str] = []
        self.entry_log = RecordingLogger()
        self.bootstrap_log = RecordingLogger()
        self._world_size = world_size
        self._fail_registry = fail_registry
        self._fail_dataset_build = fail_dataset_build
        self._install(mocker)

    def _install(self, mocker: MockerFixture) -> None:
        calls = self.calls

        def fake_admission(_params: Any, **_kwargs: Any) -> MambaAdmissionState:
            calls.append("admission")
            return MambaAdmissionState(required=False, ok=True)

        def fake_marker(*_args: Any, **_kwargs: Any) -> None:
            calls.append("marker")

        def fake_retrieve(*_args: Any) -> Any:
            if self._fail_registry:
                raise KeyError("unknown-dataset")
            return DatasetStub

        class TrainerStub:
            def __init__(self, _config: Any, **_kwargs: Any) -> None:
                calls.append("trainer")
                self.output_dir = "unused"

            def record_data_lineage(self, _lineage: Any) -> None:
                pass

            def train(self, *_args: Any, **_kwargs: Any) -> None:
                calls.append("train")

        class DatasetStub:
            @staticmethod
            def from_train_entry_config(*_args: Any, mode: ModelMode, **_kwargs: Any) -> object:
                calls.append(f"dataset:{mode.value}")
                if self._fail_dataset_build and mode is ModelMode.TRAIN:
                    raise RuntimeError("the dataset cannot be read")
                return DatasetStub()

            @staticmethod
            def supports_test_mode() -> bool:
                return False

            @staticmethod
            def data_lineage() -> None:
                return None

        def fake_start_trainer(*_args: Any, **_kwargs: Any) -> None:
            calls.append("startup")

        mocker.patch.object(train_module, "create_logger", lambda *_args, **_kwargs: self.entry_log)
        mocker.patch.object(train_module, "bootstrap_log", lambda: self.bootstrap_log)
        mocker.patch.object(train_module, "start_trainer", fake_start_trainer)
        mocker.patch.object(startup_module, "_log", self.bootstrap_log)
        mocker.patch.object(train_module, "process_group", process_group_stub(self._world_size))
        mocker.patch.object(train_module, "run_mamba_admission", fake_admission)
        mocker.patch.object(train_module, "seed_run", lambda *_args, **_kwargs: None)
        mocker.patch.object(train_module, "write_mamba_impl_marker", fake_marker)
        mocker.patch.object(train_module.torch.distributed, "get_rank", lambda: 0)
        mocker.patch.object(train_module.dataset_registry, "retrieve", fake_retrieve)
        mocker.patch.object(train_module.masked_dataset_registry, "retrieve", fake_retrieve)
        mocker.patch.object(train_module, "MegabyteTrainer", TrainerStub)
        mocker.patch.object(train_module, "MaskedTrainer", TrainerStub)

    def run(self, *, masked: bool) -> None:
        if masked:
            train_encoder_mblm(masked_entry_config())
        else:
            train_mblm(plain_entry_config())


class TestDatasetStage:
    @pytest.mark.parametrize("masked", [False, True])
    def test_a_ready_dataset_stage_lets_the_run_reach_the_trainer(
        self, mocker: MockerFixture, masked: bool
    ):
        harness = EntryHarness(mocker)

        harness.run(masked=masked)

        assert harness.calls == [
            "admission",
            "dataset:train",
            "dataset:valid",
            "trainer",
            "startup",
            "marker",
            "train",
        ]
        assert harness.entry_log.fatals == []
        assert harness.bootstrap_log.fatals == []

    @pytest.mark.parametrize("masked", [False, True])
    def test_a_failing_dataset_build_stops_the_run_before_the_trainer(
        self, mocker: MockerFixture, masked: bool
    ):
        harness = EntryHarness(mocker, fail_dataset_build=True)

        with pytest.raises(SystemExit) as exit_info:
            harness.run(masked=masked)

        assert exit_info.value.code == 1
        assert harness.calls == ["admission", "dataset:train"]
        assert harness.entry_log.fatals == []
        assert harness.bootstrap_log.fatals == [
            "startup stage 'datasets' failed: RuntimeError: the dataset cannot be read"
        ]

    def test_a_failing_registry_lookup_stops_the_run(self, mocker: MockerFixture):
        harness = EntryHarness(mocker, fail_registry=True)

        with pytest.raises(SystemExit) as exit_info:
            harness.run(masked=False)

        assert exit_info.value.code == 1
        assert harness.calls == ["admission"]
        assert len(harness.bootstrap_log.fatals) == 1
        assert "startup stage 'datasets' failed" in harness.bootstrap_log.fatals[0]
        assert "unknown-dataset" in harness.bootstrap_log.fatals[0]

    def test_a_rank_that_built_its_datasets_still_stops_when_another_rank_failed(
        self, mocker: MockerFixture
    ):
        harness = EntryHarness(mocker, world_size=2)
        mocker.patch.object(startup_module.dist, "all_gather_object", noop)
        mocker.patch.object(startup_module.dist, "get_rank", lambda: 1)

        def broadcast_object_list(decision: list, **_kwargs: Any) -> None:
            decision[0] = StageOutcome(
                stage="datasets", ok=False, reason="rank 0: RuntimeError: boom"
            )

        mocker.patch.object(startup_module.dist, "broadcast_object_list", broadcast_object_list)

        with pytest.raises(SystemExit) as exit_info:
            harness.run(masked=False)

        assert exit_info.value.code == 1
        assert harness.calls == ["admission", "dataset:train", "dataset:valid"]
        assert harness.entry_log.fatals == []
        assert harness.bootstrap_log.fatals == [
            "startup stage 'datasets' failed: rank 0: RuntimeError: boom"
        ]
