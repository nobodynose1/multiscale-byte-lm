"""The startup stages of a trainer run in a fixed order after the process group
is up, and no rank writes a run artefact before the run owns its output
directory."""

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
import torch
from pytest_mock import MockerFixture

import mblm.train.core.startup as startup_module
from mblm.train.core.config import (
    CoreIoConfig,
    CoreModelParams,
    CoreTrainConfig,
    GenericEntryConfig,
)
from mblm.train.core.startup import StageOutcome, bootstrap_log, start_trainer
from mblm.train.core.trainer import FRESH, RESUME, CoreTrainer, CoreTrainerOptions
from mblm.utils.distributed import ElasticRunVars


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

    def fatal(self, message: str, *_args: Any, **_kwargs: Any) -> None:
        self.fatals.append(message)


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


def entry_config(
    tmp_path: Path,
    *,
    target_elements: int = 2,
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
            name_model="startup",
            output_dir=str(tmp_path / "runs"),
            num_models_to_save=0,
            validate_amount=1,
            log_train_loss_amount=1,
        ),
    )


class ProbeTrainer(CoreTrainer[TinyModel, torch.Tensor, TinyParams, CoreTrainConfig, CoreIoConfig]):
    """A trainer that records the startup stages as the driver walks them."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.calls: list[str] = []
        self.parent_existed_at_components: bool | None = None
        self.output_dir_at_components: Path | None = None
        self.logger_at_components: Any = None

    def init_model(self) -> TinyModel:
        self.calls.append("init_model")
        return TinyModel()

    def model_forward(self, model: TinyModel, batch: torch.Tensor, device: str) -> torch.Tensor:
        return model(batch.to(device)).sum()

    def configure_optimizer(
        self, parameters: Iterator[torch.nn.Parameter]
    ) -> torch.optim.Optimizer:
        self.calls.append("configure_optimizer")
        return torch.optim.SGD(parameters, lr=0.1)

    def build_model(self) -> None:
        self.calls.append("build_model")
        super().build_model()

    def build_training_components(self) -> None:
        self.calls.append("build_training_components")
        self.parent_existed_at_components = Path(self.config.io.output_dir).exists()
        self.output_dir_at_components = self._output_dir
        self.logger_at_components = self._log
        super().build_training_components()

    def preflight_checkpoint(self) -> str:
        self.calls.append("preflight_checkpoint")
        return super().preflight_checkpoint()

    def create_output_dir(self) -> str | None:
        self.calls.append("create_output_dir")
        return super().create_output_dir()

    def initialize_outputs(self, output_dir: str) -> None:
        self.calls.append("initialize_outputs")
        super().initialize_outputs(output_dir)


def build_trainer(
    mocker: MockerFixture,
    tmp_path: Path,
    *,
    world_size: int = 1,
    global_rank: int = 0,
    target_elements: int = 2,
    gradient_accumulate_every: int = 1,
) -> ProbeTrainer:
    mocker.patch.object(
        ProbeTrainer, "configure_logger", lambda _self, *_args, **_kwargs: SilentLogger()
    )
    mocker.patch.object(ProbeTrainer, "_init_distributed_model", lambda _self, model: model)
    return ProbeTrainer(
        entry_config(
            tmp_path,
            target_elements=target_elements,
            gradient_accumulate_every=gradient_accumulate_every,
        ),
        run_vars=ElasticRunVars(
            local_rank=global_rank, global_rank=global_rank, world_size=world_size, is_cuda=False
        ),
        options=CoreTrainerOptions(display_progress=False),
    )


def stage_order(trainer: ProbeTrainer) -> list[str]:
    stages = {
        "build_model",
        "build_training_components",
        "preflight_checkpoint",
        "create_output_dir",
        "initialize_outputs",
    }
    return [call for call in trainer.calls if call in stages]


class SimulatedGroup:
    """
    A process group in which every rank but this one reports the outcome the
    test gives it. The unifier runs rank 0's aggregate, exactly as the process
    group does in production.
    """

    def __init__(self, mocker: MockerFixture, *, rank: int, world_size: int) -> None:
        self.rank = rank
        self.world_size = world_size
        self.rank_zero: dict[str, StageOutcome] = {}
        self.stages: list[str] = []
        self._gathered: list[StageOutcome | None] = []
        mocker.patch.object(startup_module.dist, "all_gather_object", self._all_gather_object)
        mocker.patch.object(startup_module.dist, "get_rank", lambda: rank)
        mocker.patch.object(
            startup_module.dist, "broadcast_object_list", self._broadcast_object_list
        )

    def reports_rank_zero(self, stage: str, **fields: Any) -> None:
        self.rank_zero[stage] = StageOutcome(stage=stage, **fields)

    def _all_gather_object(self, gathered: list, local: StageOutcome) -> None:
        self.stages.append(local.stage)
        # a rank that is not told otherwise agrees with this one
        agreed = StageOutcome(stage=local.stage, ok=True, state=local.state)
        for other in range(self.world_size):
            gathered[other] = self.rank_zero.get(local.stage, agreed)
        gathered[self.rank] = local
        self._gathered = list(gathered)

    def _broadcast_object_list(self, decision: list, **_kwargs: Any) -> None:
        decision[0] = startup_module._aggregate(self._gathered)


class TestStartupStageOrder:
    def test_the_stages_run_in_a_fixed_order(self, mocker: MockerFixture, tmp_path: Path):
        trainer = build_trainer(mocker, tmp_path)

        start_trainer(trainer, world_size=1)

        assert stage_order(trainer) == [
            "build_model",
            "build_training_components",
            "preflight_checkpoint",
            "create_output_dir",
            "initialize_outputs",
        ]
        # the raw model is built before anything is wrapped around it
        assert trainer.calls.index("init_model") < trainer.calls.index("configure_optimizer")

    def test_the_output_directory_is_created_after_the_checkpoint_preflight(
        self, mocker: MockerFixture, tmp_path: Path
    ):
        trainer = build_trainer(mocker, tmp_path)

        start_trainer(trainer, world_size=1)

        assert stage_order(trainer).index("preflight_checkpoint") < stage_order(trainer).index(
            "create_output_dir"
        )

    def test_a_failing_stage_stops_the_run_before_the_next_one(
        self, mocker: MockerFixture, tmp_path: Path
    ):
        trainer = build_trainer(mocker, tmp_path)
        mocker.patch.object(
            ProbeTrainer, "preflight_checkpoint", side_effect=ValueError("no checkpoint for you")
        )

        with pytest.raises(SystemExit) as exit_info:
            start_trainer(trainer, world_size=1)

        assert exit_info.value.code == 1
        assert "create_output_dir" not in trainer.calls
        assert "initialize_outputs" not in trainer.calls

    def test_a_run_that_fails_before_the_output_stage_leaves_no_directory_behind(
        self, mocker: MockerFixture, tmp_path: Path
    ):
        trainer = build_trainer(mocker, tmp_path)
        mocker.patch.object(
            ProbeTrainer, "preflight_checkpoint", side_effect=ValueError("no checkpoint for you")
        )

        with pytest.raises(SystemExit):
            start_trainer(trainer, world_size=1)

        assert not (tmp_path / "runs").exists()

    def test_micro_batches_that_do_not_fill_whole_accumulation_windows_stop_the_run(
        self, mocker: MockerFixture, tmp_path: Path
    ):
        log = RecordingLogger()
        mocker.patch.object(startup_module, "_log", log)
        trainer = build_trainer(mocker, tmp_path, target_elements=3, gradient_accumulate_every=2)

        with pytest.raises(SystemExit) as exit_info:
            start_trainer(trainer, world_size=1)

        assert exit_info.value.code == 1
        assert log.fatals == [
            "startup stage 'components' failed: ValueError: the run's 3 micro-batches are not a "
            "multiple of gradient_accumulate_every (2): it would end with a pending accumulation "
            "window"
        ]
        assert "create_output_dir" not in trainer.calls
        assert not (tmp_path / "runs").exists()

    def test_micro_batches_fewer_than_one_accumulation_window_stop_the_run(
        self, mocker: MockerFixture, tmp_path: Path
    ):
        log = RecordingLogger()
        mocker.patch.object(startup_module, "_log", log)
        trainer = build_trainer(mocker, tmp_path, target_elements=1, gradient_accumulate_every=2)

        with pytest.raises(SystemExit) as exit_info:
            start_trainer(trainer, world_size=1)

        assert exit_info.value.code == 1
        assert log.fatals == [
            "startup stage 'components' failed: ValueError: the run's 1 micro-batches are fewer "
            "than gradient_accumulate_every (2): no optimizer step would be taken"
        ]
        assert "create_output_dir" not in trainer.calls

    def test_a_run_with_no_micro_batch_at_all_stops_before_it_starts(
        self, mocker: MockerFixture, tmp_path: Path
    ):
        log = RecordingLogger()
        mocker.patch.object(startup_module, "_log", log)
        # nothing to train on: the run has to be refused here rather than started
        # and then end without a single optimizer step
        trainer = build_trainer(mocker, tmp_path, target_elements=0, gradient_accumulate_every=1)

        with pytest.raises(SystemExit) as exit_info:
            start_trainer(trainer, world_size=1)

        assert exit_info.value.code == 1
        assert log.fatals == [
            "startup stage 'components' failed: ValueError: the run's 0 micro-batches are fewer "
            "than gradient_accumulate_every (1): no optimizer step would be taken"
        ]
        assert not (tmp_path / "runs").exists()


class TestNoArtefactBeforeTheOutputStage:
    def test_nothing_exists_while_the_components_are_built(
        self, mocker: MockerFixture, tmp_path: Path
    ):
        trainer = build_trainer(mocker, tmp_path)

        start_trainer(trainer, world_size=1)

        assert trainer.parent_existed_at_components is False
        assert trainer.output_dir_at_components is None
        assert trainer.logger_at_components is bootstrap_log()

    def test_the_run_directory_holds_the_config_and_the_csv_files_only(
        self, mocker: MockerFixture, tmp_path: Path
    ):
        trainer = build_trainer(mocker, tmp_path)

        start_trainer(trainer, world_size=1)

        assert [path.name for path in (tmp_path / "runs").iterdir()] == [trainer.output_dir.name]
        assert sorted(path.name for path in trainer.output_dir.iterdir()) == [
            "config.yaml",
            "loss.csv",
            "timemem.csv",
        ]

    def test_two_runs_of_the_same_config_get_their_own_directory(
        self, mocker: MockerFixture, tmp_path: Path
    ):
        first = build_trainer(mocker, tmp_path)
        second = build_trainer(mocker, tmp_path)

        start_trainer(first, world_size=1)
        start_trainer(second, world_size=1)

        assert first.output_dir != second.output_dir
        assert first.output_dir.is_dir()
        assert second.output_dir.is_dir()


class TestEveryRankJoinsTheStageCollectives:
    def test_the_writer_creates_and_reports_the_run_directory(
        self, mocker: MockerFixture, tmp_path: Path
    ):
        group = SimulatedGroup(mocker, rank=0, world_size=2)
        trainer = build_trainer(mocker, tmp_path, world_size=2, global_rank=0)

        start_trainer(trainer, world_size=2)

        assert group.stages == ["model", "components", "checkpoint", "output_dir", "outputs"]
        assert trainer.output_dir.parent == tmp_path / "runs"
        assert trainer.output_dir.name.startswith("startup_")
        assert trainer.output_dir.is_dir()

    def test_a_non_writer_rank_attaches_to_the_directory_the_writer_created(
        self, mocker: MockerFixture, tmp_path: Path
    ):
        shared = str(tmp_path / "runs" / "startup_shared")
        group = SimulatedGroup(mocker, rank=1, world_size=2)
        group.reports_rank_zero("output_dir", ok=True, value=shared)
        trainer = build_trainer(mocker, tmp_path, world_size=2, global_rank=1)

        start_trainer(trainer, world_size=2)

        assert stage_order(trainer)[-1] == "initialize_outputs"
        assert trainer.output_dir == Path(shared)
        # rank 1 owns nothing: it neither creates the directory nor writes in it
        assert not Path(shared).exists()

    def test_a_checkpoint_state_every_rank_agrees_on_lets_the_run_continue(
        self, mocker: MockerFixture, tmp_path: Path
    ):
        group = SimulatedGroup(mocker, rank=1, world_size=2)
        group.reports_rank_zero("checkpoint", ok=True, state=FRESH)
        group.reports_rank_zero("output_dir", ok=True, value=str(tmp_path / "runs" / "startup_x"))
        trainer = build_trainer(mocker, tmp_path, world_size=2, global_rank=1)

        start_trainer(trainer, world_size=2)

        assert stage_order(trainer)[-1] == "initialize_outputs"

    def test_ranks_that_disagree_about_the_checkpoint_state_stop_the_run(
        self, mocker: MockerFixture, tmp_path: Path
    ):
        log = RecordingLogger()
        mocker.patch.object(startup_module, "_log", log)
        group = SimulatedGroup(mocker, rank=1, world_size=2)
        # rank 0 restored an explicit checkpoint while this rank found none
        group.reports_rank_zero("checkpoint", ok=True, state=RESUME)
        trainer = build_trainer(mocker, tmp_path, world_size=2, global_rank=1)

        with pytest.raises(SystemExit) as exit_info:
            start_trainer(trainer, world_size=2)

        assert exit_info.value.code == 1
        assert log.fatals == [
            "startup stage 'checkpoint' failed: ranks disagree on the state of stage "
            "'checkpoint': ['fresh', 'resume']"
        ]
        assert "create_output_dir" not in trainer.calls

    def test_a_stage_that_failed_on_another_rank_stops_this_rank_too(
        self, mocker: MockerFixture, tmp_path: Path
    ):
        group = SimulatedGroup(mocker, rank=1, world_size=2)
        group.reports_rank_zero("components", ok=False, reason="RuntimeError: no process group")
        trainer = build_trainer(mocker, tmp_path, world_size=2, global_rank=1)

        with pytest.raises(SystemExit):
            start_trainer(trainer, world_size=2)

        assert "preflight_checkpoint" not in trainer.calls
        assert "create_output_dir" not in trainer.calls
