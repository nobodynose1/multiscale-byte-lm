from typing import Any, Callable

import pytest

import mblm.train.core.startup as startup_module
from mblm.train.core.startup import StageOutcome, required_stage, unify_stage_outcome


class RecordingLogger:
    def __init__(self) -> None:
        self.fatals: list[str] = []

    def info(self, _message: str) -> None:
        pass

    def fatal(self, message: str, *_args: Any, **_kwargs: Any) -> None:
        self.fatals.append(message)


def noop(*_args: Any, **_kwargs: Any) -> None:
    """Stand-in for a collective call in the single-rank tests."""


def fake_gather(*states: StageOutcome) -> Callable[[list, StageOutcome], None]:
    def all_gather_object(gathered: list, _local: StageOutcome) -> None:
        for index, state in enumerate(states):
            gathered[index] = state

    return all_gather_object


def test_a_ready_stage_on_a_single_rank_lets_the_run_continue():
    body: list[str] = []

    with required_stage("datasets", world_size=1):
        body.append("ran")

    assert body == ["ran"]


def test_a_ready_stage_creates_no_logger(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(startup_module, "_log", None)
    monkeypatch.setattr(
        startup_module, "create_logger", lambda *_args, **_kwargs: pytest.fail("no logger")
    )

    with required_stage("datasets", world_size=1):
        pass


def test_ready_ranks_unify_to_ready(monkeypatch: pytest.MonkeyPatch):
    local = StageOutcome(stage="datasets", ok=True)
    monkeypatch.setattr(
        startup_module.dist,
        "all_gather_object",
        fake_gather(
            StageOutcome(stage="datasets", ok=True), StageOutcome(stage="datasets", ok=True)
        ),
    )
    monkeypatch.setattr(startup_module.dist, "get_rank", lambda: 0)
    monkeypatch.setattr(startup_module.dist, "broadcast_object_list", noop)

    decision = unify_stage_outcome(local, world_size=2)

    assert decision == StageOutcome(stage="datasets", ok=True)


def test_one_failing_rank_fails_the_whole_stage(monkeypatch: pytest.MonkeyPatch):
    local = StageOutcome(stage="datasets", ok=True)
    monkeypatch.setattr(
        startup_module.dist,
        "all_gather_object",
        fake_gather(
            local,
            StageOutcome(stage="datasets", ok=False, reason="RuntimeError: unreadable corpus"),
        ),
    )
    monkeypatch.setattr(startup_module.dist, "get_rank", lambda: 0)
    monkeypatch.setattr(startup_module.dist, "broadcast_object_list", noop)

    decision = unify_stage_outcome(local, world_size=2)

    assert not decision.ok
    assert decision.reason is not None
    assert "rank 1" in decision.reason
    assert "unreadable corpus" in decision.reason


def test_ranks_that_disagree_on_the_stage_fail_together(monkeypatch: pytest.MonkeyPatch):
    local = StageOutcome(stage="datasets", ok=True)
    monkeypatch.setattr(
        startup_module.dist,
        "all_gather_object",
        fake_gather(local, StageOutcome(stage="components", ok=True)),
    )
    monkeypatch.setattr(startup_module.dist, "get_rank", lambda: 0)
    monkeypatch.setattr(startup_module.dist, "broadcast_object_list", noop)

    decision = unify_stage_outcome(local, world_size=2)

    assert not decision.ok
    assert decision.reason is not None and "datasets" in decision.reason


def test_ranks_that_disagree_on_the_stage_state_fail_together(monkeypatch: pytest.MonkeyPatch):
    local = StageOutcome(stage="checkpoint", ok=True, state="fresh")
    monkeypatch.setattr(
        startup_module.dist,
        "all_gather_object",
        fake_gather(local, StageOutcome(stage="checkpoint", ok=True, state="resume")),
    )
    monkeypatch.setattr(startup_module.dist, "get_rank", lambda: 0)
    monkeypatch.setattr(startup_module.dist, "broadcast_object_list", noop)

    decision = unify_stage_outcome(local, world_size=2)

    assert not decision.ok
    assert decision.reason is not None
    assert "fresh" in decision.reason and "resume" in decision.reason


def test_a_stage_value_written_by_two_ranks_fails_the_stage(monkeypatch: pytest.MonkeyPatch):
    local = StageOutcome(stage="output_dir", ok=True, value="/runs/a")
    monkeypatch.setattr(
        startup_module.dist,
        "all_gather_object",
        fake_gather(local, StageOutcome(stage="output_dir", ok=True, value="/runs/b")),
    )
    monkeypatch.setattr(startup_module.dist, "get_rank", lambda: 0)
    monkeypatch.setattr(startup_module.dist, "broadcast_object_list", noop)

    decision = unify_stage_outcome(local, world_size=2)

    assert not decision.ok
    assert decision.reason is not None
    assert "more than one rank" in decision.reason


def test_the_value_of_one_rank_reaches_every_other_rank(monkeypatch: pytest.MonkeyPatch):
    local = StageOutcome(stage="output_dir", ok=True)
    monkeypatch.setattr(
        startup_module.dist,
        "all_gather_object",
        fake_gather(StageOutcome(stage="output_dir", ok=True, value="/runs/a"), local),
    )
    monkeypatch.setattr(startup_module.dist, "get_rank", lambda: 0)
    monkeypatch.setattr(startup_module.dist, "broadcast_object_list", noop)

    decision = unify_stage_outcome(local, world_size=2)

    assert decision.ok
    assert decision.value == "/runs/a"


def test_a_failure_on_this_rank_is_reported_on_stderr_and_exits(monkeypatch: pytest.MonkeyPatch):
    log = RecordingLogger()
    monkeypatch.setattr(startup_module, "_log", log)

    with pytest.raises(SystemExit) as exit_info:
        with required_stage("datasets", world_size=1):
            raise RuntimeError("the dataset cannot be read")

    assert exit_info.value.code == 1
    assert log.fatals == [
        "startup stage 'datasets' failed: RuntimeError: the dataset cannot be read"
    ]


def test_a_rank_whose_own_stage_was_ready_still_exits(monkeypatch: pytest.MonkeyPatch):
    log = RecordingLogger()
    monkeypatch.setattr(startup_module, "_log", log)
    monkeypatch.setattr(startup_module.dist, "all_gather_object", noop)
    monkeypatch.setattr(startup_module.dist, "get_rank", lambda: 1)

    def broadcast_object_list(decision: list, **_kwargs: Any) -> None:
        decision[0] = StageOutcome(stage="datasets", ok=False, reason="rank 0: RuntimeError: boom")

    monkeypatch.setattr(startup_module.dist, "broadcast_object_list", broadcast_object_list)

    with pytest.raises(SystemExit) as exit_info:
        with required_stage("datasets", world_size=2):
            pass

    assert exit_info.value.code == 1
    assert log.fatals == ["startup stage 'datasets' failed: rank 0: RuntimeError: boom"]


def test_interrupts_are_not_part_of_the_protocol(monkeypatch: pytest.MonkeyPatch):
    log = RecordingLogger()
    monkeypatch.setattr(startup_module, "_log", log)
    monkeypatch.setattr(
        startup_module.dist, "all_gather_object", lambda *_args, **_kwargs: pytest.fail("gathered")
    )

    with pytest.raises(KeyboardInterrupt):
        with required_stage("datasets", world_size=2):
            raise KeyboardInterrupt

    assert log.fatals == []
