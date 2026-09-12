"""One evaluation pass is unified over the ranks: every rank leaves it with the
same state, the same counts of processed and skipped batches and the same loss,
and a pass that failed or that the ranks disagree on exits all of them rather
than letting one rank record a result the others did not reach."""

from typing import Any

import pytest

import mblm.train.core.startup as startup_module
from mblm.train.core.startup import (
    EVAL_COMPLETE,
    EVAL_ERROR,
    EVAL_INCOMPLETE,
    EVAL_SCHEDULED,
    EvalOutcome,
    unify_eval_outcome,
)


class RecordingLogger:
    def __init__(self) -> None:
        self.fatals: list[str] = []

    def info(self, _message: str) -> None:
        pass

    def fatal(self, message: str, *_args: Any, **_kwargs: Any) -> None:
        self.fatals.append(message)


def noop(*_args: Any, **_kwargs: Any) -> None:
    """Stand-in for a collective call in the single-rank tests."""


def fake_gather(*outcomes: EvalOutcome):
    def all_gather_object(gathered: list, _local: EvalOutcome) -> None:
        for index, outcome in enumerate(outcomes):
            gathered[index] = outcome

    return all_gather_object


def unify_over_ranks(
    monkeypatch: pytest.MonkeyPatch, local: EvalOutcome, *others: EvalOutcome
) -> EvalOutcome:
    """Unify a pass over the ranks the test supplies, without a process group."""
    monkeypatch.setattr(startup_module.dist, "all_gather_object", fake_gather(local, *others))
    monkeypatch.setattr(startup_module.dist, "get_rank", lambda: 0)
    monkeypatch.setattr(startup_module.dist, "broadcast_object_list", noop)
    return unify_eval_outcome(local, world_size=1 + len(others))


def test_a_pass_the_ranks_completed_unifies_to_one_loss(monkeypatch: pytest.MonkeyPatch):
    local = EvalOutcome(stage=EVAL_SCHEDULED, state=EVAL_COMPLETE, loss_sum=4.0, ok_batches=2)

    decision = unify_over_ranks(
        monkeypatch,
        local,
        EvalOutcome(stage=EVAL_SCHEDULED, state=EVAL_COMPLETE, loss_sum=8.0, ok_batches=2),
    )

    # the loss comes from the reduced sums, not from either rank's own average
    assert decision.state == EVAL_COMPLETE
    assert decision.loss_sum == 12.0
    assert decision.ok_batches == 4
    assert decision.loss == 3.0


def test_a_pass_with_skipped_batches_carries_its_skips(monkeypatch: pytest.MonkeyPatch):
    local = EvalOutcome(
        stage=EVAL_SCHEDULED,
        state=EVAL_INCOMPLETE,
        loss_sum=3.0,
        ok_batches=2,
        skipped_batches=1,
    )

    decision = unify_over_ranks(
        monkeypatch,
        local,
        EvalOutcome(
            stage=EVAL_SCHEDULED,
            state=EVAL_INCOMPLETE,
            loss_sum=3.0,
            ok_batches=2,
            skipped_batches=1,
        ),
    )

    # the denominator is the batches that succeeded, on every rank at once
    assert decision.state == EVAL_INCOMPLETE
    assert decision.ok_batches == 4
    assert decision.skipped_batches == 2
    assert decision.loss == 1.5


def test_a_single_rank_pass_is_its_own_decision(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(
        startup_module.dist, "all_gather_object", lambda *_args, **_kwargs: pytest.fail("gathered")
    )
    local = EvalOutcome(stage=EVAL_SCHEDULED, state=EVAL_COMPLETE, loss_sum=2.0, ok_batches=2)

    decision = unify_eval_outcome(local, world_size=1)

    assert decision == local
    assert decision.loss == 1.0


def test_a_failed_pass_carries_no_loss():
    outcome = EvalOutcome(stage=EVAL_SCHEDULED, state=EVAL_ERROR, reason="RuntimeError: boom")

    assert outcome.loss is None


def test_a_pass_that_failed_on_one_rank_exits_every_rank(
    monkeypatch: pytest.MonkeyPatch,
):
    log = RecordingLogger()
    monkeypatch.setattr(startup_module, "_log", log)
    local = EvalOutcome(stage=EVAL_SCHEDULED, state=EVAL_COMPLETE, loss_sum=1.0, ok_batches=1)

    with pytest.raises(SystemExit) as exit_info:
        unify_over_ranks(
            monkeypatch,
            local,
            EvalOutcome(stage=EVAL_SCHEDULED, state=EVAL_ERROR, reason="FloatingPointError: bad"),
        )

    assert exit_info.value.code == 1
    assert log.fatals == [
        "evaluation 'scheduled_validation' failed: rank 1: FloatingPointError: bad"
    ]


def test_ranks_that_disagree_on_the_processed_batches_exit_together(
    monkeypatch: pytest.MonkeyPatch,
):
    log = RecordingLogger()
    monkeypatch.setattr(startup_module, "_log", log)
    local = EvalOutcome(stage=EVAL_SCHEDULED, state=EVAL_COMPLETE, loss_sum=1.0, ok_batches=1)

    with pytest.raises(SystemExit) as exit_info:
        unify_over_ranks(
            monkeypatch,
            local,
            EvalOutcome(stage=EVAL_SCHEDULED, state=EVAL_COMPLETE, loss_sum=1.0, ok_batches=2),
        )

    assert exit_info.value.code == 1
    assert log.fatals == [
        "evaluation 'scheduled_validation' failed: ranks disagree on the batches evaluation "
        "'scheduled_validation' processed and skipped: [(1, 0), (2, 0)]"
    ]


def test_ranks_that_disagree_on_the_state_exit_together(monkeypatch: pytest.MonkeyPatch):
    log = RecordingLogger()
    monkeypatch.setattr(startup_module, "_log", log)
    local = EvalOutcome(stage=EVAL_SCHEDULED, state=EVAL_COMPLETE, loss_sum=1.0, ok_batches=1)

    with pytest.raises(SystemExit) as exit_info:
        unify_over_ranks(
            monkeypatch,
            local,
            EvalOutcome(
                stage=EVAL_SCHEDULED,
                state=EVAL_INCOMPLETE,
                loss_sum=1.0,
                ok_batches=1,
                skipped_batches=1,
            ),
        )

    assert exit_info.value.code == 1
    assert log.fatals == [
        "evaluation 'scheduled_validation' failed: ranks disagree on the state of evaluation "
        "'scheduled_validation': ['complete', 'incomplete']"
    ]


def test_ranks_that_disagree_on_the_pass_exit_together(monkeypatch: pytest.MonkeyPatch):
    log = RecordingLogger()
    monkeypatch.setattr(startup_module, "_log", log)
    local = EvalOutcome(stage=EVAL_SCHEDULED, state=EVAL_COMPLETE, loss_sum=1.0, ok_batches=1)

    with pytest.raises(SystemExit) as exit_info:
        unify_over_ranks(
            monkeypatch,
            local,
            EvalOutcome(stage="test", state=EVAL_COMPLETE, loss_sum=1.0, ok_batches=1),
        )

    assert exit_info.value.code == 1
    assert log.fatals == [
        "evaluation 'scheduled_validation' failed: ranks disagree on the evaluation: "
        "['scheduled_validation', 'test']"
    ]
