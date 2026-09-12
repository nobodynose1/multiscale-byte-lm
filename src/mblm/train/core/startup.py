__copyright__ = """MIT License

Copyright (c) 2024 - IBM Research

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE."""

"""
Fixed-shape outcomes for the startup stages of a training entry.

The stages that run before a run owns any output artefact (datasets, model
components, checkpoint preflight) must not let a single rank run ahead: an
ordinary exception on one rank would otherwise leave the other ranks waiting
inside the next collective. Every rank therefore serializes its stage result,
the ranks unify the results in the stage's collective, and only a stage that is
ready on every rank lets the run continue. A stage that fails anywhere is
reported on stderr and exits every rank non-zero before any run artefact is
written.

The same fixed-shape unification carries the evaluation passes and the choice
of the state a run delivers for testing, so that a rank never records a result
the others did not reach, and never waits inside a collective the others left.
"""

import logging
import sys
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

import torch.distributed as dist
from pydantic import BaseModel

from mblm.utils.logging import create_logger

if TYPE_CHECKING:
    from mblm.train.core.trainer import CoreTrainer

# The startup stages, in the order a run executes them. The dataset stages are
# driven by the training entries, the rest by `start_trainer`.
STAGE_DATASETS = "datasets"
STAGE_MODEL = "model"
STAGE_COMPONENTS = "components"
STAGE_CHECKPOINT = "checkpoint"
STAGE_OUTPUT_DIR = "output_dir"
STAGE_OUTPUTS = "outputs"
STAGE_TEST_DATASETS = "test_datasets"
STAGE_TEST_MODEL = "test_model"
STAGE_CHECKPOINT_SAVE = "checkpoint_save"
STAGE_DELIVERY = "delivery"

# The evaluation passes a run unifies across ranks, named as the audit source
# their loss rows carry.
EVAL_SCHEDULED = "scheduled_validation"
EVAL_RESUME = "resume_evaluation"
EVAL_TEST = "test"

# The states an evaluation outcome can carry: every batch was evaluated, some
# batches were skipped, or the pass produced no usable result at all.
EVAL_COMPLETE = "complete"
EVAL_INCOMPLETE = "incomplete"
EVAL_ERROR = "error"

_log: logging.Logger | None = None


class StageOutcome(BaseModel):
    """
    Serialized outcome of one startup stage on a single rank, and of the run
    once unified.

    `state` is the stage's chosen state, e.g. whether a checkpoint preflight
    took the `fresh` or the `resume` path; every rank must reach the same one.
    `value` is the stage's payload, which exactly one rank - the writer that
    owns it - produces for all the others.
    """

    stage: str
    ok: bool
    reason: str | None = None
    state: str | None = None
    value: str | None = None


class StageReport(BaseModel):
    """
    The handle a stage body fills in: it records what this rank decided about
    the stage, and after the collective it carries the run's unified decision.
    """

    stage: str
    state: str | None = None
    value: str | None = None


def bootstrap_log() -> logging.Logger:
    """
    Logger for the stages that run before the run has an output directory:
    stderr only, no file handler, no directory creation.
    """
    global _log
    if _log is None:
        _log = create_logger(__name__)
    return _log


def _aggregate(outcomes: Sequence[StageOutcome | None]) -> StageOutcome:
    """
    Unify the per-rank outcomes of one stage into the run's decision.
    """
    ranked = [(rank, outcome) for rank, outcome in enumerate(outcomes) if outcome is not None]
    first = ranked[0][1]

    stages = {outcome.stage for _, outcome in ranked}
    if len(stages) > 1:
        return StageOutcome(
            stage=first.stage,
            ok=False,
            reason=f"ranks disagree on the startup stage: {sorted(stages)}",
        )

    failed = [(rank, outcome) for rank, outcome in ranked if not outcome.ok]
    if failed:
        reasons = "; ".join(
            f"rank {rank}: {outcome.reason or 'the stage failed'}" for rank, outcome in failed
        )
        return StageOutcome(stage=first.stage, ok=False, reason=reasons)

    states = {outcome.state for _, outcome in ranked}
    if len(states) > 1:
        return StageOutcome(
            stage=first.stage,
            ok=False,
            reason=f"ranks disagree on the state of stage '{first.stage}': "
            f"{sorted(str(state) for state in states)}",
        )

    written = [(rank, outcome.value) for rank, outcome in ranked if outcome.value is not None]
    if len(written) > 1:
        return StageOutcome(
            stage=first.stage,
            ok=False,
            reason=f"stage '{first.stage}' was produced by more than one rank: "
            f"{sorted(rank for rank, _ in written)}",
        )

    return StageOutcome(
        stage=first.stage,
        ok=True,
        state=first.state,
        value=written[0][1] if written else None,
    )


def unify_stage_outcome(local: StageOutcome, *, world_size: int) -> StageOutcome:
    """
    Unify one stage's per-rank outcome over the process group. Every rank leaves
    with the same decision, so a stage that failed on one rank is not entered by
    another rank, and no rank waits inside the next collective.
    """
    if world_size == 1:
        return local

    gathered: list[StageOutcome | None] = [None] * world_size
    dist.all_gather_object(gathered, local)

    decision: list[StageOutcome | None] = [_aggregate(gathered) if dist.get_rank() == 0 else None]
    dist.broadcast_object_list(decision, src=0)

    assert decision[0] is not None
    return decision[0]


class EvalOutcome(BaseModel):
    """
    Serialized outcome of one evaluation pass on a single rank, and of the pass
    once unified.

    `state` is `complete` when every batch of the pass was evaluated and
    `incomplete` when some batches were skipped; both carry the pass' loss as
    the sum over the batches that succeeded, which the ranks reduce into the one
    value the run keeps. `error` carries a reason instead and no loss at all, so
    a failed pass can never be recorded as a metric.
    """

    stage: str
    state: str
    reason: str | None = None
    loss_sum: float = 0.0
    ok_batches: int = 0
    skipped_batches: int = 0

    @property
    def loss(self) -> float | None:
        """
        The pass' mean loss over the batches that succeeded, or `None` when no
        batch of the pass succeeded.
        """
        if self.ok_batches == 0:
            return None
        return self.loss_sum / self.ok_batches


def _aggregate_eval(outcomes: Sequence[EvalOutcome | None]) -> EvalOutcome:
    """
    Unify the per-rank outcomes of one evaluation pass into the run's result.
    """
    ranked = [(rank, outcome) for rank, outcome in enumerate(outcomes) if outcome is not None]
    first = ranked[0][1]

    stages = {outcome.stage for _, outcome in ranked}
    if len(stages) > 1:
        return EvalOutcome(
            stage=first.stage,
            state=EVAL_ERROR,
            reason=f"ranks disagree on the evaluation: {sorted(stages)}",
        )

    failed = [(rank, outcome) for rank, outcome in ranked if outcome.state == EVAL_ERROR]
    if failed:
        reasons = "; ".join(
            f"rank {rank}: {outcome.reason or 'the evaluation failed'}" for rank, outcome in failed
        )
        return EvalOutcome(stage=first.stage, state=EVAL_ERROR, reason=reasons)

    states = {outcome.state for _, outcome in ranked}
    if len(states) > 1:
        return EvalOutcome(
            stage=first.stage,
            state=EVAL_ERROR,
            reason=f"ranks disagree on the state of evaluation '{first.stage}': "
            f"{sorted(str(state) for state in states)}",
        )

    counts = {(outcome.ok_batches, outcome.skipped_batches) for _, outcome in ranked}
    if len(counts) > 1:
        return EvalOutcome(
            stage=first.stage,
            state=EVAL_ERROR,
            reason=f"ranks disagree on the batches evaluation '{first.stage}' processed and "
            f"skipped: {sorted(counts)}",
        )

    return EvalOutcome(
        stage=first.stage,
        state=first.state,
        loss_sum=sum(outcome.loss_sum for _, outcome in ranked),
        ok_batches=sum(outcome.ok_batches for _, outcome in ranked),
        skipped_batches=sum(outcome.skipped_batches for _, outcome in ranked),
    )


def unify_eval_outcome(local: EvalOutcome, *, world_size: int) -> EvalOutcome:
    """
    Unify one evaluation pass over the process group.

    Every rank leaves with the same result: the same state, the same counts of
    evaluated and skipped batches, and the same loss, formed from the reduced
    sums rather than from any one rank's average. A pass that failed on any
    rank, or on which the ranks disagree, is reported on stderr and exits every
    rank non-zero together: a rank that evaluated nothing must not record a
    result the others did not reach.
    """
    if world_size == 1:
        decision = _aggregate_eval([local])
    else:
        gathered: list[EvalOutcome | None] = [None] * world_size
        dist.all_gather_object(gathered, local)

        decisions: list[EvalOutcome | None] = [
            _aggregate_eval(gathered) if dist.get_rank() == 0 else None
        ]
        dist.broadcast_object_list(decisions, src=0)

        assert decisions[0] is not None
        decision = decisions[0]

    if decision.state == EVAL_ERROR:
        bootstrap_log().fatal(f"evaluation '{decision.stage}' failed: {decision.reason}")
        sys.exit(1)
    return decision


@contextmanager
def required_stage(stage: str, *, world_size: int) -> Iterator[StageReport]:
    """
    Run one startup stage whose result must be ready on every rank before the
    run continues.

    An ordinary exception raised by the body is turned into this rank's `error`
    outcome; all ranks then join the stage's collective and, unless every rank
    reported ready, the stage is reported on stderr and all ranks exit non-zero
    together. Interrupt signals (`BaseException`) are not part of this protocol
    and propagate unchanged.

    The body records what this rank decided about the stage on the report it is
    handed; after the collective that report carries the run's unified decision.
    """
    report = StageReport(stage=stage)
    outcome = StageOutcome(stage=stage, ok=True)
    try:
        yield report
        outcome = StageOutcome(stage=stage, ok=True, state=report.state, value=report.value)
    except Exception as error:
        outcome = StageOutcome(
            stage=stage,
            ok=False,
            reason=f"{type(error).__name__}: {error}",
            state=report.state,
        )

    decision = unify_stage_outcome(outcome, world_size=world_size)
    if not decision.ok:
        bootstrap_log().fatal(f"startup stage '{decision.stage}' failed: {decision.reason}")
        sys.exit(1)
    # a unified ready outcome means every rank was ready, including this one
    assert outcome.ok
    report.state = decision.state
    report.value = decision.value


def start_trainer(trainer: "CoreTrainer[Any, Any, Any, Any, Any]", *, world_size: int) -> None:
    """
    Drive a trainer through the startup stages it must complete before it owns
    anything.

    The order is fixed: the model is built and unified on every rank before it
    is wrapped for distributed training, the training components are built and
    unified before any checkpoint is read into them, the checkpoint preflight
    decides `fresh` or `resume` on every rank, and only then does the global
    rank 0 writer create the run's output directory and every rank attach to it.
    """
    with required_stage(STAGE_MODEL, world_size=world_size):
        trainer.build_model()

    with required_stage(STAGE_COMPONENTS, world_size=world_size):
        trainer.build_training_components()

    with required_stage(STAGE_CHECKPOINT, world_size=world_size) as checkpoint:
        checkpoint.state = trainer.preflight_checkpoint()
    bootstrap_log().info(f"Checkpoint preflight: {checkpoint.state}")

    with required_stage(STAGE_OUTPUT_DIR, world_size=world_size) as output:
        output.value = trainer.create_output_dir()
    assert output.value is not None, "the output directory stage produced no path"

    with required_stage(STAGE_OUTPUTS, world_size=world_size):
        trainer.initialize_outputs(output.value)


__all__ = [
    "EVAL_COMPLETE",
    "EVAL_ERROR",
    "EVAL_INCOMPLETE",
    "EVAL_RESUME",
    "EVAL_SCHEDULED",
    "EVAL_TEST",
    "STAGE_CHECKPOINT",
    "STAGE_CHECKPOINT_SAVE",
    "STAGE_COMPONENTS",
    "STAGE_DATASETS",
    "STAGE_DELIVERY",
    "STAGE_MODEL",
    "STAGE_OUTPUT_DIR",
    "STAGE_OUTPUTS",
    "STAGE_TEST_DATASETS",
    "STAGE_TEST_MODEL",
    "EvalOutcome",
    "StageOutcome",
    "StageReport",
    "bootstrap_log",
    "required_stage",
    "start_trainer",
    "unify_eval_outcome",
    "unify_stage_outcome",
]
