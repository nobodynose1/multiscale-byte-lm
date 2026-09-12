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
"""

import logging
import sys
from collections.abc import Iterator, Sequence
from contextlib import contextmanager

import torch.distributed as dist
from pydantic import BaseModel

from mblm.utils.logging import create_logger

_log: logging.Logger | None = None


class StageOutcome(BaseModel):
    """
    Serialized outcome of one startup stage on a single rank, and of the run
    once unified.
    """

    stage: str
    ok: bool
    reason: str | None = None


def _bootstrap_log() -> logging.Logger:
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

    return first


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


@contextmanager
def required_stage(stage: str, *, world_size: int) -> Iterator[None]:
    """
    Run one startup stage whose result must be ready on every rank before the
    run continues.

    An ordinary exception raised by the body is turned into this rank's `error`
    outcome; all ranks then join the stage's collective and, unless every rank
    reported ready, the stage is reported on stderr and all ranks exit non-zero
    together. Interrupt signals (`BaseException`) are not part of this protocol
    and propagate unchanged.
    """
    outcome = StageOutcome(stage=stage, ok=True)
    try:
        yield
    except Exception as error:
        outcome = StageOutcome(stage=stage, ok=False, reason=f"{type(error).__name__}: {error}")

    decision = unify_stage_outcome(outcome, world_size=world_size)
    if not decision.ok:
        _bootstrap_log().fatal(f"startup stage '{decision.stage}' failed: {decision.reason}")
        sys.exit(1)
    # a unified ready outcome means every rank was ready, including this one
    assert outcome.ok


__all__ = ["StageOutcome", "required_stage", "unify_stage_outcome"]
