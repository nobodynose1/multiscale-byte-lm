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
Admission of the Mamba engine a run declares.

The parsed configuration decides which engine a Mamba stage block must be built
with (`mamba_backend`). Only once the process group exists does this module
probe that engine for real, so that:

1. importing the configuration never imports or probes a Mamba backend, and
2. a run whose declared engine cannot be bound fails on all ranks instead of
   silently being built with the other engine.

The probe state of every rank is serialized and gathered into a single outcome
before anything is built; on failure all ranks exit non-zero together. The
outcome is audited in the run's `mamba_impl.txt`.
"""

import logging
import os
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch.distributed as dist
from pydantic import BaseModel

from mblm.model import mamba_shim
from mblm.model.mamba import MambaBlock
from mblm.utils.logging import create_logger

MARKER_FILE_NAME = "mamba_impl.txt"

_log: logging.Logger | None = None


class MambaAdmissionError(RuntimeError):
    """
    Raised when a run's Mamba declarations cannot be resolved consistently.
    """


class MambaAdmissionState(BaseModel):
    """
    Serialized admission state of a single rank, and of the run once unified.
    """

    required: bool
    ok: bool
    declared: str | None = None
    backend: str | None = None
    version: str | None = None
    reason: str | None = None


def _bootstrap_log() -> logging.Logger:
    """
    Logger for the admission stage: stderr only, no file handlers, safe to use
    before any run artefact exists.
    """
    global _log
    if _log is None:
        _log = create_logger(__name__)
    return _log


def _stage_blocks(params: Any) -> tuple[Any, ...]:
    blocks = getattr(params, "block", None)
    if blocks is None:
        nested = getattr(params, "mblm_config", None)
        blocks = getattr(nested, "block", None) if nested is not None else None
    if blocks is None:
        return ()
    if isinstance(blocks, (list, tuple)):
        return tuple(blocks)
    return (blocks,)


def declared_mamba_backend(params: Any) -> str | None:
    """
    The Mamba engine the parsed model parameters declare, or `None` when no
    stage block uses Mamba. Stages that declare different engines cannot be
    served by a single admission and raise `MambaAdmissionError`.
    """
    declared = {
        block.mamba_backend for block in _stage_blocks(params) if isinstance(block, MambaBlock)
    }
    if len(declared) > 1:
        raise MambaAdmissionError(f"the run declares conflicting Mamba engines: {sorted(declared)}")
    return next(iter(declared), None)


def probe_mamba_backend(params: Any) -> MambaAdmissionState:
    """
    Probe the engine the parameters declare on this rank. Performs the actual
    import; no communication and no process exit, so the outcome can be
    gathered.
    """
    try:
        declared = declared_mamba_backend(params)
    except MambaAdmissionError as error:
        return MambaAdmissionState(required=True, ok=False, reason=str(error))

    if declared is None:
        # no Mamba stage: the Mamba backends are never imported
        return MambaAdmissionState(required=False, ok=True)

    probe = (
        mamba_shim.probe_mamba2() if declared == mamba_shim.MAMBA2 else mamba_shim.probe_mamba1()
    )
    return MambaAdmissionState(
        required=True,
        ok=probe.backend == declared,
        declared=declared,
        backend=probe.backend,
        version=probe.version,
        reason=probe.reason,
    )


def _aggregate(states: Sequence[MambaAdmissionState | None]) -> MambaAdmissionState:
    """
    Unify the probe states of all ranks into the run's outcome.
    """
    ranked = [(rank, state) for rank, state in enumerate(states) if state is not None]
    first = ranked[0][1]
    assert first is not None

    if len({state.required for _, state in ranked}) > 1:
        return MambaAdmissionState(
            required=True, ok=False, reason="ranks disagree on whether the run requires Mamba"
        )

    failed = [(rank, state) for rank, state in ranked if not state.ok]
    if failed:
        reasons = "; ".join(
            f"rank {rank}: {state.reason or 'probe failed'}" for rank, state in failed
        )
        return MambaAdmissionState(
            required=first.required,
            ok=False,
            declared=first.declared,
            reason=f"declared Mamba engine '{first.declared}' is not available: {reasons}",
        )

    if not first.required:
        return first

    backends = {state.backend for _, state in ranked}
    if len(backends) > 1:
        return MambaAdmissionState(
            required=True,
            ok=False,
            declared=first.declared,
            reason=f"ranks bound different Mamba engines: {sorted(backends, key=str)}",
        )

    versions = {state.version for _, state in ranked}
    if len(versions) > 1:
        return MambaAdmissionState(
            required=True,
            ok=False,
            declared=first.declared,
            backend=first.backend,
            reason=f"ranks bound different Mamba package versions: {sorted(versions, key=str)}",
        )

    return first


def _collective_decision(local: MambaAdmissionState, world_size: int) -> MambaAdmissionState:
    if world_size == 1:
        return local

    gathered: list[MambaAdmissionState | None] = [None] * world_size
    dist.all_gather_object(gathered, local)

    decision: list[MambaAdmissionState | None] = [
        _aggregate(gathered) if dist.get_rank() == 0 else None
    ]
    dist.broadcast_object_list(decision, src=0)

    assert decision[0] is not None
    return decision[0]


def run_mamba_admission(params: Any, *, run_vars: Any) -> MambaAdmissionState:
    """
    Admit the Mamba engine of a run: probe on every rank, unify the outcome over
    the process group and bind the admitted engine for model construction.

    An outcome that requires Mamba but cannot bind the declared engine is
    reported on stderr and exits this process non-zero on every rank.
    """
    local = probe_mamba_backend(params)
    decision = _collective_decision(local, run_vars.world_size)

    if not decision.ok:
        _bootstrap_log().fatal(f"Mamba admission failed: {decision.reason}")
        sys.exit(1)

    if decision.required:
        assert decision.backend is not None
        mamba_shim.bind(decision.backend)
        _bootstrap_log().info(
            f"Mamba admission: declared '{decision.declared}', bound '{decision.backend}'"
            f", package version {decision.version}"
        )
    else:
        _bootstrap_log().info("Mamba admission: not applicable, no stage block uses Mamba")

    return decision


def _marker_text(state: MambaAdmissionState) -> str:
    return (
        "\n".join(
            [
                "mblm mamba engine audit",
                f"declared_backend: {state.declared}",
                f"actual_backend: {state.backend}",
                f"package_version: {state.version}",
                f"failure_reason: {state.reason}",
            ]
        )
        + "\n"
    )


def write_mamba_impl_marker(output_dir: str | Path, state: MambaAdmissionState) -> Path | None:
    """
    Atomically write the run's Mamba engine audit. Called by the global rank 0
    writer only, once the admission and the run's output directory both exist.
    """
    if not state.required:
        return None

    path = Path(output_dir) / MARKER_FILE_NAME
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        tmp_path.write_text(_marker_text(state), encoding="utf-8")
        tmp_path.replace(path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()
    return path


__all__ = [
    "MARKER_FILE_NAME",
    "MambaAdmissionError",
    "MambaAdmissionState",
    "declared_mamba_backend",
    "probe_mamba_backend",
    "run_mamba_admission",
    "write_mamba_impl_marker",
]
