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
Module shims for Mamba. Because the official Mamba package relies on a Linux
environment and a few special dependencies that require CUDA, this module
provides a shim to enable development across platforms.

Importing this module never imports and never probes a Mamba backend: a run
binds an engine explicitly through `probe_mamba2` / `probe_mamba1`, which report
the actual backend, its package version and the reason an import failed. The
probe never falls back to another engine - the declared backend either binds or
the run fails.

1. mamba2: the official mamba_ssm package [https://arxiv.org/pdf/2405.21060]
2. mamba1: the pure-PyTorch community implementation [https://arxiv.org/pdf/2312.00752]

"""

import importlib.metadata
import warnings
from functools import partial
from typing import TYPE_CHECKING, Any, NamedTuple, cast

import torch

if TYPE_CHECKING:
    from mamba_ssm.utils.generation import (  # type: ignore[import-not-found]
        InferenceParams,
    )

MAMBA1 = "mamba1"
MAMBA2 = "mamba2"
UNRESOLVED = "unresolved"


class MambaBackendUnavailableError(RuntimeError):
    """
    Raised when a Mamba backend is used before it has been bound to this process.
    """


class MambaProbeResult(NamedTuple):
    """
    Outcome of a backend probe: the bound backend with its package version, or
    the reason why the backend could not be imported.
    """

    backend: str | None
    version: str | None
    reason: str | None


_MAMBA2_BINDINGS: dict[str, Any] | None = None
_MAMBA1_BINDINGS: dict[str, Any] | None = None
_ADMITTED_BACKEND: str | None = None


def _import_mamba2() -> dict[str, Any]:
    with warnings.catch_warnings():
        warnings.simplefilter(action="ignore", category=FutureWarning)
        # mamba-ssm and its triton dependency issue a handful of FutureWarnings,
        # that's none of our business
        from mamba_ssm.models.mixer_seq_simple import (
            _init_weights,  # type: ignore[import-not-found]
            create_block,  # type: ignore[import-not-found]
        )
        from mamba_ssm.ops.triton.layer_norm import (
            RMSNorm,  # type: ignore[import-not-found]
            layer_norm_fn,  # type: ignore[import-not-found]
        )

    return {
        "_init_weights": _init_weights,
        "create_block": create_block,
        "RMSNorm": RMSNorm,
        "layer_norm_fn": layer_norm_fn,
    }


def _import_mamba1() -> dict[str, Any]:
    from mambapy.mamba import Mamba, MambaConfig

    return {"Mamba": Mamba, "MambaConfig": MambaConfig}


def _package_version(distribution: str) -> str | None:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def _reason(error: BaseException) -> str:
    return f"{type(error).__name__}: {error}"


def probe_mamba2() -> MambaProbeResult:
    """
    Import the official mamba_ssm package and bind it for model construction.

    A missing package, a broken install (e.g. an unloadable `.so`) or any other
    import failure is reported through `reason`; this never falls back to
    another engine.
    """
    global _MAMBA2_BINDINGS
    try:
        _MAMBA2_BINDINGS = _import_mamba2()
    except Exception as error:
        _MAMBA2_BINDINGS = None
        return MambaProbeResult(backend=None, version=None, reason=_reason(error))
    return MambaProbeResult(backend=MAMBA2, version=_package_version("mamba-ssm"), reason=None)


def probe_mamba1() -> MambaProbeResult:
    """
    Import the pure-PyTorch mambapy implementation and bind it for model
    construction. Failures are reported through `reason`.
    """
    global _MAMBA1_BINDINGS
    try:
        _MAMBA1_BINDINGS = _import_mamba1()
    except Exception as error:
        _MAMBA1_BINDINGS = None
        return MambaProbeResult(backend=None, version=None, reason=_reason(error))
    return MambaProbeResult(backend=MAMBA1, version=_package_version("mambapy"), reason=None)


def bind(backend: str) -> None:
    """
    Record the engine this process has been admitted to use. Called once the
    admission of a run has settled on a backend.
    """
    global _ADMITTED_BACKEND
    _ADMITTED_BACKEND = backend


def bound_backend() -> str | None:
    """
    The backend this process is admitted to use, `None` before admission.
    """
    return _ADMITTED_BACKEND


def _mamba2_bindings() -> dict[str, Any]:
    if _MAMBA2_BINDINGS is None:
        raise MambaBackendUnavailableError("the mamba2 backend is not bound in this process")
    return _MAMBA2_BINDINGS


def mamba1_bindings() -> tuple[Any, Any]:
    """
    The bound mambapy `Mamba` and `MambaConfig` classes.
    """
    if _MAMBA1_BINDINGS is None:
        raise MambaBackendUnavailableError("the mamba1 backend is not bound in this process")
    return _MAMBA1_BINDINGS["Mamba"], _MAMBA1_BINDINGS["MambaConfig"]


class Mamba2Mixer(torch.nn.Module):
    """
    Simplified and typed version of
    mamba_ssm.models.mixer_seq_simple.MixerModel without the embeddings, our
    wrapper model takes care of these.

    Notation used in Mamba paper:
        d_model: Model dimension [D]
        d_state: SSM state dimension/state size/state expansion factor  [N]
        d_conv: Local convolution width
        expand: Block expansion factor [E]

        Sequence length [L]
    """

    def __init__(
        self,
        d_model: int,
        n_layers: int,
        d_state: int,
        d_conv: int,
        headdim: int,
        expand: int,
        intermediate_factor: int = 0,
        norm_epsilon: float = 1e-5,
        dropout: float = 0.0,
        residual_in_fp32: bool = False,
        fused_add_norm: bool = True,
    ):
        super().__init__()
        bindings = _mamba2_bindings()
        create_block = bindings["create_block"]
        _init_weights = bindings["_init_weights"]
        rms_norm = bindings["RMSNorm"]

        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm
        self.dropout = dropout

        # raw params that are passed to the Mamba 1/2 block
        ssm_cfg = {
            # required according to create_block fn
            "layer": "Mamba2",
            "d_state": d_state,
            "d_conv": d_conv,
            "expand": expand,
            "headdim": headdim,
        }

        self.layers = torch.nn.ModuleList(
            [
                create_block(
                    d_model,
                    d_intermediate=intermediate_factor * d_model,
                    ssm_cfg=ssm_cfg,
                    norm_epsilon=norm_epsilon,
                    rms_norm=True,
                    residual_in_fp32=residual_in_fp32,
                    fused_add_norm=fused_add_norm,
                    layer_idx=i,
                )
                for i in range(n_layers)
            ]
        )
        self.norm_f = rms_norm(d_model, eps=norm_epsilon, dropout_p=dropout)
        self.apply(
            partial(
                _init_weights,
                n_layer=n_layers,
                initializer_range=0.02,  # Now only used for embedding layer.
                rescale_prenorm_residual=True,
                n_residuals_per_layer=1 if intermediate_factor == 0 else 2,  # 2 if we have MLP
            )
        )

    def allocate_inference_cache(self, batch_size: int, max_seqlen: int, dtype=None, **kwargs):
        return {
            i: layer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)
            for i, layer in enumerate(self.layers)
        }

    def forward(
        self,
        input_ids: torch.Tensor,
        inference_params: "InferenceParams | None" = None,
        **mixer_kwargs,
    ):
        layer_norm_fn = _mamba2_bindings()["layer_norm_fn"]
        rms_norm = _mamba2_bindings()["RMSNorm"]

        hidden_states: torch.Tensor = input_ids
        residual: torch.Tensor | None = None
        for layer in self.layers:
            hidden_states, residual = layer(
                hidden_states, residual, inference_params=inference_params, **mixer_kwargs
            )
        if not self.fused_add_norm:
            residual = (hidden_states + residual) if residual is not None else hidden_states
            hidden_states = self.norm_f(residual.to(dtype=self.norm_f.weight.dtype))
        else:
            # Set prenorm=False here since we don't need the residual
            hidden_states = cast(
                torch.Tensor,
                layer_norm_fn(
                    hidden_states,
                    self.norm_f.weight,
                    self.norm_f.bias,
                    eps=self.norm_f.eps,
                    residual=residual,
                    prenorm=False,
                    dropout_p=self.dropout,
                    residual_in_fp32=self.residual_in_fp32,
                    is_rms_norm=isinstance(self.norm_f, rms_norm),
                ),
            )
        return hidden_states


__all__ = [
    "MAMBA1",
    "MAMBA2",
    "UNRESOLVED",
    "Mamba2Mixer",
    "MambaBackendUnavailableError",
    "MambaProbeResult",
    "bind",
    "bound_backend",
    "mamba1_bindings",
    "probe_mamba1",
    "probe_mamba2",
]
