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


from typing import Literal

from pydantic import Field, model_validator

from mblm.model import mamba_shim
from mblm.model.block import StageBlock


class MambaBlock(StageBlock):
    """
    General config for creating a Mamba block inside MBLM.
    Uses roughly 3 * expand * d_model^2 parameters.

    Parameters in brackets [x] denote the notation used in Mambabyte

    Parameters:
        n_layers: Number of layers [n]
        d_model: Residual stream dimension [d]
        expand: Linear layers expansion factor (always 2) [e]
        d_state: SSM state dimension [n_state]
        d_conv: Convolutional kernel size (always 4) [k]
        headdim: Head dimension
        dropout: The dropout in the Mamba block
        mamba_backend: The Mamba engine this block is built with. Declared per
            run and checked against the engine the admission bound; an
            unavailable declared engine fails the run instead of silently
            falling back to another one
    Note:
        `headdim` is a Mamba2 parameter only. There is no `dt_rank` [r]
        low-rank projection dimension in Mamba2
    """

    d_state: int
    d_conv: int
    expand: int
    headdim: int  # Mamba2 only
    dropout: float = 0.0  # Mamba2 only. Default for backwards compatibility
    mamba_backend: Literal["mamba1", "mamba2"] = Field(
        description="The Mamba engine to build this block with, e.g. 'mamba2'"
    )

    block_type: str = Field(init=False, default=mamba_shim.UNRESOLVED)

    def to_model(self, model_dim, num_layers):
        backend = mamba_shim.bound_backend()
        if backend is None:
            raise mamba_shim.MambaBackendUnavailableError(
                f"Mamba block declares '{self.mamba_backend}' but no Mamba engine was admitted "
                "in this process; run the Mamba admission before building a model"
            )
        if backend != self.mamba_backend:
            raise mamba_shim.MambaBackendUnavailableError(
                f"Mamba block declares '{self.mamba_backend}' but '{backend}' is the admitted engine"
            )

        # the runtime engine is the one the admission bound, not an import-time guess
        self.block_type = backend

        if backend == mamba_shim.MAMBA2:
            return mamba_shim.Mamba2Mixer(
                d_model=model_dim,
                n_layers=num_layers,
                d_state=self.d_state,
                d_conv=self.d_conv,
                headdim=self.headdim,
                expand=self.expand,
                dropout=self.dropout,
            )

        mamba1, mamba1_config = mamba_shim.mamba1_bindings()
        return mamba1(
            mamba1_config(
                d_model=model_dim,
                n_layers=num_layers,
                d_state=self.d_state,
                d_conv=self.d_conv,
                expand_factor=self.expand,
            )
        )

    @model_validator(mode="after")
    def validate_block_type(self):
        if self.block_type not in (mamba_shim.UNRESOLVED, mamba_shim.MAMBA1, mamba_shim.MAMBA2):
            raise ValueError("This model is a mamba block")
        return self
