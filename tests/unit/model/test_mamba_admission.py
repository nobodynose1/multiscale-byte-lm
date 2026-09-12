import subprocess
import sys
import textwrap
from typing import Any

import pytest
import torch
from pydantic import ValidationError

from mblm.model import mamba_admission, mamba_shim
from mblm.model.config import MBLMModelConfig
from mblm.model.mamba import MambaBlock
from mblm.model.mamba_admission import (
    MARKER_FILE_NAME,
    MambaAdmissionError,
    MambaAdmissionState,
    declared_mamba_backend,
    probe_mamba_backend,
    run_mamba_admission,
    write_mamba_impl_marker,
)
from mblm.model.mamba_shim import MambaBackendUnavailableError
from mblm.model.transformer import TransformerBlock
from mblm.utils.distributed import ElasticRunVars


def noop(*_args: Any, **_kwargs: Any) -> None:
    """Stand-in for a collective call in the single-rank tests."""


def mamba_block(backend: Any = "mamba1") -> MambaBlock:
    return MambaBlock(
        d_state=128,
        d_conv=4,
        expand=2,
        headdim=64,
        mamba_backend=backend,
        pos_emb_type=None,
    )


def transformer_block() -> TransformerBlock:
    return TransformerBlock(
        attn_head_dims=64,
        attn_num_heads=16,
        attn_use_rot_embs=True,
        use_flash_attn=True,
        pos_emb_type="fixed",
    )


def model_params(block: Any) -> MBLMModelConfig:
    num_stages = len(block) if isinstance(block, list) else 1
    return MBLMModelConfig(
        num_tokens=257,
        hidden_dims=[64] * num_stages,
        seq_lens=[8] * num_stages,
        num_layers=[1] * num_stages,
        pad_token_id=256,
        train_checkpoint_chunks=None,
        block=block,
    )


def single_process() -> ElasticRunVars:
    return ElasticRunVars(local_rank=0, world_size=1, is_cuda=False)


def test_backends_are_not_imported_when_the_configuration_is_imported():
    code = textwrap.dedent(
        """
        import sys

        import mblm
        from mblm.model import mamba_admission, mamba_shim
        from mblm.model.config import MBLMModelConfig
        from mblm.model.mamba import MambaBlock

        for package in ("mamba_ssm", "mambapy"):
            assert package not in sys.modules, f"{package} must not be imported at import time"
        """
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("backend", ["mamba1", "mamba2"])
def test_declared_engines_parse_to_a_mamba_block(backend: str):
    parsed = model_params(
        {
            "block_type": mamba_shim.UNRESOLVED,
            "d_state": 128,
            "d_conv": 4,
            "expand": 2,
            "headdim": 64,
            "mamba_backend": backend,
        }
    )
    assert isinstance(parsed.block, MambaBlock)
    assert parsed.block.mamba_backend == backend
    # the runtime engine is only known once an engine has been admitted
    assert parsed.block.block_type == mamba_shim.UNRESOLVED


@pytest.mark.parametrize("backend", ["auto", None, "mamba3", "mamba"])
def test_unsupported_engine_declarations_are_rejected(backend: Any):
    with pytest.raises(ValidationError) as error:
        mamba_block(backend)
    assert "mamba_backend" in str(error.value)

    # the configuration as a whole is rejected, it does not fall back to another block
    with pytest.raises(ValidationError):
        model_params(
            {
                "block_type": mamba_shim.UNRESOLVED,
                "d_state": 128,
                "d_conv": 4,
                "expand": 2,
                "headdim": 64,
                "mamba_backend": backend,
            }
        )


def test_missing_engine_declaration_is_rejected():
    with pytest.raises(ValidationError) as error:
        MambaBlock(  # type: ignore[call-arg]
            d_state=128, d_conv=4, expand=2, headdim=64, pos_emb_type=None
        )
    assert "mamba_backend" in str(error.value)

    with pytest.raises(ValidationError):
        model_params(
            {
                "block_type": mamba_shim.UNRESOLVED,
                "d_state": 128,
                "d_conv": 4,
                "expand": 2,
                "headdim": 64,
            }
        )


def test_declared_engine_is_read_from_the_parsed_configuration():
    assert declared_mamba_backend(model_params(mamba_block("mamba2"))) == "mamba2"
    assert declared_mamba_backend(model_params(transformer_block())) is None


def test_stages_declaring_different_engines_cannot_be_admitted():
    params = model_params([mamba_block("mamba1"), mamba_block("mamba2")])
    with pytest.raises(MambaAdmissionError):
        declared_mamba_backend(params)

    state = probe_mamba_backend(params)
    assert not state.ok
    assert state.reason is not None and "conflicting" in state.reason


def test_non_mamba_configuration_probes_no_backend(monkeypatch: pytest.MonkeyPatch):
    probed: list[str] = []
    monkeypatch.setattr(mamba_shim, "probe_mamba2", lambda: probed.append("mamba2"))
    monkeypatch.setattr(mamba_shim, "probe_mamba1", lambda: probed.append("mamba1"))

    state = probe_mamba_backend(model_params(transformer_block()))

    assert probed == []
    assert not state.required
    assert state.ok


def test_model_cannot_be_built_before_an_engine_is_admitted(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(mamba_shim, "_ADMITTED_BACKEND", None)

    block = mamba_block("mamba1")
    with pytest.raises(MambaBackendUnavailableError):
        block.to_model(64, 2)


def test_model_is_not_built_with_an_engine_other_than_the_declared_one(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(mamba_shim, "_ADMITTED_BACKEND", mamba_shim.MAMBA1)
    block = mamba_block("mamba2")
    with pytest.raises(MambaBackendUnavailableError):
        block.to_model(64, 2)


def test_admitted_engine_is_bound_and_used(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(mamba_shim, "_ADMITTED_BACKEND", None)

    state = run_mamba_admission(model_params(mamba_block("mamba1")), run_vars=single_process())

    assert state.ok and state.backend == "mamba1"
    assert mamba_shim.bound_backend() == "mamba1"

    block = mamba_block("mamba1")
    model = block.to_model(64, 2)
    assert isinstance(model, torch.nn.Module)
    assert block.block_type == "mamba1"


def test_unavailable_declared_engine_fails_the_run(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(
        mamba_shim,
        "probe_mamba2",
        lambda: mamba_shim.MambaProbeResult(backend=None, version=None, reason="no mamba_ssm"),
    )

    with pytest.raises(SystemExit) as error:
        run_mamba_admission(model_params(mamba_block("mamba2")), run_vars=single_process())

    assert error.value.code == 1


def test_ranks_with_different_engines_fail_together(monkeypatch: pytest.MonkeyPatch):
    local = MambaAdmissionState(required=True, ok=True, declared="mamba2", backend="mamba2")

    def fake_all_gather(gathered: list, _local: MambaAdmissionState) -> None:
        gathered[0] = MambaAdmissionState(
            required=True, ok=True, declared="mamba2", backend="mamba2", version="2.2.2"
        )
        gathered[1] = MambaAdmissionState(
            required=True, ok=True, declared="mamba2", backend="mamba2", version="2.1.0"
        )

    monkeypatch.setattr(mamba_admission.dist, "all_gather_object", fake_all_gather)
    monkeypatch.setattr(mamba_admission.dist, "get_rank", lambda: 0)
    monkeypatch.setattr(mamba_admission.dist, "broadcast_object_list", noop)

    decision = mamba_admission._collective_decision(local, 2)

    assert not decision.ok
    assert decision.reason is not None and "versions" in decision.reason


def test_ranks_that_disagree_on_the_probe_fail_together(monkeypatch: pytest.MonkeyPatch):
    local = MambaAdmissionState(required=True, ok=True, declared="mamba2", backend="mamba2")

    def fake_all_gather(gathered: list, _local: MambaAdmissionState) -> None:
        gathered[0] = local
        gathered[1] = MambaAdmissionState(
            required=True, ok=False, declared="mamba2", reason="ImportError: broken .so"
        )

    monkeypatch.setattr(mamba_admission.dist, "all_gather_object", fake_all_gather)
    monkeypatch.setattr(mamba_admission.dist, "get_rank", lambda: 0)
    monkeypatch.setattr(mamba_admission.dist, "broadcast_object_list", noop)

    decision = mamba_admission._collective_decision(local, 2)

    assert not decision.ok
    assert decision.reason is not None and "broken .so" in decision.reason


def test_marker_records_the_admitted_engine(tmp_path):
    state = MambaAdmissionState(
        required=True, ok=True, declared="mamba2", backend="mamba2", version="2.2.2"
    )

    path = write_mamba_impl_marker(tmp_path, state)

    assert path == tmp_path / MARKER_FILE_NAME
    content = path.read_text(encoding="utf-8")
    assert "declared_backend: mamba2" in content
    assert "actual_backend: mamba2" in content
    assert "package_version: 2.2.2" in content
    assert "failure_reason: None" in content
    assert [p.name for p in tmp_path.iterdir()] == [MARKER_FILE_NAME]


def test_marker_is_not_written_for_runs_without_mamba(tmp_path):
    assert write_mamba_impl_marker(tmp_path, MambaAdmissionState(required=False, ok=True)) is None
    assert list(tmp_path.iterdir()) == []


def test_marker_overwrites_in_place(tmp_path):
    first = MambaAdmissionState(required=True, ok=True, declared="mamba1", backend="mamba1")
    second = MambaAdmissionState(
        required=True, ok=True, declared="mamba2", backend="mamba2", version="2.2.2"
    )

    write_mamba_impl_marker(tmp_path, first)
    path = write_mamba_impl_marker(tmp_path, second)

    assert path is not None
    assert [p.name for p in tmp_path.iterdir()] == [MARKER_FILE_NAME]
    assert "actual_backend: mamba2" in path.read_text(encoding="utf-8")
