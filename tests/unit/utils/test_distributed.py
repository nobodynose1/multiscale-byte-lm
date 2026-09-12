from datetime import timedelta

import pytest
from pytest_mock import MockerFixture

import mblm.utils.distributed as distributed_module
from mblm.utils.distributed import ElasticRunVars, process_group


def fake_torchrun_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LOCAL_RANK", "0")
    monkeypatch.setenv("WORLD_SIZE", "1")


class TestProcessGroupTimeout:
    def test_the_configured_timeout_is_forwarded_to_torch(
        self, mocker: MockerFixture, monkeypatch: pytest.MonkeyPatch
    ):
        init_process_group = mocker.patch.object(distributed_module, "init_process_group")
        mocker.patch.object(distributed_module, "destroy_process_group")
        mocker.patch.object(distributed_module, "get_rank", lambda: 0)
        fake_torchrun_environment(monkeypatch)

        with process_group(backend="gloo", timeout=timedelta(seconds=600)) as run_vars:
            assert run_vars == ElasticRunVars(
                local_rank=0, global_rank=0, world_size=1, is_cuda=False
            )

        init_process_group.assert_called_once_with(backend="gloo", timeout=timedelta(seconds=600))

    def test_without_a_timeout_torch_keeps_its_default(
        self, mocker: MockerFixture, monkeypatch: pytest.MonkeyPatch
    ):
        init_process_group = mocker.patch.object(distributed_module, "init_process_group")
        mocker.patch.object(distributed_module, "destroy_process_group")
        mocker.patch.object(distributed_module, "get_rank", lambda: 0)
        fake_torchrun_environment(monkeypatch)

        with process_group(backend="gloo"):
            pass

        init_process_group.assert_called_once_with(backend="gloo", timeout=None)
