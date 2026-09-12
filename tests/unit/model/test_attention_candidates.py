"""A flash-attention request keeps every SDPA backend as a candidate: the device
capability no longer prunes the list, so flash stays available where torch can
run it and the math / efficient fallbacks stay available where it cannot."""

import contextlib
from collections.abc import Iterator

import pytest
import torch
from torch.nn.attention import SDPBackend

from mblm.model.transformer import AttendWithMask

ALL_BACKENDS = [
    SDPBackend.FLASH_ATTENTION,
    SDPBackend.MATH,
    SDPBackend.EFFICIENT_ATTENTION,
]


class DeviceProperties:
    def __init__(self, major: int, minor: int) -> None:
        self.major = major
        self.minor = minor


def no_device_query(*_args: object) -> None:
    raise AssertionError("the device capability must not decide the SDPA candidates")


def make_cuda_available(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_properties", no_device_query)


def test_a_non_a100_device_no_longer_loses_flash(monkeypatch: pytest.MonkeyPatch):
    make_cuda_available(monkeypatch)
    monkeypatch.setattr(
        torch.cuda, "get_device_properties", lambda _device: DeviceProperties(major=8, minor=9)
    )

    attend = AttendWithMask(causal=True, flash=True)

    assert SDPBackend.FLASH_ATTENTION in attend.attn_cfg
    assert attend.attn_cfg == ALL_BACKENDS


def test_an_a100_device_no_longer_loses_the_fallbacks(monkeypatch: pytest.MonkeyPatch):
    make_cuda_available(monkeypatch)
    monkeypatch.setattr(
        torch.cuda, "get_device_properties", lambda _device: DeviceProperties(major=8, minor=0)
    )

    attend = AttendWithMask(causal=True, flash=True)

    assert SDPBackend.MATH in attend.attn_cfg
    assert SDPBackend.EFFICIENT_ATTENTION in attend.attn_cfg


def test_the_device_capability_is_never_queried(monkeypatch: pytest.MonkeyPatch):
    make_cuda_available(monkeypatch)

    attend = AttendWithMask(causal=True, flash=True)

    assert attend.attn_cfg == ALL_BACKENDS


def test_the_candidates_are_untouched_when_flash_is_disabled(monkeypatch: pytest.MonkeyPatch):
    make_cuda_available(monkeypatch)

    attend = AttendWithMask(causal=True, flash=False)

    assert attend.attn_cfg == ALL_BACKENDS


def test_the_candidates_are_untouched_without_cuda(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    attend = AttendWithMask(causal=True, flash=True)

    assert attend.attn_cfg == ALL_BACKENDS


def test_the_candidates_reach_the_sdpa_call(monkeypatch: pytest.MonkeyPatch):
    make_cuda_available(monkeypatch)
    seen: list[list[SDPBackend]] = []
    original = torch.nn.attention.sdpa_kernel

    @contextlib.contextmanager
    def recording_kernel(backends: list[SDPBackend]) -> Iterator[None]:
        seen.append(list(backends))
        with original(backends):
            yield

    monkeypatch.setattr(torch.nn.attention, "sdpa_kernel", recording_kernel)

    attend = AttendWithMask(causal=True, flash=True)
    q = torch.randn(2, 4, 5, 8)

    attend(q, q, q)

    assert seen == [ALL_BACKENDS]
