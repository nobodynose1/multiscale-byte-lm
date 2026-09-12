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

import csv
import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Generic, NamedTuple, TypeAlias, TypeVar

import torch
import yaml
from filelock import FileLock
from pydantic import BaseModel
from torch import load as torch_load
from torch import save as torch_save
from torch.nn import Module as TorchModule
from torch.serialization import MAP_LOCATION


def _to_path(path: str | Path) -> Path:
    return Path(path) if isinstance(path, str) else path


# Although type parameter lists exist, they're only available in Python >=3.12.
# Use TypeVar and ParamSpec objects instead, which are backwards compatible with
# 3.10. See https://docs.python.org/3/library/typing.html#typing.ParamSpec
_T = TypeVar("_T")
_TBaseModel = TypeVar("_TBaseModel", bound=BaseModel)


def load_yml(
    path: str | Path, parse_to: type[_TBaseModel], try_yaml_suffixes: bool = False
) -> _TBaseModel:
    """
    Load any configuration from a (nested) yaml file.
    Args:
        path (str | Path): The path to the yaml file
        parse_to: An instance of a Pydantic model
        check_yaml_suffixes (bool=False): If `True`, try
            loading the file with either a `yaml` or
            `yml` extension
    Returns:
        A populated instance of the Pydantic model
    """
    path = _to_path(path)
    yaml_suffixes = {".yaml", ".yml"}
    assert path.suffix in yaml_suffixes, f"{path} is not a yaml file"
    try:
        with Path.open(path, "r") as file:
            yaml_data = yaml.safe_load(file)
            return parse_to.model_validate(yaml_data)
    except FileNotFoundError as exception:
        if try_yaml_suffixes:
            yaml_suffixes.remove(path.suffix)
            new_suffix = next(iter(yaml_suffixes))
            return load_yml(path.with_suffix(new_suffix), parse_to, try_yaml_suffixes=False)
        raise exception


def dump_yml(path: str | Path, data: BaseModel) -> Path:
    """
    Atomically dump a model to a yaml file by writing a sibling temp file first.
    A reader either sees the previous complete file or the new complete one.
    """
    path = _to_path(path).with_suffix(".yaml")
    tmp_file = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with tmp_file.open("w", encoding="utf-8") as file:
            yaml.safe_dump(data.model_dump(), file)
        tmp_file.replace(path)
    finally:
        if tmp_file.exists():
            tmp_file.unlink()
    return path


_TNamedTuple = TypeVar("_TNamedTuple", bound=NamedTuple)

# dont pollute with package internal logs
logging.getLogger("filelock").setLevel(logging.WARNING)


class CSVWriter(Generic[_TNamedTuple]):
    def __init__(
        self,
        output_dir: str | Path,
        file_name: str,
        noop: bool = False,
    ) -> None:
        self.noop = noop
        if noop:
            return
        output_dir = _to_path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        self._file = (output_dir / file_name).with_suffix(".csv")

        with self._file.open("w", encoding="utf-8"):
            # clear existing file contents
            pass
        self._lock = FileLock(str(self._file) + ".lock")

    def _file_is_empty(self) -> bool:
        """Check if the file is empty."""
        return self._file.stat().st_size == 0

    def write_row(self, row: _TNamedTuple):
        """
        Write a row.

        Args:
            row: Named tuple to write.
        """
        if self.noop:
            return

        with self._lock:
            file_empty = self._file_is_empty()
            with self._file.open("a", encoding="utf-8", newline="") as f:
                writer = csv.writer(f)
                if file_empty:
                    # automatically write a header once
                    writer.writerow(row._fields)
                writer.writerow(row)


class NDJSONWriter(Generic[_TBaseModel]):
    def __init__(self, path: Path | str):
        self.path = _to_path(path)
        self.stream_positions: list[int] = []

    def write_line(self, obj: _TBaseModel) -> None:
        with self.path.open("a", encoding="utf8") as f:
            self.stream_positions.append(f.tell())
            f.write(obj.model_dump_json() + "\n")

    def remove_last_line(self) -> None:
        if len(self.stream_positions) == 0:
            return None
        last_position = self.stream_positions.pop()
        with self.path.open("a", encoding="utf8") as f:
            f.truncate(last_position)


def read_jsonl(
    path: Path | str,
    parse_lines_to: Callable[..., _T],
) -> list[_T]:
    with _to_path(path).open("r", encoding="utf8") as f:
        return [parse_lines_to(**json.loads(line)) for line in f.readlines()]


StateDict: TypeAlias = dict[str, Any]


def atomic_torch_save(snapshot: StateDict, file: str | Path) -> Path:
    """
    Atomically save a torch checkpoint by writing a sibling temp file first.
    """
    file = _to_path(file)
    file.parent.mkdir(parents=True, exist_ok=True)
    tmp_file = file.with_name(f".{file.name}.{os.getpid()}.tmp")
    try:
        torch_save(snapshot, tmp_file)
        tmp_file.replace(file)
    finally:
        if tmp_file.exists():
            tmp_file.unlink()
    return file


def save_model_state(
    dir: str | Path,
    checkpoint_name: str,
    model: TorchModule | StateDict,
    loss: float,
) -> tuple[bool, Path]:
    """
    Save a model's state_dict to a checkpoint. Will overwrite a file with the same name.

    Args:
        dir (str | Path): If it does not exist, it will be created recursively
        checkpoint_name (str): The suffix will be set to `.pth` automatically.
        model (torch.nn.Module | StateDict): Any class that inherits from
            `torch.nn.Module` or the actual state dict obtained from `model.state_dict()`

    Returns:
        did_overwrite (bool): Whether or not the file was overwritten
    """
    path = _to_path(dir)
    path.mkdir(parents=True, exist_ok=True)
    file = path.joinpath(checkpoint_name).with_suffix(".pth")

    if isinstance(model, TorchModule):
        model = model.state_dict()
    did_overwrite = file.exists()
    snapshot = {"MODEL": model, "LOSS": loss}
    atomic_torch_save(snapshot, file)
    return did_overwrite, file


_TModule = TypeVar("_TModule", bound=TorchModule)


def _state_dict_or_none(obj: Any | None) -> StateDict | None:
    if obj is None:
        return None
    return obj.state_dict()


def save_training_checkpoint_state(
    dir: str | Path,
    checkpoint_name: str,
    *,
    model: TorchModule | StateDict,
    loss: float,
    optimizer: Any | None = None,
    scheduler: Any | None = None,
    grad_scaler: Any | None = None,
    epoch: int,
    batch: int,
    cum_batch: int,
) -> tuple[bool, Path]:
    """
    Save a full training checkpoint suitable for exact interruption resume.
    """
    path = _to_path(dir)
    path.mkdir(parents=True, exist_ok=True)
    file = path.joinpath(checkpoint_name).with_suffix(".pth")

    if isinstance(model, TorchModule):
        model = model.state_dict()

    snapshot = {
        "MODEL": model,
        "LOSS": loss,
        "OPTIMIZER": _state_dict_or_none(optimizer),
        "SCHEDULER": _state_dict_or_none(scheduler),
        "GRAD_SCALER": _state_dict_or_none(grad_scaler),
        "EPOCH": epoch,
        "BATCH": batch,
        "CUM_BATCH": cum_batch,
    }
    did_overwrite = file.exists()
    atomic_torch_save(snapshot, file)
    return did_overwrite, file


def load_checkpoint_snapshot(
    checkpoint_file: str | Path,
    map_location: MAP_LOCATION | None = None,
) -> StateDict:
    return torch_load(
        checkpoint_file,
        map_location=map_location,
        weights_only=True,
    )


@dataclass(frozen=True)
class CheckpointCursor:
    """
    The training position a checkpoint was written at.
    """

    epoch: int
    batch: int
    cum_batch: int


_CURSOR_KEYS = ("EPOCH", "BATCH", "CUM_BATCH")


def read_checkpoint_cursor(snapshot: StateDict) -> CheckpointCursor:
    """
    Read the cursor a checkpoint carries. The checkpoint is the only source of
    the resume position, so a checkpoint without a usable cursor is an error
    rather than a reason to fall back to a config value.
    """
    values: dict[str, int] = {}
    for key in _CURSOR_KEYS:
        if key not in snapshot:
            raise ValueError(f"Checkpoint is missing the {key} cursor entry")
        value = snapshot[key]
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"Checkpoint cursor {key} is not an integer: {value!r}")
        if value < 0:
            raise ValueError(f"Checkpoint cursor {key} is negative: {value}")
        values[key] = value
    return CheckpointCursor(
        epoch=values["EPOCH"],
        batch=values["BATCH"],
        cum_batch=values["CUM_BATCH"],
    )


def _move_optimizer_state_to_device(optimizer: Any, device: str) -> None:
    for state in optimizer.state.values():
        for key, value in state.items():
            if isinstance(value, torch.Tensor):
                state[key] = value.to(device)


_COMPONENT_KEYS = ("MODEL", "OPTIMIZER", "SCHEDULER", "GRAD_SCALER")


@torch.no_grad()
def load_training_checkpoint_state(
    checkpoint_file: str | Path,
    model: _TModule,
    *,
    optimizer: Any,
    scheduler: Any,
    grad_scaler: Any,
    gradient_accumulate_every: int,
    map_location: MAP_LOCATION | None = None,
    optimizer_device: str | None = None,
) -> tuple[CheckpointCursor, float | None]:
    """
    Restore a complete training checkpoint into the objects of the running
    process.

    A checkpoint is only usable as a resume point when every component of the
    training state is present and is accepted by the object it belongs to; a
    model-only file is not a training checkpoint, and there is no fallback to a
    partial restore. Failures are reported with the checkpoint path so the
    caller can name the file it rejected.

    Returns:
        cursor (CheckpointCursor): The position the checkpoint was written at
        loss (float | None): The historical loss stored alongside it, if any
    """
    try:
        snapshot = load_checkpoint_snapshot(checkpoint_file, map_location=map_location)
    except Exception as error:
        raise ValueError(f"checkpoint {checkpoint_file} cannot be read: {error}") from error

    missing = [key for key in _COMPONENT_KEYS if snapshot.get(key) is None]
    if missing:
        raise ValueError(
            f"checkpoint {checkpoint_file} is not a complete training checkpoint, "
            f"missing: {', '.join(missing)}"
        )

    try:
        cursor = read_checkpoint_cursor(snapshot)
    except ValueError as error:
        raise ValueError(f"checkpoint {checkpoint_file} has no usable cursor: {error}") from error
    if cursor.cum_batch % gradient_accumulate_every != 0:
        raise ValueError(
            f"checkpoint {checkpoint_file} sits at cumulative micro-batch {cursor.cum_batch}, "
            f"which is not a multiple of gradient_accumulate_every "
            f"({gradient_accumulate_every})"
        )

    try:
        model.load_state_dict(snapshot["MODEL"], strict=True)
        optimizer.load_state_dict(snapshot["OPTIMIZER"])
        scheduler.load_state_dict(snapshot["SCHEDULER"])
        grad_scaler.load_state_dict(snapshot["GRAD_SCALER"])
    except Exception as error:
        raise ValueError(
            f"checkpoint {checkpoint_file} does not match this run: {type(error).__name__}: {error}"
        ) from error

    if optimizer_device:
        _move_optimizer_state_to_device(optimizer, optimizer_device)

    loss = snapshot.get("LOSS")
    return cursor, None if loss is None else float(loss)
