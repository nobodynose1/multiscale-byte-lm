# pyright: reportIndexIssue=false, reportArgumentType=false

import csv
import tempfile
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Any, NamedTuple

import pytest
import torch
from pydantic import BaseModel

from mblm.utils.io import (
    CheckpointCursor,
    CSVWriter,
    NDJSONWriter,
    dump_yml,
    load_checkpoint_snapshot,
    load_training_checkpoint_state,
    load_yml,
    read_checkpoint_cursor,
    read_jsonl,
    save_model_state,
    save_training_checkpoint_state,
)

# TODO: Python 3.12, assert_type


class TestYMLUtils:
    def test_cfrom_yml(self):
        class Klass(BaseModel):
            num: int

        with tempfile.TemporaryDirectory() as temp_dir:
            kls = Klass(num=5)
            dumped_to = dump_yml(Path(temp_dir) / "file", kls)
            restored = load_yml(dumped_to, Klass)
            # assert_type(restored, Klass)
            assert isinstance(restored, Klass)


class DummyCSVEntry(NamedTuple):
    kind: str
    idx: int
    time: str


class TestCSVWriter:
    def test_parallel(self):
        """Simulate parallel writes to the same file with index and timestamp."""

        # Verify the file content
        with tempfile.TemporaryDirectory() as tmpdir:
            temp_output_dir = Path(tmpdir)
            csv_writer = CSVWriter[DummyCSVEntry](output_dir=temp_output_dir, file_name="test")

            def write_row(index):
                row = DummyCSVEntry(
                    kind="test",
                    idx=index,
                    time=datetime.now().isoformat(),
                )
                csv_writer.write_row(row)

            with ThreadPoolExecutor(max_workers=4) as executor:
                futures = [executor.submit(write_row, i) for i in range(10)]
                for future in futures:
                    future.result()
                csv_file = temp_output_dir / "test.csv"
                with csv_file.open("r", encoding="utf-8") as f:
                    reader = list(csv.reader(f))
                    assert reader[0] == list(DummyCSVEntry._fields)
                    assert len(reader) == 11

                    indexes = []
                    for row in reader[1:]:
                        assert row[0] == "test"
                        assert len(row[2]) > 0
                        indexes.append(row[1])
                    # Order may differ due to concurrent writing
                    assert list(map(str, range(10))) == sorted(indexes)


class TestCheckpointCursor:
    def test_a_written_cursor_is_read_back_from_the_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            _, checkpoint = save_training_checkpoint_state(
                tmpdir,
                "latest",
                model=torch.nn.Linear(2, 2),
                loss=1.5,
                epoch=3,
                batch=7,
                cum_batch=42,
            )
            snapshot = load_checkpoint_snapshot(checkpoint)

        cursor = read_checkpoint_cursor(snapshot)
        assert (cursor.epoch, cursor.batch, cursor.cum_batch) == (3, 7, 42)

    @pytest.mark.parametrize("missing", ["EPOCH", "BATCH", "CUM_BATCH"])
    def test_a_missing_cursor_entry_is_rejected(self, missing: str):
        snapshot = {"EPOCH": 1, "BATCH": 2, "CUM_BATCH": 3}
        snapshot.pop(missing)

        with pytest.raises(ValueError, match=missing):
            read_checkpoint_cursor(snapshot)

    @pytest.mark.parametrize("value", [True, "3", 1.5, None])
    def test_a_non_integer_cursor_entry_is_rejected(self, value):
        with pytest.raises(ValueError, match="BATCH"):
            read_checkpoint_cursor({"EPOCH": 1, "BATCH": value, "CUM_BATCH": 3})

    def test_a_negative_cursor_entry_is_rejected(self):
        with pytest.raises(ValueError, match="negative"):
            read_checkpoint_cursor({"EPOCH": -1, "BATCH": 0, "CUM_BATCH": 0})


def loading_targets() -> tuple[torch.nn.Linear, Any, Any, Any]:
    """Objects of the running process that a checkpoint is restored into."""
    model = torch.nn.Linear(2, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    scheduler = torch.optim.lr_scheduler.PolynomialLR(optimizer, total_iters=10, power=1.0)
    return model, optimizer, scheduler, torch.GradScaler(device="cpu")


def full_snapshot(*, loss: float | None = 1.0, cum_batch: int = 4) -> dict[str, Any]:
    """The whole training state as a checkpoint stores it."""
    model = torch.nn.Linear(2, 2)
    with torch.no_grad():
        model.weight.fill_(0.25)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    scheduler = torch.optim.lr_scheduler.PolynomialLR(optimizer, total_iters=10, power=1.0)
    return {
        "MODEL": model.state_dict(),
        "OPTIMIZER": optimizer.state_dict(),
        "SCHEDULER": scheduler.state_dict(),
        "GRAD_SCALER": torch.GradScaler(device="cpu").state_dict(),
        "EPOCH": 1,
        "BATCH": 2,
        "CUM_BATCH": cum_batch,
        **({} if loss is None else {"LOSS": loss}),
    }


def write_snapshot(path: Path, snapshot: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(snapshot, path)
    return path


def load_into_targets(
    path: Path, *, gradient_accumulate_every: int
) -> tuple[CheckpointCursor, float | None]:
    model, optimizer, scheduler, grad_scaler = loading_targets()
    return load_training_checkpoint_state(
        path,
        model,
        optimizer=optimizer,
        scheduler=scheduler,
        grad_scaler=grad_scaler,
        gradient_accumulate_every=gradient_accumulate_every,
    )


class TestTrainingCheckpointPreflight:
    """A checkpoint is a resume point only when it carries every component of
    the training state and this run's objects accept it."""

    def test_a_complete_checkpoint_restores_the_components_and_the_cursor(self, tmp_path: Path):
        snapshot = full_snapshot(loss=4.5, cum_batch=6)
        path = write_snapshot(tmp_path / "latest.pth", snapshot)

        cursor, loss = load_into_targets(path, gradient_accumulate_every=2)

        assert cursor == CheckpointCursor(epoch=1, batch=2, cum_batch=6)
        assert loss == 4.5

    def test_the_restored_model_carries_the_stored_weights(self, tmp_path: Path):
        snapshot = full_snapshot()
        path = write_snapshot(tmp_path / "latest.pth", snapshot)
        model, optimizer, scheduler, grad_scaler = loading_targets()

        load_training_checkpoint_state(
            path,
            model,
            optimizer=optimizer,
            scheduler=scheduler,
            grad_scaler=grad_scaler,
            gradient_accumulate_every=1,
        )

        assert torch.equal(model.weight, snapshot["MODEL"]["weight"])

    @pytest.mark.parametrize("missing", ["MODEL", "OPTIMIZER", "SCHEDULER", "GRAD_SCALER"])
    def test_a_checkpoint_missing_one_component_is_rejected(self, missing: str, tmp_path: Path):
        snapshot = full_snapshot()
        snapshot.pop(missing)
        path = write_snapshot(tmp_path / "latest.pth", snapshot)

        with pytest.raises(ValueError) as error:
            load_into_targets(path, gradient_accumulate_every=1)

        assert "not a complete training checkpoint" in str(error.value)
        assert missing in str(error.value)

    def test_a_model_only_file_is_refused_as_a_resume_point(self, tmp_path: Path):
        model = torch.nn.Linear(2, 2)
        _, path = save_model_state(tmp_path, "best", model=model, loss=1.0)

        with pytest.raises(ValueError, match="not a complete training checkpoint"):
            load_into_targets(path, gradient_accumulate_every=1)

    def test_a_cursor_off_the_accumulation_boundary_is_rejected(self, tmp_path: Path):
        path = write_snapshot(tmp_path / "latest.pth", full_snapshot(cum_batch=5))

        with pytest.raises(ValueError, match="not a multiple of gradient_accumulate_every"):
            load_into_targets(path, gradient_accumulate_every=2)

    def test_a_checkpoint_without_a_stored_loss_still_resumes(self, tmp_path: Path):
        path = write_snapshot(tmp_path / "latest.pth", full_snapshot(loss=None, cum_batch=4))

        cursor, loss = load_into_targets(path, gradient_accumulate_every=1)

        assert cursor == CheckpointCursor(epoch=1, batch=2, cum_batch=4)
        assert loss is None

    def test_a_snapshot_without_a_cursor_is_rejected(self, tmp_path: Path):
        snapshot = full_snapshot()
        snapshot.pop("EPOCH")
        path = write_snapshot(tmp_path / "latest.pth", snapshot)

        with pytest.raises(ValueError, match="has no usable cursor"):
            load_into_targets(path, gradient_accumulate_every=1)

    def test_a_file_that_is_not_a_checkpoint_is_rejected(self, tmp_path: Path):
        path = tmp_path / "latest.pth"
        path.write_bytes(b"not a checkpoint")

        with pytest.raises(ValueError, match="cannot be read"):
            load_into_targets(path, gradient_accumulate_every=1)

    def test_a_checkpoint_of_another_model_is_rejected(self, tmp_path: Path):
        path = write_snapshot(tmp_path / "latest.pth", full_snapshot())
        model = torch.nn.Linear(3, 3)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        scheduler = torch.optim.lr_scheduler.PolynomialLR(optimizer, total_iters=10, power=1.0)

        with pytest.raises(ValueError, match="does not match this run"):
            load_training_checkpoint_state(
                path,
                model,
                optimizer=optimizer,
                scheduler=scheduler,
                grad_scaler=torch.GradScaler(device="cpu"),
                gradient_accumulate_every=1,
            )


class TestNDJSONWriter:
    class MyClass(BaseModel):  # noqa: D106
        data: str

    first_entry = MyClass(data="a")
    temp_entry = MyClass(data="bbbbbbb")  # long entry
    second_entry = MyClass(data="c")

    def test_write_and_remove(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            file = Path(tmpdir) / "file.jsonl"
            writer = NDJSONWriter[TestNDJSONWriter.MyClass](file)
            writer.write_line(self.first_entry)
            writer.write_line(self.temp_entry)
            writer.remove_last_line()
            writer.write_line(self.second_entry)

            result = read_jsonl(file, parse_lines_to=self.MyClass)
            assert len(result) == 2
            assert result[0] == self.first_entry
            assert result[1] == self.second_entry

    def test_write_and_remove_multiple(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            file = Path(tmpdir) / "file.jsonl"
            writer = NDJSONWriter[TestNDJSONWriter.MyClass](file)
            writer.write_line(self.first_entry)
            writer.write_line(self.temp_entry)
            writer.remove_last_line()
            writer.remove_last_line()
            writer.remove_last_line()

            result = read_jsonl(file, parse_lines_to=self.MyClass)
            assert len(result) == 0
