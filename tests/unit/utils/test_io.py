# pyright: reportIndexIssue=false, reportArgumentType=false

import csv
import tempfile
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import NamedTuple

import pytest
import torch
from pydantic import BaseModel

from mblm.utils.io import (
    CSVWriter,
    NDJSONWriter,
    dump_yml,
    load_checkpoint_snapshot,
    load_yml,
    read_checkpoint_cursor,
    read_jsonl,
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
