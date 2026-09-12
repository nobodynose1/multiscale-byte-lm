"""The order a dataset enumerates its input directory in is fixed by the names
it finds there, and the run can attest which entries it read in that order."""

import json
from pathlib import Path

import pytest

import mblm.data.dataset.clevr as clevr_module
from mblm.data.dataset.clevr import Clevr, ClevrOptionalArgs
from mblm.data.dataset.pg19 import PG19
from mblm.data.dataset.pg19_masked import PG19Masked
from mblm.data.types import ModelMode
from mblm.data.utils import (
    ordered_name_size_sha256,
    sorted_dir_entries,
)

CLEVR_FIXTURE_PATH = Path("tests/fixtures/clevr")

PG19_BOOK_BYTES = 64
PG19_TRAIN_NAMES = ["a.txt", "b.txt"]


def sorted_names(directory: Path) -> list[str]:
    return [entry.name for entry in sorted_dir_entries(directory)]


def digest_of(directory: Path) -> str:
    return ordered_name_size_sha256(sorted_dir_entries(directory))


def build_pg19_tree(root: Path) -> Path:
    """
    A minimal PG19 tree whose train folder holds two books that a file system is
    free to return in either order.
    """
    for split in ("train", "validation"):
        (root / split).mkdir()
    (root / "train" / "b.txt").write_bytes(b"B" * PG19_BOOK_BYTES)
    (root / "train" / "a.txt").write_bytes(b"A" * PG19_BOOK_BYTES)
    return root


def clevr_from_fixture() -> Clevr:
    return Clevr(
        CLEVR_FIXTURE_PATH,
        mode=ModelMode.VALID,
        pad_token_id=1001,
        optional_args=ClevrOptionalArgs(qiqa_loss_mask=(1.0, 1.0, 1.0, 1.0), target_mode="a"),
        seq_len=500_000,
        worker_id=0,
        num_workers=1,
    )


class RecordingImagePipeline:
    """Stand-in for the image pipeline that records which path it was handed."""

    handed: list[str] = []

    def __init__(self, path: Path, _color_space: object) -> None:
        self.handed.append(Path(path).name)

    def to_tensor(self) -> str:
        return self.handed[-1]


class TestEnumerationOrder:
    def test_entries_are_ordered_by_the_bytes_of_their_names(self, tmp_path: Path):
        for name in ("b.txt", "a.txt", "C.txt", "10.txt", "2.txt"):
            (tmp_path / name).write_bytes(b"x")

        assert sorted_names(tmp_path) == ["10.txt", "2.txt", "C.txt", "a.txt", "b.txt"]

    def test_ordering_changes_the_sequence_but_not_the_members(self, tmp_path: Path):
        (tmp_path / "b.txt").write_bytes(b"x")
        (tmp_path / "a.txt").write_bytes(b"x")
        # a subdirectory is part of the enumeration: this fixes the order only
        (tmp_path / "sub").mkdir()

        entries = sorted_dir_entries(tmp_path)

        assert {entry.name for entry in entries} == {entry.name for entry in tmp_path.iterdir()}
        assert len(entries) == len(list(tmp_path.iterdir()))
        assert sorted_names(tmp_path) == ["a.txt", "b.txt", "sub"]

    def test_the_order_and_the_digest_ignore_what_the_directory_returns(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        for name in ("b.txt", "a.txt", "c.txt"):
            (tmp_path / name).write_bytes(b"payload")
        expected_names = ["a.txt", "b.txt", "c.txt"]

        assert sorted_names(tmp_path) == expected_names
        expected_digest = ordered_name_size_sha256([tmp_path / name for name in expected_names])

        original_iterdir = Path.iterdir
        monkeypatch.setattr(Path, "iterdir", lambda self: reversed(list(original_iterdir(self))))

        assert sorted_names(tmp_path) == expected_names
        assert digest_of(tmp_path) == expected_digest


class TestNameSizeDigest:
    def test_the_digest_changes_when_an_entry_is_added_removed_renamed_or_resized(
        self, tmp_path: Path
    ):
        (tmp_path / "a.txt").write_bytes(b"aaa")
        (tmp_path / "b.txt").write_bytes(b"bb")
        baseline = digest_of(tmp_path)

        (tmp_path / "c.txt").write_bytes(b"c")
        added = digest_of(tmp_path)
        assert added != baseline
        (tmp_path / "c.txt").unlink()
        assert digest_of(tmp_path) == baseline

        (tmp_path / "a.txt").write_bytes(b"aaaa")
        resized = digest_of(tmp_path)
        assert resized != baseline
        (tmp_path / "a.txt").write_bytes(b"aaa")
        assert digest_of(tmp_path) == baseline

        (tmp_path / "a.txt").rename(tmp_path / "z.txt")
        assert digest_of(tmp_path) != baseline

    def test_the_digest_does_not_cover_the_content_of_a_file(self, tmp_path: Path):
        """
        The record attests which entries were read in which order, not what they
        held: a same size rewrite under the same name keeps the digest.
        """
        book = tmp_path / "a.txt"
        book.write_bytes(b"0123456789")
        before = digest_of(tmp_path)

        book.write_bytes(b"9876543210")

        assert book.stat().st_size == 10
        assert digest_of(tmp_path) == before

    def test_a_name_and_a_size_cannot_share_one_encoding(self, tmp_path: Path):
        # a plain concatenation would spell "a12" for both of these
        left, right = tmp_path / "left", tmp_path / "right"
        left.mkdir()
        right.mkdir()
        (left / "a").write_bytes(b"x" * 12)
        (right / "a1").write_bytes(b"x" * 2)

        assert digest_of(left) != digest_of(right)


class TestReadPathsUseTheSameOrder:
    def test_pg19_reads_its_books_in_name_byte_order(self, tmp_path: Path):
        root = build_pg19_tree(tmp_path)
        dataset = PG19(
            data_dir=root,
            mode=ModelMode.TRAIN,
            display_load_progress=False,
            seq_len=PG19_BOOK_BYTES,
            worker_id=0,
            num_workers=1,
        )

        assert [entry.name for entry in dataset.txt_files] == PG19_TRAIN_NAMES
        assert dataset.data[:PG19_BOOK_BYTES].tolist() == [ord("A")] * PG19_BOOK_BYTES
        assert dataset.data_lineage() == ordered_name_size_sha256(dataset.txt_files)
        assert dataset.data_lineage() == digest_of(root / "train")

    def test_pg19_masked_reads_its_books_in_name_byte_order(self, tmp_path: Path):
        root = build_pg19_tree(tmp_path)
        dataset = PG19Masked(
            data_dir=root,
            mode=ModelMode.TRAIN,
            masked_token_id=-100,
            masking_proba=0.15,
            padding_token_id=-101,
            display_load_progress=False,
            seq_len=PG19_BOOK_BYTES,
            worker_id=0,
            num_workers=1,
        )

        assert [entry.name for entry in dataset.txt_files] == PG19_TRAIN_NAMES
        assert dataset.data[:PG19_BOOK_BYTES].tolist() == [ord("A")] * PG19_BOOK_BYTES
        assert dataset.data_lineage() == ordered_name_size_sha256(dataset.txt_files)
        assert dataset.data_lineage() == digest_of(root / "train")

    def test_clevr_lists_its_images_in_name_byte_order(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        images = tmp_path / "images" / "val"
        images.mkdir(parents=True)
        (tmp_path / "questions").mkdir()
        for name in ("2.png", "1.png"):
            (images / name).write_bytes(b"never decoded by this test")
        (tmp_path / "questions" / "CLEVR_val_questions.json").write_text(
            json.dumps(
                {
                    "questions": [
                        {
                            "question": "how many cubes",
                            "answer": "1",
                            "image_filename": "1.png",
                            "question_family_index": 0,
                            "program": [],
                        }
                    ]
                }
            ),
            encoding="utf8",
        )
        monkeypatch.setattr(clevr_module, "ImagePipeline", RecordingImagePipeline)
        clevr = Clevr(
            tmp_path,
            mode=ModelMode.VALID,
            pad_token_id=1001,
            optional_args=ClevrOptionalArgs(qiqa_loss_mask=(1.0, 1.0, 1.0, 1.0), target_mode="a"),
            seq_len=1,
            worker_id=0,
            num_workers=1,
        )

        assert list(clevr.iter_images()) == ["1.png", "2.png"]
        assert sorted(RecordingImagePipeline.handed) == ["1.png", "2.png"]

        original_iterdir = Path.iterdir
        monkeypatch.setattr(Path, "iterdir", lambda self: reversed(list(original_iterdir(self))))

        assert list(clevr.iter_images()) == ["1.png", "2.png"]


class TestClevrLineage:
    def test_clevr_records_no_lineage(self):
        """
        CLEVR reads its questions from a single JSON file rather than by
        enumerating a directory, so its lineage is not applicable, not
        uncollected.
        """
        assert clevr_from_fixture().data_lineage() is None
