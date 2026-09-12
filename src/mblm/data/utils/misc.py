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

import hashlib
import os
from pathlib import Path
from typing import Iterable

import torch

# The rule a run's datasets enumerate a directory with. It labels the order the
# run's data was read in, so a run can be compared with the ones that read the
# same data before the order was fixed.
DATA_ENUMERATION_VERSION = "sorted-name-bytes-v1"


def sorted_dir_entries(directory: Path) -> list[Path]:
    """
    The entries of `directory`, ordered by the file system bytes of their names.

    Sorting by the name fixes the sequence of an enumeration that would
    otherwise be whatever the file system happens to return, so the same
    directory gives the same sequence on every machine and every run. The
    entries are not filtered: the result holds exactly what the directory holds.
    """
    return sorted(directory.iterdir(), key=lambda entry: os.fsencode(entry.name))


def ordered_name_size_sha256(entries: Iterable[Path]) -> str:
    """
    Digest of the entries' names and sizes, in the order they are given.

    The digest identifies which entries a dataset read and in which order - the
    data a run's sequence is built from - but not their content: replacing a
    file's bytes while keeping its name and size leaves the digest unchanged.
    Each name is written as its raw file system bytes, preceded by the length of
    those bytes, so no two name/size sequences can encode to the same input.
    """
    digest = hashlib.sha256()
    for entry in entries:
        name = os.fsencode(entry.name)
        digest.update(len(name).to_bytes(8, "big"))
        digest.update(name)
        digest.update(entry.stat().st_size.to_bytes(8, "big"))
    return digest.hexdigest()


def target_loss_mask(
    tensor_weight_pairs: Iterable[tuple[torch.Tensor, float]],
) -> torch.Tensor:
    """
    Create a loss mask for an iterable of pairs of tensor, mask_weight.
    """
    loss_mask_tensors: list[torch.Tensor] = []
    for tensor, loss_weight in tensor_weight_pairs:
        loss_mask_tensors.append(torch.full_like(tensor, dtype=torch.float, fill_value=loss_weight))
    return torch.concat(loss_mask_tensors)


def shift_remap_tensor(
    tensor: torch.Tensor, range_start: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Remaps the values of a tensor to a new range starting from `range_start`.
    Returns the remapped tensor of the same shape and dtype and two tensors to
    map back to the original.

    Example:

        shifted, unshift, indices = remap_values(input_tensor, range_start=11)
        assert input_tensor.equal(unshift[indices])
    """
    uniqued: tuple[torch.Tensor, torch.Tensor] = torch.unique(
        tensor, return_inverse=True, sorted=True
    )
    # inverse_indices is the indices of the original elements in the returned
    # unique_values tensor
    unique_values, inverse_indices = uniqued
    remapped_values = torch.arange(
        range_start, range_start + unique_values.size(0), dtype=tensor.dtype
    )
    remapped_tensor = remapped_values[inverse_indices].reshape_as(tensor)

    return remapped_tensor, unique_values, inverse_indices
