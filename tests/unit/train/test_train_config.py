from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from mblm.train.core.config import CoreTrainConfig
from mblm.train.mblm import TrainEntryConfig
from mblm.utils.io import load_yml


def train_config(**overrides: Any) -> CoreTrainConfig:
    return CoreTrainConfig(
        target_elements=1,
        target_elements_strategy="batch",
        batch_size=1,
        learning_rate=0.001,
        gradient_accumulate_every=1,
        **overrides,
    )


class TestDistributedTimeoutSeconds:
    def test_the_default_timeout_is_six_hundred_seconds(self):
        config = train_config()

        assert config.distributed_timeout_seconds == 600
        assert config.model_dump()["distributed_timeout_seconds"] == 600

    def test_an_explicit_timeout_is_kept(self):
        assert train_config(distributed_timeout_seconds=1234).distributed_timeout_seconds == 1234

    @pytest.mark.parametrize("invalid", [0, -1])
    def test_a_non_positive_timeout_is_rejected(self, invalid: int):
        with pytest.raises(ValidationError):
            train_config(distributed_timeout_seconds=invalid)


DUAL_LINE_CONFIGS = (
    Path("config/pg19_30bb_360m_1d_s.yaml"),
    Path("config/pg19_30bb_360m_1d_t.yaml"),
)


def dual_line_seeds() -> dict[str, int | None]:
    return {
        config_path.name: load_yml(config_path, parse_to=TrainEntryConfig).train.seed
        for config_path in DUAL_LINE_CONFIGS
    }


class TestDualLineBaseSeed:
    def test_both_lines_declare_a_base_seed(self):
        seeds = dual_line_seeds()

        assert all(
            seed is not None for seed in seeds.values()
        ), f"a dual line does not declare train.seed: {seeds}"

    def test_the_two_lines_declare_the_same_base_seed(self):
        seeds = dual_line_seeds()

        assert len(set(seeds.values())) == 1, f"the dual lines disagree on the base seed: {seeds}"
