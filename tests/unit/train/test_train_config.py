from typing import Any

import pytest
from pydantic import ValidationError

from mblm.train.core.config import CoreTrainConfig


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
