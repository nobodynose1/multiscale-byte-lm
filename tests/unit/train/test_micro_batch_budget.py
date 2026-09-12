"""The micro-batches a rank trains on come from the config and the world size,
never from a data loader, and a run only starts when that count fills whole
accumulation windows: a partial window would leave gradients that are never
stepped and never cleared."""

from collections.abc import Iterator
from pathlib import Path
from typing import Literal

import pytest
import torch

from mblm.train.core.config import (
    CoreIoConfig,
    CoreModelParams,
    CoreTrainConfig,
    GenericEntryConfig,
)
from mblm.train.core.trainer import CoreTrainer
from mblm.train.mblm import MegabyteTrainer, TrainEntryConfig
from mblm.utils.distributed import ElasticRunVars
from mblm.utils.io import load_yml

WORLD_SIZES = (1, 2, 4)

# the target the PG19 lines carry after the accumulation fix
PG19_TARGET = 30_000_807_936
PG19_SEQ_LEN = 8192
# the target they carried before it, which only happened to divide on two ranks
PG19_TARGET_BEFORE = 30_000_000_000

SHIPPED_CONFIGS = sorted(Path("config").glob("*.yaml"))


class TinyParams(CoreModelParams):
    pass


class TinyEntryConfig(GenericEntryConfig[TinyParams, CoreTrainConfig, CoreIoConfig]):
    pass


class TinyModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor([1.0]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.weight


class TinyTrainer(CoreTrainer[TinyModel, torch.Tensor, TinyParams, CoreTrainConfig, CoreIoConfig]):
    def init_model(self) -> TinyModel:
        return TinyModel()

    def model_forward(self, model: TinyModel, batch: torch.Tensor, device: str) -> torch.Tensor:
        return model(batch.to(device)).sum()

    def configure_optimizer(
        self, parameters: Iterator[torch.nn.Parameter]
    ) -> torch.optim.Optimizer:
        return torch.optim.SGD(parameters, lr=0.1)


def run_vars(world_size: int) -> ElasticRunVars:
    return ElasticRunVars(local_rank=0, global_rank=0, world_size=world_size, is_cuda=False)


def tiny_config(
    *,
    target_elements: int = PG19_TARGET,
    gradient_accumulate_every: int,
    batch_size: int = 1,
    input_seq_len: int = 1,
    strategy: Literal["batch", "sequence"] = "batch",
) -> TinyEntryConfig:
    return TinyEntryConfig(
        params=TinyParams(input_seq_len=input_seq_len),
        train=CoreTrainConfig(
            target_elements=target_elements,
            target_elements_strategy=strategy,
            batch_size=batch_size,
            learning_rate=0.1,
            gradient_accumulate_every=gradient_accumulate_every,
        ),
        io=CoreIoConfig(
            name_model="budget",
            output_dir="unused",
            num_models_to_save=0,
            validate_amount=1,
            log_train_loss_amount=1,
        ),
    )


def pg19_config(
    *,
    batch_size: int,
    gradient_accumulate_every: int,
    target_elements: int = PG19_TARGET,
) -> TinyEntryConfig:
    """One of the PG19 30B recipes: sequences of 8192 counted as elements."""
    return tiny_config(
        target_elements=target_elements,
        gradient_accumulate_every=gradient_accumulate_every,
        batch_size=batch_size,
        input_seq_len=PG19_SEQ_LEN,
        strategy="sequence",
    )


def budget(config: TinyEntryConfig, *, world_size: int) -> TinyTrainer:
    return TinyTrainer(config, run_vars=run_vars(world_size))


def validate(config: TinyEntryConfig, *, world_size: int) -> None:
    budget(config, world_size=world_size)._validate_config()


class TestTheMicroBatchBudgetFollowsTheWorldSize:
    @pytest.mark.parametrize("world_size,expected_steps", [(1, 76_296), (2, 38_148), (4, 19_074)])
    @pytest.mark.parametrize(
        "batch_size,gradient_accumulate_every",
        [(2, 24), (4, 12), (6, 8)],
        ids=["1d_s", "1d_t", "2d"],
    )
    def test_a_pg19_recipe_fills_whole_windows_on_every_world_size(
        self,
        world_size: int,
        expected_steps: int,
        batch_size: int,
        gradient_accumulate_every: int,
    ):
        """The three shipped recipes move 48 sequences per optimizer step, so
        every world size takes the same number of steps."""
        config = pg19_config(
            batch_size=batch_size, gradient_accumulate_every=gradient_accumulate_every
        )
        trainer = budget(config, world_size=world_size)

        validate(config, world_size=world_size)
        assert trainer.local_gradient_steps() == expected_steps
        assert trainer.local_batch_iters() == expected_steps * gradient_accumulate_every

    def test_the_target_the_lines_carried_before_is_refused_where_it_had_to_be(self):
        old = pg19_config(
            batch_size=2,
            gradient_accumulate_every=24,
            target_elements=PG19_TARGET_BEFORE,
        )

        for world_size in (1, 4):
            with pytest.raises(ValueError, match="not a multiple of gradient_accumulate_every"):
                validate(old, world_size=world_size)

        # two ranks divided evenly by accident, which is why this was easy to miss
        validate(old, world_size=2)


class TestARunMustFillWholeAccumulationWindows:
    def test_micro_batches_fewer_than_one_window_are_refused(self):
        config = tiny_config(target_elements=1, gradient_accumulate_every=2)

        with pytest.raises(ValueError, match="1 micro-batches are fewer than"):
            validate(config, world_size=1)

    def test_a_run_with_no_micro_batch_is_refused(self):
        config = tiny_config(target_elements=0, gradient_accumulate_every=1)

        with pytest.raises(ValueError, match="0 micro-batches are fewer than"):
            validate(config, world_size=1)

    def test_a_partial_window_at_the_end_is_refused(self):
        config = tiny_config(target_elements=3, gradient_accumulate_every=2)

        with pytest.raises(ValueError, match="would end with a pending accumulation window"):
            validate(config, world_size=1)

    def test_a_single_whole_window_is_accepted(self):
        validate(tiny_config(target_elements=2, gradient_accumulate_every=2), world_size=1)


class TestTheShippedConfigsFillWholeAccumulationWindows:
    @pytest.mark.parametrize("config_path", SHIPPED_CONFIGS, ids=lambda path: path.name)
    @pytest.mark.parametrize("world_size", WORLD_SIZES)
    def test_a_shipped_config_starts_on_every_supported_world_size(
        self, config_path: Path, world_size: int
    ):
        config = load_yml(config_path, parse_to=TrainEntryConfig)
        trainer = MegabyteTrainer(config, run_vars=run_vars(world_size))
        accumulation = config.train.gradient_accumulate_every
        batch_iters = trainer.local_batch_iters()

        assert batch_iters >= accumulation
        assert batch_iters % accumulation == 0
        trainer._validate_config()
