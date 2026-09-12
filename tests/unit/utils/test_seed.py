import random

import numpy as np
import torch

from mblm.utils.seed import seed_run


def draws() -> tuple[float, float, float]:
    return (random.random(), float(torch.rand(1)), float(np.random.rand()))


class TestSeedRun:
    def test_without_a_base_seed_the_rng_streams_stay_untouched(self):
        seed_run(3, rank=0)
        assert seed_run(None, rank=0) is None
        with_unseeded_call = draws()

        seed_run(3, rank=0)
        without_unseeded_call = draws()

        assert with_unseeded_call == without_unseeded_call

    def test_the_same_base_seed_and_rank_reproduce_the_same_draws(self):
        first = seed_run(7, rank=0)
        first_draws = draws()

        second = seed_run(7, rank=0)
        second_draws = draws()

        assert first == second == 7
        assert second_draws == first_draws

    def test_every_rank_derives_its_own_effective_seed(self):
        effective_seeds = [seed_run(7, rank=rank) for rank in range(4)]

        assert effective_seeds == [7, 8, 9, 10]

    def test_ranks_do_not_share_one_rng_stream(self):
        seed_run(7, rank=0)
        rank_0_draw = random.random()

        seed_run(7, rank=1)
        rank_1_draw = random.random()

        assert rank_0_draw != rank_1_draw
