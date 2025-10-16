"""Deterministic seeding utilities."""
from __future__ import annotations

import os
import random
from contextlib import contextmanager
from typing import Iterator, Optional

import numpy as np
import torch


def set_seed(seed: int, deterministic: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


@contextmanager
def temp_seed(seed: int, deterministic: bool = True) -> Iterator[None]:
    state = random.getstate()
    np_state = np.random.get_state()
    torch_state = torch.random.get_rng_state()
    cuda_state = torch.cuda.get_rng_state_all()

    set_seed(seed, deterministic=deterministic)
    try:
        yield
    finally:
        random.setstate(state)
        np.random.set_state(np_state)
        torch.random.set_rng_state(torch_state)
        torch.cuda.set_rng_state_all(cuda_state)


__all__ = ["set_seed", "temp_seed"]
