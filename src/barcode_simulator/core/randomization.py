"""
Randomization manager ensuring deterministic reproducibility for all simulation stages.
"""

from __future__ import annotations

import random
from typing import Any, List, Optional, Sequence, Tuple, TypeVar
import numpy as np

T = TypeVar("T")


class SeededRNG:
    """Deterministic random number generator wrapper."""

    def __init__(self, seed: Optional[int] = None):
        if seed is None:
            # Generate a reproducible 32-bit positive integer seed
            seed = random.SystemRandom().randint(1, 2_147_483_647)
        self.seed = int(seed)
        self.np_rng = np.random.default_rng(self.seed)
        self.py_random = random.Random(self.seed)

    def spawn(self, sub_key: Optional[Union[str, int]] = None) -> SeededRNG:
        """Spawn an isolated deterministic child RNG based on current seed and sub_key."""
        if sub_key is None:
            child_seed = int(self.np_rng.integers(1, 2_147_483_647))
        elif isinstance(sub_key, int):
            child_seed = (self.seed * 31 + sub_key) % 2_147_483_647
        else:
            child_seed = (self.seed * 31 + abs(hash(sub_key))) % 2_147_483_647
        return SeededRNG(child_seed)

    def uniform(self, low: float, high: float) -> float:
        return float(self.np_rng.uniform(low, high))

    def integer(self, low: int, high: int) -> int:
        """Returns integer in [low, high] inclusive."""
        return int(self.np_rng.integers(low, high + 1))

    def choice(self, seq: Sequence[T]) -> T:
        idx = int(self.np_rng.integers(0, len(seq)))
        return seq[idx]

    def choices(self, population: Sequence[T], weights: Optional[Sequence[float]] = None, k: int = 1) -> List[T]:
        if weights is not None:
            probs = np.array(weights, dtype=np.float64)
            probs = probs / np.sum(probs)
            indices = self.np_rng.choice(len(population), size=k, p=probs)
            return [population[i] for i in indices]
        indices = self.np_rng.choice(len(population), size=k)
        return [population[i] for i in indices]

    def weighted_choice(self, choices_dict: dict[T, float]) -> T:
        keys = list(choices_dict.keys())
        weights = list(choices_dict.values())
        return self.choices(keys, weights=weights, k=1)[0]

    def normal(self, mean: float = 0.0, stddev: float = 1.0) -> float:
        return float(self.np_rng.normal(mean, stddev))

    def boolean(self, p: float = 0.5) -> bool:
        return float(self.np_rng.random()) < p
