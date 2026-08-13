"""
Automated unit tests verifying strict determinism and reproducibility.
Running the simulator with the same seed must produce identical package properties, trajectories, and annotations.
"""

import pytest

from barcode_simulator.core.config import SimulationConfig
from barcode_simulator.core.randomization import SeededRNG
from barcode_simulator.simulation.engine import SimulationEngine


def test_seed_reproducibility_packages():
    cfg1 = SimulationConfig()
    cfg1.simulation.seed = 42
    cfg1.simulation.duration_seconds = 2.0
    cfg1.packages.count = 5

    cfg2 = SimulationConfig()
    cfg2.simulation.seed = 42
    cfg2.simulation.duration_seconds = 2.0
    cfg2.packages.count = 5

    eng1 = SimulationEngine(config=cfg1, rng=SeededRNG(42))
    eng2 = SimulationEngine(config=cfg2, rng=SeededRNG(42))

    # Check identical packages generated
    assert len(eng1.packages) == len(eng2.packages)
    for p1, p2 in zip(eng1.packages, eng2.packages):
        assert p1.id == p2.id
        assert p1.width == p2.width
        assert p1.height == p2.height
        assert p1.data.barcode_id == p2.data.barcode_id
        assert p1.data.barcode_data.encoded_value == p2.data.barcode_data.encoded_value
        assert p1.data.barcode_data.barcode_type == p2.data.barcode_data.barcode_type
        assert p1.data.spawn_time == pytest.approx(p2.data.spawn_time)


def test_seed_reproducibility_frame_annotations():
    cfg1 = SimulationConfig()
    cfg1.simulation.seed = 1234
    cfg1.simulation.duration_seconds = 1.0
    cfg1.simulation.fps = 10
    cfg1.packages.count = 4

    cfg2 = SimulationConfig()
    cfg2.simulation.seed = 1234
    cfg2.simulation.duration_seconds = 1.0
    cfg2.simulation.fps = 10
    cfg2.packages.count = 4

    eng1 = SimulationEngine(config=cfg1, rng=SeededRNG(1234))
    eng2 = SimulationEngine(config=cfg2, rng=SeededRNG(1234))

    for frame_idx in range(10):
        _, scene1, anns1 = eng1.step(frame_idx)
        _, scene2, anns2 = eng2.step(frame_idx)

        assert len(anns1) == len(anns2)
        for a1, a2 in zip(anns1, anns2):
            assert a1.barcode_id == a2.barcode_id
            assert a1.barcode_value == a2.barcode_value
            assert a1.bbox == pytest.approx(a2.bbox, abs=1e-2)
            assert a1.visibility == pytest.approx(a2.visibility, abs=1e-3)
            assert a1.occlusion == pytest.approx(a2.occlusion, abs=1e-3)
