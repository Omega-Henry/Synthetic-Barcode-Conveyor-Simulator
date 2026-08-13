"""
Automated unit tests for configuration loading, validation, and serialization.
"""

from pathlib import Path
import pytest
from pydantic import ValidationError

from barcode_simulator.core.config import SimulationConfig, load_config, save_config


def test_default_config_validity():
    cfg = SimulationConfig()
    assert cfg.camera.width == 1280
    assert cfg.camera.height == 720
    assert cfg.simulation.fps == 30
    assert cfg.simulation.time_step == pytest.approx(1.0 / 30)


def test_yaml_presets_loading():
    for preset_name in ["easy.yaml", "medium.yaml", "hard.yaml", "thesis_example.yaml"]:
        preset_path = Path("configs") / preset_name
        assert preset_path.exists(), f"Missing config preset: {preset_path}"
        cfg = load_config(preset_path)
        assert cfg.simulation.fps > 0
        assert cfg.camera.width > 0
        assert cfg.packages.count > 0


def test_config_invalid_constraints_raise():
    # Negative duration
    with pytest.raises(ValidationError):
        SimulationConfig(simulation={"duration_seconds": -5.0})

    # FPS out of range
    with pytest.raises(ValidationError):
        SimulationConfig(simulation={"fps": 0})

    # Resolution out of range
    with pytest.raises(ValidationError):
        SimulationConfig(camera={"width": 50})


def test_config_yaml_roundtrip(tmp_path):
    cfg1 = SimulationConfig()
    cfg1.simulation.seed = 999
    cfg1.conveyor.speed_min = 250.0
    cfg1.effects.rotation.min_degrees = -30.0

    save_path = tmp_path / "test_roundtrip.yaml"
    cfg1.to_yaml(save_path)

    cfg2 = load_config(save_path)
    assert cfg2.simulation.seed == 999
    assert cfg2.conveyor.speed_min == 250.0
    assert cfg2.effects.rotation.min_degrees == -30.0
