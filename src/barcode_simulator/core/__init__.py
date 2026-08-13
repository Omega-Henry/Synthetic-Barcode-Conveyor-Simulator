"""
Core data structures, configurations, trajectories, and scene abstractions.
"""

from barcode_simulator.core.config import SimulationConfig, load_config, save_config
from barcode_simulator.core.models import (
    Point2D,
    Polygon2D,
    BoundingBox,
    BarcodeData,
    PackageData,
    FrameAnnotation,
    GroundTruthManifest,
)
from barcode_simulator.core.randomization import SeededRNG
from barcode_simulator.core.scene import SceneDescription
from barcode_simulator.core.trajectory import LinearTrajectory, Trajectory

__all__ = [
    "SimulationConfig",
    "load_config",
    "save_config",
    "Point2D",
    "Polygon2D",
    "BoundingBox",
    "BarcodeData",
    "PackageData",
    "FrameAnnotation",
    "GroundTruthManifest",
    "SeededRNG",
    "SceneDescription",
    "LinearTrajectory",
    "Trajectory",
]
