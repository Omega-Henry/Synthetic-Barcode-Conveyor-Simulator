"""
Renderer-agnostic Scene representation.
Enables full decoupling between simulation logic and visual rendering (OpenCV 2D or Blender 3D).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from barcode_simulator.core.models import BoundingBox, PackageData, Point2D, Polygon2D
from barcode_simulator.core.trajectory import Trajectory


@dataclass
class ConveyorState:
    direction: str
    lane_y: float
    lane_height: float
    belt_color: Tuple[int, int, int]
    current_offset_x: float = 0.0


@dataclass
class CameraState:
    width: int
    height: int
    zoom: float = 1.0
    center: Point2D = field(default_factory=lambda: Point2D(640.0, 360.0))
    exposure: float = 0.0


@dataclass
class ActivePackageInstance:
    package_data: PackageData
    trajectory: Trajectory
    current_position: Point2D
    current_velocity_px_s: float
    package_polygon: Polygon2D
    barcode_polygon: Polygon2D
    barcode_bounding_box: BoundingBox
    z_index: int = 0
    layer_occluded: bool = False
    rotation_deg: float = 0.0
    tilt_x_deg: float = 0.0
    tilt_y_deg: float = 0.0
    motion_blur_kernel: int = 1


@dataclass
class LightingState:
    ambient_brightness: float = 1.0
    contrast: float = 1.0
    gamma: float = 1.0
    sensor_noise_stddev: float = 0.0
    glare_sources: List[Dict[str, Any]] = field(default_factory=list)
    shadow_regions: List[Polygon2D] = field(default_factory=list)


@dataclass
class SceneDescription:
    frame_index: int
    time_seconds: float
    conveyor: ConveyorState
    camera: CameraState
    lighting: LightingState
    active_packages: List[ActivePackageInstance] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
