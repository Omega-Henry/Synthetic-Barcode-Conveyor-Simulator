"""
Virtual camera model for frame capture and viewport mapping.
"""

from __future__ import annotations

from barcode_simulator.core.config import CameraSettings
from barcode_simulator.core.models import Point2D
from barcode_simulator.core.scene import CameraState


class VirtualCamera:
    """
    Simulates camera viewport, resolution, and exposure settings.
    """

    def __init__(self, settings: CameraSettings):
        self.settings = settings
        self.width = settings.width
        self.height = settings.height
        self.zoom = settings.zoom
        self.center = Point2D(
            settings.width * settings.position_x,
            settings.height * settings.position_y,
        )
        self.exposure = settings.exposure_compensation

    def get_state(self) -> CameraState:
        return CameraState(
            width=self.width,
            height=self.height,
            zoom=self.zoom,
            center=self.center,
            elevation_angle_deg=self.settings.elevation_angle_deg,
            enable_3d_cuboid=self.settings.enable_3d_cuboid,
            exposure=self.exposure,
        )
