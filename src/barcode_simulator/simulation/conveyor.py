"""
Conveyor belt physical model and state management.
"""

from __future__ import annotations

from typing import Tuple
from barcode_simulator.core.config import ConveyorSettings
from barcode_simulator.core.scene import ConveyorState


class ConveyorBelt:
    """
    Simulates industrial conveyor belt parameters and moving texture offset.
    """

    def __init__(self, settings: ConveyorSettings, canvas_height: int):
        self.settings = settings
        self.canvas_height = canvas_height
        self.lane_y = canvas_height * settings.lane_y_ratio
        self.lane_height = canvas_height * settings.lane_height_ratio
        self.belt_color = settings.belt_color
        self.current_offset_x = 0.0

    def step(self, dt: float, mean_speed: float) -> None:
        """Advance conveyor belt surface offset."""
        direction_factor = 1.0 if self.settings.direction == "left_to_right" else -1.0
        self.current_offset_x += direction_factor * mean_speed * dt

    def get_state(self) -> ConveyorState:
        return ConveyorState(
            direction=self.settings.direction,
            lane_y=self.lane_y,
            lane_height=self.lane_height,
            belt_color=self.belt_color,
            current_offset_x=self.current_offset_x,
        )
