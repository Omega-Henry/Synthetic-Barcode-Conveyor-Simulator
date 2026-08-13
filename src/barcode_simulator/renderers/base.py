"""
Abstract Renderer interface.
Ensures the simulation engine remains completely decoupled from rendering technology.
Both OpenCV 2D and future Blender 3D renderers implement this interface.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional
import numpy as np

from barcode_simulator.core.scene import SceneDescription


class Renderer(ABC):
    """Abstract renderer interface."""

    @abstractmethod
    def initialize(self, initial_scene: SceneDescription) -> None:
        """Initialize renderer assets, viewport canvas, or 3D engine scene graph."""
        pass

    @abstractmethod
    def render_frame(self, scene: SceneDescription) -> np.ndarray:
        """
        Render a single frame for the given scene description.
        Returns:
            RGB uint8 numpy array with shape (height, width, 3).
        """
        pass

    @abstractmethod
    def shutdown(self) -> None:
        """Release GPU contexts, renderer processes, or cached framebuffer resources."""
        pass
