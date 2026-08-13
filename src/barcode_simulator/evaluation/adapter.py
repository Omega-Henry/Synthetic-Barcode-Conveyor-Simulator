"""
Scanner Adapter interface for integrating external computer vision and barcode decoding models.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import numpy as np

from barcode_simulator.core.models import BoundingBox, Polygon2D


@dataclass
class ScanResult:
    """Standardized output of a barcode scanner / detection model."""
    value: str
    barcode_type: Optional[str] = None
    bbox: Optional[BoundingBox] = None
    polygon: Optional[Polygon2D] = None
    confidence: float = 1.0
    latency_ms: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "value": self.value,
            "barcode_type": self.barcode_type,
            "bbox": self.bbox.to_list() if self.bbox else None,
            "polygon": self.polygon.to_list() if self.polygon else None,
            "confidence": round(self.confidence, 4),
            "latency_ms": round(self.latency_ms, 2),
        }


class ScannerAdapter(ABC):
    """
    Abstract interface for any external barcode scanner.
    Decoupled from simulation logic so external research algorithms can be plugged in directly.
    """

    @abstractmethod
    def scan(self, frame_rgb: np.ndarray) -> List[ScanResult]:
        """
        Scan a single RGB video frame.
        Returns:
            List of detected and decoded ScanResult objects.
        """
        pass
