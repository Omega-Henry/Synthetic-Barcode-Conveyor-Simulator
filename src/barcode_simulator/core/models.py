"""
Domain models and data structures for the Barcode Conveyor Simulator.
Designed to be independent of rendering technology (OpenCV, Blender, etc.).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import numpy as np


class BarcodeType(str, Enum):
    CODE128 = "CODE128"
    EAN13 = "EAN13"
    EAN8 = "EAN8"
    UPCA = "UPCA"
    QRCODE = "QRCODE"
    DATAMATRIX = "DATAMATRIX"


class PackageMaterial(str, Enum):
    CARDBOARD = "cardboard"
    WHITE_CARTON = "white_carton"
    COLORED_CARTON = "colored_carton"
    POLY_MAILER = "poly_mailer"
    KRAFT_PAPER = "kraft_paper"


@dataclass
class Point2D:
    x: float
    y: float

    def to_tuple(self) -> Tuple[float, float]:
        return (self.x, self.y)

    def to_int_tuple(self) -> Tuple[int, int]:
        return (int(round(self.x)), int(round(self.y)))

    def to_list(self) -> List[float]:
        return [self.x, self.y]

    def distance_to(self, other: Point2D) -> float:
        return math.hypot(self.x - other.x, self.y - other.y)

    def __add__(self, other: Union[Point2D, Tuple[float, float]]) -> Point2D:
        if isinstance(other, Point2D):
            return Point2D(self.x + other.x, self.y + other.y)
        return Point2D(self.x + other[0], self.y + other[1])

    def __sub__(self, other: Union[Point2D, Tuple[float, float]]) -> Point2D:
        if isinstance(other, Point2D):
            return Point2D(self.x - other.x, self.y - other.y)
        return Point2D(self.x - other[0], self.y - other[1])


@dataclass
class BoundingBox:
    xmin: float
    ymin: float
    xmax: float
    ymax: float

    @property
    def width(self) -> float:
        return max(0.0, self.xmax - self.xmin)

    @property
    def height(self) -> float:
        return max(0.0, self.ymax - self.ymin)

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def center(self) -> Point2D:
        return Point2D((self.xmin + self.xmax) / 2.0, (self.ymin + self.ymax) / 2.0)

    def to_list(self) -> List[float]:
        return [self.xmin, self.ymin, self.xmax, self.ymax]

    def to_int_list(self) -> List[int]:
        return [
            int(round(self.xmin)),
            int(round(self.ymin)),
            int(round(self.xmax)),
            int(round(self.ymax)),
        ]

    def to_coco(self) -> List[float]:
        """[x_min, y_min, width, height] format."""
        return [self.xmin, self.ymin, self.width, self.height]

    def to_yolo(self, img_width: float, img_height: float) -> List[float]:
        """[x_center_norm, y_center_norm, width_norm, height_norm] format."""
        x_center = (self.xmin + self.xmax) / (2.0 * img_width)
        y_center = (self.ymin + self.ymax) / (2.0 * img_height)
        norm_w = self.width / img_width
        norm_h = self.height / img_height
        return [
            float(np.clip(x_center, 0.0, 1.0)),
            float(np.clip(y_center, 0.0, 1.0)),
            float(np.clip(norm_w, 0.0, 1.0)),
            float(np.clip(norm_h, 0.0, 1.0)),
        ]

    def iou(self, other: BoundingBox) -> float:
        ixmin = max(self.xmin, other.xmin)
        iymin = max(self.ymin, other.ymin)
        ixmax = min(self.xmax, other.xmax)
        iymax = min(self.ymax, other.ymax)

        iw = max(0.0, ixmax - ixmin)
        ih = max(0.0, iymax - iymin)
        intersection = iw * ih

        union = self.area + other.area - intersection
        if union <= 0.0:
            return 0.0
        return intersection / union

    def clip_to_frame(self, width: float, height: float) -> Optional[BoundingBox]:
        xmin = max(0.0, min(width, self.xmin))
        ymin = max(0.0, min(height, self.ymin))
        xmax = max(0.0, min(width, self.xmax))
        ymax = max(0.0, min(height, self.ymax))
        if xmax <= xmin or ymax <= ymin:
            return None
        return BoundingBox(xmin, ymin, xmax, ymax)


@dataclass
class Polygon2D:
    vertices: List[Point2D]

    @classmethod
    def from_array(cls, points: Sequence[Sequence[float]]) -> Polygon2D:
        return cls([Point2D(float(p[0]), float(p[1])) for p in points])

    def to_list(self) -> List[List[float]]:
        return [v.to_list() for v in self.vertices]

    def to_int_list(self) -> List[List[int]]:
        return [v.to_int_tuple() for v in self.vertices]

    def to_numpy(self) -> np.ndarray:
        return np.array([[v.x, v.y] for v in self.vertices], dtype=np.float32)

    def to_flat_coco(self) -> List[float]:
        """Flat list of coordinates [x1, y1, x2, y2, ...] for COCO segmentation."""
        flat = []
        for v in self.vertices:
            flat.extend([v.x, v.y])
        return flat

    @property
    def bounding_box(self) -> BoundingBox:
        if not self.vertices:
            return BoundingBox(0.0, 0.0, 0.0, 0.0)
        xs = [v.x for v in self.vertices]
        ys = [v.y for v in self.vertices]
        return BoundingBox(min(xs), min(ys), max(xs), max(ys))

    @property
    def area(self) -> float:
        """Shoelace formula for polygon area."""
        n = len(self.vertices)
        if n < 3:
            return 0.0
        area = 0.0
        for i in range(n):
            j = (i + 1) % n
            area += self.vertices[i].x * self.vertices[j].y
            area -= self.vertices[j].x * self.vertices[i].y
        return abs(area) / 2.0

    @property
    def center(self) -> Point2D:
        if not self.vertices:
            return Point2D(0.0, 0.0)
        xs = [v.x for v in self.vertices]
        ys = [v.y for v in self.vertices]
        return Point2D(sum(xs) / len(xs), sum(ys) / len(ys))


@dataclass
class BarcodeData:
    barcode_id: str
    package_id: str
    barcode_type: BarcodeType
    encoded_value: str
    image_path: Optional[str] = None
    width_px: int = 240
    height_px: int = 120
    human_readable_text: bool = True
    quiet_zone_px: int = 10
    image: Optional[np.ndarray] = None  # RGB / RGBA uint8 numpy array

    def to_dict(self) -> Dict[str, Any]:
        return {
            "barcode_id": self.barcode_id,
            "package_id": self.package_id,
            "barcode_type": self.barcode_type.value,
            "encoded_value": self.encoded_value,
            "image_path": self.image_path,
            "width_px": self.width_px,
            "height_px": self.height_px,
            "human_readable_text": self.human_readable_text,
        }


@dataclass
class PackageData:
    package_id: str
    width: float  # In canvas / physical coordinate units
    height: float
    depth: float = 0.2  # For 3D Blender representation
    material: PackageMaterial = PackageMaterial.CARDBOARD
    color: Tuple[int, int, int] = (185, 145, 100)  # Base RGB
    barcode_id: Optional[str] = None
    barcode_data: Optional[BarcodeData] = None
    barcode_rel_x: float = 0.5  # Relative normalized pos [0.0, 1.0] on package
    barcode_rel_y: float = 0.5
    barcode_rel_scale: float = 0.8  # Scale factor relative to package space
    barcode_rotation_deg: float = 0.0
    has_tape: bool = True
    has_labels: bool = True
    has_text: bool = True
    spawn_time: float = 0.0
    trajectory_id: Optional[str] = None
    texture_image: Optional[np.ndarray] = None  # Cached package face texture

    def to_dict(self) -> Dict[str, Any]:
        return {
            "package_id": self.package_id,
            "width": self.width,
            "height": self.height,
            "depth": self.depth,
            "material": self.material.value,
            "color": list(self.color),
            "barcode_id": self.barcode_id,
            "barcode_rel_x": self.barcode_rel_x,
            "barcode_rel_y": self.barcode_rel_y,
            "barcode_rel_scale": self.barcode_rel_scale,
            "barcode_rotation_deg": self.barcode_rotation_deg,
            "spawn_time": self.spawn_time,
        }


@dataclass
class FrameAnnotation:
    frame_id: int
    timestamp: float
    package_id: str
    barcode_id: str
    barcode_type: str
    barcode_value: str
    bbox: List[float]  # [xmin, ymin, xmax, ymax]
    polygon: List[List[float]]  # [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
    visibility: float  # [0.0 - 1.0], 1.0 = completely visible, 0.0 = completely occluded/offscreen
    occlusion: float  # [0.0 - 1.0], 1.0 - visibility
    rotation_degrees: float
    motion_blur_kernel: int
    velocity_px_s: float
    is_in_frame: bool = True
    difficulty_tags: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "frame_id": self.frame_id,
            "timestamp": round(self.timestamp, 4),
            "package_id": self.package_id,
            "barcode_id": self.barcode_id,
            "barcode_type": self.barcode_type,
            "barcode_value": self.barcode_value,
            "bbox": [round(c, 2) for c in self.bbox],
            "polygon": [[round(p[0], 2), round(p[1], 2)] for p in self.polygon],
            "visibility": round(self.visibility, 3),
            "occlusion": round(self.occlusion, 3),
            "rotation_degrees": round(self.rotation_degrees, 2),
            "motion_blur": self.motion_blur_kernel,
            "velocity_px_s": round(self.velocity_px_s, 2),
            "is_in_frame": self.is_in_frame,
            "difficulty_tags": self.difficulty_tags,
        }


@dataclass
class GroundTruthManifest:
    simulation_id: str
    seed: int
    created_at: str
    simulator_version: str
    renderer: str
    resolution: Tuple[int, int]
    fps: int
    duration_seconds: float
    total_frames: int
    number_of_packages: int
    number_of_barcodes: int
    barcode_types: Dict[str, int]
    materials: Dict[str, int]
    parameters: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "simulation_id": self.simulation_id,
            "seed": self.seed,
            "created_at": self.created_at,
            "simulator_version": self.simulator_version,
            "renderer": self.renderer,
            "resolution": list(self.resolution),
            "fps": self.fps,
            "duration_seconds": self.duration_seconds,
            "total_frames": self.total_frames,
            "number_of_packages": self.number_of_packages,
            "number_of_barcodes": self.number_of_barcodes,
            "barcode_types": self.barcode_types,
            "materials": self.materials,
            "parameters": self.parameters,
        }
