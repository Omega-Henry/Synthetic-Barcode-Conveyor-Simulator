"""
Configuration system for the Barcode Conveyor Simulator.
Uses Pydantic v2 for strict schema definition, validation, and serialization.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple, Union
import yaml
from pydantic import BaseModel, Field, field_validator, model_validator


class SimulationSettings(BaseModel):
    seed: Optional[int] = Field(default=None, description="Random seed for reproducibility. If None, random seed is generated.")
    duration_seconds: float = Field(default=10.0, ge=0.1, le=3600.0, description="Total duration of the simulation in seconds.")
    fps: int = Field(default=30, ge=1, le=120, description="Frames per second.")
    time_step: Optional[float] = Field(default=None, description="Internal physics/trajectory delta t. Defaults to 1/fps.")

    @model_validator(mode="after")
    def compute_defaults(self) -> SimulationSettings:
        if self.time_step is None:
            self.time_step = 1.0 / self.fps
        return self


class OutputSettings(BaseModel):
    output_dir: str = Field(default="outputs", description="Base output directory.")
    save_frames: bool = Field(default=True, description="Save individual frame images (e.g. 000001.jpg).")
    save_video: bool = Field(default=True, description="Encode and save MP4 video.")
    save_annotations: bool = Field(default=True, description="Save frame-by-frame ground truth jsonl.")
    save_coco: bool = Field(default=True, description="Export COCO format annotation dataset.")
    save_yolo: bool = Field(default=True, description="Export YOLO format annotation dataset.")
    save_debug: bool = Field(default=False, description="Render and save debug frames with visual overlays.")
    video_codec: str = Field(default="h264", description="Video codec for MP4 export ('h264', 'mp4v', 'xvid').")
    video_crf: int = Field(default=18, ge=0, le=51, description="FFmpeg Constant Rate Factor (lower = better quality).")
    video_bitrate: Optional[str] = Field(default="4M", description="Video bitrate fallback (e.g. '4M', '8M').")


class CameraSettings(BaseModel):
    width: int = Field(default=1280, ge=160, le=7680, description="Frame width in pixels.")
    height: int = Field(default=720, ge=120, le=4320, description="Frame height in pixels.")
    zoom: float = Field(default=1.0, ge=0.1, le=10.0, description="Camera zoom factor.")
    position_x: float = Field(default=0.5, description="Normalized camera center X [0.0 - 1.0].")
    position_y: float = Field(default=0.5, description="Normalized camera center Y [0.0 - 1.0].")
    exposure_compensation: float = Field(default=0.0, description="Exposure bias (-2.0 to 2.0).")


class ConveyorSettings(BaseModel):
    direction: Literal["left_to_right", "right_to_left"] = Field(
        default="left_to_right", description="Conveyor movement direction."
    )
    speed_min: float = Field(default=120.0, ge=0.0, description="Minimum conveyor speed in px/s.")
    speed_max: float = Field(default=180.0, ge=0.0, description="Maximum conveyor speed in px/s.")
    speed_distribution: Literal["uniform", "constant", "normal"] = Field(
        default="uniform", description="Speed sampling distribution per package."
    )
    acceleration: float = Field(default=0.0, description="Optional conveyor acceleration in px/s^2.")
    jitter_stddev: float = Field(default=0.0, ge=0.0, description="Trajectory jitter standard deviation.")
    belt_color: Tuple[int, int, int] = Field(default=(45, 45, 48), description="RGB conveyor belt base color.")
    lane_y_ratio: float = Field(default=0.5, ge=0.1, le=0.9, description="Center Y position of the conveyor lane.")
    lane_height_ratio: float = Field(default=0.7, ge=0.2, le=1.0, description="Height ratio of the conveyor belt on screen.")


class PackageSizeSettings(BaseModel):
    min_width: int = Field(default=140, ge=40, description="Minimum package width in pixels.")
    max_width: int = Field(default=280, ge=40, description="Maximum package width in pixels.")
    min_height: int = Field(default=100, ge=30, description="Minimum package height in pixels.")
    max_height: int = Field(default=220, ge=30, description="Maximum package height in pixels.")


class PackageSettings(BaseModel):
    count: int = Field(default=20, ge=1, description="Total number of packages to generate.")
    spawn_interval_min: float = Field(default=0.6, ge=0.05, description="Min seconds between package spawns.")
    spawn_interval_max: float = Field(default=1.8, ge=0.05, description="Max seconds between package spawns.")
    size: PackageSizeSettings = Field(default_factory=PackageSizeSettings)
    materials: Dict[str, float] = Field(
        default_factory=lambda: {
            "cardboard": 0.50,
            "white_carton": 0.25,
            "colored_carton": 0.15,
            "poly_mailer": 0.10,
        },
        description="Material distribution weights.",
    )
    add_shipping_labels: bool = Field(default=True, description="Add non-barcode shipping label graphics.")
    add_text_markings: bool = Field(default=True, description="Add printed non-barcode text (fragile, arrows, text).")
    tape_probability: float = Field(default=0.7, ge=0.0, le=1.0, description="Probability of adding packing tape.")


class BarcodeSettings(BaseModel):
    types: Dict[str, float] = Field(
        default_factory=lambda: {"CODE128": 0.6, "EAN13": 0.3, "QRCODE": 0.1},
        description="Probability distribution over barcode types (CODE128, EAN13, QRCODE, DATAMATRIX, etc.).",
    )
    human_readable_text: bool = Field(default=True, description="Include human readable alphanumeric text under bars.")
    scale_min: float = Field(default=0.6, ge=0.1, le=2.0, description="Minimum relative barcode scale.")
    scale_max: float = Field(default=1.0, ge=0.1, le=2.0, description="Maximum relative barcode scale.")
    rotation_min: float = Field(default=0.0, description="Min barcode rotation relative to package (deg).")
    rotation_max: float = Field(default=0.0, description="Max barcode rotation relative to package (deg).")
    margin_padding: int = Field(default=6, ge=0, description="White label quiet zone border in pixels.")
    allow_partial_clip: bool = Field(default=False, description="Allow barcode to overflow package boundaries.")


# Individual Effect Modules
class RotationEffect(BaseModel):
    enabled: bool = Field(default=True)
    min_degrees: float = Field(default=-15.0)
    max_degrees: float = Field(default=15.0)


class PerspectiveEffect(BaseModel):
    enabled: bool = Field(default=True)
    strength_min: float = Field(default=0.0, ge=0.0, le=1.0)
    strength_max: float = Field(default=0.15, ge=0.0, le=1.0)
    tilt_x_max: float = Field(default=12.0, description="Max tilt around X-axis in degrees.")
    tilt_y_max: float = Field(default=12.0, description="Max tilt around Y-axis in degrees.")


class MotionBlurEffect(BaseModel):
    enabled: bool = Field(default=True)
    factor: float = Field(default=0.04, ge=0.0, description="Kernel length multiplier = speed * factor.")
    min_kernel: int = Field(default=1, ge=1)
    max_kernel: int = Field(default=15, ge=1)


class GaussianBlurEffect(BaseModel):
    enabled: bool = Field(default=False)
    min_sigma: float = Field(default=0.5, ge=0.1)
    max_sigma: float = Field(default=2.5, ge=0.1)


class BrightnessEffect(BaseModel):
    enabled: bool = Field(default=True)
    min_factor: float = Field(default=0.8, ge=0.1)
    max_factor: float = Field(default=1.2, ge=0.1)


class ContrastEffect(BaseModel):
    enabled: bool = Field(default=True)
    min_factor: float = Field(default=0.85, ge=0.1)
    max_factor: float = Field(default=1.15, ge=0.1)


class GammaEffect(BaseModel):
    enabled: bool = Field(default=False)
    min_gamma: float = Field(default=0.8, ge=0.1)
    max_gamma: float = Field(default=1.2, ge=0.1)


class SensorNoiseEffect(BaseModel):
    enabled: bool = Field(default=True)
    stddev_min: float = Field(default=0.0, ge=0.0)
    stddev_max: float = Field(default=6.0, ge=0.0)


class JpegCompressionEffect(BaseModel):
    enabled: bool = Field(default=False)
    min_quality: int = Field(default=40, ge=10, le=100)
    max_quality: int = Field(default=95, ge=10, le=100)


class GlareEffect(BaseModel):
    enabled: bool = Field(default=True)
    probability: float = Field(default=0.2, ge=0.0, le=1.0)
    intensity_min: float = Field(default=0.2, ge=0.0, le=1.0)
    intensity_max: float = Field(default=0.7, ge=0.0, le=1.0)
    radius_min: int = Field(default=30, ge=5)
    radius_max: int = Field(default=120, ge=5)


class ShadowEffect(BaseModel):
    enabled: bool = Field(default=True)
    probability: float = Field(default=0.3, ge=0.0, le=1.0)
    darkness_min: float = Field(default=0.2, ge=0.0, le=0.9)
    darkness_max: float = Field(default=0.5, ge=0.0, le=0.9)


class OcclusionEffect(BaseModel):
    enabled: bool = Field(default=True)
    probability: float = Field(default=0.15, ge=0.0, le=1.0)
    max_occlusion_ratio: float = Field(default=0.6, ge=0.0, le=1.0)


class EffectsSettings(BaseModel):
    rotation: RotationEffect = Field(default_factory=RotationEffect)
    perspective: PerspectiveEffect = Field(default_factory=PerspectiveEffect)
    motion_blur: MotionBlurEffect = Field(default_factory=MotionBlurEffect)
    gaussian_blur: GaussianBlurEffect = Field(default_factory=GaussianBlurEffect)
    brightness: BrightnessEffect = Field(default_factory=BrightnessEffect)
    contrast: ContrastEffect = Field(default_factory=ContrastEffect)
    gamma: GammaEffect = Field(default_factory=GammaEffect)
    noise: SensorNoiseEffect = Field(default_factory=SensorNoiseEffect)
    jpeg_compression: JpegCompressionEffect = Field(default_factory=JpegCompressionEffect)
    glare: GlareEffect = Field(default_factory=GlareEffect)
    shadows: ShadowEffect = Field(default_factory=ShadowEffect)
    occlusion: OcclusionEffect = Field(default_factory=OcclusionEffect)


class SimulationConfig(BaseModel):
    simulation: SimulationSettings = Field(default_factory=SimulationSettings)
    output: OutputSettings = Field(default_factory=OutputSettings)
    camera: CameraSettings = Field(default_factory=CameraSettings)
    conveyor: ConveyorSettings = Field(default_factory=ConveyorSettings)
    packages: PackageSettings = Field(default_factory=PackageSettings)
    barcodes: BarcodeSettings = Field(default_factory=BarcodeSettings)
    effects: EffectsSettings = Field(default_factory=EffectsSettings)
    renderer: Literal["2d", "blender_3d"] = Field(default="2d", description="Renderer backend to use.")

    @classmethod
    def from_yaml(cls, path_or_yaml: Union[str, Path]) -> SimulationConfig:
        """Load configuration from a YAML file or string."""
        if isinstance(path_or_yaml, (str, Path)) and os.path.exists(str(path_or_yaml)):
            with open(path_or_yaml, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
        else:
            data = yaml.safe_load(str(path_or_yaml)) or {}
        return cls.model_validate(data)

    def to_yaml(self, path: Optional[Union[str, Path]] = None) -> str:
        """Serialize configuration to a clean YAML string or save to file."""
        dump_dict = self.model_dump(mode="json")
        yaml_str = yaml.dump(dump_dict, sort_keys=False, default_flow_style=False)
        if path:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                f.write(yaml_str)
        return yaml_str


def load_config(config_path: Union[str, Path]) -> SimulationConfig:
    return SimulationConfig.from_yaml(config_path)


def save_config(config: SimulationConfig, output_path: Union[str, Path]) -> None:
    config.to_yaml(output_path)
