# Synthetic Industrial Barcode Conveyor-Belt Video Generation System

A modular, scientifically reproducible synthetic video and dataset generation framework for developing, testing, benchmarking, and training barcode-scanning and computer-vision algorithms in industrial logistics environments.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Architecture: Extensible](https://img.shields.io/badge/Renderer-OpenCV%202D%20%7C%20Blender%203D-green.svg)](#future-blender-3d-integration)

---

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Architecture & Decoupling Principle](#architecture--decoupling-principle)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [CLI Reference](#cli-reference)
- [Configuration System](#configuration-system)
- [Difficulty Presets](#difficulty-presets)
- [Dataset & Annotation Formats](#dataset--annotation-formats)
- [External Scanner Integration & Evaluation](#external-scanner-integration--evaluation)
- [Parameter Sweep Experiments (Thesis Support)](#parameter-sweep-experiments-thesis-support)
- [Future Blender 3D Integration](#future-blender-3d-integration)
- [Running Automated Tests](#running-automated-tests)
- [License](#license)

---

## Overview

In industrial warehouse automation, barcode scanners operate on conveyor belts under challenging real-world optical conditions: motion blur from belt speed, non-planar perspective angles, varied box materials, variable lighting, camera sensor noise, and multi-package occlusions.

This simulator generates high-fidelity synthetic conveyor-belt video sequences with **exact sub-pixel ground truth** (4-corner polygons, bounding boxes, visibility percentage, occlusion fraction, blur parameters, rotation angles). All barcodes are **mathematically and programmatically generated** (Code 128, EAN-13 with valid checksums, QR codes) with zero generative AI hallucinations.

---

## Key Features

1. **Mathematically Valid Barcodes**:
   - **Code 128**: High-density alphanumeric industrial barcodes with quiet zones.
   - **EAN-13**: Standard 13-digit GS1 barcodes with exact modulo-10 check digit verification.
   - **QR Code**: ISO/IEC 18004 2D matrix symbology with Reed-Solomon error correction.
   - Configurable human-readable text labels and quiet zones.
2. **Procedural Package Synthesis**:
   - Realistic cardboard corrugation and fiber grain textures.
   - White cartons, colored cartons, poly mailers, kraft paper.
   - Packing tape seams, shipping labels, and printed handling markings (FRAGILE, orientation arrows).
3. **Parametric Visual Domain Randomization**:
   - **Geometric**: Rotation, scale, translation, 4-point 3D perspective quadrilateral warp, skew.
   - **Motion Blur**: Directional velocity-vector kernel ($L \propto v$).
   - **Photometric & Sensor**: Gaussian blur, brightness, contrast, gamma, sensor noise, lossy JPEG compression artifacts.
   - **Environment**: Cast shadows, moving industrial spotlight glare reflections, animated conveyor belt rollers.
   - **Occlusion**: Multi-package overlap, viewport boundary clipping, exact visibility calculation via Sutherland-Hodgman polygon clipping.
4. **Comprehensive Ground Truth**:
   - Per-frame JSON Lines (`ground_truth.jsonl`) with exact polygon coordinates, visibility $[0.0, 1.0]$, and difficulty tags.
   - Automatic exporter for **COCO JSON** (instance segmentation / detection) and **YOLO txt** formats.
   - Optional visual debug overlay renderer.
5. **Video Encoding**:
   - Broadcast-quality H.264 MP4 export via FFmpeg with automatic fallback to OpenCV VideoWriter.
6. **Scientific Rigor & Thesis Support**:
   - Strict seed-based determinism and reproducibility.
   - Built-in parameter sweep runner and evaluation framework with stratified metrics.

---

## Architecture & Decoupling Principle

Simulation logic is **completely independent from rendering technology**:

```text
                  Simulation Configuration (YAML / Pydantic)
                                      |
                                      v
                             Scenario Generator
                                      |
                  +-------------------+-------------------+
                  |                                       |
                  v                                       v
         Package / Barcodes                      Conveyor / Camera
             Generator                               Generator
                  |                                       |
                  +-------------------+-------------------+
                                      |
                                      v
                             SceneDescription
                         (Renderer-Agnostic State)
                                      |
                         +------------+------------+
                         |                         |
                         v                         v
                  OpenCV2DRenderer          BlenderRenderer (3D)
                   (Implemented)              (Roadmap Design)
                         |                         |
                         +------------+------------+
                                      |
                                      v
                                 RGB Frames
                                      |
                  +-------------------+-------------------+
                  |                                       |
                  v                                       v
            Video Encoder                           Ground Truth
             (MP4 H.264)                         (JSONL/COCO/YOLO)
                  |                                       |
                  +-------------------+-------------------+
                                      |
                                      v
                              Evaluation Adapter
                                      |
                                      v
                          External Barcode Scanner
```

The renderer consumes a generic `SceneDescription` at time $t$ via the abstract `Renderer` interface (`initialize`, `render_frame`, `shutdown`). Replacing the 2D renderer with Blender 3D requires zero changes to the simulation core.

---

## Installation

### Prerequisites
- Python 3.10+
- (Optional but recommended) FFmpeg on system PATH for fast H.264 encoding.

### Setup Virtual Environment

**Windows (PowerShell):**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e .
```

**Linux / macOS:**
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

---

## Quick Start

### 1. Generate Your First Video (Easy Preset)
```bash
python -m barcode_simulator generate --config configs/easy.yaml --seed 42 --debug
```

### 2. Generate Realistic Medium Conveyor Video
```bash
python -m barcode_simulator generate --config configs/medium.yaml
```

### 3. Generate Stress-Test Hard Video
```bash
python -m barcode_simulator generate --config configs/hard.yaml
```

Outputs are automatically organized into:
```text
outputs/
└── run_20260813_230000_seed_42/
    ├── video.mp4               # Encoded H.264 MP4 video
    ├── config.yaml             # Exact configuration used for this run
    ├── manifest.json           # Simulation metadata manifest
    ├── ground_truth.jsonl      # Frame-by-frame exact annotations
    ├── summary.json            # Run summary & performance metrics
    ├── frames/                 # Individual frame JPEGs (000001.jpg, ...)
    ├── annotations/
    │   ├── coco_annotations.json
    │   └── yolo/
    │       ├── dataset.yaml
    │       ├── classes.txt
    │       └── labels/ (000001.txt, ...)
    └── debug/                  # Visual ground-truth overlay frames
```

---

## CLI Reference

### `generate`
Generate synthetic conveyor video and dataset.
```bash
python -m barcode_simulator generate [OPTIONS]

Options:
  -c, --config PATH          Path to YAML configuration file.
  -s, --seed INT             Random seed for reproducibility.
  -o, --output-dir PATH      Output destination folder.
  -d, --duration FLOAT       Simulation duration in seconds.
  --fps INT                  Frames per second (e.g. 30, 60).
  -p, --packages INT         Total number of packages.
  --debug                    Render visual ground-truth debug overlays.
  --no-video                 Skip MP4 video encoding.
  --no-frames                Skip saving individual frame images.
  --benchmark-baseline       Run baseline scanner evaluation during generation.
```

### `experiment`
Run systematic parameter sweeps for scientific thesis research.
```bash
python -m barcode_simulator experiment \
    --config configs/thesis_example.yaml \
    --parameter conveyor.speed_min \
    --values 80,140,200,260,320
```

### `evaluate`
Evaluate external scanner predictions against ground truth.
```bash
python -m barcode_simulator evaluate \
    --ground-truth outputs/run_.../ground_truth.jsonl \
    --predictions predictions.json \
    --output evaluation_summary.json
```

---

## Configuration System

All simulation parameters live in clean, validated YAML configuration files:

```yaml
simulation:
  seed: 42
  duration_seconds: 12.0
  fps: 30

camera:
  width: 1280
  height: 720
  zoom: 1.0

conveyor:
  direction: "left_to_right"   # "left_to_right" or "right_to_left"
  speed_min: 130.0             # Minimum belt velocity (px/s)
  speed_max: 190.0             # Maximum belt velocity (px/s)
  speed_distribution: "uniform" # "uniform", "constant", "normal"
  acceleration: 0.0            # Optional belt acceleration (px/s^2)
  jitter_stddev: 0.8           # Random trajectory jitter
  belt_color: [45, 45, 48]     # RGB color of conveyor belt

packages:
  count: 18
  spawn_interval_min: 0.6      # Seconds between package arrivals
  spawn_interval_max: 1.4
  size:
    min_width: 130
    max_width: 250
    min_height: 90
    max_height: 190
  materials:
    cardboard: 0.45
    white_carton: 0.30
    colored_carton: 0.15
    poly_mailer: 0.10

barcodes:
  types:
    CODE128: 0.55
    EAN13: 0.35
    QRCODE: 0.10
  human_readable_text: true    # Print text value below bars
  scale_min: 0.55              # Relative scale on package face
  scale_max: 0.90
  rotation_min: -15.0          # Relative rotation on package (deg)
  rotation_max: 15.0
  margin_padding: 6            # Quiet zone border in pixels

effects:
  rotation:
    enabled: true
    min_degrees: -18.0
    max_degrees: 18.0
  perspective:
    enabled: true
    strength_min: 0.02
    strength_max: 0.10
    tilt_x_max: 10.0
    tilt_y_max: 10.0
  motion_blur:
    enabled: true
    factor: 0.035              # Kernel size = speed * factor
    min_kernel: 1
    max_kernel: 9
  brightness:
    enabled: true
    min_factor: 0.85
    max_factor: 1.15
  contrast:
    enabled: true
    min_factor: 0.85
    max_factor: 1.15
  noise:
    enabled: true
    stddev_min: 1.0
    stddev_max: 4.0
  glare:
    enabled: true
    probability: 0.20
    intensity_min: 0.2
    intensity_max: 0.6
  occlusion:
    enabled: true
    probability: 0.15
    max_occlusion_ratio: 0.35
```

---

## Difficulty Presets

| Parameter | Easy (`configs/easy.yaml`) | Medium (`configs/medium.yaml`) | Hard (`configs/hard.yaml`) |
| :--- | :--- | :--- | :--- |
| **Belt Speed** | $70 - 100\text{ px/s}$ | $130 - 190\text{ px/s}$ | $220 - 340\text{ px/s}$ |
| **Package Density** | 8 pkgs, long intervals | 18 pkgs, moderate | 30 pkgs, dense overlap |
| **Barcode Scale** | $0.80 - 1.00$ | $0.55 - 0.90$ | $0.35 - 0.75$ |
| **Package Rotation** | $\pm 5^\circ$ | $\pm 18^\circ$ | $\pm 35^\circ$ |
| **3D Perspective Tilt** | Minimal ($\le 3^\circ$) | Moderate ($\le 10^\circ$) | Strong ($\le 18^\circ$) |
| **Motion Blur Kernel** | Disabled / 1px | $1 - 9\text{ px}$ | $3 - 17\text{ px}$ |
| **Illumination** | Clean ($0.95 - 1.05$) | Varied ($0.85 - 1.15$) | Adverse ($0.65 - 1.35$) |
| **Sensor Noise** | Disabled | $1.0 - 4.0\text{ stddev}$ | $3.0 - 9.0\text{ stddev}$ |
| **JPEG Artifacts** | Disabled | Disabled | Lossy (Quality $45 - 75$) |
| **Occlusions** | None ($0\%$) | $15\%$ prob, up to $35\%$ | $30\%$ prob, up to $65\%$ |

---

## Dataset & Annotation Formats

### 1. Frame Ground Truth (`ground_truth.jsonl`)
Every visible or partially visible barcode in every frame is recorded:
```json
{
  "frame_id": 142,
  "timestamp": 4.7333,
  "package_id": "PKG_000004",
  "barcode_id": "BC_000004",
  "barcode_type": "CODE128",
  "barcode_value": "PKG-000004",
  "bbox": [534.2, 285.6, 712.8, 388.1],
  "polygon": [
    [540.1, 290.4],
    [712.8, 304.2],
    [705.3, 388.1],
    [534.2, 372.5]
  ],
  "visibility": 0.942,
  "occlusion": 0.058,
  "rotation_degrees": 6.8,
  "motion_blur": 5,
  "velocity_px_s": 155.0,
  "is_in_frame": true,
  "difficulty_tags": ["perspective_tilt", "high_motion_blur"]
}
```

### 2. COCO Format (`annotations/coco_annotations.json`)
Compatible with detectron2, MMDetection, and standard object detection / instance segmentation pipelines.

### 3. YOLO Format (`annotations/yolo/`)
Contains `dataset.yaml`, `classes.txt`, and per-frame label files (`000001.txt`, ...) with normalized `[class_id x_center y_center width height]`.

---

## External Scanner Integration & Evaluation

The system provides a clean Python interface for benchmarking external algorithms:

```python
from barcode_simulator.core.config import load_config
from barcode_simulator.simulation.engine import SimulationEngine
from barcode_simulator.evaluation.adapter import ScannerAdapter, ScanResult
from barcode_simulator.evaluation.metrics import EvaluationMetrics

class MyCustomScanner(ScannerAdapter):
    def scan(self, frame_rgb) -> list[ScanResult]:
        # Your custom detector & decoder model here
        return [ScanResult(value="PKG-000001", confidence=0.98)]

config = load_config("configs/medium.yaml")
engine = SimulationEngine(config)
my_scanner = MyCustomScanner()
evaluator = EvaluationMetrics(iou_threshold=0.5)

for frame_rgb, ground_truth in engine:
    predictions = my_scanner.scan(frame_rgb)
    evaluator.update(predictions, ground_truth)

summary = evaluator.compute_summary()
print(f"Overall Recall: {summary['recall']:.2%}")
print(f"Precision:      {summary['precision']:.2%}")
print(f"Stratified Breakdown by Difficulty: {summary['by_difficulty_tag']}")
```

---

## Parameter Sweep Experiments (Thesis Support)

Easily study the scientific impact of individual environmental variables (e.g., speed, camera angle, motion blur, barcode scale) on scanner decoding accuracy:

```bash
python -m barcode_simulator experiment \
    --config configs/thesis_example.yaml \
    --parameter conveyor.speed_min \
    --values 80,140,200,260,320
```

Produces an `experiment_summary.json` mapping parameter values to detection recall, decoding accuracy, and mean IoU.

---

## Future Blender 3D Integration

The simulation core outputs a renderer-agnostic `SceneDescription`. See [docs/BLENDER_ROADMAP.md](docs/BLENDER_ROADMAP.md) for the complete design specification covering:
- Procedural 3D cuboid meshes with bevel modifiers.
- Principled BSDF procedural materials (cardboard, paper, plastic).
- Barcode image texture projection decals.
- Cycles raytraced lighting, specular glare, and motion blur.
- Exact 2D bounding box and polygon projection via $P \cdot V \cdot M$ matrices.

---

## Running Automated Tests

Run the complete test suite:
```bash
pytest -v
```

Tests cover:
- EAN-13 modulo-10 check digits & Code 128 formatting.
- Sutherland-Hodgman polygon clipping & exact visibility calculations.
- Strict seed reproducibility.
- Pydantic configuration validation.
- End-to-end rendering pipeline smoke test.
- Evaluation metrics calculation.

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
