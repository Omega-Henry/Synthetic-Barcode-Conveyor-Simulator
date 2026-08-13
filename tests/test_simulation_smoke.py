"""
End-to-end smoke test for complete simulation generation, frames, ground truth, and video encoding.
"""

import argparse
from pathlib import Path
import pytest

from barcode_simulator.cli import run_generate
from barcode_simulator.core.config import SimulationConfig
from barcode_simulator.utils.io import load_json, read_jsonl


def test_simulation_smoke_pipeline(tmp_path):
    # Tiny fast simulation
    cfg = SimulationConfig()
    cfg.simulation.seed = 42
    cfg.simulation.duration_seconds = 0.5  # 15 frames @ 30 fps
    cfg.simulation.fps = 30
    cfg.camera.width = 640
    cfg.camera.height = 360
    cfg.packages.count = 3
    cfg.output.output_dir = str(tmp_path)
    cfg.output.save_frames = True
    cfg.output.save_video = True
    cfg.output.save_debug = True
    cfg.output.save_coco = True
    cfg.output.save_yolo = True

    config_path = tmp_path / "smoke_config.yaml"
    cfg.to_yaml(config_path)

    args = argparse.Namespace(
        config=str(config_path),
        seed=42,
        output_dir=str(tmp_path),
        duration=None,
        fps=None,
        packages=None,
        debug=True,
        no_video=False,
        no_frames=False,
        benchmark_baseline=True,
    )

    run_dir = run_generate(args)
    assert run_dir.exists()

    # 1. Check manifest
    manifest_file = run_dir / "manifest.json"
    assert manifest_file.exists()
    manifest_data = load_json(manifest_file)
    assert manifest_data["seed"] == 42
    assert manifest_data["total_frames"] == 15
    assert manifest_data["number_of_packages"] == 3

    # 2. Check summary
    summary_file = run_dir / "summary.json"
    assert summary_file.exists()

    # 3. Check ground truth jsonl
    gt_file = run_dir / "ground_truth.jsonl"
    assert gt_file.exists()
    records = list(read_jsonl(gt_file))
    assert len(records) > 0

    # 4. Check frames
    frames_dir = run_dir / "frames"
    assert frames_dir.exists()
    frame_files = list(frames_dir.glob("*.jpg"))
    assert len(frame_files) == 15

    # 5. Check debug frames
    debug_dir = run_dir / "debug"
    assert debug_dir.exists()
    debug_files = list(debug_dir.glob("*.jpg"))
    assert len(debug_files) == 15

    # 6. Check COCO and YOLO
    coco_file = run_dir / "annotations" / "coco_annotations.json"
    assert coco_file.exists()

    yolo_yaml = run_dir / "annotations" / "yolo" / "dataset.yaml"
    assert yolo_yaml.exists()

    # 7. Check MP4 video
    video_file = run_dir / "video.mp4"
    assert video_file.exists()
    assert video_file.stat().st_size > 1000
