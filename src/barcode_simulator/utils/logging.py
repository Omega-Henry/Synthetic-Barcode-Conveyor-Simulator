"""
Logging and console reporting utilities.
"""

from __future__ import annotations

import logging
import sys
from typing import Any, Dict, Optional


def setup_logger(name: str = "barcode_simulator", level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(level)
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(level)
        formatter = logging.Formatter(
            fmt="%(asctime)s [%(levelname)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def print_run_summary(summary_data: Dict[str, Any]) -> None:
    """Print an industrial-grade formatted summary box to console."""
    run_id = summary_data.get("simulation_id", "RUN_UNKNOWN")
    renderer = summary_data.get("renderer", "2D").upper()
    seed = summary_data.get("seed", "N/A")
    resolution = summary_data.get("resolution", (1280, 720))
    fps = summary_data.get("fps", 30)
    duration = summary_data.get("duration_seconds", 0)
    packages = summary_data.get("number_of_packages", 0)
    barcodes = summary_data.get("number_of_barcodes", 0)
    barcode_types = summary_data.get("barcode_types", {})
    frames_rendered = summary_data.get("total_frames", 0)
    video_path = summary_data.get("video_path", "N/A")
    gt_path = summary_data.get("ground_truth_path", "N/A")
    manifest_path = summary_data.get("manifest_path", "N/A")

    lines = [
        "=" * 60,
        "          BARCODE CONVEYOR SIMULATION COMPLETE",
        "=" * 60,
        f"Run ID:              {run_id}",
        f"Renderer:            {renderer}",
        f"Seed:                {seed}",
        "",
        f"Resolution:          {resolution[0]}x{resolution[1]}",
        f"FPS:                 {fps}",
        f"Duration:            {duration:.1f} s",
        f"Frames Rendered:     {frames_rendered}",
        "",
        f"Total Packages:      {packages}",
        f"Total Barcodes:      {barcodes}",
    ]

    for btype, count in barcode_types.items():
        lines.append(f"  - {btype:<16}: {count}")

    lines.extend([
        "",
        "Outputs:",
        f"  Video:             {video_path}",
        f"  Ground Truth:      {gt_path}",
        f"  Manifest:          {manifest_path}",
        "=" * 60,
    ])

    print("\n" + "\n".join(lines) + "\n")
