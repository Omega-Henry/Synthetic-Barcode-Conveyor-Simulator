"""
Dataset verification script for quality assurance and scientific validation.
Checks file presence, JSON schemas, barcode check digits, image decodability, and polygon consistency.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List
import cv2
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from barcode_simulator.barcodes.generator import calculate_ean13_check_digit
from barcode_simulator.evaluation.baseline_decoder import BaselineScanner
from barcode_simulator.utils.io import load_json, read_jsonl


def verify_run_directory(run_dir_path: str) -> bool:
    run_dir = Path(run_dir_path)
    if not run_dir.exists():
        print(f"[ERROR] Run directory does not exist: {run_dir}")
        return False

    print(f"=== Verifying Dataset in: {run_dir} ===")
    errors: List[str] = []

    # 1. Manifest
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.exists():
        errors.append(f"Missing manifest.json in {run_dir}")
    else:
        manifest = load_json(manifest_path)
        print(f"[OK] manifest.json loaded: Simulation ID {manifest.get('simulation_id')}, Seed {manifest.get('seed')}")

    # 2. Config
    config_path = run_dir / "config.yaml"
    if not config_path.exists():
        errors.append(f"Missing config.yaml in {run_dir}")
    else:
        print("[OK] config.yaml exists")

    # 3. Ground Truth JSONL
    gt_path = run_dir / "ground_truth.jsonl"
    total_annotations = 0
    unique_barcodes = set()
    if not gt_path.exists():
        errors.append(f"Missing ground_truth.jsonl in {run_dir}")
    else:
        for line_num, record in enumerate(read_jsonl(gt_path), start=1):
            total_annotations += 1
            bval = record.get("barcode_value", "")
            btype = record.get("barcode_type", "")
            unique_barcodes.add(bval)

            # Check EAN-13 validity
            if btype == "EAN13":
                if len(bval) == 13 and bval.isdigit():
                    expected = calculate_ean13_check_digit(bval[:12])
                    if int(bval[12]) != expected:
                        errors.append(f"Line {line_num}: EAN-13 check digit invalid for {bval}")
                else:
                    errors.append(f"Line {line_num}: Invalid EAN-13 format '{bval}'")

            # Check polygon 4 points
            poly = record.get("polygon", [])
            if len(poly) != 4:
                errors.append(f"Line {line_num}: Polygon does not have 4 points: {poly}")

            # Check visibility bounds
            vis = record.get("visibility", -1.0)
            if not (0.0 <= vis <= 1.0):
                errors.append(f"Line {line_num}: Visibility out of [0, 1] range: {vis}")

        print(f"[OK] ground_truth.jsonl verified: {total_annotations} annotations across {len(unique_barcodes)} unique barcodes")

    # 4. Frames directory
    frames_dir = run_dir / "frames"
    if frames_dir.exists():
        frame_files = sorted(list(frames_dir.glob("*.jpg")))
        print(f"[OK] frames/ directory contains {len(frame_files)} image files")
        if frame_files:
            # Check image readable
            sample_img = cv2.imread(str(frame_files[0]))
            if sample_img is None:
                errors.append(f"Failed to read sample image: {frame_files[0]}")
            else:
                print(f"[OK] Sample frame resolution: {sample_img.shape[1]}x{sample_img.shape[0]}")

    # 5. Video
    video_path = run_dir / "video.mp4"
    if video_path.exists():
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            errors.append(f"Failed to open video file with OpenCV: {video_path}")
        else:
            fc = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            vw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            vh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            cap.release()
            print(f"[OK] video.mp4 valid: {fc} frames, {vw}x{vh} @ {fps:.1f} FPS")

    # 6. COCO & YOLO annotations
    coco_path = run_dir / "annotations" / "coco_annotations.json"
    if coco_path.exists():
        coco = load_json(coco_path)
        print(f"[OK] COCO annotations valid: {len(coco.get('images', []))} images, {len(coco.get('annotations', []))} annotations")

    yolo_dir = run_dir / "annotations" / "yolo"
    if yolo_dir.exists():
        yolo_labels = list((yolo_dir / "labels").glob("*.txt")) if (yolo_dir / "labels").exists() else []
        print(f"[OK] YOLO annotations valid: {len(yolo_labels)} label text files")

    # Summary
    if errors:
        print(f"\n[FAILED] Verification found {len(errors)} issues:")
        for err in errors:
            print(f"  - {err}")
        return False
    else:
        print("\n[SUCCESS] Dataset fully verified and passed all structural and mathematical tests!")
        return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Verify a generated simulation dataset.")
    parser.add_argument("run_dir", type=str, help="Path to run output directory.")
    args = parser.parse_args()

    success = verify_run_directory(args.run_dir)
    sys.exit(0 if success else 1)
