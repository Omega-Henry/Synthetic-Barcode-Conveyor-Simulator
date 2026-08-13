"""
Command-line interface for the Synthetic Industrial Barcode Conveyor-Belt Video Generation System.
Supports video generation, parameter sweep experiments, and scanner evaluation.
"""

from __future__ import annotations

import argparse
import datetime
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional
from tqdm import tqdm

from barcode_simulator.annotations.generator import GroundTruthAnnotationManager
from barcode_simulator.core.config import SimulationConfig, load_config
from barcode_simulator.core.models import FrameAnnotation
from barcode_simulator.core.randomization import SeededRNG
from barcode_simulator.evaluation.baseline_decoder import BaselineScanner
from barcode_simulator.evaluation.metrics import EvaluationMetrics
from barcode_simulator.renderers.debug_renderer import DebugOverlayRenderer
from barcode_simulator.simulation.engine import SimulationEngine
from barcode_simulator.utils.io import ensure_dir, load_json, save_image, save_json
from barcode_simulator.utils.logging import print_run_summary, setup_logger
from barcode_simulator.video.encoder import VideoEncoder

logger = setup_logger("cli")


def run_generate(args: argparse.Namespace) -> Path:
    """Execute video simulation and dataset generation."""
    # 1. Load configuration
    if args.config and os.path.exists(args.config):
        config = load_config(args.config)
        logger.info(f"Loaded configuration from: {args.config}")
    else:
        config = SimulationConfig()
        logger.info("Using default simulation configuration.")

    # 2. Apply CLI overrides
    if args.seed is not None:
        config.simulation.seed = args.seed
    if args.duration is not None:
        config.simulation.duration_seconds = args.duration
    if args.fps is not None:
        config.simulation.fps = args.fps
    if args.packages is not None:
        config.packages.count = args.packages
    if args.debug:
        config.output.save_debug = True
    if args.no_video:
        config.output.save_video = False
    if args.no_frames:
        config.output.save_frames = False

    # 3. Determine run directory
    base_out = Path(args.output_dir or config.output.output_dir)
    seed = config.simulation.seed
    if seed is None:
        seed = SeededRNG().seed
        config.simulation.seed = seed

    timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = base_out / f"run_{timestamp_str}_seed_{seed}"
    ensure_dir(run_dir)

    frames_dir = run_dir / "frames"
    debug_dir = run_dir / "debug"
    if config.output.save_frames:
        ensure_dir(frames_dir)
    if config.output.save_debug:
        ensure_dir(debug_dir)

    # 4. Save effective config
    config_save_path = run_dir / "config.yaml"
    config.to_yaml(config_save_path)

    # 5. Initialize Engine, Annotator, and Video Encoder
    rng = SeededRNG(seed)
    engine = SimulationEngine(config=config, rng=rng)
    ann_manager = GroundTruthAnnotationManager(output_dir=run_dir)
    debug_renderer = DebugOverlayRenderer() if config.output.save_debug else None

    video_path = run_dir / "video.mp4"
    video_encoder: Optional[VideoEncoder] = None
    if config.output.save_video:
        video_encoder = VideoEncoder(
            output_path=str(video_path),
            width=config.camera.width,
            height=config.camera.height,
            fps=config.simulation.fps,
            codec=config.output.video_codec,
            crf=config.output.video_crf,
            bitrate=config.output.video_bitrate,
        )

    evaluator = EvaluationMetrics() if args.benchmark_baseline else None
    baseline_scanner = BaselineScanner() if args.benchmark_baseline else None

    # 6. Render Loop
    total_frames = engine.total_frames
    logger.info(f"Starting simulation run '{run_dir.name}': {total_frames} frames ({config.simulation.duration_seconds:.1f}s @ {config.simulation.fps} fps)")

    start_perf = time.perf_counter()

    for frame_idx in tqdm(range(total_frames), desc="Rendering frames", unit="frame"):
        frame_rgb, scene, annotations = engine.step(frame_idx)
        frame_num = frame_idx + 1

        # Record annotations
        ann_manager.record_frame(annotations)

        # Write to video
        if video_encoder:
            video_encoder.write_frame(frame_rgb)

        # Save individual frame image
        if config.output.save_frames:
            frame_file = frames_dir / f"{frame_num:06d}.jpg"
            save_image(frame_rgb, frame_file)

        # Save debug overlay frame
        if debug_renderer and config.output.save_debug:
            debug_frame = debug_renderer.render_debug_frame(frame_rgb, scene, annotations)
            debug_file = debug_dir / f"{frame_num:06d}.jpg"
            save_image(debug_frame, debug_file)

        # Baseline scanner benchmark
        if baseline_scanner and evaluator:
            scan_results = baseline_scanner.scan(frame_rgb)
            evaluator.update(scan_results, annotations)

    # 7. Finalize outputs
    if video_encoder:
        video_encoder.close()

    if config.output.save_coco:
        ann_manager.export_coco(width=config.camera.width, height=config.camera.height)

    if config.output.save_yolo:
        ann_manager.export_yolo(width=config.camera.width, height=config.camera.height)

    # Save manifest
    manifest = engine.get_manifest()
    manifest_path = run_dir / "manifest.json"
    save_json(manifest.to_dict(), manifest_path)

    render_time = time.perf_counter() - start_perf
    fps_actual = total_frames / max(1e-5, render_time)

    # Save summary.json
    summary_data = {
        "simulation_id": manifest.simulation_id,
        "seed": manifest.seed,
        "renderer": manifest.renderer,
        "resolution": manifest.resolution,
        "fps": manifest.fps,
        "duration_seconds": manifest.duration_seconds,
        "total_frames": total_frames,
        "render_time_seconds": round(render_time, 2),
        "rendering_speed_fps": round(fps_actual, 2),
        "number_of_packages": manifest.number_of_packages,
        "number_of_barcodes": manifest.number_of_barcodes,
        "barcode_types": manifest.barcode_types,
        "video_path": str(video_path) if config.output.save_video else None,
        "ground_truth_path": str(ann_manager.jsonl_path),
        "manifest_path": str(manifest_path),
    }

    if evaluator:
        eval_summary = evaluator.compute_summary()
        summary_data["baseline_evaluation"] = eval_summary
        save_json(eval_summary, run_dir / "baseline_evaluation.json")

    save_json(summary_data, run_dir / "summary.json")

    # Print formatted summary box
    print_run_summary(summary_data)

    return run_dir


def run_experiment(args: argparse.Namespace) -> None:
    """Run parameter sweep experiments for thesis research."""
    base_config = load_config(args.config) if args.config else SimulationConfig()
    param_path = args.parameter
    values_str = args.values.split(",")

    output_dir = Path(args.output_dir or "outputs/experiments") / f"sweep_{param_path.replace('.', '_')}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    ensure_dir(output_dir)

    logger.info(f"Starting parameter sweep: '{param_path}' over values {values_str}")

    results_table = []

    for val_str in values_str:
        val_str = val_str.strip()
        # Parse value (float, int, bool, or str)
        if val_str.lower() in ("true", "false"):
            val: Any = val_str.lower() == "true"
        else:
            try:
                val = float(val_str) if "." in val_str else int(val_str)
            except ValueError:
                val = val_str

        # Clone config
        sweep_cfg = SimulationConfig.model_validate(base_config.model_dump())

        # Set nested attribute
        parts = param_path.split(".")
        target = sweep_cfg
        for p in parts[:-1]:
            target = getattr(target, p)
        setattr(target, parts[-1], val)

        sub_out = output_dir / f"{parts[-1]}_{val}"
        logger.info(f"\n--- Running Sweep Point: {param_path} = {val} ---")

        # Run generation
        gen_args = argparse.Namespace(
            config=None,
            seed=sweep_cfg.simulation.seed or 42,
            output_dir=str(output_dir),
            duration=sweep_cfg.simulation.duration_seconds,
            fps=sweep_cfg.simulation.fps,
            packages=sweep_cfg.packages.count,
            debug=False,
            no_video=False,
            no_frames=False,
            benchmark_baseline=True,
        )
        # Apply config directly
        saved_run = run_generate(gen_args)

        # Load baseline evaluation
        eval_json = saved_run / "baseline_evaluation.json"
        if eval_json.exists():
            eval_data = load_json(eval_json)
            results_table.append({
                "parameter": param_path,
                "value": val,
                "recall": eval_data.get("recall", 0.0),
                "precision": eval_data.get("precision", 0.0),
                "accuracy": eval_data.get("f1_score", 0.0),
                "mean_iou": eval_data.get("mean_iou", 0.0),
                "run_dir": str(saved_run),
            })

    # Save sweep summary
    sweep_summary_path = output_dir / "experiment_summary.json"
    save_json(results_table, sweep_summary_path)
    logger.info(f"\nParameter sweep finished. Summary saved to: {sweep_summary_path}")


def run_evaluate(args: argparse.Namespace) -> None:
    """Evaluate external scanner predictions against ground truth jsonl."""
    from barcode_simulator.utils.io import read_jsonl
    from barcode_simulator.evaluation.adapter import ScanResult
    from barcode_simulator.core.models import FrameAnnotation, BoundingBox

    gt_records = list(read_jsonl(args.ground_truth))
    # Group GT by frame_id
    gt_by_frame: Dict[int, List[FrameAnnotation]] = {}
    for r in gt_records:
        fid = r["frame_id"]
        ann = FrameAnnotation(
            frame_id=fid,
            timestamp=r.get("timestamp", 0.0),
            package_id=r.get("package_id", ""),
            barcode_id=r.get("barcode_id", ""),
            barcode_type=r.get("barcode_type", ""),
            barcode_value=r.get("barcode_value", ""),
            bbox=r.get("bbox", [0, 0, 0, 0]),
            polygon=r.get("polygon", []),
            visibility=r.get("visibility", 1.0),
            occlusion=r.get("occlusion", 0.0),
            rotation_degrees=r.get("rotation_degrees", 0.0),
            motion_blur_kernel=r.get("motion_blur", 1),
            velocity_px_s=r.get("velocity_px_s", 0.0),
            is_in_frame=r.get("is_in_frame", True),
            difficulty_tags=r.get("difficulty_tags", []),
        )
        gt_by_frame.setdefault(fid, []).append(ann)

    pred_records = load_json(args.predictions)
    pred_by_frame: Dict[int, List[ScanResult]] = {}
    for p in pred_records:
        fid = p.get("frame_id", 1)
        bbox = BoundingBox(*p["bbox"]) if "bbox" in p and p["bbox"] else None
        res = ScanResult(
            value=p.get("value", ""),
            barcode_type=p.get("barcode_type"),
            bbox=bbox,
            confidence=p.get("confidence", 1.0),
            latency_ms=p.get("latency_ms", 0.0),
        )
        pred_by_frame.setdefault(fid, []).append(res)

    evaluator = EvaluationMetrics()
    all_frames = sorted(set(list(gt_by_frame.keys()) + list(pred_by_frame.keys())))

    for fid in all_frames:
        gts = gt_by_frame.get(fid, [])
        preds = pred_by_frame.get(fid, [])
        evaluator.update(preds, gts)

    summary = evaluator.compute_summary()
    out_path = Path(args.output or "evaluation_results.json")
    save_json(summary, out_path)
    logger.info(f"Evaluation complete. Results saved to: {out_path}")
    print(save_json(summary, sys.stdout, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="barcode-simulator",
        description="Synthetic Industrial Barcode Conveyor-Belt Video Generation & Benchmarking System",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # 1. generate
    gen_parser = subparsers.add_parser("generate", help="Generate synthetic conveyor video and dataset.")
    gen_parser.add_argument("--config", "-c", type=str, default=None, help="Path to YAML configuration file.")
    gen_parser.add_argument("--seed", "-s", type=int, default=None, help="Random seed for reproducibility.")
    gen_parser.add_argument("--output-dir", "-o", type=str, default=None, help="Output destination folder.")
    gen_parser.add_argument("--duration", "-d", type=float, default=None, help="Simulation duration in seconds.")
    gen_parser.add_argument("--fps", type=int, default=None, help="Simulation frames per second.")
    gen_parser.add_argument("--packages", "-p", type=int, default=None, help="Total packages to generate.")
    gen_parser.add_argument("--debug", action="store_true", help="Render debug overlay frames.")
    gen_parser.add_argument("--no-video", action="store_true", help="Skip MP4 video encoding.")
    gen_parser.add_argument("--no-frames", action="store_true", help="Skip saving individual frame jpg files.")
    gen_parser.add_argument("--benchmark-baseline", action="store_true", help="Run baseline scanner evaluation.")

    # 2. experiment
    exp_parser = subparsers.add_parser("experiment", help="Run parameter sweep experiments for scientific research.")
    exp_parser.add_argument("--config", "-c", type=str, default=None, help="Base YAML configuration file.")
    exp_parser.add_argument("--parameter", "-p", type=str, required=True, help="Nested parameter path (e.g. conveyor.speed_min).")
    exp_parser.add_argument("--values", "-v", type=str, required=True, help="Comma-separated sweep values (e.g. '80,140,200').")
    exp_parser.add_argument("--output-dir", "-o", type=str, default="outputs/experiments", help="Output directory.")

    # 3. evaluate
    eval_parser = subparsers.add_parser("evaluate", help="Evaluate external scanner predictions.")
    eval_parser.add_argument("--ground-truth", "-g", type=str, required=True, help="Path to ground_truth.jsonl.")
    eval_parser.add_argument("--predictions", "-p", type=str, required=True, help="Path to predictions JSON file.")
    eval_parser.add_argument("--output", "-o", type=str, default="evaluation_results.json", help="Summary output JSON path.")

    args = parser.parse_args()

    if args.command == "generate":
        run_generate(args)
    elif args.command == "experiment":
        run_experiment(args)
    elif args.command == "evaluate":
        run_evaluate(args)


if __name__ == "__main__":
    main()
