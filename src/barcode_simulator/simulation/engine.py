"""
Simulation Engine orchestrating time-stepping, trajectories, scene synthesis, and ground-truth metadata generation.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Iterator, List, Optional, Tuple
import numpy as np

from barcode_simulator.barcodes.generator import BarcodeGenerator
from barcode_simulator.core.config import SimulationConfig
from barcode_simulator.core.models import (
    BarcodeData,
    BoundingBox,
    FrameAnnotation,
    GroundTruthManifest,
    Point2D,
    Polygon2D,
)
from barcode_simulator.core.randomization import SeededRNG
from barcode_simulator.core.scene import (
    ActivePackageInstance,
    CameraState,
    ConveyorState,
    LightingState,
    SceneDescription,
)
from barcode_simulator.core.trajectory import LinearTrajectory, Trajectory
from barcode_simulator.products.generator import PackageGenerator
from barcode_simulator.products.package import Package
from barcode_simulator.renderers.base import Renderer
from barcode_simulator.renderers.renderer_2d import OpenCV2DRenderer
from barcode_simulator.simulation.camera import VirtualCamera
from barcode_simulator.simulation.conveyor import ConveyorBelt
from barcode_simulator.utils.geometry import (
    apply_transform_to_points,
    compute_visibility,
    create_perspective_quad,
    get_perspective_matrix,
)


class SimulationEngine:
    """
    Core deterministic simulation engine.
    """

    def __init__(
        self,
        config: SimulationConfig,
        renderer: Optional[Renderer] = None,
        rng: Optional[SeededRNG] = None,
    ):
        self.config = config
        self.rng = rng if rng is not None else SeededRNG(config.simulation.seed)
        self.seed = self.rng.seed

        # Initialize Subsystems
        self.camera = VirtualCamera(config.camera)
        self.conveyor = ConveyorBelt(config.conveyor, config.camera.height)
        self.barcode_gen = BarcodeGenerator()
        self.pkg_gen = PackageGenerator(
            package_settings=config.packages,
            barcode_settings=config.barcodes,
            barcode_generator=self.barcode_gen,
        )

        # Renderer
        if renderer is not None:
            self.renderer = renderer
        else:
            self.renderer = OpenCV2DRenderer(self.rng.spawn("renderer"))

        # Generate packages and assign simulation parameters
        self.packages: List[Package] = []
        self.package_trajectories: Dict[str, Trajectory] = {}
        self.package_effects: Dict[str, Dict[str, Any]] = {}
        self._initialize_scenario()

    def _initialize_scenario(self) -> None:
        """Deterministically instantiate packages, trajectories, and domain randomization parameters."""
        scenario_rng = self.rng.spawn("scenario_init")
        self.packages = self.pkg_gen.generate_batch(scenario_rng)

        w = self.camera.width
        h = self.camera.height
        lane_y = self.conveyor.lane_y

        for i, pkg in enumerate(self.packages):
            pkg_rng = self.rng.spawn(f"pkg_dyn_{pkg.id}")

            # 1. Sample package speed
            speed_min = self.config.conveyor.speed_min
            speed_max = self.config.conveyor.speed_max
            if self.config.conveyor.speed_distribution == "constant":
                speed = (speed_min + speed_max) / 2.0
            elif self.config.conveyor.speed_distribution == "normal":
                speed = float(np.clip(
                    pkg_rng.normal(mean=(speed_min + speed_max) / 2.0, stddev=(speed_max - speed_min) / 4.0),
                    speed_min,
                    speed_max,
                ))
            else:
                speed = pkg_rng.uniform(speed_min, speed_max)

            # 2. Starting position
            pkg_w = pkg.width
            if self.config.conveyor.direction == "left_to_right":
                start_x = -pkg_w / 2.0 - 50.0
            else:
                start_x = w + pkg_w / 2.0 + 50.0

            # Vertical offset on lane
            lane_h = self.conveyor.lane_height
            max_y_offset = max(10.0, (lane_h - pkg.height) * 0.35)
            start_y = lane_y + pkg_rng.uniform(-max_y_offset, max_y_offset)

            # Trajectory
            traj = LinearTrajectory(
                start_time=pkg.data.spawn_time,
                start_x=start_x,
                start_y=start_y,
                speed=speed,
                direction=self.config.conveyor.direction,
                acceleration=self.config.conveyor.acceleration,
                jitter_stddev=self.config.conveyor.jitter_stddev,
                jitter_seed=pkg_rng.integer(1, 2_147_483_647),
            )
            self.package_trajectories[pkg.id] = traj

            # 3. Geometric domain randomization
            rot_deg = 0.0
            if self.config.effects.rotation.enabled:
                rot_deg = pkg_rng.uniform(
                    self.config.effects.rotation.min_degrees,
                    self.config.effects.rotation.max_degrees,
                )

            tilt_x = 0.0
            tilt_y = 0.0
            if self.config.effects.perspective.enabled:
                strength = pkg_rng.uniform(
                    self.config.effects.perspective.strength_min,
                    self.config.effects.perspective.strength_max,
                )
                tilt_x = strength * self.config.effects.perspective.tilt_x_max * pkg_rng.choice([-1.0, 1.0])
                tilt_y = strength * self.config.effects.perspective.tilt_y_max * pkg_rng.choice([-1.0, 1.0])

            # Motion blur kernel factor
            mb_kernel = 1
            if self.config.effects.motion_blur.enabled:
                k_val = int(round(speed * self.config.effects.motion_blur.factor))
                mb_kernel = int(np.clip(
                    k_val,
                    self.config.effects.motion_blur.min_kernel,
                    self.config.effects.motion_blur.max_kernel,
                ))

            # Store package effects
            self.package_effects[pkg.id] = {
                "rotation_deg": rot_deg,
                "tilt_x_deg": tilt_x,
                "tilt_y_deg": tilt_y,
                "motion_blur_kernel": mb_kernel,
                "z_index": i,  # Later spawned packages can overlap earlier ones
            }

        # Initialize renderer
        init_scene = self.build_scene(frame_index=0, time_seconds=0.0)
        self.renderer.initialize(init_scene)

    @property
    def total_frames(self) -> int:
        return int(round(self.config.simulation.duration_seconds * self.config.simulation.fps))

    def build_scene(self, frame_index: int, time_seconds: float) -> SceneDescription:
        """Construct the complete abstract SceneDescription at the given timestamp."""
        w = self.camera.width
        h = self.camera.height
        dt = 1.0 / self.config.simulation.fps

        # Advance conveyor belt offset
        mean_speed = (self.config.conveyor.speed_min + self.config.conveyor.speed_max) / 2.0
        self.conveyor.step(dt, mean_speed)

        # 1. Identify active packages in canvas
        active_instances: List[ActivePackageInstance] = []

        for pkg in self.packages:
            traj = self.package_trajectories[pkg.id]
            eff = self.package_effects[pkg.id]

            if not traj.is_active(time_seconds, canvas_width=w, package_width=pkg.width):
                continue

            pos = traj.get_position(time_seconds)
            vel = traj.get_velocity(time_seconds)

            # Compute package quadrilateral with perspective and rotation
            pkg_poly = create_perspective_quad(
                center_x=pos.x,
                center_y=pos.y,
                width=pkg.width,
                height=pkg.height,
                rotation_deg=eff["rotation_deg"],
                tilt_x_deg=eff["tilt_x_deg"],
                tilt_y_deg=eff["tilt_y_deg"],
            )

            # Compute Homography from package face to screen polygon
            tex_w, tex_h = float(pkg.width), float(pkg.height)
            src_corners = np.array([
                [0.0, 0.0],
                [tex_w, 0.0],
                [tex_w, tex_h],
                [0.0, tex_h]
            ], dtype=np.float64)
            dst_corners = pkg_poly.to_numpy()
            h_matrix = get_perspective_matrix(src_corners, dst_corners)

            # Transform Barcode polygon to screen space
            bc_poly_on_pkg = pkg.barcode_polygon_on_package
            if len(bc_poly_on_pkg.vertices) >= 4:
                bc_pts_screen = apply_transform_to_points(bc_poly_on_pkg.vertices, h_matrix)
                bc_poly_screen = Polygon2D(bc_pts_screen)
                bc_bbox_screen = bc_poly_screen.bounding_box
            else:
                bc_poly_screen = Polygon2D([])
                bc_bbox_screen = BoundingBox(0, 0, 0, 0)

            inst = ActivePackageInstance(
                package_data=pkg.data,
                trajectory=traj,
                current_position=pos,
                current_velocity_px_s=vel,
                package_polygon=pkg_poly,
                barcode_polygon=bc_poly_screen,
                barcode_bounding_box=bc_bbox_screen,
                z_index=eff["z_index"],
                rotation_deg=eff["rotation_deg"],
                tilt_x_deg=eff["tilt_x_deg"],
                tilt_y_deg=eff["tilt_y_deg"],
                motion_blur_kernel=eff["motion_blur_kernel"],
            )
            active_instances.append(inst)

        # 2. Lighting & Environment state
        env_rng = self.rng.spawn(f"env_frame_{frame_index}")

        ambient_b = 1.0
        if self.config.effects.brightness.enabled:
            ambient_b = env_rng.uniform(
                self.config.effects.brightness.min_factor,
                self.config.effects.brightness.max_factor,
            )

        contrast_v = 1.0
        if self.config.effects.contrast.enabled:
            contrast_v = env_rng.uniform(
                self.config.effects.contrast.min_factor,
                self.config.effects.contrast.max_factor,
            )

        gamma_v = 1.0
        if self.config.effects.gamma.enabled:
            gamma_v = env_rng.uniform(
                self.config.effects.gamma.min_gamma,
                self.config.effects.gamma.max_gamma,
            )

        noise_v = 0.0
        if self.config.effects.noise.enabled:
            noise_v = env_rng.uniform(
                self.config.effects.noise.stddev_min,
                self.config.effects.noise.stddev_max,
            )

        # Glare spots
        glares: List[Dict[str, Any]] = []
        if self.config.effects.glare.enabled and env_rng.boolean(self.config.effects.glare.probability):
            glares.append({
                "x": env_rng.uniform(w * 0.2, w * 0.8),
                "y": self.conveyor.lane_y + env_rng.uniform(-100, 100),
                "intensity": env_rng.uniform(self.config.effects.glare.intensity_min, self.config.effects.glare.intensity_max),
                "radius": env_rng.integer(self.config.effects.glare.radius_min, self.config.effects.glare.radius_max),
            })

        lighting = LightingState(
            ambient_brightness=ambient_b,
            contrast=contrast_v,
            gamma=gamma_v,
            sensor_noise_stddev=noise_v,
            glare_sources=glares,
        )

        metadata: Dict[str, Any] = {}
        if self.config.effects.jpeg_compression.enabled:
            metadata["jpeg_quality"] = env_rng.integer(
                self.config.effects.jpeg_compression.min_quality,
                self.config.effects.jpeg_compression.max_quality,
            )

        return SceneDescription(
            frame_index=frame_index,
            time_seconds=time_seconds,
            conveyor=self.conveyor.get_state(),
            camera=self.camera.get_state(),
            lighting=lighting,
            active_packages=active_instances,
            metadata=metadata,
        )

    def compute_frame_annotations(self, scene: SceneDescription) -> List[FrameAnnotation]:
        """Compute ground-truth bounding boxes, polygons, exact visibility, and occlusion metrics."""
        annotations: List[FrameAnnotation] = []
        w = scene.camera.width
        h = scene.camera.height

        # Sort packages by z_index ascending (lower z is underneath, higher z can occlude lower z)
        sorted_pkgs = sorted(scene.active_packages, key=lambda p: p.z_index)

        for i, target_pkg in enumerate(sorted_pkgs):
            bc_data = target_pkg.package_data.barcode_data
            if bc_data is None or len(target_pkg.barcode_polygon.vertices) < 4:
                continue

            # Occluders: any packages layered above target_pkg (i.e. index > i)
            occluders = [p.package_polygon for p in sorted_pkgs[i + 1:]]

            # Calculate exact visibility and occlusion ratio
            vis, occ = compute_visibility(
                barcode_polygon=target_pkg.barcode_polygon,
                viewport_width=w,
                viewport_height=h,
                occluder_polygons=occluders if self.config.effects.occlusion.enabled else None,
            )

            # Check if within or intersecting viewport
            clipped_bbox = target_pkg.barcode_bounding_box.clip_to_frame(w, h)
            is_in_frame = (clipped_bbox is not None) and (vis > 0.001)

            # Determine difficulty tags
            difficulty_tags = []
            if vis < 0.6:
                difficulty_tags.append("heavily_occluded" if vis < 0.3 else "partially_occluded")
            if target_pkg.motion_blur_kernel >= 5:
                difficulty_tags.append("high_motion_blur")
            if abs(target_pkg.tilt_x_deg) > 8 or abs(target_pkg.tilt_y_deg) > 8:
                difficulty_tags.append("perspective_tilt")
            if abs(target_pkg.rotation_deg) > 12:
                difficulty_tags.append("rotated")
            if target_pkg.package_data.barcode_rel_scale < 0.6:
                difficulty_tags.append("small_barcode")

            ann = FrameAnnotation(
                frame_id=scene.frame_index,
                timestamp=scene.time_seconds,
                package_id=target_pkg.package_data.package_id,
                barcode_id=bc_data.barcode_id,
                barcode_type=bc_data.barcode_type.value,
                barcode_value=bc_data.encoded_value,
                bbox=target_pkg.barcode_bounding_box.to_list(),
                polygon=target_pkg.barcode_polygon.to_list(),
                visibility=vis,
                occlusion=occ,
                rotation_degrees=target_pkg.rotation_deg,
                motion_blur_kernel=target_pkg.motion_blur_kernel,
                velocity_px_s=target_pkg.current_velocity_px_s,
                is_in_frame=is_in_frame,
                difficulty_tags=difficulty_tags,
            )
            annotations.append(ann)

        return annotations

    def step(self, frame_index: int) -> Tuple[np.ndarray, SceneDescription, List[FrameAnnotation]]:
        """
        Advance simulation to frame_index, render frame, and compute annotations.
        """
        t = frame_index / float(self.config.simulation.fps)
        scene = self.build_scene(frame_index=frame_index, time_seconds=t)
        annotations = self.compute_frame_annotations(scene)
        frame_rgb = self.renderer.render_frame(scene)
        return frame_rgb, scene, annotations

    def __iter__(self) -> Iterator[Tuple[np.ndarray, List[FrameAnnotation]]]:
        """Generator interface allowing `for frame, ground_truth in engine:` consumption."""
        for frame_idx in range(self.total_frames):
            frame_rgb, _, annotations = self.step(frame_idx)
            yield frame_rgb, annotations

    def get_manifest(self) -> GroundTruthManifest:
        """Create the complete experiment metadata manifest."""
        import datetime
        btypes_count: Dict[str, int] = {}
        materials_count: Dict[str, int] = {}

        for pkg in self.packages:
            if pkg.data.barcode_data:
                bt = pkg.data.barcode_data.barcode_type.value
                btypes_count[bt] = btypes_count.get(bt, 0) + 1
            mat = pkg.data.material.value
            materials_count[mat] = materials_count.get(mat, 0) + 1

        return GroundTruthManifest(
            simulation_id=f"RUN_{self.seed:08d}",
            seed=self.seed,
            created_at=datetime.datetime.now().isoformat(),
            simulator_version="0.1.0",
            renderer=self.config.renderer,
            resolution=(self.camera.width, self.camera.height),
            fps=self.config.simulation.fps,
            duration_seconds=self.config.simulation.duration_seconds,
            total_frames=self.total_frames,
            number_of_packages=len(self.packages),
            number_of_barcodes=len([p for p in self.packages if p.data.barcode_data]),
            barcode_types=btypes_count,
            materials=materials_count,
            parameters=self.config.model_dump(mode="json"),
        )
