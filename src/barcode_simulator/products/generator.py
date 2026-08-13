"""
Batch generation of synthetic packages with associated barcodes and procedural properties.
"""

from __future__ import annotations

from typing import List, Optional
import numpy as np

from barcode_simulator.barcodes.generator import BarcodeGenerator
from barcode_simulator.core.config import BarcodeSettings, PackageSettings
from barcode_simulator.core.models import BarcodeData, BarcodeType, PackageData, PackageMaterial
from barcode_simulator.core.randomization import SeededRNG
from barcode_simulator.products.package import Package


class PackageGenerator:
    """
    Generates a deterministic sequence of simulated packages with valid machine-readable barcodes.
    """

    def __init__(
        self,
        package_settings: PackageSettings,
        barcode_settings: BarcodeSettings,
        barcode_generator: BarcodeGenerator,
    ):
        self.pkg_cfg = package_settings
        self.bc_cfg = barcode_settings
        self.bc_gen = barcode_generator

    def generate_batch(self, rng: SeededRNG) -> List[Package]:
        """
        Generate list of simulated packages with assigned barcodes and spawn times.
        """
        packages: List[Package] = []
        current_spawn_time = 0.0

        for i in range(self.pkg_cfg.count):
            pkg_idx = i + 1
            pkg_id = f"PKG_{pkg_idx:06d}"
            bc_id = f"BC_{pkg_idx:06d}"

            # 1. Spawn sub-RNG for this package to ensure independent determinism
            pkg_rng = rng.spawn(f"pkg_{pkg_idx}")

            # 2. Dimensions
            w = pkg_rng.integer(self.pkg_cfg.size.min_width, self.pkg_cfg.size.max_width)
            h = pkg_rng.integer(self.pkg_cfg.size.min_height, self.pkg_cfg.size.max_height)

            # 3. Material selection
            material_name = pkg_rng.weighted_choice(self.pkg_cfg.materials)
            try:
                material = PackageMaterial(material_name)
            except ValueError:
                material = PackageMaterial.CARDBOARD

            # 4. Barcode type selection
            btype_str = pkg_rng.weighted_choice(self.bc_cfg.types)
            try:
                btype = BarcodeType(btype_str)
            except ValueError:
                btype = BarcodeType.CODE128

            # 5. Generate barcode
            bc_data = self.bc_gen.generate_barcode_data(
                barcode_id=bc_id,
                package_id=pkg_id,
                barcode_type=btype,
                rng=pkg_rng,
                index=pkg_idx,
                human_readable_text=self.bc_cfg.human_readable_text,
                quiet_zone_px=self.bc_cfg.margin_padding,
            )

            # 6. Barcode relative placement & scale on package face
            rel_scale = pkg_rng.uniform(self.bc_cfg.scale_min, self.bc_cfg.scale_max)
            rel_x = pkg_rng.uniform(0.3, 0.7)
            rel_y = pkg_rng.uniform(0.3, 0.7)

            if abs(self.bc_cfg.rotation_max - self.bc_cfg.rotation_min) > 0.01:
                bc_rot = pkg_rng.uniform(self.bc_cfg.rotation_min, self.bc_cfg.rotation_max)
            else:
                bc_rot = self.bc_cfg.rotation_min

            # 7. Spawn interval
            interval = pkg_rng.uniform(self.pkg_cfg.spawn_interval_min, self.pkg_cfg.spawn_interval_max)
            current_spawn_time += interval

            # 8. Construct PackageData
            pkg_data = PackageData(
                package_id=pkg_id,
                width=float(w),
                height=float(h),
                depth=0.2,
                material=material,
                barcode_id=bc_id,
                barcode_data=bc_data,
                barcode_rel_x=rel_x,
                barcode_rel_y=rel_y,
                barcode_rel_scale=rel_scale,
                barcode_rotation_deg=bc_rot,
                has_tape=pkg_rng.boolean(self.pkg_cfg.tape_probability),
                has_labels=self.pkg_cfg.add_shipping_labels,
                has_text=self.pkg_cfg.add_text_markings,
                spawn_time=current_spawn_time,
            )

            package = Package(data=pkg_data, rng=pkg_rng)
            packages.append(package)

        return packages
