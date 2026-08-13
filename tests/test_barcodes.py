"""
Automated unit tests for barcode generation and structural validation.
"""

import numpy as np
import pytest

from barcode_simulator.barcodes.generator import (
    BarcodeGenerator,
    calculate_ean13_check_digit,
    generate_code128_value,
    generate_ean13_value,
    generate_qrcode_value,
)
from barcode_simulator.barcodes.types import BarcodeFormat
from barcode_simulator.barcodes.validation import validate_barcode_data
from barcode_simulator.core.models import BarcodeData, BarcodeType
from barcode_simulator.core.randomization import SeededRNG


def test_ean13_check_digit_calculation():
    # Known GS1 test vectors
    # "400638133393" -> check digit is 1
    assert calculate_ean13_check_digit("400638133393") == 1
    # "590123412345" -> check digit is 7
    assert calculate_ean13_check_digit("590123412345") == 7
    # "012345678901" -> check digit is 2
    assert calculate_ean13_check_digit("012345678901") == 2


def test_ean13_generator_produces_valid_codes():
    rng = SeededRNG(42)
    for _ in range(50):
        val = generate_ean13_value(rng)
        assert len(val) == 13
        assert val.isdigit()
        base12 = val[:12]
        expected_check = calculate_ean13_check_digit(base12)
        assert int(val[12]) == expected_check


def test_code128_generator_uniqueness():
    rng = SeededRNG(42)
    values = set()
    for i in range(100):
        val = generate_code128_value(rng, i + 1)
        assert val.startswith("PKG-")
        values.add(val)
    assert len(values) == 100


def test_barcode_rendering_and_validation():
    gen = BarcodeGenerator()
    rng = SeededRNG(123)

    # 1. Code 128
    bc_c128 = gen.generate_barcode_data(
        barcode_id="BC_001",
        package_id="PKG_001",
        barcode_type=BarcodeType.CODE128,
        rng=rng,
        index=1,
        human_readable_text=True,
    )
    assert bc_c128.image is not None
    assert bc_c128.image.ndim == 3
    is_valid, errors = validate_barcode_data(bc_c128)
    assert is_valid, f"Code 128 validation failed: {errors}"

    # 2. EAN-13
    bc_ean = gen.generate_barcode_data(
        barcode_id="BC_002",
        package_id="PKG_002",
        barcode_type=BarcodeType.EAN13,
        rng=rng,
        index=2,
        human_readable_text=True,
    )
    assert bc_ean.image is not None
    is_valid, errors = validate_barcode_data(bc_ean)
    assert is_valid, f"EAN-13 validation failed: {errors}"

    # 3. QR Code
    bc_qr = gen.generate_barcode_data(
        barcode_id="BC_003",
        package_id="PKG_003",
        barcode_type=BarcodeType.QRCODE,
        rng=rng,
        index=3,
    )
    assert bc_qr.image is not None
    is_valid, errors = validate_barcode_data(bc_qr)
    assert is_valid, f"QR validation failed: {errors}"


def test_barcode_validation_catches_invalid_check_digit():
    # Corrupt EAN-13 check digit
    corrupted_data = BarcodeData(
        barcode_id="BC_CORRUPT",
        package_id="PKG_001",
        barcode_type=BarcodeType.EAN13,
        encoded_value="4006381333939",  # Correct is ...931
        image=np.full((100, 200, 3), 128, dtype=np.uint8),
    )
    is_valid, errors = validate_barcode_data(corrupted_data)
    assert not is_valid
    assert any("checksum mismatch" in err for err in errors)
