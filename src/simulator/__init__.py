"""
Alias module allowing python -m simulator CLI execution.
"""

from barcode_simulator import *
from barcode_simulator.cli import main

__all__ = ["main"]
