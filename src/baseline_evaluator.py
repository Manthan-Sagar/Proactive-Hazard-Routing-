"""
baseline_evaluator.py
=====================
Benchmark harness reproducing the baseline greedy-avoidance performance
under corridor chokepoint fire conditions (Experiment 2b / Origin A in Paper 2).
Uses the self-contained local floor loader with zero external dependencies.
"""

import os
import sys
import numpy as np
import pandas as pd
from PIL import Image

from src.floor_loader import load_floor
from src.config import SimulationConfig, BenchmarkLocations


def load_ground_floor():
    """Loads Ground Floor raster and XML annotations from isolated local data."""
    return load_floor("ground")
