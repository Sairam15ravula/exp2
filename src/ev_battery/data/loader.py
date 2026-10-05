"""Load NASA Li-ion Battery Aging Dataset .mat files.

The NASA dataset stores each battery as a .mat file containing a struct array.
Each struct element is one cycle with fields: type, time, voltage, current,
temperature, and (for discharge) capacity.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class CycleData:
    """One charge/discharge/impedance cycle."""

    battery_id: str
    cycle_number: int
    cycle_type: str  # 'charge', 'discharge', 'impedance'
    time: np.ndarray
    voltage: np.ndarray
    current: np.ndarray
    temperature: np.ndarray
    capacity: float | None


def load_nasa_mat(filepath: str | Path) -> list[CycleData]:
    """Load a NASA .mat file and return list of CycleData.

    Args:
        filepath: Path to the .mat file (e.g. B0005.mat).

    Returns:
        List of CycleData, one per cycle in the file.

    Raises:
        ValueError: If the file has no 'data' field or is malformed.
        FileNotFoundError: If the file does not exist.
    """
    from scipy.io import loadmat

    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    mat = loadmat(filepath, squeeze_me=True, struct_as_record=False)
    battery_id = filepath.stem

    if "data" not in mat:
        raise ValueError(f"No 'data' field in {filepath}")

    data = mat["data"]

    # Handle single-cycle files (struct instead of array)
    if not isinstance(data, np.ndarray):
        data = np.array([data])

    cycles: list[CycleData] = []
    for i, cycle in enumerate(data):
        cycle_type = str(cycle.type).strip().lower()
        time = np.atleast_1d(np.array(cycle.time, dtype=float))
        voltage = np.atleast_1d(np.array(cycle.voltage, dtype=float))
        current = np.atleast_1d(np.array(cycle.current, dtype=float))
        temperature = np.atleast_1d(np.array(cycle.temperature, dtype=float))

        capacity = None
        if hasattr(cycle, "capacity") and cycle.capacity is not None:
            cap_arr = np.atleast_1d(np.array(cycle.capacity, dtype=float))
            capacity = float(cap_arr[0]) if cap_arr.size > 0 else None

        cycles.append(CycleData(
            battery_id=battery_id,
            cycle_number=i + 1,
            cycle_type=cycle_type,
            time=time,
            voltage=voltage,
            current=current,
            temperature=temperature,
            capacity=capacity,
        ))

    return cycles


def load_nasa_directory(directory: str | Path) -> list[CycleData]:
    """Load all .mat files in a directory.

    Args:
        directory: Path to directory containing .mat files.

    Returns:
        Combined list of CycleData from all files.
    """
    directory = Path(directory)
    all_cycles: list[CycleData] = []
    for mat_file in sorted(directory.glob("*.mat")):
        all_cycles.extend(load_nasa_mat(mat_file))
    return all_cycles
