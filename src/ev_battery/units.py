"""Unit conversions and physical constants.

Rule 5: One units module. Cell-level models must never receive pack-level values.
All conversions happen here.
"""

from __future__ import annotations

# --- Cell-level constants (typical Li-ion 18650) ---
NOMINAL_CELL_VOLTAGE = 3.6  # V
NOMINAL_CELL_CAPACITY = 2.0  # Ah

# --- Safe operating ranges ---
MIN_CELL_VOLTAGE = 2.5  # V
MAX_CELL_VOLTAGE = 4.2  # V
MIN_TEMPERATURE = -40.0  # °C
MAX_TEMPERATURE = 100.0  # °C


def pack_to_cell_voltage(pack_voltage: float, n_series: int) -> float:
    """Convert pack voltage to per-cell voltage.

    Args:
        pack_voltage: Total pack voltage in volts.
        n_series: Number of cells in series.

    Returns:
        Per-cell voltage in volts.

    Raises:
        ValueError: If n_series is not positive.
    """
    if n_series <= 0:
        raise ValueError(f"n_series must be positive, got {n_series}")
    return pack_voltage / n_series


def pack_to_cell_current(pack_current: float, n_parallel: int) -> float:
    """Convert pack current to per-cell current.

    Args:
        pack_current: Total pack current in amps.
        n_parallel: Number of parallel cell groups.

    Returns:
        Per-cell current in amps.

    Raises:
        ValueError: If n_parallel is not positive.
    """
    if n_parallel <= 0:
        raise ValueError(f"n_parallel must be positive, got {n_parallel}")
    return pack_current / n_parallel


def validate_voltage(voltage: float) -> bool:
    """Check if voltage is within safe cell-level range."""
    return MIN_CELL_VOLTAGE <= voltage <= MAX_CELL_VOLTAGE


def validate_temperature(temp: float) -> bool:
    """Check if temperature is within safe operating range."""
    return MIN_TEMPERATURE <= temp <= MAX_TEMPERATURE
