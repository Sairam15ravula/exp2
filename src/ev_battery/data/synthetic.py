"""Synthetic battery data generator for testing and demos.

Rule 8: All synthetic data is clearly labelled synthetic. The battery_id
prefix 'SYNTH_' makes it impossible to confuse with real NASA data.
"""

from __future__ import annotations

import numpy as np

from ev_battery.data.loader import CycleData


def generate_synthetic_battery(
    battery_id: str = "SYNTH_001",
    n_cycles: int = 100,
    initial_capacity: float = 2.0,
    degradation_rate: float = 0.001,
    noise_level: float = 0.01,
    seed: int = 42,
) -> list[CycleData]:
    """Generate synthetic battery discharge data.

    Simulates capacity fade over cycles with realistic discharge curves.
    All data is synthetic — battery_id should start with 'SYNTH_'.

    Args:
        battery_id: Identifier (use 'SYNTH_' prefix).
        n_cycles: Number of discharge cycles to generate.
        initial_capacity: Starting capacity in Ah.
        degradation_rate: Capacity loss per cycle (fraction).
        noise_level: Relative noise amplitude.
        seed: Random seed for reproducibility.

    Returns:
        List of CycleData with discharge cycles only.
    """
    rng = np.random.default_rng(seed)
    cycles: list[CycleData] = []

    for i in range(n_cycles):
        # Capacity fades exponentially
        capacity = initial_capacity * np.exp(-degradation_rate * i)

        fade = 1.0 - (capacity / initial_capacity)

        # Generate discharge curve (100 points)
        # Discharge duration scales with capacity (Q = I * t)
        duration = (capacity / initial_capacity) * 3600.0
        n_points = 100
        t = np.linspace(0, duration, n_points)

        # Voltage: starts at 4.2, drops to ~3.0 with a plateau
        progress = np.linspace(0, 1.0, n_points)
        voltage = 4.2 - 1.2 * progress
        # Add plateau in the middle
        plateau = np.exp(-((progress - 0.5) ** 2) / 0.02) * 0.3
        voltage = voltage + plateau
        # Degradation: internal resistance increase & voltage drop proportional to capacity fade
        voltage = voltage - 0.35 * fade * progress
        # Add noise
        voltage = voltage + rng.normal(0, noise_level, n_points)
        voltage = np.clip(voltage, 2.5, 4.2)

        # Current: constant discharge at -2A with noise
        current = np.full(n_points, -2.0) + rng.normal(0, 0.05, n_points)

        # Temperature: rises during discharge, higher heat generation as battery ages
        temp_aging = 6.0 * fade
        temperature = 25.0 + (10.0 + temp_aging) * progress + rng.normal(0, 0.5, n_points)

        cycles.append(CycleData(
            battery_id=battery_id,
            cycle_number=i + 1,
            cycle_type="discharge",
            time=t,
            voltage=voltage,
            current=current,
            temperature=temperature,
            capacity=float(capacity),
        ))

    return cycles


def generate_synthetic_with_fault(
    battery_id: str = "SYNTH_FAULT",
    n_cycles: int = 100,
    fault_cycle: int = 50,
    seed: int = 42,
) -> list[CycleData]:
    """Generate synthetic data with an injected fault at a specific cycle.

    The fault causes a sudden capacity drop and elevated temperature.

    Args:
        battery_id: Identifier (use 'SYNTH_' prefix).
        n_cycles: Total cycles.
        fault_cycle: Cycle number where fault is injected.
        seed: Random seed.

    Returns:
        List of CycleData with a fault at the specified cycle.
    """
    cycles = generate_synthetic_battery(
        battery_id=battery_id,
        n_cycles=n_cycles,
        seed=seed,
    )

    rng = np.random.default_rng(seed + 1)

    for i, cycle in enumerate(cycles):
        if cycle.cycle_number >= fault_cycle:
            # Sudden capacity drop (15% loss)
            cycle.capacity = cycle.capacity * 0.85
            n_pts = len(cycle.time)
            # Duration shortens proportionally to available capacity (Q = I * t)
            dur = (cycle.capacity / 2.0) * 3600.0
            cycle.time = np.linspace(0, dur, n_pts)
            # Elevated temperature with extra joule-heating rise
            cycle.temperature = cycle.temperature + 15.0 + 5.0 * np.linspace(0, 1.0, n_pts)
            # Voltage drops faster due to impedance surge
            cycle.voltage = cycle.voltage - 0.15 * np.linspace(0, 1.0, n_pts)
            cycle.voltage = np.clip(cycle.voltage, 2.5, 4.2)

    return cycles
