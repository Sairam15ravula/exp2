"""Tests for NASA .mat file loader."""

from __future__ import annotations

import numpy as np
import pytest

from ev_battery.data.loader import CycleData, load_nasa_mat
from ev_battery.data.synthetic import generate_synthetic_battery


class TestLoadNasaMat:
    def test_load_synthetic_mat(self, tmp_path):
        """Create a synthetic .mat file and load it."""
        from scipy.io import savemat

        # Create a simple .mat file with one discharge cycle
        time = np.linspace(0, 3600, 10)
        voltage = np.linspace(4.2, 3.0, 10)
        current = np.full(10, -2.0)
        temperature = np.full(10, 25.0)

        mat_data = np.array([{
            "type": "discharge",
            "time": time,
            "voltage": voltage,
            "current": current,
            "temperature": temperature,
            "capacity": 1.9,
        }], dtype=object)

        filepath = tmp_path / "B0005.mat"
        savemat(str(filepath), {"data": mat_data})

        cycles = load_nasa_mat(filepath)
        assert len(cycles) == 1
        assert cycles[0].battery_id == "B0005"
        assert cycles[0].cycle_type == "discharge"
        assert cycles[0].capacity == pytest.approx(1.9)
        assert len(cycles[0].time) == 10

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_nasa_mat(tmp_path / "nonexistent.mat")

    def test_no_data_field_raises(self, tmp_path):
        from scipy.io import savemat
        filepath = tmp_path / "bad.mat"
        savemat(str(filepath), {"not_data": np.array([1, 2, 3])})
        with pytest.raises(ValueError, match="No 'data' field"):
            load_nasa_mat(filepath)

    def test_multiple_cycles(self, tmp_path):
        from scipy.io import savemat

        cycles_data = []
        for i in range(3):
            cycles_data.append({
                "type": "discharge",
                "time": np.linspace(0, 3600, 10),
                "voltage": np.linspace(4.2, 3.0, 10),
                "current": np.full(10, -2.0),
                "temperature": np.full(10, 25.0),
                "capacity": 2.0 - 0.1 * i,
            })

        filepath = tmp_path / "B0006.mat"
        savemat(str(filepath), {"data": np.array(cycles_data, dtype=object)})

        cycles = load_nasa_mat(filepath)
        assert len(cycles) == 3
        assert cycles[0].cycle_number == 1
        assert cycles[2].cycle_number == 3
        assert cycles[0].capacity == pytest.approx(2.0)
        assert cycles[2].capacity == pytest.approx(1.8)
