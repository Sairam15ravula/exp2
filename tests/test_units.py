"""Tests for unit conversions (rule 5)."""

from __future__ import annotations

import pytest

from ev_battery.units import (
    MAX_CELL_VOLTAGE,
    MIN_CELL_VOLTAGE,
    pack_to_cell_current,
    pack_to_cell_voltage,
    validate_temperature,
    validate_voltage,
)


class TestPackToCellVoltage:
    def test_basic_conversion(self):
        # 10 cells in series, 36V pack -> 3.6V per cell
        assert pack_to_cell_voltage(36.0, 10) == pytest.approx(3.6)

    def test_single_cell(self):
        assert pack_to_cell_voltage(3.7, 1) == pytest.approx(3.7)

    def test_zero_series_raises(self):
        with pytest.raises(ValueError, match="n_series must be positive"):
            pack_to_cell_voltage(36.0, 0)

    def test_negative_series_raises(self):
        with pytest.raises(ValueError, match="n_series must be positive"):
            pack_to_cell_voltage(36.0, -1)


class TestPackToCellCurrent:
    def test_basic_conversion(self):
        # 4 parallel groups, 8A pack -> 2A per cell
        assert pack_to_cell_current(8.0, 4) == pytest.approx(2.0)

    def test_single_parallel(self):
        assert pack_to_cell_current(2.0, 1) == pytest.approx(2.0)

    def test_zero_parallel_raises(self):
        with pytest.raises(ValueError, match="n_parallel must be positive"):
            pack_to_cell_current(8.0, 0)


class TestValidateVoltage:
    def test_valid_range(self):
        assert validate_voltage(3.6) is True
        assert validate_voltage(MIN_CELL_VOLTAGE) is True
        assert validate_voltage(MAX_CELL_VOLTAGE) is True

    def test_out_of_range(self):
        assert validate_voltage(2.0) is False
        assert validate_voltage(4.5) is False


class TestValidateTemperature:
    def test_valid_range(self):
        assert validate_temperature(25.0) is True
        assert validate_temperature(-20.0) is True
        assert validate_temperature(80.0) is True

    def test_out_of_range(self):
        assert validate_temperature(-50.0) is False
        assert validate_temperature(120.0) is False
