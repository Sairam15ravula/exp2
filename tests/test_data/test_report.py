"""Tests for data report generation."""

from __future__ import annotations

from ev_battery.data.report import generate_report
from ev_battery.data.synthetic import generate_synthetic_battery


class TestGenerateReport:
    def test_basic_report(self):
        cycles = generate_synthetic_battery(n_cycles=10)
        report = generate_report(cycles)
        assert report["n_batteries"] == 1
        assert report["n_cycles_total"] == 10
        assert report["n_discharge_cycles"] == 10
        assert report["n_issues"] == 0

    def test_capacity_summary(self):
        cycles = generate_synthetic_battery(n_cycles=10, initial_capacity=2.0)
        report = generate_report(cycles)
        cap = report["capacity_summary"]["SYNTH_001"]
        assert cap["first_capacity"] == 2.0
        assert cap["last_capacity"] < cap["first_capacity"]
        assert cap["n_cycles"] == 10

    def test_multiple_batteries(self):
        cycles = []
        cycles.extend(generate_synthetic_battery("SYNTH_001", n_cycles=5))
        cycles.extend(generate_synthetic_battery("SYNTH_002", n_cycles=5))
        report = generate_report(cycles)
        assert report["n_batteries"] == 2
        assert report["n_cycles_total"] == 10

    def test_temperature_summary(self):
        cycles = generate_synthetic_battery(n_cycles=5)
        report = generate_report(cycles)
        assert report["temperature_summary"]["overall_max"] is not None
        assert report["temperature_summary"]["overall_mean"] is not None
