# Phase 2 Learning Notes: Data Pipeline

## What was built

| File | Purpose |
|---|---|
| `src/ev_battery/data/loader.py` | Load NASA .mat files into `CycleData` structs |
| `src/ev_battery/data/segmentation.py` | Filter to discharge cycles, build flat DataFrame |
| `src/ev_battery/data/features.py` | Extract 10 features per discharge cycle |
| `src/ev_battery/data/validation.py` | Check data quality (range, NaN, length mismatches) |
| `src/ev_battery/data/report.py` | Generate summary report |
| `src/ev_battery/data/synthetic.py` | Synthetic data generator (labelled `SYNTH_`) |
| `src/ev_battery/units.py` | Pack-to-cell conversions, safe range checks |
| `scripts/download_nasa_data.py` | Download NASA dataset |

## Key concepts in plain English

**NASA .mat files** — MATLAB format. Each file is one battery. Inside is a struct
array where each element is one cycle (charge, discharge, or impedance). We use
`scipy.io.loadmat` to read them.

**Cycle segmentation** — The NASA data already separates cycles, but we only care
about **discharge cycles** because those have capacity measurements. Charge and
impedance cycles are filtered out.

**Feature extraction** — Raw voltage/temperature curves are too big for ML. We
compress each cycle into 10 numbers that capture degradation signals:
- `voltage_drop_rate` — how fast voltage falls (batteries degrade → faster drop)
- `internal_resistance_proxy` — voltage drop divided by current (resistance grows as battery ages)
- `temp_rise` — temperature increase during discharge (higher resistance → more heat)
- `time_in_voltage_window` — time spent between 3.0V and 4.2V

**No target leakage (rule 2)** — Features never include `capacity` or
`initial_capacity`. The test `test_no_capacity_leakage` enforces this.

**Synthetic data (rule 8)** — All synthetic batteries use `SYNTH_` prefix. The
generator creates realistic discharge curves with exponential capacity fade.
A fault injection function simulates sudden capacity drops for testing.

**Units module (rule 5)** — `pack_to_cell_voltage(36V, 10 cells) = 3.6V`.
All conversions happen in one place. Tests verify cell-level models never
receive pack-level values.

## What could go wrong

1. **NASA download fails** — URLs change. The script prints a manual fallback link.
2. **.mat file format changes** — NASA may update their format. The loader
   raises clear errors if fields are missing.
3. **Feature extraction on short cycles** — If a cycle has < 2 points, some
   features return 0. Validation catches this.
4. **Synthetic data used as real** — The `SYNTH_` prefix and `SYNTH_FAULT`
   naming make this obvious. Never remove the prefix.

## Viva questions

**Q1: Why only discharge cycles? Why not charge or impedance?**

A: Discharge cycles have capacity measurements. Capacity is the target variable
for SOH. Charge cycles don't measure capacity. Impedance cycles measure internal
resistance but at a different operating point. Mixing them would confuse the model.

**Q2: What is "internal resistance proxy" and why "proxy"?**

A: True internal resistance requires a pulse test (instant current step, measure
voltage jump). We approximate it: voltage drop from start to mid-discharge divided
by average current. It's a "proxy" because it's not the true value, but it trends
the same way — as the battery ages, the proxy increases.

**Q3: How does the synthetic data generator work?**

A: It creates 100-point discharge curves. Voltage starts at 4.2V and drops to 3.0V
with a plateau in the middle. Capacity fades exponentially: `C(n) = C0 * exp(-k*n)`.
Noise is added with a fixed seed so tests are reproducible. The fault function
injects a sudden 15% capacity drop and +15°C temperature rise at a specified cycle.

## Test results

```
63 passed in 0.60s
```

Breakdown:
- `test_config.py` — 9 tests (Phase 1)
- `test_logging.py` — 6 tests (Phase 1)
- `test_units.py` — 10 tests
- `test_data/test_loader.py` — 4 tests
- `test_data/test_segmentation.py` — 4 tests
- `test_data/test_features.py` — 8 tests
- `test_data/test_validation.py` — 8 tests
- `test_data/test_report.py` — 4 tests
- `test_data/test_synthetic.py` — 8 tests

## Known limitations

- NASA data not yet downloaded (requires internet + manual URL verification)
- Features are simple — no differential voltage analysis (dV/dQ) yet
- No cycle-level train/test split yet (Phase 4 adds leave-one-battery-out)
- Synthetic data is realistic but not identical to NASA data
- No dV/dQ or incremental capacity analysis (ICA/DCA) — those come in Phase 4
