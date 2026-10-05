# Phase 3 Learning Notes: SOC Estimation

## What was built

| File | Purpose |
|---|---|
| `src/ev_battery/soc/equivalent_circuit.py` | 1RC ECM: OCV(SOC), terminal voltage, simulation |
| `src/ev_battery/soc/coulomb_counting.py` | Simple Coulomb counting (baseline) |
| `src/ev_battery/soc/ekf.py` | Extended Kalman Filter for SOC estimation |
| `src/ev_battery/soc/evaluation.py` | RMSE, MAE, max error, convergence time |
| `tests/test_soc/` | 32 tests covering all SOC components |

## Key concepts in plain English

**Coulomb counting** — Integrate current over time to track SOC.
`SOC(t) = SOC(0) + (1/Q) * ∫I dt`. Simple but drifts because:
1. Current sensors have noise → error accumulates
2. Initial SOC is often unknown → constant offset
3. Capacity changes as battery ages → wrong denominator

**Equivalent-circuit model (ECM)** — A simple electrical model of the battery:
```
V_terminal = OCV(SOC) + I*R0 - V1
```
- `OCV(SOC)`: Open-circuit voltage (what you'd measure with no load)
- `I*R0`: Instant voltage drop across internal resistance
- `V1`: Slow transient from an RC branch (models the "lag" in voltage response)

**Extended Kalman Filter (EKF)** — Combines ECM prediction with voltage measurements:
1. **Predict**: Use current to advance SOC and V1 (ECM model)
2. **Update**: Compare predicted voltage with measured voltage, correct SOC

The EKF converges from wrong initial SOC because the voltage measurement
provides a "pull" toward the true value. The Kalman gain controls how much
we trust the model vs the measurement.

**Sign convention** — Discharge current is NEGATIVE in our code. This means:
- SOC update: `SOC(k+1) = SOC(k) + (I * dt) / Q` (I negative → SOC decreases)
- Terminal voltage: `V = OCV + I*R0 - V1` (I negative → V < OCV during discharge)

## What could go wrong

1. **Sign errors** — The most common bug. If discharge current is negative but
   the formula assumes positive, SOC increases during discharge. Always test
   with a known scenario.
2. **EKF divergence** — If process noise is too small, the EKF ignores
   measurements. If too large, it's noisy. Tuning Q and R is essential.
3. **OCV-SOC mismatch** — The polynomial OCV is an approximation. Real cells
   have a different curve. This causes systematic error.
4. **Wrong initial SOC** — EKF converges, but slowly. If the initial guess is
   very wrong, it may take hundreds of cycles.

## Viva questions

**Q1: Why does the EKF beat Coulomb counting?**

A: Coulomb counting integrates noise — errors accumulate linearly with time.
The EKF uses voltage measurements to correct the estimate, so errors don't
accumulate. The EKF also converges from a wrong initial SOC, while Coulomb
counting has a constant offset equal to the initial error.

**Q2: What is the "RC branch" in the ECM?**

A: A resistor (R1) and capacitor (C1) in parallel. It models the fact that
battery voltage doesn't instantly reach steady state when current changes —
there's a transient. The capacitor "charges up" over time, causing a gradual
voltage drop. This is why voltage sags under load and recovers when load is removed.

**Q3: How does the EKF know how much to trust the model vs the measurement?**

A: The Kalman gain (K) is computed from the ratio of process noise (Q) to
measurement noise (R). If Q >> R, K is large → trust the measurement more.
If R >> Q, K is small → trust the model more. The EKF automatically balances
these based on the estimated uncertainty.

## Test results

```
95 passed in 1.18s (63 from Phase 1+2, 32 new)
```

SOC-specific results:
- Coulomb counting: 5 tests
- ECM: 13 tests
- EKF: 6 tests
- Evaluation: 8 tests

## Known limitations

- OCV is a polynomial approximation, not a real cell curve
- EKF parameters (Q, R) are hand-tuned, not optimized
- No temperature dependence in the ECM
- No aging effects (capacity fade) in the ECM
- Synthetic data used for testing — real NASA data may behave differently
- EKF convergence time depends on noise levels and initial error
