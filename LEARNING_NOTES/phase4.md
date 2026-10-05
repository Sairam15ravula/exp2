# Phase 4 Learning Notes: SOH Estimation

## What was built

| File | Purpose |
|---|---|
| `src/ev_battery/soh/dataset.py` | Prepares non-leaky feature matrices, calculates $SOH = \frac{C}{C_0}$, validates Rule 2 (no target leakage), and generates Leave-One-Battery-Out (LOBO) splits |
| `src/ev_battery/soh/baselines.py` | Implements required Rule 3 baselines: Predict-the-Mean and Linear-in-Cycle models |
| `src/ev_battery/soh/model.py` | XGBoost regressor with monotonic constraint support and Rule 6 metadata persistence (SHA-256 data hash, hyperparams, library versions) |
| `src/ev_battery/soh/evaluation.py` | LOBO cross-validation engine calculating per-fold and aggregated (mean ± std) RMSE, MAE, and $R^2$ across all folds |
| `src/ev_battery/soh/explain.py` | Feature importance extraction (gain) and TreeSHAP impact analysis |
| `tests/test_soh/` | 24 tests covering leakage prevention, baseline mechanics, monotonic constraints, LOBO beating baselines, and SHAP explainability |

---

## Key concepts in plain English

**1. State of Health (SOH)**  
SOH measures the present health and remaining storage capability of a battery cell compared to its fresh state:
$$\text{SOH} = \frac{\text{Current Capacity } (C)}{\text{Initial / Rated Capacity } (C_0)}$$
When a cell is brand new, $\text{SOH} = 1.0$ (100%). Over continuous charging and discharging, chemical degradation (SEI layer thickening, lithium plating, active material loss) shrinks its usable capacity until it reaches end-of-life (typically $\text{SOH} \le 80\%$).

**2. No Target Leakage (Rule 2)**  
Because SOH is defined directly from capacity, feeding `capacity` or `initial_capacity` into the model as an input feature would be like giving a student the answer key before an exam. The model would learn a trivial formula ($y = X_1 / X_2$) and fail completely on real unmeasured BMS telemetry. Our dataset pipeline explicitly guards against this with `validate_no_target_leakage()`.

**3. Leave-One-Battery-Out (LOBO) Cross-Validation (Rule 1)**  
Never shuffle and randomly split cycle data across rows. Consecutive cycles from the same battery cell are heavily autocorrelated and share physical quirks. In LOBO, if we have 3 batteries (A, B, C):
- Fold 1: Train on B & C $\rightarrow$ Test on A
- Fold 2: Train on A & C $\rightarrow$ Test on B
- Fold 3: Train on A & B $\rightarrow$ Test on C  
The test battery has never been seen during training, proving the model generalizes across manufacturing variations and usage profiles.

**4. Beating Baselines Across ALL Folds (Rule 3)**  
A machine learning model is useless if it cannot beat simple heuristics:
- **Predict-the-Mean:** Predicts the constant average SOH of training data.
- **Linear-in-Cycle:** Fits $SOH = a \cdot \text{cycle} + b$.  
We evaluate every fold and compute the **mean $\pm$ standard deviation across ALL folds**. Reporting only a single "lucky fold" is strictly forbidden.

**5. Monotonic Constraints**  
Under normal usage without cell reconditioning, battery health degrades monotonically—a battery does not spontaneously gain capacity. XGBoost allows specifying monotonic constraints (`monotone_constraints={"cycle_number": -1}`) to guarantee that predicted SOH never increases as cycle count advances.

---

## What could go wrong

1. **Target Leakage via Proxy Columns:** Even if `capacity` is omitted, including a feature like "total discharge amphours" would reintroduce target leakage. Only instantaneous or shape-based measurements (voltage drops, internal resistance proxy, temperature rise, duration) are permitted.
2. **Overfitting to Cycle Count:** If a model relies 99% on `cycle_number` and ignores telemetry features, it will fail when a battery ages prematurely due to high operating temperatures or aggressive fast charging.
3. **Distribution Shift Across Battery Chemistries:** An XGBoost model trained on NCA cells (NASA dataset) will predict poorly on LFP cells without recalibration, because LFP cells have extremely flat voltage plateaus.
4. **Disjoint LOBO Data Scarcity:** LOBO CV variance can be high if we only have 3 or 4 batteries in total. Standard deviations reflect this honest inter-cell variance.

---

## Viva questions & answers

**Q1: Why is random K-fold cross-validation considered data leakage for battery cycle data?**  
*Answer:* Consecutive cycles from the same battery cell share latent environmental conditions, manufacturing batch characteristics, and internal impedance levels. If cycles 1, 3, and 5 are in the training set and cycle 2 is in the test set, the model interpolates between almost identical adjacent rows rather than learning genuine degradation physics. Leave-One-Battery-Out (LOBO) ensures complete physical separation between training and evaluation cells.

**Q2: What is the purpose of XGBoost monotonic constraints in battery health prognosis?**  
*Answer:* Unconstrained decision trees can produce non-monotonic step functions where small sensor noise at cycle 85 could predict a higher SOH than cycle 80. By enforcing monotonic constraints (such as non-increasing with cycle number or internal resistance), we incorporate thermodynamic domain knowledge into the tree splits, preventing physically impossible upward SOH predictions.

**Q3: How does TreeSHAP explain individual battery degradation predictions?**  
*Answer:* TreeSHAP computes the exact Shapley values for tree-based ensemble models based on cooperative game theory. It breaks down the model's SOH prediction into additive contributions from each input feature relative to the expected base value:
$$\hat{y}(x) = \phi_0 + \sum_{i=1}^{M} \phi_i(x)$$
For example, it can prove that a low SOH prediction for a cell was driven primarily by an elevated `internal_resistance_proxy` (+0.03 drop) and a compressed `duration_s`, providing clear auditability for BMS engineers.

---

## Results Table: Model vs Baselines (LOBO Evaluation)

Evaluated across 3 distinct battery degradation profiles with 50 discharge cycles each:

| Model | RMSE (mean ± std) | MAE (mean ± std) | R² (mean ± std) |
|---|---|---|---|
| **Predict-the-mean** | 0.0392 ± 0.0104 | 0.0340 ± 0.0081 | -2.2540 ± 2.6924 |
| **Linear-in-cycle** | 0.0266 ± 0.0184 | 0.0231 ± 0.0160 | -1.9901 ± 3.5729 |
| **XGBoost (SOH)** | **0.0142 ± 0.0085** | **0.0093 ± 0.0052** | **0.7427 ± 0.1754** |

### Key takeaways:
- XGBoost cuts the error of the Linear Baseline by **46.6%** (RMSE 0.0142 vs 0.0266) and the Mean Baseline by **63.8%** (0.0142 vs 0.0392).
- The baseline models fail to generalize ($R^2 < 0$) when the held-out battery has a different degradation rate from the training batteries.
- XGBoost generalizes across unseen batteries with $R^2 = 0.7427 \pm 0.1754$.

### Top Feature Importances (TreeSHAP Mean Absolute Impact):
1. `duration_s` (0.0266) — Usable discharge duration under load
2. `time_in_voltage_window` (0.0020) — Time between 3.0V and 4.2V
3. `voltage_drop_rate` (0.0008) — Rate of voltage sag under constant discharge current
4. `voltage_mean` (0.0006) — Average discharge plateau voltage
5. `cycle_number` (0.0001) — Raw cycle index

---

## Test Results

```
119 passed in 1.90s
```

Test coverage breakdown:
- Phase 1 & 2: 63 tests
- Phase 3 (SOC estimation): 32 tests
- Phase 4 (SOH estimation): 24 tests
  - `test_leakage.py` (5 tests): Rule 1 & Rule 2 enforcement
  - `test_baselines.py` (6 tests): Predict-the-mean & Linear-in-cycle verification
  - `test_model.py` (4 tests): SOHModel, monotonic constraints, and Rule 6 metadata persistence
  - `test_evaluation.py` (6 tests): LOBO cross-validation and baseline beating
  - `test_explain.py` (3 tests): Feature importance and TreeSHAP calculations

---

## Known limitations

1. **Synthetic Telemetry Baseline**: Validated using physics-grounded synthetic discharge curves. Next steps on real NASA hardware data will introduce real sensor anomalies and cell relaxation periods.
2. **Operating Temperature Variation**: The current dataset assumes nominal operating temperatures. Phase 6 and 7 will evaluate performance under severe thermal stress.
3. **No Direct ICA/DCA Peaks**: Features currently compress the voltage curve using scalar statistics (`mean`, `drop_rate`, `time_in_window`) rather than explicit $dQ/dV$ differential capacity peak tracking.
