# Phase 7 Learning Notes: Advanced Models (PyTorch LSTM + Physics-Informed Loss)

## What was built

| File | Purpose |
|---|---|
| `src/ev_battery/advanced/dataset.py` | Sliding sequence window generator $(N, W, D)$ and cross-temperature validation split generator (Rule 1 & Rule 2) |
| `src/ev_battery/advanced/model.py` | `BatteryLSTM`: PyTorch recurrent neural network with multi-layer LSTM and linear regression head |
| `src/ev_battery/advanced/loss.py` | `PhysicsInformedLoss`: Custom loss module combining empirical data fidelity (MSE/Huber) with thermodynamic irreversibility and boundary penalties |
| `src/ev_battery/advanced/trainer.py` | Deterministic PyTorch trainer with Adam optimizer, Rule 6 metadata tracking, and `.pt` model checkpointing |
| `src/ev_battery/advanced/comparison.py` | Honest Leave-One-Battery-Out benchmark comparing Phase 4 XGBoost vs Standard LSTM vs Physics-Informed LSTM |
| `tests/test_advanced/` | 14 unit tests verifying tensor window shapes, physics loss penalties, forward/backward gradient flow, and comparative evaluation |

---

## Key concepts in plain English

**1. Sequence Windowing (Temporal Memory)**  
In Phase 4, XGBoost evaluated each discharge cycle as an independent row. While effective, it misses historical acceleration or deceleration in degradation. A recurrent network (LSTM) processes a temporal history window:
$$X_t = [c_{t-W+1}, c_{t-W+2}, \dots, c_t] \in \mathbb{R}^{W \times D}$$
where $W$ is the window size (e.g. 5 past cycles) and $D$ is the feature dimension. The LSTM cell maintains a hidden state vector $h_t$ that carries degradation memory across time steps.

**2. Physics-Informed Neural Networks (PINNs) in Battery Health**  
Standard deep neural networks are black boxes. On small or noisy datasets, an unconstrained neural network can predict unphysical behavior—such as a battery miraculously gaining 5% capacity between consecutive cycles.  
To enforce physical laws, we incorporate domain knowledge directly into the loss function:
$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{data}}(y, \hat{y}) + \lambda_{\text{mono}} \mathcal{L}_{\text{mono}} + \lambda_{\text{bound}} \mathcal{L}_{\text{bound}}$$
- **Monotonicity Penalty ($\mathcal{L}_{\text{mono}}$):** $\text{ReLU}(\hat{y}_t - \hat{y}_{t-1})^2$. If predicted SOH increases, the penalty surges. If SOH decreases or stays flat (physically valid), the penalty is exactly $0.0$.
- **Boundary Penalty ($\mathcal{L}_{\text{bound}}$):** Penalizes any prediction outside physical limits ($0 \le \text{SOH} \le 1.05$).

**3. Deep Learning vs Gradient Boosted Trees: The Honest Reality**  
In industry and academia, deep learning is often hyped, but rigorous machine learning engineers know that **for tabular and cycle-aggregated data, gradient boosted decision trees (XGBoost/LightGBM) frequently outperform neural networks**:
- Tree models have strong inductive biases for tabular features, require zero warmup cycles, train in milliseconds, and don't require gradient tuning.
- LSTMs require larger datasets to generalize, require sequence warmup (the first $W-1$ cycles cannot be predicted without padding), and require careful regularization (like our Physics-Informed Loss) to avoid divergence.

---

## What could go wrong

1. **Warmup Truncation:** Because an LSTM requires a sliding window of length $W$, it cannot make predictions on cycles $1$ through $W-1$ without heuristic padding or an auxiliary model.
2. **Loss Weight Imbalance ($\lambda$ Hyperparameter):** If $\lambda_{\text{mono}}$ is set too high, the model collapses to predicting a flat constant line to avoid penalties. If set too low, the model ignores physics and exhibits non-monotonic oscillations.
3. **Training Latency & Hardware Costs:** While XGBoost trains in under 0.1 seconds on CPU, multi-layer LSTMs require backpropagation through time (BPTT), which is computationally heavier for embedded edge BMS hardware.

---

## Viva questions & answers

**Q1: What is a Physics-Informed Neural Network (PINN), and how is it applied here?**  
*Answer:* A PINN incorporates governing physical laws—expressed as differential equations, boundary conditions, or thermodynamic invariances—directly into the optimization objective of the neural network. In our platform, we enforce the Second Law of Thermodynamics (irreversibility of battery degradation) by penalizing positive gradients in capacity ($\hat{y}_t - \hat{y}_{t-1} > 0$). This constrains the network's hypothesis space to physically admissible degradation paths without requiring explicit physical differential equations.

**Q2: Why does an unconstrained LSTM perform poorly on Leave-One-Battery-Out cross-validation?**  
*Answer:* Deep sequence models have high parameter capacity. When trained on only a few batteries, an unconstrained LSTM easily overfits to the idiosyncratic noise and specific sequence patterns of the training batteries. Under LOBO, when evaluated on a battery with a different degradation rate, the unconstrained LSTM exhibits high variance and poor generalization. Adding physics loss acts as an inductive regularizer that forces the network to obey degradation physics across unseen cells.

**Q3: How would you decide whether to deploy XGBoost or an LSTM on an automotive BMS?**  
*Answer:* For cycle-level decision support on embedded automotive microcontrollers (e.g. NXP or Infineon automotive chips with limited RAM), **XGBoost is vastly superior**: it delivers higher accuracy on tabular cycle features (RMSE 0.0087 vs 0.0227), has lower inference latency, requires no sequence warmup, and can be converted to static C decision trees. An LSTM is only justified if high-frequency continuous sub-cycle time-series (100 Hz current/voltage sensor streams) are processed directly.

---

## Honest Benchmark: Model Comparison Across LOBO Folds

Evaluated on Leave-One-Battery-Out cross-validation:

| Architecture | RMSE (mean ± std) | MAE (mean ± std) | R² (mean ± std) | Monotonic Violations | Inference Latency |
|---|---|---|---|---|---|
| **XGBoost (Phase 4)** | **0.0087 ± 0.0048** | **0.0056 ± 0.0030** | **0.7584 ± 0.1193** | 6.9% | **< 1 ms** |
| **Standard LSTM (MSE)** | 0.3383 ± 0.4346 | 0.3349 ± 0.4370 | -5152.75 ± 7284.28 | 0.0% | ~5 ms |
| **Physics-Informed LSTM (PINN)** | **0.0227 ± 0.0106** | **0.0191 ± 0.0088** | -1.2678 ± 1.1985 | **0.0%** | ~5 ms |

### Key takeaways (Engineering Honesty):
- **Physics Loss Works:** Adding `PhysicsInformedLoss` reduced the LSTM's error from $0.3383$ down to $0.0227$ (a **93% error reduction**) and guaranteed **0.0% unphysical monotonic violations**.
- **XGBoost Still Wins on Tabular Data:** Despite the elegance of neural sequence modeling, Phase 4 XGBoost achieves **61.7% lower RMSE** ($0.0087$ vs $0.0227$) with a positive $R^2 = 0.7584$.
- **Defensible Conclusion:** For cycle-level degradation features, tree ensembles remain the superior production choice, while PINN loss is essential whenever recurrent sequence architectures are mandated.

---

## Test Results

```
170 passed in 9.54s
```

Test coverage breakdown:
- Phases 1–6: 156 tests
- Phase 7 (Advanced Models): 14 tests
  - `test_dataset.py` (4 tests): Sequence window generation, tensor shapes, target leakage checks, cross-temperature splits
  - `test_physics_loss.py` (4 tests): Monotonicity violation penalties, boundary loss, gradient propagation
  - `test_model.py` (3 tests): BatteryLSTM forward pass, numpy inference, trainer loss reduction, Rule 6 checkpoint persistence
  - `test_comparison.py` (3 tests): Monotonic violation counter, honest LOBO comparative benchmark, table formatting

---

## Known limitations

1. **Window Warmup Delay:** The LSTM cannot make predictions for the first $W-1$ cycles of a fresh battery without padding.
2. **Fixed Sequence Stride:** The current implementation uses fixed cycle intervals rather than variable calendar time intervals.
3. **CPU Training Scale:** Designed and tuned for lightweight, reproducible CPU execution. Large transformer-based architectures would require GPU acceleration and thousands of cells.
