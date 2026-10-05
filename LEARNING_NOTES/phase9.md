# Phase 9 Learning Notes: Explainability (SHAP & Grounded Rule-Based Explanations)

## What was built

| File | Purpose |
|---|---|
| `src/ev_battery/explain/shap_explainer.py` | `BatterySHAPExplainer`: Exact TreeSHAP feature attributions for SOH and RUL models with local efficiency verification ($\phi_0 + \sum \phi_i = \hat{y}$) |
| `src/ev_battery/explain/rule_explainer.py` | `BatteryRuleExplainer`: Domain-grounded physical diagnostic rules evaluating impedance growth, thermal stress, voltage polarization, and Rule 4 EOL-80% trajectory |
| `src/ev_battery/explain/narrative.py` | Grounded narrative synthesis engine and algorithmic anti-hallucination auditor (`verify_grounded_text`) |
| `src/ev_battery/schemas/explain.py` | Pydantic v2 schemas for explainability requests and responses with Rule 2 target leakage rejection |
| `src/ev_battery/api/explain.py` | REST API endpoints (`/api/v1/explain/soh`, `/api/v1/explain/rul`) providing feature attributions, rule findings, and grounded narrative summaries |
| `tests/test_explain/` | 15 automated unit and integration tests covering SHAP additivity, rule triggers, narrative grounding, and REST endpoints |

---

## Key concepts in plain English

**1. Shapley Values and Local Additivity ($\phi_0 + \sum_{i=1}^M \phi_i = \hat{y}$)**  
In complex machine learning models (such as gradient boosted decision trees), a single feature's impact cannot be understood in isolation because features interact non-linearly across split paths.  
Shapley values originate from cooperative game theory. They calculate the marginal contribution of each input feature across all possible subsets of features.  
A critical property of TreeSHAP is **local additivity (efficiency)**:
$$\hat{y} = \mathbb{E}[f(x)] + \sum_{i=1}^M \phi_i$$
where $\mathbb{E}[f(x)] = \phi_0$ is the base expected value across the training dataset, and $\phi_i$ is the exact additive contribution of feature $i$. If a battery cell's SOH is predicted as 84.0% when the population average is 92.0%, the sum of all $\phi_i$ values will equal exactly $-8.0\%$. This guarantees that our model explanations are mathematically faithful to the model's actual computations.

**2. Physical Rules Grounded in Electrochemical Numbers**  
While SHAP tells us *which mathematical features* the model relied upon, engineers need to know *what physical degradation mechanisms* are occurring inside the cell:
- **Ohmic Impedance Rise:** An increase in internal resistance proxy ($R_{proxy} > 0.08\,\Omega$) reflects degradation of current collectors, electrolyte dry-out, and thickening of the solid electrolyte interphase (SEI) layer. This causes $I^2 R$ Joule heating and premature voltage cut-off.
- **Thermal Stress Escalation:** Peak temperatures exceeding $35^\circ\text{C}$ accelerate parasitic side-reactions; exceeding $45^\circ\text{C}$ damages separator coatings; and exceeding $60^\circ\text{C}$ initiates exothermic decomposition of the metastable SEI layer, risking thermal runaway.
- **Rule 4 RUL Trajectory:** Remaining Useful Life is strictly tied to the $80.0\%$ SOH boundary ($20\%$ capacity fade). The rule engine contextualizes the $90\%$ prediction interval $[q_{0.05}, q_{0.95}]$ against the cell's historical degradation rate.

**3. Anti-Hallucination Guardrails for Autonomous AI**  
A dangerous failure mode in modern AI applications is the tendency of Large Language Models (LLMs) to invent numbers, fabricate failure causes, or state confident conclusions unsupported by sensor data.  
Under our non-negotiable rule:
> *"Any LLM text must only rephrase computed values, never invent."*  

Our `verify_grounded_text` engine algorithmically extracts all quantitative tokens (integers, percentages, decimals) from generated narrative text and verifies set containment against an evidence dictionary of computed scalars. If a narrative claims an unverified temperature or invented cycle lifespan, the audit flag triggers immediately.

---

## What could go wrong (and how we prevented it)

1. **Non-Additive Explanations:**
   - *Risk:* Using heuristic feature importances (e.g. tree split count or Gini impurity) which do not sum to the prediction and fail to reflect instance-level conditions.
   - *Mitigation:* We use exact TreeSHAP with an explicit assertion verifying $\phi_0 + \sum \phi_i \approx \hat{y}$ within a $10^{-3}$ tolerance.
2. **Target Leakage During Explainability:**
   - *Risk:* An engineer requests an SOH explanation but accidentally includes raw `capacity` in the input feature payload.
   - *Mitigation:* The `ExplainSOHRequest` Pydantic validator checks for forbidden target substrings (`capacity`, `initial_capacity`) and raises an HTTP 422 error before invoking SHAP or rule engines.
3. **AI Text Hallucinations in Safety Audits:**
   - *Risk:* Diagnostic summaries claim safety risks or operating numbers that do not match sensor telemetry.
   - *Mitigation:* Deterministic synthesis templates and numerical regex verification ensure 100% of cited numbers originate from computed telemetry or registered constants.

---

## 3 Viva Questions & Answers

**Q1: What is the local additivity property in SHAP, and why is it important for explaining battery SOH predictions?**  
*Answer:* Local additivity (or efficiency) states that the sum of the feature attributions $\phi_i$ plus the model's base expected value $\phi_0$ equals the exact model prediction $\hat{y}$. In battery diagnostics, this prevents contradictory or ungrounded explanations: an engineer can inspect the waterfall breakdown and see that $+0.02$ SOH came from low operating temperature, while $-0.08$ SOH was subtracted due to high cycle count and elevated internal resistance, perfectly accounting for the final 0.84 SOH prediction.

**Q2: Why is pure ML feature importance insufficient on its own for battery diagnostics, and how do rule-based physical models complement it?**  
*Answer:* Pure machine learning feature importance tells you correlation—for example, that `temp_rise` was important in tree splits—but it cannot explain the physical mechanism or determine whether the value is safe or hazardous. Rule-based physical models embed domain knowledge (e.g. SEI decomposition begins above 60°C, normal fresh cell impedance is ~0.05Ω). Combining TreeSHAP with physical rules gives a hybrid explanation: the ML model identifies the primary statistical drivers, while the physical rules diagnose electrochemical health.

**Q3: How does the platform enforce the non-negotiable rule that "any LLM text must only rephrase computed values, never invent"?**  
*Answer:* The platform uses a two-layer guardrail. First, narrative synthesis is driven by structured evidence bundles containing all computed values (SHAP attributions, rule findings, sensor extrema). Second, an automated auditor (`verify_grounded_text`) parses every number in the generated text using regular expressions and verifies that each number matches a verified scalar in the evidence dictionary within tolerance. Any invented value causes an audit failure.

---

## Known limitations

1. **Tree-Specific Explainer:** `shap.TreeExplainer` is specialized for tree ensembles (XGBoost, Gradient Boosting). Explaining neural network models (such as the Phase 7 PyTorch LSTM) requires `shap.GradientExplainer` or Integrated Gradients, which have higher computational complexity.
2. **Feature Collinearity in Attributions:** Highly correlated features (e.g. `duration_s` and `time_in_voltage_window`) can share attribution mass in TreeSHAP. While mathematically valid under conditional expectation, domain interpretation should consider feature interactions.
3. **Template-Based Synthesis vs Full LLM Integration:** The current synthesis engine uses strictly deterministic sentence templates to eliminate hallucination risk. Future extension to external LLMs (e.g. Gemini API) must pipe outputs through `verify_grounded_text` before returning responses to the user.
