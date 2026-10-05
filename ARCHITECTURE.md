# System Architecture: EV Battery Intelligence Platform

The **EV Battery Intelligence Platform** is an automotive-grade decision-support system that extends conventional Battery Management Systems (BMS). It ingests high-frequency or segmented battery telemetry (terminal voltage, pack current, cell temperatures, cycle count) and produces a continuously updated digital representation of battery state, degradation trajectories, safety risks, and explainable diagnostics.

---

## 1. Core Engineering Paradigm: Hybrid Physics + Data-Driven

```mermaid
flowchart TD
    subgraph SENSORS["1. Physical Battery Hardware / CAN Telemetry"]
        PACK_CAN["Pack Telemetry (V_pack, I_pack, T_pack)"]
        CELL_CAN["Cell Telemetry (V_cell, I_cell, T_cell)"]
    end

    subgraph GATEWAY["2. Boundary Gateway & Scaling (Rule 5 & Rule 8)"]
        SCALE["Units Scaling Gateway\nV_cell = V_pack / N_s\nI_cell = I_pack / N_p"]
        RANGE_CHECK{"Physical Bounds Check\n2.5V <= V <= 4.2V\n-40°C <= T <= 100°C"}
        LABEL["Rule 8 Synthetic Labeler\n(flags SYNTH_ records)"]
    end

    subgraph PHYSICS["3. Physics Baseline Engine"]
        ECM["1RC Equivalent Circuit Model\n(OCV-SOC curve, R0, R1, C1)"]
        CC["Coulomb Counting Integrator"]
        EKF["Extended Kalman Filter (EKF)\nState: [SOC, V_1]^T"]
        RESID["Innovation Residual Engine\ny = V_meas - V_pred"]
    end

    subgraph ML_MODELS["4. Machine Learning Intelligence Engines"]
        SOH_XGB["SOH Model (XGBoost)\nNon-leaky features\nMonotonic constraints"]
        RUL_QGB["RUL Model (Quantile GB)\nMedian & 90% Confidence Interval\nStrict EOL-80% boundary (Rule 4)"]
        PINN_LSTM["Advanced PINN LSTM (PyTorch)\nMonotonic irreversibility penalty"]
        ANOMALY["Hybrid Anomaly & Thermal Risk Engine\n0-100 Continuous Escalation Score"]
    end

    subgraph EXPLAIN["5. Explainability & Anti-Hallucination (Phase 9)"]
        TREE_SHAP["TreeSHAP Engine\nExact local attributions\nphi_0 + sum(phi_i) = y_hat"]
        RULES["Electrochemical Diagnostic Rules\n(Impedance rise, SEI breakdown, polarization)"]
        AUDITOR["Anti-Hallucination Verifier\nConfirms 100% of cited numbers in text"]
    end

    subgraph STORAGE["6. Relational Database Layer (SQLAlchemy 2.0 + PostgreSQL/SQLite)"]
        DB[(10 Core Tables\nbattery_packs, cells, telemetry, cycles,\nstate_estimates, predictions, anomaly_events,\nmodel_registry, alerts, users)]
    end

    subgraph BACKEND["7. REST API Layer (FastAPI)"]
        ROUTERS["REST Endpoints (/api/v1)\n/auth, /packs, /telemetry, /inference, /explain, /alerts, /models"]
        SECURITY["Security Guard (Rule 7)\nBcrypt hash, JWT token, Server-side roles,\nCORS allow-list: http://localhost:5173"]
    end

    subgraph UI["8. Engineering Dashboard (React + Vite + TypeScript)"]
        DASH["Instrument Dashboard\n- Pack Hierarchy & Cell Matrix Grid\n- EKF vs CC SOC Observer\n- SOH & RUL 90% Uncertainty Envelope to EOL-80%\n- 0-100 Thermal Runaway Gauge\n- SHAP Feature Impact & Verified Narrative\n- Prominent Rule 8 Synthetic Badges"]
    end

    %% Flow connections
    PACK_CAN --> SCALE
    CELL_CAN --> SCALE
    SCALE --> RANGE_CHECK
    RANGE_CHECK --> LABEL
    LABEL --> STORAGE
    LABEL --> CC
    LABEL --> EKF
    EKF --> RESID
    RESID --> ANOMALY
    STORAGE --> SOH_XGB
    STORAGE --> RUL_QGB
    STORAGE --> PINN_LSTM
    SOH_XGB --> TREE_SHAP
    RUL_QGB --> TREE_SHAP
    TREE_SHAP --> RULES
    RULES --> AUDITOR
    AUDITOR --> STORAGE
    STORAGE <--> ROUTERS
    SECURITY --- ROUTERS
    ROUTERS <--> DASH
```

---

## 2. Ingestion & Physical Scaling Sequence (Rule 5)

```mermaid
sequenceDiagram
    autonumber
    actor BMS as Vehicle BMS / Hardware Gateway
    participant API as FastAPI Ingestion Endpoint (/telemetry)
    participant Units as Units Scaling Module (Rule 5)
    participant DB as Database Session
    participant Observer as Physics Observer (EKF)

    BMS->>API: POST /telemetry/ingest-pack (V_pack=14.8V, I_pack=4.4A, T=28.5°C)
    API->>Units: pack_to_cell_voltage(14.8V, N_s=4) -> 3.70V
    API->>Units: pack_to_cell_current(4.4A, N_p=2) -> 2.20A
    Units->>Units: validate_voltage(3.70V) in [2.5V, 4.2V] -> OK
    Units->>Units: validate_temperature(28.5°C) in [-40°C, 100°C] -> OK
    API->>DB: INSERT INTO telemetry (for all 8 cells in pack)
    API->>Observer: run_ekf(cell_i=2.2A, cell_v=3.70V)
    Observer-->>API: Return [soc_ekf=0.82, soc_cc=0.81, error_bound=0.01]
    API->>DB: INSERT INTO state_estimates
    API-->>BMS: HTTP 201 Created (Ingested 8 cells, 0 warnings)
```

---

## 3. Database Schema Entity-Relationship Diagram

```mermaid
erDiagram
    USERS {
        int id PK
        string username
        string hashed_password
        string role
        boolean is_active
        datetime created_at
    }

    BATTERY_PACKS {
        string id PK
        string name
        string chemistry
        int series_cells
        int parallel_strings
        float nominal_cell_voltage_v
        float nominal_cell_capacity_ah
        float total_voltage_v
        float total_capacity_ah
        boolean is_synthetic
        datetime created_at
    }

    CELLS {
        string id PK
        string pack_id FK
        int cell_index
        string cell_serial
        float initial_capacity_ah
        float nominal_voltage_v
        boolean is_synthetic
        datetime created_at
    }

    TELEMETRY {
        int id PK
        string cell_id FK
        datetime timestamp
        float voltage_v
        float current_a
        float temperature_c
        float soc_reported
        int cycle_count
        boolean is_synthetic
    }

    CYCLES {
        int id PK
        string cell_id FK
        int cycle_number
        float capacity_ah
        float duration_s
        float avg_temp_c
        float max_temp_c
        boolean is_synthetic
        datetime created_at
    }

    STATE_ESTIMATES {
        int id PK
        string cell_id FK
        datetime timestamp
        float soc_ekf
        float soc_cc
        float error_bound
        boolean is_synthetic
    }

    PREDICTIONS {
        int id PK
        string cell_id FK
        int cycle_number
        string target_type
        float predicted_value
        float lower_bound
        float upper_bound
        float confidence_interval
        int model_id FK
        boolean is_synthetic
        datetime created_at
    }

    ANOMALY_EVENTS {
        int id PK
        string cell_id FK
        int cycle_number
        string anomaly_type
        string severity
        float score
        json evidence
        boolean is_synthetic
        datetime created_at
    }

    MODEL_REGISTRY {
        int id PK
        string model_name
        string target_type
        string version
        string file_path
        string dataset_version
        string data_sha256
        json feature_list
        json metrics
        json library_versions
        boolean is_active
        datetime created_at
    }

    ALERTS {
        int id PK
        string cell_id FK
        string pack_id FK
        string alert_type
        string severity
        text message
        json evidence
        string status
        datetime created_at
        datetime resolved_at
    }

    BATTERY_PACKS ||--o{ CELLS : "contains"
    BATTERY_PACKS ||--o{ ALERTS : "has"
    CELLS ||--o{ TELEMETRY : "measures"
    CELLS ||--o{ CYCLES : "segments"
    CELLS ||--o{ STATE_ESTIMATES : "tracks"
    CELLS ||--o{ PREDICTIONS : "forecasts"
    CELLS ||--o{ ANOMALY_EVENTS : "detects"
    CELLS ||--o{ ALERTS : "triggers"
    MODEL_REGISTRY ||--o{ PREDICTIONS : "generates"
```

---

## 4. Enforcement of the 10 Non-Negotiable Rules

| Rule | Requirement | Implementation & Architectural Enforcement |
|---|---|---|
| **1. No Data Leakage** | Group splits strictly by battery ID | Leave-One-Battery-Out (LOBO) cross-validation in `src/ev_battery/soh/evaluation.py` and `src/ev_battery/rul/evaluation.py`. Never random row split. |
| **2. No Target Leakage** | $\text{SOH} = C / C_0$. Capacity cannot be an input. | `validate_no_target_leakage()` in datasets, Pydantic field validators on API schemas, and unit test assertions that fail if `capacity` is supplied. |
| **3. Beat a Baseline** | Report against Predict-the-Mean & Linear models | Every evaluation computes `mean ± std across ALL folds` against mean & linear baselines in `baselines.py` and logs to `model_registry`. |
| **4. RUL Definition** | RUL = cycles until SOH $\le$ 80% (EOL) | Explicitly calculated in `src/ev_battery/rul/dataset.py` using `find_eol_cycle()`. Quantile gradient boosting predicts median and 90% confidence bands $[q_{0.05}, q_{0.95}]$. |
| **5. Units & Scale** | Cell models never receive pack values | Single units gateway (`src/ev_battery/units.py`). Pack telemetry is scaled via series cells ($N_s$) and parallel strings ($N_p$) with bounds checks ($[2.5\text{V}, 4.2\text{V}]$, $[-40^\circ\text{C}, 100^\circ\text{C}]$). |
| **6. Reproducible** | Pinned versions, fixed seeds, SHA256 hashes | Pinned dependencies in `requirements.txt` and `package-lock.json`. Every model stores SHA256 of training data, feature lists, library versions, and metrics in `model_registry`. |
| **7. Security** | No hardcoded secrets, server-side roles, CORS | Pydantic Settings fails at startup if `SECRET_KEY` missing. Passwords hashed with bcrypt. Roles loaded server-side from JWT tokens. CORS allow-list (`http://localhost:5173`). |
| **8. Synthetic Labelling** | Synthetic data clearly labelled everywhere | All tables store `is_synthetic: boolean`. File prefixes use `SYNTH_`. UI renders glowing amber `[SYNTHETIC DATA]` badges. |
| **9. Tests Everywhere** | 100% test pass required | Comprehensive test suite covering data, physics, ML, deep learning, backend, explainability, and E2E integration. |
| **10. Keep It Simple** | Direct connections, no redundant gateways | FastAPI talks directly to the database and serves the React dashboard without unnecessary BFF proxies. |
