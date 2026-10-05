# EV Battery Intelligence Platform

An automotive-grade decision-support system that extends a conventional Battery Management System (BMS). It transforms raw battery telemetry (voltage, current, temperature, SOC, cycle count) into a continuously updated digital representation of battery state, degradation trajectories, safety risks, and explainable diagnostics.

---

## Core Approach: Hybrid Physics + Data-Driven

- **Physics (1RC Equivalent-Circuit Model + Extended Kalman Filter)** provides a mathematically stable, bounded baseline.
- **Machine Learning (Monotonic XGBoost, Quantile Gradient Boosting, Physics-Informed LSTM)** captures nonlinear degradation, temperature sensitivity, and impedance escalation.
- **Every prediction carries uncertainty. Every alert carries evidence.**
- **Explainability:** TreeSHAP attributions combined with electrochemical domain rules, audited by algorithmic anti-hallucination verification.

---

## Architecture & Benchmark Results

- **Complete Architecture & System Flow:** See [ARCHITECTURE.md](file:///C:/Users/Ravula%20Sairam/downloads/exp2/ARCHITECTURE.md)
- **Consolidated Benchmark Results (Rule 3):** See [RESULTS.md](file:///C:/Users/Ravula%20Sairam/downloads/exp2/RESULTS.md)
- **Phase-by-Phase Learning Notes:** See [`LEARNING_NOTES/`](file:///C:/Users/Ravula%20Sairam/downloads/exp2/LEARNING_NOTES/)

---

## Quick Start

### 1. Python Environment Setup
```bash
# 1. Ensure Python 3.11+ is active in virtual environment
# 2. Install pinned dependencies (Rule 6)
pip install -r requirements.txt

# 3. Configure environment variables (Rule 7)
cp .env.example .env
```

### 2. Database & Migrations
```bash
# Start PostgreSQL (or utilize default SQLite for tests/local dev)
docker compose up -d

# Run Alembic migrations to apply all 10 core tables
alembic upgrade head
```

### 3. Run Test Suite
```bash
# Run complete test suite across all 11 phases
pytest -q

# Run end-to-end integration lifecycle test
pytest -v tests/test_e2e/test_full_system.py
```

### 4. Start Backend API
```bash
uvicorn ev_battery.api.app:app --host 0.0.0.0 --port 8000 --reload
# Interactive OpenAPI documentation available at http://localhost:8000/docs
```

### 5. Launch Engineering Dashboard
```bash
cd frontend
npm install
npm run dev
# Dashboard launches at http://localhost:5173
```

---

## Project Structure

```
├── alembic/                # Alembic database migrations (Rule 6)
├── data/                   # Datasets (gitignored — raw data is never committed)
├── frontend/               # React + Vite + TypeScript + Recharts Dashboard (Phase 10)
│   ├── src/components/     # PackOverview, SOCTracker, SOHRULCard, ThermalRiskMeter, etc.
│   └── package.json        # Pinned frontend dependencies and lockfile
├── models/                 # Serialized model artifacts (.joblib, .pt) with metadata
├── src/ev_battery/         # Core application source
│   ├── data/               # NASA loader, segmentation, non-leaky feature extraction, synthetic generator
│   ├── soc/                # Coulomb counting, 1RC ECM, Extended Kalman Filter (EKF)
│   ├── soh/                # XGBoost with monotonic constraints, LOBO evaluation, TreeSHAP
│   ├── rul/                # Quantile Gradient Boosting with 90% prediction intervals (Rule 4)
│   ├── anomaly/            # Hybrid Isolation Forest + EKF residuals, 0-100 thermal risk engine
│   ├── advanced/           # PyTorch sequence LSTM with Physics-Informed thermodynamic loss
│   ├── db/                 # SQLAlchemy 2.0 ORM models (10 tables) and session manager
│   ├── security/           # Bcrypt password hashing, JWT signing, server-side RBAC (Rule 7)
│   ├── schemas/            # Pydantic v2 validation models (Rule 2 target leakage rejection)
│   ├── explain/            # TreeSHAP explainer, electrochemical rules, anti-hallucination narrative
│   ├── api/                # FastAPI REST routers (/auth, /packs, /telemetry, /inference, /explain, /alerts, /models)
│   └── units.py            # Physical constants & unit conversions gateway (Rule 5)
├── tests/                  # Complete automated test suite (> 214 tests)
│   ├── test_data/          # Phase 2 pipeline tests
│   ├── test_soc/           # Phase 3 physics SOC tests
│   ├── test_soh/           # Phase 4 SOH estimation & leakage tests
│   ├── test_rul/           # Phase 5 RUL prediction tests
│   ├── test_anomaly/       # Phase 6 anomaly & fault injection tests
│   ├── test_advanced/      # Phase 7 PyTorch PINN tests
│   ├── test_backend/       # Phase 8 DB, auth, migrations, and API tests
│   ├── test_explain/       # Phase 9 SHAP & rule explainability tests
│   └── test_e2e/           # Phase 11 full system lifecycle integration test
├── ARCHITECTURE.md         # System flowcharts, sequence diagrams, and ERD
├── RESULTS.md              # All-folds benchmark comparisons against baselines (Rule 3)
└── LEARNING_NOTES/         # Viva questions, concepts, and pitfalls for every phase
```

---

## The 10 Non-Negotiable Engineering Rules

1. **NO DATA LEAKAGE:** Split train/test strictly by battery (Leave-One-Battery-Out / GroupKFold). Never random row splits on cycle data.
2. **NO TARGET LEAKAGE:** $\text{SOH} = C / C_0$. Therefore, `capacity` and `initial_capacity` are strictly forbidden as input features. Verified by automated tests and schema validators.
3. **ALWAYS BEAT A BASELINE:** Report every model against (a) predict-the-mean and (b) a linear-in-cycle model. Report **mean $\pm$ std across ALL folds**, never a single lucky fold.
4. **RUL DEFINITION:** $\text{RUL} = \text{cycles until SOH} \le 80\%$ (End-of-Life, EOL). **NOT** cycles until the experiment ended.
5. **UNITS AND SCALE:** Single units gateway (`units.py`). Cell-level models (~2 Ah, 3.6 V) must never receive pack-level values. Convert pack $\rightarrow$ cell via series/parallel configuration ($N_s, N_p$) with strict physical bounds checking.
6. **REPRODUCIBLE:** Pin every dependency (`requirements.txt` and `package-lock.json`), fix random seeds, and store training metadata (dataset version, metrics, feature list, library versions, data SHA256) next to each model.
7. **SECURITY:** No hardcoded secrets (fail at startup if `.env` missing). Auth on every write route. Roles are **NEVER** accepted from client (always server-side token). CORS is an allow-list (`http://localhost:5173`). Passwords hashed with bcrypt.
8. **SYNTHETIC DATA LABELLING:** Synthetic data must be clearly labelled synthetic everywhere (file names `SYNTH_`, database flags, and UI badges). Never present synthetic data as real.
9. **TESTS FOR EVERY MODULE:** A phase is not done until `pytest` passes. 100% test pass rate maintained.
10. **KEEP IT SIMPLE:** No unnecessary services or layers. React talks directly to FastAPI.

---

## Status: All 11 Phases Completed

- [x] Phase 1: Foundation (repo layout, pinned deps, config, units, docker-compose, logging)
- [x] Phase 2: Data pipeline (NASA loader, cycle segmentation, 10 non-leaky features, synthetic generator)
- [x] Phase 3: SOC baseline (Coulomb counting, 1RC Equivalent Circuit Model, Extended Kalman Filter)
- [x] Phase 4: SOH estimation (Monotonic XGBoost, LOBO evaluation beating baselines, TreeSHAP)
- [x] Phase 5: RUL prediction (Strict EOL-80% definition, Quantile GB with 90% prediction intervals)
- [x] Phase 6: Anomaly & thermal risk (Hybrid Isolation Forest + EKF residuals, 0-100 thermal score)
- [x] Phase 7: Advanced models (PyTorch sequence LSTM, physics-informed monotonic loss)
- [x] Phase 8: Backend API (SQLAlchemy 2.0 schema, Alembic migrations, REST API, bcrypt auth)
- [x] Phase 9: Explainability (TreeSHAP attributions, electrochemical rules, anti-hallucination verification)
- [x] Phase 10: Dashboard (React + Vite + TypeScript + Recharts, pack hierarchy, live telemetry)
- [x] Phase 11: Hardening (Full lifecycle E2E test, ARCHITECTURE.md, RESULTS.md, final docs)
