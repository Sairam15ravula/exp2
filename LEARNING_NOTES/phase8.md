# Phase 8 Learning Notes: Backend Architecture & Database

## What was built

| File | Purpose |
|---|---|
| `src/ev_battery/db/base.py` | SQLAlchemy 2.0 `DeclarativeBase` and `TimestampMixin` for automatic UTC timestamps |
| `src/ev_battery/db/models.py` | 10 database tables: `users`, `battery_packs`, `cells`, `telemetry`, `cycles`, `state_estimates`, `predictions`, `anomaly_events`, `model_registry`, `alerts` |
| `src/ev_battery/db/session.py` | Connection engine and session maker with `get_db` FastAPI dependency (supporting SQLite in tests and PostgreSQL in dev/production) |
| `alembic/env.py` & `alembic/versions/` | Alembic migration framework tracking database schema evolution with automated upgrade and rollback capability |
| `src/ev_battery/security/auth.py` | Rule 7 security: Bcrypt password hashing, cryptographic JWT signing/verification, and server-side role-based authorization |
| `src/ev_battery/schemas/` | Pydantic v2 schemas validating pack creation, telemetry ingestion, inference payloads (Rule 2 target leakage prevention), alerts, and model provenance |
| `src/ev_battery/api/packs.py` | Pack and cell management endpoints, automating pack architecture scaling ($V_{pack} = N_s \cdot V_{cell}, C_{pack} = N_p \cdot C_{cell}$) |
| `src/ev_battery/api/telemetry.py` | Ingestion pipeline with Rule 5 physical bounds validation ([2.5V, 4.2V], [-40°C, 100°C]) and Rule 8 synthetic labeling |
| `src/ev_battery/api/inference.py` | Live inference endpoints for Phase 3 EKF/Coulomb counting SOC, Phase 4 SOH XGBoost, Phase 5 Quantile RUL, and Phase 6 Thermal Risk Scoring |
| `src/ev_battery/api/alerts.py` | Safety alert lifecycle management (`ACTIVE` -> `ACKNOWLEDGED` -> `RESOLVED`) with audit evidence |
| `src/ev_battery/api/models.py` | Model Registry inspection endpoints exposing dataset SHA-256 hashes, feature sets, and baseline comparisons (Rule 3 & Rule 6) |
| `src/ev_battery/api/app.py` | FastAPI application factory configured with strict CORS allow-list (Rule 7, never `*`) and system health checks |
| `tests/test_backend/` | 28 automated tests covering Alembic migrations, database models, auth role enforcement, telemetry scaling, and REST endpoints |

---

## Key concepts in plain English

**1. Pack-to-Cell Voltage & Current Scaling (Rule 5: Units & Scale)**  
In electric vehicles, battery packs consist of dozens or hundreds of individual cells arranged in series ($N_s$) and parallel ($N_p$). For example, a 400V EV pack might use a 96S2P configuration.  
Physics models (Equivalent Circuit Models, Kalman filters) and ML models (trained on NASA or CALCE data) are calibrated strictly to single-cell electrochemistry (~3.6V nominal, ~2.0Ah capacity). If a pack-level telemetry reading of 380V or 200A is passed directly into a cell model, mathematical singularities occur, Kalman filters diverge, and neural network weights produce garbage.  
Our ingestion layer converts pack measurements to cell-level quantities at the boundary:
$$V_{cell} = \frac{V_{pack}}{N_s}, \quad I_{cell} = \frac{I_{pack}}{N_p}$$
and immediately validates them against physical cell safety limits ($2.5\text{V} \le V_{cell} \le 4.2\text{V}$).

**2. Server-Authoritative Role-Based Access Control (Rule 7: Security)**  
A common security vulnerability in junior developer APIs is allowing the client to send its desired role in a request body (e.g. `{"role": "admin"}`) or trusting unverified headers.  
Under Rule 7, roles are strictly immutable from the client perspective:
- When a user logs in with valid bcrypt-verified credentials, the server generates a cryptographically signed JSON Web Token (JWT) containing `{"sub": username, "role": user.role}` signed with `SECRET_KEY`.
- On every protected write route (`POST`, `PUT`, `PATCH`, `DELETE`), the `require_write_access` dependency extracts the identity from the verified signature, re-validates against the database, and confirms the account has `engineer` or `admin` status. Viewers attempting write operations are rejected with HTTP 403 Forbidden.

**3. Zero-Target-Leakage Validation at API Boundaries (Rule 2: Target Leakage)**  
Because $\text{SOH} = \text{capacity} / \text{initial\_capacity}$, passing `capacity` or `initial_capacity` into a State of Health model is a fatal data leakage bug that produces artificially perfect test scores that collapse in the field.  
In addition to unit test assertions in Phase 4, Phase 8 embeds this rule into the API runtime itself: Pydantic field validators reject any inference request whose feature dictionary contains `"capacity"`, `"initial_capacity"`, or related substrings with HTTP 422 Unprocessable Entity.

**4. Model Provenance & Reproducibility (Rule 6 & Rule 3)**  
Machine learning in mission-critical automotive systems cannot be a black box. The `model_registry` table stores complete lineage next to every serialized model artifact:
- Deterministic SHA-256 hash of the exact training dataset
- Dataset version string
- Explicit list of non-leaky input feature names
- Benchmark evaluation metrics reported against baselines (Predict-the-Mean, Linear-in-cycle)
- Exact library versions (`xgboost==2.1.3`, `scikit-learn==1.6.0`, etc.)

---

## What could go wrong (and how we prevented it)

1. **Client Role Elevation Attacks:**
   - *Risk:* An unauthorized viewer modifies HTTP request payloads to gain administrative control or inject false telemetry.
   - *Mitigation:* Roles are never read from request payloads. The role claim is verified from the server-signed JWT signature and checked against the database.
2. **Pack Telemetry Mismatch:**
   - *Risk:* Ingesting pack telemetry when the pack topology has not yet been registered or has 0 cells.
   - *Mitigation:* The `/telemetry/ingest-pack` route verifies pack existence and ensures child cells are registered before distributing scaled telemetry.
3. **Target Leakage at the REST Gateway:**
   - *Risk:* A third-party client includes raw capacity measurements in the feature dictionary when requesting SOH predictions.
   - *Mitigation:* Pydantic field validators explicitly inspect incoming dictionaries and abort with a 422 error before invoking ML inference.
4. **Hardcoded Secrets:**
   - *Risk:* Accidental leak of JWT secret keys or database passwords.
   - *Mitigation:* Pydantic Settings enforces `.env` configuration; missing `SECRET_KEY` aborts application launch at startup. `.env` is pinned in `.gitignore`.

---

## 3 Viva Questions & Answers

**Q1: Why must battery pack telemetry be normalized to cell-level quantities before running SOC/SOH inference?**  
*Answer:* Battery electrochemical equations (e.g. Nernst open-circuit voltage curves, Butler-Volmer kinetics) and equivalent circuit RC models are parameterized based on the physical chemistry of a single cell (nominal 3.6V, 2.0Ah, 2.5V–4.2V operating window). Feeding pack-level telemetry (e.g. 400V, 150A) into cell-level models would violate physical voltage limits, cause Kalman filters to diverge, and produce nonsensical SOH predictions. By scaling via series cells ($N_s$) and parallel strings ($N_p$), the models always receive physically valid cell-level data.

**Q2: How does the system enforce Rule 7 (roles never accepted from client) and why is it critical in battery management?**  
*Answer:* Client-supplied roles can be easily spoofed using tools like Postman or curl. In our architecture, the user's role is stored exclusively in the server-side database. Upon successful login, the server signs a JWT containing the role claim using a secret key (HS256). Every protected write route validates the JWT cryptographic signature and loads the user from the database. A viewer or unauthenticated user cannot perform dangerous BMS operations, such as modifying safety thresholds or clearing thermal runaway alerts.

**Q3: How does the Model Registry table fulfill Rule 6 (reproducibility) for ML lifecycle auditing?**  
*Answer:* In automotive and energy storage safety standards (e.g. ISO 26262), every predictive algorithm in production must be auditable. The `model_registry` table stores the dataset version, the exact SHA-256 hash of the training data array, the explicit list of input features, the pinned library versions (`requirements.txt`), and the baseline comparison metrics across all Leave-One-Battery-Out folds. This ensures that any model can be deterministically reproduced, verified, and audited.

---

## Known limitations

1. **SQLite vs PostgreSQL Concurrency:** In unit tests, SQLite in-memory with static pooling is used for rapid isolation. In multi-worker production with PostgreSQL, connection pooling and row-level locking should be managed via Alembic migrations under heavy telemetry ingest concurrency.
2. **Synchronous Inference Overhead:** The current `/inference/*` endpoints perform on-the-fly model inference synchronously. For high-frequency pack telemetry streams (e.g. 100 Hz from CAN bus), ingestion should be decoupled using an asynchronous task queue (e.g. Celery or Redis Stream) with batch inference.
3. **Mock Default Models on Fresh Disk:** If a serialized model artifact (`.joblib`) is missing from the `models/` directory, the backend automatically trains a lightweight surrogate model to maintain test and dev environment uptime without manual model training steps. In production, models must be explicitly registered through `/models`.
