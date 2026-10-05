# Phase 10 Learning Notes: Engineering Dashboard (React + Vite + TypeScript + Recharts)

## What was built

| File / Component | Purpose |
|---|---|
| `frontend/package.json` & `package-lock.json` | Pinned React 18, Vite, TypeScript, Recharts, and Lucide dependencies ensuring deterministic builds (Rule 6) |
| `frontend/vite.config.ts` | Development server configuration on port 5173 with direct reverse proxy to FastAPI on port 8000 (Rule 7 & Rule 10) |
| `frontend/src/types/battery.ts` | Strict TypeScript interfaces matching backend SQLAlchemy models and Pydantic schemas |
| `frontend/src/api/client.ts` | Direct REST API client communicating with FastAPI endpoints without any intermediate Node gateway |
| `frontend/src/components/Header.tsx` | App bar with live backend health connectivity, polling controls, and prominent **Rule 8 Global Synthetic Indicator** |
| `frontend/src/components/PackOverview.tsx` | Series-parallel architecture display ($N_s \times N_p$) with cell balance matrix and per-cell drill-down |
| `frontend/src/components/SOCTracker.tsx` | Real-time Recharts visualization comparing Extended Kalman Filter (1RC physics) vs Coulomb counting with error bounds |
| `frontend/src/components/SOHRULCard.tsx` | SOH degradation trajectory with 90% confidence uncertainty envelope $[q_{0.05}, q_{0.95}]$ terminating at the red **80% SOH End-of-Life boundary** (**Rule 4**) |
| `frontend/src/components/ThermalRiskMeter.tsx` | Continuous 0–100 thermal runaway escalation gauge with categorical severity tags (`NORMAL`, `ELEVATED`, `CRITICAL`) and sensor evidence |
| `frontend/src/components/AlertsPanel.tsx` | Active safety alerts feed with instant acknowledge and resolve actions |
| `frontend/src/components/ExplanationPanel.tsx` | Phase 9 explainability panel: TreeSHAP feature attribution bar chart, electrochemical rule findings, and anti-hallucination verified narrative summary |
| `frontend/src/components/LiveTelemetry.tsx` | High-frequency rolling telemetry stream (voltage, current, temperature) plotted against **Rule 5** physical operating bounds |
| `frontend/src/App.tsx` & `main.tsx` | Main dashboard state orchestrator with auto-refresh polling loop and demo dataset fallback |

---

## Key concepts in plain English

**1. Visualizing Uncertainty Bands for Safety-Critical Decisions (Rule 4)**  
In high-voltage automotive battery management, point predictions (e.g. "RUL = 60 cycles") are dangerously deceptive because degradation accelerates unpredictably near end-of-life.  
Our dashboard renders an explicit **90% confidence uncertainty envelope** $[q_{0.05}, q_{0.95}]$ shaded around the median trajectory. Crucially, the trajectory terminates against a bold red reference line at **80.0% SOH**—reinforcing **Rule 4**:
$$\text{RUL} = \text{cycles remaining until SOH} \le 80\%$$
rather than the arbitrary point where a laboratory experiment happened to conclude.

**2. Direct-to-FastAPI Architecture (Rule 10: Keep It Simple)**  
Many junior architectures introduce an unnecessary Node.js "Backend-For-Frontend" (BFF) gateway between React and Python. This adds network latency, redundant serialization schemas, and an extra failure point.  
Per Rule 10, our React application connects **directly to FastAPI**. In development, Vite's internal development server proxies `/api` calls to FastAPI on port 8000. In production, Vite compiles into static HTML/JS/CSS assets that can be served directly by FastAPI or a CDN, eliminating all redundant server layers.

**3. Prominent Synthetic Labelling in UI (Rule 8: Synthetic Labelling)**  
When simulating battery degradation or injecting artificial faults for safety testing, synthetic data can easily look identical to real cell telemetry.  
To prevent lab engineers, field technicians, or researchers from confusing simulated degradation with real battery data:
- Every synthetic pack displays an amber warning banner: `[RULE 8 NOTICE: SYNTHETIC DATA]`.
- Every synthetic cell in the matrix displays an amber `SYN` chip.
- The global header displays an unmistakable glowing indicator: `Synthetic Telemetry Active`.

---

## What could go wrong (and how we prevented it)

1. **CORS Connection Refusals:**
   - *Risk:* Browsers block cross-origin requests from the React development server to FastAPI.
   - *Mitigation:* `Settings.cors_origins` in `src/ev_battery/config.py` explicitly whitelists `http://localhost:5173` (Rule 7 allow-list), and Vite proxies API routes.
2. **Schema Drift Between Frontend and Backend:**
   - *Risk:* Renaming a backend field breaks frontend chart rendering silently.
   - *Mitigation:* Strict TypeScript types in `frontend/src/types/battery.ts` mirror backend Pydantic schemas. TypeScript compilation (`tsc`) halts production builds if an invalid property is accessed.
3. **Chart Rendering Hangs from Large Telemetry Streams:**
   - *Risk:* Ingesting thousands of high-frequency telemetry points chokes the SVG canvas in Recharts.
   - *Mitigation:* Telemetry queries are constrained to rolling windows (e.g. latest 20 to 50 readings) with monotonic step indexes.

---

## 3 Viva Questions & Answers

**Q1: How does the dashboard visualize Rule 4 (RUL defined as cycles until SOH <= 80%) to the operator?**  
*Answer:* The degradation chart plots the battery's SOH history and projects the future degradation trajectory down to a prominent red dashed reference line at 80.0% SOH. A shaded area chart represents the 90% confidence uncertainty interval $[q_{0.05}, q_{0.95}]$ generated by our quantile gradient boosting models. This visually clarifies that RUL is the operational lifespan remaining before the 80% decommissioning threshold is breached, rather than when the battery completely dies or when the test stopped.

**Q2: Why is synthetic telemetry labelled with prominent badges across the UI (Rule 8), and how is this implemented?**  
*Answer:* In EV battery testing, synthetic datasets generated by mathematical simulators are often used to test edge cases (e.g. thermal runaway or rapid overcurrent). If synthetic data were presented without clear labels, engineers could misdiagnose a healthy pack as failing or make incorrect warranty and recycling decisions. Rule 8 mandates that synthetic data must be explicitly marked: the backend flags `is_synthetic: true` on packs, cells, and telemetry, and the UI renders bold amber `[SYNTHETIC]` badges in the header, pack cards, and cell grid.

**Q3: How does the dashboard integrate explainable AI (SHAP and rule findings) into an actionable engineering workflow?**  
*Answer:* Instead of presenting raw ML weights, the Explanation Panel provides a three-part diagnostic breakdown: (1) a horizontal bar chart showing the exact numerical SHAP impacts of top features (e.g. how much internal resistance growth or peak temperature contributed to SOH loss), (2) electrochemical rule findings flagging physical breaches (e.g. SEI degradation limits), and (3) an anti-hallucination verified narrative summary that an engineer can read and copy directly into a maintenance report.

---

## Known limitations

1. **Static Bundle Size:** The production bundle is ~602 kB uncompressed due to comprehensive Recharts and Lucide icon inclusion. In a multi-page enterprise dashboard, React lazy loading (`React.lazy()` / `Suspense`) should be applied to code-split charts.
2. **Polling vs Full WebSockets:** The dashboard currently uses configurable 3s interval polling against FastAPI REST endpoints. While simple and reliable (Rule 10), high-frequency 100 Hz cell telemetry streaming would benefit from a dedicated WebSocket channel.
