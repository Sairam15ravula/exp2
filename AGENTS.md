# AGENTS.md — Engineering Rules

These rules are **non-negotiable**. They exist because a previous attempt failed.

## 1. NO LEAKAGE
Split train/test **BY BATTERY** (leave-one-battery-out or GroupKFold).
Never use random row splits on cycle data — rows from the same battery are correlated.

## 2. NO TARGET LEAKAGE
SOH = capacity / initial_capacity. Therefore **capacity and initial_capacity must NOT be
input features** when predicting SOH. Write a test that fails if a feature is a direct
function of the target.

## 3. ALWAYS BEAT A BASELINE
Report every model against:
- (a) predict-the-mean
- (b) a simple linear-in-cycle model

Report **mean ± std across ALL folds**, never a single lucky fold.

## 4. RUL DEFINITION
RUL = cycles until SOH ≤ 80% (end of life).
**NOT** cycles until the experiment ended. Document this in code and README.

## 5. UNITS AND SCALE
One units module. Cell-level models (~2 Ah, 3.6 V) must never receive pack-level values.
Convert pack → cell via series/parallel configuration. Reject or flag out-of-range inputs.

## 6. REPRODUCIBLE
- Pin every dependency (`requirements.txt` with `==` versions)
- Fix random seeds
- Store training metadata next to each model:
  - dataset version
  - metrics
  - feature list
  - library versions
  - SHA256 of training data

## 7. SECURITY
- No hardcoded secrets — use `.env`, fail at startup if missing
- Auth on every write route
- Roles are **NEVER** accepted from the client (always from server-side token)
- CORS is an allow-list (not `*`)
- Passwords hashed (bcrypt/argon2)

## 8. SYNTHETIC DATA LABELLING
Synthetic data must be clearly labelled synthetic everywhere:
- file names
- metadata
- UI badges
Never present synthetic data as real.

## 9. TESTS FOR EVERY MODULE
A phase is not done until `pytest` passes. No exceptions.

## 10. KEEP IT SIMPLE
No unnecessary services or layers. If you can delete it without breaking anything, delete it.
