# Phase 1 Learning Notes: Foundation

## What was built

| File | Purpose |
|---|---|
| `src/ev_battery/config.py` | Pydantic settings — loads from `.env`, fails at startup if missing |
| `src/ev_battery/logging_config.py` | JSON structured logging — every log line is valid JSON |
| `tests/test_config.py` | 9 tests — defaults, validation, caching |
| `tests/test_logging.py` | 6 tests — JSON format, extra fields, exceptions |
| `docker-compose.yml` | PostgreSQL 16 with health check |
| `Makefile` | `make test`, `make up`, `make down` |
| `AGENTS.md` | All 10 engineering rules documented |
| `requirements.txt` | 20 pinned dependencies |
| `pyproject.toml` | Package config + pytest `pythonpath` |

## Key concepts in plain English

**Pydantic Settings** — Instead of reading environment variables with `os.getenv()`
and manually checking each one, Pydantic validates them at import time. If
`SECRET_KEY` is missing or too short, the app crashes immediately with a clear
error. This is "fail fast" — rule 7.

**JSON logging** — Regular log files are hard to search. JSON logs are structured:
every line has `timestamp`, `level`, `message`, and any extra fields. Tools like
Elasticsearch or CloudWatch can index and query them.

**src/ layout** — Source code lives in `src/ev_battery/`, not in the repo root.
This prevents Python from accidentally importing the wrong thing. The trade-off is
you need `pythonpath = ["src"]` in pytest config or `pip install -e .` to run tests.

**Pinned dependencies** — `==` not `>=`. If numpy 2.3 breaks something, your
project still works because it's pinned to 2.2.1. Rule 6.

## What could go wrong

1. **`.env` not created** — App crashes on startup. Fix: `cp .env.example .env`
2. **Docker not running** — `docker compose up -d` fails. Fix: start Docker Desktop
3. **Port 5432 already in use** — Another PostgreSQL running. Fix: stop it or change port
4. **Python < 3.11** — Some syntax won't work. Fix: upgrade Python

## Viva questions

**Q1: Why does the app fail at startup if SECRET_KEY is missing, instead of using a default?**

A: A default secret key means anyone can forge authentication tokens. Failing fast
forces the operator to set a real secret before the app can be used. Security over
convenience — rule 7.

**Q2: Why JSON logging instead of plain text?**

A: Plain text logs (`2026-10-01 INFO Battery B001 SOC=0.85`) are hard to query.
JSON logs (`{"battery_id":"B001","soc":0.85}`) can be indexed by any field. When
you have 10,000 log lines and need "all warnings for battery B001", JSON wins.

**Q3: Why `src/` layout instead of putting code in the repo root?**

A: Without `src/`, Python might import a local file instead of the installed package.
With `src/`, there's no ambiguity — `import ev_battery` always means the package.
The cost is one extra line in `pyproject.toml`.

## Test results

```
15 passed in 0.09s
```

## Known limitations

- No actual database connection test yet (Phase 8 will add that)
- No authentication yet (Phase 8)
- Logging goes to stdout only — no file output or log rotation yet
- No CI pipeline file yet (GitHub Actions comes in Phase 11)
