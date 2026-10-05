.PHONY: help install test lint up down clean

help:
	@echo "Available targets:"
	@echo "  install   Install dependencies"
	@echo "  test      Run pytest with coverage"
	@echo "  lint      Run ruff linter"
	@echo "  up        Start PostgreSQL via docker-compose"
	@echo "  down      Stop PostgreSQL"
	@echo "  clean     Remove caches and build artifacts"

install:
	pip install -r requirements.txt

test:
	pytest tests/ -v --cov=src/ev_battery --cov-report=term-missing

lint:
	ruff check src/ tests/

up:
	docker compose up -d

down:
	docker compose down

clean:
	rm -rf .pytest_cache .coverage htmlcov __pycache__ */__pycache__ */*/__pycache__
