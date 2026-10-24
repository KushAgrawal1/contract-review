install:
	pip install -r requirements.txt -r requirements-dev.txt

test:
	python -m pytest -q

lint:
	ruff check .

format:
	ruff format .

up:
	docker compose up --build

.PHONY: install test lint format up
