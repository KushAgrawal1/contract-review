# Contributing

## Setup

    pip install -r requirements.txt -r requirements-dev.txt
    cp .env.example .env

## Tests

The unit tests need no database or API key:

    python -m pytest -q

## Style

Run `ruff check . --fix` and `ruff format .` before committing.
Pre-commit hooks are available: `pre-commit install`.

## Pull requests

Keep changes small and focused. If a change touches prompts, chunking or
retrieval, re-run the eval in `evals/` and note the before and after numbers.
