.PHONY: lint format check test

lint:
	uv run ruff check --fix
	uv run ruff format

format:
	uv run ruff format

check:
	uv run ruff check
	uv run ruff format --check
	uv run ty check

test:
	uv run pytest

all: lint check test
