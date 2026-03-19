.PHONY: lint format check test smoke smoke-list

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

smoke:
	uv run vla-eval test --all

smoke-list:
	uv run vla-eval test --list

all: lint check test
