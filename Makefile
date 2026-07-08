.PHONY: project-init project-setup test lint

# Run the first time initialization of the project
project-init:
	uv lock && \
	uv sync --all-groups && \
	git init && \
	uv run pre-commit install

# Set up project as a developer
project-setup:
	uv sync --all-groups
	uv run pre-commit install

# Run tests
test:
	uv run pytest -v

# Run linting
lint:
	uv run ruff check .
