# Contributing to dept-similarity

Thank you for your interest in contributing to dept-similarity! This document provides guidelines and information for contributors.

## Getting Started

1. Fork the repository
2. Clone your fork:
   ```bash
   git clone https://github.com/<your-username>/dept-similarity.git
   cd dept-similarity
   ```
3. Set up the development environment:
   ```bash
   uv sync --all-groups
   uv run pre-commit install
   ```

## Development Workflow

1. Create a branch for your feature or fix:
   ```bash
   git checkout -b feature/your-feature-name
   ```
2. Make your changes
3. Run linting and tests:
   ```bash
   uv run ruff check .
   uv run pytest -v tests/
   ```
4. Commit with a clear message and push to your fork
5. Open a Pull Request against `main`

## Code Style

- We use [ruff](https://docs.astral.sh/ruff/) for linting and [black](https://black.readthedocs.io/) for formatting
- Line length limit: 100 characters
- Target Python version: 3.10+

## Testing

- All new features should include tests
- Tests live in the `tests/` directory
- Run the full suite with `uv run pytest -v tests/`

## Reporting Issues

- Use the [GitHub issue tracker](../../issues)
- Include your Python version, OS, and steps to reproduce

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
