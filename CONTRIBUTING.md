# Contributing to footytrackr

Thanks for your interest in contributing to footytrackr! This document explains how to report issues, suggest improvements, and submit code changes.

## Reporting Issues

Found a bug or have a question? Please open an issue on GitHub. Before creating a new issue, check if someone has already reported it.

When reporting an issue, include:

- What you were trying to do
- What you expected to happen
- What actually happened
- Steps to reproduce (if it's a bug)
- Your Python version and OS

## Suggesting Changes

Have an idea for an improvement? Open an issue first so we can discuss it before you spend time on implementation. This helps avoid duplicated effort and ensures the change aligns with the project's goals.

For this project, prioritize changes that:

- Improve model reproducibility or documentation
- Strengthen data engineering practices
- Make the codebase easier to understand or maintain
- Add meaningful tests or diagnostics
- Fix security or reliability issues

Avoid changes that just add flashy features without improving the core portfolio signals.

## Setting Up Development

Clone the repository:

```bash
git clone https://github.com/mmashaire/footytrackr.git
cd footytrackr
```

Create a virtual environment and install in editable mode:

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -e ".[dev,api]"
```

Verify setup by running tests:

```bash
pytest tests/ -v
```

## Making Changes

### Code style

- Follow PEP 8
- Use type hints where they improve clarity
- Keep functions small and focused
- Write docstrings for public functions and non-obvious logic

The project uses:

- **black** for formatting (88 char line length)
- **isort** for import organization
- **flake8** for linting

Run these before committing:

```bash
black footytrackr/ tests/
isort footytrackr/ tests/
flake8 footytrackr/ tests/ --max-line-length=88
```

### Documentation

- Update README.md if your change affects usage
- Update MODELCARD.md if your change affects model behavior
- Add docstrings explaining why a choice was made, not just what the code does
- Keep documentation conversational, not corporate or robotic

### Testing

All non-trivial changes should include tests. Aim for:

- Unit tests for helper functions and validation
- Integration tests for end-to-end workflows
- Edge case coverage (missing files, invalid input, etc.)

Run tests locally before pushing:

```bash
pytest tests/ -v
```

## Submitting a Pull Request

1. Create a feature branch: `git checkout -b feature/your-feature`
2. Make your changes, following the code style above
3. Add or update tests
4. Update documentation if needed
5. Run the full test suite and linting checks
6. Push your branch and open a pull request

In your PR description, explain:

- What problem does this solve?
- How does it improve the project?
- Any tradeoffs or limitations you considered

## What to Expect

The project maintainer will review your PR and may ask for changes. This is normal! The goal is to keep the codebase clean, maintainable, and aligned with the project's focus on reproducibility and transparency.

If a change is outside the scope of the project, it may be declined. This isn't personal—it just means the project is focused on specific signals for a data/IT portfolio.

## Questions?

Open an issue or start a discussion. No question is too basic.

---

**Thanks for contributing!**
