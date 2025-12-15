# Contributing to RaagHMM

We welcome contributions to the RaagHMM project! This document provides guidelines and instructions for contributing to the project.

## Table of Contents
1. [Getting Started](#getting-started)
2. [Development Setup](#development-setup)
3. [Code Style](#code-style)
4. [Testing](#testing)
5. [Pull Request Process](#pull-request-process)
6. [Issue Reporting](#issue-reporting)
7. [Community](#community)

## Getting Started

Before contributing, please:
- Read our [Code of Conduct](CODE_OF_CONDUCT.md)
- Familiarize yourself with the [existing issues](https://github.com/raaghmm/raag-hmm/issues)
- Check the project's [README](README.md) for project overview and architecture

## Development Setup

1. Fork the repository
2. Clone your fork:
   ```bash
   git clone https://github.com/YOUR_USERNAME/raag-hmm.git
   cd raag-hmm
   ```

3. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

4. Install dependencies:
   ```bash
   pip install -e ".[dev]"
   ```

5. Install pre-commit hooks:
   ```bash
   pre-commit install
   ```

## Code Style

### Python
- Follow [PEP 8](https://pep8.org/) style guide
- Use [Black](https://black.readthedocs.io/) for code formatting
- Use [flake8](https://flake8.pycqa.org/) for linting
- Use type hints for all public functions

### Git
- Write clear, descriptive commit messages
- Use present tense ("Add feature" not "Added feature")
- Reference issues when applicable

## Testing

### Running Tests
```bash
# Run all tests
pytest

# Run tests with coverage
pytest --cov=src/raag_hmm

# Run specific test file
pytest tests/test_specific_module.py
```

### Adding Tests
- Add unit tests for new functionality
- Add integration tests for complex workflows
- Maintain good test coverage (>80%)

## Pull Request Process

1. Create a feature branch (`git checkout -b feature/amazing-feature`)
2. Make your changes
3. Add tests for your changes
4. Run tests and ensure they pass
5. Run linters and formatters
6. Commit your changes with clear message
7. Push to your fork
8. Open a pull request

### Before Submitting
- Ensure all tests pass
- Update documentation as needed
- Add type annotations
- Follow code style guidelines

## Issue Reporting

When reporting issues, please include:
- Clear, descriptive title
- Detailed description of the problem
- Steps to reproduce
- Expected vs actual behavior
- Relevant environment information
- Screenshots if applicable

## Community

- Be respectful and considerate
- Ask questions in our issues or discussions
- Help others when you can
- Contribute to design discussions

## License

By contributing to RaagHMM, you agree that your contributions will be licensed under the MIT License.