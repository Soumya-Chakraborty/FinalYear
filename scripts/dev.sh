#!/bin/bash
# Development scripts for RaagHMM

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}RaagHMM Development Scripts${NC}"
echo "=========================="

usage() {
    echo "Usage: $0 {test|lint|format|install-dev|clean|help}"
    echo
    echo "Commands:"
    echo "  test         Run all tests"
    echo "  lint         Run linters (flake8, mypy)"
    echo "  format       Format code with black"
    echo "  install-dev  Install development dependencies"
    echo "  clean        Clean build artifacts"
    echo "  help         Show this help message"
    echo
}

case "$1" in
    test)
        echo -e "${BLUE}Running tests...${NC}"
        python -m pytest tests/ -v
        ;;
    lint)
        echo -e "${BLUE}Running linters...${NC}"
        echo "Checking with flake8..."
        python -m flake8 src/ tests/
        echo "Checking with mypy..."
        python -m mypy src/
        ;;
    format)
        echo -e "${BLUE}Formatting code...${NC}"
        python -m black src/ tests/ examples/ scripts/
        echo -e "${GREEN}Code formatted successfully${NC}"
        ;;
    install-dev)
        echo -e "${BLUE}Installing development dependencies...${NC}"
        pip install -e ".[dev]"
        echo -e "${GREEN}Development dependencies installed${NC}"
        ;;
    clean)
        echo -e "${BLUE}Cleaning build artifacts...${NC}"
        find . -type d -name "__pycache__" -exec rm -rf {} +
        find . -type d -name ".pytest_cache" -exec rm -rf {} +
        find . -type d -name ".mypy_cache" -exec rm -rf {} +
        find . -type f -name "*.pyc" -delete
        find . -type f -name ".coverage" -delete
        rm -rf .coverage.xml
        rm -rf htmlcov/
        echo -e "${GREEN}Cleaned successfully${NC}"
        ;;
    help|"")
        usage
        ;;
    *)
        echo -e "${RED}Error: Unknown command '$1'${NC}" >&2
        echo
        usage
        exit 1
        ;;
esac