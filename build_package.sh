#!/bin/bash
# Script to build the TEMPO package for distribution

# Ensure the script exits if any command fails
set -e

echo "Cleaning up previous builds..."
rm -rf dist/ build/ *.egg-info/

echo "Installing build dependencies..."
pip install --upgrade pip
pip install --upgrade setuptools wheel twine build

echo "Building the package..."
python -m build

echo "Package built successfully! Distribution files:"
ls -l dist/

echo "To upload to PyPI, run: python -m twine upload dist/*"
echo "To upload to Test PyPI, run: python -m twine upload --repository-url https://test.pypi.org/legacy/ dist/*"
echo "To install locally, run: pip install dist/*.whl" 