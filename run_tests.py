#!/usr/bin/env python3
"""
Test runner script for TEMPO project.
Run this script to execute tests with proper environment setup.
"""

import os
import sys
import subprocess
import argparse


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run TEMPO tests")
    parser.add_argument("--unit-only", action="store_true", help="Run only unit tests")
    parser.add_argument(
        "--integration-only", action="store_true", help="Run only integration tests"
    )
    parser.add_argument("--cov", action="store_true", help="Generate coverage report")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    return parser.parse_args()


def run_tests(args):
    """Run tests based on command line arguments."""
    # Prepare command
    cmd = ["pytest"]

    # Add verbose flag if requested
    if args.verbose:
        cmd.append("-v")

    # Add coverage flags if requested
    if args.cov:
        cmd.extend(["--cov=src", "--cov=api", "--cov-report=term-missing"])

    # Filter tests if requested
    if args.unit_only:
        cmd.append("tests/unit/")
    elif args.integration_only:
        cmd.append("tests/integration/")

    # Print command
    print(f"Running: {' '.join(cmd)}")

    # Run tests
    try:
        result = subprocess.run(cmd, check=True)
        return result.returncode
    except subprocess.CalledProcessError as e:
        print(f"Tests failed with code {e.returncode}")
        return e.returncode


if __name__ == "__main__":
    # Ensure we're in the project root directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)

    # Parse arguments
    args = parse_args()

    # Run tests
    sys.exit(run_tests(args))
