"""Simple test runner script."""
import sys
import subprocess
from pathlib import Path


def main():
    """Run pytest with common options."""
    test_dir = Path(__file__).parent / "tests"

    # Default pytest arguments
    args = [
        "pytest",
        str(test_dir),
        "-v",  # Verbose
        "--tb=short",  # Short traceback
        "-x",  # Stop on first failure (optional)
    ]

    # Add user arguments
    if len(sys.argv) > 1:
        args.extend(sys.argv[1:])

    print(f"Running tests in: {test_dir}")
    print(f"Command: {' '.join(args)}\n")

    # Run pytest
    result = subprocess.run(args)

    return result.returncode


if __name__ == "__main__":
    sys.exit(main())
