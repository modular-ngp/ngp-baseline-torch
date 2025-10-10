"""Test Suite Summary and Runner

New Test Structure:
===================

Test 1: Core Components (test_1_core_components.py)
- Hash Grid initialization (CRITICAL: uniform [-1e-4, 1e-4])
- MLP with truncated exponential activation
- Sigma bias initialization (-1.5)
- RGB head with view-dependent rendering

Test 2: Integrator (test_2_integrator.py)
- Sample jittering during training (CRITICAL)
- Volume rendering compositing
- Early stopping
- White background handling

Test 3: Training Components (test_3_training_components.py)
- Separate learning rates for encoder/MLP
- Optimizer functionality
- Loss functions
- End-to-end gradient flow

Test 4: E2E Convergence (test_4_e2e_convergence.py) ⚠️ CRITICAL
- Single image overfit (500 iters → 25+ dB)
- Fast convergence (1000 iters → 20+ dB)
- Full convergence (5000 iters → 28+ dB) ← Main quality gate

Running Tests:
==============

# Run all fast tests (excludes convergence tests)
python run_tests.py

# Run quick convergence check (single image overfit)
python run_tests.py --quick

# Run full convergence test (5000 iterations)
python run_tests.py --full

# Run all tests including slow ones
python run_tests.py --all

Expected Results:
=================

Test 4 is the CRITICAL quality gate:
- 5000 iterations on Lego scene
- Must achieve: 28+ dB PSNR (Instant-NGP baseline)
- Typical Instant-NGP: 30-32 dB at full convergence

If Test 4 fails (<28 dB), there is a fundamental implementation issue.
"""

import sys
import subprocess
from pathlib import Path


def run_tests(mode='fast'):
    """Run test suite with specified mode."""

    if mode == 'fast':
        # Run all tests except slow convergence tests
        print("\n" + "="*70)
        print("Running FAST tests (excludes convergence tests)")
        print("="*70 + "\n")
        cmd = ['python', '-m', 'pytest', 'tests/', '-v', '-m', 'not slow']

    elif mode == 'quick':
        # Run quick convergence check (single image overfit)
        print("\n" + "="*70)
        print("Running QUICK convergence test (500 iterations)")
        print("="*70 + "\n")
        cmd = ['python', '-m', 'pytest', 'tests/test_4_e2e_convergence.py::TestQuickOverfit', '-v', '-s']

    elif mode == 'full':
        # Run full convergence test (5000 iterations)
        print("\n" + "="*70)
        print("Running FULL convergence test (5000 iterations)")
        print("Target: 28+ dB PSNR on Lego scene")
        print("="*70 + "\n")
        cmd = ['python', '-m', 'pytest', 'tests/test_4_e2e_convergence.py::TestFullConvergence', '-v', '-s']

    elif mode == 'all':
        # Run everything
        print("\n" + "="*70)
        print("Running ALL tests (including slow convergence)")
        print("="*70 + "\n")
        cmd = ['python', '-m', 'pytest', 'tests/', '-v', '-s']

    else:
        print(f"Unknown mode: {mode}")
        return 1

    result = subprocess.run(cmd)
    return result.returncode


if __name__ == "__main__":
    if len(sys.argv) > 1:
        arg = sys.argv[1].lower()
        if arg in ['--quick', '-q']:
            exit_code = run_tests('quick')
        elif arg in ['--full', '-f']:
            exit_code = run_tests('full')
        elif arg in ['--all', '-a']:
            exit_code = run_tests('all')
        else:
            print(__doc__)
            exit_code = 0
    else:
        # Default: fast tests
        exit_code = run_tests('fast')

    sys.exit(exit_code)
"""Pytest configuration for NGP baseline tests."""
import pytest
import torch


def pytest_configure(config):
    """Configure pytest markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )


@pytest.fixture(scope="session")
def device():
    """Global device fixture."""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


@pytest.fixture(scope="session")
def seed():
    """Set random seed for reproducibility."""
    torch.manual_seed(1337)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(1337)
    return 1337

