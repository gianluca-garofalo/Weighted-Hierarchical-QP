#!/usr/bin/env python3
"""
Test script for pyhqp Python bindings.
"""

import os
import sys

import numpy as np

# Add the current directory to the path so we can import pyhqp
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    import pyhqp
except ImportError as e:
    print(f"Failed to import pyhqp: {e}")
    print("Make sure to build the module first with:")
    print("  python setup.py build_ext --inplace")
    sys.exit(1)


def test_basic_solver_creation():
    """Test that we can create a HierarchicalQP solver."""
    solver = pyhqp.HierarchicalQP(3, 2)
    assert solver is not None

    # Test that tolerance is accessible
    assert hasattr(solver, "tolerance")
    default_tolerance = solver.tolerance
    assert default_tolerance == 1e-9

    # Test that we can change tolerance
    solver.tolerance = 1e-6
    assert solver.tolerance == 1e-6


def test_basic_problem_solving():
    """
    Test basic problem solving functionality.

    This example solves a simple 2-variable hierarchical QP problem:
    - Higher priority task: x[0] = 0 (constraint)
    - Lower priority task: x[1] = 8 (objective)
    """

    solver = pyhqp.HierarchicalQP(3, 2)

    # Set up the problem
    A = np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 1.0]])

    # Use large finite values instead of infinity
    large_val = 1e9
    lower = np.array([0.0, 8.0, -large_val])
    upper = np.array([0.0, large_val, 8.0])
    breaks = np.array([1, 3], dtype=np.int32)

    # Set the problem
    solver.set_problem(A, lower, upper, breaks)

    # Solve
    solution = solver.get_primal()

    # Check the solution
    expected_solution = np.array([0.0, 8.0])
    assert np.allclose(
        solution, expected_solution, atol=1e-6
    ), f"Expected {expected_solution}, got {solution}"


def test_metric_setting():
    """Test setting a custom metric matrix."""
    solver = pyhqp.HierarchicalQP(2, 2)

    # Create a custom metric
    metric = np.array([[2.0, 0.0], [0.0, 1.0]])

    # This should not raise an exception
    solver.set_metric(metric)


def test_problem_dimensions():
    """Test that the solver handles different problem dimensions."""
    # Test different sizes
    for m, n in [(2, 2), (5, 3), (10, 5)]:
        solver = pyhqp.HierarchicalQP(m, n)
        assert solver is not None


def run_all_tests():
    """Run all tests manually."""
    test_functions = [
        test_basic_solver_creation,
        test_basic_problem_solving,
        test_metric_setting,
        test_problem_dimensions,
    ]

    passed = 0
    failed = 0

    for test_func in test_functions:
        try:
            print(f"Running {test_func.__name__}... ", end="")
            test_func()
            print("PASSED")
            passed += 1
        except Exception as e:
            print(f"FAILED: {e}")
            failed += 1

    print(f"\nTest Results: {passed} passed, {failed} failed")
    return failed == 0


if __name__ == "__main__":
    print("Testing pyhqp Python bindings...")
    print("=" * 50)

    success = run_all_tests()

    if success:
        print("\nAll tests passed! ✅")
        sys.exit(0)
    else:
        print("\nSome tests failed! ❌")
        sys.exit(1)
