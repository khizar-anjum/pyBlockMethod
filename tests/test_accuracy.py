#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Accuracy-based tests for the Volkov solver.

These tests validate numerical accuracy by comparing solutions against
pre-computed reference data (golden baseline) using Mean Squared Error (MSE).

Key features:
- 72 reference solutions covering 2 geometries × 2 n × 2 radial × 3 overlap × 3 boundary conditions
- Machine precision comparison (MSE tolerance ~1e-14) for perfect regression detection
- Only tests stable radial_heuristic values (0.85, 0.95) which show perfect reproducibility
- Removed radial_heuristic=0.75 cases due to algorithmic non-determinism

Tolerance rationale:
- MSE tolerance of 1e-14 (machine precision) - stable parameters show perfect reproducibility
- Block counts match exactly (deterministic for stable parameters)
- Statistics match to machine precision
- Any deviation indicates a real algorithmic change
"""

import numpy as np
import pytest
from pathlib import Path
from pyBlockGrid import polygon, volkovSolver

# Test configuration (after removing unstable rad=0.75 cases)
TOLERANCE_MSE = 1e-14       # Relative MSE tolerance - stable parameters show perfect reproducibility
TOLERANCE_STATS = 1e-12     # Relative tolerance for statistics (machine precision)
TOLERANCE_ABSOLUTE = 1e-14  # Absolute tolerance for near-zero values
TOLERANCE_BLOCK_COUNT = 0   # Exact match for block counts (deterministic for stable parameters)

# Path to reference data
REFERENCE_DATA_DIR = Path(__file__).parent / 'reference_data'

def load_reference_data(filename):
    """Load reference solution data from file."""
    filepath = REFERENCE_DATA_DIR / filename
    if not filepath.exists():
        pytest.skip(f"Reference data not found: {filename}")

    data = np.load(filepath, allow_pickle=True)
    return {
        'solution': data['solution'],
        'mask': data['mask'],
        'block_counts': data['block_counts'].item(),
        'parameters': data['parameters'].item(),
        'solution_stats': data['solution_stats'].item()
    }

def generate_current_solution(params):
    """Generate solution with current implementation."""
    # Create polygon
    poly = polygon(params['vertices'])

    # Create solver
    solver = volkovSolver(
        poly=poly,
        boundary_conditions=params['boundary_conditions'],
        is_dirichlet=params['is_dirichlet'],
        delta=params['delta'],
        n=params['n'],
        max_iter=params['max_iter'],
        radial_heuristic=params['radial_heuristic'],
        overlap_heuristic=params['overlap_heuristic']
    )

    # Solve
    solution = solver.solve(verbose=False)

    # Get block covering info
    N, L, M, _ = solver.find_block_covering()

    return solution, {'N': N, 'L': L, 'M': M}

def compare_solutions(current_solution, reference_solution, reference_mask):
    """Compare current solution against reference using MSE tolerance."""
    # Reconstruct reference masked array
    ref_masked = np.ma.masked_array(reference_solution, mask=reference_mask)

    # Check shapes match
    assert current_solution.shape == ref_masked.shape, \
        f"Solution shape mismatch: {current_solution.shape} vs {ref_masked.shape}"

    # Check masks match
    np.testing.assert_array_equal(current_solution.mask, ref_masked.mask,
                                err_msg="Solution masks don't match")

    # Compare using Mean Squared Error
    current_valid = current_solution.compressed()
    ref_valid = ref_masked.compressed()

    # Calculate MSE
    mse = np.mean((current_valid - ref_valid) ** 2)

    # Calculate relative MSE (normalized by mean square of reference)
    ref_mean_square = np.mean(ref_valid ** 2)
    if ref_mean_square > TOLERANCE_ABSOLUTE:
        relative_mse = mse / ref_mean_square
        assert relative_mse < TOLERANCE_MSE, \
            f"Solution MSE too high: relative_mse={relative_mse:.2e}, tolerance={TOLERANCE_MSE:.2e}"
    else:
        # Handle case where reference solution is near zero
        assert mse < TOLERANCE_ABSOLUTE, \
            f"Solution MSE too high for near-zero reference: mse={mse:.2e}"

def compare_statistics(current_solution, reference_stats):
    """Compare solution statistics."""
    current_stats = {
        'min': float(current_solution.min()),
        'max': float(current_solution.max()),
        'mean': float(current_solution.mean()),
        'std': float(current_solution.std()),
        'valid_points': int(current_solution.compressed().size)
    }

    for stat_name, ref_value in reference_stats.items():
        current_value = current_stats[stat_name]

        if stat_name == 'valid_points':
            # Exact match for point count
            assert current_value == ref_value, \
                f"Valid points mismatch: {current_value} vs {ref_value}"
        elif abs(ref_value) < TOLERANCE_ABSOLUTE:
            # Near zero - use absolute tolerance
            assert abs(current_value - ref_value) < TOLERANCE_ABSOLUTE, \
                f"Statistic {stat_name} mismatch: {current_value} vs {ref_value} (absolute)"
        else:
            # Non-zero - use relative tolerance
            rel_error = abs(current_value - ref_value) / abs(ref_value)
            assert rel_error < TOLERANCE_STATS, \
                f"Statistic {stat_name} mismatch: {current_value} vs {ref_value} (relative error: {rel_error})"

def compare_block_counts(current_blocks, reference_blocks):
    """Compare block counts (should match exactly for stable parameters)."""
    for count_type in ['N', 'L', 'M']:
        current_count = current_blocks[count_type]
        ref_count = reference_blocks[count_type]
        assert current_count == ref_count, \
            f"Block count {count_type} mismatch: {current_count} vs {ref_count}"

# Generate test cases by discovering reference data files
def get_test_cases():
    """Generate test case parameters from reference data files."""
    if not REFERENCE_DATA_DIR.exists():
        return []

    test_cases = []
    for filename in REFERENCE_DATA_DIR.glob('*.npz'):
        test_cases.append(filename.name)

    return sorted(test_cases)

# Parametrized test function
@pytest.mark.parametrize("reference_filename", get_test_cases())
def test_solution_accuracy(reference_filename):
    """Test that current solution matches reference data with machine precision.

    After removing unstable radial_heuristic=0.75 cases, all remaining parameter
    combinations show perfect reproducibility.
    """
    # Load reference data
    reference = load_reference_data(reference_filename)

    # Generate current solution
    current_solution, current_blocks = generate_current_solution(reference['parameters'])

    # Compare solutions using MSE (should be essentially zero)
    compare_solutions(current_solution, reference['solution'], reference['mask'])

    # Compare statistics (should match exactly)
    compare_statistics(current_solution, reference['solution_stats'])

    # Compare block counts (should match exactly)
    compare_block_counts(current_blocks, reference['block_counts'])

# Additional focused tests for specific geometries
class TestUnitSquare:
    """Focused tests for unit square geometry."""

    def test_hot_bottom_basic(self):
        """Test unit square with hot bottom edge - basic parameters."""
        filename = "unit_square_n10_rad0.85_ovl0.1_hot_bottom.npz"
        reference = load_reference_data(filename)

        current_solution, current_blocks = generate_current_solution(reference['parameters'])

        # Basic checks
        assert current_solution.min() >= -TOLERANCE_ABSOLUTE, "Solution has negative values"
        assert current_solution.max() <= 1.0 + TOLERANCE_ABSOLUTE, "Solution exceeds boundary value"

        # Detailed comparison
        compare_solutions(current_solution, reference['solution'], reference['mask'])
        compare_statistics(current_solution, reference['solution_stats'])
        compare_block_counts(current_blocks, reference['block_counts'])

class TestLShape:
    """Focused tests for L-shaped geometry."""

    def test_hot_left_basic(self):
        """Test L-shape with hot left edge - basic parameters."""
        filename = "l_shape_n10_rad0.85_ovl0.1_hot_left.npz"
        reference = load_reference_data(filename)

        current_solution, current_blocks = generate_current_solution(reference['parameters'])

        # Basic checks
        assert current_solution.min() >= -TOLERANCE_ABSOLUTE, "Solution has negative values"
        assert current_solution.max() <= 1.0 + TOLERANCE_ABSOLUTE, "Solution exceeds boundary value"

        # Detailed comparison
        compare_solutions(current_solution, reference['solution'], reference['mask'])
        compare_statistics(current_solution, reference['solution_stats'])
        compare_block_counts(current_blocks, reference['block_counts'])

# Test data integrity
def test_reference_data_exists():
    """Verify that reference data directory and files exist."""
    assert REFERENCE_DATA_DIR.exists(), f"Reference data directory not found: {REFERENCE_DATA_DIR}"

    reference_files = list(REFERENCE_DATA_DIR.glob('*.npz'))
    assert len(reference_files) > 0, "No reference data files found"
    assert len(reference_files) == 72, f"Expected 72 reference files (stable parameters only), found {len(reference_files)}"

def test_reference_data_format():
    """Verify that reference data files have correct format."""
    # Test one file to verify format
    test_files = list(REFERENCE_DATA_DIR.glob('unit_square_n10_rad0.85_ovl0.1_hot_bottom.npz'))
    if not test_files:
        pytest.skip("No reference file found for format testing")

    data = np.load(test_files[0], allow_pickle=True)

    # Check required keys
    required_keys = ['solution', 'mask', 'block_counts', 'parameters', 'solution_stats']
    for key in required_keys:
        assert key in data.files, f"Missing required key: {key}"

    # Check data types
    assert isinstance(data['solution'], np.ndarray), "Solution should be numpy array"
    assert isinstance(data['mask'], np.ndarray), "Mask should be numpy array"
    assert isinstance(data['block_counts'].item(), dict), "Block counts should be dict"
    assert isinstance(data['parameters'].item(), dict), "Parameters should be dict"
    assert isinstance(data['solution_stats'].item(), dict), "Solution stats should be dict"

if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])