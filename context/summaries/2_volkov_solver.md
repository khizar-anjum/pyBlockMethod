# VolkovSolver Implementation Summary

## Overview
The `volkovSolver` class implements E.A. Volkov's block grid method for solving the Laplace equation on arbitrary polygonal domains. This method provides high-accuracy numerical solutions for boundary value problems with mixed Dirichlet/Neumann conditions.

**REFACTORING STATUS**: ✅ **COMPLETED** - The monolithic 980-line implementation has been successfully refactored into a modular orchestrator that delegates to specialized mathematical and algorithmic modules while maintaining complete backward compatibility.

**HOLE SUPPORT STATUS**: ✅ **IMPLEMENTED** - The solver now fully supports multiply-connected domains (polygons with holes), with all three block types properly handling hole constraints and boundary conditions.

## Mathematical Foundation

### Problem Formulation
Solves the boundary value problem:
- **PDE**: -Δu = 0 in domain Ω (Laplace equation)
- **Domain**: Simply or multiply-connected polygonal regions
- **Boundary**: νⱼ·u + (1-νⱼ)·∂ⱼu = φⱼ on boundary Γⱼ
  - νⱼ = 1: Dirichlet condition (specified value)
  - νⱼ = 0: Neumann condition (specified derivative)
  - φⱼ: Polynomial boundary condition on edge j
  - **NEW**: Separate boundary conditions for main polygon and each hole

## Core Algorithm Steps

### 1. Block Covering (`find_block_covering`)
**Implementation**: Delegates to `BlockCovering` class in `core/block_covering.py`

Creates a covering of the polygonal domain using three types of blocks:

#### First Kind Blocks (Vertices)
- Circular sectors centered at polygon vertices
- Angle equals interior angle at vertex
- Radius determined by distance to non-adjacent edges
- Number denoted as N
- **NEW**: Created for both main polygon vertices and hole vertices
- **NEW**: Considers all constraining edges (main + holes) for radius
- **Implementation**: `BlockCovering._create_first_kind_blocks()`

#### Second Kind Blocks (Edges)
- Half-disks centered on polygon edges
- Fill gaps between first kind blocks
- Adaptive sizing to prevent overlap
- Can be broken into smaller blocks if needed
- Number denoted as L (includes N)
- **NEW**: Created for both main polygon edges and hole edges
- **NEW**: Uses filtered edge constraints to avoid zero distances
- **Implementation**: `BlockCovering._create_second_kind_blocks()`

#### Third Kind Blocks (Interior)
- Full disks covering remaining interior points
- Placed using greedy algorithm
- Size limited by distance to edges
- Number denoted as M (total blocks)
- **NEW**: Validates placement to ensure blocks don't overlap holes
- **NEW**: Uses all constraining edges (main + holes) for sizing
- **Implementation**: `BlockCovering._create_third_kind_blocks()`

### 2. Solution Initialization (`initialize_solution`)
**Implementation**: Uses `SolutionState.from_polygon_and_blocks()` factory method

- Creates discretized grid with spacing δ (done in `__init__` for proper execution order)
- Masks points outside polygon
- Assigns points to containing blocks
- Initializes solution arrays and parameters
- Calculates block parameters (α, β, θ, r₀)

### 3. Boundary Estimation (`estimate_solution_over_curved_boundaries`)
**Implementation**: Delegates to `BoundaryEstimator` class in `mathematical/boundary_estimator.py`

- Iteratively estimates solution on block boundaries
- Uses carrier functions and Poisson kernels
- Implements equations (4.14) and (4.15) from Volkov's book
- Iterates until max_iter reached

### 4. Interior Solution (`estimate_solution_over_inner_points`)
**Implementation**: Delegates to `InteriorSolver` class in `mathematical/interior_solver.py`

- Calculates solution at all interior grid points
- Uses equation (5.1) combining:
  - Q: Carrier function values (from `CarrierFunctions`)
  - R: Poisson kernel values (from `PoissonKernels`)
  - Boundary estimates from step 3

## Key Mathematical Components (Now Modularized)

### Carrier Functions (`mathematical/carrier_functions.py`)
**Class**: `CarrierFunctions` - Extracted pure mathematical calculations

Different formulations based on block type and boundary conditions:
- Handle mixed Dirichlet/Neumann conditions
- Account for singularities near boundaries
- Use logarithmic terms for certain configurations
- **Key Method**: `calculate(block_kind, block_boundary_identifier, r_, theta_, k, a_, b_, alpha_j)`

### Poisson Kernels (`mathematical/poisson_kernels.py`)
**Class**: `PoissonKernels` - Fundamental solutions for each block type

- **First kind**: Complex formula with scaling factor λⱼ
- **Second kind**: Standard Poisson kernel with reflection
- **Third kind**: Simple radial Poisson kernel
- **Key Method**: `calculate(block_kind, nu_i, nu_j, r_, r0_, theta_, eta_, alpha_j)`
- **Implements**: Equations (3.12), (3.13), (3.15), (3.16), (3.17) from Volkov's book

### Coordinate Transformations (`utils/coordinate_transforms.py`)
**Class**: `CoordinateTransforms` - Extracted transformation utilities

- Cartesian ↔ Polar conversions relative to block centers
- Reference angle tracking for consistent orientation
- Vectorized operations for efficiency
- **Methods**: `cartesian_to_polar()`, `polar_to_cartesian()`, batch versions

### Boundary Estimation (`mathematical/boundary_estimator.py`)
**Class**: `BoundaryEstimator` - Iterative boundary solution algorithm

- **Key Method**: `estimate(state, max_iter=100)`
- **Implements**: Equations (4.14) and (4.15) from Volkov's book
- Manages iterative convergence for boundary values

### Interior Solution (`mathematical/interior_solver.py`)
**Class**: `InteriorSolver` - Interior point calculation algorithm

- **Key Method**: `solve(state, tolerance=1e-10)`
- **Implements**: Equation (5.1) from Volkov's book
- Computes final solution at all interior grid points

## Parameters

### Required
- `poly`: Polygon object defining domain
- `boundary_conditions`: List of polynomial coefficients per edge
- `is_dirichlet`: Boolean flags for boundary type per edge

### Numerical Parameters
- `n`: Angular divisions (default 10, controls accuracy)
- `delta`: Grid spacing (default 0.01, resolution)
- `max_iter`: Iteration limit (default 100)
- `tolerance`: Floating point tolerance (default 1e-10)

### Heuristic Parameters
- `radial_heuristic`: Block size scaling (0.707-1.0, default 0.8)
- `overlap_heuristic`: Block overlap control (0-0.5, default 0.1)
- `verify_solution`: Check uniqueness condition (default False)

## Implementation Details

### Efficiency Optimizations
- Vectorized NumPy operations throughout
- Masked arrays for irregular domains
- Pre-computed block relationships
- Batch processing of grid points

### Numerical Stability
- Special handling for singularities
- Adaptive block breaking for edge cases
- Tolerance checks for near-zero values
- Verification of solution uniqueness

### Data Structures (Now in SolutionState)
**Centralized in**: `core/solution_state.py` - `SolutionState` class

- `blocks`: List of block objects
- `solution`: Masked array of solution values
- `inside_block_ids`: Grid point to block mapping
- `boundary_estimates`: Solution on block boundaries
- `quantized_boundary_points`: Discretized boundary points
- **Factory Method**: `SolutionState.from_polygon_and_blocks()` for initialization

## Solution Process Flow

1. **Setup Phase**
   - Create rectangular grid encompassing polygon
   - Mask exterior points
   - Initialize solution arrays

2. **Block Generation**
   - Place first kind blocks at vertices
   - Fill edge gaps with second kind blocks
   - Cover interior with third kind blocks
   - Verify complete coverage

3. **Numerical Solution**
   - Compute carrier functions for boundary conditions
   - Calculate Poisson kernels for block interactions
   - Iterate boundary estimates
   - Compute final interior solution

4. **Output**
   - Return masked array with solution values
   - Provide visualization methods
   - Support gradient computation

## Key Methods (Refactored Architecture)

### Main Orchestrator (volkov.py)
- `solve()`: Main entry point, orchestrates solution process (now ~484 lines vs 980)
- `find_block_covering()`: Delegates to `BlockCovering` class
- `initialize_solution()`: Uses `SolutionState.from_polygon_and_blocks()`
- `plot_block_covering()`: Delegates to `volkov_plots.plot_block_covering()`
- `plot_solution()`: Delegates to `volkov_plots.plot_solution_heatmap()`
- `plot_gradient()`: Delegates to `volkov_plots.plot_gradient_field()`

### Extracted Mathematical Methods
- `CarrierFunctions.calculate()`: Pure mathematical carrier function evaluation
- `PoissonKernels.calculate()`: Pure mathematical Poisson kernel evaluation
- `BoundaryEstimator.estimate()`: Boundary iteration algorithm
- `InteriorSolver.solve()`: Interior point solution calculation

### Extracted Algorithmic Methods
- `BlockCovering._create_first_kind_blocks()`: Vertex block creation
- `BlockCovering._create_second_kind_blocks()`: Edge block creation
- `BlockCovering._create_third_kind_blocks()`: Interior block creation
- `CoordinateTransforms.cartesian_to_polar()`: Coordinate conversions

## Hole Support

### Creating Domains with Holes
```python
# Create main polygon
poly = polygon(main_vertices)

# Add holes with boundary conditions
hole = poly.add_hole(hole_vertices, hole_boundary_conditions, hole_is_dirichlet)
```

### Key Implementation Details
1. **Hole Representation**:
   - Holes use `PolygonHole` class with clockwise vertices
   - Each hole maintains its own boundary conditions
   - Automatic validation of hole placement

2. **Block Generation**:
   - First kind blocks created at all vertices (main + holes)
   - Second kind blocks placed on all edges (main + holes)
   - Third kind blocks validated to not overlap with holes

3. **Grid Creation**:
   - `polygon.is_inside()` now excludes points inside holes
   - Vectorized implementation for efficiency
   - Proper masking of invalid domain points

4. **Edge Constraints**:
   - `_get_all_constraining_edges()` unifies edge handling
   - `_distances_from_all_edges_filtered()` excludes current edge
   - Prevents zero-distance issues for blocks on edges

## Error Handling
- Verifies counterclockwise vertex ordering (main polygon)
- Verifies clockwise vertex ordering (holes)
- Validates holes are inside main polygon
- Checks holes don't overlap
- Checks boundary condition count matches edges
- Validates heuristic parameter ranges
- Optional solution uniqueness verification
- Handles degenerate geometries gracefully