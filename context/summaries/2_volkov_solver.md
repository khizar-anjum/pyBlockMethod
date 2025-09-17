# VolkovSolver Implementation Summary

## Overview
The `volkovSolver` class implements E.A. Volkov's block grid method for solving the Laplace equation on arbitrary polygonal domains. This method provides high-accuracy numerical solutions for boundary value problems with mixed Dirichlet/Neumann conditions.

## Mathematical Foundation

### Problem Formulation
Solves the boundary value problem:
- **PDE**: -Δu = 0 in domain Ω (Laplace equation)
- **Boundary**: νⱼ·u + (1-νⱼ)·∂ⱼu = φⱼ on boundary Γⱼ
  - νⱼ = 1: Dirichlet condition (specified value)
  - νⱼ = 0: Neumann condition (specified derivative)
  - φⱼ: Polynomial boundary condition on edge j

## Core Algorithm Steps

### 1. Block Covering (`find_block_covering`)
Creates a covering of the polygonal domain using three types of blocks:

#### First Kind Blocks (Vertices)
- Circular sectors centered at polygon vertices
- Angle equals interior angle at vertex
- Radius determined by distance to non-adjacent edges
- Number denoted as N

#### Second Kind Blocks (Edges)
- Half-disks centered on polygon edges
- Fill gaps between first kind blocks
- Adaptive sizing to prevent overlap
- Can be broken into smaller blocks if needed
- Number denoted as L (includes N)

#### Third Kind Blocks (Interior)
- Full disks covering remaining interior points
- Placed using greedy algorithm
- Size limited by distance to edges
- Number denoted as M (total blocks)

### 2. Solution Initialization (`initialize_solution`)
- Creates discretized grid with spacing δ
- Masks points outside polygon
- Assigns points to containing blocks
- Initializes solution arrays and parameters
- Calculates block parameters (α, β, θ, r₀)

### 3. Boundary Estimation (`estimate_solution_over_curved_boundaries`)
- Iteratively estimates solution on block boundaries
- Uses carrier functions and Poisson kernels
- Implements equations (4.14) and (4.15) from Volkov's book
- Iterates until max_iter reached

### 4. Interior Solution (`estimate_solution_over_inner_points`)
- Calculates solution at all interior grid points
- Uses equation (5.1) combining:
  - Q: Carrier function values
  - R: Poisson kernel values
  - Boundary estimates from step 3

## Key Mathematical Components

### Carrier Functions
Different formulations based on block type and boundary conditions:
- Handle mixed Dirichlet/Neumann conditions
- Account for singularities near boundaries
- Use logarithmic terms for certain configurations

### Poisson Kernels
Fundamental solutions for each block type:
- **First kind**: Complex formula with scaling factor λⱼ
- **Second kind**: Standard Poisson kernel with reflection
- **Third kind**: Simple radial Poisson kernel

### Coordinate Transformations
- Cartesian ↔ Polar conversions relative to block centers
- Reference angle tracking for consistent orientation
- Vectorized operations for efficiency

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

### Data Structures
- `blocks`: List of block objects
- `solution`: Masked array of solution values
- `inside_block_ids_`: Grid point to block mapping
- `boundary_estimates`: Solution on block boundaries
- `quantized_boundary_points`: Discretized boundary points

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

## Key Methods

- `solve()`: Main entry point, orchestrates solution process
- `find_block_covering()`: Creates block covering
- `calculate_carrier_function_value()`: Evaluates carrier functions
- `calculate_poisson_kernel_value()`: Evaluates Poisson kernels
- `plot_block_covering()`: Visualizes block arrangement
- `plot_solution()`: Creates solution heatmap
- `plot_gradient()`: Shows solution gradient field

## Error Handling
- Verifies counterclockwise vertex ordering
- Checks boundary condition count matches edges
- Validates heuristic parameter ranges
- Optional solution uniqueness verification
- Handles degenerate geometries gracefully