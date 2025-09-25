# Codebase Structure Summary

## Overview
This project implements the Volkov Method for solving Laplace equations on polygonal domains, based on E.A. Volkov's book "Block Method for Solving the Laplace Equation and for Constructing Conformal Mappings (1994)".

**REFACTORING STATUS**: ✅ **COMPLETED** - The monolithic volkov.py (980 lines) has been successfully refactored into a modular architecture with ~484 lines in the main solver and mathematical algorithms separated into dedicated modules.

**HOLE SUPPORT STATUS**: ✅ **IMPLEMENTED** - The solver now fully supports polygonal domains with holes, including proper block generation, boundary condition handling, and visualization for multiply-connected domains.

## Important: Virtual Environment
**Always use the virtual environment when running code:**
```bash
source venv/bin/activate  # Activate the venv first
python examples/script.py  # Then run scripts
```

## Directory Structure

### Root Level
- **setup.py**: Package installation configuration
- **pyproject.toml**: Modern Python project configuration
- **requirements-test.txt**: Testing dependencies (pytest)
- **LICENSE**: MIT License
- **README.md**: Basic project description

### src/pyBlockGrid/
Main package directory containing the implementation:

#### Core Components (`core/`)
- **polygon.py** (362 lines): Defines the `polygon` class for representing polygonal domains
  - Handles vertex ordering verification (counterclockwise)
  - Calculates interior angles, edges, and inner connections
  - Provides area and perimeter calculations
  - Implements point-in-polygon testing using ray tracing
  - **NEW**: Supports holes via `add_hole()` method and hole-aware `is_inside()`
  - **NEW**: `_point_in_polygon()` method for single-point checks with holes
  - **NEW**: `area_with_holes()` for effective domain area calculation
  - Includes plotting methods with hole visualization

- **polygon_hole.py** (263 lines): **NEW** - Defines the `PolygonHole` class for holes
  - Represents clockwise-oriented holes within polygons
  - Maintains boundary conditions for each hole edge
  - Provides hole-specific geometric calculations
  - Implements point-in-hole testing
  - Validates hole geometry and placement

- **block.py** (53 lines): Defines the `block` class for the block grid method
  - Three types of blocks:
    1. First kind: Sectors extending from vertices
    2. Second kind: Half-disks extending from edges
    3. Third kind: Full disks inside the polygon
  - Stores block geometry (center, angle, radii r and r0)
  - **NEW**: Tracks boundary association (main polygon vs holes)
  - **NEW**: Includes `boundary_type`, `boundary_id`, and `vertex_id` for hole support

- **solution_state.py** (316 lines): **NEW** - Comprehensive data structure for solution state management
  - Organizes all arrays and parameters needed during computation
  - Factory method for initialization from polygon and blocks
  - Handles grid creation, masking, and parameter setup
  - **UPDATED**: Extended to handle hole boundaries and their conditions
  - **NEW**: Flattens boundary conditions from main polygon and all holes
  - **NEW**: Tracks boundary segments with (boundary_type, boundary_id, edge_id)

- **block_covering.py** (957 lines): **NEW** - Extracted block covering strategies
  - Implements algorithms for creating all three block types
  - Handles block placement, overlap detection, and coverage verification
  - Contains the core logic for finding uncovered points and creating third kind blocks
  - **UPDATED**: All block creation methods now hole-aware
  - **NEW**: `_get_all_constraining_edges()` for unified edge handling
  - **NEW**: `_distances_from_all_edges_filtered()` for proper edge exclusion
  - **NEW**: Hole-aware methods for all three block kinds

#### Mathematical Module (`mathematical/`) - **NEW**
**Purpose**: Pure mathematical algorithms extracted from the monolithic solver

- **__init__.py** (25 lines): Module exports for mathematical components
- **carrier_functions.py** (148 lines): **NEW** - Pure mathematical carrier function calculations
  - Handles all four boundary condition combinations for different block types
  - Extracted from the original complex conditional logic
- **poisson_kernels.py** (134 lines): **NEW** - Poisson kernel calculations for all block types
  - Implements equations (3.12), (3.13), (3.15), (3.16), (3.17) from Volkov's book
- **boundary_estimator.py** (102 lines): **NEW** - Iterative boundary solution estimation
  - Implements equations (4.14) and (4.15) from Volkov's book
- **interior_solver.py** (154 lines): **NEW** - Interior point solution computation
  - Implements equation (5.1) for computing solution at interior grid points

#### Solvers (`solvers/`)
- **base.py** (17 lines): Abstract base class `PDESolver` defining the solver interface
- **volkov.py** (540 lines): **REFACTORED** - Clean orchestrator of the Volkov method
  - **Reduced from 980 to 540 lines** (~45% reduction)
  - Now acts as a clean orchestrator delegating to specialized modules
  - Maintains exact same API and functionality
  - Improved readability and maintainability
  - **UPDATED**: Grid creation now uses hole-aware `is_inside()` method
  - **NEW**: `create_third_kind_blocks()` handles holes properly
  - **NEW**: `_validate_third_kind_block()` ensures blocks don't overlap holes
- **volkov_original.py** (980 lines): Original monolithic implementation preserved for reference

#### Utilities (`utils/`)
- **geometry.py** (80 lines): Geometric utility functions
  - Random polygon generation for testing
- **coordinate_transforms.py** (141 lines): **NEW** - Coordinate transformation utilities
  - Cartesian ↔ Polar conversions relative to block centers
  - Vectorized batch transformations for efficiency

#### Visualization (`visualization/`)
- **plotting.py** (268 lines): General visualization functions
  - Multi-polygon comparison plots
  - Heuristic analysis plots
- **volkov_plots.py** (223 lines): **NEW** - Volkov-specific visualization functions
  - Block covering visualization with proper block type coloring
  - Solution heatmaps with masked array support
  - Gradient field visualization

### tests/
- **test_solver.py**: Functional tests using pytest (3 tests)
  - Tests polygon geometric calculations (area, convexity, vertex validation)
  - Tests block covering mathematical relationships (N, L, M counts)
  - Clean, focused tests for core component functionality
- **test_accuracy.py**: Machine-precision accuracy tests (76 tests)
  - MSE-based validation against 72 reference solutions
  - Tests all stable parameter combinations (radial_heuristic ≥ 0.85)
  - Infrastructure tests for reference data integrity
- **generate_reference_data.py**: Utility script for generating golden baseline solutions
- **reference_data/**: 72 compressed .npz files (732KB) containing reference solutions
- **conftest.py**: Pytest configuration

### examples/
- **simple_square.py**: Basic square domain example with temperature distribution
- **simple_polygon.py**: General polygon example
- **complex_polygon.py**: Complex geometry demonstrations
- **random_polygons.py**: Random polygon generation and solving
- **simple_square_with_hole.py**: **NEW** - Square domain with square hole demonstration
  - Shows hole creation and validation
  - Demonstrates block covering for multiply-connected domains
  - Visualizes valid domain (inside main, outside hole)

### context/summaries/
Documentation directory containing these summary files.

## Refactoring Achievements

### 📊 Metrics
- **Original**: volkov.py = 980 lines (monolithic)
- **Refactored**: volkov.py = 484 lines (~50% reduction)
- **New Modules**: 7 new files for mathematical algorithms
- **Total Lines**: ~2400 lines well-organized across 17 files

### 🎯 Key Improvements
1. **Separation of Concerns**: Mathematical algorithms now in dedicated `mathematical/` module
2. **Maintainability**: Each component has a single, clear responsibility
3. **Testability**: Individual mathematical functions can be tested in isolation
4. **Extensibility**: New block types or algorithms can be added easily
5. **Readability**: Main solver now acts as a clear orchestrator

## Key Design Patterns

1. **Modular Architecture**: Clean separation between mathematical algorithms, core data structures, and solver orchestration
2. **Abstract Base Classes**: PDESolver provides interface for future solver implementations
3. **Factory Pattern**: SolutionState.from_polygon_and_blocks() for complex object initialization
4. **Strategy Pattern**: Different algorithms for each block type in block_covering.py
5. **Vectorized Operations**: Heavy use of NumPy for efficient numerical computation
6. **Masked Arrays**: Efficient handling of irregular domains within rectangular grids
7. **Specialized Visualization**: Separate plotting modules for different visualization needs

## Dependencies
- **NumPy**: Core numerical operations and array handling
- **Matplotlib**: Visualization and plotting
- **pytest**: Testing framework (dev dependency)

## Package Import Structure
```python
from pyBlockGrid import polygon, block, volkovSolver
```

The package exposes three main classes at the top level for ease of use.

## Refactoring Architecture

### Mathematical Module Hierarchy
```
mathematical/
├── __init__.py (exports)
├── carrier_functions.py (boundary condition calculations)
├── poisson_kernels.py (fundamental solutions)
├── boundary_estimator.py (iterative boundary estimation)
└── interior_solver.py (interior point calculations)
```

### Core Data Structures
```
core/
├── polygon.py (domain geometry)
├── block.py (block definitions)
├── solution_state.py (solution data organization)
└── block_covering.py (block placement algorithms)
```

### Solver Architecture
```
solvers/
├── base.py (abstract interface)
├── volkov.py (refactored orchestrator)
└── volkov_original.py (preserved original)
```