# Codebase Structure Summary

## Overview
This project implements the Volkov Method for solving Laplace equations on polygonal domains, based on E.A. Volkov's book "Block Method for Solving the Laplace Equation and for Constructing Conformal Mappings (1994)".

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
- **polygon.py**: Defines the `polygon` class for representing polygonal domains
  - Handles vertex ordering verification (counterclockwise)
  - Calculates interior angles, edges, and inner connections
  - Provides area and perimeter calculations
  - Implements point-in-polygon testing using ray tracing
  - Includes plotting methods

- **block.py**: Defines the `block` class for the block grid method
  - Three types of blocks:
    1. First kind: Sectors extending from vertices
    2. Second kind: Half-disks extending from edges  
    3. Third kind: Full disks inside the polygon
  - Stores block geometry (center, angle, radii r and r0)
  - Tracks block relationships and overlaps

#### Solvers (`solvers/`)
- **base.py**: Abstract base class `PDESolver` defining the solver interface
- **volkov.py**: Main `volkovSolver` class implementing the Volkov method
  - Complex numerical solver with ~950 lines of code
  - Handles Dirichlet and Neumann boundary conditions
  - Implements block covering algorithm
  - Solves using carrier functions and Poisson kernels

#### Utilities (`utils/`)
- **geometry.py**: Geometric utility functions
  - Random polygon generation for testing

#### Visualization (`visualization/`)
- **plotting.py**: Visualization functions
  - Block covering visualization
  - Solution heatmaps
  - Gradient vector fields
  - Heuristic analysis plots
  - Multi-polygon comparison plots

### tests/
- **test_solver.py**: Unit tests using pytest
  - Tests polygon functionality
  - Tests solver basic operations
  - Tests block covering algorithm
  - Tests plotting functionality
- **conftest.py**: Pytest configuration
- **run_examples.py**: Example runner script

### examples/
- **simple_square.py**: Basic square domain example with temperature distribution
- **simple_polygon.py**: General polygon example
- **complex_polygon.py**: Complex geometry demonstrations
- **random_polygons.py**: Random polygon generation and solving

### context/summaries/
Documentation directory containing these summary files.

## Key Design Patterns

1. **Object-Oriented Architecture**: Clean separation between polygon geometry, blocks, and solver
2. **Abstract Base Classes**: PDESolver provides interface for future solver implementations
3. **Vectorized Operations**: Heavy use of NumPy for efficient numerical computation
4. **Masked Arrays**: Efficient handling of irregular domains within rectangular grids
5. **Modular Visualization**: Separate plotting module for different visualization needs

## Dependencies
- **NumPy**: Core numerical operations and array handling
- **Matplotlib**: Visualization and plotting
- **pytest**: Testing framework (dev dependency)

## Package Import Structure
```python
from pyBlockGrid import polygon, block, volkovSolver
```

The package exposes three main classes at the top level for ease of use.