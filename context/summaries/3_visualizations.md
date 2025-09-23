# Visualization Components Summary

## Overview
The visualization module provides comprehensive plotting capabilities for understanding the Volkov method's operation, from block covering to solution visualization and analysis.

**REFACTORING STATUS**: ✅ **COMPLETED** - Visualization functionality has been reorganized into two modules:
- `visualization/volkov_plots.py` (223 lines): Volkov-specific plotting functions extracted from the main solver
- `visualization/plotting.py` (268 lines): General analysis and comparison plots

## Main Visualization Functions

## Volkov-Specific Visualization (`volkov_plots.py`)

### 1. Block Covering Visualization
**Method**: `volkovSolver.plot_block_covering()` → Delegates to `plot_block_covering()`

Displays the block grid structure overlaid on the polygon:
- **First Kind Blocks (Blue)**
  - Circular sectors at vertices
  - Solid line for inner radius (r)
  - Dashed line for outer radius (r₀)
  - Center points labeled as P₁, P₂, etc.

- **Second Kind Blocks (Red)**
  - Half-disks along edges
  - Solid and dashed radii as above
  - Sequential numbering after vertices

- **Third Kind Blocks (Green)**
  - Full circles in interior
  - Cover remaining uncovered points
  - Numbered continuing from second kind

**Optional Features**:
- Boundary condition labels on edges (φⱼ values)
- Quantized boundary points with solution values
- Uncovered points highlighting (red dots)

### 2. Solution Heatmap
**Method**: `volkovSolver.plot_solution()` → Delegates to `plot_solution_heatmap()`

Visualizes the computed solution as a color-coded heatmap:
- Uses `pcolormesh` for efficient rendering
- Masked array handling for irregular domains
- Customizable colormap (default: viridis)
- Automatic colorbar with scale
- Optional vmin/vmax for consistent scaling across plots

### 3. Gradient Field Visualization
**Method**: `volkovSolver.plot_gradient()` → Delegates to `plot_gradient_field()`

Shows the solution gradient as a vector field:
- Computes gradients using `np.gradient`
- Normalizes vectors for uniform arrow length
- Quiver plot overlaid on domain
- Parameters:
  - `decimation_factor`: Reduces vector density (default 2)
  - `scale`: Controls arrow size (default 20)

## General Analysis Visualization (`plotting.py`)

### 4. Three-Polygon Comparison
**Function**: `plot_three_polygons_block_covering()` (in `plotting.py`)

Creates publication-ready figures of three standard geometries:
- Rectangle (3:1 aspect ratio)
- Obtuse triangle (2:1 aspect ratio)
- Regular hexagon (1:1 aspect ratio)

Features:
- Removes axes and labels for clean presentation
- Saves each as separate PDF
- Consistent width scaling
- Demonstrates block covering on different shapes

### 5. Solution Steps Visualization
**Function**: `plot_3by3_solution_steps()` (in `plotting.py`)

Creates comprehensive 3x3 grid showing complete solution process:

**Row 1**: Block Covering
- Three different polygons side by side
- Shows all block types and arrangements

**Row 2**: Solution Heatmaps
- Temperature/potential distributions
- Consistent colormap scaling across all three

**Row 3**: Gradient Fields
- Flow/flux visualization
- Normalized vector fields

Output Options:
- Display on screen
- Save as PDF (block_covering.pdf, solution.pdf, gradient.pdf)

### 6. Heuristic Analysis Plot
**Function**: `plot_heuristic_analysis()` (in `plotting.py`)

Analyzes impact of heuristic parameters on solver performance:

**Analysis Dimensions**:
- Radial heuristics: [0.75, 0.8, 0.85, 0.90, 0.95, 0.99]
- Overlap heuristics: [0.1, 0.2, 0.3, 0.4, 0.45]
- Polygon complexities: 10, 20, 30 vertices

**Visualization Features**:
- Three subplots for different vertex counts
- Error bars showing standard deviation
- Percentage change in block count (M)
- Different markers for overlap values
- Grid lines for easy reading

**Statistical Analysis**:
- Averages over 5 random polygons
- Reports mean area and variance
- Shows optimization trends
- Helps parameter selection

## Utility Functions

### Simple Block Plotting
**Function**: `plot_blocks()`
- Basic circle rendering for blocks
- Used for debugging block placement
- Shows block centers and radii

## Visualization Best Practices

### Color Schemes
- Blue: First kind blocks (vertex)
- Red: Second kind blocks (edge)
- Green: Third kind blocks (interior)
- Viridis: Solution heatmaps
- Black: Polygon boundaries

### Layout Conventions
- Equal aspect ratio for geometric accuracy
- Grid lines for reference
- Colorbars for quantitative data
- Clear labels and legends

### Performance Considerations
- Masked arrays for efficient irregular domain handling
- Decimation for large gradient fields
- Vectorized plotting operations
- Batch figure generation for comparisons

## Integration with Solver (Refactored)

The visualization modules are now cleanly separated but well-integrated:

### Volkov-Specific Integration (`volkov_plots.py`)
- Clean delegation pattern from solver methods
- Direct access to solver state and block data
- Real-time plotting during solution with proper data access
- Specialized functions for Volkov method visualization

### Analysis Integration (`plotting.py`)
- Independent analysis and comparison functions
- Parameter sensitivity analysis
- Multi-solver comparison capabilities
- Publication-ready output generation

### Key Improvements
- **Separation of Concerns**: Volkov-specific vs general plotting
- **Testability**: Visualization functions can be tested independently
- **Reusability**: General plotting functions work with any solver
- **Maintainability**: Clear boundaries between solver logic and visualization

## Output Formats
- Interactive matplotlib figures
- PDF export for publications
- Configurable DPI and sizes
- Batch processing support

## Common Use Cases

1. **Algorithm Understanding**
   - Visualize block covering strategy
   - See how gaps are filled
   - Understand overlap regions

2. **Solution Verification**
   - Check boundary conditions are satisfied
   - Verify smooth solutions
   - Identify numerical artifacts

3. **Parameter Tuning**
   - Analyze heuristic impacts
   - Find optimal settings
   - Trade-off accuracy vs. efficiency

4. **Publication Figures**
   - Clean, professional outputs
   - Consistent formatting
   - Batch generation
   - Vector graphics (PDF)