import numpy as np
from pyBlockGrid import LaplaceEquationSolver, Polygon

def main():
    # Example polygon
    vertices = np.array([[0, 0], [2, 0], [2, 1], [1, 1], [1, 2], [0, 2]])
    poly = Polygon(vertices)
    
    # Boundary conditions
    boundary_conditions = [[0.0] for _ in range(len(vertices))]
    boundary_conditions[0] = [1.0]
    boundary_conditions[2] = [1.0]
    
    is_dirichlet = [True for _ in range(len(vertices))]
    
    # Solve
    solver = LaplaceEquationSolver(
        poly=poly,
        boundary_conditions=boundary_conditions,
        is_dirichlet=is_dirichlet,
        delta=0.05,
        n=50,
        max_iter=10
    )
    
    solution = solver.solve(verbose=True)
    
    # Plot results
    import matplotlib.pyplot as plt
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    solver.plot_block_covering(ax=ax1)
    solver.plot_solution(ax=ax2, solution=solution)
    plt.show()

if __name__ == "__main__":
    main() 