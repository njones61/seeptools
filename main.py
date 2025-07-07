from seep import import_seep2d, run_analysis, print_seep_data_diagnostics, export_solution_csv
from plot import plot_mesh, plot_solution
import numpy as np


# Load input
seep_data = import_seep2d("samples/s2unc/s2unc.s2d")

# Print diagnostics
# print_seep_data_diagnostics(seep_data)

# Plot mesh
plot_mesh(seep_data, show_nodes=True, show_bc=True)

# Run analysis
solution = run_analysis(seep_data)

# Export solution to CSV
export_solution_csv("seep_solution.csv", seep_data, solution)

# Plot solution
plot_solution(seep_data, solution, base_mat=1, fill_contours=True, phreatic=True) 