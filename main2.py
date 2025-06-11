from seep2d import Seep2D, read_seep2d_input, diagnose_exit_face, compare_python_fortran_nodal_results
from plot import plot_mesh, plot_solution
import numpy as np

# Load input
model = Seep2D()
#model_data = read_seep2d_input("samples/s2unc/s2unc.s2d")
model_data = read_seep2d_input("samples/s2con/s2con.s2d")

# Populate model with input data
model.coords = model_data["coords"]
model.elements = model_data["elements"]
model.nbc = model_data["nbc"]
model.fx = model_data["fx"]
model.element_materials = model_data["element_materials"]
model.k1_by_mat = model_data["k1_by_mat"]
model.k2_by_mat = model_data["k2_by_mat"]
model.angle_by_mat = model_data["angle_by_mat"]
model.kr0_by_mat = model_data["kr0_by_mat"]
model.h0_by_mat = model_data["h0_by_mat"]
model.unit_weight = model_data["unit_weight"]

model.export_path = "solution.csv"


# Run solver
model.run_analysis()


# # Compare with FORTRAN results
# compare_python_fortran_nodal_results(
#     python_head=model.solution["head"],
#     python_q=model.solution["q"],
#     fortran_out_path="samples/s2unc/s2unc.out",
#     nbc=model.nbc
# )

# Plot mesh
# plot_mesh(model.coords, model.elements, model.element_materials,
#           show_nodes=True, show_bc=True, nbc=model.nbc)



# Plot solution
plot_solution(model.coords,
              model.elements,
              model.solution["head"],
              phi=model.solution["phi"],
              flowrate=model.solution["flowrate"],
              base_mat=1,  # assuming material ID 1 is the base material
              k1_by_mat=model.k1_by_mat,
              fill_contours=False,  # Set to False for black lines only
              phreatic=True,       # Set to False to hide phreatic surface
              element_materials=model.element_materials)  # Add material coloring

# Plot flownet
#plot_flownet(model.coords, model.elements, model.solution["head"], model.solution["phi"], flowrate=model.solution["flowrate"])

