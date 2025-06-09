from seep2d import Seep2D, read_seep2d_input, diagnose_exit_face
from plot import plot_mesh, plot_solution
import numpy as np

# Load input
model = Seep2D()
model_data = read_seep2d_input("samples/s2unc/s2unc.s2d")
#model_data = read_seep2d_input("samples/s2con/s2con.s2d")

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


# Add these debug prints to main2.py after loading the model data

print("\n=== Material Properties Debug ===")
print(f"Number of materials: {len(model.k1_by_mat)}")
for i, (k1, k2, angle, kr0, h0) in enumerate(zip(
    model.k1_by_mat, model.k2_by_mat, model.angle_by_mat,
    model.kr0_by_mat, model.h0_by_mat)):
    print(f"Material {i+1}: k1={k1}, k2={k2}, angle={angle}, kr0={kr0}, h0={h0}")

print("\n=== Boundary Conditions Debug ===")
print(f"Total nodes: {len(model.nbc)}")
print(f"Fixed head nodes (nbc=1): {np.sum(model.nbc == 1)}")
print(f"Exit face nodes (nbc=2): {np.sum(model.nbc == 2)}")
print(f"No-flow nodes (nbc=0): {np.sum(model.nbc == 0)}")

# Check head values at boundaries
fixed_heads = [(i, model.fx[i]) for i in range(len(model.nbc)) if model.nbc[i] == 1]
print(f"\nFixed head values: min={min(h for _, h in fixed_heads):.3f}, max={max(h for _, h in fixed_heads):.3f}")

exit_elevations = [(i, model.coords[i,1]) for i in range(len(model.nbc)) if model.nbc[i] == 2]
if exit_elevations:
    print(f"Exit face elevations: min={min(y for _, y in exit_elevations):.3f}, max={max(y for _, y in exit_elevations):.3f}")

print("\n=== Element Materials Debug ===")
unique_mats, counts = np.unique(model.element_materials, return_counts=True)
for mat, count in zip(unique_mats, counts):
    print(f"Material {mat}: {count} elements")

# Check for any zero or negative conductivities
k1_arr = model.k1_by_mat[model.element_materials - 1]
k2_arr = model.k2_by_mat[model.element_materials - 1]
if np.any(k1_arr <= 0) or np.any(k2_arr <= 0):
    print("\nWARNING: Found zero or negative conductivities!")
    print(f"k1 range: [{np.min(k1_arr)}, {np.max(k1_arr)}]")
    print(f"k2 range: [{np.min(k2_arr)}, {np.max(k2_arr)}]")


# Run solver
model.run_analysis()


diagnose_exit_face(model.coords, model.nbc, model.solution["head"],
                   model.solution["q"], model.fx)


# Plot mesh
plot_mesh(model.coords, model.elements, model.element_materials,
          show_nodes=True, show_bc=True, nbc=model.nbc)





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

