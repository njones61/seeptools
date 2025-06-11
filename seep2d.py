"""
SEEP2D Python Translation
-------------------------

Finite element seepage analysis tool originally developed in Fortran.
Ported to Python for improved modularity and extensibility.

Author (Original): Fred Tracy, ERDC
Python Port: [Your Name or Group]
"""

import numpy as np
import matplotlib.pyplot as plt


# Constants (from seep.inc)
MXNODS = 1000000   # Max number of nodes
MXELES = 1000000   # Max number of elements
MXBNDW = 700       # Max matrix bandwidth
MXMATS = 100       # Max material zones


class Seep2D:
    def __init__(self):
        # Simulation control flags and metadata
        self.stayopen = ''
        self.usegeo = False

        # Problem size (to be set during input)
        self.num_nodes = 0
        self.num_elements = 0

        # Data structures
        self.nodes = np.zeros((MXNODS, 2))        # Node coordinates
        self.elements = np.zeros((MXELES, 3), dtype=int)  # Element connectivity
        self.material_ids = np.zeros(MXELES, dtype=int)   # Material ID per element

        # Placeholder: boundary conditions, loads, solution variables, etc.
        # self.boundary_conditions = ...
        # self.hydraulic_heads = ...

    def load_input(self, filename):
        print(f"Loading input from {filename}")
        import sys
        import os

        print("Entering seepage analysis")

        if filename == "-getArraySizes":
            with open("arraySizes.txt", "w") as f:
                f.write(f"MaxNode     {MXNODS}\n")
                f.write(f"MaxElement  {MXELES}\n")
                f.write(f"MaxBandwidth {MXBNDW}\n")
                f.write(f"MaxMaterials {MXMATS}\n")
            print("Wrote arraySizes.txt and exiting.")
            sys.exit(0)

        if filename.strip() == "":
            filename = input("Enter the name of the Seep2D super file: ").strip()

        if not os.path.exists(filename):
            raise FileNotFoundError(f"Super file '{filename}' not found.")

        with open(filename, "r") as f:
            sptype = f.readline().strip()

        if not sptype.startswith("SEEPSUP"):
            raise ValueError("This file is not a GMS Seep2D superfile.")

        print(f"Superfile validated: {filename}")

        # Read associated data files
        with open(filename, "r") as f:
            lines = f.readlines()[1:]  # skip sptype line

        for line in lines:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            fltype, flname = parts[0], parts[1]
            flname = os.path.join(os.path.dirname(filename), flname)

            if fltype == "SEEP":
                self.seep_file = flname
                print(f"SEEP file: {flname}")
            elif fltype == "ODAT":
                self.odat_file = flname
                print(f"ODAT file: {flname}")
            elif fltype == "OGEO":
                self.ogeo_file = flname
                self.usegeo = True
                print(f"OGEO file: {flname}")
            elif fltype == "DSET":
                self.dset_file = flname
                print(f"DSET file: {flname}")

    def run_analysis(self):
        """
        Unified SEEP2D analysis driver for both confined and unconfined flow.
        """
        is_unconfined = np.any(self.nbc == 2)
        flow_type = "unconfined" if is_unconfined else "confined"
        print(f"Solving {flow_type.upper()} SEEP2D problem...")
        print("Number of fixed-head nodes:", np.sum(self.nbc == 1))
        print("Number of exit face nodes:", np.sum(self.nbc == 2))

        # Dirichlet BCs: fixed head (nbc == 1) and possibly exit face (nbc == 2)
        bcs = [(i, self.fx[i]) for i in range(len(self.nbc)) if self.nbc[i] in (1, 2)]

        # Material properties (per element)
        mat_ids = self.element_materials - 1
        k1 = self.k1_by_mat[mat_ids]
        k2 = self.k2_by_mat[mat_ids]
        angle = self.angle_by_mat[mat_ids]

        # Solve for head, stiffness matrix A, and nodal flow vector q
        if is_unconfined:
            # Get kr0 and h0 values per element based on material
            kr0_per_element = self.kr0_by_mat[mat_ids]
            h0_per_element = self.h0_by_mat[mat_ids]

            head, A, q, total_flow = solve_unsaturated(
                coords=self.coords,
                elements=self.elements,
                nbc=self.nbc,
                fx=self.fx,
                kr0=kr0_per_element,
                h0=h0_per_element,
                k1_vals=k1,
                k2_vals=k2,
                angles=angle,
                max_iter=200,
                tol=1e-4
            )
            # Solve for potential function φ for flow lines
            dirichlet_phi_bcs = create_flow_potential_bc(self.coords, self.elements, q)
            phi = solve_flow_function_unsaturated(self.coords, self.elements, head, k1, k2, angle, kr0_per_element, h0_per_element, dirichlet_phi_bcs)
            print(f"phi min: {np.min(phi):.3f}, max: {np.max(phi):.3f}")
        else:
            head, A, q, total_flow = solve_confined(self.coords, self.elements, self.nbc, bcs, k1, k2, angle)
            # Solve for potential function φ for flow lines
            dirichlet_phi_bcs = create_flow_potential_bc(self.coords, self.elements, q)
            phi = solve_flow_function_confined(self.coords, self.elements, dirichlet_phi_bcs)
            print(f"phi min: {np.min(phi):.3f}, max: {np.max(phi):.3f}")

        gamma_w = self.unit_weight
        pressure = gamma_w * (head - self.coords[:, 1])
        velocity = compute_velocity(self.coords, self.elements, head, k1, k2, angle)

        self.solution = {
            "head": head,
            "pressure": pressure,
            "velocity": velocity,
            "q": q,
            "phi": phi,
            "flowrate": total_flow
        }

        if hasattr(self, "export_path"):
            export_solution_csv(self.export_path, self.coords, head, pressure, velocity, q, phi, total_flow)

        # After computing nodal flows
        #debug_nodal_flows_above_phreatic(self.coords, head, q, title='Nodal Flows vs Phreatic Surface')

    def save_results(self, filename):
        print(f"Saving results to {filename}")
        # TODO: Implement result writer


def main():
    model = Seep2D()
    model.load_input("input.seep")
    model.run_analysis()
    model.save_results("results.out")


    if __name__ == "__main__":
        main()

    import re
    import numpy as np

    element_lines = lines[skip_header + num_nodes:]
    elements = []
    mat_ids = []

    for line in element_lines:
        nums = [int(n) for n in re.findall(r'\d+', line)]
        if len(nums) >= 6:
            _, n1, n2, n3, _, mat = nums[:6]
            elements.append([n1, n2, n3])
            mat_ids.append(mat)

    return np.array(elements) - 1, np.array(mat_ids)  # zero-based indexing

def read_seep2d_input(filepath):
    """
    Reads SEEP2D .s2d input file and returns mesh, materials, and BC data.

    Returns:
        {
            "coords": np.ndarray (n_nodes, 2),
            "node_ids": np.ndarray (n_nodes,),
            "node_materials": np.ndarray (n_nodes,),
            "nbc": np.ndarray (n_nodes,),   # boundary condition flags
            "fx": np.ndarray (n_nodes,),    # boundary condition values (head or elevation)
            "elements": np.ndarray (n_elements, 3),
            "element_materials": np.ndarray (n_elements,)
        }
    """
    import re
    import numpy as np

    with open(filepath, "r", encoding="latin-1") as f:
        lines = [line.rstrip() for line in f if line.strip()]

    title = lines[0]                  # First line is the title (any text)
    parts = lines[1].split()          # Second line contains analysis parameters

    num_nodes = int(parts[0])         # Number of nodes
    num_elements = int(parts[1])      # Number of elements
    num_materials = int(parts[2])     # Number of materials
    datum = float(parts[3])           # Datum elevation (not used, assume 0.0)

    problem_type = parts[4]           # "PLNE" = planar, otherwise axisymmetric (we only support "PLNE")
    analysis_flag = parts[5]          # Unknown integer (ignore)
    flow_flag = parts[6]              # "F" or "T" = compute flowlines (ignore)
    unit_weight = float(parts[7])     # Unit weight of water (e.g. 62.4 lb/ft³ or 9.81 kN/m³)
    model_type = int(parts[8])        # 1 = linear front, 2 = van Genuchten (we only support 0)

    assert problem_type == "PLNE", "Only planar problems are supported"
    assert model_type == 1, "Only linear front models are supported"

    unit_weight = float(parts[7])   # the unit weight
    mat_props = []
    line_offset = 2
    while len(mat_props) < num_materials:
        nums = [float(n) if '.' in n or 'e' in n.lower() else int(n)
                for n in re.findall(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?', lines[line_offset])]
        if len(nums) >= 6:
            mat_props.append(nums[:6])
        line_offset += 1
    mat_props = np.array(mat_props)
    k1_array = mat_props[:, 1]
    k2_array = mat_props[:, 2]
    angle_array = mat_props[:, 3]
    kr0_array = mat_props[:, 4]
    h0_array = mat_props[:, 5]
    node_lines = lines[line_offset:line_offset + num_nodes]
    element_lines = lines[line_offset + num_nodes:]

    coords = []
    node_materials = []
    node_ids = []
    nbc_flags = []
    fx_vals = []

    for line in node_lines:
        try:
            node_id = int(line[0:5])
            bc_type = int(line[7:10])
            x = float(line[10:25])
            y = float(line[25:40])

            if bc_type == 1 and len(line) >= 41:
                fx_val = float(line[40:55])
            elif bc_type == 2:
                fx_val = y
            else:
                fx_val = 0.0

            node_ids.append(node_id)
            nbc_flags.append(bc_type)
            fx_vals.append(fx_val)
            coords.append((x, y))
            node_materials.append(0)

        except Exception as e:
            print(f"Warning: skipping node due to error: {e}")

    elements = []
    element_mats = []

    for line in element_lines:
        nums = [int(n) for n in re.findall(r'\d+', line)]
        if len(nums) >= 6:
            _, n1, n2, n3, _, mat = nums[:6]
            elements.append([n1, n2, n3])
            element_mats.append(mat)

    return {
        "coords": np.array(coords),
        "node_ids": np.array(node_ids, dtype=int),
        "node_materials": np.array(node_materials),
        "nbc": np.array(nbc_flags, dtype=int),
        "fx": np.array(fx_vals),
        "elements": np.array(elements, dtype=int) - 1,
        "element_materials": np.array(element_mats),
        "k1_by_mat": k1_array,
        "k2_by_mat": k2_array,
        "angle_by_mat": angle_array,
        "kr0_by_mat": kr0_array,
        "h0_by_mat": h0_array,
        "unit_weight": unit_weight
    }

def export_solution_csv(filename, coords, head, pressure, velocity, q, phi, flowrate):
    """Exports nodal results to a CSV file."""
    import pandas as pd
    df = pd.DataFrame({
        "x": coords[:, 0],
        "y": coords[:, 1],
        "head": head,
        "pressure": pressure,
        "v_x": velocity[:, 0],
        "v_y": velocity[:, 1],
        "v_mag": np.linalg.norm(velocity, axis=1),
        "q": q,
        "phi": phi
    })
    # Write to file, then append flowrate as comment
    with open(filename, "w") as f:
        df.to_csv(f, index=False)
        f.write(f"# Total Flowrate: {flowrate:.6f}\n")

    print(f"Exported solution to {filename}")

def load_solution_csv(filename):
    """Loads solution results from a CSV file."""
    import pandas as pd
    df = pd.read_csv(filename)
    coords = df[["x", "y"]].values
    head = df["head"].values
    pressure = df["pressure"].values
    velocity = df[["vx", "vy"]].values
    return coords, head, pressure, velocity


def solve_confined(coords, elements, nbc, dirichlet_bcs, k1_vals, k2_vals, angles=None):
    """
    FEM solver for confined seepage with anisotropic conductivity.
    Parameters:
        coords : (n_nodes, 2) array of node coordinates
        elements : (n_elements, 3) triangle node indices
        dirichlet_bcs : list of (node_id, head_value)
        k1_vals : (n_elements,) or scalar, major axis conductivity
        k2_vals : (n_elements,) or scalar, minor axis conductivity
        angles : (n_elements,) or scalar, angle in degrees (from x-axis)
    Returns:
        head : (n_nodes,) array of nodal heads
    """
    import numpy as np
    from scipy.sparse import lil_matrix
    from scipy.sparse.linalg import spsolve

    n_nodes = coords.shape[0]
    A = lil_matrix((n_nodes, n_nodes))
    b = np.zeros(n_nodes)

    for idx, tri in enumerate(elements):
        i, j, k = tri
        xi, yi = coords[i]
        xj, yj = coords[j]
        xk, yk = coords[k]

        area = 0.5 * np.linalg.det([[1, xi, yi], [1, xj, yj], [1, xk, yk]])
        if area <= 0:
            continue

        beta = np.array([yj - yk, yk - yi, yi - yj])
        gamma = np.array([xk - xj, xi - xk, xj - xi])
        grad = np.array([beta, gamma]) / (2 * area)

        # Get anisotropic conductivity
        k1 = k1_vals[idx]
        k2 = k2_vals[idx]
        theta = angles[idx]

        theta_rad = np.radians(theta)
        c, s = np.cos(theta_rad), np.sin(theta_rad)
        R = np.array([[c, s], [-s, c]])
        Kmat = R.T @ np.diag([k1, k2]) @ R

        ke = area * grad.T @ Kmat @ grad

        for a in range(3):
            for b_ in range(3):
                A[tri[a], tri[b_]] += ke[a, b_]

    from scipy.sparse import csr_matrix

    A_full = A.copy()  # Keep original matrix for computing q

    for node, value in dirichlet_bcs:
        A[node, :] = 0
        A[node, node] = 1
        b[node] = value

    head = spsolve(A.tocsr(), b)
    q = A_full.tocsr() @ head


    total_flow = 0.0

    for node_idx in range(len(nbc)):
        if q[node_idx] > 0:  # Positive flow
            total_flow += q[node_idx]

    return head, A, q, total_flow   


def solve_unsaturated(coords, elements, nbc, fx, kr0=0.001, h0=-1.0,
                      k1_vals=1.0, k2_vals=1.0, angles=0.0,
                      max_iter=200, tol=1e-4):
    """
    Iterative FEM solver for unconfined flow using linear kr frontal function.
    """
    import numpy as np
    from scipy.sparse import lil_matrix, csr_matrix
    from scipy.sparse.linalg import spsolve

    n_nodes = coords.shape[0]
    y = coords[:, 1]

    # Initialize heads
    h = np.zeros(n_nodes)
    for node_idx in range(n_nodes):
        if nbc[node_idx] == 1:
            h[node_idx] = fx[node_idx]
        elif nbc[node_idx] == 2:
            h[node_idx] = y[node_idx]
        else:
            fixed_heads = fx[nbc == 1]
            h[node_idx] = np.mean(fixed_heads) if len(fixed_heads) > 0 else np.mean(y)

    # Track which exit face nodes are active (saturated)
    exit_face_active = np.ones(n_nodes, dtype=bool)
    exit_face_active[nbc != 2] = False

    # Store previous iteration values
    h_last = h.copy()

    # Get material properties per element
    if np.isscalar(kr0):
        kr0 = np.full(len(elements), kr0)
    if np.isscalar(h0):
        h0 = np.full(len(elements), h0)

    # Set convergence tolerance based on domain height
    ymin, ymax = np.min(y), np.max(y)
    eps = (ymax - ymin) * 0.0001

    print("Starting unsaturated flow iteration...")
    print(f"Convergence tolerance: {eps:.6e}")

    # Track convergence history
    residuals = []
    relax = 1.0  # Initial relaxation factor
    prev_residual = float('inf')

    for iteration in range(1, max_iter + 1):
        # Reset diagnostics for this iteration
        kr_diagnostics = []

        # Build global stiffness matrix
        A = lil_matrix((n_nodes, n_nodes))
        b = np.zeros(n_nodes)

        # Compute pressure head at nodes
        p_nodes = h - y

        # Element assembly with element-wise kr computation
        for idx, tri in enumerate(elements):
            i, j, k = tri
            xi, yi = coords[i]
            xj, yj = coords[j]
            xk, yk = coords[k]

            # Element area
            area = 0.5 * abs((xj - xi) * (yk - yi) - (xk - xi) * (yj - yi))
            if area <= 0:
                continue

            # Shape function derivatives
            beta = np.array([yj - yk, yk - yi, yi - yj])
            gamma = np.array([xk - xj, xi - xk, xj - xi])
            grad = np.array([beta, gamma]) / (2 * area)

            # Get material properties for this element
            k1 = k1_vals[idx] if hasattr(k1_vals, '__len__') else k1_vals
            k2 = k2_vals[idx] if hasattr(k2_vals, '__len__') else k2_vals
            theta = angles[idx] if hasattr(angles, '__len__') else angles

            # Anisotropic conductivity matrix
            theta_rad = np.radians(theta)
            c, s = np.cos(theta_rad), np.sin(theta_rad)
            R = np.array([[c, s], [-s, c]])
            Kmat = R.T @ np.diag([k1, k2]) @ R

            # Compute element pressure (centroid)
            p_elem = (p_nodes[i] + p_nodes[j] + p_nodes[k]) / 3.0

            # Get kr for this element based on its material properties
            kr_elem = kr_frontal(p_elem, kr0[idx], h0[idx])

            # Element stiffness matrix with kr
            ke = kr_elem * area * grad.T @ Kmat @ grad

            # Inside the element assembly loop, after kr_elem is computed:
            kr_diagnostics.append({
                'element': idx,
                'p_elem': p_elem,
                'kr_elem': kr_elem,
                'y_centroid': (yi + yj + yk) / 3.0,
                'h_centroid': (h[i] + h[j] + h[k]) / 3.0
            })

            # Assembly
            for row in range(3):
                for col in range(3):
                    A[tri[row], tri[col]] += ke[row, col]

        # Store unmodified matrix for flow computation
        A_full = A.tocsr()

        # Apply boundary conditions
        for node_idx in range(n_nodes):
            if nbc[node_idx] == 1:
                A[node_idx, :] = 0
                A[node_idx, node_idx] = 1
                b[node_idx] = fx[node_idx]
            elif nbc[node_idx] == 2 and exit_face_active[node_idx]:
                A[node_idx, :] = 0
                A[node_idx, node_idx] = 1
                b[node_idx] = y[node_idx]

        # Convert to CSR and solve
        A_csr = A.tocsr()
        h_new = spsolve(A_csr, b)

        # FORTRAN-style relaxation strategy
        if iteration > 20:
            relax = 0.5
        if iteration > 40:
            relax = 0.2
        if iteration > 60:
            relax = 0.1
        if iteration > 80:
            relax = 0.05
        if iteration > 100:
            relax = 0.02
        if iteration > 120:
            relax = 0.01

        # Apply relaxation
        h_new = relax * h_new + (1 - relax) * h_last

        # Compute flows at all nodes (not used for closure, but for exit face logic)
        q = A_full @ h_new

        # Update exit face boundary conditions with hysteresis
        n_active_before = np.sum(exit_face_active)
        hyst = 0.001 * (ymax - ymin)  # Hysteresis threshold

        for node_idx in range(n_nodes):
            if nbc[node_idx] == 2:
                if exit_face_active[node_idx]:
                    # Check if node should become inactive
                    if h_new[node_idx] < y[node_idx] - hyst or q[node_idx] > 0:
                        exit_face_active[node_idx] = False
                else:
                    # Check if node should become active again
                    if h_new[node_idx] >= y[node_idx] + hyst and q[node_idx] <= 0:
                        exit_face_active[node_idx] = True
                        h_new[node_idx] = y[node_idx]  # Reset to elevation

        n_active_after = np.sum(exit_face_active)

        # Compute relative residual
        residual = np.max(np.abs(h_new - h)) / (np.max(np.abs(h)) + 1e-10)
        residuals.append(residual)

        # Print detailed iteration info
        if iteration <= 3 or iteration % 20 == 0 or n_active_before != n_active_after:
            print(f"Iteration {iteration}: residual = {residual:.6e}, relax = {relax:.3f}")
            print(f"  BCs: {np.sum(nbc == 1)} fixed head, {n_active_after}/{np.sum(nbc == 2)} exit face active")

        # Check convergence
        if residual < eps:
            print(f"Converged in {iteration} iterations")
            break

        # Update for next iteration
        h = h_new.copy()
        h_last = h_new.copy()

    else:
        print(f"Warning: Did not converge in {max_iter} iterations")
        print("\nConvergence history:")
        for i, r in enumerate(residuals):
            if i % 20 == 0 or i == len(residuals) - 1:
                print(f"  Iteration {i+1}: residual = {r:.6e}")


    q_final = q

    # Flow potential closure check - FORTRAN-style
    total_inflow = 0.0
    total_outflow = 0.0
    
    for node_idx in range(n_nodes):
        if nbc[node_idx] == 1:  # Fixed head boundary
            if q_final[node_idx] > 0:
                total_inflow += q_final[node_idx]
            elif q_final[node_idx] < 0:
                total_outflow -= q_final[node_idx]
        elif nbc[node_idx] == 2 and exit_face_active[node_idx]:  # Active exit face
            if q_final[node_idx] < 0:
                total_outflow -= q_final[node_idx]

    closure_error = abs(total_inflow - total_outflow)
    print(f"Flow potential closure check: error = {closure_error:.6e}")
    print(f"Total inflow: {total_inflow:.6e}")
    print(f"Total outflow: {total_outflow:.6e}")

    if closure_error > 0.01 * max(abs(total_inflow), abs(total_outflow)):
        print(f"Warning: Large flow potential closure error = {closure_error:.6e}")
        print("This may indicate:")
        print("  - Non-conservative flow field")
        print("  - Incorrect boundary identification")
        print("  - Numerical issues in the flow solution")


    return h, A_csr, q_final, total_inflow

def kr_frontal(p, kr0, h0):
    """
    Fortran-compatible relative permeability function (front model).
    This matches the fkrelf function in the Fortran code exactly.
    """
    if p >= 0.0:
        return 1.0
    elif p > h0:
        return kr0 + (1.0 - kr0) * p / h0
    else:
        return kr0


def diagnose_exit_face(coords, nbc, h, q, fx):
    """
    Diagnostic function to understand exit face behavior
    """
    import numpy as np

    print("\n=== Exit Face Diagnostics ===")
    exit_nodes = np.where(nbc == 2)[0]
    y = coords[:, 1]

    print(f"Total exit face nodes: {len(exit_nodes)}")
    print("\nNode | x      | y      | h      | h-y    | q        | Status")
    print("-" * 65)

    for node in exit_nodes:
        x_coord = coords[node, 0]
        y_coord = y[node]
        head = h[node]
        pressure = head - y_coord
        flow = q[node]

        if head >= y_coord:
            status = "SATURATED"
        else:
            status = "UNSATURATED"

        print(f"{node:4d} | {x_coord:6.2f} | {y_coord:6.2f} | {head:6.3f} | {pressure:6.3f} | {flow:8.3e} | {status}")

    # Summary statistics
    saturated = np.sum(h[exit_nodes] >= y[exit_nodes])
    print(f"\nSaturated nodes: {saturated}/{len(exit_nodes)}")

    # Check phreatic surface
    print("\n=== Phreatic Surface Location ===")
    # Find where the phreatic surface intersects the exit face
    for i in range(len(exit_nodes) - 1):
        n1, n2 = exit_nodes[i], exit_nodes[i + 1]
        if (h[n1] >= y[n1]) and (h[n2] < y[n2]):
            # Interpolate intersection point
            y1, y2 = y[n1], y[n2]
            h1, h2 = h[n1], h[n2]
            y_intersect = y1 + (y2 - y1) * (h1 - y1) / (h1 - y1 - h2 + y2)
            print(f"Phreatic surface exits between nodes {n1} and {n2}")
            print(f"Approximate exit elevation: {y_intersect:.3f}")
            break

def create_flow_potential_bc(coords, elements, q, debug=False):
    """
    Generates Dirichlet BCs for flow potential φ by marching around the boundary
    and accumulating q to assign φ, ensuring closed-loop conservation.

    Improved version that handles numerical noise and different boundary types.

    Parameters:
        coords : (n_nodes, 2) array of node coordinates
        elements : (n_elements, 3) triangle node indices
        q : (n_nodes,) nodal flow vector
        debug : bool, if True prints detailed diagnostic information

    Returns:
        List of (node_id, phi_value) tuples
    """
    import numpy as np
    from collections import defaultdict

    if debug:
        print("=== FLOW POTENTIAL BC DEBUG ===")

    # Step 1: Build edge dictionary and count how many times each edge appears
    edge_counts = defaultdict(list)
    for idx, (i, j, k) in enumerate(elements):
        edges = [(i, j), (j, k), (k, i)]
        for a, b in edges:
            edge = tuple(sorted((a, b)))
            edge_counts[edge].append(idx)

    # Step 2: Extract boundary edges (appear only once)
    boundary_edges = [edge for edge, elems in edge_counts.items() if len(elems) == 1]

    if debug:
        print(f"Found {len(boundary_edges)} boundary edges")

    # Step 3: Build connectivity for the boundary edges
    neighbor_map = defaultdict(list)
    for a, b in boundary_edges:
        neighbor_map[a].append(b)
        neighbor_map[b].append(a)

    # Step 4: Walk the boundary in order (clockwise or counterclockwise)
    start_node = boundary_edges[0][0]
    ordered_nodes = [start_node]
    visited = {start_node}
    current = start_node

    while True:
        neighbors = [n for n in neighbor_map[current] if n not in visited]
        if not neighbors:
            break
        next_node = neighbors[0]
        ordered_nodes.append(next_node)
        visited.add(next_node)
        current = next_node
        if next_node == start_node:
            break  # closed loop

    # Debug boundary flow statistics
    if debug:
        boundary_nodes = sorted(set(ordered_nodes))
        print(f"Boundary nodes: {len(boundary_nodes)}")
        print(f"Flow statistics on boundary:")
        q_boundary = [q[node] for node in boundary_nodes]
        print(f"  Min q: {min(q_boundary):.6e}")
        print(f"  Max q: {max(q_boundary):.6e}")
        print(f"  Mean |q|: {np.mean([abs(qval) for qval in q_boundary]):.6e}")
        print(f"  Std |q|: {np.std([abs(qval) for qval in q_boundary]):.6e}")

        # Count "small" flows
        thresholds = [1e-12, 1e-10, 1e-8, 1e-6, 1e-4]
        for thresh in thresholds:
            count = sum(1 for qval in q_boundary if abs(qval) < thresh)
            print(f"  Nodes with |q| < {thresh:.0e}: {count}/{len(boundary_nodes)}")

    # Step 5: Find starting point - improved algorithm
    start_idx = None
    n = len(ordered_nodes)

    # Define threshold for "effectively zero" flow based on the magnitude of flows
    q_boundary = [abs(q[node]) for node in ordered_nodes]
    q_max = max(q_boundary) if q_boundary else 1.0
    q_threshold = max(1e-10, q_max * 1e-6)  # Adaptive threshold

    if debug:
        print(f"Flow analysis: max |q| = {q_max:.3e}, threshold = {q_threshold:.3e}")

    # Strategy 1: Look for transition from significant flow to near-zero flow
    for i in range(n):
        current_q = abs(q[ordered_nodes[i]])
        next_q = abs(q[ordered_nodes[(i + 1) % n]])

        if current_q > q_threshold and next_q <= q_threshold:
            start_idx = (i + 1) % n
            if debug:
                print(f"Found transition at node {ordered_nodes[i]} -> {ordered_nodes[start_idx]}")
            break

    # Strategy 2: If no clear transition, find the node with minimum |q|
    if start_idx is None:
        min_q_idx = min(range(n), key=lambda i: abs(q[ordered_nodes[i]]))
        start_idx = min_q_idx
        if debug:
            print(
                f"No clear transition found, starting at minimum |q| node {ordered_nodes[start_idx]} (|q|={abs(q[ordered_nodes[start_idx]]):.3e})")

    # Strategy 3: If all flows are significant, look for flow direction changes
    if start_idx is None or abs(q[ordered_nodes[start_idx]]) > q_threshold:
        # Look for where flow changes sign (inflow vs outflow)
        for i in range(n):
            current_q = q[ordered_nodes[i]]
            next_q = q[ordered_nodes[(i + 1) % n]]

            if current_q * next_q < 0:  # Sign change
                start_idx = i if abs(current_q) < abs(next_q) else (i + 1) % n
                if debug:
                    print(f"Found sign change, starting at node {ordered_nodes[start_idx]}")
                break

    # Strategy 4: Default fallback - start at node 0
    if start_idx is None:
        start_idx = 0
        if debug:
            print(f"Using fallback: starting at first boundary node {ordered_nodes[start_idx]}")

    # Step 6: Assign φ = 0 to the starting node, then accumulate q
    phi = {}
    phi_val = 0.0

    if debug:
        print(f"Starting flow potential calculation at node {ordered_nodes[start_idx]}")

    for i in range(n):
        idx = (start_idx + i) % n
        node = ordered_nodes[idx]
        phi[node] = phi_val
        phi_val += q[node]

        if debug and (i < 5 or i >= n - 5):  # Print first and last few for debugging
            print(f"  Node {node}: φ = {phi[node]:.6f}, q = {q[node]:.6f}")

    # Check closure - should be close to zero for a proper flow field
    closure_error = phi_val - q[ordered_nodes[start_idx]]

    if debug or abs(closure_error) > 1e-3:
        print(f"Flow potential closure check: error = {closure_error:.6e}")

        if abs(closure_error) > 1e-3:
            print(f"Warning: Large flow potential closure error = {closure_error:.6e}")
            print("This may indicate:")
            print("  - Non-conservative flow field")
            print("  - Incorrect boundary identification")
            print("  - Numerical issues in the flow solution")

    if debug:
        print("✓ Flow potential BC creation succeeded")

    return list(phi.items())

def solve_flow_function_confined(coords, elements, dirichlet_nodes):
    """
    Solves Laplace equation for flow function Phi on the same mesh,
    assigning Dirichlet values along no-flow boundaries.

    Parameters:
        coords : (n_nodes, 2) array of node coordinates
        elements : (n_elements, 3) triangle node indices
        velocity : (n_nodes, 2) Darcy velocity vectors (currently unused)
        dirichlet_nodes : list of (node_id, phi_value)

    Returns:
        phi : (n_nodes,) stream function (flow function) values
    """

    from scipy.sparse import lil_matrix
    from scipy.sparse.linalg import spsolve

    n_nodes = coords.shape[0]
    A = lil_matrix((n_nodes, n_nodes))
    b = np.zeros(n_nodes)

    for tri in elements:
        i, j, k = tri
        xi, yi = coords[i]
        xj, yj = coords[j]
        xk, yk = coords[k]

        area = 0.5 * np.linalg.det([[1, xi, yi], [1, xj, yj], [1, xk, yk]])
        if area <= 0:
            continue

        beta = np.array([yj - yk, yk - yi, yi - yj])
        gamma = np.array([xk - xj, xi - xk, xj - xi])
        grad = np.array([beta, gamma]) / (2 * area)

        ke = area * grad.T @ grad  # Isotropic Laplace stiffness

        for a in range(3):
            for b_ in range(3):
                A[tri[a], tri[b_]] += ke[a, b_]

    for node, phi_value in dirichlet_nodes:
        A[node, :] = 0
        A[node, node] = 1
        b[node] = phi_value

    phi = spsolve(A.tocsr(), b)
    return phi

def solve_flow_function_unsaturated(coords, elements, head, k1_vals, k2_vals, angles, kr0, h0, dirichlet_nodes):
    """
    Solves the flow function Phi using the correct ke for unsaturated flow.
    For flowlines, assemble the element matrix using the inverse of kr_elem and Kmat, matching the FORTRAN logic.
    """
    from scipy.sparse import lil_matrix
    from scipy.sparse.linalg import spsolve
    import numpy as np

    n_nodes = coords.shape[0]
    A = lil_matrix((n_nodes, n_nodes))
    b = np.zeros(n_nodes)

    y = coords[:, 1]
    p_nodes = head - y

    for idx, tri in enumerate(elements):
        i, j, k = tri
        xi, yi = coords[i]
        xj, yj = coords[j]
        xk, yk = coords[k]

        area = 0.5 * abs((xj - xi) * (yk - yi) - (xk - xi) * (yj - yi))
        if area <= 0:
            continue

        beta = np.array([yj - yk, yk - yi, yi - yj])
        gamma = np.array([xk - xj, xi - xk, xj - xi])
        grad = np.array([beta, gamma]) / (2 * area)  # grad is (2,3)

        # Get material properties for this element
        k1 = k1_vals[idx] if hasattr(k1_vals, '__len__') else k1_vals
        k2 = k2_vals[idx] if hasattr(k2_vals, '__len__') else k2_vals
        theta = angles[idx] if hasattr(angles, '__len__') else angles

        theta_rad = np.radians(theta)
        c, s = np.cos(theta_rad), np.sin(theta_rad)
        R = np.array([[c, s], [-s, c]])
        Kmat = R.T @ np.diag([k1, k2]) @ R  # Kmat is (2,2)

        # Compute element pressure (centroid)
        p_elem = (p_nodes[i] + p_nodes[j] + p_nodes[k]) / 3.0
        kr_elem = kr_frontal(p_elem, kr0[idx], h0[idx])

        # Assemble using the inverse of kr_elem and Kmat
        # If kr_elem is very small, avoid division by zero
        if kr_elem > 1e-12:
            ke = (1.0 / kr_elem) * area * grad.T @ np.linalg.inv(Kmat) @ grad
        else:
            ke = 1e12 * area * grad.T @ np.linalg.inv(Kmat) @ grad  # Large value for near-zero kr

        for a in range(3):
            for b_ in range(3):
                A[tri[a], tri[b_]] += ke[a, b_]

    for node, phi_value in dirichlet_nodes:
        A[node, :] = 0
        A[node, node] = 1
        b[node] = phi_value

    phi = spsolve(A.tocsr(), b)
    return phi


def compute_velocity(coords, elements, head, k1_vals, k2_vals, angles):
    """
    Compute nodal velocities by averaging element-wise Darcy velocities.

    Parameters:
        coords : (n_nodes, 2) array of node coordinates
        elements : (n_elements, 3) triangle node indices
        head : (n_nodes,) nodal head solution
        k1_vals, k2_vals, angles : per-element anisotropic properties (or scalar)

    Returns:
        velocity : (n_nodes, 2) array of nodal velocity vectors [vx, vy]
    """
    n_nodes = coords.shape[0]
    velocity = np.zeros((n_nodes, 2))
    count = np.zeros(n_nodes)

    scalar_k = np.isscalar(k1_vals)

    for idx, tri in enumerate(elements):
        i, j, k = tri
        xi, yi = coords[i]
        xj, yj = coords[j]
        xk, yk = coords[k]

        area = 0.5 * np.linalg.det([[1, xi, yi], [1, xj, yj], [1, xk, yk]])
        if area <= 0:
            continue

        beta = np.array([yj - yk, yk - yi, yi - yj])
        gamma = np.array([xk - xj, xi - xk, xj - xi])
        grad = np.array([beta, gamma]) / (2 * area)

        h_vals = head[[i, j, k]]
        grad_h = grad @ h_vals

        if scalar_k:
            k1 = k1_vals
            k2 = k2_vals
            theta = angles
        else:
            k1 = k1_vals[idx]
            k2 = k2_vals[idx]
            theta = angles[idx]

        theta_rad = np.radians(theta)
        c, s = np.cos(theta_rad), np.sin(theta_rad)
        R = np.array([[c, s], [-s, c]])
        K = R.T @ np.diag([k1, k2]) @ R

        v_elem = -K @ grad_h

        for node in tri:
            velocity[node] += v_elem
            count[node] += 1

    count[count == 0] = 1  # Avoid division by zero
    velocity /= count[:, None]
    return velocity

def compare_python_fortran_nodal_results(python_head, python_q, fortran_out_path, node_offset=1, verbose=True, nbc=None):
    """
    Compare Python and FORTRAN nodal heads and flows and save results to Excel.
    - python_head: numpy array of nodal heads (Python, index 0 = node 1 in FORTRAN)
    - python_q: numpy array of nodal flows (same indexing)
    - fortran_out_path: path to FORTRAN .out file
    - node_offset: 1 if FORTRAN nodes are 1-based, 0 if 0-based
    - verbose: if True, print summary statistics
    - nbc: array of boundary condition types (0=interior, 1=fixed head, 2=exit face)
    """
    import re
    import numpy as np
    import pandas as pd
    from pathlib import Path

    # Debug prints for nbc array
    print("\n=== Debug: Boundary Conditions ===")
    print(f"nbc is None: {nbc is None}")
    if nbc is not None:
        print(f"nbc type: {type(nbc)}")
        print(f"nbc shape: {nbc.shape if hasattr(nbc, 'shape') else len(nbc)}")
        print(f"nbc dtype: {nbc.dtype if hasattr(nbc, 'dtype') else type(nbc[0])}")
        print(f"nbc unique values: {np.unique(nbc)}")
        print(f"nbc first 10 values: {nbc[:10]}")
        print(f"Number of interior nodes (0): {np.sum(nbc == 0)}")
        print(f"Number of fixed head nodes (1): {np.sum(nbc == 1)}")
        print(f"Number of exit face nodes (2): {np.sum(nbc == 2)}")

    # Read the FORTRAN output file and find the relevant section
    with open(fortran_out_path, 'r') as f:
        lines = f.readlines()

    # Find the start of the 'Nodal Flows and Heads' section
    start_idx = None
    for i, line in enumerate(lines):
        if 'Nodal Flows and Heads' in line:
            start_idx = i
            break
    if start_idx is None:
        raise ValueError('Could not find "Nodal Flows and Heads" section in FORTRAN output.')

    # Skip header lines to the data
    data_start = start_idx + 5  # 5 lines after header is usually where data starts
    nodal_data = []
    for line in lines[data_start:]:
        if not line.strip():
            continue
        # Stop if we hit a non-data line (e.g., next section)
        if re.match(r'\s*\d+\s+\d', line) is None and re.match(r'\s*\d+\s+\d', line.strip()) is None:
            # If line doesn't start with a node number, break
            if not re.match(r'\s*\d+', line):
                break
        # Parse node, head, percent, (optional) flow
        m = re.match(r'\s*(\d+)\s+([\d.Ee+-]+)\s+([\d.Ee+-]+)\s*%?\s*([\d.Ee+-]*)', line)
        if m:
            node = int(m.group(1))
            head = float(m.group(2))
            # percent = float(m.group(3))  # Not used
            flow_str = m.group(4)
            flow = float(flow_str) if flow_str.strip() else np.nan
            nodal_data.append((node, head, flow))
        else:
            # If the line doesn't match, stop parsing
            break

    # Convert to arrays
    fortran_nodes = np.array([n[0] for n in nodal_data], dtype=int)
    fortran_head = np.array([n[1] for n in nodal_data], dtype=float)
    fortran_flow = np.array([n[2] for n in nodal_data], dtype=float)

    # Align indices: assume node 1 in FORTRAN is index 0 in Python
    n_compare = min(len(fortran_head), len(python_head))
    head_diff = python_head[:n_compare] - fortran_head[:n_compare]
    q_diff = python_q[:n_compare] - fortran_flow[:n_compare]

    # Create BC type mapping
    bc_type_map = {0: 'Interior', 1: 'Fixed Head', 2: 'Exit Face'}
    bc_types = np.array(['Unknown'] * n_compare)
    if nbc is not None:
        print(f"\nDebug: Creating BC types array")
        print(f"n_compare: {n_compare}")
        print(f"nbc length: {len(nbc)}")
        # Convert nbc to numpy array if it isn't already
        nbc = np.asarray(nbc)
        # Create BC types array
        bc_types = np.array([bc_type_map.get(int(nbc[i]), 'Unknown') for i in range(n_compare)])
        print(f"First 10 BC types: {bc_types[:10]}")
        print(f"Unique BC types: {np.unique(bc_types)}")

    # Create DataFrame for comparison
    df = pd.DataFrame({
        'Node': np.arange(node_offset, n_compare + node_offset),
        'BC_Type': bc_types,
        'Python_Head': python_head[:n_compare],
        'FORTRAN_Head': fortran_head[:n_compare],
        'Head_Diff': head_diff,
        'Abs(Head_Diff)': np.abs(head_diff),
        'Python_Q': python_q[:n_compare],
        'FORTRAN_Q': fortran_flow[:n_compare],
        'Q_Diff': q_diff,
        'Abs(Q_Diff)': np.abs(q_diff)
    })

    # Add summary statistics
    summary_stats = {
        'Metric': [
            'Max abs(head diff)',
            'Mean abs(head diff)',
            'Std abs(head diff)',
            'Max abs(q diff)',
            'Mean abs(q diff)',
            'Std abs(q diff)'
        ],
        'Value': [
            np.max(np.abs(head_diff)),
            np.mean(np.abs(head_diff)),
            np.std(head_diff),
            np.max(np.abs(q_diff[~np.isnan(fortran_flow[:n_compare])])),
            np.mean(np.abs(q_diff[~np.isnan(fortran_flow[:n_compare])])),
            np.std(q_diff[~np.isnan(fortran_flow[:n_compare])])
        ]
    }
    summary_df = pd.DataFrame(summary_stats)

    # Add BC-specific statistics if nbc is provided
    if nbc is not None:
        bc_stats = []
        for bc_type in [0, 1, 2]:
            mask = nbc[:n_compare] == bc_type
            if np.any(mask):
                bc_name = bc_type_map[bc_type]
                # Head statistics
                head_mask = mask
                if np.any(head_mask):
                    bc_stats.extend([
                        [f'{bc_name} - Max abs(head diff)', np.max(np.abs(head_diff[head_mask]))],
                        [f'{bc_name} - Mean abs(head diff)', np.mean(np.abs(head_diff[head_mask]))]
                    ])
                
                # Flow statistics (only for nodes with valid FORTRAN flow data)
                flow_mask = mask & ~np.isnan(fortran_flow[:n_compare])
                if np.any(flow_mask):
                    bc_stats.extend([
                        [f'{bc_name} - Max abs(q diff)', np.max(np.abs(q_diff[flow_mask]))],
                        [f'{bc_name} - Mean abs(q diff)', np.mean(np.abs(q_diff[flow_mask]))]
                    ])
        
        if bc_stats:
            bc_summary_df = pd.DataFrame(bc_stats, columns=['Metric', 'Value'])
            summary_df = pd.concat([summary_df, bc_summary_df], ignore_index=True)

    # Create Excel writer
    output_path = Path(fortran_out_path).with_suffix('.xlsx')
    with pd.ExcelWriter(output_path) as writer:
        # Write main comparison data
        df.to_excel(writer, sheet_name='Nodal Comparison', index=False)
        
        # Write summary statistics
        summary_df.to_excel(writer, sheet_name='Summary Statistics', index=False)
        
        # Auto-adjust column widths
        for sheet_name in writer.sheets:
            worksheet = writer.sheets[sheet_name]
            for idx, col in enumerate(df.columns):
                max_length = max(
                    df[col].astype(str).apply(len).max(),
                    len(col)
                )
                worksheet.column_dimensions[chr(65 + idx)].width = max_length + 2

    if verbose:
        print(f"\nComparison results saved to: {output_path}")
        print("\n=== Python vs FORTRAN Nodal Head Comparison ===")
        print(f"Max abs(head diff): {np.max(np.abs(head_diff)):.4e}")
        print(f"Mean abs(head diff): {np.mean(np.abs(head_diff)):.4e}")
        print(f"Std abs(head diff): {np.std(head_diff):.4e}")

        print("\n=== Python vs FORTRAN Nodal Flow Comparison ===")
        # Only compare where FORTRAN flow is not nan
        valid = ~np.isnan(fortran_flow[:n_compare])
        if np.any(valid):
            print(f"Max abs(q diff): {np.max(np.abs(q_diff[valid])):.4e}")
            print(f"Mean abs(q diff): {np.mean(np.abs(q_diff[valid])):.4e}")
            print(f"Std abs(q diff): {np.std(q_diff[valid]):.4e}")
        else:
            print("No FORTRAN flow data to compare.")

        if nbc is not None:
            print("\n=== Boundary Condition Statistics ===")
            for bc_type in [0, 1, 2]:
                mask = nbc[:n_compare] == bc_type
                if np.any(mask):
                    bc_name = bc_type_map[bc_type]
                    print(f"\n{bc_name} nodes:")
                    print(f"  Count: {np.sum(mask)}")
                    # Head statistics
                    print(f"  Max abs(head diff): {np.max(np.abs(head_diff[mask])):.4e}")
                    print(f"  Mean abs(head diff): {np.mean(np.abs(head_diff[mask])):.4e}")
                    # Flow statistics (only for nodes with valid FORTRAN flow data)
                    valid_q = mask & ~np.isnan(fortran_flow[:n_compare])
                    if np.any(valid_q):
                        print(f"  Max abs(q diff): {np.max(np.abs(q_diff[valid_q])):.4e}")
                        print(f"  Mean abs(q diff): {np.mean(np.abs(q_diff[valid_q])):.4e}")
                    else:
                        print("  No valid flow data for comparison")

    return head_diff, q_diff

def plot_kr_field(coords, elements, kr_vals, title='Kr Field'):
    """
    Plot the kr field at element centroids.
    """
    import numpy as np
    centroids = np.mean(coords[elements], axis=1)
    plt.figure(figsize=(10, 4))
    sc = plt.scatter(centroids[:, 0], centroids[:, 1], c=kr_vals, cmap='viridis', s=30, edgecolor='k')
    plt.colorbar(sc, label='k_r')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(title)
    plt.gca().set_aspect('equal')
    plt.tight_layout()
    plt.show()

def plot_nodal_flows(coords, q, title='Nodal Flow Values', show_values=False, cmap='RdBu_r'):
    """
    Plot nodal flow values for debugging.
    
    Parameters:
    -----------
    coords : ndarray
        Node coordinates array (n_nodes x 2)
    q : ndarray
        Nodal flow values array (n_nodes)
    title : str, optional
        Plot title
    show_values : bool, optional
        Whether to show the actual flow values on the plot
    cmap : str, optional
        Colormap to use for the scatter plot
    """
    plt.figure(figsize=(10, 8))
    
    # Create scatter plot of nodes colored by flow value
    scatter = plt.scatter(coords[:, 0], coords[:, 1], 
                         c=q, cmap=cmap, 
                         s=100, edgecolor='k')
    
    # Add colorbar
    cbar = plt.colorbar(scatter)
    cbar.set_label('Flow Value')
    
    # Add node numbers and flow values if requested
    if show_values:
        for i, (x, y) in enumerate(coords):
            plt.text(x, y, f'Node {i+1}\n{q[i]:.2e}', 
                    ha='center', va='center',
                    fontsize=8)
    
    plt.title(title)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.grid(True)
    plt.axis('equal')
    
    return plt.gcf()

def classify_nodes_by_phreatic_surface(coords, head):
    """
    Returns a boolean mask: True if node is above the phreatic surface (unsaturated), False otherwise.
    """
    return head < coords[:, 1]

def debug_nodal_flows_above_phreatic(coords, head, q, title='Nodal Flows vs Phreatic Surface'):
    above_mask = classify_nodes_by_phreatic_surface(coords, head)
    below_mask = ~above_mask

    # Print summary statistics
    print(f'Nodes above phreatic surface: {np.sum(above_mask)}')
    print(f'Nodes below phreatic surface: {np.sum(below_mask)}')
    print('--- Above phreatic surface ---')
    print(f'Max |q|: {np.max(np.abs(q[above_mask])):.3e}')
    print(f'Mean |q|: {np.mean(np.abs(q[above_mask])):.3e}')
    print(f'Min |q|: {np.min(np.abs(q[above_mask])):.3e}')
    print('--- Below phreatic surface ---')
    print(f'Max |q|: {np.max(np.abs(q[below_mask])):.3e}')
    print(f'Mean |q|: {np.mean(np.abs(q[below_mask])):.3e}')
    print(f'Min |q|: {np.min(np.abs(q[below_mask])):.3e}')

    # Plot
    plt.figure(figsize=(10, 8))
    plt.scatter(coords[below_mask, 0], coords[below_mask, 1], c=q[below_mask], cmap='Blues', label='Below Phreatic', s=60)
    plt.scatter(coords[above_mask, 0], coords[above_mask, 1], c=q[above_mask], cmap='Reds', label='Above Phreatic', s=60, marker='^')
    plt.colorbar(label='Nodal q')
    plt.legend()
    plt.title(title)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.grid(True)
    plt.axis('equal')
    plt.show()

def debug_kr_above_phreatic(coords, head, kr, title='kr vs Phreatic Surface'):
    """
    Plot and summarize kr values above and below the phreatic surface.
    coords: (n_nodes, 2) array of node coordinates
    head: (n_nodes,) array of nodal heads
    kr: (n_nodes,) array of nodal kr values
    """
    import numpy as np
    import matplotlib.pyplot as plt
    above_mask = classify_nodes_by_phreatic_surface(coords, head)
    below_mask = ~above_mask

    # Print summary statistics
    print(f'Nodes above phreatic surface: {np.sum(above_mask)}')
    print(f'Nodes below phreatic surface: {np.sum(below_mask)}')
    print('--- Above phreatic surface ---')
    print(f'Max kr: {np.max(kr[above_mask]):.3e}')
    print(f'Mean kr: {np.mean(kr[above_mask]):.3e}')
    print(f'Min kr: {np.min(kr[above_mask]):.3e}')
    print('--- Below phreatic surface ---')
    print(f'Max kr: {np.max(kr[below_mask]):.3e}')
    print(f'Mean kr: {np.mean(kr[below_mask]):.3e}')
    print(f'Min kr: {np.min(kr[below_mask]):.3e}')

    # Plot
    plt.figure(figsize=(10, 8))
    plt.scatter(coords[below_mask, 0], coords[below_mask, 1], c=kr[below_mask], cmap='Blues', label='Below Phreatic', s=60)
    plt.scatter(coords[above_mask, 0], coords[above_mask, 1], c=kr[above_mask], cmap='Reds', label='Above Phreatic', s=60, marker='^')
    plt.colorbar(label='kr')
    plt.legend()
    plt.title(title)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.grid(True)
    plt.axis('equal')
    plt.show()

def debug_kr_elements(coords, elements, head, kr_elem, title='Element kr vs Phreatic Surface'):
    """
    Plot and summarize kr values at the element level, classified by phreatic surface.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection

    # Compute average head and average elevation for each element
    avg_head = np.mean(head[elements], axis=1)
    avg_elev = np.mean(coords[elements, 1], axis=1)
    above_mask = avg_head < avg_elev
    below_mask = ~above_mask

    print(f'Elements above phreatic surface: {np.sum(above_mask)}')
    print(f'Elements below phreatic surface: {np.sum(below_mask)}')
    print('--- Above phreatic surface ---')
    print(f'Max kr: {np.max(kr_elem[above_mask]):.3e}')
    print(f'Mean kr: {np.mean(kr_elem[above_mask]):.3e}')
    print(f'Min kr: {np.min(kr_elem[above_mask]):.3e}')
    print('--- Below phreatic surface ---')
    print(f'Max kr: {np.max(kr_elem[below_mask]):.3e}')
    print(f'Mean kr: {np.mean(kr_elem[below_mask]):.3e}')
    print(f'Min kr: {np.min(kr_elem[below_mask]):.3e}')

    # Plot
    verts = [coords[tri] for tri in elements]
    fig, ax = plt.subplots(figsize=(10, 8))
    pc = PolyCollection(verts, array=kr_elem, cmap='viridis', edgecolor='k')
    ax.add_collection(pc)
    ax.autoscale()
    plt.colorbar(pc, ax=ax, label='kr')
    plt.title(title)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.grid(True)
    plt.show()

def debug_ke_elements(coords, elements, head, ke_list, title='Element ke vs Phreatic Surface'):
    """
    Print and compare the norm of ke for elements above vs below the phreatic surface.
    ke_list: list or array of 3x3 element stiffness matrices
    """
    import numpy as np
    # Compute average head and average elevation for each element
    avg_head = np.mean(head[elements], axis=1)
    avg_elev = np.mean(coords[elements, 1], axis=1)
    above_mask = avg_head < avg_elev
    below_mask = ~above_mask

    # Compute Frobenius norm of each ke
    ke_norms = np.array([np.linalg.norm(ke) if ke is not None else np.nan for ke in ke_list])

    print(f'Elements above phreatic surface: {np.sum(above_mask)}')
    print(f'Elements below phreatic surface: {np.sum(below_mask)}')
    print('--- Above phreatic surface ---')
    print(f'Max ke norm: {np.nanmax(ke_norms[above_mask]):.3e}')
    print(f'Mean ke norm: {np.nanmean(ke_norms[above_mask]):.3e}')
    print(f'Min ke norm: {np.nanmin(ke_norms[above_mask]):.3e}')
    print('--- Below phreatic surface ---')
    print(f'Max ke norm: {np.nanmax(ke_norms[below_mask]):.3e}')
    print(f'Mean ke norm: {np.nanmean(ke_norms[below_mask]):.3e}')
    print(f'Min ke norm: {np.nanmin(ke_norms[below_mask]):.3e}')