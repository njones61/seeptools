"""
SEEP2D Python Translation
-------------------------

Finite element seepage analysis tool originally developed in Fortran.
Ported to Python for improved modularity and extensibility.

Author (Original): Fred Tracy, ERDC
Python Port: [Your Name or Group]
"""

import numpy as np


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
            kr0 = float(self.kr0_by_mat[0])
            h0 = float(self.h0_by_mat[0])
            head, A, q = solve_unsaturated(
                self.coords,
                self.elements,
                self.nbc,
                self.fx,
                kr0=kr0,
                h0=h0,
                k1_vals=k1,
                k2_vals=k2,
                angles=angle
            )
        else:
            head, A, q = solve_confined(self.coords, self.elements, bcs, k1, k2, angle)

        gamma_w = self.unit_weight
        pressure = gamma_w * (head - self.coords[:, 1])
        velocity = compute_velocity(self.coords, self.elements, head, k1, k2, angle)
        flowrate = q[(self.nbc == 1) & (q > 0)].sum()

        # Solve for potential function φ for flow lines
        dirichlet_phi_bcs = create_flow_potential_bc(self.coords, self.elements, q)
        phi = solve_flow_function(self.coords, self.elements, velocity, dirichlet_phi_bcs)

        print(f"phi min: {np.min(phi):.3f}, max: {np.max(phi):.3f}")

        self.solution = {
            "head": head,
            "pressure": pressure,
            "velocity": velocity,
            "q": q,
            "phi": phi,
            "flowrate": flowrate
        }

        if hasattr(self, "export_path"):
            export_solution_csv(self.export_path, self.coords, head, pressure, velocity, q, phi, flowrate)


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

            if bc_type == 1 and len(line) >= 54:
                fx_val = float(line[40:54])
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

def solve_confined(coords, elements, dirichlet_bcs, k1_vals, k2_vals, angles=None):
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

    return head, A, q


def solve_unsaturated(coords, elements, nbc, fx, kr0=0.1, h0=-1.0,
                      k1_vals=1.0, k2_vals=1.0, angles=0.0,
                      max_iter=50, tol=1e-4):
    """
    Complete FORTRAN-based unsaturated flow solver following the exact sequence:
    1. Initialize arrays (lines 100, 360-400)
    2. Iteration loop (420-490)
    3. FORTRAN relaxation and convergence logic
    4. Exit face handling with count array
    """
    import numpy as np
    from scipy.sparse import lil_matrix
    from scipy.sparse.linalg import spsolve

    n_nodes = coords.shape[0]
    y = coords[:, 1]

    # FORTRAN initialization (lines 100, 200)
    hlast = np.full(n_nodes, 1e30)  # hlast(n) = 1.d30
    count = np.zeros(n_nodes)  # count(n) = 0.d0
    flow = np.zeros(n_nodes)  # flow(n) = 0.d0

    # FORTRAN: Pre-process fx for exit face nodes (line 200)
    fx_corrected = fx.copy()
    for n in range(n_nodes):
        if nbc[n] == 1:
            fx_corrected[n] = fx[n]  # Fixed head
        elif nbc[n] == 2:
            fx_corrected[n] = y[n]  # Exit face: fx(n) = y(n)

    # FORTRAN count array initialization (lines 360-400)
    # Find lowest exit face nodes and activate them
    exit_nodes = np.where(nbc == 2)[0]
    if len(exit_nodes) > 0:
        exit_elevations = y[exit_nodes]
        nsmall = []
        ysmall = []

        # Find 2 lowest exit face nodes (FORTRAN nsmall logic)
        sorted_indices = np.argsort(exit_elevations)
        for i in range(min(2, len(exit_nodes))):
            nsmall.append(exit_nodes[sorted_indices[i]])
            ysmall.append(exit_elevations[sorted_indices[i]])

        # Activate lowest exit face nodes (lines 390-400)
        if len(nsmall) > 0:
            count[nsmall[0]] = 1.0
            if len(nsmall) > 1:
                count[nsmall[1]] = 1.0

        print(f"FORTRAN: Activated exit face nodes {nsmall}")

    # FORTRAN convergence tolerance (line 400)
    ymin, ymax = y.min(), y.max()
    eps = (ymax - ymin) * 0.0001
    relax = 1.0

    print(f"FORTRAN: eps = {eps:.6e}, ymin = {ymin:.3f}, ymax = {ymax:.3f}")

    # Initialize head field
    head_initial = np.mean(fx_corrected[nbc == 1]) if np.any(nbc == 1) else 1.0
    r = np.full(n_nodes, head_initial)  # FORTRAN r array

    # FORTRAN main iteration loop with correct convergence logic
    iconv = 0  # FORTRAN convergence counter

    for k in range(1, max_iter + 1):
        print(f"FORTRAN iteration {k}")

        # FORTRAN relaxation schedule (lines 420-440)
        if k > 20:
            relax = 0.5
        if k > 40:
            relax = 0.2
        if k > 60:
            relax = 0.1
        if k > 80:
            relax = 0.05
        if k > 100:
            relax = 0.02
        if k > 120:
            relax = 0.01

        # Assemble stiffness matrix
        A = lil_matrix((n_nodes, n_nodes))
        b = np.zeros(n_nodes)

        # Compute relative permeability
        p = r - y
        kr_node = kr_frontal(p, kr0, h0)

        # Element assembly
        for idx, tri in enumerate(elements):
            i, j, k_node = tri
            xi, yi = coords[i]
            xj, yj = coords[j]
            xk, yk = coords[k_node]

            area = 0.5 * np.linalg.det([[1, xi, yi], [1, xj, yj], [1, xk, yk]])
            if area <= 0:
                continue

            beta = np.array([yj - yk, yk - yi, yi - yj])
            gamma = np.array([xk - xj, xi - xk, xj - xi])
            grad = np.array([beta, gamma]) / (2 * area)

            # Material properties
            if np.isscalar(k1_vals):
                k1, k2, theta = k1_vals, k2_vals, angles
            else:
                k1, k2, theta = k1_vals[idx], k2_vals[idx], angles[idx]

            # Anisotropic conductivity
            theta_rad = np.radians(theta)
            c, s = np.cos(theta_rad), np.sin(theta_rad)
            R = np.array([[c, s], [-s, c]])
            Kmat = R.T @ np.diag([k1, k2]) @ R

            # Average relative permeability
            kr_avg = np.mean([kr_node[i], kr_node[j], kr_node[k_node]])
            ke = kr_avg * area * grad.T @ Kmat @ grad

            for a in range(3):
                for b_ in range(3):
                    A[tri[a], tri[b_]] += ke[a, b_]

        # FORTRAN boundary condition application (modify subroutine)
        for i in range(n_nodes):
            if nbc[i] <= 0:  # No BC
                continue
            elif nbc[i] == 2:  # Exit face
                if count[i] == 0:  # Inactive exit face
                    continue
                h_bc = y[i]  # Use elevation
            else:  # Fixed head (nbc == 1)
                h_bc = fx_corrected[i]

            # Apply Dirichlet BC
            A[i, :] = 0
            A[i, i] = 1
            b[i] = h_bc

        # Solve system
        r_new = spsolve(A.tocsr(), b)

        # FORTRAN convergence check and relaxation (lines 430-440)
        ifirst = 1
        dhmax = 0.0

        for i in range(n_nodes):
            # Apply relaxation (line 430)
            if k > 1:
                r_new[i] = r_new[i] * relax + hlast[i] * (1 - relax)

            # Check if head >= elevation (line 430-440)
            if r_new[i] >= y[i]:
                dh = abs(r_new[i] - hlast[i]) if k > 1 else 0
                dhmax = max(dhmax, dh)
            else:
                ifirst = 0  # At least one node below ground

        # Update hlast (line 450)
        hlast = r.copy()

        # FORTRAN exit face node updates (lines 460-480)
        for i in range(n_nodes):
            if nbc[i] != 2:  # Only exit face nodes
                continue

            if count[i] == 0:  # Inactive exit face
                if r_new[i] >= y[i] - 0.01:  # Becomes wet
                    count[i] = 1.0
                    r_new[i] = y[i]
                    hlast[i] = y[i]
                    print(f"  Activated exit face node {i}")
            else:  # Active exit face
                if r_new[i] < y[i]:
                    r_new[i] = y[i]  # Force to elevation

        # Update solution
        r = r_new.copy()

        # FORTRAN convergence logic (CORRECT VERSION)
        print(f"  dhmax = {dhmax:.6e}, eps = {eps:.6e}, relax = {relax:.3f}")

        if dhmax > eps:
            iconv = 0  # Reset counter
        else:
            iconv += 1  # Increment counter

        print(f"  iconv = {iconv} (need 2 consecutive for convergence)")

        if iconv == 2:
            print(f"FORTRAN convergence: 2 consecutive iterations with dhmax ≤ eps")
            break

        # Early exit for ifirst=1 (all nodes above ground - rare in unsaturated flow)
        if ifirst == 1:
            print(f"FORTRAN early exit: all nodes above ground surface")
            break

    else:
        print(f"WARNING: Did not converge after {max_iter} iterations, dhmax = {dhmax:.6e}")

    # Final flow computation
    A_final = A.copy()
    q = A_final.tocsr() @ r

    return r, A_final, q


def kr_frontal(p, kr0, h0):
    """
    FORTRAN fkrelf function - relative permeability for linear frontal model
    """
    import numpy as np
    p = np.asarray(p)
    kr = np.ones_like(p)

    # Saturated: p >= 0, kr = 1.0
    # Transition: h0 < p < 0, linear
    # Dry: p <= h0, kr = kr0

    mask_transition = (p < 0) & (p > h0)
    mask_dry = (p <= h0)

    kr[mask_transition] = kr0 + (1 - kr0) * p[mask_transition] / h0
    kr[mask_dry] = kr0

    return kr

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
def solve_flow_function(coords, elements, velocity, dirichlet_nodes):
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