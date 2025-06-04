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
        
    if self.iuntyp == 0:
        print("Solving confined SEEP2D problem (linear)...")
        # Apply dummy Dirichlet BCs: fix head at 5 nodes
        n = self.coords.shape[0]
        bcs = [(i, self.fx[i]) for i in range(len(self.nbc)) if self.nbc[i] == 1]
        
        k1 = self.k1_by_mat[self.element_materials]
        k2 = self.k2_by_mat[self.element_materials]
        angle = self.angle_by_mat[self.element_materials]

        head = solve_confined(self.coords, self.elements, bcs)
        pressure = head - self.coords[:, 1]
        velocity = np.zeros_like(self.coords)  # Placeholder
        self.solution = {"head": head, "pressure": pressure, "velocity": velocity}
        if hasattr(self, "export_path"):
            export_solution_csv(self.export_path, self.coords, head, pressure, velocity)
        return
    elif self.iuntyp == 2:
        print(\"Solving unconfined SEEP2D problem using kr frontal function...\")
        bcs = [(i, self.fx[i]) for i in range(len(self.nbc)) if self.nbc[i] in (1, 2)]
        
        k1 = self.k1_by_mat[self.element_materials]
        k2 = self.k2_by_mat[self.element_materials]
        angle = self.angle_by_mat[self.element_materials]

        h, pressure, velocity = solve_unsaturated(self.coords, self.elements, self.nbc, self.fx)
        self.solution = {\"head\": h, \"pressure\": pressure, \"velocity\": velocity}
        if hasattr(self, \"export_path\"):
            export_solution_csv(self.export_path, self.coords, h, pressure, velocity)

            print("Running seepage analysis...")
    # Placeholder dummy results for demo purposes
    n = self.coords.shape[0]
    head = np.ones(n)
    pressure = np.zeros(n)
    velocity = np.zeros((n, 2))

    self.solution = {
        "head": head,
        "pressure": pressure,
        "velocity": velocity
    }

    if hasattr(self, "coords") and hasattr(self, "export_path"):
        export_solution_csv(self.export_path, self.coords, head, pressure, velocity)

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
        lines = [line.strip() for line in f if line.strip()]

    assert lines[0].startswith("GMS SEEP2D Simulation"), "Invalid SEEP2D header"
    parts = lines[1].split()
    num_nodes = int(parts[0])
    num_elements = int(parts[1])
    num_materials = int(parts[2])
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
    node_lines = lines[2:2 + num_nodes]
    element_lines = lines[2 + num_nodes:]

    coords = []
    node_materials = []
    node_ids = []
    nbc_flags = []
    fx_vals = []

    for line in node_lines:
        nums = [float(n) if '.' in n or 'e' in n.lower() else int(n)
                for n in re.findall(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?', line)]
        if len(nums) >= 5:
            nid, flag, mat, x, y = nums[:5]
        elif len(nums) == 4:
            nid, flag, x, y = nums
            mat = 0
        elif len(nums) == 3:
            nid, x, y = nums
            flag = mat = 0
        else:
            continue
        node_ids.append(int(nid))
        node_materials.append(int(mat))
        coords.append((x, y))
        nbc_flags.append(int(flag))
        fx_vals.append(float(y) if flag == 2 else 0.0)  # set fx to elevation for exit face

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
        "h0_by_mat": h0_array
    }
        "node_ids": np.array(node_ids, dtype=int),
        "node_materials": np.array(node_materials),
        "elements": np.array(elements, dtype=int) - 1,
        "element_materials": np.array(element_mats)
    }

def export_solution_csv(filename, coords, head, pressure, velocity):
    """Exports nodal results to a CSV file."""
    import pandas as pd
    df = pd.DataFrame({
        "x": coords[:, 0],
        "y": coords[:, 1],
        "head": head,
        "pressure": pressure,
        "vx": velocity[:, 0],
        "vy": velocity[:, 1]
    })
    df.to_csv(filename, index=False)
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



def solve_confined(coords, elements, dirichlet_bcs, k1_vals=None, k2_vals=None, angles=None):
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

        # Get anisotropic conductivity
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
        Kmat = R.T @ np.diag([k1, k2]) @ R

        ke = area * grad.T @ Kmat @ grad

        for a in range(3):
            for b_ in range(3):
                A[tri[a], tri[b_]] += ke[a, b_]

    for node, value in dirichlet_bcs:
        A[node, :] = 0
        A[node, node] = 1
        b[node] = value

    head = spsolve(A.tocsr(), b)
    return head, A

def kr_frontal(p, kr0, h0):
    """Compute relative conductivity using frontal function."""
    import numpy as np
    kr = np.ones_like(p)
    mask1 = (p <= 0) & (p > h0)
    mask2 = (p <= h0)
    kr[mask1] = kr0 + (1 - kr0) * p[mask1] / h0
    kr[mask2] = kr0
    return kr


def solve_unsaturated(coords, elements, nbc, fx, kr0=0.1, h0=-1.0,
                      k1_vals=1.0, k2_vals=1.0, angles=0.0,
                      max_iter=50, tol=1e-4):
    """
    Iterative FEM solver for unsaturated flow using frontal kr function with anisotropic K.
    """
    import numpy as np
    from scipy.sparse import lil_matrix
    from scipy.sparse.linalg import spsolve

    n_nodes = coords.shape[0]
    y = coords[:, 1]
    h = np.full(n_nodes, np.mean(fx[nbc == 1]) if np.any(nbc == 1) else 1.0)
    bcs = [(i, fx[i]) for i in range(n_nodes) if nbc[i] in (1, 2)]

    scalar_k = np.isscalar(k1_vals)

    for iteration in range(max_iter):
        A = lil_matrix((n_nodes, n_nodes))
        b = np.zeros(n_nodes)
        p = h - y
        kr = kr_frontal(p, kr0, h0)

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

            # Get anisotropic K matrix
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
            Kmat = R.T @ np.diag([k1, k2]) @ R

            kr_elem = np.mean([kr[i], kr[j], kr[k]])
            ke = kr_elem * area * grad.T @ Kmat @ grad

            for a in range(3):
                for b_ in range(3):
                    A[tri[a], tri[b_]] += ke[a, b_]

        for node, value in bcs:
            A[node, :] = 0
            A[node, node] = 1
            b[node] = value

        h_new = spsolve(A.tocsr(), b)
        if np.max(np.abs(h_new - h)) < tol:
            break
        h = h_new

    pressure = h - y
    velocity = np.zeros_like(coords)  # Placeholder
    return h, pressure, velocity, A