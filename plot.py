
import matplotlib.pyplot as plt
import matplotlib.tri as tri
from matplotlib.ticker import MaxNLocator
import numpy as np
from scipy.interpolate import griddata


def plot_mesh(coords, elements, element_materials, show_nodes=False, show_bc=False, nbc=None):
    """
    Plots a mesh colored by material zone.
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(12, 5))
    materials = np.unique(element_materials)
    cmap = plt.get_cmap("tab10", len(materials))
    mat_to_color = {mat: cmap(i) for i, mat in enumerate(materials)}

    for idx, tri_nodes in enumerate(elements):
        polygon = coords[tri_nodes]
        color = mat_to_color[element_materials[idx]]
        ax.fill(*zip(*polygon), edgecolor='k', facecolor=color, linewidth=0.5)

    if show_nodes:
        ax.plot(coords[:, 0], coords[:, 1], 'k.', markersize=2)

    legend_handles = [
        plt.Line2D([0], [0], color=cmap(i), lw=4, label=f"Material {mat}")
        for i, mat in enumerate(materials)
    ]

    if show_bc and nbc is not None:
        bc1 = coords[nbc == 1]
        bc2 = coords[nbc == 2]
        if len(bc1) > 0:
            h1, = ax.plot(bc1[:, 0], bc1[:, 1], 'ro', label="Fixed Head (nbc=1)")
            legend_handles.append(h1)
        if len(bc2) > 0:
            h2, = ax.plot(bc2[:, 0], bc2[:, 1], 'bs', label="Exit Face (nbc=2)")
            legend_handles.append(h2)

    # Single combined legend outside the plot
    ax.legend(
        handles=legend_handles,
        loc='upper center',
        bbox_to_anchor=(0.5, -0.1),
        ncol=3,  # or more, depending on how many items you have
        frameon=False
    )
    ax.set_aspect("equal")
    ax.set_title("SEEP2D Mesh with Material Zones")
    # plt.subplots_adjust(bottom=0.2)  # Add vertical cushion
    plt.tight_layout()
    plt.show()


def plot_solution_OLD(coords, elements, head, flowrate=None, levels=20):
    """
    Plots filled contour of total head and overlays solid contour lines.
    """

    plt.figure(figsize=(12, 5))

    triang = tri.Triangulation(coords[:, 0], coords[:, 1], elements) # Create triangulation from coordinates and elements

    # Filled contours
    vmin = np.min(head)
    vmax = np.max(head)
    contour_levels = np.linspace(vmin, vmax, levels)

    contourf = plt.tricontourf(triang, head, levels=levels, cmap="Spectral_r", vmin=vmin, vmax=vmax)

    # Options for cmap:
    # plasma, viridis, inferno, magma, cividis, coolwarm, Spectral, Blues, YlGnBu, RdYlBu, etc
    # Append "_r" to reverse the colormap (e.g., "Spectral_r")
    # You can also use a custom colormap if desired, e.g., from matplotlib.colors


    # Solid contour lines
    contour = plt.tricontour(triang, head, levels=levels, colors="k", linewidths=0.5)

    # Colorbar with nicely rounded ticks
    cbar = plt.colorbar(contourf, label="Total Head")
    cbar.locator = MaxNLocator(nbins=10, steps=[1, 2, 5])  # Pick good-looking tick steps
    cbar.update_ticks()

    #plt.clabel(contour, inline=True, fontsize=8, fmt="%.2f", colors="k") # Uncomment to label contours
    #plt.triplot(triang, color="k", linewidth=0.2, alpha=0.4) # Uncomment if you want to show mesh edges

    title = "Total Head Contours"
    if flowrate is not None:
        title += f" — Total Flowrate: {flowrate:.3f}"
    plt.title(title)
    plt.gca().set_aspect("equal")
    plt.tight_layout()
    plt.show()


def plot_solution(coords, elements, head, phi=None, flowrate=None, levels=20, base_mat=None, k1_by_mat=None):
    """
    Plots head contours and optionally overlays flowlines (phi) based on flow function.

    Arguments:
        coords: (n_nodes, 2) array of node coordinates
        elements: (n_elements, 3) array of triangle indices
        element_materials: (n_elements,) array of material IDs (1-based)
        head: (n_nodes,) array of total head values
        phi: (n_nodes,) array of flow potential values (optional)
        flowrate: total flowrate (optional, required for phi contours)
        levels: number of head contour levels
        base_mat: material ID (1-based) used to compute k for flow function
        k1_by_mat: (n_materials,) array of k1 values by material ID (required if base_mat is given)
    """
    import matplotlib.pyplot as plt
    import matplotlib.tri as tri
    from matplotlib.ticker import MaxNLocator
    import numpy as np

    triang = tri.Triangulation(coords[:, 0], coords[:, 1], elements)
    plt.figure(figsize=(12, 5))

    vmin = np.min(head)
    vmax = np.max(head)
    hdrop = vmax - vmin
    contour_levels = np.linspace(vmin, vmax, levels)
    contourf = plt.tricontourf(triang, head, levels=contour_levels, cmap="Spectral_r", vmin=vmin, vmax=vmax, alpha=0.5)
    cbar = plt.colorbar(contourf, label="Total Head")
    cbar.locator = MaxNLocator(nbins=10, steps=[1, 2, 5])
    cbar.update_ticks()

    # Solid lines for head contours
    plt.tricontour(triang, head, levels=contour_levels, colors="k", linewidths=0.5)

    # Overlay flowlines if phi is available
    if phi is not None and flowrate is not None and base_mat is not None and k1_by_mat is not None:
        # Materials are 1-based, so adjust index
        base_k = k1_by_mat[base_mat - 1]
        ne = levels - 1
        nf = (flowrate * ne) / (base_k * hdrop)
        phi_levels = round(nf) + 1
        print(f"Computed nf: {nf:.2f}, using {phi_levels} φ contours (base k={base_k}, head drop={hdrop:.3f})")
        phi_contours = np.linspace(np.min(phi), np.max(phi), phi_levels)
        plt.tricontour(triang, phi, levels=phi_contours, colors="blue", linewidths=0.7, linestyles="solid")

    # Plot the mesh boundary
    boundary = get_ordered_boundary(coords, elements)
    plt.plot(boundary[:, 0], boundary[:, 1], color="black", linewidth=1.0, label="Mesh Boundary")

    title = "Flow Net: Head Contours"
    if phi is not None:
        title += " and Flowlines"
    if flowrate is not None:
        title += f" — Total Flowrate: {flowrate:.3f}"
    plt.title(title)
    plt.gca().set_aspect("equal")
    plt.tight_layout()
    plt.show()



def get_ordered_boundary(coords, elements):
    """
    Extracts the outer boundary of the mesh and returns it as an ordered array of points.

    Returns:
        np.ndarray of shape (N, 2): boundary coordinates in order (closed loop)
    """
    import numpy as np
    from collections import defaultdict, deque

    # Step 1: Count all edges
    edge_count = defaultdict(int)
    edge_to_nodes = {}

    for tri in elements:
        for i in range(3):
            a, b = sorted((tri[i], tri[(i + 1) % 3]))
            edge_count[(a, b)] += 1
            edge_to_nodes[(a, b)] = (tri[i], tri[(i + 1) % 3])  # preserve direction

    # Step 2: Keep only boundary edges (appear once)
    boundary_edges = [edge_to_nodes[e] for e, count in edge_count.items() if count == 1]

    if not boundary_edges:
        raise ValueError("No boundary edges found.")

    # Step 3: Build adjacency for boundary walk
    adj = defaultdict(list)
    for a, b in boundary_edges:
        adj[a].append(b)
        adj[b].append(a)

    # Step 4: Walk the boundary in order
    start = boundary_edges[0][0]
    boundary_loop = [start]
    visited = set([start])
    current = start

    while True:
        neighbors = [n for n in adj[current] if n not in visited]
        if not neighbors:
            break
        next_node = neighbors[0]
        boundary_loop.append(next_node)
        visited.add(next_node)
        current = next_node
        if current == start:
            break

    return coords[boundary_loop]