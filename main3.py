import numpy as np
import matplotlib.pyplot as plt
from seep2d import solve_unsaturated


# Add this complete debugging script to check what's actually happening
def debug_s2unc_boundary_conditions():
    """
    Complete debug of the s2unc boundary conditions to understand the flow pattern issue.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from seep2d import read_seep2d_input

    print("=== COMPLETE s2unc BOUNDARY CONDITION DEBUG ===")

    # Load the problem
    model_data = read_seep2d_input("samples/s2unc/s2unc.s2d")

    coords = model_data['coords']
    nbc = model_data['nbc']
    fx = model_data['fx']

    print(f"Mesh: {len(coords)} nodes")
    print(f"Coordinate ranges:")
    print(f"  X: {coords[:, 0].min():.1f} to {coords[:, 0].max():.1f}")
    print(f"  Y: {coords[:, 1].min():.1f} to {coords[:, 1].max():.1f}")

    # Analyze each boundary condition type
    for bc_type in [0, 1, 2]:
        nodes = np.where(nbc == bc_type)[0]
        if len(nodes) == 0:
            continue

        print(f"\n--- BC Type {bc_type} ({len(nodes)} nodes) ---")

        if bc_type == 0:
            print("No-flow (impermeable) boundary")
        elif bc_type == 1:
            print("Fixed head boundary")
            unique_heads = np.unique(fx[nodes])
            print(f"  Head values: {unique_heads}")
        elif bc_type == 2:
            print("Exit face (seepage face) boundary")

        # Spatial distribution
        node_coords = coords[nodes]
        print(f"  X range: {node_coords[:, 0].min():.1f} to {node_coords[:, 0].max():.1f}")
        print(f"  Y range: {node_coords[:, 1].min():.1f} to {node_coords[:, 1].max():.1f}")

        # Show first few node details
        print(f"  First 5 nodes:")
        for i in range(min(5, len(nodes))):
            node = nodes[i]
            x, y = coords[node]
            if bc_type == 1:
                print(f"    Node {node}: ({x:.1f}, {y:.1f}), head={fx[node]:.1f}")
            else:
                print(f"    Node {node}: ({x:.1f}, {y:.1f})")

    # Check if boundary conditions make physical sense
    print("\n=== PHYSICAL SENSE CHECK ===")

    fixed_nodes = np.where(nbc == 1)[0]
    exit_nodes = np.where(nbc == 2)[0]

    if len(fixed_nodes) > 0:
        fixed_coords = coords[fixed_nodes]
        fixed_heads = fx[fixed_nodes]

        # Check if fixed heads are on the upstream (left) side
        x_min = coords[:, 0].min()
        x_threshold = x_min + (coords[:, 0].max() - x_min) * 0.2  # 20% from left
        upstream_fixed = np.sum(fixed_coords[:, 0] <= x_threshold)

        print(f"Fixed head nodes on upstream side: {upstream_fixed}/{len(fixed_nodes)}")
        print(f"Fixed head values: {np.unique(fixed_heads)}")

        if upstream_fixed == 0:
            print("⚠️  WARNING: No fixed head nodes on upstream (left) side!")
            print("   This could explain wrong flow direction.")

    if len(exit_nodes) > 0:
        exit_coords = coords[exit_nodes]

        # Check if exit faces are on the downstream (right) side
        x_max = coords[:, 0].max()
        x_threshold = x_max - (coords[:, 0].max() - coords[:, 0].min()) * 0.2  # 20% from right
        downstream_exit = np.sum(exit_coords[:, 0] >= x_threshold)

        print(f"Exit face nodes on downstream side: {downstream_exit}/{len(exit_nodes)}")

        if downstream_exit == 0:
            print("⚠️  WARNING: No exit face nodes on downstream (right) side!")

    # Create a visualization of boundary conditions
    plt.figure(figsize=(12, 6))

    # Plot all nodes
    plt.scatter(coords[:, 0], coords[:, 1], c='lightgray', s=5, alpha=0.5, label='No-flow')

    # Overlay boundary condition nodes
    if len(fixed_nodes) > 0:
        fixed_coords = coords[fixed_nodes]
        plt.scatter(fixed_coords[:, 0], fixed_coords[:, 1], c='red', s=20, label='Fixed Head', zorder=5)

    if len(exit_nodes) > 0:
        exit_coords = coords[exit_nodes]
        plt.scatter(exit_coords[:, 0], exit_coords[:, 1], c='blue', s=20, label='Exit Face', zorder=5)

    # Add annotations for head values
    if len(fixed_nodes) > 0:
        for i in range(min(3, len(fixed_nodes))):  # Annotate first few
            node = fixed_nodes[i]
            x, y = coords[node]
            plt.annotate(f'h={fx[node]:.0f}', (x, y), xytext=(5, 5),
                         textcoords='offset points', fontsize=8, color='red')

    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('s2unc Boundary Conditions')
    plt.legend()
    plt.axis('equal')
    plt.grid(True, alpha=0.3)
    plt.show()

    return model_data


# Add this to debug what happens during the solution process
def debug_unsaturated_iteration():
    """
    Debug what happens during the first few iterations of the unsaturated solver.
    """
    import numpy as np
    from seep2d import read_seep2d_input

    model_data = read_seep2d_input("samples/s2unc/s2unc.s2d")

    coords = model_data['coords']
    nbc = model_data['nbc']
    fx = model_data['fx']

    print("\n=== ITERATION DEBUG ===")

    # Show what boundary conditions are actually being applied
    print("Initial boundary conditions that will be applied:")

    fixed_nodes = np.where(nbc == 1)[0]
    exit_nodes = np.where(nbc == 2)[0]

    print(f"Fixed head BCs ({len(fixed_nodes)} nodes):")
    for node in fixed_nodes[:5]:  # Show first 5
        print(f"  Node {node}: h = {fx[node]:.1f} at ({coords[node, 0]:.1f}, {coords[node, 1]:.1f})")

    print(f"Exit face BCs ({len(exit_nodes)} nodes):")
    for node in exit_nodes[:5]:  # Show first 5
        elev = coords[node, 1]
        print(f"  Node {node}: h = {elev:.1f} (elevation) at ({coords[node, 0]:.1f}, {coords[node, 1]:.1f})")

    # Check if there are any nodes with very high fixed heads in wrong locations
    if len(fixed_nodes) > 0:
        max_head = fx[fixed_nodes].max()
        max_head_node = fixed_nodes[np.argmax(fx[fixed_nodes])]
        max_head_coord = coords[max_head_node]

        print(
            f"\nHighest fixed head: {max_head:.1f} at node {max_head_node} ({max_head_coord[0]:.1f}, {max_head_coord[1]:.1f})")

        # Check if this high head is in a problematic location
        x_center = (coords[:, 0].min() + coords[:, 0].max()) / 2
        if max_head_coord[0] > x_center:
            print("⚠️  WARNING: Highest fixed head is on the RIGHT side of the domain!")
            print("   This will cause flow to appear to originate from the right.")


# Run the debug functions
if __name__ == "__main__":
    model_data = debug_s2unc_boundary_conditions()
    debug_unsaturated_iteration()