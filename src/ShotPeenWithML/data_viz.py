import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.collections import PolyCollection
import matplotlib.collections as mc

def load_data(file_path, description=""):
    try:
        data = np.load(file_path)
        return data
    except Exception as e:
        print(f"Error loading {description} data from {file_path}: {e}")
        return None

def visualize_checkerboard(simulation_folder):
    try:
        checkerboard_path = os.path.join(simulation_folder, 'checkerboard.npy')
        checkerboard = load_data(checkerboard_path, "checkerboard")
        if checkerboard is None:
            return

        plt.figure(figsize=(6, 6))
        plt.imshow(checkerboard, cmap='viridis', origin='lower', extent=[0, 1, 0, 1])
        plt.colorbar(label='Expansion Coefficient')
        plt.title('Checkerboard Pattern of Expansion Coefficients')
        plt.xlabel('X Position (m)')
        plt.ylabel('Y Position (m)')
        plt.grid(False)
        plt.show()
    except Exception as e:
        print(f"Error visualizing checkerboard: {e}")

def compute_deformed_mesh(simulation_folder, scale_factor=1):
    try:
        # Load node data
        node_coords = load_data(os.path.join(simulation_folder, 'node_coords.npy'), "node coordinates")
        node_labels = load_data(os.path.join(simulation_folder, 'node_labels.npy'), "node labels")
        displacements = load_data(os.path.join(simulation_folder, 'displacements.npy'), "displacements")
        disp_node_labels = load_data(os.path.join(simulation_folder, 'disp_node_labels.npy'), "displacement node labels")

        # Check for missing data
        if any(obj is None for obj in [node_coords, node_labels, displacements, disp_node_labels]):
            print("One or more required files are missing.")
            return None, None, None

        # Map node labels to indices
        node_label_to_index = {label: idx for idx, label in enumerate(node_labels)}

        # Align displacements with node coordinates
        aligned_displacements = np.zeros_like(node_coords)
        for idx, label in enumerate(disp_node_labels):
            node_idx = node_label_to_index[label]
            aligned_displacements[node_idx] = displacements[idx]

        # Compute deformed coordinates
        deformed_coords = node_coords + scale_factor * aligned_displacements

        # Load element connectivity
        element_connectivity = load_data(os.path.join(simulation_folder, 'element_connectivity.npy'), "element connectivity")
        if element_connectivity is None:
            print("Element connectivity file is missing.")
            return None, None, None

        # Map element connectivity to node indices
        element_nodes = [
            [node_label_to_index[label] for label in elem]
            for elem in element_connectivity
        ]

        return node_coords, deformed_coords, element_nodes
    except Exception as e:
        print(f"Error computing deformed mesh: {e}")
        return None, None, None

def visualize_mesh(node_coords, deformed_coords, element_nodes):
    try:
        def create_mesh_lines(coords, elements):
            lines = []
            for elem in elements:
                element_coords = coords[elem]
                for i in range(len(element_coords)):
                    start = element_coords[i][:2]
                    end = element_coords[(i + 1) % len(element_coords)][:2]
                    lines.append([start, end])
            return lines

        undeformed_lines = create_mesh_lines(node_coords, element_nodes)
        deformed_lines = create_mesh_lines(deformed_coords, element_nodes)

        fig, ax = plt.subplots(figsize=(10, 10))
        lc_undeformed = mc.LineCollection(undeformed_lines, colors='gray', linewidths=0.5)
        lc_deformed = mc.LineCollection(deformed_lines, colors='blue', linewidths=0.5)
        ax.add_collection(lc_undeformed)
        ax.add_collection(lc_deformed)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        ax.set_aspect('equal')
        ax.set_title('Undeformed (gray) and Deformed (blue) Mesh')
        ax.set_xlabel('X Position (m)')
        ax.set_ylabel('Y Position (m)')
        plt.show()
    except Exception as e:
        print(f"Error visualizing mesh: {e}")

def visualize_stress_field(simulation_folder, deformed_coords, element_nodes):
    try:
        stresses = load_data(os.path.join(simulation_folder, 'stresses.npy'), "stresses")
        stress_element_labels = load_data(os.path.join(simulation_folder, 'stress_element_labels.npy'), "stress element labels")
        if stresses is None or stress_element_labels is None:
            return

        S11, S22, S12 = stresses[:, 0], stresses[:, 1], stresses[:, 3]
        von_mises_stress = np.sqrt(S11**2 - S11*S22 + S22**2 + 3*S12**2)
        element_label_to_stress = {label: vm for label, vm in zip(stress_element_labels, von_mises_stress)}

        element_polygons = []
        element_stress_values = []

        for elem_indices, elem_label in zip(element_nodes, stress_element_labels):
            coords = deformed_coords[elem_indices][:, :2]
            element_polygons.append(coords)
            element_stress_values.append(element_label_to_stress[elem_label])

        collection = PolyCollection(element_polygons, array=element_stress_values, cmap='jet', edgecolors='k', linewidths=0.5)
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.add_collection(collection)
        ax.autoscale_view()
        ax.set_aspect('equal')
        plt.colorbar(collection, ax=ax, label='Von Mises Stress (Pa)')
        ax.set_title('Stress Field on Deformed Mesh')
        ax.set_xlabel('X Position (m)')
        ax.set_ylabel('Y Position (m)')
        plt.show()
    except Exception as e:
        print(f"Error visualizing stress field: {e}")

def visualize_deformation(simulation_folder, deformed_coords, element_nodes, aligned_displacements):
    try:
        deformation_magnitude = np.linalg.norm(aligned_displacements, axis=1)

        element_deformation_values = [
            deformation_magnitude[elem_indices].mean()
            for elem_indices in element_nodes
        ]

        element_polygons = [
            deformed_coords[elem_indices][:, :2] for elem_indices in element_nodes
        ]

        collection_def = PolyCollection(element_polygons, array=element_deformation_values, cmap='plasma', edgecolors='k', linewidths=0.5)

        fig, ax = plt.subplots(figsize=(10, 10))
        ax.add_collection(collection_def)
        ax.autoscale_view()
        ax.set_aspect('equal')
        plt.colorbar(collection_def, ax=ax, label='Deformation Magnitude (m)')
        ax.set_title('Deformation Magnitude on Deformed Mesh')
        ax.set_xlabel('X Position (m)')
        ax.set_ylabel('Y Position (m)')
        plt.show()
    except Exception as e:
        print(f"Error visualizing deformation: {e}")

def main():
    # Path to the simulation folder
    simulation_folder = r'\\udrive.uw.edu\onestr\Shot Peening\Checkerboard\Method1\TestBatch\Simulation_0'

    # Deformation Scale (play around with it)
    scale_factor = 1 

    # Step 1: Visualize the Checkerboard Pattern
    print("Visualizing Checkerboard Pattern...")
    visualize_checkerboard(simulation_folder)

    # Step 2: Compute Deformed Mesh
    print("Computing Deformed Mesh...")
    node_coords, deformed_coords, element_nodes = compute_deformed_mesh(simulation_folder, scale_factor)

    # Check if any of the outputs are None
    if any(obj is None for obj in [node_coords, deformed_coords, element_nodes]):
        print("Error in computing deformed mesh. Exiting.")
        return

    # Step 3: Visualize the Undeformed and Deformed Mesh
    print("Visualizing Mesh (Undeformed and Deformed)...")
    visualize_mesh(node_coords, deformed_coords, element_nodes)

    # Step 4: Visualize the Stress Field on the Deformed Mesh
    print("Visualizing Stress Field on Deformed Mesh...")
    visualize_stress_field(simulation_folder, deformed_coords, element_nodes)

    # Step 5: Visualize the Deformation Magnitude on Deformed Mesh
    print("Visualizing Deformation Magnitude on Deformed Mesh...")
    aligned_displacements = deformed_coords - node_coords
    visualize_deformation(simulation_folder, deformed_coords, element_nodes, aligned_displacements)

if __name__ == "__main__":
    main()
