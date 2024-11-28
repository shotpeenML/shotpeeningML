#%%
import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.collections import PolyCollection
import matplotlib.collections as mc

# Path to simulation folder
simulation_folder = r'\\udrive.uw.edu\onestr\Shot Peening\Checkerboard\TestBatch2_with node info\Simulation_0'  # Change to the desired simulation index

# Deformation Scale (play around with it)
scale_factor = 1 


### 1. Visualize the Checkerboard Pattern ###

# Load checkerboard pattern
checkerboard = np.load(os.path.join(simulation_folder, 'checkerboard.npy'))

# Plot the checkerboard pattern
plt.figure(figsize=(6, 6))
plt.imshow(checkerboard, cmap='viridis', origin='lower', extent=[0, 1, 0, 1])
plt.colorbar(label='Expansion Coefficient')
plt.title('Checkerboard Pattern of Expansion Coefficients')
plt.xlabel('X Position (m)')
plt.ylabel('Y Position (m)')
plt.grid(False)
plt.show()

### 2. Visualize the Undeformed and Deformed Mesh ###

# Load node coordinates and labels
node_coords = np.load(os.path.join(simulation_folder, 'node_coords.npy'))
node_labels = np.load(os.path.join(simulation_folder, 'node_labels.npy'))

# Load displacements and corresponding node labels
displacements = np.load(os.path.join(simulation_folder, 'displacements.npy'))
disp_node_labels = np.load(os.path.join(simulation_folder, 'disp_node_labels.npy'))

# Create a mapping from node label to index in node_coords
node_label_to_index = {label: idx for idx, label in enumerate(node_labels)}

# Initialize an array to store displacements aligned with node_coords
aligned_displacements = np.zeros_like(node_coords)

for idx, label in enumerate(disp_node_labels):
    node_idx = node_label_to_index[label]
    aligned_displacements[node_idx] = displacements[idx]

# Compute deformed node positions
deformed_coords = node_coords + scale_factor * aligned_displacements

# Load element connectivity
element_connectivity = np.load(os.path.join(simulation_folder, 'element_connectivity.npy'))

# Convert node labels in connectivity to indices
element_nodes = []

for elem in element_connectivity:
    indices = [node_label_to_index[label] for label in elem]
    element_nodes.append(indices)

# Creating mesh lines
def create_mesh_lines(node_coords, element_nodes):
    lines = []
    for elem in element_nodes:
        coords = node_coords[elem]
        for i in range(len(coords)):
            start = coords[i][:2]
            end = coords[(i + 1) % len(coords)][:2]
            lines.append([start, end])
    return lines

# Create lines for undeformed and deformed mesh
undeformed_lines = create_mesh_lines(node_coords, element_nodes)
deformed_lines = create_mesh_lines(deformed_coords, element_nodes)

'''
# Plot undeformed and deformed mesh
fig, ax = plt.subplots(figsize=(10, 10))
lc_undeformed = mc.LineCollection(undeformed_lines, colors='gray', linewidths=0.5)
ax.add_collection(lc_undeformed)
lc_deformed = mc.LineCollection(deformed_lines, colors='blue', linewidths=0.5)
ax.add_collection(lc_deformed)
ax.set_xlim([0, 1])
ax.set_ylim([0, 1])
ax.set_aspect('equal')
ax.set_title('Undeformed (gray) and Deformed (blue) Mesh')
ax.set_xlabel('X Position (m)')
ax.set_ylabel('Y Position (m)')
plt.show()
'''

### 3. Visualize the Stress Field on Deformed Mesh ###

# Load stress data and labels
stresses = np.load(os.path.join(simulation_folder, 'stresses.npy'))
stress_element_labels = np.load(os.path.join(simulation_folder, 'stress_element_labels.npy'))

# Compute von Mises stress
S11 = stresses[:, 0]
S22 = stresses[:, 1]
S12 = stresses[:, 3]
von_mises_stress = np.sqrt(S11**2 - S11*S22 + S22**2 + 3*S12**2)

# Map stress data to elements
element_label_to_stress = {label: vm for label, vm in zip(stress_element_labels, von_mises_stress)}

# Prepare data for plotting
element_polygons = []
element_stress_values = []

for elem_indices, elem_label in zip(element_nodes, stress_element_labels):
    coords = deformed_coords[elem_indices][:, :2]
    element_polygons.append(coords)
    stress_value = element_label_to_stress[elem_label]
    element_stress_values.append(stress_value)

# Create PolyCollection
collection = PolyCollection(element_polygons, array=element_stress_values, cmap='jet', edgecolors='k', linewidths=0.5)

# Plot stress field on deformed mesh
fig, ax = plt.subplots(figsize=(10, 10))
ax.add_collection(collection)
ax.autoscale_view()
ax.set_aspect('equal')
plt.colorbar(collection, ax=ax, label='Von Mises Stress (Pa)')
ax.set_title('Stress Field on Deformed Mesh')
ax.set_xlabel('X Position (m)')
ax.set_ylabel('Y Position (m)')
plt.show()

### 4. Visualize the Deformation Magnitude on Deformed Mesh ###

# Compute deformation magnitude at each node
deformation_magnitude = np.linalg.norm(aligned_displacements, axis=1)

# Normalize deformation magnitude for color mapping (optional)
# deformation_magnitude_normalized = (deformation_magnitude - deformation_magnitude.min()) / (deformation_magnitude.max() - deformation_magnitude.min())

# Prepare data for plotting
# Since deformation is a nodal value, we need to map it to elements for plotting
element_deformation_values = []

for elem_indices in element_nodes:
    # Average deformation magnitude of the nodes in the element
    avg_deformation = deformation_magnitude[elem_indices].mean()
    element_deformation_values.append(avg_deformation)

# Create PolyCollection for deformation
collection_def = PolyCollection(element_polygons, array=element_deformation_values, cmap='plasma', edgecolors='k', linewidths=0.5)

# Plot deformation magnitude on deformed mesh
fig, ax = plt.subplots(figsize=(10, 10))
ax.add_collection(collection_def)
ax.autoscale_view()
ax.set_aspect('equal')
plt.colorbar(collection_def, ax=ax, label='Deformation Magnitude (m)')
ax.set_title('Deformation Magnitude on Deformed Mesh')
ax.set_xlabel('X Position (m)')
ax.set_ylabel('Y Position (m)')
plt.show()
