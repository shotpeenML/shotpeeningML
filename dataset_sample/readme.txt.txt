# Dataset 1

The script for dataset 1 generates a dataset of random checkerboard patterns with expansion coefficients assigned to alternating squares in a 5x5 checkerboard grid for material simulations. The script iterates through a predefined number of simulations, creating a unique checkerboard pattern for each. Four different expansion coefficients (0.005, 0.01, 0.015, 0.02) are randomly assigned to the squares of the checkerboard grid. The script prepares the geometry, assigns materials with different thermal expansion properties, and saves the resulting patterns as numerical arrays for reference. For each simulation, a new model is initialized, and relevant boundary conditions and temperature fields are applied. The simulation results are saved, including displacement and stress data, as well as the material distribution for later analysis. The primary goal is to generate a comprehensive dataset to study the effects of material expansion on the checkerboard pattern's mechanical response under uniform thermal conditions.

# Dataset 2

The script for dataset 2 is similar with that of dataset 1 however, it is designed to systematically apply each of the four expansion coefficients (0.005, 0.01, 0.015, 0.02) to one square at a time in a 5x5 checkerboard grid, resulting in 100 simulations (4 coefficients * 25 squares). Unlike the original script, where coefficients are randomly distributed across the grid, this version focuses on isolating the effect of a single coefficient applied to a specific square while all other squares have zero expansion. For each simulation, the script initializes the model, assigns the "zero expansion" material to all squares, and applies the target coefficient to one square. It generates a checkerboard pattern reflecting this setup, saves the coefficient distribution as numerical arrays, and runs the simulation. Results, including displacement and stress data, are saved for each case. The primary goal of this script is to build a dataset where the isolated effect of individual expansion coefficients can be analyzed systematically, offering a controlled environment for studying localized material behavior under thermal conditions.

# Simulation Data Files

Each folder in both datasets contains data generated from a eigenstrain model shot peen FEA simulation. Below is a description of each file and its purpose. 
Each file is saved in npy and csv. Npy is easier to import and manage in python, but csv is easier to visualize outside of python.

---

## Abaqus Model and Output Files

- **Checkerboard.cae**
    - **Description**: This file is the Abaqus CAE model database that contains the complete setup of the simulation, including the geometry, materials, boundary conditions, and mesh.
    - **Purpose**: The `.cae` file allows the user to reopen the simulation model in Abaqus/CAE for further analysis, modifications, or to rerun the simulation with different parameters.
    - **Usage**: Load this file in Abaqus/CAE to access and edit the simulation setup.

- **Checkerboard.odb**
    - **Description**: This is the Abaqus Output Database (ODB) file that stores the simulation results, including field outputs such as displacements, stresses, and any other specified output variables.
    - **Purpose**: The `.odb` file contains the computed results from the simulation, which can be viewed, post-processed, and analyzed within Abaqus/CAE. It is also the source of data for extracting displacement, stress, and other fields for further analysis or visualization outside Abaqus.
    - **Usage**: Open this file in Abaqus/CAE to view the simulation results or use Python scripts to extract specific data fields.

---

## Checkerboard Pattern Files

- **checkerboard.npy** and **checkerboard.csv**
    - **Description**: These files store the checkerboard pattern of thermal expansion coefficients assigned to different regions on the plate.
    - **Purpose**: This pattern is used as input for the simulation to analyze how different coefficients influence mechanical behavior.

---

## Mesh and Geometry Data

- **node_labels.npy** and **node_labels.csv**
    - **Description**: These files contain the labels (unique identifiers) for each node in the mesh.
    - **Purpose**: Node labels are essential for aligning displacement data and reconstructing the mesh structure.

- **node_coords.npy** and **node_coords.csv**
    - **Description**: These files store the original (undeformed) coordinates of each node in the mesh.
    - **Purpose**: Node coordinates are required to visualize the mesh and compute deformed positions after displacement is applied.

- **element_labels.npy** and **element_labels.csv**
    - **Description**: These files contain the labels (unique identifiers) for each element in the mesh.
    - **Purpose**: Element labels are used to align stress data with specific elements for accurate visualization.

- **element_connectivity.npy** and **element_connectivity.csv**
    - **Description**: These files contain the connectivity information for each element, specifying which nodes form each element.
    - **Purpose**: Element connectivity defines the mesh structure, allowing visualization of the element shapes and reconstruction of the mesh.

---

## Simulation Results

### Displacement Data

- **displacements.npy** and **displacements.csv**
    - **Description**: These files contain the displacement vectors `(U1, U2, U3)` for each node in the mesh, representing the node’s deformation from its original position.
    - **Purpose**: Displacement data is used to compute the deformed mesh for visualization and to analyze the deformation pattern caused by the thermal expansion.

- **disp_node_labels.npy** and **disp_node_labels.csv**
    - **Description**: These files provide the labels for nodes associated with each displacement vector in `displacements.npy`.
    - **Purpose**: Node labels are required to correctly align displacement data with corresponding nodes in the mesh.

### Stress Data

- **stresses.npy** and **stresses.csv**
    - **Description**: These files contain the stress components `(S11, S22, S33, S12, S13, S23)` for each element, which measure the internal forces within each element.
    - **Purpose**: Stress data helps evaluate the internal response of the material to the applied thermal expansion and loading conditions.

- **stress_element_labels.npy** and **stress_element_labels.csv**
    - **Description**: These files contain the labels for each element associated with the stress values in `stresses.npy`.
    - **Purpose**: Element labels are needed to correctly align stress data with the corresponding elements in the mesh for visualization.

---

## Visualization and Analysis (visualized used data_viz.py)

These files are used to visualize and analyze the simulation results:
- **Checkerboard Pattern**: Visualized as an image to see the distribution of expansion coefficients.
- **Stress Field**: Displayed on the deformed mesh to illustrate the stress distribution.
- **Displacement Magnitude**: Plotted on the deformed mesh to show the extent of deformation across the plate.

---
