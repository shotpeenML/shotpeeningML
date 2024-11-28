# Import necessary modules
import os
import random
from abaqus import *
from abaqusConstants import *
from caeModules import *
from driverUtils import executeOnCaeStartup
import visualization
import numpy as np
import shutil
import sys


'''
Generates a dataset of random checkerboard patterns with expansion coefficients assigned to alternating squares in a 5x5 checkerboard 
grid for material simulations. The script iterates through a predefined number of simulations, creating a unique checkerboard pattern for each. 
Four different expansion coefficients (0.005, 0.01, 0.015, 0.02) are randomly assigned to the squares of the checkerboard grid. The script prepares 
the geometry, assigns materials with different thermal expansion properties, and saves the resulting patterns as numerical arrays for reference. 
For each simulation, a new model is initialized, and relevant boundary conditions and temperature fields are applied. The simulation results are 
saved, including displacement and stress data, as well as the material distribution for later analysis. The primary goal is to generate a 
comprehensive dataset to study the effects of material expansion on the checkerboard pattern's mechanical response under uniform thermal conditions.
'''

# Initialize the CAE startup sequence
executeOnCaeStartup()

# Number of simulations
num_simulations = 2000  # Adjust as needed

# Directory to save all simulations
main_directory = r'U:\Shot Peening\Checkerboard\Dataset1_Random_Board'  # Change this to your desired path

# Create the main directory if it doesn't exist
if not os.path.exists(main_directory):
    os.makedirs(main_directory)

# Loop over the number of simulations
for sample_idx in range(num_simulations):
    print("\nStarting simulation {}/{}\n".format(sample_idx + 1, num_simulations))
    
    # Create a unique folder for this simulation
    simulation_folder = os.path.join(main_directory, 'Simulation_{}'.format(sample_idx))
    if not os.path.exists(simulation_folder):
        os.makedirs(simulation_folder)
    
    # Create a new model database for each simulation
    Mdb()  # Clear existing models
    mdb.Model(name='Script', modelType=STANDARD_EXPLICIT)
    
    # Create a new sketch for the part
    s = mdb.models['Script'].ConstrainedSketch(name='__profile__', sheetSize=200.0)
    s.setPrimaryObject(option=STANDALONE)
    
    # Draw a rectangle with opposite corners at (0,0) and (1,1)
    square_length = 1  # meters
    s.rectangle(point1=(0.0, 0.0), point2=(square_length, square_length))
    
    # Create a new part named 'Sheet' with a 3D planar deformable body
    p = mdb.models['Script'].Part(name='Sheet', dimensionality=THREE_D, type=DEFORMABLE_BODY)
    
    # Create a shell from the sketch
    p.BaseShell(sketch=s)
    
    # Unset the sketch as the primary object
    s.unsetPrimaryObject()
    
    # Delete the sketch as it is no longer needed
    del mdb.models['Script'].sketches['__profile__']
    
    # Create a set called 'Sheet-all' including all of the geometry of the part 'Sheet'
    p.Set(name='Sheet-all', faces=p.faces[:])
    
    # Create top and bottom surfaces for Sheet
    sheet_faces = p.faces[:]
    p.Surface(name='Sheet-top', side1Faces=sheet_faces)
    p.Surface(name='Sheet-bottom', side2Faces=sheet_faces)
    
    # Create Copy of Sheet
    p_peen = mdb.models['Script'].Part(name='Peen', objectToCopy=mdb.models['Script'].parts['Sheet'])
    
    # Create Material(s) and Section(s)
    # For Peen
    # Create a list of tuples of expansion coefficients
    expansion_data = [
        (0.005, '005'),
        (0.01, '01'),
        (0.015, '015'),
        (0.02, '02')
    ]
    
    # Loop through the expansion coefficients and create materials
    for coef, name_suffix in expansion_data:
        material_name = 'Aluminum-' + name_suffix
        section_name = 'Peen-' + name_suffix
    
        # Create the material with the given name
        mdb.models['Script'].Material(name=material_name)
    
        # Set the elastic properties for the material
        mdb.models['Script'].materials[material_name].Elastic(table=((68000000000.0, 0.36), ))
    
        # Set the expansion coefficient for the material
        mdb.models['Script'].materials[material_name].Expansion(table=((coef, ), ))
    
        # Create the section with the given name
        mdb.models['Script'].HomogeneousShellSection(name=section_name, preIntegrate=OFF,
                                                     material=material_name, thicknessType=UNIFORM, thickness=0.0002,
                                                     thicknessField='', nodalThicknessField='', idealization=NO_IDEALIZATION,
                                                     poissonDefinition=DEFAULT, thicknessModulus=None, temperature=GRADIENT,
                                                     useDensity=OFF, integrationRule=SIMPSON, numIntPts=5)
    
    # For Sheet
    # Create the material with the given name
    mdb.models['Script'].Material(name='Aluminum-Sheet')
    mdb.models['Script'].materials['Aluminum-Sheet'].Elastic(table=((68000000000.0, 0.36), ))
    
    # Create the section with the given name
    mdb.models['Script'].HomogeneousShellSection(name='Sheet', preIntegrate=OFF,
                                                 material='Aluminum-Sheet', thicknessType=UNIFORM, thickness=0.005,
                                                 thicknessField='', nodalThicknessField='', idealization=NO_IDEALIZATION,
                                                 poissonDefinition=DEFAULT, thicknessModulus=None, temperature=GRADIENT,
                                                 useDensity=OFF, integrationRule=SIMPSON, numIntPts=5)
    
    
    # Assign section 'Sheet' to the set 'Sheet-all'
    p_sheet = mdb.models['Script'].parts['Sheet']
    region = p_sheet.sets['Sheet-all']
    p_sheet.SectionAssignment(region=region, sectionName='Sheet',
        offsetType=SINGLE_VALUE, offset=0.52, offsetField='',
        thicknessAssignment=FROM_SECTION)
    
    
    # Create checkerboard pattern on the Peen part
    part = mdb.models['Script'].parts['Peen']
    checker_size = 0.2  # make sure this does not return float pattern_number
    pattern_number = int(square_length / checker_size)
    
    # Create the checkerboard sketch
    sketch = mdb.models['Script'].ConstrainedSketch(name='__checkerboard__', sheetSize=1.0)
    for i in range(pattern_number):
        for j in range(pattern_number):
            if (i + j) % 2 == 0:  # Checkerboard pattern
                x1 = i * checker_size
                y1 = j * checker_size
                x2 = x1 + checker_size
                y2 = y1 + checker_size
                sketch.rectangle(point1=(x1, y1), point2=(x2, y2))
    
    # Partition the face using the checkerboard sketch
    faces = part.faces.findAt(((0.05, 0.05, 0.0),))
    part.PartitionFaceBySketch(faces=faces, sketch=sketch)
    
    # Create sets for the checkerboard pattern
    sets = []
    for i in range(pattern_number):
        for j in range(pattern_number):
            x = i * checker_size + checker_size / 2
            y = j * checker_size + checker_size / 2
            face = part.faces.findAt(((x, y, 0.0),))
            set_name = 'Peen-set-{}_{}'.format(i, j)
            part.Set(name=set_name, faces=face)
            sets.append(set_name)
    
    # Rename top and bottom surfaces and set for Peen
    part.surfaces.changeKey(fromName='Sheet-bottom', toName='Peen-bottom')
    part.surfaces.changeKey(fromName='Sheet-top', toName='Peen-top')
    part.sets.changeKey(fromName='Sheet-all', toName='Peen-all')
    
    # Randomly assign materials to the sets and record the checkerboard pattern
    sections = ['Peen-005', 'Peen-01', 'Peen-015', 'Peen-02']
    expansion_coefficients = {
        'Peen-005': 0.005,
        'Peen-01': 0.01,
        'Peen-015': 0.015,
        'Peen-02': 0.02
    }
    
    random.shuffle(sets)  # Shuffle the sets for random assignment
    
    # Initialize a 2D array to store the expansion coefficients
    checkerboard_array = np.zeros((pattern_number, pattern_number))
    
    for idx, set_name in enumerate(sets):
        section_name = sections[idx % len(sections)]  # Cycle through the sections
        region = part.sets[set_name]
        part.SectionAssignment(region=region, sectionName=section_name, offset=-5.0,
                               offsetType=MIDDLE_SURFACE, offsetField='',
                               thicknessAssignment=FROM_SECTION)
        # Extract indices from set_name
        indices = set_name.split('-')[-1].split('_')
        i_idx = int(indices[0])
        j_idx = int(indices[1])
        # Assign the expansion coefficient to the array
        checkerboard_array[i_idx, j_idx] = expansion_coefficients[section_name]
    
    # Save the checkerboard pattern as .npy and .csv in the simulation folder
    np.save(os.path.join(simulation_folder, 'checkerboard.npy'), checkerboard_array)
    np.savetxt(os.path.join(simulation_folder, 'checkerboard.csv'), checkerboard_array, delimiter=',')
    
    # Create Instances
    a = mdb.models['Script'].rootAssembly
    
    # Instance for Sheet
    a.Instance(name='Sheet-1', part=p_sheet, dependent=ON)
    
    # Instance for Peen
    a.Instance(name='Peen-1', part=part, dependent=ON)
    
    # Create tie constraint between Sheet-top and Peen-bottom
    region1 = a.instances['Sheet-1'].surfaces['Sheet-top']
    region2 = a.instances['Peen-1'].surfaces['Peen-bottom']
    mdb.models['Script'].Tie(name='Constraint-1', main=region1, secondary=region2, positionToleranceMethod=COMPUTED, adjust=ON, tieRotations=ON, thickness=ON)
    
    ## Constraints / Boundary Conditions
    # Constrain the bottom left corner of the Sheet part
    bottom_left_vertex = a.instances['Sheet-1'].vertices.findAt(((0.0, 0.0, 0.0),))
    mdb.models['Script'].DisplacementBC(name='BC-Fixed', createStepName='Initial', 
        region=regionToolset.Region(vertices=bottom_left_vertex), u1=SET, u2=SET, u3=SET)
    
    # Roll constrain the other two corners of the Sheet part
    top_left_vertex = a.instances['Sheet-1'].vertices.findAt(((0.0, square_length, 0.0),))
    bottom_right_vertex = a.instances['Sheet-1'].vertices.findAt(((square_length, 0.0, 0.0),))
    
    # Top left corner roll constraint (only u1 and ur3 free)
    mdb.models['Script'].DisplacementBC(name='BC-Roll-TopLeft', createStepName='Initial', 
        region=regionToolset.Region(vertices=top_left_vertex), u1=SET, u2=UNSET, u3=UNSET)
    
    # Bottom right corner roll constraint (only u2 and ur3 free)
    mdb.models['Script'].DisplacementBC(name='BC-Roll-BottomRight', createStepName='Initial', 
        region=regionToolset.Region(vertices=bottom_right_vertex), u1=UNSET, u2=SET, ur3=UNSET)
    
    # Create another step
    mdb.models['Script'].StaticStep(name='Step-1', previous='Initial')
    
    # Apply a predefined temperature field on Peen-all
    region = a.instances['Peen-1'].sets['Peen-all']
    mdb.models['Script'].Temperature(name='Predefined Field-1', 
        createStepName='Step-1', region=region, distributionType=UNIFORM, 
        crossSectionDistribution=CONSTANT_THROUGH_THICKNESS, magnitudes=(1.0, ))
    
    # Generate mesh for Peen part
    part.seedPart(size=0.02, deviationFactor=0.1, minSizeFactor=0.1)
    part.generateMesh()
    
    # Generate mesh for Sheet part
    p_sheet.seedPart(size=0.02, deviationFactor=0.1, minSizeFactor=0.1)
    p_sheet.generateMesh()
    
    # Create unique job name
    job_name = 'Checkerboard_{}'.format(sample_idx)
    
    # Save the CAE file in the simulation folder
    cae_file_path = os.path.join(simulation_folder, '{}.cae'.format(job_name))
    mdb.saveAs(pathName=cae_file_path)
    
    # Create job
    mdb.Job(name=job_name, model='Script', description='', type=ANALYSIS,
            atTime=None, waitMinutes=0, waitHours=0, queue=None, memory=90,
            memoryUnits=PERCENTAGE, getMemoryFromAnalysis=True,
            explicitPrecision=SINGLE, nodalOutputPrecision=SINGLE, echoPrint=OFF,
            modelPrint=OFF, contactPrint=OFF, historyPrint=OFF,
            userSubroutine='', scratch='', multiprocessingMode=DEFAULT,
            numCpus=1, numGPUs=0)
    
    # Submit the job
    mdb.jobs[job_name].submit()
    mdb.jobs[job_name].waitForCompletion()
    
    ## Processing Results

    # Open ODB file
    odb_path = os.path.join(r'C:\temp', '{}.odb'.format(job_name))
    odb = visualization.openOdb(path=odb_path)

    # Access the desired step
    step = odb.steps['Step-1']

    # Access the last frame (final results)
    last_frame = step.frames[-1]

    # FIELD Output
    disp_field = last_frame.fieldOutputs['U']  # Get displacement field output
    stress_field = last_frame.fieldOutputs['S']  # Get stress field output

    ## NODAL Output

    # Nodal Displacement
    # Initialize lists to store data
    disp_node_labels = []
    displacements = []

    for value in disp_field.values:
        disp_node_labels.append(value.nodeLabel)
        # value.data returns a tuple (U1, U2, U3)
        displacements.append(value.data)

    # Element Stresses
    # Initialize lists to store data
    stress_element_labels = []
    stresses = []

    for value in stress_field.values:
        stress_element_labels.append(value.elementLabel)
        # value.data returns stress components, e.g., (S11, S22, S33, S12, S13, S23)
        stresses.append(value.data)

    # Convert lists to NumPy arrays
    displacements = np.array(displacements)  # Shape: (num_nodes, 3)
    stresses = np.array(stresses)            # Shape: (num_elements, 6)
    disp_node_labels = np.array(disp_node_labels)
    stress_element_labels = np.array(stress_element_labels)

    # Save displacements and stresses with node/element labels
    np.save(os.path.join(simulation_folder, 'displacements.npy'), displacements)
    np.save(os.path.join(simulation_folder, 'disp_node_labels.npy'), disp_node_labels)
    np.savetxt(os.path.join(simulation_folder, 'displacements.csv'), displacements, delimiter=',')
    np.savetxt(os.path.join(simulation_folder, 'disp_node_labels.csv'), disp_node_labels, fmt='%d', delimiter=',')

    np.save(os.path.join(simulation_folder, 'stresses.npy'), stresses)
    np.save(os.path.join(simulation_folder, 'stress_element_labels.npy'), stress_element_labels)
    np.savetxt(os.path.join(simulation_folder, 'stresses.csv'), stresses, delimiter=',')
    np.savetxt(os.path.join(simulation_folder, 'stress_element_labels.csv'), stress_element_labels, fmt='%d', delimiter=',')

    ## Extract Node Coordinates and Element Connectivity

    # Access the assembly
    assembly = odb.rootAssembly

    # Initialize lists to store node labels and coordinates
    node_labels = []
    node_coords = []

    # Initialize lists to store element labels and connectivity
    element_labels = []
    element_connectivity = []

    # Loop through instances in the assembly
    for instance in assembly.instances.values():
        # Nodes
        for node in instance.nodes:
            node_labels.append(node.label)
            node_coords.append(node.coordinates)
        # Elements
        for element in instance.elements:
            element_labels.append(element.label)
            element_connectivity.append(element.connectivity)

    # Convert to NumPy arrays
    node_labels = np.array(node_labels)  # Shape: (num_nodes,)
    node_coords = np.array(node_coords)  # Shape: (num_nodes, 3)
    element_labels = np.array(element_labels)  # Shape: (num_elements,)
    element_connectivity = np.array(element_connectivity)  # Shape: (num_elements, nodes_per_element)

    # Save node labels and coordinates
    np.save(os.path.join(simulation_folder, 'node_labels.npy'), node_labels)
    np.save(os.path.join(simulation_folder, 'node_coords.npy'), node_coords)
    np.savetxt(os.path.join(simulation_folder, 'node_labels.csv'), node_labels, fmt='%d', delimiter=',')
    np.savetxt(os.path.join(simulation_folder, 'node_coords.csv'), node_coords, delimiter=',')

    # Save element labels and connectivity
    np.save(os.path.join(simulation_folder, 'element_labels.npy'), element_labels)
    np.save(os.path.join(simulation_folder, 'element_connectivity.npy'), element_connectivity)
    np.savetxt(os.path.join(simulation_folder, 'element_labels.csv'), element_labels, fmt='%d', delimiter=',')
    np.savetxt(os.path.join(simulation_folder, 'element_connectivity.csv'), element_connectivity, fmt='%d', delimiter=',')

    # Close the ODB file
    odb.close()

    # Move the ODB file to the simulation folder
    odb_destination = os.path.join(simulation_folder, '{}.odb'.format(job_name))
    if os.path.exists(odb_path):
        shutil.move(odb_path, odb_destination)

    # (Rest of your script)

    
    # Optionally, delete the job from the job manager to free up resources
    del mdb.jobs[job_name]
    
    # Clear the model database for the next simulation
    mdb.close()

print("\nAll simulations completed.\n")
