import pytest
import numpy as np
import os
import sys

# Add the src directory to the Python module search path
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../src/ShotPeenWithML'))
sys.path.append(src_path)

# Import functions from data_viz.py
from data_viz import load_data, compute_deformed_mesh


# Setup: Create dummy data for testing
@pytest.fixture(scope="module")
def create_dummy_simulation_folder(tmpdir_factory):
    folder = tmpdir_factory.mktemp("simulation")
    np.save(os.path.join(folder, 'checkerboard.npy'), np.random.rand(10, 10))
    np.save(os.path.join(folder, 'node_coords.npy'), np.random.rand(10, 3))
    np.save(os.path.join(folder, 'node_labels.npy'), np.arange(10))
    np.save(os.path.join(folder, 'displacements.npy'), np.random.rand(10, 3))
    np.save(os.path.join(folder, 'disp_node_labels.npy'), np.arange(10))
    np.save(os.path.join(folder, 'element_connectivity.npy'), np.array([[0, 1, 2], [3, 4, 5]]))
    return folder

# Smoke Test
def test_load_data_smoke(create_dummy_simulation_folder):
    file_path = os.path.join(create_dummy_simulation_folder, 'checkerboard.npy')
    data = load_data(file_path)
    assert data is not None, "Smoke test failed: load_data returned None."

# One-shot Tests
def test_compute_deformed_mesh_correct_data(create_dummy_simulation_folder):
    folder = create_dummy_simulation_folder
    node_coords, deformed_coords, element_nodes = compute_deformed_mesh(folder)
    assert node_coords is not None, "One-shot test failed: Node coordinates are None."
    assert deformed_coords is not None, "One-shot test failed: Deformed coordinates are None."
    assert element_nodes is not None, "One-shot test failed: Element nodes are None."

def test_load_data_invalid_file():
    data = load_data("non_existent_file.npy", "nonexistent")
    assert data is None, "One-shot test failed: Invalid file should return None."

# Edge Case Tests
def test_compute_deformed_mesh_missing_file(create_dummy_simulation_folder):
    folder = create_dummy_simulation_folder
    os.remove(os.path.join(folder, 'node_coords.npy'))
    node_coords, deformed_coords, element_nodes = compute_deformed_mesh(folder)
    assert node_coords is None, "Edge test failed: Should return None if a required file is missing."

def test_load_data_empty_file(tmpdir):
    empty_file = tmpdir.join("empty.npy")
    empty_file.write("")
    data = load_data(str(empty_file), "empty file")
    assert data is None, "Test failed: load_data should return None for an empty file."

# Running Tests
if __name__ == "__main__":
    pytest.main()
