"""
Tests for the data visualization module in the peen-ml project.
"""

# Import functions from data_viz.py
import os
import sys
import numpy as np
import pytest
# Add the src directory to the Python module search path
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../src/peen-ml'))
sys.path.append(src_path)
from data_viz import load_data, compute_deformed_mesh # pylint: disable=wrong-import-position

# Setup: Create dummy data for testing
@pytest.fixture(scope="module")
def create_simulation(tmpdir_factory):
    """Create dummy simulation folder for testing."""
    folder = tmpdir_factory.mktemp("simulation")
    np.save(os.path.join(folder, 'checkerboard.npy'), np.random.rand(10, 10))
    np.save(os.path.join(folder, 'node_coords.npy'), np.random.rand(10, 3))
    np.save(os.path.join(folder, 'node_labels.npy'), np.arange(10))
    np.save(os.path.join(folder, 'displacements.npy'), np.random.rand(10, 3))
    np.save(os.path.join(folder, 'disp_node_labels.npy'), np.arange(10))
    np.save(os.path.join(folder, 'element_connectivity.npy'), np.array([[0, 1, 2], [3, 4, 5]]))
    return folder

# Smoke Test
def test_load_data_smoke(create_simulation):
    """Smoke test to verify load_data runs without errors."""
    file_path = os.path.join(create_simulation, 'checkerboard.npy')
    data = load_data(file_path)
    assert data is not None, "Smoke test failed: load_data returned None."

# One-shot Tests
def test_compute_deformed_mesh_correct_data(create_simulation):
    """Test compute_deformed_mesh with correct data."""
    node_coords, deformed_coords, element_nodes = compute_deformed_mesh(create_simulation)
    assert node_coords is not None, "One-shot test failed: Node coordinates are None."
    assert deformed_coords is not None, "One-shot test failed: Deformed coordinates are None."
    assert element_nodes is not None, "One-shot test failed: Element nodes are None."

def test_load_data_invalid_file():
    """Test load_data with an invalid file."""
    data = load_data("non_existent_file.npy", "nonexistent")
    assert data is None, "One-shot test failed: Invalid file should return None."

# Edge Case Tests
def test_compute_deformed_mesh_missing_file(create_simulation):
    """Test compute_deformed_mesh when a required file is missing."""
    os.remove(os.path.join(create_simulation, 'node_coords.npy'))
    node_coords, _, _ = compute_deformed_mesh(create_simulation)
    assert node_coords is None, "Edge test failed: Should return None if a required file is missing"

def test_load_data_empty_file(tmpdir):
    """Test load_data with an empty file."""
    empty_file = tmpdir.join("empty.npy")
    empty_file.write("")
    data = load_data(str(empty_file), "empty file")
    assert data is None, "Test failed: load_data should return None for an empty file."

# Running Tests
if __name__ == "__main__":
    pytest.main()
