import pytest
import numpy as np
import os
import torch
import sys

# Add the src directory to the Python module search path
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../src/ShotPeenWithML'))
sys.path.append(src_path)
from model import (
    load_all_npy_files,
    CheckerboardDataset,
    NormalizedDataset,
    ChannelAttention,
    SpatialAttention,
    DisplacementPredictor,
)

# Smoke Test
def test_load_all_npy_files_smoke():
    """Smoke test to check if the function runs without errors."""
    base_folder = "./test_simulations"
    os.makedirs(base_folder, exist_ok=True)
    for i in range(2):  # Create dummy simulations
        sim_folder = os.path.join(base_folder, f"Simulation_{i}")
        os.makedirs(sim_folder, exist_ok=True)
        np.save(os.path.join(sim_folder, "checkerboard.npy"), np.random.rand(10, 10))
        np.save(os.path.join(sim_folder, "displacements.npy"), np.random.rand(10, 3))
    
    result = load_all_npy_files(base_folder, 2)
    assert "checkerboard" in result
    assert "displacements" in result
    assert result["checkerboard"].shape[0] == 2  # 2 simulations
    assert result["checkerboard"].shape[1:] == (10, 10)

# One-Shot Tests
def test_checkerboard_dataset_one_shot():
    """Check basic functionality of CheckerboardDataset."""
    checkerboards = np.random.rand(10, 10, 10)
    displacements = np.random.rand(10, 5, 3)
    dataset = CheckerboardDataset(checkerboards, displacements)
    assert len(dataset) == 10
    checkerboard, displacement = dataset[0]
    assert checkerboard.shape == (1, 10, 10)  # Single channel
    assert displacement.shape == (5, 3)

def test_channel_attention_one_shot():
    """Check if ChannelAttention layer can forward a tensor."""
    layer = ChannelAttention(channels=32)
    input_tensor = torch.rand(4, 32, 10, 10)
    output_tensor = layer(input_tensor)
    assert output_tensor.shape == input_tensor.shape  # Should not change shape

# Edge Tests
def test_load_all_npy_files_edge_missing_file():
    """Test load_all_npy_files with a missing file."""
    base_folder = "./test_simulations_edge"
    os.makedirs(base_folder, exist_ok=True)
    for i in range(1):  # Create a single simulation folder
        sim_folder = os.path.join(base_folder, f"Simulation_{i}")
        os.makedirs(sim_folder, exist_ok=True)
        np.save(os.path.join(sim_folder, "checkerboard.npy"), np.random.rand(10, 10))
    with pytest.raises(FileNotFoundError):
        load_all_npy_files(base_folder, 1, skip_missing=False)



if __name__ == "__main__":
    pytest.main()
