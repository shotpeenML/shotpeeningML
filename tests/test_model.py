"""
Tests for the data visualization module in the peen-ml project.
"""

import os
import sys
import numpy as np
import torch
import pytest

# Add the src directory to the Python module search path
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../src/peen-ml'))
sys.path.append(src_path)

from model import (
    load_all_npy_files,
    CheckerboardDataset,
    ChannelAttention,
)

# Smoke Test
def test_load_all_npy_files_smoke():
    """Smoke test to check if the function runs without errors."""
    base_folder = "./test_simulations"
    result = load_all_npy_files(base_folder, ("checkerboard", "displacements"))
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
    with pytest.raises(FileNotFoundError):
        load_all_npy_files(base_folder, ("checkerboard", "displacements"), skip_missing=False)

if __name__ == "__main__":
    pytest.main()
