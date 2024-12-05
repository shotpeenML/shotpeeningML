import pytest
from unittest.mock import MagicMock, patch
import tkinter as tk
import os
import sys

# Add the src directory to the Python module search path
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../src/ShotPeenWithML'))
sys.path.append(src_path)
from Shotpeen_Gui import App


# Smoke Test
def test_app_initialization():
    """Smoke Test: Ensure the app initializes without errors."""
    root = tk.Tk()
    app = App(root)
    assert isinstance(app, App), "App did not initialize correctly."


# One-Shot Test 1: Test Main Menu Navigation
def test_main_menu():
    """Test that the main menu is displayed."""
    root = tk.Tk()
    app = App(root)
    assert len(app.root.winfo_children()) > 0, "Main menu widgets were not created."


# One-Shot Test 2: Test Browse File Functionality
@patch("Shotpeen_Gui.filedialog.askopenfilename", return_value="/path/to/mock/file")
def test_browse_file(mock_askopenfilename):
    """Test the browse file functionality."""
    root = tk.Tk()
    app = App(root)
    mock_variable = MagicMock()
    app.browse_file(mock_variable)
    mock_askopenfilename.assert_called_once()
    mock_variable.set.assert_called_with("/path/to/mock/file")


# Edge Test 1: Test Empty File Selection
@patch("Shotpeen_Gui.filedialog.askopenfilename", return_value="")
def test_browse_file_empty(mock_askopenfilename):
    """Test browsing with no file selected."""
    root = tk.Tk()
    app = App(root)
    mock_variable = MagicMock()
    app.browse_file(mock_variable)
    mock_askopenfilename.assert_called_once()
    mock_variable.set.assert_not_called()


# Edge Test 2: Test Training Completion
@patch("Shotpeen_Gui.App.finish_training")
def test_training_completion(mock_finish_training):
    """Test that training process completes."""
    root = tk.Tk()
    app = App(root)
    log_widget = MagicMock()
    progress_bar = MagicMock()
    
    # Start training (this schedules the finish_training call)
    app.start_training(log_widget, progress_bar)
    
    # Simulate the execution of the `after` scheduled task
    app.root.update_idletasks()  # Process any pending events
    app.root.after_idle(mock_finish_training.assert_called_once_with, log_widget, progress_bar)



if __name__ == "__main__":
    pytest.main()
