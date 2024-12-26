"""
Tests for the GUI module in the peen-ml project.
"""

from shotpeen_gui import App  # pylint: disable=wrong-import-position
import os
import sys
from unittest.mock import patch
import pytest

src_path = os.path.abspath(os.path.join(
    os.path.dirname(__file__), '../src/peen-ml'))
sys.path.append(src_path)


@pytest.fixture
def app():
    """Create an instance of the App class for testing, mocking GUI operations."""
    with patch("tkinter.Tk") as mocktk, \
            patch("PIL.Image.open"), \
            patch("PIL.ImageTk.PhotoImage"):
        mock_root = mocktk()
        app_instance = App(mock_root)
    return app_instance


def test_smoke_app_initialization(app):  # pylint: disable=redefined-outer-name
    """Smoke test to check if the App initializes without crashing."""
    assert app is not None


def test_train_model_directory_not_exists(app):  # pylint: disable=redefined-outer-name
    """One-shot test: Check if train_model correctly handles a non-existent directory."""
    non_existing_dir = "/tmp/non_existent_data_folder"

    with patch("tkinter.messagebox.showerror") as mock_error, \
            patch("os.path.exists", return_value=False):
        app.train_model(non_existing_dir)
        mock_error.assert_called_once_with("Error",
                                           f"The folder path does not exist: {non_existing_dir}")


def test_preview_file_directory_not_exists(app):  # pylint: disable=redefined-outer-name
    """Edge test: Check if preview_file handles a non-existent directory."""
    non_existing_dir = "/tmp/non_existent_data_folder"

    with patch("tkinter.messagebox.showerror") as mock_error:
        app.preview_file(non_existing_dir)
        mock_error.assert_called_once_with("Error",
                                           f"The Folder path does not exist: {non_existing_dir}")


def test_preview_file_directory_empty(app):  # pylint: disable=redefined-outer-name
    """Edge test: Check if preview_file handles an empty directory."""
    empty_dir = "/tmp/empty_folder"
    os.makedirs(empty_dir, exist_ok=True)
    with patch("tkinter.messagebox.showwarning") as mock_warning:
        app.preview_file(empty_dir)
        mock_warning.assert_called_once_with(
            "Warning", "The directory is empty.")
    os.rmdir(empty_dir)
