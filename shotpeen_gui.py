"""
This is the main calling and GUI script for the shot peening backend.
This script allows lay users to train and test shot peening models,
and it serves as a usage example of calling the trained models.

Workflow:
- The application presents a graphical user interface (GUI) where the user can choose
     to train a new model or load an existing model.
- If the user selects "Train Model," the application opens a dialog to configure the training,
     including selecting input files (training and testing data) and tracking the 
     training progress via a log and progress bar.
- If the user selects "Load Model," the application allows them to load an existing model and 
     related files (STEP file, model file), with an option to preview the STEP file.
- The script also checks and installs any required dependencies that are missing.

Dependencies:
- requests>=2.25.1
- numpy>=1.20.0
- matplotlib>=3.4.0
- pandas>=1.3.0
- pytorch>=1.9.0
- tkinter
- pillow

Function Definitions:
1. `check_install(package_id: str)`: Checks if a required package is installed, and
     installs it using either `pip` or `conda` if it's missing.
2. `__init__(self, root_tk)`: Initializes the main application window and
     calls the `main_menu` function to display the initial GUI.
3. `main_menu(self)`: Displays the main menu of the application with options to train or
     load a model.
4. `train_model_dialog(self)`: Opens a dialog window for training a model,
     allowing the user to select training data and showing a log and progress bar.
5. `load_model_dialog(self)`: Opens a dialog window for loading an existing model,
     including options to select model files, STEP files, and output paths.
6. `browse_file(self, variable)`: Opens a file dialog to allow the user to select a file and
     stores the file path in the provided variable.
7. `browse_directory(self, variable)`: Opens a directory dialog to allow the user to select a
    directory and stores the directory path in the provided variable.
8. `preview_file(self, file_path)`: Previews the selected STEP file by displaying its path in
     a message box.
9. `start_training(self, log_widget, progress_bar)`: Simulates the model training process,
     updating the log and progress bar.
10. `finish_training(self, log_widget, progress_bar)`: Completes the training process,
     stops the progress bar, and updates the training log.

Usage:
- Upon running the script, a GUI window opens with the option to train a model or
     load an existing one.
- Selecting "Train Model" opens a dialog to input training data, with a training log and
     progress bar to track progress.
- Selecting "Load Model" opens a dialog to load an existing model and
     related files (STEP and output paths).
Author:
- Harshavardhan Raje
"""
import tkinter as tk
from tkinter import filedialog, ttk, messagebox
import sys
import subprocess
import shutil
import os
import threading
import torch
from PIL import Image, ImageTk
# Append src folder to path such that the called python files can be called.
sys.path.append(os.path.join(os.path.dirname(__file__), 'src', 'peen-ml'))
# Deviating from PEP8 to make sure that this script can call the backend
from model import train_model, create_data_loaders, create_model, evaluate_model
from model import train_save_gui
from data_viz import visualize_checkerboard, compute_deformed_mesh, visualize_mesh
from data_viz import visualize_stress_field


def check_install(package_id: str):
    """
    Checks if a package is installed and installs it if missing.

    Args:
        package_id (str): The name of the package to check/install.
    """
    try:
        # Try importing the package to check if it's installed
        __import__(package_id)
        print(f"{package_id} is already installed.")
    except ModuleNotFoundError:
        print(f"{package_id} is not installed.")
        try:
            # Try installing using pip
            print(f"Trying to install {package_id} using pip...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package_id])
        except subprocess.CalledProcessError as pip_error:
            print(f"Pip installation failed: {pip_error}")
            try:
                # Try installing using conda (only if conda is available)
                if shutil.which("conda"):  # Check if conda is available
                    print(f"Trying to install {package_id} using conda...")
                    subprocess.check_call(["conda", "install", package_id, "-y"])
                else:
                    print("Conda is not available. Please install using pip.")
            except subprocess.CalledProcessError as conda_error:
                print(f"Conda installation failed: {conda_error}")
                print(f"Unable to install {package_id} with pip or conda.")
# import model as md

# viz.main()
class App:
    """
    This app provides a main menu with options to train a new model or load
    an existing model, along with dialogs to configure the training parameters
    and load models from files.
    """
    def __init__(self, root_tk):
        """
        Initializes the application window.
        
        Args:
            root (tk.Tk): The root Tkinter window to create the GUI.

        Sets up the main window with a title and geometry, and calls the main menu.
        """
        self.root = root_tk
        self.root.title("Model GUI")
        self.root.geometry("800x600")
        self.test_train_data_path = ""
        self.parent_process = None  # Initialize the attribute
        self.main_menu()

    def main_menu(self):
        """
        Displays the main menu of the application with options to train or load a model.

        Clears the window and adds buttons for training and loading models. 
        Each button triggers its respective dialog (training or loading).
        """
        # Clear window
        for widget in self.root.winfo_children():
            widget.destroy()

        # Main Menu Layout
        main_frame = tk.Frame(self.root, padx=20, pady=20)
        main_frame.pack(expand=True)
        try:
            bullet_bill_path = os.path.join(os.path.dirname(__file__), 'src', 'peen-ml', 'bullet_bill.png')
            image = Image.open(bullet_bill_path)
            image = image.resize((400, 250), Image.Resampling.LANCZOS)
            self.splash_image = ImageTk.PhotoImage(image)
            ttk.Label(main_frame, image=self.splash_image).grid(row=0,
                                              column=0,
                                                columnspan=2,
                                                  pady=20)
        except FileNotFoundError as e:
            raise f"Could not locate Bullet bill logo {e}"

        tk.Label(main_frame,
                  text="Main Menu",
                    font=("Arial", 24)).grid(row=1,
                                              column=0,
                                                columnspan=2,
                                                  pady=20)

        tk.Button(main_frame,
                   text="Train Model",
                     command=self.train_model_dialog,
                       width=20,
                         height=2).grid(row=2,
                                         column=0,
                                           padx=20,
                                             pady=20)
        tk.Button(main_frame,
                   text="Load Model",
                     command=self.load_model_dialog,
                       width=20,
                         height=2).grid(row=2,
                                         column=1,
                                           padx=20,
                                             pady=20)

    def get_file_path(self, relative_path):
        """
        Gets the file path of the requested file, works in development or .exe mode
        mainly for finding the path this repository is in.
        Args:
            relative_path (str): The relative path to the desired file.
        
        Returns:
            str: The absolute path to the file.
        """
        # Use the _MEIPASS attribute or abs path
        base_path = getattr(sys, '_MEIPASS', os.path.abspath("."))
        return os.path.join(base_path, relative_path)

    def train_model_dialog(self):
        """
        Opens a dialog window for training a model.
        
        Displays fields for selecting training and testing data, a log area for 
        showing training progress, and a progress bar to visually track the 
        training process.
        """
        dialog = tk.Toplevel(self.root)
        dialog.title("Train Model")
        dialog.geometry("1000x700")

        # Training Layout
        frame = tk.Frame(dialog, padx=20, pady=20)
        frame.pack(expand=True, fill=tk.BOTH)
        # TODO: look for folder path
        # File Selection
        tk.Label(frame,
                  text="Training and Testing Data Folder",
                    font=("Arial", 12)).grid(row=0,
                                              column=0,
                                                sticky="w",
                                                  pady=10)
        data_folder_var = tk.StringVar()
        tk.Entry(frame,
                  textvariable=data_folder_var,
                    width=50).grid(row=0,
                                   column=1,
                                     padx=10,
                                       pady=5)
        tk.Button(frame,
                   text="Browse",
                     command=lambda:
                       self.browse_directory(data_folder_var)).grid(row=0,
                                                             column=2,
                                                              pady=5)
        self.test_train_data_path = data_folder_var

        # Log and Progress Bar
        tk.Label(frame,
                  text="Training Log",
                    font=("Arial", 12)).grid(row=1,
                                              column=0,
                                                sticky="w",
                                                  pady=10)
        log = tk.Text(frame, height=10, width=60, state='disabled')
        log.grid(row=2, column=0, columnspan=3, pady=10)

        progress = ttk.Progressbar(frame,
                                    orient=tk.HORIZONTAL,
                                      length=500,
                                        mode='determinate')
        progress.grid(row=3, column=0, columnspan=3, pady=10)

        # Buttons
        tk.Button(frame,
                   text="Train",
                     command=lambda:
                      train_save_gui(data_folder_var.get()), width=15).grid(row=4,
                                                                          column=0,
                                                                           pady=20)
        tk.Button(frame,
                   text="Back to Main Menu",
                   command=dialog.destroy, width=15).grid(row=4, column=2, pady=20)

    def load_model_dialog(self):
        """
        Opens a dialog window for loading an existing model.
        
        Allows the user to select a model file, a STEP file, and an output path.
        Provides options to preview the STEP file and go back to the main menu.
        """
        dialog = tk.Toplevel(self.root)
        dialog.title("Load Model")
        dialog.geometry("900x500")

        # Load Model Layout
        frame = tk.Frame(dialog, padx=20, pady=20)
        frame.pack(expand=True)

        # Configure grid to center contents
        for i in range(5):  # Assuming there are up to 5 rows
            frame.grid_rowconfigure(i, weight=1)
        for i in range(3):  # Assuming 3 columns: Label, Entry, and Button
            frame.grid_columnconfigure(i, weight=1)

        # Model File Selection
        tk.Label(frame,
                  text="Model File",
                    font=("Arial", 12)).grid(row=0, column=0, sticky="e", pady=10)
        model_file_var = tk.StringVar()
        tk.Entry(frame,
                  textvariable=model_file_var,
                    width=50).grid(row=0, column=1, padx=10, pady=5)
        tk.Button(frame,
                   text="Browse",
                     command=lambda:
                       self.browse_file(model_file_var)).grid(row=0,
                                                               column=2,
                                                                 pady=5)

        # STEP File Selection
        tk.Label(frame,
                  text="Peen Intensity Folder",
                    font=("Arial", 12)).grid(row=1, column=0, sticky="e",
                                              pady=10, padx=10)
        checkerboard_file_var = tk.StringVar()
        tk.Entry(frame,
                  textvariable=checkerboard_file_var,
                    width=50).grid(row=1, column=1, padx=10, pady=5)
        tk.Button(frame,
                   text="Browse",
                     command=lambda:
                       self.browse_directory(checkerboard_file_var)).grid(row=1,
                                                            column=2,
                                                              pady=5)

        # Output Path Selection
        tk.Label(frame,
                  text="Output Path",
                    font=("Arial", 12)).grid(row=2, column=0, sticky="e", pady=10)
        output_path_var = tk.StringVar()
        tk.Entry(frame,
                  textvariable=output_path_var,
                    width=50).grid(row=2, column=1, padx=10, pady=5)
        tk.Button(frame,
                   text="Browse",
                     command=lambda: self.browse_directory(output_path_var)).grid(row=2,
                                                                                   column=2,
                                                                                     pady=5)
        # TODO: Checkerboard Pattern, numpy array
        # Buttons at the bottom
        tk.Button(frame,
                   text="Input Peen Intensity Preview",
                     command=lambda:
                       self.preview_file(checkerboard_file_var.get()),
                         width=25).grid(row=3,
                             column=0,
                               pady=20,
                                   sticky="n")
        tk.Button(frame,
                   text="Predicted Deformation Preview",
                     command=lambda:
                       self.preview_file(checkerboard_file_var.get()),
                         width=25).grid(row=3,
                             column=2,
                               pady=20,
                                   sticky="n")
        tk.Button(frame,
                   text="Back to Main Menu",
                     command=dialog.destroy, width=30).grid(row=4,
                                                             column=1,
                                                               pady=20
                                                               , sticky="n")

    def browse_file(self, variable):
        """
        Opens a file dialog to allow the user to select a file.
        
        Args:
            variable (tk.StringVar): The variable to store the selected file path.
        """
        filepath = filedialog.askopenfilename()
        if filepath:
            variable.set(filepath)

    def browse_directory(self, variable):
        """
        Opens a directory dialog to allow the user to select a directory.
        
        Args:
            variable (tk.StringVar): The variable to store the selected directory path.
        """
        dirpath = filedialog.askdirectory()
        if dirpath:
            variable.set(dirpath)

    def preview_file(self, folder_path):
        """
        Previews the selected .npy file by displaying a message with its path.
        Helps preview the checkerboard pattern sent in and pushed out.
        
        Args:
            file_path (str): The path of the npy file to preview.
        """
        # if folder_path:
        #     messagebox.showinfo("Preview", f"Previewing: {folder_path}")
        # else:
        #     messagebox.showerror("Error", "No file selected!")
        print("preview_task")
        # Check if the provided path is valid and exists
        if not os.path.exists(folder_path):
            messagebox.showerror("Error", f"The Folder path does not exist: {folder_path}")
            return
        # Ensure the path is a directory or handle invalid input
        if not os.path.isdir(folder_path):
            messagebox.showerror("Error", f"The provided path is not a directory: {folder_path}")
            return

        try:
            entry_values = os.listdir(folder_path)  # List contents of the directory
            if not entry_values:
                messagebox.showwarning("Warning", "The directory is empty.")
                return
        except OSError as e:
            messagebox.showerror("Error", f"Failed to access the directory: {e}")
            return
        # Run the preview in a separate thread
        try:
            thread = threading.Thread(target=self.run_preview, args=(folder_path,))
            thread.start()
        except RuntimeError as e:
            messagebox.showerror("Error", f"Threading error: {e}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to start the preview thread: {e}")

    def run_preview(self, geometry_folder_path):
        """
        This function runs the visualizer in a seperate thread 
        """
        print(geometry_folder_path)

        # TODO Edit the python version in production
        # command = [sys.executable, "Step_file_visualizer.py", geometry_file_path]
        # process = subprocess.Popen(command,shell=True,
        #                             stderr=subprocess.PIPE,
        #                               stdout=subprocess.PIPE)
        # shell_pid=process.pid
        # self.parent_process = psutil.Process(shell_pid)
        # stdout, stderr = process.communicate()

        visualize_checkerboard(geometry_folder_path)

    def train_model(self, data_folder):
        """
        Trains a model using the data in the specified folder.
        """
        if not os.path.exists(data_folder):
            messagebox.showerror("Error", f"The folder path does not exist: {data_folder}")
            return
        # num_simulations = self.num_of_simulations(data_folder)
        train_loader, val_loader, test_loader, _ = create_data_loaders(data_folder)
        model = create_model(input_channels=1, num_nodes=5202)
        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        train_model(model, train_loader, val_loader, criterion, optimizer, None)

        messagebox.showinfo("Training Complete", "Model training has completed successfully!")

    def start_training(self, log_widget, progress_bar):
        """
        Starts the model training process and updates the log and progress bar.
        
        Args:
            log_widget (tk.Text): The widget to display training log messages.
            progress_bar (ttk.Progressbar): The progress bar to show training progress.
        """
        # Simulate training process
        log_widget.config(state='normal')
        log_widget.insert(tk.END, "Training started...\n")
        log_widget.see(tk.END)
        log_widget.config(state='disabled')
        progress_bar.start(10)  # Simulate progress
        self.root.after(3000, lambda: self.finish_training(log_widget, progress_bar))

    def finish_training(self, log_widget, progress_bar):
        """
        Completes the training process, stops the progress bar, and updates the log.
        
        Args:
            log_widget (tk.Text): The widget to display training log messages.
            progress_bar (ttk.Progressbar): The progress bar to stop after training completion.
        """
        progress_bar.stop()
        log_widget.config(state='normal')
        log_widget.insert(tk.END, "Training completed!\n")
        log_widget.see(tk.END)
        log_widget.config(state='disabled')
    def num_of_simulations(self, folder_path):
        simulation_folders = [
              os.path.join(folder_path, folder)
              for folder in os.listdir(folder_path)
              if folder.startswith("Simulation_") and folder[len("Simulation_"):].isdigit()
          ]
        print(f"# Simulation folders: {len(simulation_folders)}")
        return len(simulation_folders)


# Check for missing modules
dependencies = [
    "requests",
    "numpy",
    "matplotlib",
    "pandas",
    "pytorch",
    "tkinter",
    "pillow"
]

# for library in dependencies:
#     check_install(library)

# Run the Application
if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()
