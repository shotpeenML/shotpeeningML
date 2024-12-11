# peen-ml

**Machine learning solution to predict deformation from shot peening as an alternative to dynamic simulation.** This repository provides tools and workflows to input shot peening parameters and geometry, train machine learning (ML) models to predict resulting deformations, visualize outcomes, and interact with the models through a user-friendly GUI. It aims to streamline the process of exploring and optimizing shot peening recipes, reducing reliance on time-consuming finite element analysis (FEA) simulations.

## Table of Contents
- [Overview](#overview)
- [Key Features](#key-features)
- [Repository Structure](#repository-structure)
- [Core Components](#core-components)
- [User Roles and Use Cases](#user-roles-and-use-cases)
- [Installation and Setup](#installation-and-setup)
- [Running the GUI](#running-the-gui)
- [Training and Evaluating the ML Model](#training-and-evaluating-the-ml-model)
- [Data Visualization](#data-visualization)
- [Testing](#testing)
- [License](#license)
- [Authors](#authors)
- [Acknowledgments](#acknowledgments)

## Overview
Shot peening is a manufacturing process used to improve material properties by bombarding a surface with small beads (shots). Predicting deformation due to shot peening typically involves complex FEA simulations. This repository provides a machine learning-driven approach to quickly approximate deformation results, enabling engineers to:
- Rapidly iterate on shot peening parameters.
- Compare multiple recipes without running time-consuming simulations.
- Visualize predicted outcomes and analyze their effects on component geometry.


This project was developed as part of the CSE 583 Software Development for Data Scientists course at the University of Washington, aiming to demonstrate best practices in code organization, testing, documentation, and continuous integration.

### Style
To ensure clean and maintainable code, we have followed **PEP 8** guidelines and linted the code to the best of our ability. However, we decided to relax the line length limitations to maintain code readability and logical flow where necessary.

### Replication
This repository includes resources to replicate experiments and demonstrate functionality:  
- **Demo Simulations:** The `tests/simulation_0` directory contains a sample simulation that can be used to explore and validate the repository’s workflows.  
- **Full Dataset Training:** For generating complete datasets, use the provided Abaqus scripts in `dataset1_script.py` and `dataset2_script.py`. These scripts facilitate dataset creation for model training.  

**Note:** The Abaqus scripts are provided solely for reproducibility purposes and have not been linted or formatted for PEP 8 compliance. They serve as supplementary tools and do not impact the repository’s primary functionality.

## Key Features
- **GUI for Ease of Use**: A graphical user interface (`shotpeen_gui.py`) to train models and load existing ones without requiring deep ML expertise.
- **ML-Based Prediction Engine**: A CNN-based model (with attention mechanisms) that predicts deformation from shot peening recipes and geometry (`model.py`).
- **Data Visualization Tools**: Scripts (`data_viz.py`) to visualize checkerboard patterns, stress fields, and deformation magnitudes, aiding in interpretation and analysis.
- **Modular Architecture**: Separate modules for input data handling, visualization, prediction, and storage, supporting extensibility and maintenance.
- **Comparison and Analysis**: Tools to compare deformation predictions across multiple shot peening recipes, helping users find optimal process parameters.
- **Continuous Integration**: A GitHub Actions workflow for style checking (`.github/workflows/pylint.yml`).

## Repository Structure
```
[Repository Root]
├─ .github/
│  └─ workflows/
│     └─ pylint.yml          # CI pipeline (Pylint checks)
├─ blueprint/
│  ├─ Components.md          # Description of system components
│  ├─ Describing_a_usecase.md
│  └─ User_story.md
├─ dataset_sample/
│  ├─ dataset1_sample.rar    # Sample dataset files (for demonstration)
│  ├─ dataset2_sample.rar
│  └─ readme.txt.txt
├─ src/
│  └─ peen-ml/
│     ├─ data_viz.py         # Visualization tools
│     ├─ dataset1_script.py  # Script to generate dataset1
│     ├─ dataset2_script.py  # Script to generate dataset2
│     ├─ model.py            # ML model definition, training & evaluation
│     ├─ model_notebook_v2.ipynb  # Model evaluation & results 
│     └─ model_notebook_v3.ipynb  # Model evaluation & results
├─ tests/
│  ├─ simulation_0           # Example Data for Test cases
│  ├─ simulation_1           # Example Data for Test cases
│  ├─ test_Shotpeen_Gui.py   # GUI tests
│  ├─ test_data_viz.py       # Visualization tests
│  └─ test_model.py          # Model tests
├─ .gitignore
├─ LICENSE
├─ README.md                 # This README file
├─ pyproject.toml            # Project configuration & dependencies
└─ shotpeen_gui.py           # Main GUI application
```

## Core Components

- **Data Input Module** (in `shotpeen_gui.py` and `src/peen-ml/` scripts):  
  Collects geometry and shot peening parameters, validates data, and feeds it into the ML model.

- **Prediction Engine (ML Model)** (in `model.py`):  
  A CNN with attention mechanisms to predict deformation from input parameters. Trained on historical or simulated FEA data, it approximates deformation outcomes quickly.

- **Visualization Module** (in `data_viz.py`):  
  Tools to visualize mesh deformation, stress fields, and checkerboard patterns. Helps engineers understand the predicted outcomes intuitively.

- **Recipe & Result Storage**:  
  Enables saving shot peening recipes and predictions for future retrieval and analysis.

- **Comparison & Analysis Module** (future add-on):  
  Allows side-by-side comparison of multiple shot peening recipes to identify the most effective parameters.

- **Alert & Reporting System** (future add-on):  
  Intended to notify users when predicted deformation exceeds thresholds and to generate reports for quality control.

## User Roles and Use Cases
Usecases and users that this was created for and defined in `blueprint/`:

- **Mechanical Engineer (Alex)**: Inputs geometry and recipe parameters, quickly gets deformation predictions.
- **Process Engineer (Jordan)**: Compares multiple recipes to find optimal shot peening settings.
- **Quality Control Engineer (Taylor)**: Validates predicted results against actual deformation data, refining the model over time.
- **Data Scientist (Sam)**: Improves ML models by experimenting with algorithms and leveraging stored datasets.

Typical uses include:  
- **Input Shot Peening Recipe and Geometry**: Load CAD files and shot parameters to generate predictions.  
- **Predict and Visualize Deformation**: Instantly see deformation overlays on a 3D model.  
- **Compare Recipes**: Evaluate which shot peening parameters yield the desired deformation efficiently.  
- **Store and Retrieve Historical Data**: Build a knowledge base of recipes and predictions to inform future decisions.

## Installation and Setup
**Prerequisites:**
- Python 3.7–3.12
- Git
- (Optional) Conda for environment management
- requests 2.25.1+
- numpy 1.20.0+
- matplotlib 3.4.0+
- pandas 1.3.0+
- torch 1.9.0+
- tinker
- pillow

**Steps:**
1. **Clone the Repository:**
   ```bash
   git clone SSH
   ```

2. **Create and Activate a Virtual Environment (Recommended):**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install Dependencies:**

   Directly via pyproject.toml using `pip` (make sure this is run within peen-ml):
   ```bash
   pip install .
   ```
   
   **Dependencies include:**  
   - `requests>=2.25.1`  
   - `numpy>=1.20.0`  
   - `matplotlib>=3.4.0`  
   - `pandas>=1.3.0`  
   - `torch>=1.9.0`  
   - `pillow`  
   - `tkinter` (usually comes with Python, but may need separate installation on some systems)

## Running the GUI
The GUI (`shotpeen_gui.py`) provides an accessible interface for training new models and loading existing ones.
![Gui_Main_Menu](https://raw.githubusercontent.com/onestr1/peen-ml/refs/heads/main/images/gui_main_menu.png)

**Launch the GUI:**

All the commands are launched from the root directory `~/peen-ml$`
```bash
python shotpeen_gui.py
```

**Features via GUI:**
- **Train Model**:  
  Opens a dialog to select training and testing data, shows a training log, and displays a progress bar.
  
- **Load Model**:  
  Opens a dialog to load existing models and step files for review.
  

## Training and Evaluating the ML Model
The ML workflow is defined in `src/peen-ml/model.py`:

1. **Prepare Data**:  
   Place your simulation data (`.npy` files) into a structured directory (e.g., `Dataset1_Random_Board` with `Simulation_0`, `Simulation_1`, etc.).

2. **Train Model or Load Existing Model via the GUI**:
   ```
   Harsh include images from GUI and explain
   ```
   ![Load_model_page](https://raw.githubusercontent.com/onestr1/peen-ml/refs/heads/main/images/load_model_page.png)
   ![train_model_page](https://raw.githubusercontent.com/onestr1/peen-ml/refs/heads/main/images/train_model_page.png)


   This will:
   - do this
   - do that
   - do the other thing

3. **Evaluate the Model**:
   After training, the script evaluates the model on the test set, reporting MSE and sMAPE metrics.

## Data Visualization
`data_viz.py` provides functions to visualize checkerboard patterns, stress fields, and deformation fields. These functions are called in 'shotpeen_gui.py' to display the results of the trained model or the checkerboard pattern intensity.
A sample usage of the data visualization functions is shown below. You can run the `main()` script in `data_viz.py` for a demo:
```bash
python src/peen-ml/data_viz.py
```
This generates plots for:
- Checkerboard patterns of expansion coefficients.
- Undeformed vs. Deformed meshes.
- Stress fields (e.g., von Mises stress).
- Deformation magnitudes.

## Testing
Tests are located in the `tests/` directory and can be run using `pytest` or `unittest`.

**Run Tests:**
```bash
pytest tests
```

This includes:
- `test_Shotpeen_Gui.py`: Tests GUI functionality.
- `test_data_viz.py`: Tests data visualization utilities.
- `test_model.py`: Tests model loading, training, and evaluation functionalities.

## License
This project is licensed under the [MIT License](LICENSE).

## Authors
- [Onest Rexhepi](mailto:onestr@uw.edu)  
  *Contributions:*  
  - Established and managed the GitHub repository, including all documentation and replication workflows.  
  - Generated data for both datasets utilized by the model.  
  - Designed and implemented data visualizations scripts for model inputs and outputs that were integrated into the GUI.  

- [Harshavardhan Sameer Raje](mailto:harshr@uw.edu)  
  *Contributions:*  
  - Developed the graphical user interface (GUI) to ensure user-friendly interaction with the model.  
  - Integrated the GUI with backend scripts for training and visualization.  

- [Jiachen Zhong](mailto:jczhong@uw.edu)  
  *Contributions:*  
  - Designed and implemented the CNN model architecture for deformation prediction.  
  - Optimized the model pipeline for efficient training and inference.  

- [Xuanyu Shen](mailto:xshen20@uw.edu)  
  *Contributions:*  
  - Developed test cases to ensure functionality and reliability of scripts and models.  




## Acknowledgments
- University of Washington CSE 583 course staff for guidance on best practices in software development for data scientists.
- The broader Python and open-source community for providing tools and libraries that made this project possible.
