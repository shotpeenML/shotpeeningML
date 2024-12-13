"""
Module: Displacement Prediction using CNN with Attention Mechanisms

This module contains classes and functions for loading and processing simulation data,
defining neural network models with attention mechanisms, and training and evaluating the models
for displacement prediction.

Features:
1. Loading .npy files from simulation datasets.
2. Custom PyTorch Dataset classes for checkerboard and displacement data.
3. Channel and spatial attention modules for feature enhancement.
4. A CNN model for displacement prediction.
5. Data loader creation, training, and evaluation utilities.

Author:
    Jiachen Zhong
Date:
    Dec 10, 2024
"""

import os
from pathlib import Path

import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader, random_split
import matplotlib.pyplot as plt


# 1. Load All Numpy Files Function
def load_all_npy_files(base_folder,
                        load_files=("checkerboard", "displacements"),
                          skip_missing=True):
    """
    Load specified .npy files from multiple simulation folders.

    Args:
        base_folder (str): The base folder containing simulation subfolders.
        load_files (tuple): Names of the files to load (default: ("checkerboard", "displacements")).
        skip_missing (bool): If True, skip missing files; otherwise, raise an error.

    Returns:
        dict: A dictionary containing loaded data arrays for the specified files.
              Keys are file names, and values are stacked arrays.
    """
    # Find all folders matching the pattern "Simulation_\d+"
    simulation_folders = [
        folder for folder in os.listdir(base_folder)
        if os.path.isdir(os.path.join(base_folder, folder)) and folder.startswith("Simulation_")
    ]

    # Sort folders numerically by the index after "Simulation_"
    simulation_folders.sort(key=lambda x: int(x.split("_")[1]))

    # Initialize dictionaries to store data
    data_dict = {key: [] for key in load_files}

    for simulation_folder in simulation_folders:
        simulation_path = os.path.join(base_folder, simulation_folder)

        for file_name in load_files:
            data_file_path = os.path.join(simulation_path, f"{file_name}.npy")

            if os.path.exists(data_file_path):
                # Load the file and append to the respective list
                data_dict[file_name].append(np.load(data_file_path))
                print(f"{file_name.capitalize()} from {simulation_folder} loaded successfully!")
            else:
                # Handle missing files
                if skip_missing:
                    print(f"{file_name.capitalize()} File not found in {simulation_folder}! Skipping...")
                else:
                    raise FileNotFoundError(f"{file_name.capitalize()} File not found in {simulation_folder}!")

    # Stack data from all simulations along a new axis
    stacked_data = {}
    for key, data_list in data_dict.items():
        if data_list:
            stacked_data[key] = np.stack(data_list)  # Stack along a new axis
        else:
            stacked_data[key] = None  # No data loaded for this key

    print("All specified data loaded and stacked successfully!")
    return stacked_data

# 2. Dataset Classes
class CheckerboardDataset(Dataset):
    """
    A PyTorch Dataset class for checkerboard patterns and displacement data.

    Args:
        checkerboards (numpy array): Array of checkerboard patterns (batch_size, height, width).
        displacements (numpy array): Array of displacements (batch_size, num_nodes, 3).
    """
    def __init__(self, checkerboards, displacements):
        """
        Args:
            checkerboards (numpy array): Array of checkerboard patterns (batch_size, height, width).
            displacements (numpy array): Array of displacements (batch_size, num_nodes, 3).
        """
        self.checkerboards = checkerboards
        self.displacements = displacements

    def __len__(self):
        """Returns the total number of samples in the dataset."""
        return len(self.checkerboards)

    def __getitem__(self, idx):
        """
        Retrieves a sample by index.

        Args:
            idx (int): Index of the sample.

        Returns:
            tuple: A tuple containing the checkerboard tensor and the displacement tensor.
        """
        checkerboard = self.checkerboards[idx]
        displacement = self.displacements[idx]

        # Add a channel dimension to checkerboard (1 channel) to match with CNN expectations
        checkerboard = torch.tensor(checkerboard, dtype=torch.float32).unsqueeze(0)  # (1, height, width)
        displacement = torch.tensor(displacement, dtype=torch.float32)  # (num_nodes, 3)

        return checkerboard, displacement

class NormalizedDataset(Dataset):
    """
    A wrapper for normalizing datasets. Takes a base dataset and applies normalization to its features.

    Args:
        base_dataset (Dataset): The original dataset to normalize.
    """
    def __init__(self, base_dataset):
        self.base_dataset = base_dataset
        self.checkerboards = torch.cat([data[0] for data in base_dataset], dim=0)  # Collect all checkerboards
        self.min_val = self.checkerboards.min()
        self.max_val = self.checkerboards.max()

    def __len__(self):
        """Returns the total number of samples in the dataset."""
        return len(self.base_dataset)

    def __getitem__(self, idx):
        """
        Retrieves a sample by index and normalizes the checkerboard.

        Args:
            idx (int): Index of the sample.

        Returns:
            tuple: A tuple containing the normalized checkerboard tensor and the displacement tensor.
        """
        checkerboard, displacement = self.base_dataset[idx]
        normalized_checkerboard = (checkerboard - self.min_val) / (self.max_val - self.min_val)
        return normalized_checkerboard, displacement

# 3. Attention Modules
class ChannelAttention(nn.Module):
    """
    Channel Attention module for emphasizing relevant feature channels.

    Args:
        channels (int): Number of input channels.
        reduction (int): Reduction ratio for channel compression (default: 16).
    """
    def __init__(self, channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.fc1 = nn.Conv2d(channels, channels // reduction, kernel_size=1)
        self.fc2 = nn.Conv2d(channels // reduction, channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        Forward pass of the Channel Attention module.

        Args:
            x (Tensor): Input feature map.

        Returns:
            Tensor: Feature map after channel attention.
        """
        avg_pool = torch.mean(x, dim=(2, 3), keepdim=True)  # Global average pooling
        max_pool = torch.max(torch.max(x, dim=2, keepdim=True).values, dim=3, keepdim=True).values  # Global max pooling
        scale = self.fc1(avg_pool) + self.fc1(max_pool)
        scale = self.fc2(torch.relu(scale))
        return self.sigmoid(scale) * x

class SpatialAttention(nn.Module):
    """
    Spatial Attention module for emphasizing relevant spatial regions.
    """
    def __init__(self):
        super(SpatialAttention, self).__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size=7, padding=3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        Forward pass of the Spatial Attention module.

        Args:
            x (Tensor): Input feature map.

        Returns:
            Tensor: Feature map after spatial attention.
        """
        avg_pool = torch.mean(x, dim=1, keepdim=True)  # Channel-wise average
        max_pool = torch.max(x, dim=1, keepdim=True).values  # Channel-wise max
        scale = torch.cat([avg_pool, max_pool], dim=1)
        return self.sigmoid(self.conv1(scale)) * x

# 4. CNN Model with Attention
class DisplacementPredictor(nn.Module):
    """
    A CNN model with channel and spatial attention for displacement prediction.

    Args:
        input_channels (int): Number of input channels.
        num_nodes (int): Number of nodes in the displacement data.
    """
    def __init__(self, input_channels, num_nodes):
        super(DisplacementPredictor, self).__init__()

        # Convolutional layers for spatial feature extraction
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )
        self.ca1 = ChannelAttention(32)
        self.sa1 = SpatialAttention()

        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.ca2 = ChannelAttention(64)
        self.sa2 = SpatialAttention()

        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )
        self.ca3 = ChannelAttention(128)
        self.sa3 = SpatialAttention()

        # Fully connected layers for displacement prediction
        self.fc = nn.Sequential(
            nn.Linear(128 * 5 * 5, 512),  # Adjust based on flattened dimension after conv layers
            nn.ReLU(),
            nn.Linear(512, num_nodes * 3)  # Output size = num_nodes * 3 (displacement components)
        )

    def forward(self, x):
        """
        Forward pass of the displacement predictor model.

        Args:
            x (Tensor): Input tensor of shape (batch_size, input_channels, height, width).

        Returns:
            Tensor: Predicted displacement tensor of shape (batch_size, num_nodes, 3).
        """
        x = self.conv1(x)
        x = self.ca1(x)
        x = self.sa1(x)

        x = self.conv2(x)
        x = self.ca2(x)
        x = self.sa2(x)

        x = self.conv3(x)
        x = self.ca3(x)
        x = self.sa3(x)

        # Flatten the output for fully connected layers
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        # Reshape output to (batch_size, num_nodes, 3)
        return x.view(x.size(0), -1, 3)

# 5. Data Loader Creation Function
def create_data_loaders(base_folder, load_files=("checkerboard", "displacements"), skip_missing=True, batch_size=15):
    """
    Create PyTorch DataLoaders for training, validation, and testing.

    Args:
        base_folder (str): Path to the folder containing simulation data.
        num_simulations (int): Number of simulation subfolders to process.
        load_files (tuple): Names of the files to load (default: ("checkerboard", "displacements")).
        skip_missing (bool): Whether to skip missing files or raise an error.
        batch_size (int): Batch size for DataLoaders.

    Returns:
        tuple: DataLoaders for training, validation, and testing, and the loaded data dictionary.
    """
    loaded_data = load_all_npy_files(base_folder, load_files, skip_missing)
    checkerboard = loaded_data["checkerboard"]
    displacements = loaded_data["displacements"]

    # Set Random State for Reproducibility
    torch.manual_seed(2024)
    np.random.seed(2024)

    # Create dataset
    full_dataset = CheckerboardDataset(checkerboard, displacements)

    # Split into train, validation, and test sets
    train_size = int(0.7 * len(full_dataset))
    val_size = int(0.15 * len(full_dataset))
    test_size = len(full_dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(full_dataset, [train_size, val_size, test_size])

    # Wrap subsets with normalization
    train_dataset = NormalizedDataset(train_dataset)
    val_dataset = NormalizedDataset(val_dataset)
    test_dataset = NormalizedDataset(test_dataset)

    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, loaded_data

# 6. Model Creation Function
def create_model(input_channels, num_nodes):
    """
    Create a DisplacementPredictor model.

    Args:
        input_channels (int): Number of input channels.
        num_nodes (int): Number of nodes in the displacement data.

    Returns:
        DisplacementPredictor: The instantiated model.
    """
    model = DisplacementPredictor(input_channels, num_nodes)
    return model

# 7. Training Function
def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, epochs=10, patience=5):
    """
    Train the model with early stopping.

    Args:
        model (nn.Module): The PyTorch model to train.
        train_loader (DataLoader): DataLoader for training data.
        val_loader (DataLoader): DataLoader for validation data.
        criterion (nn.Module): Loss function.
        optimizer (torch.optim.Optimizer): Optimizer for training.
        scheduler (torch.optim.lr_scheduler._LRScheduler): Learning rate scheduler.
        epochs (int): Maximum number of training epochs.
        patience (int): Number of epochs to wait for improvement before stopping early.

    Returns:
        tuple: Lists of training and validation losses per epoch.
    """
    best_val_loss = float('inf')
    early_stop_counter = 0

    train_losses = []
    val_losses = []

    # Initialize plot (temporarily disable to ensure print statements work)
    plt.ion()
    fig, ax = plt.subplots()
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training and Validation Loss')
    line1, = ax.plot([], [], label='Training Loss', color='blue')
    line2, = ax.plot([], [], label='Validation Loss', color='orange')
    ax.legend()

    for epoch in range(epochs):
        # Training
        model.train()
        epoch_loss = 0.0

        for checkerboard, displacement in train_loader:
            optimizer.zero_grad()
            predicted_displacements = model(checkerboard)
            loss = criterion(predicted_displacements, displacement)
            epoch_loss += loss.item()
            loss.backward()
            optimizer.step()

        scheduler.step()
        train_loss = epoch_loss / len(train_loader)
        train_losses.append(train_loss)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for checkerboard, displacement in val_loader:
                predicted_displacements = model(checkerboard)
                loss = criterion(predicted_displacements, displacement)
                val_loss += loss.item()
        val_loss /= len(val_loader)
        val_losses.append(val_loss)

        # Early Stopping Check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            early_stop_counter = 0
        else:
            early_stop_counter += 1
            if early_stop_counter >= patience:
                print(f"Early stopping triggered after {epoch+1} epochs.")
                break

        # Update plot (temporarily disable)
        line1.set_xdata(range(1, len(train_losses) + 1))
        line1.set_ydata(train_losses)
        line2.set_xdata(range(1, len(val_losses) + 1))
        line2.set_ydata(val_losses)
        ax.relim()
        ax.autoscale_view()
        fig.canvas.draw()
        fig.canvas.flush_events()

        # Print Losses
        print(f"Epoch {epoch+1}/{epochs}, Training Loss: {train_loss:.10f}, Validation Loss: {val_loss:.10f}")

    plt.ioff()
    plt.show()
    return train_losses, val_losses

# 8. Evaluation Function
def smape(y_true, y_pred):
    """
    Calculate Symmetric Mean Absolute Percentage Error (sMAPE).

    Args:
        y_true (Tensor): Ground truth tensor.
        y_pred (Tensor): Predicted tensor.

    Returns:
        float: sMAPE value.
    """
    numerator = torch.abs(y_true - y_pred)
    denominator = (torch.abs(y_true) + torch.abs(y_pred)) / 2
    smape_value = torch.mean(numerator / denominator)
    return smape_value

def evaluate_model(model, test_loader, criterion):
    """
    Evaluate the model on the test set.

    Args:
        model (nn.Module): The trained model.
        test_loader (DataLoader): DataLoader for test data.
        criterion (nn.Module): Loss function.

    Returns:
        float: Overall Mean Squared Error (MSE) on the test set.
    """
    model.eval()
    total_mse = 0.0  # Initialize total MSE for all batches
    total_smape = 0.0  # Initialize total sMAPE for all batches

    batch_count = 0


    with torch.no_grad():
        for checkerboard, displacement in test_loader:
            # Forward pass to get predictions
            predicted_displacements = model(checkerboard)

            # Calculate batch MSE
            batch_mse = criterion(predicted_displacements, displacement).item()  # Compute MSE loss for the batch
            total_mse += batch_mse

            # Calculate batch sMAPE
            batch_smape = smape(displacement, predicted_displacements).item() # sMAPE
            total_smape += batch_smape

            batch_count += 1

            # Display results for the first batch
            if batch_count == 1:
                print("\nCheckerboard Input:")
                print(checkerboard[0][0].numpy())  # Show first checkerboard in the batch
                print("\nPredicted Displacement (First 5 Nodes):")
                print(predicted_displacements[0][:5].numpy())  # Predicted displacement for first 5 nodes
                print("\nGround Truth Displacement (First 5 Nodes):")
                print(displacement[0][:5].numpy())  # Ground truth displacement for first 5 nodes

    # Calculate and print overall MSE
    overall_mse = total_mse / batch_count
    overall_smape = total_smape / batch_count

    print(f"Overall Mean Squared Error (MSE) on Test Set: {overall_mse:.10f}")
    print(f"Overall Symmetric Mean Absolute Percentage Error (sMAPE) on Test Set: {overall_smape * 100:.10f}%")
    return overall_mse

# 9. Main Function
def main():
    """
    Main function to load data, train the model, and evaluate it.

    Steps:
    1. Load data from the specified folder.
    2. Create the model and initialize training components.
    3. Train the model with early stopping.
    4. Evaluate the model on the test set.
    5. Save the trained model.
    """
    ### Change the path to your local data directory
    data_path1 = r"C:\Users\Lenovo\Desktop\CSE 583 Software Development for Data Scientists\Project\Dataset1_Random_Board\Dataset1_Random_Board"


    # Create DataLoaders
    print("Loading data...")
    train_loader, val_loader, test_loader, _ = create_data_loaders(
        base_folder=data_path1,
        load_files=("checkerboard", "displacements")
    )

    # Model, Loss, and Optimizer
    input_channels = 1  # Checkerboard has 1 channel
    num_nodes = 5202  # Number of nodes
    model = create_model(input_channels, num_nodes)
    print("Model created.")

    criterion = nn.MSELoss()  # Loss function
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)  # Optimizer
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.5)  # Reduce LR every 2 epochs

    # Training
    epochs = 10
    patience = 5  # Number of epochs to wait for improvement before stopping early
    print("Starting training...")
    train_losses, val_losses = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        epochs=epochs,
        patience=patience
    )
    print(
        f"Training completed. The last training loss is: {train_losses[-1]:.10f}, "
        f"and the last validation loss is: {val_losses[-1]:.10f}."
    )
    # Testing and Evaluation
    print("Evaluating model on test set...")
    evaluate_model(
        model=model,
        test_loader=test_loader,
        criterion=criterion
    )
    print("Evaluation completed.")

if __name__ == "__main__":
    main()


def train_save_gui(data_path):
    # Create DataLoaders
    print("Loading data...")
    train_loader, val_loader, _, _ = create_data_loaders(
        base_folder=data_path,
        load_files=("checkerboard", "displacements")
    )

    # Model, Loss, and Optimizer
    input_channels = 1  # Checkerboard has 1 channel
    num_nodes = 5202  # Number of nodes
    model = create_model(input_channels, num_nodes)
    print("Model created.")

    criterion = nn.MSELoss()  # Loss function
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)  # Optimizer
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.5)  # Reduce LR every 2 epochs

    # Training
    epochs = 10
    patience = 5  # Number of epochs to wait for improvement before stopping early
    print("Starting training...")
    train_losses, val_losses = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        epochs=epochs,
        patience=patience
    )
    print(
        f"Training completed. The last training loss is: {train_losses[-1]:.10f}, "
        f"and the last validation loss is: {val_losses[-1]:.10f}."
    )
    #create the saved_model folder and save the trained model
    save_dir = Path(data_path) / "saved_model"
    save_dir.mkdir(parents=True, exist_ok=True)  # Create the path if not exist
    save_path = save_dir / "trained_displacement_predictor_full_model.pth"  # Model name

    torch.save(model, save_path)
    print(f"Trained model has been saved to {save_path}.")


### Evaluation_GUI part
def create_test_loader(test_data_path, load_files=("checkerboard", "displacements"), batch_size=1):
    """
    Create a DataLoader using the entire dataset from test_data_path.

    Args:
        test_data_path (str): Path to the folder containing the test data.
        load_files (tuple): Names of the files to load (default: ("checkerboard", "displacements")).
        batch_size (int): Batch size for the DataLoader.

    Returns:
        DataLoader: DataLoader for the entire dataset in test_data_path.
    """
    # Load the entire dataset from the specified path
    loaded_data = load_all_npy_files(test_data_path, load_files, skip_missing=True)
    checkerboards = loaded_data["checkerboard"]
    displacements = loaded_data["displacements"]

    # Create a dataset using the entire loaded data
    full_dataset = CheckerboardDataset(checkerboards, displacements)
    normalized_dataset = NormalizedDataset(full_dataset)

    # Create a DataLoader for the entire dataset
    test_loader = DataLoader(normalized_dataset, batch_size=batch_size, shuffle=False)

    return test_loader


def evaluate_model_gui(model, test_loader, criterion, pred_save_dir):
    """
    Evaluate the model on the test set and save predictions.

    Args:
        model (nn.Module): The trained model.
        test_loader (DataLoader): DataLoader for test data.
        criterion (nn.Module): Loss function.
        pred_save_dir (str): Directory to save the predicted displacements.

    Returns:
        float: Overall Mean Squared Error (MSE) on the test set.
    """
    model.eval()
    total_mse = 0.0  # Initialize total MSE for all batches
    total_smape = 0.0  # Initialize total sMAPE for all batches

    batch_count = 0

    os.makedirs(pred_save_dir, exist_ok=True)  # Ensure the directory exists

    with torch.no_grad():
        for batch_idx, (checkerboard, displacement) in enumerate(test_loader):
            # print(f"Processing batch {batch_idx}, Checkerboard size: {checkerboard.size()}")

            # Forward pass to get predictions
            predicted_displacements = model(checkerboard)

            # Save predictions to individual files
            batch_dir = os.path.join(pred_save_dir, f"Simulation_{batch_idx}")
            os.makedirs(batch_dir, exist_ok=True)

            # Save as .npy
            npy_path = os.path.join(batch_dir, "pred_displacements.npy")
            np.save(npy_path, predicted_displacements.cpu().numpy())


            # Save as CSV
            flat_predictions = predicted_displacements.cpu().numpy().reshape(-1, 3)
            csv_path = os.path.join(batch_dir, "pred_displacements.csv")
            np.savetxt(csv_path, flat_predictions, delimiter=",")


            # Calculate batch MSE
            batch_mse = criterion(predicted_displacements, displacement).item()  # Compute MSE loss for the batch
            total_mse += batch_mse

            # Calculate batch sMAPE
            batch_smape = smape(displacement, predicted_displacements).item()  # sMAPE
            total_smape += batch_smape

            batch_count += 1

            # Display results for the first batch
            if batch_count == 1:
                print("\nCheckerboard Input:")
                print(checkerboard[0][0].numpy())  # Show first checkerboard in the batch
                print("\nPredicted Displacement (First 5 Nodes):")
                print(predicted_displacements[0][:5].numpy())  # Predicted displacement for first 5 nodes
                print("\nGround Truth Displacement (First 5 Nodes):")
                print(displacement[0][:5].numpy())  # Ground truth displacement for first 5 nodes

    # Calculate and print overall MSE
    overall_mse = total_mse / batch_count
    overall_smape = total_smape / batch_count

    print(f"Overall Mean Squared Error (MSE) on Test Set: {overall_mse:.10f}")
    print(f"Overall Symmetric Mean Absolute Percentage Error (sMAPE) on Test Set: {overall_smape * 100:.10f}%")
    return overall_mse


def load_and_evaluate_model_gui(model_path, test_data_path, pred_save_dir):
    # Load the model
    model = torch.load(model_path)
    model.eval()
    print("Model loaded successfully.")

    # Use the entire test data as the DataLoader
    test_loader = create_test_loader(test_data_path, batch_size=1)
    print("Test data loaded successfully.")

    # Define loss function
    criterion = nn.MSELoss()

    # Evaluate the model
    print("Evaluating the model...")
    evaluate_model_gui(
        model=model,
        test_loader=test_loader,
        criterion=criterion,
        pred_save_dir=pred_save_dir
    )
    print("Evaluation completed, Predicted Displacements saved.")
