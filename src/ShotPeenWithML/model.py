import os
import pandas as pd
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader, random_split

# 1. Load All Numpy Files Function
def load_all_npy_files(base_folder, num_simulations, load_files=("checkerboard", "displacements"), skip_missing=True):
    """
    Load specified .npy files from multiple simulation folders.

    Args:
        base_folder (str): The base folder containing simulation subfolders.
        num_simulations (int): The number of simulations (Simulation_0 to Simulation_(num_simulations-1)).
        load_files (tuple): Names of the files to load (default: ("checkerboard", "displacements")).
        skip_missing (bool): If True, skip missing files; otherwise, raise an error.

    Returns:
        dict: A dictionary containing loaded data arrays for the specified files.
              Keys are file names, and values are stacked arrays.
    """
    # Initialize dictionaries to store data
    data_dict = {key: [] for key in load_files}

    for i in range(num_simulations):
        simulation_folder = f"Simulation_{i}"  # Dynamically construct folder name
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
    def __init__(self, checkerboards, displacements):
        """
        Args:
            checkerboards (numpy array): Array of checkerboard patterns (batch_size, height, width).
            displacements (numpy array): Array of displacements (batch_size, num_nodes, 3).
        """
        self.checkerboards = checkerboards
        self.displacements = displacements

    def __len__(self):
        return len(self.checkerboards)

    def __getitem__(self, idx):
        checkerboard = self.checkerboards[idx]
        displacement = self.displacements[idx]

        # Add a channel dimension to checkerboard (1 channel) to match with CNN expectations
        checkerboard = torch.tensor(checkerboard, dtype=torch.float32).unsqueeze(0)  # (1, height, width)
        displacement = torch.tensor(displacement, dtype=torch.float32)  # (num_nodes, 3)

        return checkerboard, displacement

class NormalizedDataset(Dataset):
    """
    A wrapper for normalizing datasets. Takes a base dataset and applies normalization to its features.
    """
    def __init__(self, base_dataset):
        self.base_dataset = base_dataset
        self.checkerboards = torch.cat([data[0] for data in base_dataset], dim=0)  # Collect all checkerboards
        self.min_val = self.checkerboards.min()
        self.max_val = self.checkerboards.max()

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        checkerboard, displacement = self.base_dataset[idx]
        normalized_checkerboard = (checkerboard - self.min_val) / (self.max_val - self.min_val)
        return normalized_checkerboard, displacement

# 3. Attention Modules
class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.fc1 = nn.Conv2d(channels, channels // reduction, kernel_size=1)
        self.fc2 = nn.Conv2d(channels // reduction, channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_pool = torch.mean(x, dim=(2, 3), keepdim=True)  # Global average pooling
        max_pool = torch.max(torch.max(x, dim=2, keepdim=True).values, dim=3, keepdim=True).values  # Global max pooling
        scale = self.fc1(avg_pool) + self.fc1(max_pool)
        scale = self.fc2(torch.relu(scale))
        return self.sigmoid(scale) * x

class SpatialAttention(nn.Module):
    def __init__(self):
        super(SpatialAttention, self).__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size=7, padding=3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_pool = torch.mean(x, dim=1, keepdim=True)  # Channel-wise average
        max_pool = torch.max(x, dim=1, keepdim=True).values  # Channel-wise max
        scale = torch.cat([avg_pool, max_pool], dim=1)
        return self.sigmoid(self.conv1(scale)) * x

# 4. CNN Model with Attention
class DisplacementPredictor(nn.Module):
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
def create_data_loaders(base_folder, num_simulations, load_files=("checkerboard", "displacements"), skip_missing=True, batch_size=15):
    loaded_data = load_all_npy_files(base_folder, num_simulations, load_files, skip_missing)
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
    model = DisplacementPredictor(input_channels, num_nodes)
    return model

# 7. Training Function
def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, epochs=10, patience=5):
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
    """Calculate sMAPE for two tensors."""
    numerator = torch.abs(y_true - y_pred)
    denominator = (torch.abs(y_true) + torch.abs(y_pred)) / 2
    smape_value = torch.mean(numerator / denominator) 
    return smape_value
    
def evaluate_model(model, test_loader, criterion):
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
    ### Change the path to your local data directory 
    data_path1 = r"C:\Users\Lenovo\Desktop\CSE 583 Software Development for Data Scientists\Project\Dataset1_Random_Board\Dataset1_Random_Board"
    num_simulations1 = 1531  # change the number of simulatiions to your actual data size 

    # Create DataLoaders
    print("Loading data...")
    train_loader, val_loader, test_loader, loaded_data1 = create_data_loaders(
        base_folder=data_path1,
        num_simulations=num_simulations1,
        load_files=("checkerboard", "displacements")
    )

    # Access the returned data
    checkerboard1 = loaded_data1["checkerboard"]
    displacements1 = loaded_data1["displacements"]

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
    print("Training completed.")

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
