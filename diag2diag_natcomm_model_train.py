import pickle
import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import time
import datetime
import h5py
import glob
import scipy.signal
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, default_collate
from torch.nn.utils.rnn import pad_sequence


def mycollate_fn(batch):
    """
    Custom collate function used by DataLoader to handle variable-length sequences.
    
    Args:
        batch (list of tuples): Each tuple contains (X_tensor, Y_tensor).
    
    Returns:
        tuple: Padded inputs (X_padded) and padded targets (Y_padded), both batched.
    """
    X_batch, Y_batch = zip(*batch)  # Unpack dataset tuples
    # Pad each sequence in the batch to the same length with zeros
    X_padded = pad_sequence(X_batch, batch_first=True, padding_value=0)
    Y_padded = pad_sequence(Y_batch, batch_first=True, padding_value=0)
    return X_padded, Y_padded


# List of diagnostics to collect from the HDF5 file
diag_list = [
    'ts_core_density',
    'ts_core_density_error',
    'ts_core_temperature',
    'ts_core_temperature_error',
    'cer_ti',
    'co2_density',
    'ece_cali',
    'magnetics',
    'mse'
]

# Pre-computed statistics (means and std) for various diagnostics
stats = {
    'ts_core_density': {
        'mean': 1.863966361105754e+19,
        'std': 2.162379391327157e+19
    },
    'ts_core_temperature': {
        'mean': 952.0972,
        'std': 852.1549
    },
    'ts_core_density_error': {
        'mean': 1.863966361105754e+19,
        'std': 2.162379391327157e+19
    },
    'ts_core_temperature_error': {
        'mean': 952.0972,
        'std': 852.1549
    },
    'ece_cali': {
        'mean': 2.5278146,
        'std': 1.2058376
    },
    'co2_density': {
        'mean': 86475240000000.0,
        'std': 3505041700000.0
    },
    'mse': {
        'mean': 3.052,
        'std': 8.720
    },
    'ece_cali_1dev': {
        'mean': 1.0057740156937882e-06,
        'std': 7.76780096672238e-05
    },
    'ece_cali_2dev': {
        'mean': 2.283253742011691e-12,
        'std': 1.7506640532075264e-07
    },
    'co2_density_1dev': {
        'mean': 9400960.77142386,
        'std': 1057202263.1658885
    },
    'co2_density_2dev': {
        'mean': -77.52600344843847,
        'std': 782965.4136278953
    },
    'magnetics': {
        'mean': -1.9582632,
        'std': 0.008118983
    },
    'cer_fz': {
        'mean': 1.3880545,
        'std': 0.4862455
    },
    'cer_nz': {
        'mean': 1.3880545,
        'std': 0.4862455
    },
    'cer_rot': {
        'mean': 1.3880545,
        'std': 0.4862455
    },
    'cer_ti': {
        'mean': 1.3880545,
        'std': 0.4862455
    }
}


def calculate_derivatives(df, window_size, smooth_len):
    """
    Calculate first and second derivatives of a time series using rolling windows.
    
    Args:
        df (pd.DataFrame): DataFrame with a time-based index and a single column of data.
        window_size (int/float): Size of the rolling window (in ms or index steps).
        smooth_len (int/float): Window size for smoothing prior to taking the derivative.
    
    Returns:
        tuple: (first_derivative_df, second_derivative_df)
            DataFrames with suffixes '_1dev' and '_2dev' for the first and second derivatives.
            If the data length is smaller than window sizes, returns 0 arrays.
    """
    # Estimate the number of points in window_size based on average time step
    window_size = np.int32(window_size / df.index.to_series().diff().fillna(method='bfill').mean())
    smooth_len = np.int32(smooth_len / df.index.to_series().diff().fillna(method='bfill').mean())

    if window_size != 0 or smooth_len != 0:
        # Time difference handling for datetime-indexed data
        if isinstance(df.index, pd.DatetimeIndex):
            time_diff = df.index.to_series().diff().dt.total_seconds().fillna(method='bfill')
            rolling_time_diff = time_diff.rolling(window=window_size, min_periods=1).sum()
        else:
            # For a non-datetime index, use the window size directly
            rolling_time_diff = window_size

        # Smooth the signal over smooth_len points and take difference over window_size
        rolling_diff = (
            df.rolling(smooth_len).mean()
              .diff(window_size)
              .fillna(method='bfill')
        )
        
        # Compute the first derivative
        first_derivative = rolling_diff / rolling_time_diff

        # Compute second derivative by taking derivative of the first derivative
        rolling_diff_first_derivative = (
            first_derivative.rolling(smooth_len).mean()
            .diff(window_size)
            .fillna(method='bfill')
        )
        second_derivative = rolling_diff_first_derivative / rolling_time_diff

        return first_derivative.add_suffix('_1dev'), second_derivative.add_suffix('_2dev')
    else:
        print('Data length smaller than the windows! Returning 0s.')
        return np.zeros(df.shape), np.zeros(df.shape)


def process_df(df):
    """
    Clean up a DataFrame by dropping columns of all NaNs and averaging repeated measurements.
    
    Args:
        df (pd.DataFrame): Original DataFrame.
    
    Returns:
        pd.DataFrame: Processed DataFrame with no all-NaN columns and averaged entries for duplicate times.
    """
    df.dropna(axis=1, how='all', inplace=True)  # Drop columns that are entirely NaN
    df = df.groupby(level=0).mean()             # Aggregate duplicate indices (times)
    return df


def collect_data(data_path, shotn, diag_list):
    """
    Read and process data from an HDF5 file for a given shot number.
    
    Args:
        data_path (str): Directory containing the HDF5 files.
        shotn (int): Shot number to process.
        diag_list (list of str): List of diagnostics to read from the file.
    
    Returns:
        dict: Dictionary of DataFrames keyed by diagnostic name. Additional derivatives
              (e.g., '_1dev', '_2dev') are also computed for specified diagnostics.
    """
    all_data = dict()
    h5_file_path = os.path.join(data_path, f'{shotn}.h5')
    
    with h5py.File(h5_file_path, 'r') as hf:
        for diagname in diag_list:
            print(diagname)
            
            # Read axes and data for the diagnostic
            sig_time = np.asarray(hf[diagname]['axis1'], dtype=np.float32)
            sig_name = np.asarray(hf[diagname]['block0_items'], dtype=str)
            sig_data = np.asarray(hf[diagname]['block0_values'], dtype=np.float32)
            
            # Create a DataFrame from the signal data
            all_data[diagname] = pd.DataFrame(
                sig_data,
                columns=[f'{diagname}_{x}' for x in sig_name],
                index=sig_time
            )
            # Process and clean the DataFrame
            all_data[diagname] = process_df(all_data[diagname])
            
            # Optionally calculate derivatives for specific diagnostics
            if diagname in ['co2_density', 'ece_cali']:
                (all_data[f'{diagname}_1dev'],
                 all_data[f'{diagname}_2dev']) = calculate_derivatives(
                     all_data[diagname], window_size=1, smooth_len=1
                 )
                
    return all_data


def extract_features(old_dfs, t, stats, res='low', augment=True):
    """
    Resample and normalize diagnostic DataFrames, select channels of interest, 
    and concatenate them into input (X) and target (Y) arrays.
    
    Args:
        old_dfs (dict): Dictionary of DataFrames keyed by diagnostic names.
        t (list or tuple): Start and end times (e.g., [0, 5000]) for data selection.
        stats (dict): Dictionary containing mean and std for each diagnostic for normalization.
        res (str): 'low' uses TS timing, 'high' uses a 1MHz sampling from t[0] to t[1].
        augment (bool): If True, data is augmented using TS uncertainty (upper and lower bounds).
    
    Returns:
        tuple: (X, Y) where
               X is a 2D array of concatenated input features,
               Y is a 2D array (or 3D if augment=True) of concatenated output features.
    """
    new_dfs = dict()
    
    # Determine reference time based on resolution
    if res == 'low':
        # Use TS density's index as reference time
        ref_time = old_dfs['ts_core_density'].index.to_numpy()
        ref_time = ref_time[(ref_time >= t[0]) & (ref_time <= t[1])]
    elif res == 'high':
        # Sample in 1 kHz steps between t[0] and t[1]
        ref_time = np.linspace(t[0], t[1], np.int32((t[1] - t[0]) * 1000))
    
    # Resample and normalize each diagnostic
    for diagname in old_dfs.keys():
        old_dfs[diagname] = old_dfs[diagname].reindex(ref_time, method='ffill')
        new_dfs[diagname] = (old_dfs[diagname] - stats[diagname]['mean']) / stats[diagname]['std']
    
    # Select magnetics channels of interest
    mag_chn = [f'magnetics_mpi.{chn}139' for chn in ['1a','2b','3a','3b','4a','4b','5a','5b']]
    new_dfs['magnetics'] = new_dfs['magnetics'][mag_chn]
    
    # Select CER channels of interest
    cer_chn = []
    for x in ['t', 'v']:
        for y in np.arange(1, 30):
            cer_chn.append(f'cer_ti_q.ti.{x}{y:02d}')
    new_dfs['cer_ti'] = new_dfs['cer_ti'][cer_chn]
    
    # Select ECE and ECE derivative channels
    ece_chn = [f'ece_cali_{x+1:02d}' for x in range(40)]
    new_dfs['ece_cali'] = new_dfs['ece_cali'][ece_chn]
    
    ece_chn = [f'ece_cali_{x+1:02d}_1dev' for x in range(40)]
    new_dfs['ece_cali_1dev'] = new_dfs['ece_cali_1dev'][ece_chn]
    
    ece_chn = [f'ece_cali_{x+1:02d}_2dev' for x in range(40)]
    new_dfs['ece_cali_2dev'] = new_dfs['ece_cali_2dev'][ece_chn]
    
    # Select first 38 MSE channels
    new_dfs['mse'] = new_dfs['mse'].iloc[:, :38]
    
    # Concatenate input features
    X = np.hstack([
        new_dfs[x].values for x in [
            'co2_density', 'co2_density_1dev', 'co2_density_2dev',
            'ece_cali', 'ece_cali_1dev', 'ece_cali_2dev',
            'cer_ti', 'mse', 'magnetics'
        ]
    ])
    
    # Concatenate output features (TS density and TS temperature)
    Y = np.hstack([
        new_dfs[x].values for x in ['ts_core_density', 'ts_core_temperature']
    ])
    
    # Data augmentation based on TS error
    if augment:
        y_err_1 = np.hstack([
            new_dfs[x].values - new_dfs[f'{x}_error'].values
            for x in ['ts_core_density', 'ts_core_temperature']
        ])
        y_err_2 = np.hstack([
            new_dfs[x].values + new_dfs[f'{x}_error'].values
            for x in ['ts_core_density', 'ts_core_temperature']
        ])
        # Stack augmented data: (original, lower bound, upper bound)
        Y = np.vstack([Y, y_err_1, y_err_2])
        X = np.vstack([X, X, X])
    
    return X, Y


def generate_datasets(data_dir, shot_list):
    """
    Generate X, Y pairs for each shot in shot_list by collecting
    and extracting features. (Dummy data is used here as an example.)
    
    Args:
        data_dir (str): Directory where shot data is stored.
        shot_list (list or np.array): List of shot numbers.
    
    Returns:
        tuple: (X_data, Y_data) where each is a list of tensors. 
               Each element of X_data/Y_data corresponds to a shot.
    """
    X_data, Y_data = [], []
    
    for shotn in tqdm(shot_list):
        # If needed, uncomment and use real data collection:
        # dfs = collect_data('../data/diag2diag-data/', shotn, diag_list)
        # X_discharge, Y_discharge = extract_features(dfs, [0, 5000], stats, 'low', True)
        
        # For demonstration, randomly generate data shapes:
        ln_list = [900, 700, 1000]
        ln = random.choice(ln_list)
        X_discharge = np.random.randn(ln, 236)  # 236 input features
        Y_discharge = np.random.randn(ln, 80)   # 80 output features
        
        # Convert NumPy arrays to PyTorch tensors
        X_data.append(torch.tensor(X_discharge, dtype=torch.float32))
        Y_data.append(torch.tensor(Y_discharge, dtype=torch.float32))
    
    return X_data, Y_data


def validate_model(model, val_loader, criterion, device):
    """
    Evaluate the model on the validation set and compute average loss.
    
    Args:
        model (nn.Module): Trained model to evaluate.
        val_loader (DataLoader): DataLoader for the validation set.
        criterion (nn.Module): Loss function.
        device (torch.device): Device ('cpu' or 'cuda') for computation.
    
    Returns:
        tuple: (all_predictions, all_targets) as concatenated tensors from the entire validation set.
    """
    model.eval()  # Set model to evaluation mode
    val_loss = 0
    all_predictions, all_targets = [], []

    with torch.no_grad():  # No gradient computation needed
        for X_batch, Y_batch in val_loader:
            X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)
            predictions = model(X_batch)
            loss = criterion(predictions, Y_batch)
            val_loss += loss.item()

            all_predictions.append(predictions.cpu())
            all_targets.append(Y_batch.cpu())

    val_loss /= len(val_loader)
    print(f"Validation Loss: {val_loss:.6f}")
    return torch.cat(all_predictions), torch.cat(all_targets)


# ------------------ Example usage of data collection and feature extraction ------------------
X_data = []
Y_data = []

shotn = 174823
dfs = collect_data('../data/diag2diag-data/', shotn, diag_list)
x, y = extract_features(dfs, [0, 5000], stats, 'low', True)
X_data.append(x)
Y_data.append(y)

# Example loading shot list
shot_list = np.loadtxt('diag2diag_shotlist.txt', dtype=np.int32)

# Generate datasets for multiple shots (using random data in this example)
data_dir = '../data/diag2diag-data/'
X_data, Y_data = generate_datasets(data_dir, shot_list)


# ------------------ Creating train/validation split and DataLoaders ------------------
train_discharges = 2000  # Number of training discharges
train_dataset = list(zip(X_data[:train_discharges], Y_data[:train_discharges]))
val_dataset = list(zip(X_data[train_discharges:], Y_data[train_discharges:]))

# Use the custom collate function to handle variable-length sequences
train_loader = DataLoader(
    train_dataset,
    batch_size=1024,
    shuffle=True,
    num_workers=0,
    collate_fn=mycollate_fn
)
val_loader = DataLoader(
    val_dataset,
    batch_size=1024,
    shuffle=False,
    num_workers=0,
    collate_fn=mycollate_fn
)

# Quick shape check
for X_batch, Y_batch in train_loader:
    print(f"X batch shape: {X_batch.shape}")
    print(f"Y batch shape: {Y_batch.shape}")
    break


# ---------------------------- Define MLP Model ----------------------------
class MLP(nn.Module):
    """
    A simple Multi-Layer Perceptron (MLP) architecture.
    """
    def __init__(self, input_size, hidden_layers, output_size, dropout_rate):
        """
        Args:
            input_size (int): Number of input features.
            hidden_layers (list of int): Sizes of each hidden layer.
            output_size (int): Number of output features.
            dropout_rate (float): Probability of dropout.
        """
        super(MLP, self).__init__()
        layers = []
        prev_size = input_size
        
        # Create hidden layers
        for layer_size in hidden_layers:
            layers.append(nn.Linear(prev_size, layer_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            prev_size = layer_size
        
        # Final output layer
        layers.append(nn.Linear(prev_size, output_size))
        
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        """
        Forward pass of the MLP.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_size).
        
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, output_size).
        """
        return self.network(x)


# ---------------------------- Model Training ----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MLP(236, [512, 256, 128], 80, 0.076).to(device)
optimizer = optim.AdamW(model.parameters(), lr=1e-3)
criterion = nn.L1Loss()

def train_model(model, train_loader, val_loader, optimizer, criterion,
                max_epochs=500, patience=20):
    """
    Train the MLP model with early stopping based on validation loss.
    
    Args:
        model (nn.Module): The neural network model to train.
        train_loader (DataLoader): DataLoader for the training set.
        val_loader (DataLoader): DataLoader for the validation set.
        optimizer (torch.optim.Optimizer): Optimizer for model parameters.
        criterion (nn.Module): Loss function.
        max_epochs (int): Maximum number of epochs to train.
        patience (int): Number of consecutive epochs without improvement before stopping.
    
    Returns:
        None
    """
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(max_epochs):
        # Set model to training mode
        model.train()
        train_loss = 0
        
        for X_batch, Y_batch in train_loader:
            # The custom collate_fn returns batched data already, so just move them to device
            X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)
            
            # Zero out gradients
            optimizer.zero_grad()
            
            # Forward pass
            predictions = model(X_batch)
            
            # Compute loss
            loss = criterion(predictions, Y_batch)
            
            # Backpropagation
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # Compute average training loss
        train_loss /= len(train_loader)
        
        # Validation step
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X_batch, Y_batch in val_loader:
                X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)
                predictions = model(X_batch)
                loss = criterion(predictions, Y_batch)
                val_loss += loss.item()
        val_loss /= len(val_loader)
        
        print(f"Epoch {epoch+1}: Train Loss = {train_loss:.6f}, Val Loss = {val_loss:.6f}")
        
        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # Save the best model
            torch.save(model.state_dict(), "best_mlp_model.pth")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered")
                break


# Train the model
train_model(model, train_loader, val_loader, optimizer, criterion)

# Validate (evaluate) the model on the validation set
val_predictions, val_targets = validate_model(model, val_loader, criterion, device)