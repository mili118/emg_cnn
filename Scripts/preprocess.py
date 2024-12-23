import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader

def load_all_files(data_dir):
    """
    Traverse the directory to find all .txt files, filter out invalid rows,
    and combine them into a single DataFrame.
    """
    all_data = []

    # Traverse directories to find .txt files
    for root, _, files in os.walk(data_dir):
        for file in files:
            if file.endswith(".txt"):
                file_path = os.path.join(root, file)
                print(f"Processing file: {file_path}")  # Debugging

                try:
                    # Read file into pandas DataFrame
                    data = pd.read_csv(file_path, delim_whitespace=True, header=0, on_bad_lines="skip")
                    
                    # Convert all data to numeric, coercing errors to NaN
                    data = data.apply(pd.to_numeric, errors="coerce")
                    
                    # Drop rows with NaN values (non-numeric rows)
                    data = data.dropna()
                    
                    # Ensure correct number of columns
                    if data.shape[1] != 10:
                        print(f"Skipping file {file_path}: Unexpected number of columns ({data.shape[1]})")
                        continue

                    # Append valid data
                    all_data.append(data)

                except Exception as e:
                    print(f"Error processing file {file_path}: {e}")

    # Check if any files were found
    if not all_data:
        raise ValueError(f"No valid .txt files found in directory: {data_dir}")

    # Combine all DataFrames into one
    combined_data = pd.concat(all_data, ignore_index=True)
    return combined_data

def preprocess_data(data):
    # Extract features (columns 1-8) and labels (column 9)
    features = data.iloc[:, 1:9].values
    labels = data.iloc[:, 9].values

    # Debugging: Check unique labels
    print("Unique labels before filtering:", np.unique(labels))

    # Filter out invalid labels if necessary
    valid_indices = labels <= 6  # Adjust based on valid label range
    '''
    features = features[valid_indices]
    labels = labels[valid_indices]
    '''
    
    # Normalize features to range [0, 1]
    features = (features - np.min(features, axis=0)) / (np.max(features, axis=0) - np.min(features, axis=0) + 1e-8)

    # Segment data into fixed-size windows
    window_size = 36
    segments, segment_labels = [], []
    for i in range(0, len(features) - window_size + 1, window_size):
        segments.append(features[i:i + window_size].T)
        segment_labels.append(labels[i + window_size - 1])

    if len(segments) == 0 or len(segment_labels) == 0:
        raise ValueError("Segmentation produced no samples. Check the window size and input data.")

    inputs = torch.tensor(np.array(segments), dtype=torch.float32)
    targets = torch.tensor(np.array(segment_labels), dtype=torch.long)

    # Debugging: Check unique labels in targets
    print("Unique labels after filtering:", targets.unique().tolist())

    return inputs, targets

def save_preprocessed_data(data_dir, save_path):
    """
    Preprocess the raw data and save the preprocessed tensors to disk.
    """
    combined_data = load_all_files(data_dir)
    inputs, targets = preprocess_data(combined_data)
    torch.save((inputs, targets), save_path)
    print(f"Preprocessed data saved to {save_path}")

def load_preprocessed_data(save_path, batch_size):
    """
    Load preprocessed data from disk and prepare DataLoaders.
    """
    inputs, targets = torch.load(save_path)
    print(f"Preprocessed data loaded from {save_path}")

    # Create a dataset from the inputs and targets
    dataset = TensorDataset(inputs, targets)

    # Split into training and validation sets (80-20 split)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    # Create DataLoaders for training and validation
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader

# Add a function to return train_loader and val_loader
def get_loaders(save_path, batch_size):
    return load_preprocessed_data(save_path, batch_size)


if __name__ == "__main__":
    # Base directory containing raw data
    data_dir = "../Data/rawdata/"
    save_path = "../Data/preprocessed_data.pt"  # Path to save preprocessed data
    batch_size = 16

    # Step 1: Preprocess and save the data (run once)
    save_preprocessed_data(data_dir, save_path)

    # Step 2: Load preprocessed data and prepare DataLoaders
    train_loader, val_loader = load_preprocessed_data(save_path, batch_size)

    print(f"Train batches: {len(train_loader)}, Validation batches: {len(val_loader)}")

