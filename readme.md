# EMG Gesture Classification with CNN

## Overview
This project is a deep learning-based approach to classify static hand gestures using surface Electromyographic (EMG) data. The gestures are detected based on data collected from a MYO Thalmic bracelet, which records EMG signals from eight sensors placed around the user's forearm. The model utilizes a Convolutional Neural Network (CNN) for efficient feature extraction and classification.

## Dataset
The dataset used for training and evaluation is the **EMG Data for Gestures** dataset from the UCI Machine Learning Repository.
- **Source**: [UCI EMG Data for Gestures Dataset](https://archive.ics.uci.edu/dataset/481/emg+data+for+gestures)
- **Description**:
  - Contains raw EMG data collected from 36 subjects.
  - Each subject performed a series of six to seven basic static hand gestures, with each gesture lasting 3 seconds followed by a 3-second pause.
  - Data includes 8 EMG channels and gesture labels.
  - Gesture classes:
    - 0: Unmarked data
    - 1: Hand at rest
    - 2: Hand clenched in a fist
    - 3: Wrist flexion
    - 4: Wrist extension
    - 5: Radial deviations
    - 6: Ulnar deviations
    - 7: Extended palm (the gesture was not performed by all subjects).



## Project Workflow

### 1. **Data Preprocessing**
The raw EMG data is preprocessed for training and evaluation:
- **File Traversal**: The raw `.txt` files are loaded and combined into a single dataset.
- **Filtering**: Invalid rows and files with incorrect formats are ignored.
- **Normalization**: The EMG signal values are normalized to the range [0, 1].
- **Segmentation**: The data is segmented into fixed-size windows (36 time steps).
- **Preprocessed Data Storage**: The processed data is saved as a `.pt` file using PyTorch.

### 2. **Model Architecture**
The project uses a CNN designed for 1D EMG signal data:
- **Convolutional Layers**:
  - Two 1D convolutional layers to extract spatial features from the 8 EMG channels.
  - Kernel size: 3
  - Padding: 1
  - Output channels: 16 and 32
- **Fully Connected Layers**:
  - One fully connected layer with 128 neurons.
  - Final layer with 7 neurons corresponding to the 7 gesture classes.
- **Activation Function**: ReLU (Rectified Linear Unit)
- **Loss Function**: CrossEntropyLoss
- **Optimizer**: Adam optimizer with a learning rate of 0.001.

### 3. **Model Training**
- **Epochs**: 10
- **Batch Size**: 16
- **Training Data**: 80% of the preprocessed data
- **Validation Data**: 20% of the preprocessed data
- The training loop calculates the loss and updates the model weights using backpropagation.


## Acknowledgments
- The UCI Machine Learning Repository for providing the dataset.
- The developers of PyTorch and other open-source libraries used in this project.

