import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from preprocess import get_loaders

# Parameters
save_path = "../data/preprocessed_data.pt"
batch_size = 16

# Get DataLoaders
train_loader, val_loader = get_loaders(save_path, batch_size)

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        # Define convolutional layers
        self.conv1 = nn.Conv1d(in_channels=8, out_channels=16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, padding=1)

        # Define fully connected layers
        self.fc1 = nn.Linear(in_features=32 * 36, out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=8)  # Adjusted for 7 classes

    def forward(self, x):
        x = self.conv1(x)
        x = nn.ReLU()(x)
        x = self.conv2(x)
        x = nn.ReLU()(x)

        # Flatten the tensor
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        return x

# Create an instance of the model
model = CNN()

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    for inputs, labels in train_loader:
        optimizer.zero_grad()  # Clear gradients

        # Forward pass
        outputs = model(inputs)

        # Compute loss
        loss = criterion(outputs, labels)

        # Backward pass and update weights
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {running_loss / len(train_loader):.4f}")

# Validation loop
model.eval()  # Set model to evaluation mode
correct, total = 0, 0

with torch.no_grad():  # Disable gradient calculations
    for inputs, labels in val_loader:
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print(f"Validation Accuracy: {accuracy:.2f}%")

# Save the model
torch.save(model.state_dict(), "emg_cnn_model.pth")
