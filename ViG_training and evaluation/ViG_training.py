#!/usr/bin/env python
# coding: utf-8

# # ViG Model Training

# ## Load Dataset

import numpy as np
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


# Define paths to the directories where you saved your datasets
dataset_dir = './FYP/DataSet/Preprocessed_DFU'

# Load augmented train set
train_data_augmented = np.load(os.path.join(dataset_dir, 'train_data.npy'))
train_labels_augmented = np.load(os.path.join(dataset_dir, 'train_labels_encoded.npy'))

# Load augmented validation set
val_data_augmented = np.load(os.path.join(dataset_dir, 'val_data.npy'))
val_labels_augmented = np.load(os.path.join(dataset_dir, 'val_labels_encoded.npy'))

# Load test set
test_data = np.load(os.path.join(dataset_dir, 'test_data.npy'))
test_labels = np.load(os.path.join(dataset_dir, 'test_labels_encoded.npy'))


# Check the shapes of the loaded datasets
print("Shape of augmented train data array:", train_data_augmented.shape)
print("Shape of augmented train labels array:", train_labels_augmented.shape)
print("Shape of augmented validation data array:", val_data_augmented.shape)
print("Shape of augmented validation labels array:", val_labels_augmented.shape)
print("Shape of test data array:", test_data.shape)
print("Shape of test labels array:", test_labels.shape)


from torchvision import transforms
# Define transformations
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Adjust mean and std if needed
])



# Convert numpy arrays to PyTorch tensors and transpose the dimensions
train_data_augmented = torch.tensor(train_data_augmented.transpose((0, 3, 1, 2)))  # Transpose from [height, width, channels] to [channels, height, width]
train_labels_augmented = torch.LongTensor(train_labels_augmented)
val_data_augmented = torch.tensor(val_data_augmented.transpose((0, 3, 1, 2)))  # Transpose from [height, width, channels] to [channels, height, width]
val_labels_augmented = torch.LongTensor(val_labels_augmented)
test_data = torch.tensor(test_data.transpose((0, 3, 1, 2)))  # Transpose from [height, width, channels] to [channels, height, width]
test_labels = torch.LongTensor(test_labels)


# Create DataLoader
train_dataset = TensorDataset(train_data_augmented, train_labels_augmented)
val_dataset = TensorDataset(val_data_augmented, val_labels_augmented)
test_dataset = TensorDataset(test_data, test_labels)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)
test_loader = DataLoader(test_dataset, batch_size=32)






# ## Load and Train Model

import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from vig import vig_ti_224_gelu 
import timm


# Define the path to the ViG model directory
vig_model_path = "./FYP/Models/Vision_GNN-main/Vision_GNN-main"



sys.path.append(vig_model_path)


# Instantiate the ViG model
model = vig_ti_224_gelu(pretrained=True)  # Set pretrained=True to use pretrained weights


# Move the model to the appropriate device (CPU or GPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)


# Loss Function and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


# Training Loop
def train_model(model, criterion, optimizer, train_loader, val_loader, num_epochs=10):
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
            _, predicted_train = torch.max(outputs, 1)
            total_train += labels.size(0)
            correct_train += (predicted_train == labels).sum().item()
        
        epoch_loss = running_loss / len(train_loader.dataset)
        train_losses.append(epoch_loss)
        train_accuracy = correct_train / total_train
        train_accuracies.append(train_accuracy)

        # Validation
        model.eval()
        correct_val = 0
        total_val = 0
        val_running_loss = 0.0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted_val = torch.max(outputs, 1)
                total_val += labels.size(0)
                correct_val += (predicted_val == labels).sum().item()
                loss = criterion(outputs, labels)
                val_running_loss += loss.item() * inputs.size(0)
        
        val_loss = val_running_loss / len(val_loader.dataset)
        val_losses.append(val_loss)
        val_accuracy = correct_val / total_val
        val_accuracies.append(val_accuracy)

        print(f"Epoch [{epoch+1}/{num_epochs}], "
              f"Train Loss: {epoch_loss:.4f}, "
              f"Train Accuracy: {train_accuracy:.2%}, "
              f"Val Loss: {val_loss:.4f}, "
              f"Val Accuracy: {val_accuracy:.2%}")

    return train_losses, val_losses, train_accuracies, val_accuracies


# Train the model
train_losses, val_losses, train_accuracies, val_accuracies = train_model(model, criterion, optimizer, train_loader, val_loader, num_epochs=10)



# # Save Trained model weights and training metrics

import os

# Define paths for saving the model weights and training metrics
model_weights_path = './FYP/Models/ViG_model/model_weights'
training_metrics_path = './FYP/Models/ViG_model/training_metrics'


# Create directories
os.makedirs(model_weights_path, exist_ok=True)
os.makedirs(training_metrics_path, exist_ok=True)

# Define filenames for saving
model_weights_file = os.path.join(model_weights_path, 'trained_model.pth')
training_metrics_file = os.path.join(training_metrics_path, 'training_metrics.npz')


# Save model weights to the specified path
torch.save(model.state_dict(), model_weights_file)

# Save training metrics to the specified path
np.savez(training_metrics_file,
         train_losses=train_losses,
         val_losses=val_losses,
         train_accuracies=train_accuracies,
         val_accuracies=val_accuracies)
