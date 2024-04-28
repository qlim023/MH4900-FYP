#!/usr/bin/env python
# coding: utf-8

# # ViG Model Evaluation

# ## Load Dataset

import numpy as np
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


# Define paths to the directories where you saved your datasets
dataset_dir = './FYP/DataSet/Preprocessed_DFU'

# Load test set
test_data = np.load(os.path.join(dataset_dir, 'test_data.npy'))
test_labels = np.load(os.path.join(dataset_dir, 'test_labels_encoded.npy'))


# Check the shapes of the loaded datasets
print("Shape of test data array:", test_data.shape)
print("Shape of test labels array:", test_labels.shape)


from torchvision import transforms
# Define transformations
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Adjust mean and std if needed
])


# Convert numpy arrays to PyTorch tensors and transpose the dimensions
test_data = torch.tensor(test_data.transpose((0, 3, 1, 2)))  # Transpose from [height, width, channels] to [channels, height, width]
test_labels = torch.LongTensor(test_labels)


# Create DataLoader
test_dataset = TensorDataset(test_data, test_labels)
test_loader = DataLoader(test_dataset, batch_size=32)





# ### Plot Training accuracy and loss

import numpy as np

# Load the training metrics
training_metrics_path = './FYP/Models/ViG_model/training_metrics/training_metrics.npz'
training_metrics = np.load(training_metrics_path)

# Extract the training metrics
train_losses = training_metrics['train_losses']
val_losses = training_metrics['val_losses']
train_accuracies = training_metrics['train_accuracies']
val_accuracies = training_metrics['val_accuracies']


# Number of epochs
num_epochs = len(train_accuracies)


import matplotlib.pyplot as plt
# Plot Accuracy
plt.figure(figsize=(10, 5))
plt.plot(range(1, num_epochs + 1), train_accuracies, label='Training Accuracy')
plt.plot(range(1, num_epochs + 1), val_accuracies, label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()
plt.grid(True)
plt.savefig('ViG_accuracy.png')
plt.show()


# Plot Loss
plt.figure(figsize=(10, 5))
plt.plot(range(1, num_epochs + 1), train_losses, label='Training Accuracy')
plt.plot(range(1, num_epochs + 1), val_losses, label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()
plt.grid(True)
plt.savefig('ViG_loss.png')
plt.show()




# # Evaluation on Test Set

import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from vig import vig_ti_224_gelu 
from torch.utils.data import DataLoader
import timm
from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report

# Define the path to the cached ViG model directory
vig_model_path = "./FYP/Models/Vision_GNN-main/Vision_GNN-main"


sys.path.append(vig_model_path)


# Instantiate the ViG model
model = vig_ti_224_gelu(pretrained=False)  # Set pretrained=True to use pretrained weights


# Load the saved model weights
model_weights_path = './FYP/Models/ViG_model/model_weights/trained_model.pth'
model.load_state_dict(torch.load(model_weights_path))


# Move the model to the appropriate device (CPU or GPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)


# Set the model to evaluation mode
model.eval()


# Define the criterion (e.g., CrossEntropyLoss) if needed
criterion = nn.CrossEntropyLoss()



# Function to evaluate the model on the test dataset
def evaluate_model(model, test_loader, criterion=None):
    model.eval()

    # Initialize variables for TP, FP, TN, FN
    true_positives = 0
    false_positives = 0
    true_negatives = 0
    false_negatives = 0

    # Lists to store predictions and ground truth labels
    all_predictions = []
    all_labels = []

    # Running loss for computing average test loss
    running_loss = 0.0

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            # Convert input tensor to float32
            inputs = inputs.float()

            # Forward pass
            outputs = model(inputs)

            # Compute loss if criterion is provided
            if criterion:
                loss = criterion(outputs, labels)
                running_loss += loss.item() * inputs.size(0)

            # Compute predictions
            _, predicted = torch.max(outputs, 1)

            # Update lists of predictions and labels
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            # Compute TP, FP, TN, FN
            true_positives += ((predicted == 1) & (labels == 1)).sum().item()
            false_positives += ((predicted == 1) & (labels == 0)).sum().item()
            true_negatives += ((predicted == 0) & (labels == 0)).sum().item()
            false_negatives += ((predicted == 0) & (labels == 1)).sum().item()

    # Calculate accuracy
    accuracy = (true_positives + true_negatives) / len(all_labels)

    # Calculate average test loss if criterion is provided
    if criterion:
        test_loss = running_loss / len(all_labels)
        print('Test Loss: {:.4f}'.format(test_loss))

    # Print test accuracy
    print('Test Accuracy: {:.2%}'.format(accuracy))

    # Calculate precision, recall, and F1-score
    precision = true_positives / (true_positives + false_positives)
    recall = true_positives / (true_positives + false_negatives)
    f1_score = 2 * (precision * recall) / (precision + recall)

    # Print precision, recall, and F1-score
    print('Precision: {:.2%}'.format(precision))
    print('Recall: {:.2%}'.format(recall))
    print('F1 Score: {:.2%}'.format(f1_score))

    # Calculate AUC
    auc_score = roc_auc_score(all_labels, all_predictions)

    # Print AUC
    print('AUC: {:.2%}'.format(auc_score))

    # Generate and print confusion matrix
    cm = confusion_matrix(all_labels, all_predictions)
    print('Confusion Matrix:')
    print(cm)

    # Generate and print classification report
    print('Classification Report:')
    print(classification_report(all_labels, all_predictions))


# Call the function to evaluate the model on the test dataset
evaluate_model(model, test_loader, criterion)

