#!/usr/bin/env python
# coding: utf-8

# # ViT Model Training and Evaluation

# ## Loading the data

import numpy as np
import os

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


# Normalize pixel values to the range [-1, 1]
train_data_normalized = (train_data_augmented / 127.5) - 1.0
val_data_normalized = (val_data_augmented / 127.5) - 1.0
test_data_normalized = (test_data / 127.5) - 1.0


print("Shape of augmented train data array:", train_data_normalized.shape)
print("Shape of augmented train labels array:", train_labels_augmented.shape)
print("Shape of augmented validation data array:", val_data_normalized.shape)
print("Shape of augmented validation labels array:", val_labels_augmented.shape)





# # Load and Train Model

import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras import layers, models


# Define the path to the cached ViT model directory
vit_model_path = "./FYP/Models/ViT_model"


# Load the ViT model from the local directory
vit_module = hub.KerasLayer(vit_model_path, trainable=False)


# Wrap the ViT module within a Sequential model
# Convert the vit_module to a Keras layer
vit_model = models.Sequential([
    layers.Lambda(lambda x: vit_module(x))
])



# Define additional layers for classification
num_classes = 2  # Number of classes in your dataset
classification_model = models.Sequential([
    vit_model,
    layers.Flatten(),
    layers.Dense(512, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(num_classes, activation='softmax')
])



# Compile the model
classification_model.compile(optimizer='adam',
                              loss='sparse_categorical_crossentropy',
                              metrics=['accuracy'])


# Train the model
history = classification_model.fit(train_data_normalized, train_labels_augmented,
                                   validation_data=(val_data_normalized, val_labels_augmented),
                                   epochs=10, batch_size=32)



# Evaluate the model on test data
test_loss, test_accuracy = classification_model.evaluate(test_data_normalized, test_labels)
print("Test Loss:", test_loss)
print("Test Accuracy:", test_accuracy)




# # Evaluate Model/ Plots

from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, average_precision_score
from sklearn.metrics import precision_recall_curve, auc, roc_curve


# Plot training and validation accuracy
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()
plt.savefig('ViT_accuracy.png')
plt.show()



# Plot training and validation loss
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.savefig('ViT_loss.png')
plt.show()



# Make Predictions
test_predictions = classification_model.predict(test_data_normalized)


# Calculate Metrics
# Accuracy
accuracy = accuracy_score(test_labels, np.argmax(test_predictions, axis=1))

# Confusion Matrix
conf_matrix = confusion_matrix(test_labels, np.argmax(test_predictions, axis=1))

# Precision, Recall, F1-Score
precision = precision_score(test_labels, np.argmax(test_predictions, axis=1))
recall = recall_score(test_labels, np.argmax(test_predictions, axis=1))
f1 = f1_score(test_labels, np.argmax(test_predictions, axis=1))

# AUC
auc = roc_auc_score(test_labels, test_predictions[:, 1])  # Assuming binary classification

# Average Precision (mAP)
mAP = average_precision_score(test_labels, test_predictions[:, 1])  # Assuming binary classification


# Print Results
print("Accuracy:", accuracy)
print("Confusion Matrix:\n", conf_matrix)
print("Precision:", precision)
print("Recall:", recall)
print("F1-Score:", f1)
print("AUC:", auc)
print("mAP:", mAP)


# Calculate the false positive rate and true positive rate for the ROC curve
fpr, tpr, _ = roc_curve(test_labels, test_predictions[:, 1])  # Assuming binary classification
roc_auc = auc(fpr, tpr)


# Plot ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='red', lw=2, linestyle='--', label='Random Guess')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.savefig('ViT_ROC.png')
plt.show()



# Plot confusion matrix
plt.figure(figsize=(8, 6))
plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.colorbar()
tick_marks = np.arange(len(conf_matrix))
plt.xticks(tick_marks, ['Normal', 'Abnormal'])
plt.yticks(tick_marks, ['Normal', 'Abnormal'])
plt.ylabel('True Label')
plt.xlabel('Predicted Label')

# Add text annotations to each cell
thresh = conf_matrix.max() / 2.
for i in range(conf_matrix.shape[0]):
    for j in range(conf_matrix.shape[1]):
        plt.text(j, i, format(conf_matrix[i, j], 'd'),
                 horizontalalignment="center",
                 color="white" if conf_matrix[i, j] > thresh else "black")

plt.tight_layout()
plt.savefig('ViT_CM.png')
plt.show()






# ## Display correct and incorrect images

import random


# Convert probabilities to class labels (0 or 1)
test_predictions_classes = (test_predictions > 0.5).astype(int)


# Convert predicted probabilities to class labels
test_predicted_labels = np.argmax(test_predictions_classes, axis=1)


# Define a function to display images
def display_images(images, titles, rows, cols):
    fig, axes = plt.subplots(rows, cols, figsize=(15, 10))
    for i, ax in enumerate(axes.flat):
        ax.imshow(images[i])
        ax.set_title(titles[i])
        ax.axis('off')
    plt.tight_layout()
    plt.show()


# Get indices of correct and incorrect predictions
incorrect_indices = []
correct_indices = []

# Compare predicted labels with true labels
for i in range(len(test_predicted_labels)):
    if test_predicted_labels[i] != test_labels[i]:
        incorrect_indices.append(i)
    else:
        correct_indices.append(i)



# Define a function to select random indices
def get_random_indices(indices, num_samples):
    return random.choices(list(indices), k=num_samples)



# Select random indices for correct and incorrect predictions
random_correct_indices = get_random_indices(correct_indices, 3)
random_incorrect_indices = get_random_indices(incorrect_indices, 3)


# Display random correct predictions
print("Random correct predictions:")
random_correct_images = test_data[random_correct_indices]
random_correct_titles = ["Actual: {}, Predicted: {}".format(test_labels[i], test_predicted_labels[i]) for i in random_correct_indices]
display_images(random_correct_images, random_correct_titles, 1, 3)


# Display incorrect predictions
print("Incorrect predictions:")
incorrect_images = test_data[incorrect_indices]
incorrect_titles = ["Actual: {}, Predicted: {}".format(test_labels[i], test_predicted_labels[i]) for i in incorrect_indices]
display_images(incorrect_images, incorrect_titles, 1, len(incorrect_indices))

