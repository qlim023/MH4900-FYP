#!/usr/bin/env python
# coding: utf-8

# # DenseNet121 Model

# ## Load Dataset

import numpy as np
import os

# Define paths to the directory of datasets
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





# ## Load and Train Model

from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam


# Load DenseNet121 model pre-trained on ImageNet
model = DenseNet121(weights='imagenet', include_top=False, input_shape=(224, 224, 3))


# Freeze the layers of the pre-trained model
for layer in model.layers:
    layer.trainable = False


# Add classification layers on top of the pre-trained DenseNet121 model
x = GlobalAveragePooling2D()(model.output)
x = Dense(128, activation='relu')(x)
predictions = Dense(1, activation='sigmoid')(x)


# Create the final model
final_model = Model(inputs=model.input, outputs=predictions)


# Compile the model 
learning_rate = 0.0001
final_model.compile(optimizer=Adam(learning_rate=learning_rate), loss='binary_crossentropy', metrics=['accuracy'])


# Print model summary
final_model.summary()


# Train the model
history = final_model.fit(
    train_data_augmented, train_labels_augmented,
    epochs=30,
    batch_size=32,
    validation_data=(val_data_augmented, val_labels_augmented)
)





# ## Plot accuracy and loss

import matplotlib.pyplot as plt

# Get training and validation accuracy
train_acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

# Get training and validation loss
train_loss = history.history['loss']
val_loss = history.history['val_loss']


# Plot training versus validation accuracy
plt.figure(figsize=(10, 5))
plt.plot(train_acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
# Save the figure as an image file
plt.savefig('DenseNet_accuracy.png')
plt.show()


# Plot training versus validation loss
plt.figure(figsize=(10, 5))
plt.plot(train_loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig('DenseNet_loss.png')
plt.show()






# ## Evaluate model performance

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score, average_precision_score
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc


# Evaluate the model on the test set
test_loss, test_accuracy = final_model.evaluate(test_data, test_labels)


# Print the test loss and accuracy
print("Test Loss:", test_loss)
print("Test Accuracy:", test_accuracy)


# Make predictions on the test set
test_predictions = final_model.predict(test_data)

# Convert probabilities to class labels (0 or 1)
test_predictions_classes = (test_predictions > 0.5).astype(int)

# Calculate accuracy
accuracy = accuracy_score(test_labels, test_predictions_classes)

# Calculate confusion matrix
conf_matrix = confusion_matrix(test_labels, test_predictions_classes)

# Calculate precision, recall, and F1-score
report = classification_report(test_labels, test_predictions_classes, target_names=['Normal', 'Abnormal'], output_dict=True)
precision = report['Abnormal']['precision']
recall = report['Abnormal']['recall']
f1_score = report['Abnormal']['f1-score']

# Calculate True Positives, False Positives, True Negatives, False Negatives
tn, fp, fn, tp = conf_matrix.ravel()

# Calculate Area Under Curve (AUC) for ROC curve
auc_roc = roc_auc_score(test_labels, test_predictions)

# Calculate Average Precision (mAP) for Precision-Recall curve
average_precision = average_precision_score(test_labels, test_predictions)


# Print the metrics
print("Accuracy:", accuracy)
print("True Positives:", tp)
print("False Positives:", fp)
print("Recall (Sensitivity):", recall)
print("Precision:", precision)
print("F1-Score:", f1_score)
print("Area Under Curve (AUC):", auc_roc)
print("Average Precision (mAP):", average_precision)
print("\nConfusion Matrix:")
print(conf_matrix)


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
plt.savefig('DenseNet_CM.png')
plt.show()




# Plot ROC curve
fpr, tpr, thresholds = roc_curve(test_labels, test_predictions)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='red', lw=2, linestyle='--', label='Random Guess')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate (FPR)')
plt.ylabel('True Positive Rate (TPR)')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.savefig('DenseNet_ROC.png')
plt.show()






# ## Display correct and incorrect predictions
import random

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
for i in range(len(test_predictions_classes)):
    if test_predictions_classes[i] != test_labels[i]:
        incorrect_indices.append(i)
        
correct_indices = []
for i in range(len(test_predictions_classes)):
    if test_predictions_classes[i] == test_labels[i]:
        correct_indices.append(i)  # Corrected this line


# Define a function to select random indices
def get_random_indices(indices, num_samples):
    return random.choices(list(indices), k=num_samples)


# Select random indices for correct and incorrect predictions
random_correct_indices = get_random_indices(correct_indices, 3)
random_incorrect_indices = get_random_indices(incorrect_indices, 3)


# Display random correct predictions
print("Random correct predictions:")
random_correct_images = test_data[random_correct_indices]
random_correct_titles = ["Actual: {}, Predicted: {}".format(test_labels[i], test_predictions_classes[i][0]) for i in random_correct_indices]
display_images(random_correct_images, random_correct_titles, 1, 3)


# Display incorrect predictions
print("Incorrect predictions:")
incorrect_images = test_data[random_incorrect_indices]
incorrect_titles = ["Actual: {}, Predicted: {}".format(test_labels[i], test_predictions_classes[i][0]) for i in random_incorrect_indices]
display_images(incorrect_images, incorrect_titles, 1, len(random_incorrect_indices))
