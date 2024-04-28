#!/usr/bin/env python
# coding: utf-8

# # Data Preprocessing

import os
from PIL import Image
import numpy as np
import random
import matplotlib.pyplot as plt


patches_dir = './FYP/DataSet'


# Load images from a directory
def load_images_from_dir(directory):
    images = []
    for subdir in os.listdir(directory):  # Iterate over subdirectories
        subdir_path = os.path.join(directory, subdir)
        if os.path.isdir(subdir_path):  # Check if it's a directory
            for filename in os.listdir(subdir_path):
                img_path = os.path.join(subdir_path, filename)
                try:
                    img = Image.open(img_path)
                    images.append(img)
                except Exception as e:
                    print(f"Error loading image {img_path}: {e}")
    return images


# Load images from directory
patches = load_images_from_dir(patches_dir)



# Load and display a sample of skin patches
def display_sample(patches_dir, num_samples=5, export_filename='sample_patches.png'):
    fig, axes = plt.subplots(1, num_samples, figsize=(15, 5))
    for i, ax in enumerate(axes):
        subdir = random.choice(os.listdir(patches_dir))
        subdir_path = os.path.join(patches_dir, subdir)
        img_filename = random.choice(os.listdir(subdir_path))
        img_path = os.path.join(subdir_path, img_filename)
        img = plt.imread(img_path)
        ax.imshow(img)
        ax.set_title(subdir)
        ax.axis('off')
    
    # Save the figure as an image file
    plt.savefig(export_filename)

    # Show the plot    
    plt.show()


# Display a sample of skin patches
display_sample(patches_dir, export_filename='sample_patches.png')




# ### Split dataset

from sklearn.model_selection import train_test_split
from collections import Counter


# Create labels for the images based on their subdirectory names
labels = []
for subdir in os.listdir(patches_dir):
    subdir_path = os.path.join(patches_dir, subdir)
    if os.path.isdir(subdir_path):
        labels.extend([subdir] * len(os.listdir(subdir_path)))


# Split the data into train and test sets (80% train, 20% test)
train_images, test_images, train_labels, test_labels = train_test_split(patches, labels, test_size=0.2, random_state=42, stratify=labels)


# Further split the training set into train and validation sets (70% train, 10% validation)
train_images, val_images, train_labels, val_labels = train_test_split(train_images, train_labels, test_size=0.1, random_state=42, stratify=train_labels)


# Print the sizes of the resulting sets
print("Total number of samples:", len(patches))
print("Number of samples in training set:", len(train_images))
print("Number of samples in validation set:", len(val_images))
print("Number of samples in test set:", len(test_images))


# Count the number of samples for each class in a given set
def count_classes(images, labels):
    class_counts = Counter(labels)
    return class_counts

# Count classes for train set
train_class_counts = count_classes(train_labels, train_labels)
print("Train set class counts:")
print("Abnormal(Ulcer):", train_class_counts['Abnormal(Ulcer)'])
print("Normal(Healthy skin):", train_class_counts['Normal(Healthy skin)'])

# Count classes for validation set
val_class_counts = count_classes(val_labels, val_labels)
print("\nValidation set class counts:")
print("Abnormal(Ulcer):", val_class_counts['Abnormal(Ulcer)'])
print("Normal(Healthy skin):", val_class_counts['Normal(Healthy skin)'])

# Count classes for test set
test_class_counts = count_classes(test_labels, test_labels)
print("\nTest set class counts:")
print("Abnormal(Ulcer):", test_class_counts['Abnormal(Ulcer)'])
print("Normal(Healthy skin):", test_class_counts['Normal(Healthy skin)'])




# ### Convert to Numpy arrays and shuffle dataset

from sklearn.utils import shuffle


def preprocess_images(directory):
    images = []
    labels = []
    for subdir in os.listdir(directory):
        subdir_path = os.path.join(directory, subdir)
        if os.path.isdir(subdir_path):
            label = 1 if subdir == 'Abnormal(Ulcer)' else 0
            for filename in os.listdir(subdir_path):
                img_path = os.path.join(subdir_path, filename)
                try:
                    img = Image.open(img_path)
                    # Resize image to 224x224
                    img = img.resize((224, 224))
                    # Convert to RGB if not already in RGB format
                    if img.mode != 'RGB':
                        img = img.convert('RGB')
                    # Convert image to numpy array
                    img_array = np.array(img)
                    images.append(img_array)
                    labels.append(label)
                except Exception as e:
                    print(f"Error loading image {img_path}: {e}")
    # Convert lists to numpy arrays
    images = np.array(images)
    labels = np.array(labels)
    # Shuffle the dataset
    images, labels = shuffle(images, labels, random_state=42)
    return images, labels


# Load and preprocess images
images, labels = preprocess_images(patches_dir)


# Print the shape of images and labels
print("Shape of images array:", images.shape)
print("Shape of labels array:", labels.shape)


def Preprocess_images(images, labels):
    image_data = []
    image_labels = []
    
    # Process images
    for img, label in zip(images, labels):
        try:
            # Resize image to 224x224
            img = img.resize((224, 224))
            # Convert to RGB if not already in RGB format
            if img.mode != 'RGB':
                img = img.convert('RGB')
            # Convert image to numpy array
            img_array = np.array(img)
            image_data.append(img_array)
            image_labels.append(label)
        except Exception as e:
            print(f"Error processing train image: {e}")
            
    # Convert lists to numpy arrays
    image_data = np.array(image_data)
    image_labels = np.array(image_labels)
    
    return image_data, image_labels



train_data, train_labels = Preprocess_images(train_images, train_labels)
val_data, val_labels = Preprocess_images(val_images, val_labels)
test_data, test_labels = Preprocess_images(test_images, test_labels)



# Print the shape of data and labels
print("Shape of train data array:", train_data.shape)
print("Shape of train labels array:", train_labels.shape)
print("Shape of validation data array:", val_data.shape)
print("Shape of validation labels array:", val_labels.shape)
print("Shape of test data array:", test_data.shape)
print("Shape of test labels array:", test_labels.shape)




# ### Data Augmentation 

from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Define augmentation parameters
zoom_range = 2
rotation_range = 90
shear_range = 0.4
width_shift_range = 0.2
height_shift_range = 0.2
horizontal_flip = True
vertical_flip = True


# Create an ImageDataGenerator instance with specified augmentation parameters
datagen = ImageDataGenerator(
    zoom_range=zoom_range,
    rotation_range=rotation_range,
    shear_range=shear_range,
    width_shift_range=width_shift_range,
    height_shift_range=height_shift_range,
    horizontal_flip=horizontal_flip,
    vertical_flip=vertical_flip
)

# Generate augmented images
def generate_augmented_images(data, labels, num_augmented_images=7):
    augmented_data = []
    augmented_labels = []
    
    # Generate augmented images for each augmentation technique
    for img, label in zip(data, labels):
        # Reshape the image to meet the requirements of ImageDataGenerator
        img = img.reshape((1,) + img.shape)
        
        # Generate augmented images for each augmentation technique
        for i in range(num_augmented_images):
            # Generate a single augmented image
            augmented_img = next(datagen.flow(img))[0]
            augmented_data.append(augmented_img)
            augmented_labels.append(label)
            
    # Convert lists to numpy arrays
    augmented_data = np.array(augmented_data)
    augmented_labels = np.array(augmented_labels)
    
    return augmented_data, augmented_labels



# Generate augmented images for train datasets
augmented_train_data, augmented_train_labels = generate_augmented_images(train_data, train_labels)



# Augment validation images
augmented_val_data, augmented_val_labels = generate_augmented_images(val_data, val_labels)



# Print the shape of augmented labels
print("Shape of augmented train labels array:", augmented_train_labels.shape)
print("Shape of augmented validation labels array:", augmented_val_labels.shape)

# Print unique classes in augmented labels
print("Unique classes in augmented train labels:", np.unique(augmented_train_labels))
print("Unique classes in augmented validation labels:", np.unique(augmented_val_labels))



# Concatenate original and augmented data and labels
train_data_augmented = np.concatenate((train_data, augmented_train_data), axis=0)
train_labels_augmented = np.concatenate((train_labels, augmented_train_labels), axis=0)
val_data_augmented = np.concatenate((val_data, augmented_val_data), axis=0)
val_labels_augmented = np.concatenate((val_labels, augmented_val_labels), axis=0)



# Shuffle augmented data
train_data_augmented, train_labels_augmented = shuffle(train_data_augmented, train_labels_augmented, random_state=42)
val_data_augmented, val_labels_augmented = shuffle(val_data_augmented, val_labels_augmented, random_state=42)


# Print the shape of augmented data and labels
print("Shape of augmented train data array:", train_data_augmented.shape)
print("Shape of augmented train labels array:", train_labels_augmented.shape)
print("Shape of augmented validation data array:", val_data_augmented.shape)
print("Shape of augmented validation labels array:", val_labels_augmented.shape)



# Function to print class counts
def print_class_counts(labels, dataset_name):
    class_counts = Counter(labels)
    print(f"Class counts for {dataset_name}:")
    print("Abnormal(Ulcer):", class_counts['Abnormal(Ulcer)'])
    print("Normal(Healthy skin):", class_counts['Normal(Healthy skin)'])



# Print class counts for augmented train set
print_class_counts(train_labels_augmented, "augmented train set")

# Print class counts for augmented validation set
print_class_counts(val_labels_augmented, "augmented validation set")

# Print class counts for test set
print_class_counts(test_labels, "test set")



# Create figure and axes
fig, axes = plt.subplots(num_images_to_display, num_augmented_images, figsize=(15, 5*num_images_to_display))

# Display augmented images with class labels
for i in range(num_images_to_display):
    # Select a random index for the original image
    original_index = np.random.randint(len(train_data_augmented))
    # Display the original image
    axes[i, 0].imshow(normalize_image(train_data_augmented[original_index]))
    axes[i, 0].set_title(train_labels_augmented[original_index])
    axes[i, 0].axis('off')

    # Display augmented images
    for j in range(1, num_augmented_images):
        # Select a random index for the augmented image
        augmented_index = np.random.randint(len(train_data_augmented))
        axes[i, j].imshow(normalize_image(train_data_augmented[augmented_index]))
        axes[i, j].set_title(train_labels_augmented[augmented_index])
        axes[i, j].axis('off')

plt.tight_layout()
# Save the figure as an image file
plt.savefig('sample_augmented_images.png')
plt.show()




# ## Encode Labels
from sklearn.preprocessing import LabelEncoder

# Initialize LabelEncoder
label_encoder = LabelEncoder()

# Fit LabelEncoder on training labels
label_encoder.fit(train_labels_augmented)

# Transform labels
train_labels_encoded = label_encoder.transform(train_labels_augmented)
val_labels_encoded = label_encoder.transform(val_labels_augmented)
test_labels_encoded = label_encoder.transform(test_labels)

# Swap the encoding values
train_labels_encoded_swapped = 1 - train_labels_encoded
val_labels_encoded_swapped = 1 - val_labels_encoded
test_labels_encoded_swapped = 1 - test_labels_encoded

# Print the original and encoded labels for comparison
print("Original Training Labels:", train_labels_augmented[:10])
print("Encoded Training Labels:", train_labels_encoded_swapped[:10])
print()
print("Original Validation Labels:", val_labels_augmented[:10])
print("Encoded Validation Labels:", val_labels_encoded_swapped[:10])
print()
print("Original test Labels:", test_labels[:10])
print("Encoded test Labels:", test_labels_encoded_swapped[:10])





# ## Export Dataset


# Define paths for saving the datasets
output_dir = './FYP/DataSet/Preprocessed_DFU'

# Define filenames for each dataset
train_data_file = 'train_data.npy'
train_labels_file = 'train_labels_encoded.npy'
val_data_file = 'val_data.npy'
val_labels_file = 'val_labels_encoded.npy'
test_data_file = 'test_data.npy'
test_labels_file = 'test_labels_encoded.npy'


# Save augmented train set
np.save(os.path.join(output_dir, train_data_file), train_data_augmented)
np.save(os.path.join(output_dir, train_labels_file), train_labels_encoded_swapped)


# Save augmented validation set
np.save(os.path.join(output_dir, val_data_file), val_data_augmented)
np.save(os.path.join(output_dir, val_labels_file), val_labels_encoded_swapped)


# Save test set
np.save(os.path.join(output_dir, test_data_file), test_data)
np.save(os.path.join(output_dir, test_labels_file), test_labels_encoded_swapped)
print("Datasets exported successfully.")
