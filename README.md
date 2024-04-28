# MH4900_FYP_Codes
Codes relating to MH4900 Final Year Project: "AI models for the Early Detection of Diabetic Feet".

## Dataset
  * Consists of a total of 1055 images, where 512 are classified as "Abnormal(Ulcer)" and 543 are "Normal(Healthy skin)"
  * Sourced from: https://www.kaggle.com/datasets/laithjj/diabetic-foot-ulcer-dfu/data

## Models
Original pre-trained models were sourced from:
  * Vision Transformer (ViT-B/16 Feature Extractor model) : https://github.com/tensorflow/tfhub.dev/blob/master/assets/docs/sayakpaul/collections/vision_transformer/1.md
     * Note model file not included due to large size
     * Tensorflow version of model can be downloaded from https://www.kaggle.com/models/spsayakpaul/vision-transformer/tensorFlow2/vit-b16-fe/1?tfhub-redirect=true 
  * Vision Graph Neural Network (ViG-Ti model): https://github.com/jichengyuan/Vision_GNN/tree/main
     * Vision_GNN-main.zip
   
## Preprocessing and Training Scripts
A brief description of the main scripts used in the project for preprocessing the dataset and training of the models is given below. All scripts use an absolute path to the dataset of model folder, which must be changed to allow use of the scripts based on User's dataset location.

### Data_preprocessing.py
Script to load data, split into train, test and validation sets, apply data augmentations and encode labels.

### DenseNet_training and evaluation.py
Script to train and evaluate the preprocessed dataset using a Convolutional Neural Network model, DenseNet121.

### ViT_training and evaluation.py
Script to train and evaluate the preprocessed dataset using a Vision Transformer model, ViT-B/16. 

### ViG_training.py and ViG_evaluation.py
Script to train and evaluate the preprocessed dataset using a Vision Graph Neural Network model, ViG-Ti. 
