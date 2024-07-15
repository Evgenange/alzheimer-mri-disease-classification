# Alzheimer MRI Disease Classification 

This repository contains code and resources for classifying Alzheimer's Disease using MRI images. The dataset used is sourced from Hugging Face.

Introduction
The Alzheimer MRI Disease Classification dataset is used to classify Alzheimer's disease based on MRI scans. The dataset consists of brain MRI images labeled into four categories:

* '0': Mild_Demented
* '1': Moderate_Demented
* '2': Non_Demented
* '3': Very_Mild_Demented

Contents
* EDA.ipynb: Exploratory Data Analysis (EDA) notebook with some basic EDA for the MRI images.
* Training.py: Script for training a VGG19 model using PyTorch.
* Testing.py: Script for testing the trained VGG19 model.
* trained_vgg19.pth: Pre-trained VGG19 model.
* requirements.txt

Results
Achieved an accuracy of 95.23% on the testing set.

Requirements
GPU is required to run the notebooks effectively.

Citation
If you use this dataset in your research or health medicine applications, we kindly request that you cite the following publication:
@dataset{alzheimer_mri_dataset,
  author = {Falah.G.Salieh},
  title = {Alzheimer MRI Dataset},
  year = {2023},
  publisher = {Hugging Face},
  version = {1.0},
  url = {https://huggingface.co/datasets/Falah/Alzheimer_MRI}
}
