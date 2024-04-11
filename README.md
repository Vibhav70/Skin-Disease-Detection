# Skin Disease Detection Model

## Overview

This repository contains the code and resources for a deep learning model designed for skin disease detection. The model is trained on a dataset comprising images from various sources, including Dermnet (from Kaggle) and Fitzpatrick17k (from GitHub). The model is based on EfficientNetV2 and achieved a training accuracy of 92% and a testing accuracy of 65%.

## Table of Contents

- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Training](#training)
- [Results](#results)

## Dataset

The dataset consists of images categorized into ten major skin disease categories:

1. Bacterial Infections
2. Chronic Skin Conditions
3. Sexually Transmitted Infections
4. Connective Tissue Disorders
5. Nail Conditions
6. Contact Dermatitis
7. Infestations and Bites
8. Fungal Infections
9. Viral Infections
10. Miscellaneous Skin Conditions

Each category contains a specific set of diseases with corresponding images.

## Model Architecture

The skin disease detection model is built on the EfficientNetV2 architecture. Transfer learning is applied using a pre-trained model on the ImageNet dataset, and fine-tuning is performed on the skin disease dataset.

## Training

The model is trained using various deep learning techniques, including transfer learning and data augmentation. Hyperparameters such as learning rate, batch size, and optimizer settings were tuned to achieve optimal performance.

## Results

The model achieved a training accuracy of 92% and a testing accuracy of 65%. Detailed results, including confusion matrices and other metrics can fe found in the training directory.
