# Robust MRI Brain Tumor Segmentation

Semantic segmentation of brain lesions in 2D MRIs (BRISC 2025 dataset) using PyTorch. This repository implements U-Net and U-Net++ architectures (EfficientNet-B3 encoders), features a custom Focal-Dice loss pipeline, and highlights model robustness against noisy, misaligned ground-truth labels through rigorous qualitative error analysis. 

## Project Overview

This project was developed to explore the limits of semantic segmentation on medical imaging and demonstrate the level of granular control PyTorch offers in a research environment. While standard evaluation metrics are crucial, this repository emphasizes **qualitative error analysis** to prove model generalization. After 74 hours of training, the models demonstrated the ability to correctly identify and segment tumors that were mislabeled or entirely missed by the dataset's human annotators. 

## The Dataset: BRISC 2025

The model is trained and evaluated on the **BRISC 2025 (Brain Tumor MRI Dataset)**. 

* **Task:** Pixel-wise semantic segmentation.
* **Modality:** 2D T1-weighted MRI slices.
* **Planes:** Axial, Coronal, and Sagittal.

## Architecture & Tech Stack

* **Framework:** PyTorch
* **Architectures:** U-Net & U-Net++ (`segmentation_models_pytorch`)
* **Encoders:** `efficientnet-b3` (initialized with ImageNet weights)
* **Data Augmentation:** Utilized `Albumentations` for spatial augmentations (ElasticTransform, GridDistortion, ShiftScaleRotate) and pixel-level adjustments (RandomBrightnessContrast) to enforce strict generalization.
* **Loss Function:** A custom weighted combination of Focal Loss (0.4) for hard examples and Dice Loss (0.6) for spatial overlap.
* **Optimizer:** Adam with Weight Decay.
* **Scheduler:** `ReduceLROnPlateau` for adaptive learning rate decay.

## Results & Error Analysis (The "Plot Twist")

**Aggregate Metrics:**
* **U-Net Test IoU:** ~0.81
* **U-Net++ Test IoU:** ~0.82

**Qualitative Analysis & Outperforming the Ground Truth:**

While an overall IoU of ~0.82 is robust, diving into the individual visual predictions revealed a significant discrepancy. On cleanly annotated slices, both models consistently achieved **>0.90 IoU**, mapping the lesions flawlessly. 

The aggregate score was dragged down by a subset of the dataset's human-annotated "Ground Truth" masks. Several of these masks were misaligned, drawn in the incorrect anatomical quadrants, or entirely missing. The trained U-Net models successfully generalized the spatial features of actual lesions, finding tumors that the original annotations missed. Because the model predicted correctly but the ground truth was flawed, these accurate predictions artificially received an IoU score of 0.000.

![Qualitative Error Analysis Output](https://github.com/AkhundMubeen/brain-tumor-segmentation/blob/main/results/ouput_img_1.png)

This repository serves as a case study in the necessity of visual validation in medical machine learning: **an aggregate metric should never be trusted blindly without analyzing the raw outputs.**

## Acknowledgments

* The BRISC 2025 dataset creators and collaborating radiologists.
* Developed as part of an ongoing initiative to build robust, academic-grade computer vision pipelines for deep learning research.
