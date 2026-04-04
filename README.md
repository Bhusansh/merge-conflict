# Flood Detection Pipeline (Phase 2)

This repository contains a high-performance 3-class flood segmentation pipeline designed for the ANRF AISEHack Theme 1 (Phase 2) competition. It utilizes a robust ensemble approach combining foundation and convolutional models to detect flood events accurately using 6-band multispectral and SAR imagery.

## Overview

The solution aims to distinguish between **No Flood (0)**, **Flood (1)**, and **Water Body (2)**. It employs a weighted ensemble of three architectures:
1. **Prithvi EO v2 (300M-TL)** - Earth Observation foundation model with an UperNet decoder.
2. **UNet++** - Convolutional model with an EfficientNet-B4 backbone.
3. **FPN (Feature Pyramid Network)** - Convolutional model with a ResNet-50 backbone.

## Key Features

- **Multi-Sensor Input Strategy**: Processes 6-channel TIFF images (SAR-HH, SAR-HV, Green, Red, NIR, SWIR).
- **Dynamic Class Weighting**: Automatically computes frequency-based class weights dynamically based on the active training splits to effectively counter class imbalances.
- **Hybrid Loss Function**: Optimizes learning through combination of `DiceLoss` (for structural agreement) and weighted `CrossEntropyLoss` (for pixel accuracy).
- **2-Phase Foundation Model Training**:
  - **Phase A**: Frozen backbone warm-up (5 epochs)
  - **Phase B**: Full end-to-end fine-tuning (25 epochs)
- **Memory-Optimized Training**: Leverages PyTorch Automatic Mixed Precision (AMP) and gradient accumulation to train heavyweight architectures on constrained T4 GPU environments.
- **Robust Inference with TTA**: Implements 4-way Test-Time Augmentation (Original, Horizontal Flip, Vertical Flip, Diagonal Flip) with a tailored weighted ensemble logic:
  - `0.50 * Prithvi + 0.35 * UNet++ + 0.15 * FPN`
- **Morphological Post-processing**: Cleans predictions by eliminating isolated noise blobs (less than 50 pixels) in the flood channel.

## Dependencies

The implementation is packaged into a standard Jupyter Notebook (`v4-antig-phase2.ipynb`) and requires:
- `PyTorch` & `Lightning`
- `terratorch` (Prithvi backbone)
- `segmentation-models-pytorch` (UNet++ / FPN)
- `albumentations` (Heavy augmentations for satellite imagery)
- `rasterio` (Geospatial data I/O)
- `scipy` (Post-processing)

## Workflow Steps Executed in the Notebook

1. **Environment Setup**: Safely installs `terratorch` and `smp` while explicitly resolving Kaggle-specific `numpy` version conflicts.
2. **Data Pipeline**: Sets up `DataLoader`s with heavy augmentations (`ShiftScaleRotate`, `GaussNoise`, `RandomBrightnessContrast`) dropping the labels channel natively when loaded.
3. **Model Initialization**: Downloads and caches ImageNet and Prithvi EO weights.
4. **Training Loop**: Fits Prithvi parameters in two phases, followed by UNet++ and FPN sequentially configured with `CosineAnnealingLR`.
5. **Ensemble Prediction**: Iterates the unlabelled testing split dataset to formulate averaged probabilities using TTA.
6. **Submission Encoding**: Converts the final Boolean raster masks (Flood variable) into column-major Run-Length Encoding (RLE) to formulate the target `submission.csv`.

---
