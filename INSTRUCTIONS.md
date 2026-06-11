# EcoClassify - MobileNetV2 Garbage Detector

This repository contains an AI-powered garbage classification system that uses Transfer Learning with a pre-trained **MobileNetV2** model. It classifies waste items into six balanced categories: **cardboard, glass, metal, paper, plastic, and trash**.

---

## Getting Started

### 1. Prerequisites & Environment Setup
First, make sure you have Python 3.12 installed.
To set up your environment, follow these commands:

```bash
# Create a virtual environment (if not already done)
py -3.12 -m venv .venv

# Activate the virtual environment
# Windows (PowerShell):
.venv\Scripts\Activate.ps1
# Windows (CMD):
.venv\Scripts\activate.bat

# Install the required dependencies
pip install -r requirements.txt
```

---

## Pipeline Execution Steps

### 2. Augment and Prepare the Dataset
We merge our local dataset with additional images from the public Hugging Face dataset `omasteam/waste-garbage-management-dataset`.
To run the automated downloader and class balancing pipeline, run:

```bash
python prepare_data.py
```
This script will:
1. Copy the local original images.
2. Stream relevant images from Hugging Face for the 6 waste categories.
3. Automatically down-sample/balance the dataset so all classes have the identical number of images, eliminating class bias.
4. Save the compiled dataset to `Garbage_Augmented/`.

### 3. Train the MobileNetV2 Model
To train the neural network, execute:

```bash
python train.py
```
This training script will:
1. Load `Garbage_Augmented/` and apply stratified splits: **70% Train, 15% Validation, 15% Test**.
2. Set up real-time data augmentation (rotation, shifts, zooms, flips).
3. Load the pre-trained `MobileNetV2` feature extractor with ImageNet weights.
4. **Stage 1 (Warmup)**: Train only the classification head with a learning rate of `1e-3` while base layers are frozen (12 epochs).
5. **Stage 2 (Fine-Tuning)**: Unfreeze the top layers (from layer 100 onwards) and train with a very small learning rate of `1e-5` (15 epochs).
6. Apply callbacks: `EarlyStopping`, `ReduceLROnPlateau`, and `ModelCheckpoint` to save the best model configuration.
7. Print final metrics: accuracy, precision, recall, F1-score, and confusion matrix on the hold-out test set.
8. Save the best model as `model.h5` and plot curves as `training_history.png`.

### 4. Run the Web Dashboard
After training completes and `model.h5` is created, launch the interactive web application:

```bash
python app.py
```

Then open your browser and navigate to:
👉 **[http://127.0.0.1:5000/](http://127.0.0.1:5000/)**

You can upload custom files or click on the test samples. The UI features a drag-and-zoom canvas cropping tool, allowing you to crop out background desks/tables to get the most accurate, background-free predictions from the model!
