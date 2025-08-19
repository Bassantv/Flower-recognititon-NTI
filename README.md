# Flower Recognition with CNN

## Overview
This project implements a Convolutional Neural Network (CNN) for flower recognition using the Flowers Recognition dataset from Kaggle. The model is built with Keras and TensorFlow, achieving approximately **69% validation accuracy** after 30 epochs of training.

## Dataset
The **Flowers Recognition** dataset from Kaggle contains 4,317 images categorized into 5 classes:
- Daisy
- Dandelion
- Rose
- Sunflower
- Tulip

*The dataset is automatically downloaded when running the notebook on Kaggle.*

## Model Architecture
The CNN model consists of the following layers:

### Convolutional Blocks
- **Input Layer**: (240, 240, 3) - RGB images resized to 240×240 pixels
- **Conv2D**: 32 filters, 3×3 kernel, ReLU activation
- **MaxPooling2D**: 2×2 pool size
- **Conv2D**: 16 filters, 3×3 kernel, ReLU activation
- **MaxPooling2D**: 2×2 pool size
- **Conv2D**: 8 filters, 3×3 kernel, ReLU activation
- **MaxPooling2D**: 2×2 pool size

### Fully Connected Layers
- **Flatten Layer**: Converts 2D features to 1D
- **Dense**: 128 units, ReLU activation
- **Dense**: 64 units, ReLU activation
- **Dense**: 32 units, ReLU activation
- **Dense**: 5 units, Softmax activation (Output layer)

**Total Parameters**: 820,125 (3.13 MB)

## Data Preprocessing
Images are preprocessed using Keras' `ImageDataGenerator` with the following augmentations:
- Rescaling (1./255)
- Rotation range: 30°
- Width and height shift range: 0.1
- Shear range: 0.1
- Zoom range: 0.3
- Horizontal flipping
- Fill mode: 'nearest'

### Dataset Split
- **Training set**: 3,457 images (80%)
- **Validation set**: 860 images (20%)

## Training Configuration
- **Optimizer**: Adam
- **Loss Function**: Categorical Crossentropy
- **Metrics**: Accuracy
- **Epochs**: 30
- **Batch Size**: 32

## Results
After 30 epochs of training:
- **Training Accuracy**: ~74.5%
- **Validation Accuracy**: ~69.1%
- **Validation Loss**: ~0.799

## Requirements
To run this notebook, you'll need:
- Python 3.11.13
- TensorFlow/Keras
- NumPy
- Kaggle environment (for dataset access)

## Usage
1. Upload the notebook to Kaggle
2. Ensure the Flowers Recognition dataset is connected
3. Run all cells sequentially
4. The model will automatically download the dataset and begin training

## Future Improvements
- Experiment with different architectures (e.g., transfer learning with VGG16, ResNet)
- Implement early stopping and learning rate scheduling
- Add more data augmentation techniques
- Tune hyperparameters for better performance
- Implement model checkpointing to save best weights

## License
This project uses the Flowers Recognition dataset from Kaggle. Please check the original dataset for specific licensing information.
