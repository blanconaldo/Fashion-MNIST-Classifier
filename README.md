# Fashion MNIST CNN Classifier

A complete convolutional neural network project for classifying clothing items, built with TensorFlow/Keras and deployed as a desktop application.

![Python Version](https://img.shields.io/badge/python-3.11%2B-blue)
![TensorFlow Version](https://img.shields.io/badge/tensorflow-2.15%2B-orange)
![License](https://img.shields.io/badge/license-MIT-green)

## Project Overview

This project implements a deep learning model that recognizes different types of clothing items from images. It includes:

- Data loading and preprocessing pipeline
- CNN model architecture design and training
- Data augmentation techniques for improved generalization
- Model evaluation and visualization
- User-friendly desktop application for real-world image classification

## Dataset

The [Fashion MNIST dataset](https://github.com/zalandoresearch/fashion-mnist) contains 70,000 grayscale images of clothing items (28x28 pixels) across 10 categories:

- T-shirt/top
- Trouser
- Pullover
- Dress
- Coat
- Sandal
- Shirt
- Sneaker
- Bag
- Ankle boot

## Project Structure

```
Fashion-MNIST-Classifier/
├── main.py               # Main script for training and evaluation
├── tkinter_example.py    # GUI application for real-world testing
├── utils/                # Helper functions and model architecture
├── models/               # Saved model files
├── data/                 # Cached dataset
├── requirements.txt      # Dependencies
├── output.txt            # Final output after training model
├── .gitignore            # Self explanatory
└── README.md             # Project documentation
```

## Requirements

- Python 3.11 or higher
- Libraries listed in requirements.txt:

```
tensorflow>=2.19.0
keras>=3.9.0
numpy>=2.1.0
matplotlib>=3.10.0
scikit-learn>=1.6.0
seaborn>=0.12.0
pillow>=11.0.0
tkinter (included in Python on Windows, python-tk package on Linux)
```

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/blanconaldo/Fashion-MNIST-Classifier.git
   cd Fashion-MNIST-Classifier
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Training the Model

Run the main script to train and evaluate the model (might take up to 2 hours):

```bash
python main.py
```

This will:
1. Download and preprocess the Fashion MNIST dataset
2. Train the CNN model with data augmentation
3. Evaluate model performance
4. Save the trained model

### Using the Desktop Application

To classify your own clothing images, you can either run the following after training the model, or you could directly use the downloaded best model in the "models" folder:

```bash
python tkinter_example.py
```

Upload any JPG, PNG, BMP, or GIF image of a clothing item to get a prediction.

## Implementation Details

### Data Preprocessing

- **Normalization**: Pixel values scaled to range [0,1]
- **Reshaping**: Added channel dimension for CNN input (28x28x1)
- **Shuffling**: Random shuffling of training data
- **Train/Test/Val Split**: The dataset is already split to training and testing - around ≈85% for training and ≈15% for testing. The training was further split to 80% training and 20% validation. The final evaluation is done using the test data.

### Data Augmentation

Improved model generalization with:
- Random rotations (±10°)
- Width/height shifts (±10%)
- Zoom variations (±10%)
- Horizontal flips
- Shear transformations

### CNN Architecture

```
Model: "sequential"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 conv2d (Conv2D)             (None, 28, 28, 32)        320       
                                                                 
 batch_normalization         (None, 28, 28, 32)        128       
                                                                 
 activation (Activation)     (None, 28, 28, 32)        0         
                                                                 
 conv2d_1 (Conv2D)           (None, 28, 28, 32)        9248      
                                                                 
 batch_normalization_1       (None, 28, 28, 32)        128       
                                                                 
 activation_1 (Activation)   (None, 28, 28, 32)        0         
                                                                 
 max_pooling2d               (None, 14, 14, 32)        0         
                                                                 
 dropout (Dropout)           (None, 14, 14, 32)        0         
                                                                 
 conv2d_2 (Conv2D)           (None, 14, 14, 64)        18496     
                                                                 
 batch_normalization_2       (None, 14, 14, 64)        256       
                                                                 
 activation_2 (Activation)   (None, 14, 14, 64)        0         
                                                                 
 conv2d_3 (Conv2D)           (None, 14, 14, 64)        36928     
                                                                 
 batch_normalization_3       (None, 14, 14, 64)        256       
                                                                 
 activation_3 (Activation)   (None, 14, 14, 64)        0         
                                                                 
 max_pooling2d_1             (None, 7, 7, 64)          0         
                                                                 
 dropout_1 (Dropout)         (None, 7, 7, 64)          0         
                                                                 
 flatten (Flatten)           (None, 3136)              0         
                                                                 
 dense (Dense)               (None, 512)               1606144   
                                                                 
 batch_normalization_4       (None, 512)               2048      
                                                                 
 activation_4 (Activation)   (None, 512)               0         
                                                                 
 dropout_2 (Dropout)         (None, 512)               0         
                                                                 
 dense_1 (Dense)             (None, 10)                5130      
                                                                 
=================================================================
Total params: 1,679,082
Trainable params: 1,677,674
Non-trainable params: 1,408
```

- **Convolution layers**: Feature extraction with 3x3 kernels
- **Batch normalization**: Stabilizes and accelerates training
- **Activation functions**: ReLU for non-linearity
- **Max pooling**: Reduces dimensionality while preserving features
- **Dropout layers**: Prevent overfitting
- **Dense layers**: Classification based on extracted features

### Training Process

- **Optimizer**: Adam with learning rate 0.001
- **Loss function**: Sparse Categorical Crossentropy
- **Batch size**: 64
- **Epochs**: 80 (with early stopping)
- **Callbacks**: 
  - Early stopping (patience=5)
  - Learning rate reduction on plateau
  - Model checkpoint to save best weights

### Evaluation

The model achieved **92.53%** accuracy on the test set with these metrics:

```
              precision    recall  f1-score   support
 T-shirt/top       0.91      0.84      0.87      1000
     Trouser       1.00      0.99      0.99      1000
    Pullover       0.94      0.86      0.90      1000
       Dress       0.95      0.91      0.93      1000
        Coat       0.90      0.89      0.89      1000
      Sandal       0.98      0.99      0.98      1000
       Shirt       0.71      0.86      0.78      1000
     Sneaker       0.95      0.97      0.96      1000
         Bag       0.99      0.99      0.99      1000
  Ankle boot       0.98      0.95      0.96      1000
```

### GUI Application

- Built with Tkinter for cross-platform compatibility
- Allows uploading and classifying custom images
- Preprocesses images to match model's expected format
- Displays both original and processed versions
- Shows prediction and confidence score
- Due to the nature of the dataset, it is limited on what images it can make confident prediction on.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- [Fashion MNIST Dataset](https://github.com/zalandoresearch/fashion-mnist) by Zalando Research
- TensorFlow and Keras documentation