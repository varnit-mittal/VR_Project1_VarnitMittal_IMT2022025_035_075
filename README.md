VR MINI PROJECT - README
============================

Project Overview
----------------

This project aims to classify images of faces as either "with mask" or "without mask" using various machine learning techniques.

Subtask (a): Binary Classification Using Handcrafted Features and ML Classifiers
--------------------------------------------------------------------------------

### Objectives

*   Extract handcrafted features from the dataset using Histogram of Oriented Gradients (HOG).
    
*   Train and evaluate at least two machine learning classifiers (SVM and Neural Network).
    
*   Report and compare the accuracy of the classifiers.
    

### Dataset

The dataset consists of two categories:

*   **With Mask**: Images of faces wearing masks.
    
*   **Without Mask**: Images of faces without masks.
    

Images are stored in respective folders:
```
./dataset/with_mask
./dataset/without_mask
```

### Implementation Details

1.  **Data Loading and Preprocessing**
    
    *   Images are read in grayscale format.
        
    *   Each image is resized to **64x64 pixels**.
        
    *   Labels are assigned: 1 for "with mask", 0 for "without mask".
        
2.  **Feature Extraction**
    
    *   Histogram of Oriented Gradients (HOG) is used to extract features from images.
        
    *   Parameters used:
        
        *   pixels\_per\_cell = (8,8)
            
        *   cells\_per\_block = (2,2)
            
        *   feature\_vector = True
            
3.  **Model Training and Evaluation**
    
    *   Data is split into **80% training** and **20% testing**.
        
    *   **SVM (Support Vector Machine)** classifier with hyperparameter tuning via GridSearchCV.
        
    *   **Neural Network (MLPClassifier)** with a 2-layer architecture (128, 64 neurons per layer) trained for 500 iterations.
        
4.  **Performance Evaluation**
    
    *   Classification reports for both models are generated.
        
    *   Metrics used: **Precision, Recall, F1-Score, and Accuracy**.

### Results
#### SVM Classifier Performance:
| Metric        | Class 0 (No Mask) | Class 1 (Mask) | Overall Accuracy |
|--------------|------------------|---------------|-----------------|
| Precision    | 0.95             | 0.96          | **0.95**        |
| Recall       | 0.94             | 0.96          |                 |
| F1-Score     | 0.95             | 0.96          |                 |

#### Neural Network Classifier Performance:
| Metric        | Class 0 (No Mask) | Class 1 (Mask) | Overall Accuracy |
|--------------|------------------|---------------|-----------------|
| Precision    | 0.91             | 0.93          | **0.92**        |
| Recall       | 0.92             | 0.93          |                 |
| F1-Score     | 0.91             | 0.93          |                 |


### Observations

*   The **SVM classifier** achieved **95% accuracy**, outperforming the Neural Network model.
    
*   The **Neural Network classifier** reached **92% accuracy**, which is slightly lower than SVM.
    
*   SVM performed slightly better in terms of precision and recall.




Subtask (b): Binary Classification using CNN
--------------------------------------------------------------------------------

## Project Overview
This task implements a binary classification model using a Convolutional Neural Network (CNN) to detect face masks in images. The dataset consists of images with and without masks, and the goal is to classify each image correctly.

## Model Architecture
The CNN model consists of:
1. **Three Convolutional Layers** with ReLU activation
2. **Max Pooling Layers** after each convolutional block
3. **Flattening Layer**
4. **Fully Connected Dense Layer** (128 neurons, ReLU activation)
5. **Dropout Layer** (0.5 dropout rate)
6. **Final Dense Layer** (1 neuron, Sigmoid activation for binary classification)

### Model Compilation
- **Loss Function:** Binary Crossentropy
- **Optimizers Tested:** Adam, SGD
- **Evaluation Metric:** Accuracy

## Hyperparameter Experiments and Results
Several hyperparameter variations were tested to improve model performance:

| Experiment Name | Learning Rate | Optimizer | Batch Size | Accuracy |
|---------------|--------------|------------|------------|----------|
| Baseline | 0.001 | Adam | 32 | to fill |
| Low LR | 0.0001 | Adam | 32 | 0.9620 |
| High LR | 0.01 | Adam | 32 | 0.9301 |
| SGD Optimizer | 0.001 | SGD | 32 | 0.8933 |
| Larger Batch | 0.001 | Adam | 64 | 0.9681 |

The best-performing model used Adam optimizer with a batch size of 64, achieving an accuracy of **96.81%**.

## Comparison with ML Classifiers
In addition to CNN, we compared its performance against traditional ML classifiers using HOG features.

### **SVM Classifier Performance**
- **Accuracy:** 0.89
- **Precision:** 0.86 (No Mask), 0.91 (Mask)
- **Recall:** 0.89 (No Mask), 0.89 (Mask)
- **F1-score:** 0.87 (No Mask), 0.90 (Mask)

### **Neural Network (MLP) Classifier Performance**
- **Accuracy:** 0.92
- **Precision:** 0.90 (No Mask), 0.93 (Mask)
- **Recall:** 0.92 (No Mask), 0.92 (Mask)
- **F1-score:** 0.91 (No Mask), 0.92 (Mask)

## Conclusion
- The **CNN model significantly outperformed the SVM classifier** (96.81% vs. 89%).
- The **Neural Network (MLP) performed well** but was slightly behind CNN (92%).
- The best CNN model used **Adam optimizer with batch size 64**.




Subtask (c): Region Segmentation Using Traditional Techniques
--------------------------------------------------------------------------------
## Overview

This subtask focuses on segmenting the mask regions for faces identified as "with mask" using traditional segmentation techniques. We implement and compare three different region-based segmentation methods:

- **Gaussian Mixture Model (GMM)**
- **Otsu's Thresholding**
- **Watershed Algorithm**

The segmentation results are evaluated against ground truth masks using Intersection over Union (IoU) and Dice Score.

## Implementation

The segmentation methods are implemented using OpenCV and Scikit-learn.

### 1. **Gaussian Mixture Model (GMM) Segmentation**

The Gaussian Mixture Model is applied to grayscale pixel intensities to separate foreground (mask) from the background.

- The grayscale image is reshaped into a 1D array of pixel intensities.
- GMM with two components (mask and background) is trained.
- The brighter region is assigned as the mask.

### 2. **Otsu's Thresholding**

A global threshold is selected automatically using Otsu’s method to separate the foreground from the background.

- The grayscale image is binarized using Otsu’s threshold.
- The obtained binary mask is used for segmentation.

### 3. **Watershed Algorithm**

This method is based on the distance transform and morphological operations.

- Noise is removed using morphological opening.
- Background and foreground regions are identified using distance transformation.
- Watershed segmentation is applied to distinguish the mask region.

## Evaluation Metrics

Two key metrics are used to evaluate the segmentation accuracy:

- **Intersection over Union (IoU):** Measures the overlap between predicted and ground truth masks.
- **Dice Score:** Computes the similarity between the predicted and ground truth masks.

## Dataset

- **Input:** Cropped face images from `../dataset/face_crop`
- **Ground Truth:** Corresponding segmentation masks from `../dataset/face_crop_segmentation`

## Results Visualization

For each image, the original, ground truth, and segmented masks (GMM, Otsu, Watershed) are displayed side by side along with the computed IoU and Dice scores.

## Example Output

The segmentation results are visualized with corresponding IoU and Dice scores:

```
GMM Segmentation
IoU: 0.65, Dice: 0.78

Otsu Segmentation
IoU: 0.63, Dice: 0.78

Watershed Segmentation
IoU: 0.02, Dice: 0.04
```

## Conclusion and Observations

- GMM provides robust segmentation when the foreground and background have distinct intensity distributions.
- Otsu’s method is simple but may fail in cases with uneven lighting.
- The Watershed algorithm works well when proper preprocessing is applied.

Each method has its strengths and weaknesses, making them suitable for different scenarios in mask segmentation.





Subtask (d): Face Mask Segmentation using U-Net
-------------------------------------------------------------------------------

## Overview
This project implements a U-Net-based deep learning model for face mask detection and segmentation. The model is trained to segment masks from facial images using a dataset of masked and unmasked faces. 

## Dataset
- **Images:** Stored in `dataset/image`
- **Masks:** Stored in `dataset/mask`
- **Preprocessing:** Images are resized to `128x128`, normalized, and converted to RGB format. Masks are binarized (0 for no mask, 1 for mask) and reshaped accordingly.
- **Training data shape:** `(1000, 128, 128, 3)` (Images)
- **Labels shape:** `(1000, 128, 128, 1)` (Mask segmentations)

## Model Architecture
The U-Net model consists of:
- **Encoder:** Four down-sampling blocks with convolutional, batch normalization, dropout, and max-pooling layers.
- **Bottleneck:** Two convolutional layers with batch normalization and dropout.
- **Decoder:** Four up-sampling blocks with transposed convolutions, concatenation, and batch normalization.
- **Output Layer:** A `1x1` convolution with a sigmoid activation function for binary mask segmentation.

## Metrics
The model evaluates segmentation using:
- **IoU (Intersection over Union)**
- **Dice Coefficient**

## Loss Function
- **Binary Crossentropy (BCE) Loss**
- **Dice Loss** (for better handling of class imbalance)
- **Combined Loss**: `BCE + Dice Loss`

## Training Details
- **Optimizer:** Adam (learning rate `1e-3`)
- **Batch Size:** `16`
- **Epochs:** `50`
- **Validation Split:** `20%`

## Model Performance

| Model       | Whole (IoU, Dice) | Validation (IoU, Dice) |
|------------|------------------|----------------------|
| Combined   | 82,89            | 75,85               |
| Dice Loss  | 81,89            | 79,88               |
| 82IoU90Dice | 82,90            | 76,86               |
| Leaky ReLU | 82,89            | 79,88               |
| 256ReLU    | 83,90            | 75,83               |
| Dice Loss, | 83,90            | 78,87               |

## Training Parameters

- **START_FILTERS** = 64  # Starting number of filters, all others are multiples of this  
- **DROPOUT_RATE** = 0.3  # Dropout rate to reduce overfitting  
- **LEARNING_RATE** = 1e-4  # Lower learning rate for stable convergence  
- **BATCH_SIZE** = 16  
- **EPOCHS** = 100  
