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

## **1. Introduction**  
This part of the mini-project focuses on segmenting masked face regions using deep learning-based image segmentation techniques. The primary objective is to train a U-Net model to precisely detect and segment mask regions in images and compare its performance with traditional segmentation methods using evaluation metrics like Intersection over Union (IoU) and Dice Score.  

## **2. Dataset**  

The dataset used for this project contains face images with ground truth mask annotations. Each image has a corresponding binary mask that marks the mask region.  

- **Source**: [Dataset](https://github.com/sadjadrz/MFSD).  
- **Structure**:  
  - `face_crop/` – Contains RGB face images.  
  - `face_crop_segmentation/` – Corresponding grayscale mask images.  

Each image and its corresponding mask were resized to a uniform dimension of **128×128** (or **256×256** in some experiments).  Due to computational constraints, the model was trained on a **subset of 1000 images** instead of the full dataset. This allowed for faster experimentation while maintaining strong segmentation performance.

---

## **3. Methodology**  

### **Data Preprocessing**  
1. Images were read using OpenCV and converted to RGB format.  
2. Resized to the chosen input size (**128×128** or **256×256**).  
3. Normalized pixel values to the range **[0,1]**.  
4. Masks were read as grayscale, resized, and expanded to include a single channel.  

---

## **4. Model Architecture**  
The U-Net model used in this project follows an encoder-decoder structure with skip connections to preserve spatial information. The key components of the architecture include:  

1. **Convolutional Blocks:**  
   - Each block consists of **two convolutional layers** with a kernel size of **3×3**, followed by **batch normalization** and **dropout** for regularization.  
   - **LeakyReLU** is used as the activation function instead of ReLU for better gradient flow.  

2. **Feature Scaling:**  
   - The number of filters starts at **64** and doubles at each downsampling stage (64 → 128 → 256 → 512 → 1024) in the encoder.  
   - In the decoder, the number of filters decreases symmetrically by a factor of **2** at each upsampling stage (1024 → 512 → 256 → 128 → 64).  

3. **Downsampling & Upsampling:**  
   - **Max-pooling** is used to reduce the spatial dimensions in the encoder.  
   - **UpSampling2D** is used to restore the resolution in the decoder, followed by concatenation with corresponding encoder feature maps.  

4. **Output Layer:**  
   - A **1×1 convolution** is applied at the final layer with a **sigmoid activation** to generate a binary mask.  

This architecture enables effective feature extraction while maintaining fine details in the segmentation output.

---

## **5. Training Details**  
- **Loss Functions**: **Dice Loss**  
- **Optimizer**: Adam with a learning rate of **1e-3**.  
- **Metrics**: IoU and Dice Score.  
- **Batch Size**: 16  
- **Epochs**: 50  
- **Dropout Rate**: 0.3
---

## **6. Post-Processing**  
After segmentation, the raw output mask may contain small artifacts or sharp boundaries. To refine the mask, the following post-processing steps were applied:  

1. **Thresholding:**  
   - The predicted mask is thresholded at **0.5** to obtain a binary mask.  

2. **Morphological Closing:**  
   - A **7×7 kernel** is used to remove small holes inside the detected mask regions, making the segmentation more coherent.  

3. **Gaussian Blurring:**  
   - A **3×3 Gaussian blur** is applied to smoothen the mask edges and create a more natural-looking boundary.  

These refinements help reduce noise, improve mask continuity, and enhance segmentation accuracy.

---

## **7. Results**  
The performance of different model variations was evaluated using **Intersection over Union (IoU)** and **Dice Score**, two key metrics for segmentation quality. The best-performing model incorporated **Dice loss, LeakyReLU activation, and post-processing techniques**, achieving:  

- **Whole dataset:** **IoU = 88.45%**, **Dice Score = 93.35%**  
- **Validation dataset:** **IoU = 84.32%**, **Dice Score = 90.79%**  

The inclusion of post-processing significantly improved the mask smoothness, reducing sharp artifacts in the segmentation output.

### **Example Predictions**  

#### **Random Sample (Good Segmentation)**  
_Example where the model successfully segments the mask:_  
![image](https://github.com/user-attachments/assets/e9105528-27d1-4317-9732-ba8fbd391eaf)

#### **Min IoU Case (Failure Case)**  
_Example where the model struggles with segmentation:_  
![image](https://github.com/user-attachments/assets/af994d16-b6b6-42ee-a6c2-2c9498c71f2d)

#### **Max IoU Case (Best Segmentation)**  
_Example where the model's prediction closely matches the ground truth:_  
![image](https://github.com/user-attachments/assets/30d0c77a-f626-429d-923d-ac58e8da178b)

---

## **8. Hyperparameters and Experiments**  
The following variations were explored:  

| **Model Variant** | **Whole Dataset (IoU, Dice)** | **Validation Set (IoU, Dice)** |  
|------------------|----------------------------|----------------------------|  
| BCE + Dice Loss | 80.23, 89.01 | 75.23, 85.d |  
| Dice Loss | 80.11, 89.21 | 77.29, 88.78 |  
| BCE Loss | 80.15, 90.85 | 76.77, 86.81 |  
| Leaky ReLU | 82.21, 89.73 | 76.12, 87.22 |  
| **256×256 Image Size, Dice Loss** | 82.10, 90.02 | 75.17, 83.06 |  
| Dice Loss + Leaky ReLU | 81.58, 89.34 | 79.51, 88.04 |  
| **Best Model: Dice Loss + Leaky ReLU + Post-Processing** | **88.45, 93.35** | **84.32, 90.79** |  

### **Key Observations:**  
- **LeakyReLU activation** improved performance compared to ReLU.  
- **Dice loss performed better** than BCE loss.  
- **Larger input size (256×256)** did not necessarily yield better validation results.  
- **Post-processing (smoothing, morphological closing) significantly improved** the results.  

---
## **9. Challenges Faced**  

1. **Training Time and Compute Limitations:**  
   - Training on high-resolution images required extensive computational resources.  
   - Due to resource constraints, the model was trained on a **subset of 1000 images** instead of the full dataset to reduce training time while maintaining reasonable performance.  

2. **Over-Sharp Predictions:**  
   - The predicted masks initially had overly sharp boundaries compared to the ground truth.  
   - This issue was mitigated using **post-processing techniques**, including **morphological closing** and **Gaussian blurring**, to produce smoother and more natural segmentation masks.
  
## **10. Comparison with Traditional Segmentation Methods**  

To assess the effectiveness of the U-Net model, its performance was compared with a **traditional threshold-based segmentation method** using **IoU and Dice Score** metrics.  

| **Method**                  | **IoU (%)** | **Dice Score (%)** |
|-----------------------------|------------|--------------------|
| **Traditional Segmentation** (Thresholding + Morphology) | 72.85       | 84.32               |
| **U-Net (Best Model with Post-Processing)** | **88.45**  | **93.35**         |

### **Why U-Net Outperforms Traditional Methods?**  

1. **Learned Features vs. Handcrafted Rules:**  
   - Traditional methods rely on fixed thresholding and morphological operations, making them sensitive to lighting variations, noise, and occlusions.  
   - U-Net **learns hierarchical features** from the data, allowing it to generalize better across varying image conditions.  

2. **Context Awareness:**  
   - Unlike traditional methods that focus only on pixel intensity, U-Net **captures spatial context** using convolutional layers and skip connections.  
   - This helps in accurately segmenting **complex mask shapes** that thresholding-based methods often fail to detect.  

3. **Robustness to Variability:**  
   - The dataset contains images with diverse lighting, occlusions, and mask variations.  
   - U-Net can adapt to these variations, whereas traditional segmentation struggles with consistency.  

Thus, U-Net **significantly outperforms** traditional methods in both **IoU and Dice Score**, making it a more reliable choice for precise mask segmentation.
