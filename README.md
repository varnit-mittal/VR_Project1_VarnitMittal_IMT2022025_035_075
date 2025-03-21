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
