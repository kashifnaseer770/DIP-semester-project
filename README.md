# Pedestrian Gender Classification with Imbalanced Data Handling

## Project Overview

This project addresses binary imbalanced classification, demonstrated with pedestrian gender recognition. It uses data augmentation and preprocessing to improve dataset quality and balance. Both low-level (HOG, LBP, GLCM) and deep (VGG19 FC7) features are extracted, fused, and reduced with PCA. A Linear SVM classifier is trained and evaluated with 10-fold cross-validation.

## Features Extracted

- Histogram of Oriented Gradients (HOG)  
- Local Binary Patterns (LBP)  
- Gray-Level Co-occurrence Matrix (GLCM)  
- Deep features from the FC7 layer of pretrained VGG19

## Methodology

1. Data augmentation (flipping, rotation) for class balance  
2. Feature extraction (low-level + deep)  
3. Feature fusion and dimensionality reduction via PCA  
4. Classification using Linear SVM  
5. Evaluation with 10-fold cross-validation (accuracy, precision, recall, F1, confusion matrix)

## Setup & Usage
- **Environment:** Jupyter Notebook  
- **Requirements:**  
  - Python 3.x
  - `numpy`, `scikit-learn`, `opencv-python`, `tensorflow` or `torch` (for VGG19), `matplotlib`

### Running the Project
1. Open the Jupyter Notebook file (`.ipynb`) included in the repo.  
2. Run the notebook cells sequentially to preprocess data, extract features, train the model, and evaluate performance.  
3. Modify parameters or dataset paths inside the notebook as needed.

## Results
The proposed approach effectively handles imbalanced data and achieves over **80% accuracy** in pedestrian gender classification. Other metrics such as precision, recall, and F1-score also demonstrate robust performance.


