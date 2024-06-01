# Overcoming Inherent Uncertainty of Atopic Dermatitis Severity Assessment

## Applying a Probabilistic Deep Learning Algorithm to Whole Body Images

![55036999_File000000_1347229217](https://github.com/nistring/AI-EASI-evaluation/assets/71208448/97356d13-0b5c-48be-99ee-192513b8a2cd)

### Abstract

Assessing the severity of atopic dermatitis (AD) is critical for tailoring effective treatment options for patients. The widely used Eczema Area and Severity Index (EASI) allows dermatologists to evaluate AD severity. However, inter-observer variability in EASI scores can lead to discrepancies in clinical decision-making. To address this, we propose a novel approach using a probabilistic deep learning model to assess AD severity across the entire body.

### Background

- **Atopic Dermatitis (AD)**: A common inflammatory skin condition characterized by pruritic, erythematous, and eczematous lesions.
- **Eczema Area and Severity Index (EASI)**: A tool for quantifying AD severity based on lesion area and intensity.

### Objective

Our goal is to develop an AI model that provides reliable and consistent AD severity assessments, reducing inter-observer variability.

### Methods

1. **Dataset**:
   - We collected full-body photographs from 14 AD patients.
   - Three dermatologists independently assessed EASI scores for each patient.

2. **Severity Categorization**:
   - We categorized whole-body EASI scores as follows:
     - Mild: EASI < 6
     - Moderate: 6 ≤ EASI ≤ 23
     - Severe: EASI > 23

3. **Probabilistic Deep Learning Model**:
   - We trained an AI model using deep learning techniques.
   - The model predicts AD severity probabilities across the entire body.

### Results

- **Correlation with Ground Truth**:
  
  ![image](https://github.com/nistring/AI-EASI-evaluation/assets/71208448/5b621382-e5b3-40c0-a72c-d98a2c83f7d6)
  
  - Pearson correlation coefficients:
    - Internal test set (a): 0.724
    - External test set (b): 0.738

- **Agreement with Dermatologists**:

  ![image](https://github.com/nistring/AI-EASI-evaluation/assets/71208448/d986dee5-6335-4648-98ea-2a5ffe19f58b)

  - In whole-body image evaluation, the AI model agreed (completely or partially) with dermatologists in 11 out of 14 cases (78.6%).

### Getting started
  ```
  git clone https://github.com/nistring/AI-EASI-evaluation.git
  pip install -r requirements.txt
  ```
- You can download the weights from [google drive](https://drive.google.com/drive/folders/12JEz5lnL-9r00-QR1cPN3fKRAB6o6RJr?usp=sharing)
- To use this model with your custom data, save the image files in the following paths.
  ```
  ├──  data  
  │    └── roi_predict  - here's the datasets folder for ROI images.
  │    └── wb_predict  - here's the datasets folder for whole-body images. The file name of the image and mask should be the same.
  │        └── image - the folder for body images.
  │        └── mask - the folder for masking unwanted areas(e.g. backgrounds, clothes). If it does not exist, the black area of the image will be the masking area.
  ```
- Then, run the `predict.sh` script in `scripts` folder.
- The format of the script is
  ```
  python main.py --phase predict --checkpoint path_to_weights --devices 2 --test-dataset roi_predict or wb_predict
  ```


### Acknowledgments
This work is based on a [pytorch implementation](https://github.com/Zerkoar/hierarchical_probabilistic_unet_pytorch) of [hierarchical probabilistic unet](https://arxiv.org/abs/1905.13077v1)
