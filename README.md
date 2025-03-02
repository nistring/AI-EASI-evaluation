# Overcoming Inherent Uncertainty of Atopic Dermatitis Severity Assessment

## Applying a Probabilistic Deep Learning Algorithm to Whole Body Images

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

2. **Probabilistic Deep Learning Model**:
   - We trained an AI model using deep learning techniques.
   - The model predicts AD severity probabilities across the entire body.

### Results

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
