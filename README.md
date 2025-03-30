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


### Getting started
  ```
  git clone https://github.com/nistring/AI-EASI-evaluation.git
  pip install -r requirements.txt
  ```
- You can download the pre-trained model weights for both ROI and whole-body models:

- **ROI Model**: [Download ROI Model Weights](https://drive.google.com/file/d/1Nir6_lJnZyHMf2BJ1yPgAdKCL3enXIzx/view?usp=drive_link)
- **Whole-body Model**: [Download Whole-body Model Weights](https://drive.google.com/file/d/1NixHaf3K1GaJji9L6sqLsq9gQHFKY2BH/view?usp=drive_link)

- After downloading, place the weight files in your project directory and specify the path to them using the `--checkpoint` parameter when running the script.

- To use this model with your custom data, save the image files in the following paths.
  ```
  ├──  data  
  │    └── roi_predict  - here's the datasets folder for ROI images.
  │    └── wb_predict  - here's the datasets folder for whole-body images. The file name of the image and mask should be the same.
  │        └── image - the folder for body images.
  │        └── mask - the folder for masking unwanted areas(e.g. backgrounds, clothes). If it does not exist, the black area of the image will be the masking area.
  ```
- Run script, for example
  ```
  python main.py --phase predict --checkpoint path_to_weights --devices 0 --test-dataset roi_predict or wb_predict
  ```
### Configuration

- The default configuration file is located at `config.yaml` in the project root directory.
- To switch between ROI and whole-body models, modify the `wholebody` parameter in the configuration file:
  - Set `wholebody: true` to use the whole-body model
  - Set `wholebody: false` to use the ROI model
- Example configuration:
  ```yaml
  train:
    wholebody: true  # Set to false for ROI model
  ```

### Acknowledgments
This work is based on a [pytorch implementation](https://github.com/Zerkoar/hierarchical_probabilistic_unet_pytorch) of [hierarchical probabilistic unet](https://arxiv.org/abs/1905.13077v1)
