﻿# Chest X-Ray Disease Detection - DAT255

**Authors:** Magne Svalland, Hans Christian Gustafsson, Jonatan Dam  
**Course:** DAT255 Deep Learning Engineering  
**Dataset:** [CheXpert](https://www.kaggle.com/datasets/ashery/chexpert) (via Kaggle) - 224,316 chest X-ray images, 14 conditions

## Project Goal
The goal of this project is to develop and compare deep learning models that can identify various conditions from chest X-ray images. We train a CNN for multi-label classification and explore explainability methods like Grad-CAM to interpret the model's predictions, with the aim of presenting results in a way that could support clinical decision-making.

## Models
- Custom CNN built from scratch with Keras (Mean AUC: 0.865)
- Transfer learning with pre-trained DenseNet121
- Grad-CAM explainability (coming)

## Pre-trained weights
Pre-trained model weights are available in the `results/` folder in this repository.
- baseline_cnn_best.keras - Baseline CNN (Mean AUC: 0.865)

## Data
CheXpert is downloaded automatically by running the first cell in `notebooks/01_eda.ipynb`. You need a Kaggle account and API token set up locally.

## References
- Chollet, F. (2025). *Deep Learning with Python*, 3rd edition. Manning Publications.
- Irvin, J. et al. (2019). CheXpert: A Large Chest Radiograph Dataset with Uncertainty Labels and Expert Comparison. *AAAI*. https://arxiv.org/abs/1901.07031