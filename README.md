# Chest X-Ray Disease Detection - DAT255

**Authors:** Magne Svalland, Hans Christian Gustafsson, Jonatan Dam  
**Course:** DAT255 Deep Learning Engineering  
**Dataset:** [CheXpert](https://www.kaggle.com/datasets/ashery/chexpert) (via Kaggle) - 224,316 chest X-ray images, 14 conditions

## Project Goal
The goal of this project is to develop and compare deep learning models that can identify various conditions from chest X-ray images. We train a CNN for multi-label classification and explore explainability methods like Grad-CAM to interpret the model's predictions, with the aim of presenting results in a way that could support clinical decision-making.

## Web Application
The model is deployed as a web application where users can upload a chest X-ray and receive predictions with Grad-CAM visualizations.

🔗 **[Try the app on Hugging Face Spaces](https://huggingface.co/spaces/Magnen/chexpert-disease-detection)**

## Models

| Model | Mean AUC |
|-------|----------|
| Baseline CNN (scratch) | 0.865 |
| DenseNet121 (transfer learning) | 0.857 |
| Stanford baseline (Irvin et al., 2019) | 0.907 |

## Pre-trained weights
- `baseline_cnn_best.keras` — Baseline CNN (5.5 MB) — available in `results/`
- `densenet_best.keras` — DenseNet121 — available in `results/`

## Data
CheXpert is downloaded automatically by running the first cell in `notebooks/01_eda.ipynb`. You need a Kaggle account and API token set up locally.

## References
- Chollet, F. (2025). *Deep Learning with Python*, 3rd edition. Manning Publications.
- Irvin, J. et al. (2019). CheXpert: A Large Chest Radiograph Dataset with Uncertainty Labels and Expert Comparison. *AAAI*. https://arxiv.org/abs/1901.07031
- Selvaraju, R.R. et al. (2019). Grad-CAM: Visual Explanations from Deep Networks via Gradient-Based Localization. *ICCV*. https://arxiv.org/abs/1610.02391