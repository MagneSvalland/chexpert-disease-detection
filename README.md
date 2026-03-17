# Chest X-Ray Disease Detection — DAT255

**Authors:** Magne Svalland, Hans Christian Gustafsson, Jonatan Dam  
**Course:** DAT255 Deep Learning Engineering  
**Dataset:** [CheXpert](https://stanfordmlgroup.github.io/competitions/chexpert/) — 224,316 chest X-ray images, 14 conditions

## Project Goal
The goal of this project is to develop and compare deep learning models that can identify various conditions from chest X-ray images. We train a CNN for multi-label classification and explore explainability methods like Grad-CAM to interpret the model's predictions, with the aim of presenting results in a way that could support clinical decision-making.

## Models
- Custom CNN built from scratch with Keras
- Transfer learning with pre-trained ResNet and DenseNet architectures

## Data
CheXpert is downloaded automatically by running the first cell in `notebooks/01_eda.ipynb`. You need a Kaggle account and API token set up locally.

## References
- F. Chollet: *Deep Learning with Python*, 3rd edition (Manning, 2025)
- Irvin et al.: *CheXpert: A Large Chest Radiograph Dataset* (Stanford, 2019)



