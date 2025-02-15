# ESMStabP: A Regression Model for Protein Thermostability Prediction

**Author:** Marcus Ramos, Robert L. Jernigan, Mesih Kilinc

**Affiliation:** Iowa State University, Ames, IA  

## Overview

ESMStabP is a regression model designed to predict protein thermostability with high accuracy. Built upon embeddings from the ESM2 protein language model, this tool integrates thermophilic classifications and other features to enhance performance. The model has applications in biotechnology, pharmaceuticals, and food science, where accurate predictions of protein melting temperatures are crucial.

## Features

- **Accuracy:** Achieves an R² of 0.94 and a Pearson correlation coefficient (PCC) of 0.92.
- **Advanced Embeddings:** Utilizes embeddings from the ESM2 (esm2_t33_650M_UR50) protein language model, with significant performance improvements observed using the final (33rd) layer.
- **Comprehensive Dataset:** Combines and cleans datasets from multiple sources, balancing thermophilic and non-thermophilic proteins for robust predictions.
- **Flexible Architecture:** Employs a random forest regressor and integrates features such as optimal growth temperature (OGT) and experimental conditions.
- **Parameter-Efficient Fine-Tuning:** Includes experiments with LoRA (Low-Rank Adaptation) for efficient model specialization.

## Repository Structure

```
.
├── Dataset Assembly
├── ESMStabP
├── LoRA
└── Web Interface
```

### 1. Dataset Assembly

Contains the code used to aggregate and preprocess the datasets used for training and testing. This includes balancing thermophilic and non-thermophilic sequences, as well as integrating data from multiple sources.

**Key Features:**
- Dataset cleaning and deduplication.
- Thermophilic classification.
- K-fold cross-validation setup.

### 2. ESMStabP

This folder contains the core code for training and testing the ESMStabP model.

**Key Features:**
- Integration of ESM2 embeddings (33rd layer).
- Random forest regressor for optimal performance.
- Model evaluation using metrics such as R², MAE, MSE, RMSE, and PCC.

### 3. LoRA

Contains all code related to the parameter-efficient fine-tuning experiments using LoRA. While LoRA experiments did not surpass ESMStabP's accuracy, they provide insights into efficient model adaptation.

**Key Features:**
- LoRA-based fine-tuning on ESM2 and Prot5.
- Comparative results for parameter-efficient methods.

### 4. Web Interface

Provides a local user interface to interact with the ESMStabP model. Note that due to GitHub file size restrictions, the model itself is not included in the repository. You will need to run `setup.py` to download and configure the model locally before launching the UI.

**Key Features:**
- Simple interface for predicting protein melting temperatures.
- Instructions for local setup and use.


## Results
- **Performance Metrics:**
  | Model      | R²  | PCC  | MAE  | MSE   | RMSE  |
  |------------|------|------|------|-------|-------|
  | ESMStabP   | 0.94 | 0.92 | 3.42 | 19.51 | 4.13  |
  | DeepStabP  | 0.81 | 0.88 | 3.62 | 18.40 | 4.32  |
  | ProTstab2  | 0.51 | 0.68 | 4.95 | 46.95 | 6.31  |

- **Embedding Layer Performance:**
  Layer 33 of ESM2 provided the best results, highlighting the importance of feature selection in leveraging protein language model embeddings.

<!-- ## Citation

If you use this code in your research, please cite:

```
@article{ramos2025esmstabp,
  title={ESMStabP: A Regression Model for Protein Thermostability Prediction},
  author={Marcus Ramos},
  journal={bioRχiv},
  year={2025}
}
``` -->

## Acknowledgments

- Iowa State University for research support.
- Facebook AI for providing the ESM2 protein language model.
- Contributors to DeepStabP and TemBERTure for inspiration and datasets.
