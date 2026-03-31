# MuSGraP

MuSGraP (Multi-Scale Graph Prognosticator) is a weakly supervised deep learning framework that fuses local topological patterns captured by dynamic graph convolution with global contextual interactions modeled by transformer self-attention to enable annotation-free, robust postoperative progression risk stratification in limited-stage small cell lung cancer from H&E-stained whole-slide images.

<p align="center">
  <img src="assets/graphical_abstract.png" alt="MuSGraP graphical abstract" width="1000">
</p>

## Overview

The repository is organized into three major stages:

1. **`01_data_processing`** – Whole-slide tiling, foundation-model feature extraction, and tile quality control.
2. **`02_model_development`** – MuSGraP training, evaluation, and interpretability analysis.
3. **`03_downstream_analysis`** – Time-dependent ROC analysis, C-index computation, multivariable Cox regression, and survival analysis.

The codebase is structured as a modular research workflow rather than a monolithic package, with most scripts exposing command-line interfaces for both single-slide and batch processing.

## Repository Layout

```text
MuSGraP/
|-- assets/
|   `-- graphical_abstract.png
|-- 01_data_processing/
|   |-- Get_foundation_model_features.py
|   |-- Patch segmentation.py
|   `-- Quality control.py
|-- 02_model_development/
|   |-- datasets/
|   |-- config/
|   |   `-- config.yaml
|   |-- Log/
|   |-- models/
|   |   |-- Attention.py
|   |   |-- augmentation.py
|   |   |-- contrastive_loss.py
|   |   |-- dataset.py
|   |   |-- Interpretability.py
|   |   |-- Model_Foundation.py
|   |   |-- resnet.py
|   |   `-- Survival.py
|   |-- Result/
|   |-- top10_spatial_results/
|   |-- utils/
|   |   |-- __init__.py
|   |   |-- save_model.py
|   |   |-- Survival.py
|   |   `-- yaml_config_hook.py
|   |-- Visual/
|   |-- eval_survival.py
|   |-- interpretability.py
|   `-- train_survival.py
|-- 03_downstream_analysis/
|   |-- cindex_nomogram_analysis.R
|   |-- Clincail_Tab.py
|   |-- KM.py
|   |-- km_and_rate_charts.R
|   |-- metastasis.py
|   |-- metastasis_forest_plot.R
|   |-- NE.py
|   |-- Stage.py
|   |-- time_roc_analysis.R
|   |-- timedep_auc_comparison.R
|   `-- treatment.py
`-- README.md
```

> **Note:** The `datasets/` and `Log/` directories are auxiliary local resources used during development and debugging. They are not required for the public workflow and should not be treated as mandatory inputs.

## System Requirements

### Hardware

- GPU: Tesla V100-PCIE-32GB (or equivalent)

### Operating System

This package has been tested on the following systems:

- Linux 3.10.0-957.el7.x86_64 (recommended)
- Windows 11 x64

## Installation

> The estimated installation time is approximately 1 hour, depending on network conditions.

1. Install [Anaconda3](https://www.anaconda.com/download).
2. Install CUDA 10.x and cuDNN.
3. Create a conda environment and install the Python dependencies:

```bash
conda create -n musgrasp python=3.10 -y
conda activate musgrasp
pip install -r requirements.txt
```

### Python Dependencies

- `torch`
- `pandas`, `numpy`, `pyarrow`
- `anndata`, `scanpy`
- `scikit-learn`
- `matplotlib`
- `gseapy`, `tangram`
- `pyyaml`
- `openslide-python`

### R Dependencies

The following downstream analysis scripts require R:

- `03_downstream_analysis/cindex_nomogram_analysis.R`
- `03_downstream_analysis/km_and_rate_charts.R`
- `03_downstream_analysis/metastasis_forest_plot.R`
- `03_downstream_analysis/time_roc_analysis.R`
- `03_downstream_analysis/timedep_auc_comparison.R`

## Usage

### Step 1 – H&E Tile Segmentation

Convert SVS files to PNG format and apply the watershed algorithm to generate binary masks. The whole-slide image is divided into 224 × 224 tiles, retaining only those with ≥ 60% tissue content.

```bash
python ./01_data_processing/Patch\ segmentation.py
```

### Step 2 – Quality Control

```bash
python ./01_data_processing/Quality\ control.py
```

### Step 3 – Feature Extraction

Before running feature extraction, organize patch images as one subdirectory per slide. Patch filenames should encode slide identity and spatial coordinates:

```text
dataset/
`-- patches/
    |-- slide_A/
    |   |-- slide_A_(23.0,17.0).jpg
    |   |-- slide_A_(23.0,18.0).jpg
    |   `-- ...
    `-- slide_B/
```

Extract foundation-model features (one feature table per slide):

```bash
python ./01_data_processing/Get_foundation_model_features.py
```

### Step 4 – Model Training & Evaluation

```bash
# Training
python ./02_model_development/train_survival.py

# Evaluation
python ./02_model_development/eval_survival.py
```

### Step 5 – Interpretability

```bash
python ./02_model_development/interpretability.py
```

## Citation

If you use MuSGraP in academic work, please cite the associated study once the manuscript or preprint is publicly available.

