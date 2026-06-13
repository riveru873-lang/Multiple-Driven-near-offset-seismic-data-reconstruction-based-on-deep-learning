# Multiple-Driven Near-Offset Seismic Data Reconstruction Based on Deep Learning

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-ee4c2c.svg)](https://pytorch.org/)

This repository contains the official implementation of the paper:
**"Multiple-Driven Near-Offset Seismic Data Reconstruction Based on Deep Learning"** 

## 📋 Overview

This work addresses a critical challenge in marine seismic exploration: the reconstruction of large contiguous near-offset data gaps. We propose a novel physics-grounded deep learning framework that integrates the physics-based transform theory with deep learning. This integration allows our method to reconstruct missing near-offset traces with higher fidelity compared to purely data-driven or conventional interpolation methods, especially in the presence of extended gaps.

---

## 🛠️ Repository Architecture

```text
├── configs/               # Hyperparameter & data path configurations
│   ├── unet_configs.py
│   ├── resunet_configs.py # Default configuration file
├── models/                # Deep learning model backbones
│   ├── Unet.py
│   ├── ResUnet.py
├── src/                   # Core data engineering modules
│   └── Seismic_dataset3.py
├── data                   #data
├── utils.py               # Weight init, evaluation metrics, and sliding-window engine
├── train.py               # Main training and validation script
├── test.py                # Whole-shot inference and quantitative evaluation script
└── requirements.txt       # Project dependencies
```

##  Data Availability

Two independent datasets are used in this study. Both are publicly archived on Zenodo with their own DOIs:

| Dataset | Description | DOI |
| :--- | :--- | :--- |
| **Synthetic Dataset** | Pluto and diffraction models with various missing gap configurations (miss20, miss40, miss60). Used for model training and ablation studies. | [![DOI](https://img.shields.io/badge/DOI-10.5281/zenodo.20603084-blue.svg)](https://doi.org/10.5281/zenodo.20603084) |
| **Field Dataset** | Real marine seismic data with 40 contiguous missing near-offset traces. Used for method validation on real-world data. | [![DOI](https://img.shields.io/badge/DOI-10.5281/zenodo.20623128-blue.svg)](https://doi.org/10.5281/zenodo.20623128) |

 **Data Placement:**
   Download and place the raw `.dat` files into your local `./data/` directory.
   
##  Getting Started
# 1. Installation
Clone the repository and install the required dependencies:
git clone [https://github.com/riveru873-lang/Multiple-Driven-near-offset-seismic-data-reconstruction-based-on-deep-learning.git]
cd your-repo-name
```text
pip install -r requirements.txt
(Dependencies include: torch, torchmetrics, torchinfo, numpy, matplotlib)
```
# 2. Dataset Preparation
Ensure your data files are in raw binary format (float32, .dat). Configure the input shapes and file paths inside the corresponding configuration file (e.g., configs/resunet_configs.py):
```text
parser.add_argument("--SAMPLE_PATH", type=str, default="./data/train-samples-data.dat")
parser.add_argument("--LABEL_PATH", type=str, default="./data/train-labels-data.dat")
parser.add_argument("--DIM", type=tuple, default=(nt, nx, n_shots)) # (nt, nx, n_shots)
```
# 3. Model Training
Run train.py to start training. The pipeline supports automatic GPU device discovery, training/validation data splitting ($80\%/20\%$), and logs real-time curves inside the ./outputs folder.
```text
python train.py
```
Note: You can switch the backbone model by modifying the network_model variable in train.py among ['unet', 'denseunet', 'resunet', 'rdbunet'].

# 4. Evaluation & Inference
Execute test.py to load a trained model checkpoint, perform whole-shot high-fidelity interpolation, and compute standard production metrics:
```text
python test.py
```
## Citation
If you find this codebase or the associated methodology useful for your research, please cite our work:
```text
@article{multi_driven_seismic_2026,
  author    = {Zhina Li and Zhichuan Yu},
  title     = {Multiple-Driven Near-Offset Seismic Data Reconstruction Based on Deep Learning},
  journal   = {Computers & Geosciences},
  year      = {2026},
  note      = {Submitted}
}
```
