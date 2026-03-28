# 🚀 Mapping AAV Capsid Sequences to Functions through Function-Guided *In Silico* Evolution

This repository provides a comprehensive implementation of the **ALICE** and **ALICE-X** architectures, along with all supporting scripts for data analysis and evaluation. 🧬✨
The project is organized as follows:

* **`./alice/`**
  Contains the full implementation of the **ALICE** framework.
  Detailed instructions for installation, usage, and training are available in the dedicated [README](./alice/README.md). 📘

* **`./alice_x/`**
  Hosts the codebase for the **ALICE-X** architecture, including experimental pipelines and usage guidelines. ⚙️

* **`./code_for_analysis/`** and **`./alice/statistical_code/`**
  Provide scripts for downstream data analysis, statistical evaluation, and result visualization. 📊📈

---

Overall, this repository serves as a unified framework for **mapping AAV capsid sequences to functional outcomes** via function-guided *in silico* evolution—bridging sequence design, evaluation, and optimization in a closed-loop workflow. 🔄🧠

---

# 🧩 ALICE-X Environment Requirements

This document outlines the required environment configuration, core dependencies, and installation guidelines for running the `ALICE-X` project.

## 1. Basic Environment

* **Operating System**: macOS / Linux / Windows (Linux or macOS recommended) 💻
* **Python Version**: **Python 3.7** is recommended for optimal compatibility with dependencies
* **CUDA Support**: A CUDA-enabled GPU is recommended for accelerating SeqGAN training and AutoGluon inference (CUDA 11.3+ preferred) ⚡

## 2. Core Dependencies

The project consists of three main modules: **FE (Feature Engineering)**, **finetuning (generation)**, and **ranking (evaluation)**.

### 2.1 Deep Learning & Generative Models (finetuning)

* **PyTorch (>= 1.10.1)** — for SeqGAN training and inference 🔥
* **Transformers (>= 4.30.2)** — for loading pretrained RoBERTa models 🤖
* **Torchvision / Torchaudio** — auxiliary PyTorch ecosystem components

### 2.2 AutoML & Evaluation (ranking)

* **AutoGluon (>= 0.8.0)** — for training and deploying GBM-based evaluation models (`autogluon.tabular`) 📦
* **XGBoost (>= 1.6.2)** — gradient boosting backend used within AutoGluon

### 2.3 Feature Engineering & Acceleration (FE)

* **Numba** — accelerates sequence alignment algorithms (e.g., Smith–Waterman) 🚀
* **NumPy (>= 1.21.6)** — numerical computation
* **Pandas (>= 1.3.5)** — data processing and CSV handling

### 2.4 Data Science & Utilities

* **Scikit-learn (>= 1.0.2)** — general ML utilities
* **SciPy (>= 1.7.3)** — statistical computation
* **Matplotlib / Seaborn** — visualization 📊
* **tqdm** — progress tracking ⏳

---

## 3. Installation Guide

### 3.1 Install via Conda (Recommended) 🧪

Create the environment using the `environment.yml` file in the project root:

```bash
conda env create -f environment.yml
conda activate alice
```

> ⚠️ Note: If `environment.yml` is unavailable or customization is needed, follow the manual steps below.

---

### 3.2 Manual Installation

1. **Install PyTorch (CUDA 11.3 example)**:

```bash
pip install torch==1.10.1+cu113 torchvision==0.11.2+cu113 torchaudio==0.10.1 -f https://download.pytorch.org/whl/cu113/torch_stable.html
```

2. **Install AutoGluon**:

```bash
pip install autogluon
```

3. **Install remaining dependencies**:

```bash
pip install transformers numba pandas numpy matplotlib seaborn scipy scikit-learn tqdm
```

---

## 4. Module Dependency Overview

| Module         | Core Dependencies          | Description                                       |
| -------------- | -------------------------- | ------------------------------------------------- |
| **FE**         | `numba`, `numpy`, `pandas` | Sequence evolution and feature generation         |
| **finetuning** | `torch`, `transformers`    | SeqGAN training and RoBERTa-based semantic tuning |
| **ranking**    | `autogluon`, `pandas`      | Model evaluation and inference                    |

---

## 5. Notes & Best Practices

* **GPU Memory**: At least **8GB VRAM** is recommended for SeqGAN training 🧠
* **Numba Caching**: The first run triggers JIT compilation; subsequent runs will be significantly faster ⚡
* **RoBERTa Weights**: Ensure the checkpoint file
  `./finetuning/model/words_model_ep3.pth`
  is available—this is critical for the finetuning stage 📂

