# Probabilistic Machine Learning & Structure Learning Portfolio

This repository contains from-scratch implementations of fundamental probabilistic models and deep learning architectures, developed during the MSc Machine Learning coursework at UCL (Gatsby Unit modules, Supervised Learning).

The implementations focus on **Structure Learning**, **Latent Dynamics**, and **Approximate Inference**, reflecting my research interests in continuous reasoning models.

## 1. Latent Dynamics & Structure Learning
* **`linear_gaussian_ssm.py`**
  * **System Identification for Linear Dynamical Systems.** Implements the M-Step updates (EM algorithm) to learn system dynamics matrices (A, C, Q, R) from data.
  * *Note: Uses `ssm_kalman.py` (provided course utility) for the E-Step inference.*
* **`gaussian_process_structure_discovery.ipynb`**
  * **Compositional Structure Search.** A PyTorch implementation of Gaussian Processes that discovers structure in time-series data (Mauna Loa CO2) by composing Linear, Periodic, and RBF kernels. Uses Log Marginal Likelihood for model selection.

## 2. Approximate Inference & Latent Variables
* **`variational_mean_field_ard.ipynb`**
  * **Variational Inference with ARD.** Implements Mean-Field Variational Inference to recover latent features. demonstratng **Automatic Relevance Determination (ARD)** to prune irrelevant latent dimensions (a key mechanism for structure learning).
* **`bernoulli_mixture_em.py`**
  * **Mixture Models (EM).** An Expectation-Maximization algorithm for clustering binary data. Features a manual implementation of the **Log-Sum-Exp trick** for numerical stability in high-dimensional probabilistic inference.

## 3. Deep Learning & Sampling
* **`pytorch_digit_classification.ipynb`**
  * **Deep Learning Pipeline (MNIST).** A complete PyTorch supervised learning workflow, featuring custom DataLoaders, K-Fold Cross-Validation, and rigorous error analysis (Confusion Matrices).
* **`mcmc_metropolis_hastings.py`**
  * **Combinatorial Optimization via Sampling.** A Metropolis-Hastings MCMC sampler designed to break substitution ciphers by sampling from the posterior distribution of decryption keys.

---
**Tech Stack:** Python, PyTorch, NumPy, Matplotlib.
**Author:** Harshad Dahake
