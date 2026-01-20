# probabilistic-ml-portfolio
Description: Probabilistic ML implementations

This repository contains from-scratch implementations of fundamental probabilistic models, developed during the MSc Machine Learning coursework at UCL.

### Contents
* **linear_gaussian_ssm.py**: System Identification for Linear Dynamical Systems. Implements the M-Step updates to learn system dynamics (Matrices A, C, Q, R) from data.
    * *Note: Uses `ssm_kalman.py` (provided course utility) for the E-Step inference.*
* **bernoulli_mixture_em.py**: An Expectation-Maximization (EM) algorithm for clustering binary data, featuring the Log-Sum-Exp trick for stability.
* **mcmc_metropolis_hastings.py**: A Markov Chain Monte Carlo sampler for decrypting substitution ciphers via combinatorial optimization.
