"""
Linear Gaussian State Space Model (LGSSM) - EM Learning
-------------------------------------------------------
Author: Harshad Dahake
Context: Gatsby Unit Coursework (Probabilistic Modelling)

Description:
  Implementation of System Identification (Parameter Learning) for an LGSSM.
  
  While the inference (Kalman Filter/Smoother) uses a provided course utility 
  ('ssm_kalman'), this script implements the Expectation-Maximization (EM) loop 
  from scratch, specifically:
  - The M-Step: Deriving and implementing the closed-form update rules for 
    parameters A (Transition), C (Emission), Q (Process Noise), and R (Obs. Noise).
  - Convergence Analysis: Tracking the regularized log-likelihood over iterations.

Key Concepts:
  - Latent Linear Dynamics
  - Expectation-Maximization (EM) for Time-Series
  - System Identification
"""

# %% IMPORT REQUIRED PACKAGES
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ssm_kalman import run_ssm_kalman
import math

# %% IMPORT DATA AND INITIALIZE PARAMETERS
## Import data: ssm_spins.txt
with open("ssm_spins.txt", "r") as f:
    data = pd.read_csv(f, sep='\s+', header=None)
data.columns = ["x1","x2","x3","x4","x5"]

X = data.T.values # get transposed data for correct dimensions, and create numpy array from the pandas data frame

## Initialisations
y_init = np.zeros(4).T # initialise y (latent) with mean = 0

Q_init = np.identity(4) # initialise Q (variance of latent linear dynamics)

# Define A, a true parameter
a = np.zeros((4,4))
a[0,0] = math.cos(np.pi/90)
a[1,1] = math.cos(np.pi/90)
a[2,2] = math.cos(np.pi/45)
a[3,3] = math.cos(np.pi/45)
a[0,1] = -math.sin(np.pi/90)
a[1,0] = math.sin(np.pi/90)
a[2,3] = -math.sin(np.pi/45)
a[3,2] = math.sin(np.pi/45)
A = 0.99*a; del a

# Define Q, a true parameter
Q = np.identity(4) - np.matmul(A,A.T)

# Define C, a true parameter
C = np.zeros((5,4))
C[0,0] = 1
C[1,1] = 1
C[3,3] = 1
C[0,2] = 1
C[1,3] = 1
C[2,0] = 1
C[2,3] = 1
C[3,2] = 1
C[4,0] = 0.5
C[4,1] = 0.5
C[4,2] = 0.5
C[4,3] = 0.5

# Define R, a true parameter
R = np.identity(5)

# %% Implement Smoothing Mode using given ssm_kalman implementation and Plot results

# Set current parameters for inputs
A0 = A
Q0 = Q
C0 = C
R0 = R

# run smoother
y_hat, V_hat, V_joint, likelihood = run_ssm_kalman(X, y_init, Q_init, A0, Q0, C0, R0)

# Plot y_hat for smooth mode
y_preds = y_hat.T

plt.figure()
plt.plot(range(y_preds.shape[0]), y_preds, alpha=0.8)
plt.show()

# Plot V_hat (log determinant) for smooth mode
v_logdets = [np.linalg.slogdet(V_hats)[1].item() for V_hats in V_hat]

plt.figure()
plt.plot(range(len(v_logdets)), v_logdets, alpha=0.8)
plt.show()

# %% Implement Filtering/Forward Mode using given ssm_kalman implementation and Plot results

# Set current parameters for inputs
A0 = A
Q0 = Q
C0 = C
R0 = R

# run filter
y_hat2, V_hat2, V_joint2, likelihood2 = run_ssm_kalman(X, y_init, Q_init, A0, Q0, C0, R0, mode="filt")

# Plot y_hat for filter mode
y_preds2 = y_hat2.T

plt.figure()
plt.plot(range(5,y_preds2.shape[0]), y_preds2[5:], alpha=0.8) # , c=cols
plt.show()

# Plot V_hat (log determinant) for filter mode
v_logdets2 = [np.linalg.slogdet(V_hats2)[1].item() for V_hats2 in V_hat2]

plt.figure()
plt.plot(range(len(v_logdets2)), v_logdets2, alpha=0.8)
plt.show()

# %% Compare min/max uncertainty (V_hat) for observation

# Compare maximum variance for smooth mode (v_logdets) and minimum variance for filter mode (v_logdets2)
print(f"Maximum uncertainty metric in 'smooth' mode: {np.max(v_logdets)}, with\nMinimum uncertainty metric in 'filter' mode: {np.min(v_logdets2)}")

# %% Part B(i): Implement M-step and the EM algorithm

# Define M-step function and EM algorithm
def M_step(X, y_hat, V_hat, V_joint):

    d, full_T = X.shape # number of observed variables and time T
    k = y_hat.shape[0] # number of latent variables

    # Implement formula for C_new 
    C2 = np.zeros((k,k))
    for t in range(full_T):
        yt = y_hat[:, t].reshape(k,1)
        yt_outer_prod = np.matmul(yt,yt.T)
        C2 += V_hat[t] + yt_outer_prod
        
    C2_inv = np.linalg.inv(C2)
    C1 = np.matmul(X,y_hat.T)

    C_new = np.matmul(C1,C2_inv); del C2, C2_inv
    
    # Implement formula for R_new 
    R1 = np.zeros((d,d))
    R2 = np.zeros((d,k))
    for t in range(full_T):
        xt = X[:,t].reshape(d,1)
        yt = y_hat[:,t].reshape(k,1)
        R1 += np.matmul(xt,xt.T)
        R2 += np.matmul(xt,yt.T)
    R_new = 1/full_T * (R1 - np.matmul(R2,C_new.T))

    # Ensure symmetric R_new covariance matrix
    R_new = (R_new + R_new.T) / 2.0

    # Implement formula for A_new 
    T_minus_1 = full_T - 1
    A1 = np.zeros((k,k))
    A2 = np.zeros((k,k))
    for t in range(T_minus_1):
        yt = y_hat[:, t].reshape(k,1)
        yt_outer_prod = np.matmul(yt,yt.T)
        A2 += V_hat[t] + yt_outer_prod
        
        yt_plus_1 = y_hat[:, t+1].reshape(k,1)
        A1_second_term = np.matmul(yt_plus_1,yt.T)
        A1 += V_joint[t] + A1_second_term
    
    A2_inv = np.linalg.inv(A2)
    A_new = np.matmul(A1,A2_inv)
    
    # Implement formula for Q_new 
    Q1 = np.zeros((k,k))
    Q2 = A1
    for t in range(1,full_T):
        yt = y_hat[:,t].reshape(k,1)
        yt_minus_1 = y_hat[:,t-1].reshape(k,1)
        Q1 += V_hat[t] + np.matmul(yt,yt.T)
    Q_new = 1/(T_minus_1) * (Q1 - np.matmul(Q2,A_new.T))

    # Ensure symmetric Q_new covariance matrix
    Q_new = (Q_new + Q_new.T) / 2.0
    
    return (A_new, Q_new, C_new, R_new)

# Define EM algorithm function
def EM_algorithm(X, y_init, Q_init, A0, Q0, C0, R0, mode: str, iters: int):
    
    # Set E-step function input parameters (initialisation only)
    A_upd = A0
    Q_upd = Q0
    C_upd = C0
    R_upd = R0

    likelihoods = []
    total_iter = iters
    epsilon = 10**-5 # threshold to terminate EM algorithm based on relative improvement of log-likelihood
    for iter in range(iters):
        # E-step for Smoother
        y_hat, V_hat, V_joint, likelihood = run_ssm_kalman(X, y_init, Q_init, A_upd, Q_upd, C_upd, R_upd, mode)
        likelihoods.append(likelihood)

        # M-step for Smoother
        A_upd, Q_upd, C_upd, R_upd = M_step(X, y_hat, V_hat, V_joint)

        # Check for convergence (relative improvement of log likelihood)
        if iter > 0:
            if np.abs(np.abs(np.sum(likelihoods[iter])) - np.abs(np.sum(likelihoods[iter-1])))/np.abs(np.sum(likelihoods[iter-1])) < epsilon:
                # print(f"Stopping now at iter #{iter+1}")
                total_iter = iter+1 # set total iterations per termination, if convergence occurs
                break
    
    return (likelihoods, A_upd, Q_upd, C_upd, R_upd, total_iter)

# %% ## Part B(ii): Run EM algorithm with varying starting parameterts
## Train model with various starting parameters:
# (a) True parameters
# (b) 10 randomised parameters

# %% ## Run EM algorithm with starting parameters same as true (generator) parameters with 600 iterations

# data
obs_data = X # train or test data

# hyperparameters
m = "smooth" # Kalman Smoothing
n = 600 # number of iterations (E and M steps) - was run for 50 steps as well, similar termination due to the convergence threshold check

# initialisations
A0 = A
R0 = R
C0 = C
Q0 = Q

y_init = y_init
Q_init = Q_init

# run EM algorithm
likelihoods, A_final, Q_final, C_final, R_final, total_iters = EM_algorithm(obs_data, y_init, Q_init, A0, Q0, C0, R0, mode=m, iters=n)
total_lh = np.sum(likelihoods, axis=1)

# get total log likelihood for true parameters post-training - this will be used to plot train vs test data performance in Part (c)
true_final_lh = total_lh[total_iters-1]

# Plot log likelihood vs iterations for EM algorithm
plt.figure()
plt.plot(range(1,total_iters+1), total_lh)
plt.title("Log Likelihood vs Iterations for EM algorithm")
plt.xlabel("Iteration")
plt.ylabel("Log likelihood")
plt.show()

# %% RUN EM ALGORITHM FOR 10 DIFFERENT RANDOMISED INITIALIZED PARAMETER VALUES: A, Q, C, R (600 iterations)

# Total runs
num_runs = 10

# Setup lists to collect results
total_lh_run = [] # log likelihood after each iteration for each run
final_lh_run = [] # final log likelihood after all iterations for each run
final_lh_run.append(true_final_lh) # Add true params. final log likelihood on training data for comparison later on
A_start = [] # Initialised A for each run
Q_start = [] # Initialised Q for each run
C_start = [] # Initialised C for each run
R_start = [] # Initialised R for each run
A_end = [] # Learned A for each run
Q_end = [] # Learned Q for each run
C_end = [] # Learned C for each run
R_end = [] # Learned R for each run
iterations = [] # Number of iterations at termination for each run

# Run EM algorithm for 10 different initialisations
for i in range(num_runs):
    # print(f"i: {i}")

    ## Randomised Initialisations
    
    # Maintain A0 and C0 values: center around 0 to allow for uniform seeds (+/-)
    A0 = 2 * np.random.rand(*A.shape) - 1
    C0 = 2 * np.random.rand(*C.shape) - 1
    
    # Ensure Q0 and R0 are symmetric (covariance metrices)
    Q_interim = 2 * np.random.rand(*Q.shape) - 1
    Q0 = np.matmul(Q_interim, Q_interim.T); del Q_interim
    
    R_interim = 2 * np.random.rand(*R.shape) - 1
    R0 = np.matmul(R_interim, R_interim.T); del R_interim    

    # data
    obs_data = X # train or test data
    
    # hyperparameters
    m = "smooth" # Kalman Smoothing
    n = 600 # number of iterations (E and M steps)

    # run EM algorithm and calculate total log likelihood across time steps for each iteration
    likelihoods, A_final, Q_final, C_final, R_final, total_iters = EM_algorithm(obs_data, y_init, Q_init, A0, Q0, C0, R0, mode=m, iters=n)
    total_lh = np.round(np.sum(likelihoods,axis=1),2)
    final_lh = total_lh[total_iters-1]
    
    # Collect results for run
    total_lh_run.append(total_lh)
    A_start.append(A0)
    C_start.append(C0)
    Q_start.append(Q0)
    R_start.append(R0)
    A_end.append(A_final)
    C_end.append(C_final)
    Q_end.append(Q_final)
    R_end.append(R_final)
    iterations.append(total_iters)
    final_lh_run.append(final_lh)


## Plot log likelihood vs iterations for 10 runs
plt.figure()
for i in range(num_runs):
    plt.plot(range(1,iterations[i]+1), total_lh_run[i], label=f'run {i+1}', alpha=0.6)
plt.title(f'EM Convergence: Log-Likelihood vs. Iteration ({num_runs} Random Starts)')
plt.xlabel('EM Iteration')
plt.ylabel('Total Log-Likelihood')
plt.grid(True, linestyle='--', alpha=0.5)
plt.show()


# %% Compare Learned Parameters across Runs and True Parameters

## Visually compare Learned Parameter A for 10 runs with True Parameter A via Heatmap
all_A_matrices = []
all_A_matrices.append(A)
for j in range(len(A_end)):
    all_A_matrices.append(A_end[j])

# 1. Calculate Global Scale (for comparison)
all_data = np.concatenate([m.flatten() for m in all_A_matrices])
global_vmin = all_data.min()
global_vmax = all_data.max()

# 2. Calculate Local Scale for A_True (for structure)
A_True = all_A_matrices[0]
local_vmin = A_True.min()
local_vmax = A_True.max()

# 3. Create a figure with a space for the comparison grid (11 matrices) 
#    and the structural plot (1 matrix)

# Create a figure large enough for the grid (3x4 = 12 spots) + extra space
fig = plt.figure(figsize=(16, 11))

# --- PART A: GLOBAL COMPARISON GRID (The 11 original plots) ---
gs_comp = fig.add_gridspec(3, 4, left=0.05, right=0.75, top=0.95, bottom=0.1) 
# Plots 1-11 go here (3 rows, 4 columns)

titles = ['True A (Global Scale)'] + [f'Run {i+1}' for i in range(10)] + ['']
for i in range(11):
    ax = fig.add_subplot(gs_comp[i])
    im = ax.imshow(all_A_matrices[i], cmap='coolwarm', 
                   vmin=global_vmin, 
                   vmax=global_vmax)
    ax.set_title(titles[i], fontsize=10)
    ax.axis('off')

# Add one common colorbar for the global grid
cbar_ax_comp = fig.add_axes([0.77, 0.1, 0.02, 0.85]) 
fig.colorbar(im, cax=cbar_ax_comp, label='Value (Global Range)')

# --- PART B: LOCAL STRUCTURE PLOT (The 12th spot, isolated) ---
gs_struct = fig.add_gridspec(1, 1, left=0.82, right=0.98, top=0.55, bottom=0.35)
ax_struct = fig.add_subplot(gs_struct[0])

# Plot True Q using its LOCAL scale (revealing the diagonal structure)
im_struct = ax_struct.imshow(A_True, cmap='viridis', 
                             vmin=local_vmin, 
                             vmax=local_vmax)
ax_struct.set_title('True A Structure (Local Scale)', fontsize=12)
ax_struct.set_xticks(np.arange(A_True.shape[0]))
ax_struct.set_yticks(np.arange(A_True.shape[0]))
ax_struct.tick_params(labelsize=8)

# Add a separate colorbar for the local structure plot
cbar_ax_struct = fig.add_axes([0.82, 0.25, 0.16, 0.02])
fig.colorbar(im_struct, cax=cbar_ax_struct, orientation='horizontal', 
             label=f'Value (Local Range: {local_vmin:.2e} to {local_vmax:.2e})')

fig.suptitle('LGSSM A Matrix Comparison: Global vs. Structural Views', fontsize=18)
plt.show()


## Visually compare Learned Parameter Q for 10 runs with True Parameter Q via Heatmap
all_Q_matrices = []
all_Q_matrices.append(Q)
for j in range(len(Q_end)):
    all_Q_matrices.append(Q_end[j])

# 1. Calculate Global Scale (for comparison)
all_data = np.concatenate([m.flatten() for m in all_Q_matrices])
global_vmin = all_data.min()
global_vmax = all_data.max()

# 2. Calculate Local Scale for Q_True (for structure)
Q_True = all_Q_matrices[0]
local_vmin = Q_True.min()
local_vmax = Q_True.max()

# 3. Create a figure with a space for the comparison grid (11 matrices) 
#    and the structural plot (1 matrix)

# Create a figure large enough for the grid (3x4 = 12 spots) + extra space
fig = plt.figure(figsize=(16, 11))

# --- PART A: GLOBAL COMPARISON GRID (The 11 original plots) ---
gs_comp = fig.add_gridspec(3, 4, left=0.05, right=0.75, top=0.95, bottom=0.1) 
# Plots 1-11 go here (3 rows, 4 columns)

titles = ['True Q (Global Scale)'] + [f'Run {i+1}' for i in range(10)] + ['']
for i in range(11):
    ax = fig.add_subplot(gs_comp[i])
    im = ax.imshow(all_Q_matrices[i], cmap='coolwarm', 
                   vmin=global_vmin, 
                   vmax=global_vmax)
    ax.set_title(titles[i], fontsize=10)
    ax.axis('off')

# Add one common colorbar for the global grid
cbar_ax_comp = fig.add_axes([0.77, 0.1, 0.02, 0.85]) 
fig.colorbar(im, cax=cbar_ax_comp, label='Value (Global Range)')

# --- PART B: LOCAL STRUCTURE PLOT (The 12th spot, isolated) ---
gs_struct = fig.add_gridspec(1, 1, left=0.82, right=0.98, top=0.55, bottom=0.35)
ax_struct = fig.add_subplot(gs_struct[0])

# Plot True Q using its LOCAL scale (revealing the diagonal structure)
im_struct = ax_struct.imshow(Q_True, cmap='viridis', 
                             vmin=local_vmin, 
                             vmax=local_vmax)
ax_struct.set_title('True Q Structure (Local Scale)', fontsize=12)
ax_struct.set_xticks(np.arange(Q_True.shape[0]))
ax_struct.set_yticks(np.arange(Q_True.shape[0]))
ax_struct.tick_params(labelsize=8)

# Add a separate colorbar for the local structure plot
cbar_ax_struct = fig.add_axes([0.82, 0.25, 0.16, 0.02])
fig.colorbar(im_struct, cax=cbar_ax_struct, orientation='horizontal', 
             label=f'Value (Local Range: {local_vmin:.2e} to {local_vmax:.2e})')

fig.suptitle('LGSSM Q Matrix Comparison: Global vs. Structural Views', fontsize=18)
plt.show()


## Visually compare Learned Parameter C for 10 runs with True Parameter C via Heatmap
all_C_matrices = []
all_C_matrices.append(C)
for j in range(len(C_end)):
    all_C_matrices.append(C_end[j])

# 1. Calculate Global Scale (for comparison)
all_data = np.concatenate([m.flatten() for m in all_C_matrices])
global_vmin = all_data.min()
global_vmax = all_data.max()

# 2. Calculate Local Scale for C_True (for structure)
C_True = all_C_matrices[0]
local_vmin = C_True.min()
local_vmax = C_True.max()

# 3. Create a figure with a space for the comparison grid (11 matrices) 
#    and the structural plot (1 matrix)

# Create a figure large enough for the grid (3x4 = 12 spots) + extra space
fig = plt.figure(figsize=(16, 11))

# --- PART A: GLOBAL COMPARISON GRID (The 11 original plots) ---
gs_comp = fig.add_gridspec(3, 4, left=0.05, right=0.75, top=0.95, bottom=0.1) 
# Plots 1-11 go here (3 rows, 4 columns)

titles = ['True C (Global Scale)'] + [f'Run {i+1}' for i in range(10)] + ['']
for i in range(11):
    ax = fig.add_subplot(gs_comp[i])
    im = ax.imshow(all_C_matrices[i], cmap='coolwarm', 
                   vmin=global_vmin, 
                   vmax=global_vmax)
    ax.set_title(titles[i], fontsize=10)
    ax.axis('off')

# Add one common colorbar for the global grid
cbar_ax_comp = fig.add_axes([0.77, 0.1, 0.02, 0.85]) 
fig.colorbar(im, cax=cbar_ax_comp, label='Value (Global Range)')

# --- PART B: LOCAL STRUCTURE PLOT (The 12th spot, isolated) ---
gs_struct = fig.add_gridspec(1, 1, left=0.82, right=0.98, top=0.55, bottom=0.35)
ax_struct = fig.add_subplot(gs_struct[0])

# Plot True C using its LOCAL scale (revealing the diagonal structure)
im_struct = ax_struct.imshow(C_True, cmap='viridis', 
                             vmin=local_vmin, 
                             vmax=local_vmax)
ax_struct.set_title('True C Structure (Local Scale)', fontsize=12)
ax_struct.set_xticks(np.arange(C_True.shape[0]))
ax_struct.set_yticks(np.arange(C_True.shape[0]))
ax_struct.tick_params(labelsize=8)

# Add a separate colorbar for the local structure plot
cbar_ax_struct = fig.add_axes([0.82, 0.25, 0.16, 0.02])
fig.colorbar(im_struct, cax=cbar_ax_struct, orientation='horizontal', 
             label=f'Value (Local Range: {local_vmin:.2e} to {local_vmax:.2e})')

fig.suptitle('LGSSM C Matrix Comparison: Global vs. Structural Views', fontsize=18)
plt.show()


## Visually compare Learned Parameter R for 10 runs with True Parameter R via Heatmap
all_R_matrices = []
all_R_matrices.append(R)
for j in range(len(R_end)):
    all_R_matrices.append(R_end[j])

# 1. Calculate Global Scale (for comparison)
all_data = np.concatenate([m.flatten() for m in all_R_matrices])
global_vmin = all_data.min()
global_vmax = all_data.max()

# 2. Calculate Local Scale for R_True (for structure)
R_True = all_R_matrices[0]
local_vmin = R_True.min()
local_vmax = R_True.max()

# 3. Create a figure with a space for the comparison grid (11 matrices) 
#    and the structural plot (1 matrix)

# Create a figure large enough for the grid (3x4 = 12 spots) + extra space
fig = plt.figure(figsize=(16, 11))

# --- PART A: GLOBAL COMPARISON GRID (The 11 original plots) ---
gs_comp = fig.add_gridspec(3, 4, left=0.05, right=0.75, top=0.95, bottom=0.1) 
# Plots 1-11 go here (3 rows, 4 columns)

titles = ['True R (Global Scale)'] + [f'Run {i+1}' for i in range(10)] + ['']
for i in range(11):
    ax = fig.add_subplot(gs_comp[i])
    im = ax.imshow(all_R_matrices[i], cmap='coolwarm', 
                   vmin=global_vmin, 
                   vmax=global_vmax)
    ax.set_title(titles[i], fontsize=10)
    ax.axis('off')

# Add one common colorbar for the global grid
cbar_ax_comp = fig.add_axes([0.77, 0.1, 0.02, 0.85]) 
fig.colorbar(im, cax=cbar_ax_comp, label='Value (Global Range)')

# --- PART B: LOCAL STRUCTURE PLOT (The 12th spot, isolated) ---
gs_struct = fig.add_gridspec(1, 1, left=0.82, right=0.98, top=0.55, bottom=0.35)
ax_struct = fig.add_subplot(gs_struct[0])

# Plot True R using its LOCAL scale (revealing the diagonal structure)
im_struct = ax_struct.imshow(R_True, cmap='viridis', 
                             vmin=local_vmin, 
                             vmax=local_vmax)
ax_struct.set_title('True R Structure (Local Scale)', fontsize=12)
ax_struct.set_xticks(np.arange(R_True.shape[0]))
ax_struct.set_yticks(np.arange(R_True.shape[0]))
ax_struct.tick_params(labelsize=8)

# Add a separate colorbar for the local structure plot
cbar_ax_struct = fig.add_axes([0.82, 0.25, 0.16, 0.02])
fig.colorbar(im_struct, cax=cbar_ax_struct, orientation='horizontal', 
             label=f'Value (Local Range: {local_vmin:.2e} to {local_vmax:.2e})')

fig.suptitle('LGSSM R Matrix Comparison: Global vs. Structural Views', fontsize=18)
plt.show()

# %% PART C: Test data performance comparison

# Import data: ssm_spins_test.txt
with open("ssm_spins_test.txt", "r") as f:
    test_data = pd.read_csv(f, sep='\s+', header=None)

# rename vector components
test_data.columns = ["x1","x2","x3","x4","x5"]

X_test = test_data.T.values # get transposed data for correct dimensions, and create numpy array from the pandas data frame

# %%  ## Evaluate likelihood of the test data under a) True parameteres and b) 10 learned parameters

## Collect parameters: true and learned x 10
all_param_sets = []
true_set_em = {
    'label': 'True',
    'A': A_final,
    'Q': Q_final,
    'C': C_final,
    'R': R_final
}
all_param_sets.append(true_set_em)

num_runs = 10 # number of randomized runs

for i in range(num_runs):
    learned_set = {
        'label': f'Run {i+1}',
        'A': A_end[i],
        'Q': Q_end[i],
        'C': C_end[i],
        'R': R_end[i]
    }
    all_param_sets.append(learned_set)

## Initializations
# variable to collect results
log_likelihood_results = {}

# initialize y_init (state) as zero vector (for all 11 evaluations)
y_init_test = np.zeros(A.shape[0])

# initialize Q_init (state covariance) to high uncertainty identity matrix (for all 11 evaluations)
Q_init_test = (10**6) * np.identity(A.shape[0]) 

## run filter mode (forward pass) for each param set for evaluation of log-likelihood for comparison
for param_set in all_param_sets:
    A_set = param_set['A']
    Q_set = param_set['Q']
    C_set = param_set['C']
    R_set = param_set['R']
    label = param_set['label']

    # run the filter using the consistent initial conditions
    y_hat, V_hat, V_joint, likelihood = run_ssm_kalman(y_init=y_init_test, Q_init=Q_init_test, X=X_test, A=A_set, Q=Q_set, C=C_set, R=R_set, mode='filt')
    
    # calculate the total log likelihood over time for comparison
    total_ll = np.round(np.sum(likelihood),2)

    log_likelihood_results[label] = total_ll

# Collect log-likelihood of test data for parameter sets
labels = []
lls = []
for label, ll in log_likelihood_results.items():
    labels.append(label)
    lls.append(ll)

# Plot train vs test performance
plt.figure()
plt.scatter(labels,final_lh_run,color="blue",label='Training Data LL',marker='x')
plt.scatter(labels,lls,color="red",label='Test Data LL',marker='o')
plt.show()

# %% 
