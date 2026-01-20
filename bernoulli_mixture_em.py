"""
Bernoulli Mixture Model via EM Algorithm
----------------------------------------
Author: Harshad Dahake
Context: Gatsby Unit Coursework (Probabilistic and Unsupervised Learning)

Description:
  Implementation of the Expectation-Maximization (EM) algorithm for clustering 
  high-dimensional binary data (digit images) using a Mixture of Bernoulli distributions.
  
  Features manual derivation and implementation of:
  - The E-step: Calculating posterior responsibilities.
  - The M-step: Closed-form parameter updates.
  - Numerical Stability: Implementation of the Log-Sum-Exp trick to prevent 
    underflow in posterior calculations.

Key Concepts:
  - Latent Variable Models
  - Maximum Likelihood Estimation
  - Numerical Stability in Probabilistic Inference
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy as sp

def main():
    # Import image data
    Y = np.loadtxt('binarydigits.txt')

    # Define E step function
    def E_step(X, P, pi, K: int):
        N, D = X.shape
        r = np.zeros((N,K))
        updated_log_lh = 0

        # Update responsibilities r_nk, and log-joint and log-likelihood
        for n in range(N):
            # print(f"n: {n}")
            X_n = X[n]
            log_joint_likelihood = np.zeros(K)
            for k in range(K):
                log_joint_likelihood[k] = np.log(pi[k]) + np.sum(X_n*np.log(P[k,:]) + (1-X_n)*np.log(1-P[k,:]))
            
            ## Use log-sum-exp trick to avoid numerical instability issues

            # compute scaling constant: max(log joint likelihood)
            log_scaling = np.max(log_joint_likelihood)
            
            # scale numerator and denominator by log_scaling
            scaled_log_joint = log_joint_likelihood - log_scaling
            
            # compute log evidence using the scaled terms
            
            # The sum_k part in the log domain:
            log_sum_exp_scaled = np.log(np.sum(np.exp(scaled_log_joint)))
            
            # Total log evidence (log P(x_n))
            log_evidence = log_scaling + log_sum_exp_scaled
            
            log_r_nk = log_joint_likelihood - log_evidence
            
            # Update responsibilities
            r[n,:] = np.exp(log_r_nk)
            
            # Sum over n to get total log likelihood
            updated_log_lh += log_evidence
        
        return (r, updated_log_lh)

    # Define M step function
    def M_step(X, r, K: int):
        N, D = X.shape
        
        ## Update pi (Mixing Proportions)
        pi_updated = (1/N)*(np.sum(r, axis=0))
        
        ## Update P (Bernoulli Parameter Vectors)
        
        # create placeholder variable for updated P
        P_updated = np.zeros((K,D))

        # update P[k,d]
        for k in range(K):
            denom = np.sum(r[:,k])
            for d in range(D):
                numerator = 0
                for n in range(N):
                    X_nd = X[n,d]
                    numerator += X_nd*r[n,k]
                P_updated[k,d] = numerator/denom
        
        # clamp P matrix to avoid P_kd = 0 (numerical instabilities)
        eps = 10**-10
        P_updated = np.clip(P_updated, eps, 1 - eps)
        
        return (P_updated, pi_updated)

    # Define EM algorithm function
    def EM_algorithm(X, K: int, iters_max: int):
        # Announce number of distributions
        # print(f"K={K}")

        # Set threshold for convergence check
        epsilon = 10**(-6)
        
        # placeholder variable for actual # iterations if the algorithm terminates (convergence) earlier than iters_max
        total_iter = iters_max

        # Get N: number of images and D: number of pixels per image
        N, D = X.shape

        # Randomly initialised mixing proportions (with constraint: sum of all mixing proportions = 1
        pi_curr = np.random.rand(K); pi_curr = pi_curr/np.sum(pi_curr)

        # Randomly initialised Bernoulli Parameters (with constraint: P[k,d] in [0,1])
        P_curr = np.random.rand(K,D)
        eps = 10**-10
        P_curr = np.clip(P_curr, eps, 1-eps)
        # print(f"Starting P:\n{P_curr}")

        # To record log likelihood for plotting
        log_lh = np.zeros(iters_max)

        # Run EM algorithm (E and M steps) until convergence or up to max number of iterations
        for iter in range(iters_max):
            # print(f"Starting iter: {iter+1}")
            
            # E step
            r_updated, log_lh[iter] = E_step(X, P_curr, pi_curr, K)
            
            # M step
            P_curr, pi_curr = M_step(X, r_updated, K)
            
            # Check for convergence (relative improvement of log likelihood)
            if iter > 0:
                if np.abs(log_lh[iter] - log_lh[iter-1])/np.abs(log_lh[iter-1]) < epsilon:
                    # print(f"Stopping now at iter #{iter+1}")
                    total_iter = iter+1 # set total iterations per termination, if convergence occurs
                    break
        
        return (r_updated, P_curr, pi_curr, log_lh[:total_iter], total_iter)

    ## Define function for visualizing learned parameters
    def visualize_learned_parameters(P_updated, K, run_num: int):
        # create subplot for each of the K components
        fig, axes = plt.subplots(1, K, figsize=(15, 3))
        
        # convert 'axes' to array if axes=1 (K=1)
        if K == 1:
            axes = np.array([axes])

        # iterate through each component (k) of the P matrix
        for k in range(K):
            # get the k-th probability vector
            p_k = P_updated[k, :]
            
            # reshape the 1D vector (64 elements) into an 8x8 matrix
            image_matrix = p_k.reshape(8, 8)
            
            # plot image
            axes[k].imshow(image_matrix, cmap='gray', vmin=0, vmax=1)
            
            # clean up axes and title
            axes[k].set_title(f'Cluster {k+1}', fontsize=10)
            axes[k].axis('off') # Hide axis ticks and labels
        
        # add main title for full image
        plt.suptitle(f'Learned Mean Images from run: {run_num} for K={K} Clusters', fontsize=14)
        plt.tight_layout(rect=[0, 0, 1, 0.95]) # Adjust layout to prevent title overlap
        plt.show()

    ### PART D - Run and compare across K: {2,3,4,7,10}

    # Values for hyperparameter K - Number of Bernoulli Distribution Components
    Ks = [2,3,4,7,10]

    # Hyperparameters: select max iterations for this run
    max_iterations = 50

    for num_dist in Ks:
        r_upd, P_upd, pi_curr, log_lh, total_iter = EM_algorithm(X=Y, iters_max=max_iterations, K=num_dist)
        
        ## Plot log likelihood vs iterations
        plt.figure()
        plt.plot(range(1,(total_iter+1)), log_lh[:total_iter], alpha=0.7)
        plt.title(f"Log likelihood vs Iterations for K={num_dist} [{total_iter} iterations]")
        plt.xlabel("Iterations")
        plt.ylabel("Log likelihood")
        plt.show()

        ## Plot Mixing Proportions for K
        component_labels = [f'Comp {k+1}' for k in range(num_dist)]
        plt.figure(figsize=(8, 5))
        plt.bar(component_labels, pi_curr, color=plt.cm.viridis(np.linspace(0, 1, num_dist)))
        plt.title('Learned Mixing Proportions (Prior Probabilities)')
        plt.ylabel('Proportion ($\pi_k$)')
        plt.xlabel('Bernoulli Component')
        plt.ylim(0, 1) # Proportions must be between 0 and 1
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.show()

        ## Plot component Bernoulli distribution parameters learned: P
        visualize_learned_parameters(P_upd, num_dist, 1)
        
        print("\n\n\n")


    ### PART E: Run with fixed parameter K from a few randomly chosen initial conditions (= "runs") and compare results across runs for different values of K ###

    ## Compare parameters found across runs

    # Values for hyperparameter K - Number of Bernoulli Distributions
    Ks = [2,3,4,7,10]

    # No. of Runs
    runs = 5

    # Hyperparameters: max iterations for this run
    max_iterations = 30

    for num_dist in Ks:
        r_upds = []
        P_upds = []
        pi_upds = []
        log_lhs = []
        total_iters = []
        
        for j in range(runs):
            r_upd, P_upd, pi_upd, log_lh, total_iter = EM_algorithm(X=Y, iters_max=max_iterations, K=num_dist)

            # append results for each run
            r_upds.append(r_upd)
            P_upds.append(P_upd)
            pi_upds.append(pi_upd)
            log_lhs.append(log_lh)
            total_iters.append(total_iter)
        
        ## Compare parameters found across runs for K by plotting
        comp_size = (8, 8)
        all_images = []

        # load all images into a flat list for each K across runs to plot
        for i in range(runs):
            for x in range(num_dist):
                all_images.append(P_upds[i][x].reshape(8,8)) 
        
        # create the main figure and subplots
        fig, axes = plt.subplots(
            nrows=runs, 
            ncols=num_dist, 
            figsize=(num_dist * 2, runs * 1.5) # Adjust figsize for clarity
        )

        # set main title for the entire comparison
        fig.suptitle(f'Comparison of Learned Parameters Across {runs} EM Runs (K={num_dist})', 
                    fontsize=14, y=1.02)
        plt.tight_layout(rect=[0, 0, 1, 0.98]) # Adjust layout to fit suptitle

        # iterate through K parameters across runs to plot
        for i in range(runs):
            for x in range(num_dist):
                ax = axes[i, x]
                idx = i * num_dist + x
                
                # plot the 8x8 image data
                ax.imshow(all_images[idx], cmap="gray", interpolation='nearest') 
                
                # clean up axes for a compact view
                ax.set_xticks([])
                ax.set_yticks([])
                
                # add sub-titles for row/column identification of each image
                if x == 0:
                    ax.set_ylabel(f'Run {i+1}', rotation=0, labelpad=30, fontsize=10)
                if i == 0:
                    ax.set_title(f'Comp {x+1}', fontsize=10)

        plt.show()

        ## Plot log-likelihood vs iterations for each K
        plt.figure()
        for z in range(runs):
            plt.plot(range(1,total_iters[z]+1), log_lhs[z], alpha=0.8)
        plt.title(f"Log-likelihood vs Iterations for {runs} runs for K={num_dist}")
        plt.xlabel("Iteration")
        plt.ylabel("Log-likelihood")
        plt.show()

        ## Plot optimized mixing proportions for each K

        # Additionally, I checked against a ballpark margin (threshold) - number of clusters with 'significant' mixing proportions
        # cluster-results with close to 0 significant mixing proportions would correspond to us exceeding appropriate values of K
        margin = 0.15
        plt.figure()
        for z in range(runs):
            plt.plot(range(1,num_dist+1), pi_upds[z], alpha=0.8)
            count=0
            for a in range(num_dist):
                if pi_upds[z][a] > margin:
                    count+=1
            print(f"Run {z} for k={num_dist} has {count} mix-props > {margin}")
        plt.title(f"Mixing Proportions for {runs} runs for K={num_dist}")
        plt.xlabel("Component (k)")
        plt.ylabel("Mixing Proportion")
        plt.show()

if __name__ == "__main__":
    main()