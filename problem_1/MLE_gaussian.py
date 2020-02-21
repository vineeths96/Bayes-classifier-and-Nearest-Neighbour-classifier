import numpy as np

DIMENSIONS = 2

def MLE_gaussian(X, N):
    mu_estimate = np.sum(X, axis=0)/N

    sigma_estimate = np.zeros([DIMENSIONS, DIMENSIONS])
    for idx in range(len(X)):
        outer_product = np.outer((X[idx] - mu_estimate), (X[idx] - mu_estimate))
        sigma_estimate += outer_product

    sigma_estimate = sigma_estimate/N
    return mu_estimate, sigma_estimate
