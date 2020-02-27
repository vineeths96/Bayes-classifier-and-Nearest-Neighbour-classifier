import numpy as np

DIMENSIONS = 2


def MLE_exponential(X, N):
    X = np.abs(X)
    lambda_estimate = N/np.sum(X, axis=0)

    return lambda_estimate


if __name__ == '__main__':
    MLE_exponential()
