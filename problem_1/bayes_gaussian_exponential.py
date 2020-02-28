import numpy as np


def bayes_gaussian_exponential(theta_0, theta_1, X_test, p0=0.5, p1=0.5):
    mu_0 = theta_0[0]
    sigma_0 = theta_0[1]
    det_sigma_0 = np.linalg.det(sigma_0)
    inv_sigma_0 = np.linalg.inv(sigma_0)

    lambda_1 = theta_1[0]
    lambda_2 = theta_1[1]

    Y_pred = np.zeros(len(X_test))

    for idx in range(len(X_test)):
        X = X_test[idx]
        q0 = 1/(2*np.pi*(det_sigma_0**0.5)) * np.exp(-0.5 * (X-mu_0)@ inv_sigma_0 @np.transpose(X-mu_0)) *p0
        X = np.abs(X)
        q1 = lambda_1 * np.exp(-lambda_1 * X[0]) * lambda_2 * np.exp(-lambda_2 * X[1]) *p1

        if q0 >= q1:
            Y_pred[idx] = 0
        else:
            Y_pred[idx] = 1

    return Y_pred


if __name__ == "__main__":
    bayes()