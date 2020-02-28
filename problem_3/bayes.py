import numpy as np


def bayes(theta_0, theta_1, X_test, p0=0.5, p1=0.5):
    mu_0 = theta_0[0]
    sigma_0 = theta_0[1]
    det_sigma_0 = np.linalg.det(sigma_0)
    inv_sigma_0 = np.linalg.inv(sigma_0)

    mu_1 = theta_1[0]
    sigma_1 = theta_1[1]
    det_sigma_1 = np.linalg.det(sigma_1)
    inv_sigma_1 = np.linalg.inv(sigma_1)

    Y_pred = np.zeros(len(X_test))

    for idx in range(len(X_test)):
        X = X_test[idx]
        q0 = 1/(2*np.pi*(det_sigma_0**0.5)) * np.exp(-0.5 * (X-mu_0)@ inv_sigma_0 @np.transpose(X-mu_0)) *p0
        q1 = 1/(2*np.pi*(det_sigma_1**0.5)) * np.exp(-0.5 * (X-mu_1)@ inv_sigma_1 @np.transpose(X-mu_1)) *p1

        if q0 >= q1:
            Y_pred[idx] = 0
        else:
            Y_pred[idx] = 1

    return Y_pred


if __name__ == "__main__":
    bayes()